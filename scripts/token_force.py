#!/usr/bin/env python
"""Token-forcing lossless gate (SpecPilot, arXiv 2611.02501 §3.3).

Generates every prompt twice on the same loaded target, once with a REFERENCE mode (default
``baseline``, plain greedy, on mlx's stock kernels) and once with a CANDIDATE mode (default:
the pair's ``--mode auto`` pick, with the kernels ``mlx-dspark generate`` applies). Each
output is then teacher-forced through the target (``mlx_dspark.token_force.score_forced``)
and every output position is checked: is the committed token the target's argmax there?

Unlike firstdiff, this keeps working past the first fp tie, because each stream is scored
on the prefix it committed itself. Near-tie flips (margin <= delta) are expected from kernel
rounding in any correct system; the gate asks whether the candidate has measurably MORE
mismatches than the reference, or any meaningful number of large-margin (> 2*delta) events:

  rate rule    fail iff the candidate's mismatch rate exceeds the reference's by more than
               --tolerance (0.5pp) with 95% confidence (one-sided two-proportion z-test,
               unpooled SE; see token_force.rate_rule)
  event rule   fail iff the candidate's large-margin event count exceeds the exact 95%
               Poisson upper limit of the reference's count at the candidate's exposure
               (zero reference events: 3 candidate events fail; see token_force.event_rule)

``--ref-mode none`` is the single-system sanity check: score the candidate only, print its
rate, max margin and mismatch list; it exits non-zero on any large-margin event.

Exit status: 0 pass, 1 gate failed (or large-margin events in single-system mode).
Greedy only (temperature 0): token forcing scores argmax decisions.

Works offline (HF_HUB_OFFLINE=1) with local or cached models; loading goes through the
library exactly as ``mlx-dspark generate`` does.

Usage:
    python scripts/token_force.py --model mlx-community/Qwen3-4B-8bit
    python scripts/token_force.py --model <repo> --mode dspark --max-draft 4 --json out.json
    python scripts/token_force.py --model <repo> --mode dspark --ref-mode dspark   # kernels vs stock
    python scripts/token_force.py --model <repo> --mode dflash --ref-mode none     # sanity only
    python scripts/token_force.py --model <repo> --prompts-file prompts.jsonl --no-bench-prompts

``--prompts-file`` is JSONL, one item per line: a JSON string, or an object with "prompt"
(a user message, chat-templated) or "messages" (an OpenAI-shape transcript), plus an
optional "name".
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import sys
from contextlib import contextmanager

import mlx.core as mx

from mlx_dspark.cli import BENCH_PROMPTS, _parse_max_draft
from mlx_dspark.generate import (
    dflash_generate,
    encode_messages,
    encode_prompt,
    greedy_generate,
    speculative_generate,
)
from mlx_dspark.load import (
    load_dflash,
    load_drafter,
    load_target,
    lookup_drafts_default,
    resolve_drafter_window,
    resolve_mode,
)
from mlx_dspark.lookup import lookup_generate
from mlx_dspark.token_force import (
    ALPHA,
    DEFAULT_DELTA,
    RATE_TOLERANCE,
    SCORE_CHUNK,
    ForceStats,
    force_stats,
    gate,
    score_forced,
)

# the MODULE (sys.modules entry), never a same-named package export
_gen = importlib.import_module("mlx_dspark.generate")
SPEC_MODES = ("dspark", "dflash")

# generate's process-wide kernel knobs: the apply_* calls the CLI makes set exactly these
# (library default: all None = mlx stock kernels). Snapshotting them per arm is what makes
# "reference = stock, candidate = CLI kernels" hold regardless of arm order.
_KERNEL_KNOBS = ("SMALL_M_IDS", "SDPA_SPLIT_CFG", "MULTIROW_CFG",
                 "WIDE_GEMM_MIN_ROWS", "WIDE_GEMM_SHAPES", "CPU_SPLIT")


@contextmanager
def kernel_scope():
    """Run an arm from the library default (every scoped kernel off) and restore the
    process-wide knobs afterwards, so each arm's kernel context is exactly what it applies."""
    saved = {k: getattr(_gen, k) for k in _KERNEL_KNOBS}
    try:
        for k in _KERNEL_KNOBS:
            setattr(_gen, k, None)
        yield
    finally:
        for k, v in saved.items():
            setattr(_gen, k, v)


def load_prompts(path: str | None, bench: bool) -> list[tuple[str, dict]]:
    items: list[tuple[str, dict]] = []
    if bench:
        items += [(name, {"prompt": p}) for name, p in BENCH_PROMPTS.items()]
    if path:
        with open(path) as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, str):
                    obj = {"prompt": obj}
                if not isinstance(obj, dict) or not ("prompt" in obj or "messages" in obj):
                    raise SystemExit(f"{path}:{ln}: expected a JSON string or an object with "
                                     f"'prompt' or 'messages'")
                items.append((str(obj.get("name") or obj.get("id") or f"line{ln}"), obj))
    if not items:
        raise SystemExit("no prompts (drop --no-bench-prompts or pass --prompts-file)")
    return items


def encode_item(tok, item: dict, use_chat: bool) -> list[int]:
    if "messages" in item:
        return encode_messages(tok, item["messages"])
    return encode_prompt(tok, item["prompt"], use_chat=use_chat)


def run_arm(role: str, mode: str, drafter_repo: str | None, *, target, tok, target_repo: str,
            prompts: list[list[int]], names: list[str], args, kernels: bool,
            max_draft, tree="off") -> dict:
    """Load this arm's drafter, apply its kernel context, derive its cap, and generate every
    prompt -- in ``cmd_generate``'s order (kernels before the cap is measured, so the cap is
    derived inside the context generation runs in). Returns token ids + metadata; the
    drafter is released on return."""
    from mlx_dspark.calibrate import (
        apply_cpu_split,
        apply_multirow_attn,
        apply_sdpa_split,
        apply_small_m,
        apply_wide_gemm,
        calibrate,
        static_cap,
        static_tree,
    )

    with kernel_scope():
        drafter = None
        if mode == "dspark":
            drafter, _ = load_drafter(drafter_repo, quantize=args.drafter_bits > 0,
                                      bits=max(args.drafter_bits, 2))
            drafter.ctx_window = resolve_drafter_window(args.drafter_window, target_repo)
        elif mode == "dflash":
            drafter, _ = load_dflash(drafter_repo, quantize=args.drafter_bits > 0,
                                     bits=max(args.drafter_bits, 2))
            drafter.bind(target.model)
        if kernels:
            apply_small_m(target, drafter, target_repo=target_repo, drafter_repo=drafter_repo,
                          verbose=False)
            apply_sdpa_split(target, target_repo=target_repo, verbose=False)
            apply_multirow_attn(target, drafter, verbose=False)
        ctrl = None
        md = max_draft
        if md == "auto":
            if mode in SPEC_MODES:
                ctrl = calibrate(target, drafter, mode=mode, target_repo=target_repo,
                                 drafter_repo=drafter_repo, verbose=False)
            md = None
        if kernels:
            apply_wide_gemm(target, drafter, target_repo=target_repo, verbose=False)
            apply_cpu_split(target, drafter, target_repo=target_repo, verbose=False)

        n = args.max_tokens
        cap = None
        lookup = None
        tree_nodes = None
        if mode == "dspark":
            cap = md
            if md is None and ctrl is None:
                cap = static_cap(target, drafter, mode="dspark", target_repo=target_repo,
                                 drafter_repo=drafter_repo)
            lookup = (args.lookup_drafts if args.lookup_drafts is not None
                      else lookup_drafts_default(target_repo))
            tree_nodes = static_tree(target, drafter, setting=tree, target_repo=target_repo,
                                     drafter_repo=drafter_repo, cap=cap, mode="dspark")

            def gen(ids):
                return speculative_generate(target, tok, drafter, prompt_ids=ids,
                                            max_new_tokens=n, max_draft_tokens=cap,
                                            cap_controller=ctrl, lookup_drafts=lookup,
                                            tree=tree_nodes)
        elif mode == "dflash":
            cap = None if (md is None or md <= 0) else md
            if md is None and ctrl is None:
                fb = max(1, int(getattr(drafter.config, "block_size", 8)) - 1)
                cap = static_cap(target, drafter, mode="dflash", target_repo=target_repo,
                                 drafter_repo=drafter_repo, fallback=fb)
            # DFlash 2 tree verify, resolved exactly as `generate` / `serve` do
            tree_nodes = static_tree(target, drafter, setting=tree, target_repo=target_repo,
                                     drafter_repo=drafter_repo, cap=cap)

            def gen(ids):
                return dflash_generate(target, tok, drafter, prompt_ids=ids,
                                       max_new_tokens=n, max_draft_tokens=cap,
                                       cap_controller=ctrl, tree=tree_nodes)
        elif mode == "lookup":
            cap = 6 if (md is None or not isinstance(md, int) or md <= 0) else md

            def gen(ids):
                return lookup_generate(target, tok, prompt_ids=ids, max_new_tokens=n,
                                       max_draft_tokens=cap, long_draft_tokens=max(cap, 32))
        else:
            def gen(ids):
                return greedy_generate(target, tok, prompt_ids=ids, max_new_tokens=n)

        desc = (f"{mode}" + (f" cap {cap}" if cap is not None else "")
                + (f" tree {'w' if tree_nodes == 0 else tree_nodes}"
                   if tree_nodes is not None else "")
                + (" auto-cap" if ctrl is not None else "")
                + ("" if lookup is None else f" lookup-drafts {'on' if lookup else 'off'}")
                + (" · CLI kernels" if kernels else " · stock kernels"))
        print(f"{role}: {desc}" + (f" · drafter {drafter_repo}" if drafter_repo else ""))
        outs = []
        for name, ids in zip(names, prompts):
            r = gen(ids)
            outs.append({"token_ids": list(r.token_ids), "finish_reason": r.finish_reason,
                         "accept": round(r.mean_accept_len, 2),
                         "decode_tok_s": round(r.decode_tokens_per_sec, 1)})
            print(f"  {name:<16} {r.num_tokens:>5} tok · {r.finish_reason:<6} · "
                  f"accept {r.mean_accept_len:.2f} · {r.decode_tokens_per_sec:.1f} tok/s")
    drafter = gen = None
    gc.collect()
    mx.clear_cache()
    return {"role": role, "mode": mode, "drafter": drafter_repo, "cap": cap,
            "auto_cap": ctrl is not None, "lookup_drafts": lookup,
            "kernels": "cli" if kernels else "stock", "outputs": outs}


def _firstdiff(a: list[int], b: list[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))


def _tokstr(tok, t: int) -> str:
    try:
        s = repr(tok.decode([t]))
    except Exception:  # noqa: BLE001 — display only
        s = "?"
    return f"{s[:14]}({t})"


def _stats_line(label: str, st: ForceStats) -> str:
    return (f"  {label:<10} {st.positions:>9} {st.n_mismatches:>11} "
            f"{st.mismatch_rate:>9.3%} {st.near_ties:>8} {st.gray:>5} {st.large_events:>6} "
            f"{st.max_margin:>11.3f}")


def _print_mismatches(label: str, st: ForceStats, names: list[str], tok, show: int) -> None:
    if not st.mismatches or show <= 0:
        return
    top = sorted(st.mismatches, key=lambda m: -m.margin)[:show]
    print(f"\n{label} mismatches (largest margin first, {len(top)} of {st.n_mismatches}):")
    for m in top:
        cls = ("LARGE" if m.margin > 2 * st.delta else
               "near-tie" if m.margin <= st.delta else "gray")
        print(f"  {names[m.sample]:<16} pos {m.position:>5}  committed "
              f"{_tokstr(tok, m.committed):<22} argmax {_tokstr(tok, m.argmax):<22} "
              f"margin {m.margin:.4f}  {cls}")


def main() -> int:
    ap = argparse.ArgumentParser(prog="token_force.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="target: registry id, HF repo, or local path")
    ap.add_argument("--drafter", default=None, help="candidate drafter (default: registry)")
    ap.add_argument("--mode", default="auto",
                    choices=["auto", "dspark", "dflash", "lookup", "baseline"],
                    help="candidate mode (auto = the pair's measured-best, as `generate`)")
    ap.add_argument("--ref-mode", default="baseline",
                    choices=["baseline", "dspark", "dflash", "lookup", "none"],
                    help="reference mode, run on stock kernels; 'none' = single-system check")
    ap.add_argument("--ref-drafter", default=None,
                    help="reference drafter (default: --drafter when the modes match, else "
                         "the registry)")
    ap.add_argument("--max-tokens", type=int, default=256, help="max new tokens per prompt")
    ap.add_argument("--max-draft", default=None,
                    help="candidate cap (int or 'auto'); unset = derived, as `generate`")
    ap.add_argument("--ref-max-draft", default=None,
                    help="reference cap when --ref-mode is a spec mode; unset = derived")
    ap.add_argument("--delta", type=float, default=DEFAULT_DELTA,
                    help="near-tie threshold in nats; > 2*delta is a large-margin event")
    ap.add_argument("--tolerance", type=float, default=RATE_TOLERANCE,
                    help="rate-rule allowance as a fraction (0.005 = 0.5pp)")
    ap.add_argument("--alpha", type=float, default=ALPHA, help="one-sided significance level")
    ap.add_argument("--prompts-file", default=None, help="JSONL prompts (see above)")
    ap.add_argument("--no-bench-prompts", action="store_true",
                    help="skip the built-in chat/code/math benchmark prompts")
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--stock-kernels", action="store_true",
                    help="run the candidate on stock kernels too (no small-M / SDPA split / "
                         "multi-row / wide-GEMM / CPU co-prefill)")
    ap.add_argument("--lookup-drafts", action=argparse.BooleanOptionalAction, default=None,
                    help="dspark hybrid lookup drafts; unset = the pair's default")
    ap.add_argument("--drafter-window", type=int, default=None, metavar="N",
                    help="DSpark drafter context window (0 = whole context; unset = pair default)")
    ap.add_argument("--drafter-bits", type=int, default=4)
    ap.add_argument("--kv-bits", type=int, default=0, choices=[0, 4, 8])
    ap.add_argument("--tree", default="auto",
                    help="candidate dflash/dspark arm: tree verify auto|on|off|N (as `generate`)")
    ap.add_argument("--ref-tree", default="off",
                    help="reference dflash/dspark arm's tree setting (default off = the chain)")
    ap.add_argument("--chunk", type=int, default=SCORE_CHUNK,
                    help="rows per teacher-forced scoring forward (memory: chunk x vocab)")
    ap.add_argument("--show", type=int, default=10, help="mismatches to list per system")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--json", default=None, help="also write the full report to this file")
    args = ap.parse_args()
    if args.trust_remote_code:
        from mlx_dspark import load as _load

        _load.TRUST_REMOTE_CODE = True
    if args.delta <= 0:
        ap.error("--delta must be > 0")

    try:
        cand_mode, target_repo, cand_drafter = resolve_mode(
            args.model, mode=args.mode, drafter=args.drafter)
        ref_mode = ref_drafter = None
        if args.ref_mode != "none":
            rd = args.ref_drafter or (args.drafter if args.ref_mode == cand_mode else None)
            ref_mode, _, ref_drafter = resolve_mode(target_repo, mode=args.ref_mode, drafter=rd)
    except ValueError as e:
        ap.error(str(e))
    max_draft = _parse_max_draft(args.max_draft, ap)
    ref_max_draft = _parse_max_draft(args.ref_max_draft, ap)

    dev = mx.device_info().get("device_name", "?")
    print(f"token forcing · {dev} · mlx {mx.__version__} · target {target_repo} · "
          f"delta {args.delta} · max {args.max_tokens} tok")
    items = load_prompts(args.prompts_file, not args.no_bench_prompts)
    target, tok = load_target(
        target_repo, require_tap=cand_mode in SPEC_MODES or ref_mode in SPEC_MODES,
        kv_bits=args.kv_bits or None)
    names = [name for name, _ in items]
    prompts = [encode_item(tok, item, not args.no_chat_template) for _, item in items]
    common = {"target": target, "tok": tok, "target_repo": target_repo, "prompts": prompts,
              "names": names, "args": args}

    arms = []
    if ref_mode is not None:                 # reference first: nothing applied yet
        arms.append(run_arm("reference", ref_mode, ref_drafter, kernels=False,
                            max_draft=ref_max_draft, tree=args.ref_tree, **common))
    arms.append(run_arm("candidate", cand_mode, cand_drafter, kernels=not args.stock_kernels,
                        max_draft=max_draft, tree=args.tree, **common))

    print("\nscoring (teacher-forced, stock kernels) …")
    stats: dict[str, ForceStats] = {}
    for arm in arms:
        scores = [score_forced(target, p, o["token_ids"], chunk=args.chunk)
                  for p, o in zip(prompts, arm["outputs"])]
        st = force_stats(scores, delta=args.delta)
        stats[arm["role"]] = st
        for o, s in zip(arm["outputs"], scores):
            o["mismatch_positions"] = s.mismatch_positions()

    cand = stats["candidate"]
    ref = stats.get("reference")
    if ref is not None:
        print(f"\n  {'prompt':<16} {'ref tok':>7} {'cand tok':>8} {'firstdiff':>9} "
              f"{'ref mism':>8} {'cand mism':>9}")
        for i, name in enumerate(names):
            ro, co = arms[0]["outputs"][i], arms[1]["outputs"][i]
            fd = _firstdiff(ro["token_ids"], co["token_ids"])
            co["firstdiff_vs_reference"] = fd
            print(f"  {name:<16} {len(ro['token_ids']):>7} {len(co['token_ids']):>8} "
                  f"{fd:>9} {len(ro['mismatch_positions']):>8} "
                  f"{len(co['mismatch_positions']):>9}")
    print(f"\n  {'':<10} {'positions':>9} {'mismatches':>11} {'rate':>9} {'near-tie':>8} "
          f"{'gray':>5} {'large':>6} {'max margin':>11}")
    if ref is not None:
        print(_stats_line("reference", ref))
    print(_stats_line("candidate", cand))
    if ref is not None:
        _print_mismatches("reference", ref, names, tok, args.show)
    _print_mismatches("candidate", cand, names, tok, args.show)

    report = {"device": dev, "mlx": mx.__version__, "target": target_repo,
              "delta": args.delta, "tolerance": args.tolerance, "alpha": args.alpha,
              "max_tokens": args.max_tokens, "prompts": names, "arms": arms,
              "stats": {k: v.to_dict() for k, v in stats.items()}}
    if ref is not None:
        verdict = gate(cand, ref, tolerance=args.tolerance, alpha=args.alpha)
        report["gate"] = verdict.to_dict()
        print("\ngate (SpecPilot §3.3):")
        for r in (verdict.rate, verdict.events):
            print(f"  {r.name:<7}{'PASS' if r.passed else 'FAIL'}  {r.detail}")
        print(f"VERDICT: {'PASS' if verdict.passed else 'FAIL'}")
        ok = verdict.passed
    else:
        ok = cand.large_events == 0
        report["single_system_clean"] = ok
        print(f"\nsingle-system check: {cand.large_events} large-margin events "
              f"(> {2 * args.delta:.2f} nats) -> {'CLEAN' if ok else 'NOT CLEAN'}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
