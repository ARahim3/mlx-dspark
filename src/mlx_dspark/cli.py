"""mlx-dspark command line.

Subcommands:
  serve      Start an OpenAI-compatible API server (LM Studio / openai SDK / curl).
  generate   One-shot local generation (DSpark / DFlash / lookup / baseline). This is also
             the default when no subcommand is given, so the historical flat invocation
             ``python -m mlx_dspark --prompt ...`` keeps working unchanged.
  benchmark  Warm, reproducible speed sweep on this machine (baseline vs the spec modes).
  models     List the built-in target+drafter presets.
  doctor     Check the environment (Apple Silicon, MLX stack, RAM vs. model size).

Run ``mlx-dspark <cmd> -h`` for a command's flags.
"""

from __future__ import annotations

import argparse
import sys

from .load import REGISTRY

_SUBCOMMANDS = ("serve", "generate", "benchmark", "models", "doctor")


def _emit(s: str) -> None:
    sys.stdout.write(s)
    sys.stdout.flush()


# --------------------------------------------------------------------------- generate


def _parse_max_draft(value: str | None, ap) -> int | str | None:
    """--max-draft accepts an int or 'auto' (calibrated per machine+model, adapts live)."""
    if value is None:
        return None
    if str(value).strip().lower() == "auto":
        return "auto"
    try:
        return int(value)
    except ValueError:
        ap.error(f"--max-draft must be an integer or 'auto', got {value!r}")


def cmd_generate(argv: list[str]) -> None:
    from .generate import dflash_generate, greedy_generate, speculative_generate
    from .load import apply_wired_limit, load_dflash, load_drafter, load_target, resolve_mode
    from .lookup import lookup_generate

    ap = argparse.ArgumentParser(prog="mlx-dspark generate")
    ap.add_argument("--mode", choices=["auto", "dspark", "dflash", "lookup", "baseline"],
                    default="dspark",
                    help="dspark = DSpark spec decoding; dflash = z-lab DFlash (block diffusion); "
                         "lookup = drafter-free prompt-lookup spec decoding (any target); "
                         "auto = best available for this target (dspark -> dflash -> lookup); "
                         "baseline = plain greedy target")
    ap.add_argument("--model", default=None,
                    help="target model: an HF repo or local path (e.g. mlx-community/Qwen3-8B-8bit). "
                         "The matched drafter auto-resolves for known targets; else pass --drafter.")
    ap.add_argument("--drafter", default=None, help="drafter repo/path (overrides auto-resolve)")
    ap.add_argument("--family", choices=["gemma4", "qwen3"], default=None,
                    help=argparse.SUPPRESS)          # deprecated alias for --model
    ap.add_argument("--target", default=None, help=argparse.SUPPRESS)  # deprecated alias for --model
    ap.add_argument("--prompt", default="Explain how rainbows form, in a few sentences.")
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--max-draft", default=None,
                    help="tokens verified per round (cap), or 'auto' to calibrate for this "
                         "machine+model and adapt live. Defaults: dspark=2, lookup=6, "
                         "dflash=full block (<=0 also means full block).")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy (exact); >0 = speculative sampling (paper setup, lossless wrt target@T)")
    ap.add_argument("--top-p", type=float, default=1.0, help="nucleus sampling (temperature > 0)")
    ap.add_argument("--top-k", type=int, default=0, help="top-k sampling (temperature > 0)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--confidence-threshold", type=float, default=0.0)
    ap.add_argument("--drafter-bits", type=int, default=4)
    ap.add_argument("--kv-bits", type=int, default=0, choices=[0, 4, 8],
                    help="quantize the target KV cache (4/8 bits; 0 = off). Cuts the KV "
                         "bandwidth bill on long contexts; mlx-lm text targets only")
    ap.add_argument("--no-lookup-drafts", action="store_true",
                    help="disable hybrid n-gram drafting inside dspark mode (on by default; "
                         "free extra speedup on copy-heavy spans, lossless either way)")
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--no-stream", action="store_true")
    args = ap.parse_args(argv)

    try:
        mode, target_repo, drafter_repo = resolve_mode(
            args.model, mode=args.mode, drafter=args.drafter,
            family=args.family, target=args.target)
    except ValueError as e:
        ap.error(str(e))
    args.mode = mode

    labels = {"dspark": "DSpark speculative", "dflash": "DFlash (z-lab) speculative",
              "lookup": "Prompt-lookup speculative (drafter-free)",
              "baseline": "Baseline (plain greedy)"}
    label = labels[args.mode]
    print(f"loading {args.mode}: target={target_repo}"
          + (f", drafter={drafter_repo}" if drafter_repo else ""))
    target, tok = load_target(target_repo, require_tap=args.mode in ("dspark", "dflash"),
                              kv_bits=args.kv_bits or None)
    drafter = None
    if args.mode == "dspark":
        drafter, _ = load_drafter(drafter_repo, quantize=args.drafter_bits > 0,
                                  bits=max(args.drafter_bits, 2))
    elif args.mode == "dflash":
        drafter, _ = load_dflash(drafter_repo, quantize=args.drafter_bits > 0,
                                 bits=max(args.drafter_bits, 2))
        drafter.bind(target.model)

    max_draft = _parse_max_draft(args.max_draft, ap)
    cap_controller = None
    if max_draft == "auto":
        if args.mode in ("dspark", "dflash"):
            from .calibrate import calibrate

            cap_controller = calibrate(target, drafter, mode=args.mode,
                                       target_repo=target_repo, drafter_repo=drafter_repo)
        max_draft = None                                   # controller (or mode default) drives

    apply_wired_limit()
    on_text = None if args.no_stream else _emit
    print("\n" + "=" * 64)
    print(f"  ▶  {label}   ·   {target_repo.split('/')[-1]}")
    print("=" * 64)

    if args.mode == "dspark":
        cap = (2 if max_draft is None and cap_controller is None else max_draft)
        res = speculative_generate(
            target, tok, drafter, args.prompt,
            max_new_tokens=args.max_new_tokens, max_draft_tokens=cap,
            cap_controller=cap_controller, lookup_drafts=not args.no_lookup_drafts,
            confidence_threshold=args.confidence_threshold,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, seed=args.seed,
            apply_chat_template=not args.no_chat_template, on_text=on_text,
        )
        extra = f" · accept {res.mean_accept_len:.2f}/round · {res.target_forwards} target fwds"
        if res.lookup_rounds:
            extra += f" · {res.lookup_rounds} lookup rounds"
    elif args.mode == "dflash":
        cap = None if (max_draft is None or max_draft <= 0) else max_draft
        res = dflash_generate(
            target, tok, drafter, args.prompt,
            max_new_tokens=args.max_new_tokens, max_draft_tokens=cap,
            cap_controller=cap_controller,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            seed=args.seed, apply_chat_template=not args.no_chat_template, on_text=on_text,
        )
        extra = f" · accept {res.mean_accept_len:.2f}/round · {res.target_forwards} target fwds"
    elif args.mode == "lookup":
        cap = 6 if (max_draft is None or not isinstance(max_draft, int) or max_draft <= 0) \
            else max_draft
        res = lookup_generate(
            target, tok, args.prompt,
            max_new_tokens=args.max_new_tokens, max_draft_tokens=cap,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, seed=args.seed,
            apply_chat_template=not args.no_chat_template, on_text=on_text,
        )
        extra = f" · accept {res.mean_accept_len:.2f}/round · {res.target_forwards} target fwds"
    else:
        res = greedy_generate(
            target, tok, args.prompt, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, seed=args.seed,
            apply_chat_template=not args.no_chat_template, on_text=on_text,
        )
        extra = ""
    if cap_controller is not None:
        extra += f" · auto-cap now {cap_controller.cap} (p≈{cap_controller.p:.2f})"
    if args.no_stream:
        print(res.text)

    print("\n" + "-" * 64)
    print(f"  {res.num_tokens} tokens · {res.seconds:.2f}s · "
          f"\033[1m{res.tokens_per_sec:.1f} tok/s\033[0m{extra}")
    print("-" * 64)


# --------------------------------------------------------------------------- serve


def cmd_serve(argv: list[str]) -> None:
    from .server import Engine, maybe_batch_engine, run_server

    ap = argparse.ArgumentParser(prog="mlx-dspark serve",
                                 description="OpenAI-compatible API server.")
    ap.add_argument("--mode", choices=["auto", "dspark", "dflash", "lookup", "baseline"],
                    default="dspark",
                    help="'auto' picks the best available speculation for the target "
                         "(dspark -> dflash -> drafter-free lookup), so any repo serves")
    ap.add_argument("--model", default=None,
                    help="target model: an HF repo or local path (e.g. mlx-community/Qwen3-8B-8bit). "
                         "Matched drafter auto-resolves for known targets; else pass --drafter. "
                         "See `mlx-dspark models`.")
    ap.add_argument("--drafter", default=None, help="drafter repo/path (overrides auto-resolve)")
    ap.add_argument("--family", choices=["gemma4", "qwen3"], default=None,
                    help=argparse.SUPPRESS)          # deprecated alias for --model
    ap.add_argument("--target", default=None, help=argparse.SUPPRESS)  # deprecated alias for --model
    ap.add_argument("--max-draft", default=None,
                    help="cap on tokens verified per round; <=0 = full block; 'auto' = calibrate "
                         "for this machine+model and adapt live. Default: dspark=2, lookup=6, "
                         "dflash=full block.")
    ap.add_argument("--default-max-tokens", type=int, default=2048,
                    help="max_tokens used when a request doesn't send one (default 2048)")
    ap.add_argument("--max-tokens-cap", type=int, default=32768,
                    help="hard ceiling on per-request max_tokens (default 32768 — thinking "
                         "models routinely need >8k)")
    ap.add_argument("--default-temperature", type=float, default=None,
                    help="temperature for requests that don't send one (overrides the model's "
                         "generation_config; note many mlx-community repos ship none, in which "
                         "case omitted temperature means greedy unless this is set)")
    ap.add_argument("--default-top-p", type=float, default=None,
                    help="top_p for requests that don't send one (see --default-temperature)")
    ap.add_argument("--default-top-k", type=int, default=None,
                    help="top_k for requests that don't send one (see --default-temperature)")
    ap.add_argument("--confidence-threshold", type=float, default=0.0)
    ap.add_argument("--drafter-bits", type=int, default=4)
    ap.add_argument("--kv-bits", type=int, default=0, choices=[0, 4, 8],
                    help="quantize the target KV cache (4/8 bits; 0 = off). Cuts the KV "
                         "bandwidth bill on long contexts; mlx-lm text targets only "
                         "(disables --max-batch)")
    ap.add_argument("--max-batch", type=int, default=1,
                    help="micro-batch up to N concurrently-queued requests through one batched "
                         "target forward (dense mlx-lm target + dspark/baseline; ~1.5-2.5x "
                         "aggregate throughput at 2-4 concurrent). 1 = serialized (default)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--api-key", default=None,
                    help="if set, requests must send 'Authorization: Bearer <key>'")
    ap.add_argument("--no-thinking", action="store_true",
                    help="default responses to non-thinking mode (Qwen3 enable_thinking=False); "
                         "clients can still override per-request")
    ap.add_argument("--no-prefix-cache", action="store_true",
                    help="disable multi-turn prefix caching (reuse the shared conversation "
                         "prefix's KV; on by default for dspark/lookup/baseline on dense or "
                         "under-window sliding-window targets)")
    ap.add_argument("--prefix-cache-slots", type=int, default=2,
                    help="number of conversations kept in the prefix cache LRU (default 2, "
                         "so an agent and a chat don't evict each other every turn)")
    ap.add_argument("--no-lookup-drafts", action="store_true",
                    help="disable hybrid n-gram drafting inside dspark mode")
    ap.add_argument("--prefix-cache-dir", default=None,
                    help="directory for the L2 SSD spill tier (enables spilling the cache to disk)")
    ap.add_argument("--prefix-cache-max-ram-mb", type=int, default=0,
                    help="spill the prefix cache to --prefix-cache-dir once it exceeds this many MB "
                         "of RAM (0 = never spill; requires --prefix-cache-dir)")
    args = ap.parse_args(argv)

    md = _parse_max_draft(args.max_draft, ap)
    if md is None or md == "auto":
        max_draft = md                                     # engine picks default / calibrates
    elif args.mode == "dflash" and md <= 0:
        max_draft = None                                   # full block
    else:
        max_draft = max(1, md)

    print(f"loading {args.mode} engine — first run downloads weights…")
    try:
        engine = Engine.load(
            mode=args.mode, model=args.model, drafter=args.drafter,
            family=args.family, target=args.target,
            drafter_bits=args.drafter_bits, max_draft_tokens=max_draft,
            confidence_threshold=args.confidence_threshold,
            enable_thinking=False if args.no_thinking else None,
            prefix_cache=not args.no_prefix_cache,
            prefix_cache_dir=args.prefix_cache_dir,
            prefix_cache_max_ram_mb=args.prefix_cache_max_ram_mb,
            default_max_tokens=args.default_max_tokens,
            max_tokens_cap=args.max_tokens_cap,
            default_temperature=args.default_temperature,
            default_top_p=args.default_top_p,
            default_top_k=args.default_top_k,
            prefix_cache_slots=args.prefix_cache_slots,
            lookup_drafts=not args.no_lookup_drafts,
            batch_widths=(sorted({2, args.max_batch}) if args.max_batch > 1 else None),
            kv_bits=args.kv_bits or None,
        )
    except ValueError as e:
        ap.error(str(e))
    engine = maybe_batch_engine(engine, args.max_batch)
    run_server(engine, host=args.host, port=args.port, api_key=args.api_key)


# --------------------------------------------------------------------------- benchmark


BENCH_PROMPTS = {
    "chat": "Explain how rainbows form.",
    "code": "Write a Python function to check if a string is a palindrome.",
    "math": "A train travels 120 km in 1.5 hours. What is its average speed in m/s? "
            "Show your work.",
}


def cmd_benchmark(argv: list[str]) -> None:
    """Warm, reproducible sweep on this machine — the numbers behind the README table.
    Runs a greedy baseline first, then each requested mode/cap on the same prompts."""
    import json as _json
    import platform

    import mlx.core as mx

    from .generate import dflash_generate, greedy_generate, speculative_generate
    from .load import apply_wired_limit, load_dflash, load_drafter, load_target, resolve_mode
    from .lookup import lookup_generate

    ap = argparse.ArgumentParser(prog="mlx-dspark benchmark")
    ap.add_argument("--model", default=None, help="target repo/path (see `mlx-dspark models`)")
    ap.add_argument("--drafter", default=None)
    ap.add_argument("--modes", default="dspark,lookup",
                    help="comma-separated: dspark, dflash, lookup (baseline always runs)")
    ap.add_argument("--caps", default="2,auto",
                    help="comma-separated caps for dspark/dflash: ints and/or 'auto'")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--json", default=None, help="also write results to this JSON file")
    args = ap.parse_args(argv)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    caps = [c.strip() for c in args.caps.split(",") if c.strip()]
    apply_wired_limit()

    dev = mx.device_info().get("device_name", "?")
    print(f"mlx-dspark benchmark · {dev} · mlx {mx.__version__} · "
          f"{platform.machine()} {platform.system()}")

    _, target_repo, _ = resolve_mode(args.model, mode="auto", drafter=args.drafter)
    print(f"target: {target_repo}\nloading + warming up…")
    target, tok = load_target(
        target_repo, require_tap=any(m in ("dspark", "dflash") for m in modes))
    greedy_generate(target, tok, "Tell me about the sea.", max_new_tokens=100)

    results = {"device": dev, "mlx": mx.__version__, "target": target_repo, "runs": []}

    def run(label, fn):
        toks = tps = accept = 0.0
        for p in BENCH_PROMPTS.values():
            r = fn(p)
            toks += r.num_tokens
            tps += r.tokens_per_sec
            accept += r.mean_accept_len
        n = len(BENCH_PROMPTS)
        row = {"run": label, "tok_s": round(tps / n, 1), "accept": round(accept / n, 2)}
        results["runs"].append(row)
        base = results["runs"][0]["tok_s"]
        speedup = f"  ({row['tok_s'] / base:.2f}x)" if label != "baseline" else ""
        print(f"  {label:<22} {row['tok_s']:>7.1f} tok/s   accept {row['accept']:.2f}{speedup}")
        return row

    print(f"\n{'run':<24} {'tok/s':>7}")
    run("baseline", lambda p: greedy_generate(
        target, tok, p, max_new_tokens=args.max_new_tokens))

    for mode in modes:
        if mode == "lookup":
            run("lookup", lambda p: lookup_generate(
                target, tok, p, max_new_tokens=args.max_new_tokens))
            continue
        try:
            _, _, drafter_repo = resolve_mode(args.model, mode=mode, drafter=args.drafter)
        except ValueError as e:
            print(f"  {mode:<22} skipped ({e})")
            continue
        if mode == "dspark":
            drafter, _ = load_drafter(drafter_repo)
        else:
            drafter, _ = load_dflash(drafter_repo)
            drafter.bind(target.model)
        for cap in caps:
            ctrl = None
            md: int | None = None
            if cap == "auto":
                from .calibrate import calibrate

                ctrl = calibrate(target, drafter, mode=mode, target_repo=target_repo,
                                 drafter_repo=drafter_repo, verbose=False)
            else:
                md = int(cap)
            if mode == "dspark":
                run(f"dspark cap={cap}", lambda p: speculative_generate(
                    target, tok, drafter, p, max_new_tokens=args.max_new_tokens,
                    max_draft_tokens=md if md else None, cap_controller=ctrl))
            else:
                run(f"dflash cap={cap}", lambda p: dflash_generate(
                    target, tok, drafter, p, max_new_tokens=args.max_new_tokens,
                    max_draft_tokens=md if md else None, cap_controller=ctrl))
        drafter = None                     # release before the next mode's drafter loads

    if args.json:
        with open(args.json, "w") as f:
            _json.dump(results, f, indent=1)
        print(f"\nwrote {args.json}")


# --------------------------------------------------------------------------- models


def cmd_models(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="mlx-dspark models")
    ap.parse_args(argv)
    print("Targets whose drafter auto-resolves — pass any of these as --model and the matched\n"
          "DSpark/DFlash drafter is picked automatically (quantization-agnostic):\n")
    rows = [("target (--model)", "DSpark drafter", "DFlash drafter", "RAM")]
    for e in REGISTRY:
        dspark = e.get("dspark", "—")
        if dspark.startswith("gguf:"):     # PrismML GGUF drafter: converted on first use
            dspark = dspark[len("gguf:"):].rsplit("/", 1)[-1] + " (auto-converted)"
        rows.append((e["target"], dspark, e.get("dflash", "—"), e["ram"]))
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    for i, r in enumerate(rows):
        print("  " + "  ".join(c.ljust(widths[j]) for j, c in enumerate(r)))
        if i == 0:
            print("  " + "  ".join("-" * widths[j] for j in range(4)))
    print("\nAny other target works too — pass its HF repo/path as --model plus --drafter <repo>.\n"
          "Use --mode baseline for plain target decoding (no drafter).")


# --------------------------------------------------------------------------- doctor


def cmd_doctor(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="mlx-dspark doctor")
    ap.parse_args(argv)
    ok = True

    def check(label: str, good: bool, detail: str = ""):
        nonlocal ok
        ok = ok and good
        mark = "\033[32m✓\033[0m" if good else "\033[31m✗\033[0m"
        print(f"  {mark} {label}" + (f"  — {detail}" if detail else ""))

    import platform

    print("mlx-dspark doctor\n")
    is_mac = sys.platform == "darwin"
    is_arm = platform.machine() == "arm64"
    check("Apple Silicon (arm64 macOS)", is_mac and is_arm,
          f"{platform.system()} {platform.machine()}")

    # mlx exposes its version on mlx.core, not the top-level package
    for pkg, ver_attr in (("mlx", "mlx.core"), ("mlx_lm", "mlx_lm"), ("mlx_vlm", "mlx_vlm")):
        try:
            __import__(pkg)
            ver = getattr(__import__(ver_attr, fromlist=["__version__"]), "__version__", "?")
            check(f"{pkg} importable", True, ver)
        except Exception as e:  # noqa: BLE001
            check(f"{pkg} importable", False, str(e))

    try:
        import mlx.core as mx  # noqa: F401
        try:
            mx.zeros((2, 2))  # exercise the Metal path
            check("MLX Metal device works", True)
        except Exception as e:  # noqa: BLE001
            check("MLX Metal device works", False, str(e))
    except Exception:
        pass

    # gemma4 preset compat: mlx-vlm 0.6.4's Gemma4UnifiedProcessor breaks under
    # transformers>=5.12 (issue #4, upstream Blaizzy/mlx-vlm#1578); load_target shims it.
    try:
        from .load import _shim_gemma4_unified_processor
        if _shim_gemma4_unified_processor():
            print("  · mlx-vlm 0.6.4 Gemma4UnifiedProcessor is incompatible with this "
                  "transformers — shimmed at load time (upstream: Blaizzy/mlx-vlm#1578)")
    except Exception:  # noqa: BLE001
        pass

    total_gb = _total_ram_gb()
    if total_gb:
        check("System RAM", total_gb >= 15,
              f"{total_gb:.0f} GB (gemma4 preset ~15 GB, qwen3 ~8 GB)")

    # wired-limit hint: with big models, letting macOS page the weights mid-generation is
    # the classic silent slowdown; a raised iogpu limit keeps them resident
    try:
        import subprocess
        wired = int(subprocess.run(["sysctl", "-n", "iogpu.wired_limit_mb"],
                                   capture_output=True, text=True).stdout.strip() or 0)
        if wired:
            print(f"  · iogpu.wired_limit_mb = {wired}")
        elif total_gb and total_gb >= 16:
            print(f"  · tip: for large models, raise the GPU wired limit, e.g. "
                  f"`sudo sysctl iogpu.wired_limit_mb={int(total_gb * 0.75 * 1024)}` "
                  f"(mlx-dspark also wires the recommended working set at start)")
    except Exception:  # noqa: BLE001
        pass

    from . import __version__
    print(f"\nmlx-dspark {__version__} — {'ready' if ok else 'issues above'}.")
    if not ok:
        sys.exit(1)


def _total_ram_gb() -> float | None:
    try:
        import subprocess
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        return int(out.stdout.strip()) / (1024 ** 3)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- dispatch


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    sub = argv[0] if (argv and not argv[0].startswith("-")) else None
    if sub in ("-h", "--help", None) and not argv:
        # bare `mlx-dspark` -> show top-level help
        print(__doc__)
        return
    if sub == "serve":
        return cmd_serve(argv[1:])
    if sub == "models":
        return cmd_models(argv[1:])
    if sub == "doctor":
        return cmd_doctor(argv[1:])
    if sub == "benchmark":
        return cmd_benchmark(argv[1:])
    if sub == "generate":
        return cmd_generate(argv[1:])
    if sub in _SUBCOMMANDS:  # defensive; unreachable
        return
    # No subcommand (flags only) -> legacy flat generate CLI (backward compatible).
    return cmd_generate(argv)


if __name__ == "__main__":
    main()
