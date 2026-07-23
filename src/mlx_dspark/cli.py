"""mlx-dspark command line.

Subcommands:
  serve      Start the API server: OpenAI (/v1/chat/completions) **and** Anthropic
             (/v1/messages) on the same port, so LM Studio clients, the openai SDK, curl,
             and Claude Code all talk to it.
  claude     Launch Claude Code against a running ``serve``, configured for this process
             only — your normal ``claude`` sessions are untouched.
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
import json
import os
import shutil
import sys
import urllib.error
import urllib.request

from .load import REGISTRY

_SUBCOMMANDS = ("serve", "claude", "generate", "benchmark", "models", "doctor")


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
    ap.add_argument("--wired-limit", action="store_true",
                    help="wire MLX's recommended working set (~75%% of RAM) so weights "
                         "can't be paged out. OFF by default and rarely worth it: wired "
                         "pages cannot be reclaimed, so on a machine whose working set is "
                         "already large this can HANG macOS hard enough to need a power "
                         "cycle (observed on an M4 Pro). Short of that it has corrupted the "
                         "verify logits on the gemma-4/mlx-vlm route, which can commit "
                         "wrong tokens rather than crash. It bought no measurable speed "
                         "where tested (<1%%). Only consider it if the model nearly fills "
                         "your RAM and you actually see paging stalls - and validate a long "
                         "run before trusting it.")
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
    ap.add_argument("--lookup-long-draft", type=int, default=32,
                    help="match-scaled long-draft ceiling for lookup drafts (dspark hybrid "
                         "+ lookup mode): a deep context match (real copy run) earns drafts "
                         "up to this length (default 32 = the measured M-series verify-width "
                         "plateau); set to the base (6) to disable. Speed-only, output "
                         "unchanged")
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

    if args.wired_limit:
        apply_wired_limit()
    on_text = None if args.no_stream else _emit
    print("\n" + "=" * 64)
    print(f"  ▶  {label}   ·   {target_repo.split('/')[-1]}")
    print("=" * 64)

    if args.mode == "dspark":
        cap = max_draft
        if max_draft is None and cap_controller is None:
            # No explicit --max-draft: derive the cap from THIS machine+model+quant's
            # measured curves (one-time ~5 s, disk-cached) instead of a hardcoded 2, which
            # is only right for a curve shape mlx 0.31.2 had. See calibrate.static_cap.
            from .calibrate import static_cap

            cap = static_cap(target, drafter, mode="dspark", target_repo=target_repo,
                             drafter_repo=drafter_repo)
            print(f"  cap {cap} (measured for this model+quant; override with --max-draft)")
        res = speculative_generate(
            target, tok, drafter, args.prompt,
            max_new_tokens=args.max_new_tokens, max_draft_tokens=cap,
            cap_controller=cap_controller, lookup_drafts=not args.no_lookup_drafts,
            lookup_long_draft=args.lookup_long_draft,
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
            long_draft_tokens=max(cap, args.lookup_long_draft),
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
    from .server import Engine, EngineHolder, maybe_batch_engine, run_server

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
    ap.add_argument("--context-window", type=int, default=None,
                    help="cap the prompt length (default: the target's own trained limit). "
                         "An over-long request is refused with the message Claude Code "
                         "recognises as a context limit, so it compacts and retries instead "
                         "of failing; lower it to keep the KV cache inside your RAM budget.")
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
    ap.add_argument("--wired-limit", action="store_true",
                    help="wire MLX's recommended working set (~75%% of RAM) so weights "
                         "can't be paged out. OFF by default and rarely worth it: wired "
                         "pages cannot be reclaimed, so on a machine whose working set is "
                         "already large this can HANG macOS hard enough to need a power "
                         "cycle (observed on an M4 Pro). Short of that it has corrupted the "
                         "verify logits on the gemma-4/mlx-vlm route, which can commit "
                         "wrong tokens rather than crash. It bought no measurable speed "
                         "where tested (<1%%). Only consider it if the model nearly fills "
                         "your RAM and you actually see paging stalls - and validate a long "
                         "run before trusting it.")
    ap.add_argument("--prefix-cache-slots", type=int, default=2,
                    help="number of conversations kept in the prefix cache LRU (default 2, "
                         "so an agent and a chat don't evict each other every turn)")
    ap.add_argument("--no-lookup-drafts", action="store_true",
                    help="disable hybrid n-gram drafting inside dspark mode")
    ap.add_argument("--lookup-long-draft", type=int, default=32,
                    help="match-scaled long-draft ceiling for lookup drafts (default 32; "
                         "set to 6 to disable — see `mlx-dspark generate -h`)")
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
    # Captured so a `/admin/load` model swap re-loads with the same server flags, changing only
    # the model — an in-place swap that keeps the port instead of a full restart.
    load_kwargs = dict(
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
        lookup_long_draft=args.lookup_long_draft,
        wired_limit=args.wired_limit,
        batch_widths=(sorted({2, args.max_batch}) if args.max_batch > 1 else None),
        kv_bits=args.kv_bits or None,
        context_window=args.context_window,
    )
    try:
        engine = Engine.load(**load_kwargs)
    except ValueError as e:
        ap.error(str(e))
    holder = EngineHolder(maybe_batch_engine(engine, args.max_batch),
                          load_kwargs, max_batch=args.max_batch)
    run_server(holder, host=args.host, port=args.port, api_key=args.api_key)


# --------------------------------------------------------------------------- claude code


def _probe_health(base: str, api_key: str | None, timeout: float = 3.0) -> dict:
    """``GET {base}/health`` -> the server's self-description (model id, context window)."""
    req = urllib.request.Request(base.rstrip("/") + "/health")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def claude_env(base: str, health: dict, api_key: str | None) -> dict:
    """The environment that points one Claude Code process at this server.

    Every name here is Claude Code's documented configuration surface, and all of it lives in
    the *child process* — see :func:`cmd_claude` for why that matters.

    * ``ANTHROPIC_BASE_URL`` sends requests here. On its own it is **not** enough: without a
      credential variable Claude Code keeps using your saved claude.ai login for auth, which
      means the session still counts against your subscription. ``ANTHROPIC_AUTH_TOKEN`` is
      what actually takes over (and takes effect immediately — ``ANTHROPIC_API_KEY`` would
      instead prompt for one-time approval), so we always set it, placeholder or not.
    * ``ANTHROPIC_MODEL`` plus the per-alias ``ANTHROPIC_DEFAULT_*_MODEL`` variables point the
      session model *and* every alias — including the ``haiku`` slot Claude Code uses for
      background work like conversation titles — at the one model that's loaded. Without the
      haiku mapping those background calls ask for a model this server has never heard of.
    * ``CLAUDE_CODE_MAX_OUTPUT_TOKENS`` keeps requested output under the server's own cap.
    * ``CLAUDE_CODE_AUTO_COMPACT_WINDOW`` triggers compaction below the model's real window —
      but Claude Code clamps it to at least 100k, so setting it under that does nothing and we
      skip it. Smaller windows are handled by the server's ``prompt is too long`` reply, which
      Claude Code recognises and recovers from by compacting.
    """
    model = health.get("model") or "local"
    label = f"{model} (mlx-dspark)"
    desc = f"Local {health.get('mode', 'dspark')} on Apple Silicon via mlx-dspark"
    env = dict(os.environ)
    # A stale key/provider selection in the ambient environment would otherwise override or
    # bypass the base URL we just set. Drop them for this child only.
    for stale in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
                  "CLAUDE_CODE_USE_FOUNDRY", "CLAUDE_CODE_USE_MANTLE",
                  "CLAUDE_CODE_USE_ANTHROPIC_AWS"):
        env.pop(stale, None)
    env["ANTHROPIC_BASE_URL"] = base.rstrip("/")
    env["ANTHROPIC_AUTH_TOKEN"] = api_key or "mlx-dspark"
    env["ANTHROPIC_MODEL"] = model
    for alias in ("OPUS", "SONNET", "HAIKU", "FABLE"):
        env[f"ANTHROPIC_DEFAULT_{alias}_MODEL"] = model
        env[f"ANTHROPIC_DEFAULT_{alias}_MODEL_NAME"] = label
        env[f"ANTHROPIC_DEFAULT_{alias}_MODEL_DESCRIPTION"] = desc
    env["ANTHROPIC_SMALL_FAST_MODEL"] = model      # pre-2.x alias for DEFAULT_HAIKU_MODEL
    out_cap = health.get("max_output_tokens")
    if isinstance(out_cap, int) and out_cap > 0:
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(out_cap)
    window = health.get("context_window")
    if isinstance(window, int) and window >= 100_000:
        env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(int(window * 0.8))
    return env


def cmd_claude(argv: list[str]) -> None:
    """Run Claude Code against a local mlx-dspark server, scoped to this process.

    The whole design goal is that using it costs you nothing afterwards. Configuration is
    passed as the launched process's environment and nothing else: no shell profile is edited,
    no ``settings.json`` is written, no login is replaced. Your other Claude Code sessions —
    running now or started later — keep their normal account, model, and endpoint, and this
    one reverts the moment it exits. ``--print-env`` dumps the same variables if you'd rather
    wire them in yourself.
    """
    ap = argparse.ArgumentParser(
        prog="mlx-dspark claude",
        description="Launch Claude Code against a running `mlx-dspark serve`. Configures the "
                    "launched process only — other Claude Code sessions are unaffected.",
        epilog="Anything after `--` is passed straight to claude, e.g. "
               "`mlx-dspark claude -- --continue`.")
    ap.add_argument("--url", default="http://127.0.0.1:8080",
                    help="base URL of the running server (default http://127.0.0.1:8080)")
    ap.add_argument("--api-key", default=None,
                    help="the server's --api-key, if it was started with one")
    ap.add_argument("--print-env", action="store_true",
                    help="print the environment as shell exports instead of launching")
    ap.add_argument("--print-settings", action="store_true",
                    help="print a .claude/settings.local.json 'env' block instead of "
                         "launching (project-scoped: applies to that project only)")
    args, passthrough = ap.parse_known_args(argv)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    try:
        health = _probe_health(args.url, args.api_key)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            ap.error(f"{args.url} rejected the credential — pass the server's --api-key value "
                     f"with --api-key")
        ap.error(f"{args.url}/health returned HTTP {e.code}")
    except (urllib.error.URLError, OSError, ValueError):
        ap.error(f"no mlx-dspark server at {args.url}.\n"
                 f"  Start one in another terminal, e.g.:\n"
                 f"    mlx-dspark serve --model mlx-community/Qwen3-8B-8bit\n"
                 f"  then re-run `mlx-dspark claude`.")

    env = claude_env(args.url, health, args.api_key)
    added = {k: v for k, v in env.items() if os.environ.get(k) != v}

    if args.print_settings:
        print(json.dumps({"env": {k: v for k, v in added.items()
                                  if k.startswith(("ANTHROPIC_", "CLAUDE_CODE_"))}}, indent=2))
        return
    if args.print_env:
        for k, v in sorted(added.items()):
            print(f"export {k}={v!r}")
        return

    exe = shutil.which("claude")
    if exe is None:
        ap.error("`claude` not found on PATH. Install Claude Code "
                 "(https://claude.com/claude-code), or use --print-env / --print-settings.")
    print(f"claude code -> {args.url}  ·  model {health.get('model')} "
          f"({health.get('mode')})  ·  this session only", flush=True)
    # exec (not spawn): Claude Code owns the TTY, signals, and the exit code directly.
    os.execve(exe, [exe, *passthrough], env)


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
    ap.add_argument("--trials", type=int, default=1,
                    help="repeat each prompt N times and report the MEDIAN. Between-trial "
                         "noise is ~14%% on an M4 Pro (machine state, not content), so a "
                         "single trial is not quotable — use 3+ for README numbers.")
    ap.add_argument("--wired-limit", action="store_true",
                    help="wire MLX's recommended working set (~75%% of RAM) so weights "
                         "can't be paged out. OFF by default and rarely worth it: wired "
                         "pages cannot be reclaimed, so on a machine whose working set is "
                         "already large this can HANG macOS hard enough to need a power "
                         "cycle (observed on an M4 Pro). Short of that it has corrupted the "
                         "verify logits on the gemma-4/mlx-vlm route, which can commit "
                         "wrong tokens rather than crash. It bought no measurable speed "
                         "where tested (<1%%). Only consider it if the model nearly fills "
                         "your RAM and you actually see paging stalls - and validate a long "
                         "run before trusting it.")
    ap.add_argument("--json", default=None, help="also write results to this JSON file")
    args = ap.parse_args(argv)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    caps = [c.strip() for c in args.caps.split(",") if c.strip()]
    if args.wired_limit:
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
        """Per-prompt medians over --trials, plus the across-prompt mean. The per-prompt
        split is what the README's 'N.NNx code · N.NNx chat' figures need — averaging the
        three prompts into one number (the old behaviour) cannot produce them."""
        import statistics as _st

        per: dict[str, dict] = {}
        for name, p in BENCH_PROMPTS.items():
            tp, ac = [], []
            for _ in range(max(1, args.trials)):
                r = fn(p)
                tp.append(r.tokens_per_sec)
                ac.append(r.mean_accept_len)
            per[name] = {"tok_s": round(_st.median(tp), 1),
                         "accept": round(_st.median(ac), 2)}
        n = len(BENCH_PROMPTS)
        row = {"run": label,
               "tok_s": round(sum(v["tok_s"] for v in per.values()) / n, 1),
               "accept": round(sum(v["accept"] for v in per.values()) / n, 2),
               "per_prompt": per}
        results["runs"].append(row)
        base_row = results["runs"][0]
        speedup = f"  ({row['tok_s'] / base_row['tok_s']:.2f}x)" if label != "baseline" else ""
        print(f"  {label:<22} {row['tok_s']:>7.1f} tok/s   accept {row['accept']:.2f}{speedup}")
        detail = []
        for name, v in per.items():
            sp = (f" {v['tok_s'] / base_row['per_prompt'][name]['tok_s']:.2f}x"
                  if label != "baseline" else "")
            detail.append(f"{name} {v['tok_s']:.1f}{sp}")
        print(f"  {'':22} {'  ·  '.join(detail)}")
        return row

    print(f"\n{'run':<24} {'tok/s':>7}   (per prompt below each row)")
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
    """Environment + model-inventory check.

    Rendering only — every fact comes from :mod:`diagnostics`, which is also what the server
    serves at ``GET /doctor``. One source of truth, so the app and the CLI can never disagree
    about whether this machine is set up correctly.
    """
    ap = argparse.ArgumentParser(prog="mlx-dspark doctor")
    ap.add_argument("--json", action="store_true",
                    help="emit the full report as JSON (same payload as the server's /doctor)")
    ap.add_argument("--models", action="store_true",
                    help="also list which registry models fit this Mac and are already downloaded")
    args = ap.parse_args(argv)

    from .diagnostics import doctor

    report = doctor()
    if args.json:
        import json as _json

        print(_json.dumps(report, indent=1))
        sys.exit(0 if report["ok"] else 1)

    env = report["environment"]

    def check(label: str, good: bool, detail: str = ""):
        mark = "\033[32m\u2713\033[0m" if good else "\033[31m\u2717\033[0m"
        print(f"  {mark} {label}" + (f"  \u2014 {detail}" if detail else ""))

    print("mlx-dspark doctor\n")
    check("Apple Silicon (arm64 macOS)", env["apple_silicon"],
          f"{env['platform']} {env['machine']}")
    for pkg in ("mlx", "mlx_lm", "mlx_vlm"):
        version = env["packages"].get(pkg)
        check(f"{pkg} importable", version is not None, version or "not installed")
    check("MLX Metal device works", env["metal_ok"], env["device"] or env["metal_error"] or "")
    if env["ram_gb"]:
        check("System RAM", env["ram_gb"] >= 15,
              f"{env['ram_gb']:.0f} GB (gemma4 preset ~15 GB, qwen3 ~8 GB)")

    # gemma4 preset compat: mlx-vlm 0.6.4's Gemma4UnifiedProcessor breaks under
    # transformers>=5.12 (issue #4, upstream Blaizzy/mlx-vlm#1578); load_target shims it.
    try:
        from .load import _shim_gemma4_unified_processor

        if _shim_gemma4_unified_processor():
            print("  \u00b7 mlx-vlm 0.6.4 Gemma4UnifiedProcessor is incompatible with this "
                  "transformers \u2014 shimmed at load time (upstream: Blaizzy/mlx-vlm#1578)")
    except Exception:  # noqa: BLE001
        pass

    if env["iogpu_wired_limit_mb"]:
        print(f"  \u00b7 iogpu.wired_limit_mb = {env['iogpu_wired_limit_mb']}")
    elif env["wired_limit_hint"]:
        print(f"  \u00b7 tip: for large models, raise the GPU wired limit, e.g. "
              f"`{env['wired_limit_hint']}` "
              f"(mlx-dspark also wires the recommended working set at start)")

    if args.models:
        print("\n  models (fits this Mac / already downloaded):")
        for row in report["models"]:
            fits = "\u2713" if row["fits"] else ("?" if row["fits"] is None else "\u2717")
            state = "ready" if row["ready"] else (
                "drafter missing" if row["target_installed"] else "not downloaded")
            print(f"    {fits} {row['id']:<20} {row['ram'] or '':<8} {state}")

    for problem in report["problems"]:
        print(f"\n  \u2717 {problem}")

    print(f"\n\u2014 mlx-dspark {env['version']} "
          f"{'ready' if report['ok'] else 'issues above'}.")
    if not report["ok"]:
        sys.exit(1)



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
    if sub == "claude":
        return cmd_claude(argv[1:])
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
