#!/usr/bin/env python
"""Measure PREFILL (prompt-processing) throughput for a target model.

Prefill is a target-only property (independent of the drafter / speculative loop),
so we measure the plain greedy baseline. ``greedy_generate`` splits the clock at the
prefill->decode boundary (``mx.eval(logits); t_prefill``), so:

    prefill tok/s = len(prompt_ids) / result.prefill_seconds

We feed a controlled ``prompt_ids`` of exact length (content is ~irrelevant to prefill
throughput), warm up first (kernel compile + clock ramp otherwise land in the first
prefill), and take the median over trials — this M4 Pro has ~14% between-trial noise.

The reported prefill figures are flat across lengths, so the ~3.5k-token row is the one
comparable to the README "Prompt processing (prefill)" table.

Usage:
    python scripts/prefill.py <model>                    # registry id, HF repo, or local path
    python scripts/prefill.py lfm2.5-1.2b
    python scripts/prefill.py mlx-community/Qwen3-4B-8bit --lengths 2048,4096 --trials 5
"""
import argparse
import os
import statistics as st
import time

import mlx.core as mx

from mlx_dspark.generate import greedy_generate
from mlx_dspark.load import _registry_entry, load_target


def main() -> None:
    ap = argparse.ArgumentParser(prog="prefill.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="registry id (see `mlx-dspark models`), HF repo, or local path")
    ap.add_argument("--lengths", default="512,2048,4096",
                    help="comma-separated prompt token counts to sweep (default 512,2048,4096)")
    ap.add_argument("--trials", type=int, default=3,
                    help="repeat each length N times and report the MEDIAN (default 3; use 3+ — "
                         "between-trial noise is ~14%% on an M4 Pro)")
    ap.add_argument("--max-new-tokens", type=int, default=8,
                    help="decode length (only used to also report decode tok/s; prefill is "
                         "unaffected). Default 8.")
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]

    # Accept a bare registry id (e.g. "lfm2.5-1.2b") as a convenience by mapping it to the
    # target repo; a full HF repo or local path passes straight through to load_target (which
    # handles the ~/.cache/mlx_dspark and hub resolution itself).
    target_repo = args.model
    entry = _registry_entry(args.model)
    if entry is not None and entry["id"] == os.path.basename(args.model.rstrip("/")).lower():
        target_repo = entry["target"]

    print(f"# {mx.device_info().get('device_name', '?')} · mlx {mx.__version__}")
    print(f"# target: {target_repo}")
    t0 = time.time()
    target, tok = load_target(target_repo, require_tap=False)
    print(f"# loaded in {time.time() - t0:.1f}s")

    # A pool of real token ids to tile into controlled-length prompts.
    pool = tok.encode("The quick brown fox jumps over the lazy dog. " * 200)

    def make_ids(n: int) -> list[int]:
        return (pool * ((n // len(pool)) + 1))[:n]

    # Warm up: past kernel compile + clock ramp before the timed runs.
    greedy_generate(target, tok, "Tell me about the sea.", max_new_tokens=32)
    for n in lengths:
        greedy_generate(target, tok, prompt_ids=make_ids(n), max_new_tokens=2)

    print(f"\n{'prompt_tok':>10} {'prefill_s':>10} {'prefill_tok/s':>14} {'decode_tok/s':>13}")
    for n in lengths:
        ids = make_ids(n)
        pf, dtps = [], []
        for _ in range(max(1, args.trials)):
            r = greedy_generate(target, tok, prompt_ids=ids, max_new_tokens=args.max_new_tokens)
            pf.append(r.prefill_seconds)
            dtps.append(r.decode_tokens_per_sec)
        pf_med = st.median(pf)
        print(f"{n:>10} {pf_med:>10.4f} {n / pf_med:>14.1f} {st.median(dtps):>13.1f}")


if __name__ == "__main__":
    main()
