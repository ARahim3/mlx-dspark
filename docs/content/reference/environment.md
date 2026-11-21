---
title: Environment and files
nav_title: Environment and files
description: The environment variables the engine reads, and where it keeps things on disk.
---

## Environment variables

`MLX_DSPARK_MODEL_DIRS`
: Extra folders to search for MLX models before downloading, `:`-separated, `~` expanded. Each folder is tried as `publisher/model`, `publisher_model` and a bare `model` folder. They appear in `mlx-dspark models` and the app's list.

`MLX_DSPARK_TRUST_REMOTE_CODE=1`
: Allow checkpoints that ship their own Python (`model_file` / `auto_map` in their config) to be imported, for the whole process. Same as `--trust-remote-code`. Off by default.

`MLX_DSPARK_CPU_SPLIT_FP32=0`
: Run the CPU share of [CPU co-prefill](/concepts/prefill/) in bf16 through BNNS instead of fp32 through Accelerate BLAS. Faster, but the BNNS route was behind every co-prefill crash ever reported; only for machines that never reach memory pressure.

`MLX_DSPARK_FORCE_SMALL_M=1`
: Enable the small-M verify kernel on M5 and newer, where it's gated off. For comparison runs only.

`MLX_DSPARK_FORCE_MULTIROW=1`
: Enable the multi-row attention kernel on M5 and newer, where it's gated off until verified. For comparison runs only.

`MLX_DSPARK_FORCE_TREE_KERNEL=1`
: Enable the tree-verify recurrence kernel (DFlash 2 trees on gated-DeltaNet targets) on M5 and newer, where it's gated off until verified. Without it those GPUs run the exact per-path fallback, and `--tree auto` measures that cost and usually keeps the chain. For comparison runs only.

`MLX_DSPARK_EAGER_COMMIT=1`
: Rebuild a hybrid target's recurrence state at every accept instead of folding the accepted rows into the next verify (the default since the deferred commit). Same tokens, a little slower. For comparison runs only.

`MLX_DSPARK_INKLING_GROUPED_MOE=0`
: Inkling 2 only: verify draft rows with mlx-vlm's stock expert path instead of mlx-dspark's grouped kernel, which loads each expert once for every draft row routed to it. Slower at every draft length. For comparison runs only.

`MLX_DSPARK_INKLING_FAST_HEAD=0`
: Inkling 2 only: compute the fp32 logits through mlx-vlm's stock path, which keeps a 1.85 GB fp32 copy of the output head, instead of mlx-dspark's kernel that reads the bf16 head directly. Slower and uses more memory. For comparison runs only.

`MLX_DSPARK_INKLING_DRAFT_HEAD_BITS`
: Inkling 2 only: bits of the drafter's own copy of the output head, which its draft logits go through (default 8; 0 uses the model's bf16 head). Drafting only — the model still checks every token. 8 bits leaves acceptance unchanged and makes each draft round about 1.7 ms cheaper; 4 bits is faster still but starts to cost acceptance.

`MLX_DSPARK_STREAM_KEEPALIVE_S`
: Seconds between keep-alive frames on quiet streams (default 15). Lower it for aggressive proxies.

`MLX_DSPARK_SLOW_ROUND_LOG_S`
: Log any gap between rounds longer than this many seconds to stderr, with mode, cap and context (default 10; 0 disables). Useful when reporting a stall.

`HF_HOME`, `HF_HUB_CACHE`
: Where the Hugging Face cache lives. Honoured everywhere, including the "is it installed?" checks.

## Files

`~/.cache/mlx_dspark/calibration.json`
: This Mac's measured curves per model, quantization and MLX version: verify cost by width, drafter cost, depth slope, kernel races and the CPU co-prefill fraction. Safe to delete; it re-measures on the next run. See [the calibration cache](/concepts/draft-caps/#the-calibration-cache).

`~/.cache/mlx_dspark/models/`
: Plain model folders the engine looks in before Hugging Face: converted drafters (like the Bonsai GGUF repack) and anything you put there by hand.

`~/.lmstudio/models/`, `~/.cache/lm-studio/models/`
: LM Studio's folders. Its MLX downloads load in place.

`--prefix-cache-dir`
: Where the optional SSD tier of the prefix cache spills, when you enable it.

The Mac app keeps its runtime, logs and chats under its own Application Support folder, outside the app bundle and separate from any `pip` install.
