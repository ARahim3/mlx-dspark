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
: Inkling 2 only: run mlx-vlm's stock expert code instead of mlx-dspark's expert kernels, which read each expert's weights once for all the rows routed to it and do the routing and the route combine in one GPU launch each. Slower for decode steps and more so for verify passes. For comparison runs only.

`MLX_DSPARK_INKLING_DFLASH_HEAD_BITS`
: Inkling 2 only: the bits of the copy of the output head that the DFlash 2 head (`incoai/Inkling-2-Mini-DFlash2`) picks its draft candidates through on the released weights: 8 (default, about 0.5 GB), 4, or 0 for the bf16 head itself (slower: it reads 0.93 GB a round). With `--target-bits 8` the target's own 8-bit head is used and nothing is copied. Drafting only — the output is checked by the model either way; 4 bits measurably reorders the candidates. For comparison runs only.

`MLX_DSPARK_INKLING_EXPERT_VIEWS=0`
: Inkling 2 only: hand the expert kernels the full expert tensors instead of small views of them. MLX ends a GPU command buffer once the arrays bound in it pass a size limit, and one stacked expert tensor is over that limit on its own, so without the views every expert launch ends its buffer (about 1.4 ms of a 30 ms decode step). Same tokens either way. For comparison runs only.

`MLX_DSPARK_INKLING_FAST_HEAD=0`
: Inkling 2 only: compute the fp32 logits through mlx-vlm's stock path, which keeps a 1.85 GB fp32 copy of the output head, instead of mlx-dspark's kernel that reads the bf16 head directly. Slower and uses more memory. With `--target-bits 8` it switches the 8-bit head's kernel back to MLX's quantized product as well. For comparison runs only.

`MLX_DSPARK_INKLING_FUSED=0`
: Inkling 2 only: run mlx-vlm's stock layer, attention and expert code for decode steps and verify passes instead of mlx-dspark's fused forward, which replaces runs of small GPU operations with single kernels (a decode step launches about 2,100 GPU kernels instead of about 7,000). Slower. For comparison runs only.

`MLX_DSPARK_INKLING_KV_CACHE=0`
: Inkling 2 only: use mlx-vlm's stock key/value caches instead of mlx-dspark's, which keep the rows in order with the attention sink's row inside the buffer so the attention reads them in place. The stock caches copy the whole cached keys and values at every layer visit and again to roll a verify pass back. Same tokens either way; slower, and more so the longer the conversation. For comparison runs only.

`MLX_DSPARK_INKLING_MASK=0`
: Inkling 2 only: build the attention's relative-position mask with mlx-vlm's stock kernel and a concatenate for the sink's column, instead of mlx-dspark's single kernel that writes both. Same values bit for bit; slower, and more so the longer the conversation. For comparison runs only.

`MLX_DSPARK_INKLING_VERIFY_ATTN=0`
: Inkling 2 only: run every verify pass's attention through MLX's own attention kernel instead of mlx-dspark's, which reads the cached keys and values once per key/value head for all the rows of the pass (MLX's reads them once per row and query head). mlx-dspark's is used for verify passes of 3 to 12 rows once a layer sees 1,024 keys, which is where it is faster; a decode step always uses MLX's. The two agree within floating-point rounding, not bit for bit. Slower on long conversations. For comparison runs only.

`MLX_DSPARK_INKLING_WIDE_ROWS=0`
: Inkling 2 only: keep mlx-dspark's fused forward to verify passes of at most 8 rows (a drafter round), so a wider one — a lookup round checking up to 32 copied tokens at once — runs mlx-vlm's stock code as before. Slower for lookup drafts (`--lookup-drafts`); nothing else changes. For comparison runs only.

`MLX_DSPARK_INKLING_FOLD_BITS`
: Inkling 2 only: bits of the drafter's Markov correction matrix, which one GPU kernel scans per drafted token (default 4; 8 is the other choice, and 0 keeps the reference's bf16 matrix with its full-vocabulary bias, softcap and argmax). Drafting only — the model still checks every token. Acceptance is the same at 4, 8 and bf16; each draft round is about 1 ms cheaper than the reference path.

`MLX_DSPARK_INKLING_DRAFT_HEAD_BITS`
: Inkling 2 only: bits of the drafter's own copy of the output head, which its draft logits go through (default 6; 0 uses the model's bf16 head). Drafting only — the model still checks every token. 6 bits leaves acceptance unchanged and makes each draft round about 2 ms cheaper than the bf16 head; 4 bits is faster still but starts to cost acceptance. With `--target-bits 8` the drafter reads the model's own 8-bit head instead of a copy.

`MLX_DSPARK_STREAM_KEEPALIVE_S`
: Seconds between keep-alive frames on quiet streams (default 15). Lower it for aggressive proxies.

`MLX_DSPARK_PARENT_PID`
: `serve` only: a process ID to exit with. When that process ends, however it ends (a crash or a force-quit included), the server stops too. The Mac app sets it to its own ID, so a crashed app can't leave its engine running and holding the port.

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
