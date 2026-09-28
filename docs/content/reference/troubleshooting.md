---
title: Troubleshooting
description: The problems people actually hit, and what to do about each. Start with mlx-dspark doctor.
---

Start here:

```bash
mlx-dspark doctor --models
```

It checks the chip, the MLX stack and your memory, lists every folder the engine searches for models, and shows which supported models fit and are already downloaded. When filing an issue, include `mlx-dspark doctor --json`.

## Speed

### My speedup is lower than the published numbers

In rough order of likelihood:

- **The content.** Open-ended chat accepts fewer drafted tokens than code or math: expect the low end of a model's range there.
- **A hot or throttled machine.** MacBooks in automatic energy mode throttle sustained speculative rounds more than plain decoding. Use High Power mode for measurements, and don't run anything else on the GPU (a download, another model).
- **A calibration taken on a hot GPU.** The first-run measurement is cached, and one taken right after heavy use reads high and can pick a too-low cap. Delete the model's entries from `~/.cache/mlx_dspark/calibration.json` and run it again from cold.
- **Long context.** At 16–32k tokens, speedups are lower for structural reasons; see [Long context](/concepts/long-context/).
- **Memory pressure.** If macOS is swapping, everything is slow. `/health` and the serve log show a warning; see below.
- **Your chip.** Numbers here are an M4 Pro's. Compare ratios with `mlx-dspark benchmark --trials 3`, not absolute tokens per second.

### Acceptance is stuck near 1.2 tokens per round

Something is mismatched: a **base** model instead of the instruct one the drafter was trained against, a very different quantization, or a new drafter packaging. Use the matched instruct model, and check [Bring your own drafter](/models/bring-your-own/#getting-a-good-pairing).

### I'm on an M5 and speculation is slower than I expected

Two of the custom kernels are switched off on M5 and newer: the small-M verify kernel stalled sustained generation there, and the attention kernel isn't verified yet. For a comparison run, `MLX_DSPARK_FORCE_SMALL_M=1` and `MLX_DSPARK_FORCE_MULTIROW=1` enable them. Paired numbers from an M5 are very welcome in an issue.

## Memory

### The Mac swaps or the model slows to a crawl

The model plus its KV cache doesn't fit. In order of effect:

1. **Cap the context:** `--context-window 32768` (or whatever fits). `serve` prints a suggested value at load when the full window wouldn't fit.
2. **Use a smaller build:** the 4-bit build of the same model resolves the same drafter.
3. **`--kv-bits 8`** halves the attention cache.

The memory guard (on by default) sheds caches under pressure, but it can't make a model that doesn't fit fast. Don't reach for `--wired-limit`: on a machine that's already short of memory it can hang macOS.

### The server crashed during prompt reading (SIGSEGV in BNNS)

That crash came from the bf16 CPU route of CPU co-prefill. Since v0.19.0 the CPU share runs in fp32 through Accelerate by default, which avoids it. If you set `MLX_DSPARK_CPU_SPLIT_FP32=0`, unset it. `--cpu-split 0` turns co-prefill off entirely. If it still happens with the defaults, please report it with the crash log.

## Loading

### "prompt is too long"

The request is longer than the context window: the model's own limit, or the `--context-window` you set. Agent clients like Claude Code compact and retry on this message automatically. Raise `--context-window` if you have the memory.

### A model is refused because it "ships its own Python"

Its `config.json` has `model_file` or `auto_map`, and loading it could run that code. If you trust the repo, start with `--trust-remote-code` (or `MLX_DSPARK_TRUST_REMOTE_CODE=1`). [Nanbeige4.2-3B](/models/nanbeige4.2-3b/) needs this on a fresh download, although nothing is actually imported for it.

### It wants to download a model I already have

- **From LM Studio:** only its **MLX** downloads are reused, and the `--model` id must match LM Studio's `publisher/model` exactly. GGUF files aren't readable.
- **In your own folders:** add them to `MLX_DSPARK_MODEL_DIRS`.
- `mlx-dspark doctor --json` shows every folder searched, in order.

### Gemma-4 fails with "Can't load video processor"

That's an old `mlx-vlm` 0.6.4 × `transformers` ≥ 5.12 incompatibility. Upgrade: `pip install -U mlx-dspark mlx-vlm`.

### The 1-bit Bonsai model is refused

Speculation is a net loss on the 1-bit pack (0.71–0.77×), so mlx-dspark points you to plain generation with `mlx-vlm` instead. The ternary 2-bit build is the one to use with a drafter.

## Clients and agents

### `mlx-dspark claude` says there's no server

Start one in another terminal (`mlx-dspark serve --model …`). If it's on another port or machine, pass `--url http://host:port`; if it has a key, pass `--api-key`.

### Claude Code still uses my subscription

A base URL alone isn't enough: Claude Code keeps authenticating with your claude.ai login unless `ANTHROPIC_AUTH_TOKEN` is set. `mlx-dspark claude` sets it for you. If you configure it by hand, see [the variables](/use/claude-code/#wiring-it-up-yourself).

### The client times out during a long prompt

Keep-alive frames go out every 15 seconds on both OpenAI and Anthropic streams, so most clients won't time out. For a proxy with a shorter idle timeout, lower `MLX_DSPARK_STREAM_KEEPALIVE_S`. `GET /events` streams prefill progress if you want to watch it.

### Thinking shows up inside the answer, or tool calls come back as text

Both should be split out on every dialect. If you see either, please open an issue with the model, the client, and whether it was streaming. For tool calls, first make sure the model can call tools at all; the server translates six native tool-call formats, but a model that doesn't emit one can't be parsed.

## Output

### The output differs from plain decoding at some token

That's a floating-point tie: at that position the model's top two tokens scored equal to the last bit, and verifying several tokens at once breaks the tie differently from decoding one at a time. From there both texts are equally the model's own greedy output. Batched serving on any engine has the same property. See [Is it really lossless?](/concepts/speculative-decoding/#is-it-really-lossless)
