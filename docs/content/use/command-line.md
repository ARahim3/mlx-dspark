---
title: Command line
description: One-shot generation, head-to-head comparisons, and the benchmark that produced every number on this site. Run it on your own Mac.
---

Six subcommands. This page is the tour; every flag is in the [CLI reference](/reference/cli/).

| Command | What it does |
|---|---|
| `mlx-dspark generate` | One prompt, one answer, printed as it streams. |
| `mlx-dspark serve` | The API server. See [Run a server](/use/server/). |
| `mlx-dspark claude` | Launch Claude Code against a running server. See [Claude Code](/use/claude-code/). |
| `mlx-dspark benchmark` | A warm, reproducible speed sweep on this Mac. |
| `mlx-dspark models` | The measured models, with their drafters and memory. |
| `mlx-dspark doctor` | Check the environment and which models fit. |

`python -m mlx_dspark …` works too, and a bare `mlx-dspark --prompt …` still means `generate`.

## Generate

```bash
# the drafter and your Mac's settings resolve on first run
mlx-dspark generate --model mlx-community/Qwen3-4B-8bit --prompt "Explain how rainbows form."

# the same prompt, plain and speculative: same text, different speed
mlx-dspark generate --model mlx-community/Qwen3-4B-8bit --mode baseline --prompt "…" --max-new-tokens 400
mlx-dspark generate --model mlx-community/Qwen3-4B-8bit --mode dspark   --prompt "…" --max-new-tokens 400

# z-lab's DFlash drafter at its native full block
mlx-dspark generate --model mlx-community/gemma-4-12B-it-8bit --mode dflash --max-draft 0 \
  --prompt "Write a binary search."

# sampled instead of greedy, still lossless with respect to the model
mlx-dspark generate --model mlx-community/Qwen3-4B-8bit --prompt "Write a short poem." \
  --temperature 1.0 --top-p 0.95 --seed 0

# any model at all, no drafter: n-gram lookup speculation
mlx-dspark generate --model mlx-community/Mistral-7B-Instruct-v0.3-4bit --mode lookup --prompt "…"
```

The modes:

`auto` (default)
: The pair's measured best: the registry's stamped mode (DFlash 2 on Qwen3.8-27B), otherwise DSpark, then DFlash, then drafter-free lookup. Any model runs.

`dspark`
: DeepSeek's DSpark drafter.

`dflash`
: z-lab's DFlash or Inco AI's DFlash 2.

`lookup`
: Drafter-free prompt-lookup speculation, for any model.

`baseline`
: Plain decoding, for comparison.

## Benchmark

`benchmark` is the same sweep the results on this site come from. It warms up first, runs three prompts (chat, code, math), and reports each mode against plain decoding on the same machine:

```bash
mlx-dspark benchmark --model mlx-community/Qwen3-8B-8bit --trials 3
```

Useful options:

`--modes dspark,dflash,lookup`
: Which methods to race against the baseline (the baseline always runs).

`--caps 2,4,7,auto`
: Sweep draft caps, to see your Mac's curve. Omit it to use the derived default.

`--trials N`
: Repeat each prompt and report the median. Run-to-run noise is about ±14% on a laptop, so a single trial isn't worth quoting. Use 3 or more.

`--json results.json`
: Also write a device-stamped result file. If you measure a pair that isn't on the [models page](/models/), please share it in an issue.

The header prints which kernels and options were active, so two runs can't be confused.

### Getting numbers you can trust

- **Warm and idle.** Close heavy apps and don't run it during a download.
- **High Power mode on laptops.** Under sustained load, a MacBook in automatic energy mode throttles speculative rounds more than plain decoding, which understates the speedup.
- **Compare ratios within one run,** not absolute tok/s across runs.
- **Don't benchmark right after the first-run calibration.** A hot GPU reads about 10% low.

## Models and doctor

```bash
mlx-dspark models           # measured targets, their drafters, memory
mlx-dspark doctor           # Apple Silicon, MLX stack, memory, model folders
mlx-dspark doctor --models  # plus: which registry models fit and are downloaded
mlx-dspark doctor --json    # the same report as the server's /doctor
```

`doctor` lists every folder the engine searches for models, in order. It's the first thing to check when the engine wants to download something you already have.
