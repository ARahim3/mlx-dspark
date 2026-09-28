---
title: Choose a model
description: Tell it your Mac's memory and what you care about. Below the picker, the reasoning behind it, so you can choose for yourself.
scripts: widgets
---
{% import "macros.html" as ui %}

{{ ui.picker(32) }}

## By the Mac you have

These are the picks the measurements support. Speedups are the M4 Pro's; your Mac derives its own settings on first run.

8 GB
: [MiniCPM5-2B](/models/minicpm5-2b/) (bf16, ~6 GB): 3.16× mean and 107–213 tok/s, and a tool-calling model. [LFM2.5-1.2B](/models/lfm2.5-1.2b/) (~4 GB) is faster still in raw tokens per second.

16 GB
: [Qwen3-8B](/models/qwen3-8b/) (8-bit, ~11 GB) is the proven all-rounder and the best local model for Claude Code. [Ornith-1.0-9B](/models/ornith-1.0-9b/) (~13 GB, tight) is the mid-size sweet spot for agentic coding, and [Nanbeige4.2-3B](/models/nanbeige4.2-3b/) reaches 4.3× on math.

24–36 GB
: [Qwen3.8-27B 4-bit](/models/qwen3.8-27b/) (~18 GB): 27B quality at 31–45 tok/s, the fastest 27B-class decode here. [Gemma-4 12B](/models/gemma-4-12b/) (~15 GB) pairs a big ratio (3.25×) with real speed (48–78 tok/s).

48 GB and up
: [Qwen3.8-27B 8-bit](/models/qwen3.8-27b/#8-bit) (~29 GB): the best ratio in the project, 3.74× mean and 4.23× on math. [Muse-Glimmer-30B 8-bit](/models/muse-glimmer-30b/#8-bit) (~40 GB) is a dense 30B at 2.47×.

Raw speed above all
: The mixtures of experts: [Qwen3.6-35B-A3B](/models/qwen3.6-35b-a3b/) decodes at 91–145 tok/s. Their *ratio* is modest because only a few billion parameters are active per token, so plain decoding is already fast. Nothing here decodes faster at that size.

## 8-bit, 4-bit or bf16

Verification dominates the cost of speculative decoding on a Mac, so the model's precision is a speed knob as much as a quality one.

- **8-bit is the sweet spot** for the speedup ratio and for quality. It's what most pairs are registered at.
- **4-bit** gives the highest absolute tokens per second and fits smaller Macs, but a smaller ratio: its verify cost starts rising at a narrower width. Ornith-1.0-9B is 2.40× at 8-bit and about 1.4× at 4-bit, yet faster in raw tok/s at 4-bit.
- **bf16** pays off for models that ship that way: [LFM2.5](/models/lfm2.5-1.2b/), [MiniCPM5](/models/minicpm5-2b/) and [Nanbeige](/models/nanbeige4.2-3b/) are among the biggest wins here. Where a model also ships an 8-bit build, 8-bit usually still wins.

The drafter itself always loads 4-bit. Its precision doesn't change how many tokens get accepted, only how fast it runs.

**Match the model's precision to what the drafter was trained against.** A drafter trained against the bf16 model accepts more on the 8-bit build than on the 4-bit one. Use the matched **instruct** model: a base model silently halves acceptance.

## For coding agents

Claude Code sends 18–26k tokens of system prompt and tool schemas with every request, so *reading the prompt* dominates the clock, not generating the answer. That changes what to optimize:

1. **Prefix caching first.** Every model here reuses the cached conversation between turns, which is what makes agents usable at all. See [Prefix caching](/concepts/prefix-caching/).
2. **A tool-calling model of about 8B or more.** Smaller models flail at multi-step tool use.
3. **Thinking off** (`--no-thinking`). A reasoning model thinks before every tool call; on Qwen3-8B the same task took 3:17 with thinking and about 2:20 without.

[Qwen3-8B](/models/qwen3-8b/) holds the measured wall-clock record under Claude Code. The agent you pick matters even more than the model: [pi](/use/agents-and-apps/#pi)'s ~1.5k-token system prompt finishes the same fix in about 6 seconds.

## Models that aren't worth a drafter

Speculation buys time in proportion to what one step of the model costs. Below about 4B dense parameters there's little to win: a Qwen3.5-0.8B drafter runs correctly but measures **0.96×**, because the model already decodes at 215 tok/s and a draft round costs nearly what it saves. The exceptions are small models with an expensive step, like looped Nanbeige, or ones that are unusually easy to draft, like LFM2.5 and MiniCPM5.

## A model that isn't listed

Every model on the [models page](/models/) is measured and vouched for. That list is *not* the list of models that run:

- **Any MLX model** gets drafter-free speculation (n-gram lookup) with the default `--mode auto`. It helps most on copy-heavy work like editing code.
- **Any matched DSpark or DFlash drafter** runs with `--drafter <repo>`, with no code change. See [Bring your own drafter](/models/bring-your-own/).

```bash
mlx-dspark serve --model mlx-community/Qwen3-32B-8bit \
  --drafter deepseek-ai/dspark_qwen3_32b_block7
```
