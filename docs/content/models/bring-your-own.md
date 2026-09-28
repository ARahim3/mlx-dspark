---
title: Bring your own drafter
description: The registry only saves you from naming the drafter. Any matched DSpark or DFlash checkpoint runs with --drafter, and any model at all runs with drafter-free lookup.
---

```bash
# a measured pair: the drafter resolves by itself
mlx-dspark generate --model mlx-community/Qwen3-14B-8bit --prompt "Explain how rainbows form."

# anything else: name the drafter; the machinery is identical from here on
mlx-dspark generate --model mlx-community/Qwen3-32B-8bit \
  --drafter deepseek-ai/dspark_qwen3_32b_block7 --prompt "Explain how rainbows form."
```

## What runs, and what doesn't

New DSpark and DFlash drafters keep landing on Hugging Face in several different packagings. The loaders recognize these, and refuse anything incompatible with an error naming the reason, never a silent mis-load.

| Checkpoint style | Example | Status |
|---|---|---|
| **DeepSpec standalone drafter** (any size or quant) | `deepseek-ai/dspark_qwen3_32b_block7` | Runs via `--drafter`. The 4B, 8B, 14B and Gemma-4 12B heads are measured and registered, so they need no flag. Larger sizes should run; reports welcome. |
| **z-lab DFlash adapter** | `z-lab/Qwen3-8B-DFlash-b16` | Runs via `--mode dflash --drafter`. |
| **DFlash 2** (Inco AI: candidate selector + dynamic convolutions) | `incoai/Qwen3.8-27B-DFlash2` | Runs via `--mode dflash --drafter`. The Qwen3.8-27B heads are registered and `--mode auto` picks them. |
| **vLLM "speculators" format**, `dspark` algorithm | `RedHatAI/Qwen3.8-27B-speculator.dspark`, `makora-ai/gemma4-26b-a4b-dspark` | Runs via `--drafter`. The config is translated on load, including reduced draft vocabularies (`draft_vocab_size` with a `d2t` table). Other speculators algorithms (EAGLE, EAGLE-3) are refused by name. |
| **SpecForge / SGLang packaging** (`dflash_config` with `projector_type: dspark`) | `Nanbeige/Nanbeige4.2-3B-DSpark` | Runs via `--drafter`, with YaRN rope where the head uses it. |
| **LiquidAI LFM2.5-DSpark** | `LiquidAI/LFM2.5-2.6B-DSpark` | Measured and registered, any quant. |
| **PrismML dspark GGUF** (Bonsai 27B) | `prism-ml/Ternary-Bonsai-27B-gguf` | Pre-converted repacks resolve automatically. A future GGUF-only drop runs via `--drafter gguf:<repo>/<file>.gguf`, converted locally once. |
| **Full model with an embedded drafter** | `deepseek-ai/DeepSeek-V4-Pro-DSpark` (893 GB) | Doesn't run: a different architecture and packaging, out of scope for a Mac. |
| **DFlash + Markov community hybrids** | `Hikari07jp/DSpark-Gemma-4-31B-draft` | Not yet. |

One measured example outside the registry: `makora-ai/gemma4-26b-a4b-dspark` on `mlx-community/gemma-4-26b-a4b-it-8bit` (Google's 26B mixture of experts) runs at **1.27×** with `--max-draft 2 --no-lookup-drafts`: 1.38× on code, 1.37× on math, 1.06× on chat. It isn't registered while the ratio is under review.

## Which models can be drafted for

- **Any dense `mlx-lm` text model** routes automatically. On load, a one-time probe checks that the hidden-state tap the drafter reads reproduces the model's own forward pass, and fails loudly if the model's family needs dedicated support.
- **Hybrid and recurrent families already supported:** Qwen3.5/3.6/3.8 (linear attention), Nemotron-H (Mamba-2), LFM2 (short convolutions), Nanbeige (looped depth), and Gemma-4 and Muse through `mlx-vlm`. Their recurrent state can't be trimmed like a KV cache, so each verify round's inputs are recorded and the state is rebuilt exactly at the accept point.
- **Any model at all** runs drafter-free with `--mode lookup`, or `--mode auto`.

## Getting a good pairing

- **Use the matched *instruct* model** the drafter was trained against. A base model drops acceptance sharply.
- **Match the model's precision to the drafter's training.** A drafter trained against the bf16 model does best on the 8-bit build, one trained against a 4-bit model on the 4-bit build. (Better training data can beat this rule: Red Hat's bf16-trained Qwen3.8 head wins on the 4-bit model too.)
- **Drafter precision doesn't matter** for acceptance. Drafters load 4-bit by default (`--drafter-bits` changes it), because that's the fastest to run.
- **Very small models aren't worth a drafter.** A Qwen3.5-0.8B DSpark head runs correctly and losslessly at **0.96×**: the model already decodes at 215 tok/s, and a draft round costs nearly what it saves. Run those models plainly.

## Measure it

```bash
mlx-dspark benchmark --model <model> --drafter <drafter> --caps 2,4,7,auto --trials 3 --json result.json
```

The registered pairs accept between 2.5 and 5.5 tokens per round. Acceptance stuck near 1.2 means something is mismatched: a base model instead of the instruct one, the wrong quantization, or a packaging detail the loader doesn't know yet. If you've measured a pair that isn't listed, the device-stamped JSON is exactly what's needed to add it. Please [open an issue](https://github.com/ARahim3/mlx-dspark/issues) with it.

## Remote code

Drafters load their weights one-to-one and never import code, so a drafter whose config carries an `auto_map` pointer needs nothing special. **Models** that ship their own Python (`model_file` or `auto_map` in `config.json`) are refused by default, because the loaders would otherwise import and run it. Opt in with `--trust-remote-code` or `MLX_DSPARK_TRUST_REMOTE_CODE=1` once you trust the repo.
