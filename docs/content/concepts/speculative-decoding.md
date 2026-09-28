---
title: Speculative decoding
description: Why a Mac can check several tokens for the price of one, what the drafters are, and exactly what "lossless" means here.
---

## The bottleneck is memory, not math

To produce one token, a language model reads every one of its weights from memory: about 29 GB for a 27B model at 8-bit. On Apple Silicon that read, not the arithmetic, is what limits decoding speed. A Mac with 273 GB/s of bandwidth can't decode that model much faster than about 9 tokens per second, however fast its GPU is.

Here's the useful part: checking several candidate tokens in *one* forward pass reads the weights **once**. If something cheap could guess the next few tokens, the big model could check all of them for about the price of generating one.

<figure class="timeline" aria-label="Plain decoding makes one pass per token; speculative decoding makes one pass per round of several tokens">
  <div class="tl-row"><span class="tl-label">Plain decoding</span>
    <span class="tl-track">{% for i in range(12) %}<span class="tl-pass"><span class="tok ok">t</span></span>{% endfor %}</span>
    <span class="tl-note">12 tokens, 12 passes</span></div>
  <div class="tl-row"><span class="tl-label">Speculative</span>
    <span class="tl-track"><span class="tl-pass wide"><span class="tok ok">t</span><span class="tok ok">t</span><span class="tok ok">t</span><span class="tok ok">t</span><span class="tok fix">t</span></span><span class="tl-pass wide"><span class="tok ok">t</span><span class="tok ok">t</span><span class="tok fix">t</span></span><span class="tl-pass wide"><span class="tok ok">t</span><span class="tok ok">t</span><span class="tok ok">t</span><span class="tok fix">t</span></span></span>
    <span class="tl-note">12 tokens, 3 passes</span></div>
  <figcaption>Each box is one pass through the model. A speculative pass costs a little more than a plain one, because the drafter runs first and a few more rows are verified, but it can commit several tokens.</figcaption>
</figure>

## One round

1. **Draft.** A small drafter model proposes the next block of tokens. The drafters here are *EAGLE-family*: they don't read the text from scratch, they read the big model's own hidden state, which makes them tiny and accurate.
2. **Verify.** The big model runs one forward pass over all drafted positions and computes, at each one, the token it would have chosen itself.
3. **Accept.** Drafts are kept up to the first one the model disagrees with. At that position the model's own token is used instead, so every round commits at least one token (the worst case, equal to plain decoding) and at most the whole block plus one.

The average number of tokens committed per round is the **acceptance length**. It's the number that matters most, and the reason each model page shows it.

## Is it really lossless?

Yes, by construction. Every token you receive is the one the big model chose at that position, given everything before it.

- **Greedy decoding** (temperature 0): a draft is accepted only if it's exactly the model's top choice. The output matches plain greedy decoding.
- **Sampling** (`--temperature > 0`, with `--top-p` / `--top-k`): drafts are accepted by the speculative-sampling rule, and a rejection is resampled from the corrected distribution. The result is an exact sample from the model at that temperature, not an approximation. On a Mac it runs at about greedy speed.

One honest footnote. Verifying seven tokens at once uses different matrix shapes than decoding one at a time, so floating-point sums happen in a different order. When the model's top two candidates score equal to the last bit (a *tie*), the two orders can break the tie differently. From that token on, the texts differ, and both are equally the model's greedy output. Batched serving and even chunked prompt reading on any engine have the same property. The project's tests check every divergence and confirm it's at a tie.

The Mac app's **Race** runs this check on your own prompt: several decoders side by side, then a token-by-token comparison.

## The drafters

All of these come from the same lineage: a small network that reads the big model's hidden state.

DSpark (DeepSeek)
: A 5-layer parallel backbone proposes a block of 7 tokens in one pass, and a rank-256 **Markov head** adds a cheap correction from the previous token. That fixes *suffix decay*, where the later positions of a parallel draft stop agreeing with each other, for about 0.6 ms per round. A confidence head scores each position (`--confidence-threshold` uses it to stop drafting early). It comes from DeepSeek's [DeepSpec](https://github.com/deepseek-ai/DeepSpec) codebase, where it accelerates DeepSeek-V4; this project is its first port to MLX. `--mode dspark`.

DFlash (z-lab)
: A **block-diffusion** drafter: it denoises a whole 16-token block in one parallel pass and reuses the big model's own embedding and output head ([paper](https://arxiv.org/abs/2602.06036), MIT). Fast, but later positions collide on open-ended text. `--mode dflash`.

DFlash 2 (Inco AI)
: The DFlash backbone plus a **candidate-path selector**: it keeps the big model's top 16 candidates at each position and walks one coherent chain through them, plus dynamic convolutions. It raises acceptance *without* widening the verify. The default on [Qwen3.8-27B](/models/qwen3.8-27b/). `--mode dflash`.

Lookup (no drafter)
: When the text being written already appeared earlier in the context (quoting, code edits, repeated structure), the continuation of that earlier match is a free draft. `--mode lookup` runs this for **any** model with no drafter at all.

`--mode auto`, the default, picks each pair's measured best: the registry's stamped mode, otherwise DSpark, then DFlash, then lookup.

### Lookup drafts

Inside DSpark mode, lookup also runs as a **hybrid**. When the current text matches something earlier in the context, that free continuation is verified instead of running the drafter that round, so copy-heavy spans commit several tokens per round. When the match runs deep (8 tokens or more), drafts grow up to about twice the matched length, to 32 tokens: verifying 16–32 rows costs only about 2.5× a single step on Apple Silicon, so a verbatim span can commit 20–30 tokens per pass. That's how file re-emission on Gemma-4 12B goes from 3.0× to **4.5×**.

A free draft still has to be verified, and on a mixture of experts every extra verified row pulls in a fresh set of experts. So lookup drafts are **off by default** for the pairs where that measured as a net loss (every MoE, and the 4-bit 27B hybrids). `--lookup-drafts` / `--no-lookup-drafts` overrides the default either way.

## DSpark or DFlash?

It depends on the model and on the MLX version, because the answer comes from two curves: how many tokens a drafter gets accepted, and how fast verification cost grows with width on your chip. The objective isn't acceptance, it's **acceptance per unit of verify width**:

- **DFlash 2 wins where its checkpoints exist** (Qwen3.8-27B): same verify width as DSpark, more tokens accepted.
- **DSpark wins everywhere else** measured on current MLX. Original DFlash's full 16-token block does accept more on code and math (about 6 tokens per round on Gemma-4 12B), but verifying 16 rows costs more than it returns on these models. On Qwen3-8B the full block is a net loss (about 0.9×), and z-lab's own optimized runner measures the same.
- **The sharpest case is a mixture of experts.** On Qwen3.6-35B-A3B, DFlash out-drafts DSpark on every prompt (9.6 tokens per round on math, the highest acceptance ever measured here) and still loses on speed, 0.94× against 1.33×, because every verified row pulls in new experts.

According to the DSpark paper (acceptance length at temperature 1, full block), DSpark beats DeepSpec's DFlash by 16–18% and EAGLE-3 by 27–31%. Greedy numbers here are lower than the paper's, because exact greedy matching is the strictest possible acceptance rule.

## What this is not

This isn't DeepSeek-V4 inference. The models here are consumer-size models (Gemma-4, Qwen3, Qwen3.8, LFM2.5, MiniCPM5, Muse-Glimmer, Nemotron, Bonsai and others) with published drafters. mlx-dspark runs the real drafter methods on a Mac, but the model producing tokens is one of those.
