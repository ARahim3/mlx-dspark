---
title: Long context
description: Agents send 10–40k-token prompts. What changes at that depth, what mlx-dspark does about it, and how to keep memory in check.
---

Agent clients like Claude Code, Codex and pi send 10–40k tokens with every request. At that depth, speculative decoding used to fade on a Mac: at 32k tokens, DSpark on Qwen3.8-27B could fall *below* plain decoding. Three engine changes (v0.20.0) hold the speedup there:

1. **A multi-row attention kernel.** MLX's decode attention re-reads the whole KV cache once per verified row. The new kernel reads each KV tile once for all rows, and also covers the DSpark drafter's own attention over the context.
2. **A preallocated drafter context.** The DSpark drafter no longer copies its whole context twice per layer per round.
3. **A 4096-token drafter window.** DeepSpec-style heads lose acceptance when they attend a context far longer than they were trained on. Attending only the most recent 4096 tokens restores their short-context acceptance at any depth. Heads trained with their own sliding window (DFlash 2, Red Hat's Qwen3.8 head, Nemotron, Muse) keep theirs.

## Measured at 16k and 32k

Decode speed, prompt reading excluded, with agent-style content (source code plus a coding task, thinking off), against plain decoding at the same depth. Accepted tokens per round in parentheses:

<div class="table-wrap"><table class="num">
<thead><tr><th>Model and drafter</th><th>Context</th><th>Plain</th><th>Cap 3</th><th>Cap 7</th></tr></thead>
<tbody>
{% for v in variants if v.long_context %}
{% for r in v.long_context %}
<tr><td>{% if loop.first or r.arm != loop.previtem.arm %}<a href="/models/{{ v.model.slug }}/">{{ v.model.name }}</a> {{ v.quant }}, {{ r.arm }}{% endif %}</td><td>{{ r.depth }}</td><td>{{ r.baseline }} tok/s</td>
<td>{{ '%.2f' | format(r.cap3[0]) }}× <span class="muted">({{ r.cap3[1] }})</span></td><td>{{ '%.2f' | format(r.cap7[0]) }}× <span class="muted">({{ r.cap7[1] }})</span></td></tr>
{% endfor %}
{% endfor %}
</tbody>
</table></div>

Speedups at depth are lower than at chat length for a structural reason: attention over a long KV cache is extra verify work that a plain decoding step pays only once, and plain decoding itself slows with depth too (Qwen3.8-27B 4-bit: about 15 tok/s at 2k, 12.9 at 32k).

**You don't need to pick the cap.** With no `--max-draft`, calibration also measures how verify cost grows with depth, and the derived cap shrinks the verify width when a narrower one pays. `--max-draft auto` adapts per round. Under 4k tokens of context neither changes anything.

`--drafter-window N` changes the drafter window (`0` = the whole context, the pre-2026-09 behaviour). It only affects drafting, so the output is unchanged.

## What else grows with a long prompt

For every decoder, this one and plain `mlx-lm` alike:

- **Time to first token.** Reading an *L*-token prompt is inherent work: about 110 s for 20k tokens on a 27B. The [prefix cache](/concepts/prefix-caching/) is the lever: every later turn of a conversation skips the part it has already read.
- **Memory.** The KV cache grows linearly with context.

## Memory at long context

On Qwen3.8-27B the KV cache costs **0.086 GB per 1,000 tokens** of context: about 11 GB on top of the weights at 128k, and about 23 GB at the full 256k. It's identical for both quantizations, because the cache is bf16 regardless of weight bits. Its hybrid design is why that's small: only 16 of its 64 layers keep a growing cache; the other 48 hold a fixed-size recurrent state.

Tools for when memory is the constraint:

`--context-window N`
: Caps the prompt length. Requests past it get the "prompt is too long" error agent clients compact on, instead of a swap storm. By default the window is the model's own maximum (262,144 on Qwen3.8-27B, about 16 GB of KV cache on top of about 29 GB of weights). If weights plus a full-window cache would overrun your GPU working set, `serve` prints a warning with a value that fits, and `/health` shows it too. Also settable per model swap as `context_window` on `/admin/load`.

`--kv-bits 8`
: Halves the attention KV cache, on hybrid models too (only their full-attention layers quantize). A memory lever, not a speed lever: at 32k the quantized attention measures slightly *slower*, so leave it off unless memory is the limit.

The memory-pressure guard
: On by default. At macOS's *warning* level it returns MLX's cached buffers and the prefix cache's shallow snapshots (about 1.7 GB on a 27B), keeping every conversation's cached prefix. At *critical* it empties the prefix cache. It buys headroom before macOS pages the model out. It can't make a swapping model fast again, which is why `--context-window` is the real fix.

`--wired-limit`
: Off, and almost certainly best left off. It pins MLX's working set so weights can't be paged out, but wired memory can't be reclaimed by macOS: on a machine already holding a large working set it can hang the Mac hard enough to need a power cycle, and it bought no measurable speed where tested.
