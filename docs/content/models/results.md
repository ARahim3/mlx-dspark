---
title: Results and method
nav_title: Results and method
description: Every measured number in one place, how it was measured, and why some models gain far more than others.
scripts: widgets
layout: wide
---
{% import "macros.html" as ui %}

{{ ui.dotplot() }}

The dashed line is plain decoding of the same model. Everything to its right is time saved, with identical output.

## The full table

Each model at the draft cap it **derives on its own** (no flags), against plain greedy decoding of the same model. M4 Pro, 48 GB, warm, 4-bit drafter, 200 new tokens, end-to-end tokens per second, median of 3 runs over the three prompts.

<div class="table-wrap"><table class="num">
<thead><tr><th>Model</th><th>Drafter</th><th>Cap</th><th>Accepted</th><th>Plain</th><th>mlx-dspark</th><th>Speedup</th><th>Chat / code / math</th><th>Measured</th></tr></thead>
<tbody>
{% for v in variants | sort(attribute='best.speedup', reverse=True) %}
{% set a = v.best %}
<tr>
<td><a href="/models/{{ v.model.slug }}/">{{ v.model.name }}</a> <span class="muted">{{ v.quant }}</span></td>
<td>{{ a.label }}</td>
<td>{{ a.cap if a.cap else a.cap_label }}</td>
<td>{{ '%.2f' | format(a.accept) if a.accept else '—' }}</td>
<td>{{ a.baseline }}</td>
<td>{{ a.tps }}</td>
<td><strong>{{ '%.2f' | format(a.speedup) }}×</strong></td>
<td>{{ '%.2f' | format(a.content.chat) }} / {{ '%.2f' | format(a.content.code) }} / {{ '%.2f' | format(a.content.math) }}</td>
<td>{{ v.measured.when }}{% if not v.measured.fresh %}*{% endif %}</td>
</tr>
{% endfor %}
</tbody>
</table></div>

Rows marked * were measured before the September 2026 verify kernels. Those kernels moved every 8-bit model re-measured since from cap 4 to cap 7 (Gemma-4 12B went from 2.82× to 3.25× in the same run), so the older 8-bit rows are probably conservative. Qwen3.6-35B-A3B's row uses `--confidence-threshold 0.3`; its zero-flag default measures 1.27×. The per-model pages have each row's flags and caveats.

Reproduce any row on your own Mac:

```bash
mlx-dspark benchmark --model mlx-community/Qwen3-8B-8bit --trials 3
```

## Two drafters on Qwen3.8-27B

Qwen3.8-27B has two strong drafters: Inco AI's **DFlash 2** (the default) and Red Hat's **DSpark** head (`--mode dspark`). Both verify the same width (cap 7). Paired in one process, arms interleaved per prompt, High Power mode:

<div class="table-wrap"><table class="num">
<thead><tr><th>Quant</th><th>Drafter</th><th>Mean</th><th>Chat</th><th>Code</th><th>Math</th><th>Accepted</th><th>tok/s</th></tr></thead>
<tbody>
{% for v in variants if v.arms | length > 1 %}
{% for arm in v.arms %}
<tr{% if arm.default %} class="is-default"{% endif %}><td>{{ v.quant }}{% if loop.first %} <span class="muted">(plain {{ arm.baseline }} tok/s)</span>{% endif %}</td><td>{{ arm.label }}{% if arm.default %} <span class="chip chip-spark">default</span>{% endif %}</td>
<td><strong>{{ '%.2f' | format(arm.speedup) }}×</strong></td><td>{{ '%.2f' | format(arm.content.chat) }}×</td><td>{{ '%.2f' | format(arm.content.code) }}×</td><td>{{ '%.2f' | format(arm.content.math) }}×</td><td>{{ '%.2f' | format(arm.accept) }}</td><td>{{ arm.tps }}</td></tr>
{% endfor %}
{% endfor %}
</tbody>
</table></div>

DFlash 2 leads at 4-bit; the two tie at 8-bit, which is the best DSpark-mode ratio in the project. DFlash 2 adds a **candidate-path selector** (the model's top-16 candidates per position, walked as one coherent chain) and **dynamic convolutions** to the DFlash backbone. Acceptance rises by about 1.1–1.4 tokens *without widening the verify*, which is exactly the kind of gain that converts on Apple Silicon.

## At long context

Agent clients send 10–40k-token prompts. Decode speed at depth, against plain decoding at the same depth, accepted tokens per round in parentheses (agent-style content, thinking off, best of 2):

<div class="table-wrap"><table class="num">
<thead><tr><th>Model and drafter</th><th>Context</th><th>Plain</th><th>Cap 3</th><th>Cap 7</th></tr></thead>
<tbody>
{% for v in variants if v.long_context %}
{% for r in v.long_context %}
<tr><td>{% if loop.first or r.arm != loop.previtem.arm %}{{ v.model.name }} {{ v.quant }}, {{ r.arm }}{% endif %}</td><td>{{ r.depth }}</td><td>{{ r.baseline }} tok/s</td>
<td>{{ '%.2f' | format(r.cap3[0]) }}× <span class="muted">({{ r.cap3[1] }})</span></td><td>{{ '%.2f' | format(r.cap7[0]) }}× <span class="muted">({{ r.cap7[1] }})</span></td></tr>
{% endfor %}
{% endfor %}
</tbody>
</table></div>

Speedups at depth are lower than at chat length for a structural reason: attention over a long KV cache is extra verify work that a single decode step pays only once. You don't need to pick the cap: with no `--max-draft`, it shrinks automatically when the measured depth cost says a narrower verify pays. See [Long context](/concepts/long-context/).

## Reading the prompt

Prefill speed decides how long you wait before the first token. With CPU co-prefill on (the default for the CLI and server), 2048-token prompt, median of 3:

<div class="table-wrap"><table class="num">
<thead><tr><th>Model</th><th>Prefill</th><th>Without CPU co-prefill</th><th>A 20k-token prompt takes</th></tr></thead>
<tbody>
{% set pre = [] %}
{% for v in variants if v.prefill %}{% set _ = pre.append(v) %}{% endfor %}
{% for v in pre | sort(attribute='prefill.tps', reverse=True) %}
{% set best_p = v.prefill.cpu_split or v.prefill.tps %}
<tr><td><a href="/models/{{ v.model.slug }}/">{{ v.model.name }}</a> <span class="muted">{{ v.quant }}</span></td><td><strong>{{ best_p }} tok/s</strong></td><td>{% if v.prefill.cpu_split %}{{ v.prefill.tps }} tok/s ({{ '%.2f' | format(v.prefill.cpu_split / v.prefill.tps) }}×){% else %}—{% endif %}</td><td>about {{ (20000 / best_p) | round | int }} s</td></tr>
{% endfor %}
</tbody>
</table></div>

Model size barely predicts this. Prefill is compute-bound, so small dense models lead, and mixtures of experts punch far above their total size: an A3B model does only about 3.8B parameters' worth of arithmetic per token. A dash means CPU co-prefill doesn't apply to that model yet (bf16 layers and expert layers aren't split). Either way this is a **first-request cost**: the [prefix cache](/concepts/prefix-caching/) skips it on every later turn.

## Copy-heavy editing goes further

The table above is fresh generation. When a model re-emits or refactors code already in its context, the everyday agent workload, lookup drafts reach well past it with output still bit-identical:

| Task | Model | Before lookup drafts | With them |
|---|---|---|---|
| Re-emit a file | Gemma-4 12B 8-bit | 3.03× | **4.51×** (75 tok/s) |
| Rename refactor | Gemma-4 12B 8-bit | | 4.33× |
| Rename refactor | Ornith-1.0-9B 8-bit | 2.79× | **3.57×** (93 tok/s) |
| Re-emit a file | Ornith-1.0-9B 8-bit | | 2.45× |

Chat and fresh code are unchanged. See [lookup drafts](/concepts/speculative-decoding/#lookup-drafts).

## Why some models gain more than others

**The drafter sets the ceiling.** Acceptance, how many drafted tokens survive per round, is set by how well the drafter matches the model. Official drafters and carefully qualified community ones accept 4–5.5 tokens per round on these prompts.

**What a step costs sets the payoff.** Speculation saves whole passes through the model. If a pass is cheap, there's little to save. That's why:

- **Mixtures of experts gain least, whatever the drafter.** Qwen3.6-35B-A3B activates about 3.8B of its 35B parameters per token, so plain decoding already runs at 87 tok/s. Its drafter accepts up to 7 tokens per round on math and it still converts to only 1.32×: a step costs about 11.5 ms, and the dense drafter costs about 5.7 ms of every round. Every extra verified row also pulls in a fresh set of experts.
- **Small, cheap models gain less** unless they're unusually easy to draft (LFM2.5, MiniCPM5).
- **Looped Nanbeige gains a lot.** It runs its layers twice, so its step is as expensive as a 6B model's, and that's exactly where a drafter pays.

**Precision changes the verify curve.** 2-bit Bonsai is compute-bound from a verify width of 2, which caps it near 1.1×. 8-bit stays flat to width 8 with the current kernels, which is why most 8-bit models draft a full block of 7. See [Draft caps](/concepts/draft-caps/).

## How it's measured

- `mlx-dspark benchmark --trials 3`: a chat, a code and a math prompt, 200 new tokens each, after a warm-up, median of three runs, end-to-end tokens per second. The plain baseline is a pipelined greedy loop that measures at parity with `mlx_lm.generate` (Qwen3-4B: 51–52 tok/s either way).
- Each pair runs at the cap it derives with no flags, so the numbers are what you get out of the box, not a tuned best.
- Run-to-run noise on a laptop is about ±14%, from the machine, not the content. Comparisons are paired within one run and quoted as ratios.
- All on one M4 Pro (16" MacBook Pro, 48 GB). Head-to-heads and the most recent rows were run in High Power mode, because in automatic energy mode sustained load throttles speculative rounds more than plain decoding.
- Every run checks output against plain decoding. Where they differ, it's at a floating-point tie: two candidate tokens scored equal to the last bit, and different arithmetic order broke the tie differently. See [Is it really lossless?](/concepts/speculative-decoding/#is-it-really-lossless)

Your Mac will differ, which is the point of deriving the cap on it. A faster chip can land on a *lower* cap, because cheaper verification shifts the optimum. Run the benchmark and compare ratios, not absolute tokens per second.
