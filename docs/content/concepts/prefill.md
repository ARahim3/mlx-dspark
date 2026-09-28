---
title: Reading the prompt
nav_title: Reading the prompt (prefill)
description: Decoding speed is half your wait. The other half is prefill, reading the prompt, and for agents it's most of it.
---

Everything else on this site measures *decoding*, how fast tokens come out once generation starts. The other half of your wall clock is **prefill**: reading the prompt. For a chat message that's nothing. For a pasted file, a long conversation or an agent (Claude Code sends 18–26k tokens with every request), it's most of the wait.

<div class="table-wrap"><table class="num">
<thead><tr><th>Model</th><th>Prefill</th><th>Without CPU co-prefill</th><th>A 20k-token prompt takes</th></tr></thead>
<tbody>
{% set pre = [] %}
{% for v in variants if v.prefill %}{% set _ = pre.append(v) %}{% endfor %}
{% for v in pre | sort(attribute='prefill.tps', reverse=True) %}
{% set best_p = v.prefill.cpu_split or v.prefill.tps %}
<tr><td><a href="/models/{{ v.model.slug }}/">{{ v.model.name }}</a> <span class="muted">{{ v.quant }}</span></td><td><strong>{{ best_p }} tok/s</strong></td><td>{% if v.prefill.cpu_split %}{{ v.prefill.tps }} tok/s{% else %}—{% endif %}</td><td>about {{ (20000 / best_p) | round | int }} s</td></tr>
{% endfor %}
</tbody>
</table></div>

M4 Pro, 2048-token prompt, median of 3, default settings.

**Model size barely predicts this.** Prefill is compute-bound, so small dense models lead, and mixtures of experts punch far above their total size: an A3B model does only about 3.8B parameters' worth of arithmetic per token, however many experts it stores. The reverse shows up on Qwen3.8-27B: the 4-bit build prefills at the same rate as the 8-bit, because weight bits change decode speed (bandwidth-bound), not prefill. Looped Nanbeige runs every token through its layers twice, so the "3B" pays a 6B's arithmetic.

## What mlx-dspark does

**Skips work.** Prefill logits that every caller throws away aren't computed, and wide quantized weights are dequantized once instead of once per output tile. That's worth 1.07–1.15×, bit-identical, with no extra memory.

**Adds a second engine: CPU co-prefill.** With that, the GPU runs prefill at about 85% of its measured matrix-multiply peak, so going faster needs more hardware. Above a measured row count, each wide quantized matmul hands a calibrated fraction of its rows (about 0.3 on an M4 Pro) to the CPU's matrix units, which run *alongside* the GPU on the same arrays, with no copy and no second thread. That's the 1.3–1.4× column above, for about 0.4 GB of extra peak memory.

- The fraction is measured once per machine and model (about 15 seconds, cached), because it's an optimum with a cliff: 0.45 is already slower than 0.30.
- It isn't bit-identical: the CPU rows accumulate in a different order. That's the same floating-point-tie class as reading a prompt in chunks, and a 64-token greedy continuation came out token-identical. So the command line and server turn it on, and the Python library leaves it off.
- The CPU share runs in **fp32 through Accelerate BLAS** by default. The faster bf16 route (through BNNS) was behind every co-prefill crash ever reported, so it's opt-in: `MLX_DSPARK_CPU_SPLIT_FP32=0`, on a machine that never reaches memory pressure. On Qwen3.8-27B 4-bit that's 157 tok/s against 170 for bf16 and 136 with the split off.
- The memory guard's first shed switches it off for the rest of the session, since under memory pressure its win isn't realized anyway.
- It doesn't apply yet to unquantized (bf16) layers or to mixture-of-experts expert layers.

`--cpu-split 0` turns it off; `/health` reports the calibrated setting (`cpu_split`), and the serve banner prints it.

The Apple Neural Engine was measured for the same job and doesn't fit: its fast weight formats can't hold MLX's group-quantized weights exactly, and the exact fp16 form of a 27B model's MLP would be a 34 GB copy.

## The real lever: don't read it twice

A 20k-token first request is slow on any engine. What makes a long conversation or an agent session usable is that **every turn after the first skips the part it has already read**: 62 s cold, then about 1 s on the next turn, on a 27B with an 8k-token system prompt. That's the [prefix cache](/concepts/prefix-caching/).

Watching a long cold prompt? `GET /events` streams prefill progress (`{"type": "prefill", "processed": …, "total": …}`), so a client can show a progress bar instead of what looks like a hung server.
