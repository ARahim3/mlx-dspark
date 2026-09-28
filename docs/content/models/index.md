---
title: Supported models
nav_title: Supported models
description: Every model here is measured and vouched for. Pass it to --model and its drafter, and that pair's best settings, resolve automatically, for any quantization of the model.
scripts: widgets
layout: wide
---

These are the pairs we have measured. They're also exactly what `--model` auto-resolves, so this list and the engine can't disagree: the page is generated from the engine's registry. Numbers are an M4 Pro's. Your Mac derives its own settings on first run.

<fieldset class="seg filters" data-model-filter data-filter-table="#models-table">
<legend class="sr-only">Show</legend>
<label><input type="radio" name="mkind" value="all" checked><span>All</span></label>
<label><input type="radio" name="mkind" value="small"><span>Small (≤ 4B)</span></label>
<label><input type="radio" name="mkind" value="dense"><span>Dense</span></label>
<label><input type="radio" name="mkind" value="hybrid"><span>Hybrid and recurrent</span></label>
<label><input type="radio" name="mkind" value="moe"><span>Mixture of experts</span></label>
</fieldset>

<div class="table-wrap"><table class="num mtable" id="models-table">
<thead><tr><th>Model</th><th>Mean speedup</th><th>Chat / code / math</th><th>Tokens / s</th><th>Memory</th><th>Default drafter</th></tr></thead>
<tbody>
{% for v in variants | sort(attribute='best.speedup', reverse=True) %}
{% set a = v.best %}
{% set kinds = [v.model.kind, 'small' if (v.model.params_b or 99) <= 4 else '', 'hybrid' if v.model.kind in ('conv-hybrid', 'looped', 'hybrid') else ''] %}
<tr data-kind="{{ kinds | join(' ') }}">
<td><a href="/models/{{ v.model.slug }}/{% if v.model.variants | length > 1 %}#{{ v.quant }}{% endif %}">{{ v.model.name }}</a> <span class="q">{{ v.quant }}</span></td>
<td><strong>{{ '%.2f' | format(a.speedup) }}×</strong></td>
<td>{{ '%.2f' | format(a.content.chat) }} / {{ '%.2f' | format(a.content.code) }} / {{ '%.2f' | format(a.content.math) }}</td>
<td>{% if a.tps_range %}{{ a.tps_range[0] }}–{{ a.tps_range[1] }}{% else %}{{ a.tps }}{% endif %}</td>
<td>{{ v.ram }}</td>
<td>{{ a.label }}</td>
</tr>
{% endfor %}
</tbody>
</table></div>

*Mean speedup* is over plain decoding of the same model, averaged across a chat, a code and a math prompt. *Tokens / s* is the range across those prompts: chat is usually the low end, because open-ended text accepts fewer drafted tokens than code or math. *Memory* is the measured peak at chat length for the model plus its 4-bit drafter; long contexts add KV cache on top. [How these are measured](/models/results/).

## Run one

```bash
mlx-dspark serve --model mlx-community/Qwen3.8-27B-4bit
```

Matching ignores quantization: a `-4bit`, `-8bit` or `-bf16` build of the same model resolves the same drafter. `mlx-dspark models` prints this list in your terminal.

## Anything else runs too

This list is what we vouch for, not what works:

- **Any MLX model** gets drafter-free lookup speculation with the default `--mode auto`, for free. It helps most on copy-heavy work like editing code.
- **Any matched DSpark or DFlash drafter** runs with `--drafter <repo>`, with no code change and no registry entry. See [Bring your own drafter](/models/bring-your-own/).

## Where models come from

`--model` takes a Hugging Face repo id or a local path, like `mlx-lm`. Before downloading anything, the engine looks, in order, at:

1. the path itself, if it's a folder (`--model ~/models/Qwen3.8-27B-4bit` always works);
2. `~/.cache/mlx_dspark/models`;
3. LM Studio's folder (`~/.lmstudio/models`): its **MLX** downloads load in place, with no second download;
4. any folders you list in `MLX_DSPARK_MODEL_DIRS` (`:`-separated), such as an external drive, `~/models` or a NAS, as `publisher/model` trees, `publisher_model` folders or bare model folders;
5. your Hugging Face cache (wherever `HF_HOME` puts it), and only then a download.

Only MLX checkpoints load from these folders (a `config.json` plus `.safetensors`). `mlx-dspark doctor --json` lists every folder searched, in order.

!!! warning "Downloads are big"
    A 27B model at 8-bit is about 29 GB. Each model page lists what you're signing up for, and the Mac app shows download size and fit before it starts.
