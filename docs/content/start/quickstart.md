---
title: Quickstart
description: From install to a faster local model in four commands, then three things worth knowing before you stop reading.
---

This page uses **Qwen3-8B** (8-bit, about 11 GB), a solid all-rounder. Swap in any model from [the models page](/models/), or let [Choose a model](/start/choose-a-model/) pick one for your Mac.

## 1. Ask it something

```bash
mlx-dspark generate --model mlx-community/Qwen3-8B-8bit \
  --prompt "Explain how rainbows form."
```

The first run downloads the model and its matched drafter, then measures your Mac once for this pair (a few seconds). The answer streams, and a summary line reports how many tokens were accepted per round and the speed.

## 2. See that nothing changed but the speed

Run the same prompt without speculation:

```bash
mlx-dspark generate --model mlx-community/Qwen3-8B-8bit --mode baseline \
  --prompt "Explain how rainbows form." --max-new-tokens 300
```

Same text, fewer tokens per second. The model verifies every drafted token, so its output is exactly what it would write on its own. `mlx-dspark benchmark` does this comparison properly, over several prompts and trials: see [Command line](/use/command-line/).

## 3. Serve it

```bash
mlx-dspark serve --model mlx-community/Qwen3-8B-8bit
```

One port, three dialects: the OpenAI API at `http://127.0.0.1:8080/v1`, Anthropic's Messages API at `/v1/messages`, and OpenAI's Responses API at `/v1/responses`.

## 4. Talk to it

<div class="tabs" data-group="client" markdown="1">
<div class="tab" data-label="curl" markdown="1">

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-8B-8bit",
       "messages": [{"role": "user", "content": "Explain rainbows briefly."}]}'
```

</div>
<div class="tab" data-label="OpenAI SDK" markdown="1">

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")
reply = client.chat.completions.create(
    model="Qwen3-8B-8bit",
    messages=[{"role": "user", "content": "Explain rainbows briefly."}],
)
print(reply.choices[0].message.content)
```

</div>
<div class="tab" data-label="Anthropic SDK" markdown="1">

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://127.0.0.1:8080", api_key="not-needed")
msg = client.messages.create(
    model="Qwen3-8B-8bit",
    max_tokens=512,
    messages=[{"role": "user", "content": "Explain rainbows briefly."}],
)
print(msg.content[-1].text)
```

</div>
<div class="tab" data-label="Claude Code" markdown="1">

```bash
# leave the server running, then in a second terminal:
mlx-dspark claude
```

For agent work, start the server with `--no-thinking`: see [Claude Code](/use/claude-code/).

</div>
</div>

Any app that talks to an OpenAI-compatible server works the same way: point it at `http://127.0.0.1:8080/v1` with any API key. See [Agents and apps](/use/agents-and-apps/).

## Three things worth knowing

**Pick a model, not a configuration.** `--model` takes any Hugging Face repo or local path, exactly like `mlx-lm`. For [the measured models](/models/), the matching drafter *and* that pair's best settings resolve automatically, for any quantization of the model. Anything else still gets drafter-free speculation with the default `--mode auto`, or name a drafter with `--drafter <repo>`.

**Don't set the draft cap.** The published speedups were measured on one M4 Pro. Your Mac's optimum is different, so mlx-dspark measures it on first run and derives its own. If you only remember one flag, `--max-draft auto` adapts the cap every round while it generates. Pin `--max-draft N` only if you've measured a better value. [Why](/concepts/draft-caps/).

**It's lossless by construction.** The model checks every drafted token, so output matches plain decoding in every mode and at every cap. Sampling with `--temperature` stays lossless too: you get an exact sample from the model at that temperature.

Prefer clicking to typing? [The Mac app](/start/mac-app/) does all of this, including the model picker and a live race that checks the output token by token.
