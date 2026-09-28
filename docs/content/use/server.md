---
title: Run a server
nav_title: Run a server
description: One process, one port, three API dialects. Anything that talks to OpenAI, Anthropic or the Responses API can use your local model, with the speedup built in.
---

```bash
mlx-dspark serve --model mlx-community/Qwen3-8B-8bit
```

That's a complete server at `http://127.0.0.1:8080`. It speaks:

| Dialect | Endpoint | Used by |
|---|---|---|
| OpenAI Chat Completions | `POST /v1/chat/completions` | The OpenAI SDKs, Open WebUI, Continue, Cline, Zed, SillyTavern, LibreChat, Raycast AI, OpenCode, pi |
| OpenAI Completions | `POST /v1/completions` | Older clients and raw prompting |
| Anthropic Messages | `POST /v1/messages`, `POST /v1/messages/count_tokens` | [Claude Code](/use/claude-code/), the Anthropic SDKs, pi |
| OpenAI Responses | `POST /v1/responses` | [Codex](/use/codex/) |

All of them stream and all of them take tools. The model list is at `GET /v1/models`; health, metrics and the control plane are in the [HTTP reference](/reference/http-api/).

The speedup is transparent. Clients don't need to know speculation is happening, and nothing about a request changes.

## The flags you'll actually use

`--model REPO_OR_PATH`
: Any Hugging Face repo or local path. For [measured models](/models/) the drafter and its tuned defaults resolve automatically.

`--no-thinking`
: Answer directly instead of reasoning first, on models that think (Qwen3, MiniCPM5, Qwen3.8, …). Clients can still turn it back on per request. The single biggest speed win for agents.

`--reasoning-effort low|medium|high|xhigh`
: A default reasoning depth for templates that support it (the Qwen3.8 class). Requests can override it. `/health` reports whether the loaded model supports it.

`--context-window N`
: Cap the prompt length below the model's own maximum, mainly to keep the KV cache inside your RAM. Over-long requests get the "prompt is too long" error that agent clients compact on.

`--max-batch N`
: Serve up to N concurrent requests in one batched forward pass, for several agents or users at once. See [Batching](/concepts/batching/).

`--host 0.0.0.0 --api-key KEY`
: Listen on your network. With a key, every route except `/health` requires it.

`--mode auto|dspark|dflash|lookup|baseline`
: `auto` (the default) picks the pair's measured best. The others force one method, for comparisons.

`--no-model`
: Start instantly with nothing loaded. A client loads a model later with `POST /admin/load`, and can swap it without the port changing.

Every flag, with its default, is in the [CLI reference](/reference/cli/#serve).

## Talk to it

<div class="tabs" data-group="client" markdown="1">
<div class="tab" data-label="curl" markdown="1">

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-8B-8bit", "stream": true,
       "messages": [{"role": "user", "content": "Write a haiku about caches."}]}'
```

</div>
<div class="tab" data-label="OpenAI SDK" markdown="1">

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")
stream = client.chat.completions.create(
    model="Qwen3-8B-8bit",
    messages=[{"role": "user", "content": "Write a haiku about caches."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

</div>
<div class="tab" data-label="Anthropic SDK" markdown="1">

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://127.0.0.1:8080", api_key="not-needed")
with client.messages.stream(
    model="Qwen3-8B-8bit",
    max_tokens=512,
    messages=[{"role": "user", "content": "Write a haiku about caches."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

</div>
</div>

The model name in a request is the last part of the repo id (`Qwen3-8B-8bit` for `mlx-community/Qwen3-8B-8bit`). The server serves one model at a time; `GET /v1/models` returns its exact name.

## What a request can ask for

- **Sampling:** `temperature`, `top_p`, `top_k`, `seed`, `max_tokens`, `stop`. Speculative sampling stays lossless: the output is an exact sample from the model at that temperature.
- **Penalties and logprobs:** `presence_penalty`, `frequency_penalty`, `logprobs`, `top_logprobs`.
- **Tools:** OpenAI `tools` / `tool_calls`, Anthropic `tool_use` / `tool_result`, and the Responses API's flat tool shape. Each is translated into whatever the loaded model's chat template expects, including multi-turn tool history. Six native tool-call syntaxes are parsed back: Hermes JSON (Qwen3), Gemma-4's, the XML `<function=…>` form (Ornith and others), Muse's ATEM calls, LFM2's Python-style call list, and MiniCPM5's `<function name="…">`.
- **Thinking:** `enable_thinking` (per request), Anthropic's `thinking: {"type": "disabled"}`, and `reasoning_effort` where the template supports it.

When no temperature is sent, the model's own `generation_config` defaults apply. Many community conversions ship none, in which case omitted means greedy; set `--default-temperature` to change that.

## Reasoning comes back separated

A reasoning model's thinking never leaks into the answer. On the OpenAI endpoints it arrives as `reasoning_content`; on the Anthropic endpoint as proper `thinking` blocks; on the Responses endpoint as reasoning items. That holds when streaming too, including for templates that pre-open the `<think>` block in the prompt.

## See the speedup on every response

Each response carries a non-standard `x_mlx_dspark` block that clients ignore and you can read:

```json
"x_mlx_dspark": {
  "mode": "dspark", "cap": 7, "accept_len": 3.4,
  "decode_tokens_per_sec": 64.5, "tokens_per_sec": 58.1,
  "prompt_tokens": 2150, "cached_tokens": 2048, "ttft_seconds": 0.21,
  "roofline_ratio": 2.1
}
```

`accept_len` is tokens committed per round, `cached_tokens` is how much of the prompt the [prefix cache](/concepts/prefix-caching/) served, and `roofline_ratio` is decode speed as a multiple of what plain decoding could reach on this Mac's measured memory bandwidth. The full field list is in the [HTTP reference](/reference/http-api/#the-x_mlx_dspark-block). The standard `usage.prompt_tokens_details.cached_tokens` is filled in too, for tools that compute cache hit rates.

## Security

The server is a local, single-user server and binds `127.0.0.1` by default.

- **Exposing it:** use `--api-key`. Then every route except `GET /health` requires `Authorization: Bearer <key>` or `x-api-key: <key>`.
- **Model code:** a checkpoint that ships its own Python (`model_file` or `auto_map` in its config) is **refused by default**, because the loaders would otherwise import and run it. `--trust-remote-code` (or `MLX_DSPARK_TRUST_REMOTE_CODE=1`) opts the whole process in. There's no per-request override.

See [SECURITY.md](https://github.com/ARahim3/mlx-dspark/blob/main/SECURITY.md) for the threat model and how to report a vulnerability.

## Keeping memory under control

- **`--context-window`** is the real fix for a model that nearly fills your RAM. At load, `serve` estimates the KV cache at the model's full context and prints a warning, with a value that fits, if it would overrun your GPU working set. The same warning appears in `/health`.
- **The memory-pressure guard** (on by default) watches macOS's memory pressure level. On a warning it returns MLX's cached buffers and the prefix cache's shallow snapshots, keeping each conversation's cached prefix. At critical pressure it empties the prefix cache. It buys headroom before macOS starts paging the model out; it can't make a swapping model fast again. `--no-memory-guard` turns it off.
- **`--kv-bits 8`** halves the attention KV cache's memory. Treat it as a memory lever, not a speed lever: at depth it measures slightly slower.
