---
title: Agents and apps
description: pi, OpenCode, Open WebUI, Continue, Cline, Zed and everything else that connects to llama.cpp or Ollama connects here by changing one URL.
---

The server's HTTP surface is the one local-model tools already speak. Anything that takes a custom OpenAI base URL works: set it to `http://127.0.0.1:8080/v1` with any API key. That covers the tools people point at `llama-server` or Ollama, including Open WebUI, SillyTavern, Continue, Cline, Zed, LibreChat and Raycast AI.

There's no GGUF interop, because mlx-dspark runs MLX weights natively. But the HTTP side is shared, and that's the part these tools talk to.

<div class="tabs" markdown="1">
<div class="tab" data-label="Any OpenAI-compatible app" markdown="1">

Paste these into the app's provider settings:

| Field | Value |
|---|---|
| Base URL | `http://127.0.0.1:8080/v1` |
| API key | anything (or your server's `--api-key`) |
| Model | the served name, e.g. `Qwen3-8B-8bit` |

</div>
<div class="tab" data-label="pi" markdown="1">

Add a provider to `~/.pi/agent/models.json`:

```json title="~/.pi/agent/models.json"
{ "providers": { "mlx-dspark": {
    "baseUrl": "http://127.0.0.1:8080", "api": "anthropic-messages", "apiKey": "mlx-dspark",
    "models": [{ "id": "Qwen3-8B-8bit", "contextWindow": 40960, "maxTokens": 8192 }] } } }
```

```bash
pi --provider mlx-dspark --model Qwen3-8B-8bit
```

To use the OpenAI endpoint instead, set `"api"` to `"openai-completions"` and `"baseUrl"` to `http://127.0.0.1:8080/v1`. Both work.

</div>
<div class="tab" data-label="OpenCode" markdown="1">

Add a provider to `opencode.json`, in your project root or `~/.config/opencode/`:

```json title="opencode.json"
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "mlx-dspark": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "mlx-dspark",
      "options": { "baseURL": "http://127.0.0.1:8080/v1", "apiKey": "mlx-dspark" },
      "models": { "Qwen3-8B-8bit": { "name": "Qwen3-8B-8bit" } }
    }
  }
}
```

</div>
<div class="tab" data-label="Claude Code" markdown="1">

```bash
mlx-dspark claude
```

See [Claude Code](/use/claude-code/).

</div>
<div class="tab" data-label="Codex" markdown="1">

Codex needs the Responses API; see [Codex](/use/codex/) for the `config.toml` provider.

</div>
</div>

A running server hands out these configs itself, filled in with its real address, model and key: `GET /admin/integrations`. The Mac app's **Agents** tab is built on it.

## pi

[pi](https://github.com/earendil-works/pi-mono) suits a local model better than Claude Code, for one reason: its system prompt is **about 1.5k tokens, against Claude Code's 18–26k**. Reading the prompt dominates an agent's clock on a Mac, so that lands directly on your wait. The same one-bug fix that takes about 2:20 in Claude Code takes **about 6 seconds**, and a four-tool task (read, two edits, read, write) finishes in 8.5 s at 24 tok/s on Qwen3-8B. Ornith-1.0-9B runs the same task in 18.8 s.

Gemma-4 12B doesn't converge on pi's tool protocol, on either endpoint, so it's the model and not the server. Use Gemma-4 with Claude Code instead.

## Practical notes for any agent

Use a tool-calling model
: These are tool-use agents first. Qwen3-8B and up handle it; smaller models flail.

Agent choice moves the clock more than model choice
: The client's prompt size is the dominant cost on a local model. A lean agent is an order of magnitude faster on the same hardware and task.

Thinking off is a speed knob, not a requirement
: Reasoning streams back properly either way. It just costs time, because the model thinks before every tool call. See [Claude Code tips](/use/claude-code/#tips).

Leave prefix caching on
: It's doing most of the work. See [Prefix caching](/concepts/prefix-caching/).

Streams stay alive, and disconnects stop generation
: Keep-alive frames go out every 15 seconds while nothing else is on the wire, so clients don't idle-timeout during a long prompt read. If a client vanishes, generation stops at the next round instead of running to `max_tokens` while other requests queue behind it.

## Already use LM Studio?

Its **MLX** downloads are reused automatically. Pass the same `publisher/model` id LM Studio shows, and the engine loads it straight from LM Studio's folder instead of downloading it again:

```bash
mlx-dspark serve --model lmstudio-community/Qwen3-8B-MLX-8bit
```

They also appear in `mlx-dspark models` and in the app's "On this Mac" list. LM Studio's **GGUF** files are a different format the loaders don't read: grab the `mlx-community` build of the same model instead.

The reverse, running mlx-dspark as an engine *inside* LM Studio's chat window, isn't possible, because LM Studio only runs its own bundled engines. But any client that takes a custom OpenAI endpoint can point at mlx-dspark.

## Serving other devices

```bash
mlx-dspark serve --model mlx-community/Qwen3-8B-8bit --host 0.0.0.0 --api-key YOUR-KEY
```

This listens on every interface. With a key set, every route except `/health` needs `Authorization: Bearer YOUR-KEY` or `x-api-key: YOUR-KEY`. In the Mac app, the same two switches are under **Settings › Local server**.
