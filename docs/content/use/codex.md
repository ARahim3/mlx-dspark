---
title: Codex
description: OpenAI's Codex CLI talks the Responses API. The server speaks it, so Codex can run on your local model.
---

[Codex](https://github.com/openai/codex) only talks to providers through OpenAI's **Responses API** (it dropped Chat Completions). mlx-dspark serves it at `/v1/responses`, so pointing Codex at a local model is a provider entry in its config.

## Set it up

Start a server:

```bash
mlx-dspark serve --model mlx-community/Qwen3-8B-8bit --no-thinking
```

Add a provider and a profile to `~/.codex/config.toml`:

```toml title="~/.codex/config.toml"
[model_providers.mlx-dspark]
name = "mlx-dspark"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
env_key = "MLX_DSPARK_KEY"

[profiles.mlx-dspark]
model_provider = "mlx-dspark"
model = "Qwen3-8B-8bit"
```

Then run it with that profile:

```bash
export MLX_DSPARK_KEY=mlx-dspark      # any value, unless the server has an --api-key
codex --profile mlx-dspark
```

`model` is the served model's name, the last part of the repo id. `curl http://127.0.0.1:8080/v1/models` prints it.

## What's supported

`POST /v1/responses`, streaming and non-streaming.

- `input` as a bare string or as the structured item list: `message`, `function_call` and `function_call_output` items.
- `tools` in the Responses API's flat shape, translated to the loaded model's own tool-call syntax, with multi-turn tool use round-tripping like the other dialects.
- Reasoning comes back as `reasoning` output items, separate from the answer.

It's a **stateless** implementation, like Ollama's: there's no `previous_response_id` or server-side conversation store. Codex resubmits its own history on every request anyway, so nothing is lost, and the [prefix cache](/concepts/prefix-caching/) makes that resubmission cheap.

The same prompt-size advice as for [Claude Code](/use/claude-code/#tips) applies: prefix caching is doing the heavy lifting, thinking off is faster, and a tool-calling model of about 8B or more works best.
