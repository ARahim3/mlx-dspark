---
title: HTTP API reference
nav_title: HTTP API
description: Every route the server answers, the control plane the Mac app is built on, and the fields it adds to responses.
---

The server listens on `http://127.0.0.1:8080` by default (`--host`, `--port`). With `--api-key`, **every route except `GET /health`** requires `Authorization: Bearer <key>` or `x-api-key: <key>`, and a wrong key gets `401`.

Every route also answers without the `/v1` prefix (`/chat/completions`, `/messages`, …).

## Generation

| Route | Dialect | Notes |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions | Streaming and not; `n`; tools; `reasoning_content` split; `stream_options.include_usage`. |
| `POST /v1/completions` | OpenAI Completions | Raw prompt, no chat template. |
| `POST /v1/messages` | Anthropic Messages | Streaming and not; tools; `thinking` blocks; `stop_sequences`. |
| `POST /v1/messages/count_tokens` | Anthropic | Token count for a would-be request. |
| `POST /v1/responses` | OpenAI Responses | Streaming and not; stateless (no `previous_response_id`). |
| `GET /v1/models` | OpenAI / Anthropic | The one loaded model, with a `display_name`. |

Generation routes answer `503` while no model is loaded (`--no-model`, or during a swap). A prompt longer than the context window is refused with *"prompt is too long"*, the wording agent clients recognize and compact on. Unknown request fields are ignored, never rejected.

Accepted request parameters: `temperature`, `top_p`, `top_k`, `max_tokens`, `stop` / `stop_sequences`, `seed`, `presence_penalty`, `frequency_penalty`, `logprobs`, `top_logprobs`, `tools`, `enable_thinking`, `thinking`, `reasoning_effort`. Unset sampling values fall back to `--default-temperature` / `--default-top-p` / `--default-top-k`, then to the model's own `generation_config`. `max_tokens` defaults to `--default-max-tokens` (2048) and is capped at `--max-tokens-cap` (32768).

### The `x_mlx_dspark` block

Every response carries this non-standard block. Clients ignore it; it's how you see the speedup.

| Field | Meaning |
|---|---|
| `mode` | `dspark`, `dflash`, `lookup` or `baseline`. |
| `cap` | Draft cap used for this request, after any depth adjustment. |
| `accept_len` | Tokens committed per round. |
| `tokens_per_sec` | End to end, including prompt reading. |
| `decode_tokens_per_sec` | Generation only. |
| `target_forwards` | Passes through the model. |
| `lookup_rounds` | Rounds whose draft came free from lookup (when any). |
| `prompt_tokens`, `cached_tokens`, `completion_tokens`, `context_tokens` | Token accounting; `cached_tokens` is what the prefix cache served. |
| `prefill_seconds`, `decode_seconds`, `ttft_seconds` | Where the time went. |
| `prefill_tokens_per_sec` | Prompt-reading speed (when at least 16 new tokens were read). |
| `ceiling_tokens_per_sec`, `roofline_ratio` | The plain-decoding ceiling this Mac's measured bandwidth allows at this depth, and decode speed as a multiple of it. |
| `decay_ratio`, `swap_delta_bytes`, `cold` | Diagnostics: late-vs-early speed within the request, swap growth during it, and whether it was the first request after a load without warm-up. |

The OpenAI endpoints also fill the standard `usage.prompt_tokens_details.cached_tokens`.

## Status and telemetry

| Route | Returns |
|---|---|
| `GET /health` | Always open, even with `--api-key` and mid-swap. `status` is `ok`, `loading` or `no_model`. When ready: model, mode, target and drafter, `max_draft`, `lookup_drafts`, `confidence_threshold`, `context_window`, `kv_bits`, the kernel and prefill switches (`small_m`, `multirow_attn`, `sdpa_split`, `cpu_split`, `drafter_window`), `warmup`, `memory_guard`, `max_output_tokens`, `supports_reasoning_effort`, `thinking_default`, `reasoning_effort`, and a `warnings` list (memory pressure, context-window RAM). While loading: `phase` (`loading` or `warming_up`) and `download` progress. |
| `GET /metrics` | Aggregate counters, allocator memory, macOS memory pressure and swap, the last roofline verdict, memory-guard state. |
| `GET /machine` | Chip, measured memory bandwidth, the loaded model's bytes per token and single-stream ceiling, and a verdict. Answers without a model too. |
| `GET /calibration` | This Mac's measured cost curves for the loaded pair. |
| `GET /events` | Server-sent events: one event per speculation round (drafted, accepted, committed, cap, timing) and prefill progress events. |
| `GET /rounds?limit=N` | Recent rounds and per-position acceptance stats, for clients that would rather poll. |
| `GET /doctor` | The same report as `mlx-dspark doctor --json`. Answers without a model. |

### `/events`

Round events have no `type` key: `{"drafted": 7, "accepted": 5, "committed": 6, "cap": 7, "source": "drafter", …}`, where `source` is `drafter`, `lookup` (a free lookup draft) or `plain` (a parked or plain step). Prefill events do: `{"type": "prefill", "req": …, "mode": …, "processed": 4096, "total": 20000}`, with positions counted from the start of the prompt (so they begin at the cached length), and a final `{"type": "prefill", …, "done": true}` on every exit path, so a progress display always clears. The stream ends when the model is swapped.

## Control plane

These are what the Mac app drives. Each is a plain HTTP endpoint first, so a script can do anything the app can.

| Route | What it does |
|---|---|
| `POST /admin/load` | Load or swap the model in place; the port stays the same, so connected clients keep working. Body below. A first-time load reports download progress in `/health` and resumes partial downloads. |
| `POST /admin/load/cancel` | Cancel a download in progress. |
| `POST /admin/unload` | Free the model; the server stays up. |
| `GET /admin/status` | Loading state and errors. |
| `GET /admin/models` | Registry models, what's on disk (Hugging Face cache, LM Studio, your model folders), disk usage, and this Mac's bandwidth relative to the M4 Pro the published numbers came from. Answers without a model. |
| `GET /admin/integrations` | Ready-to-paste configs for Claude Code, Codex, OpenCode, pi and any OpenAI-compatible app, filled in with this server's real address, model and key. |
| `POST /admin/race` | Run several decoders on one prompt and stream the result as SSE, ending with a token-by-token identical-output verdict. Body: `{"prompt": "…", "arms": [{"mode": "dspark", "cap": 4}, {"mode": "baseline"}], "max_tokens": 200}`. |

### `POST /admin/load` body

Only `model` is required. Every other key is an override for this load; omit it to keep the pair's measured default or the server's setting.

| Key | Type | Meaning |
|---|---|---|
| `model` | string | Repo id or local path. |
| `mode` | string | `auto`, `dspark`, `dflash`, `lookup` or `baseline`. |
| `max_draft` | int or `"auto"` | Pin the cap, or adapt per round. |
| `lookup_drafts` | bool | Hybrid lookup drafts on or off. |
| `confidence_threshold` | number, 0–1 | 0 disables. |
| `context_window` | int | Cap the prompt length (a RAM lever). |
| `kv_bits` | 0, 4 or 8 | Quantized KV cache; 0 = full precision. |
| `drafter_window` | int ≥ 0 | DSpark drafter context window; 0 = whole context. |
| `small_m`, `multirow_attn`, `sdpa_split` | bool | Kernel switches, for A/B runs. |
| `cpu_split` | number, 0 ≤ x < 1 | CPU co-prefill fraction; 0 = off. |
| `warmup` | bool | Run the warm-up generation after loading. |
| `memory_guard` | bool | The memory-pressure guard. |
| `enable_thinking` | bool | The thinking default for clients that don't say (sticky across swaps). |
| `reasoning_effort` | string | `low`, `medium`, `high` or `xhigh`: the default depth (sticky across swaps). |

There's deliberately no per-request way to allow remote code: that's `--trust-remote-code` for the whole process.

```bash
curl -X POST http://127.0.0.1:8080/admin/load \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/Qwen3.8-27B-4bit", "context_window": 65536, "enable_thinking": false}'
```
