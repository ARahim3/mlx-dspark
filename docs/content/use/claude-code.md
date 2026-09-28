---
title: Claude Code
description: Run Claude Code entirely on a model on your Mac, in two commands, without touching your normal Claude Code setup.
---

The server speaks Anthropic's Messages API, which is the dialect [Claude Code](https://claude.com/claude-code) talks. Start a server, then launch Claude Code against it:

```bash
mlx-dspark serve --model mlx-community/Qwen3-8B-8bit --no-thinking   # terminal 1
mlx-dspark claude                                                    # terminal 2
```

`mlx-dspark claude` finds the running server, points one Claude Code process at it, and hands over the terminal. Anything after `--` goes to `claude`:

```bash
mlx-dspark claude -- --continue
```

## It changes nothing outside that one process

The configuration lives in the launched process's environment and nowhere else. No shell profile is edited, no `settings.json` is written, and your login isn't replaced. Your other Claude Code sessions, open now or started later, keep their normal account, model and endpoint, and this one reverts the moment it exits.

Claude Code notes at startup that a credential variable takes precedence over your claude.ai login. That's just the notice: the login stays saved and untouched.

## Wiring it up yourself

If you'd rather configure it by hand, the launcher prints exactly what it would set:

```bash
mlx-dspark claude --print-env        # shell exports
mlx-dspark claude --print-settings   # an "env" block for a project's .claude/settings.local.json
```

The variables that matter:

| Variable | Why |
|---|---|
| `ANTHROPIC_BASE_URL` | Sends requests to the local server (`http://127.0.0.1:8080`). |
| `ANTHROPIC_AUTH_TOKEN` | Required. Without a credential variable Claude Code keeps authenticating with your claude.ai login, so the session still counts against your subscription. Any value works unless the server has an `--api-key`. |
| `ANTHROPIC_MODEL` | The served model's name, e.g. `Qwen3-8B-8bit`. |
| `ANTHROPIC_DEFAULT_OPUS/SONNET/HAIKU/FABLE_MODEL` | Point every model alias at the one loaded model. The **haiku** slot matters most: Claude Code sends background work like conversation titles there. |
| `CLAUDE_CODE_MAX_OUTPUT_TOKENS` | Keeps requested output under the server's cap. |

If the server was started with `--api-key`, pass the same key: `mlx-dspark claude --api-key KEY`. For a server on another machine, add `--url http://host:8080`.

## Which model

Claude Code sends **18–26k tokens** of system prompt and tool schemas with *every* request, so reading the prompt dominates the clock and reusing it between turns matters more than drafter quality. Measured on an M4 Pro, each a real Claude Code session that read a buggy file and fixed it with the Edit tool, thinking off:

| Model | Tokens accepted per round | Prefix cache | Wall clock |
|---|---|---|---|
| [Qwen3-8B](/models/qwen3-8b/) 8-bit | 3.01 | hit on 2 of 3 requests, ~26k tokens reused | **about 2:20** |
| [Ornith-1.0-9B](/models/ornith-1.0-9b/) 8-bit | **5.07** | off at the time (hybrid model) | about 4:10 |
| [Gemma-4 12B](/models/gemma-4-12b/) 8-bit | 3.68 | on, but its sliding window wraps at this prompt size | about 4:10 |

Qwen3-8B wins despite the lowest acceptance, because it re-reads the least. Ornith's 5.07 is the highest acceptance measured anywhere in this project (tool-call JSON is very predictable), but at the time it spent that win re-reading the prompt.

!!! note "Measured before hybrid models could cache"
    That table predates [checkpoint prefix caching](/concepts/prefix-caching/) (v0.7.0) and its stable-boundary rework (v0.10.1), which let hybrid models like Ornith, Qwen3.6 and Qwen3.8 reuse the prompt too, with thinking on or off. Every hybrid row should be faster now. It hasn't been re-measured, so the numbers stand as recorded.

## Tips

**Turn thinking off.** `--no-thinking` isn't required: reasoning streams back as proper `thinking` blocks either way. It's a speed knob. A reasoning model thinks before *every* tool call: on Qwen3-8B the same task took 3:17 and 2762 output tokens with thinking, against about 2:20 and 169 without. A client that sends `thinking: {"type": "disabled"}` gets the same effect per request. It's a no-op on Gemma-4, which doesn't think by default.

**Use a tool-calling model of about 8B or more.** Claude Code is a tool-use agent first, and smaller models flail at it.

**Leave prefix caching on.** It's doing most of the work. The first request to a cold server is the slow one. After that, even a new session over a system prompt the server has already seen reuses part of it.

**Context limits recover by themselves.** An over-long request is refused with the exact wording Claude Code recognizes as a context limit, so it compacts and retries instead of failing. `--context-window N` lowers that limit on purpose, for example to keep the KV cache inside your RAM.

**Long silences are fine.** Keep-alive frames flow every 15 seconds while nothing else is on the wire (a long prompt being read, or a tool call being assembled), so the client doesn't time out. If the client disconnects, generation stops at the next round instead of running on.

**Consider a leaner agent.** Claude Code's prompt is large. [pi](/use/agents-and-apps/#pi)'s is about 1.5k tokens, and the same one-bug fix takes about 6 seconds there.

## What gets translated

`POST /v1/messages` (streaming and non-streaming) and `POST /v1/messages/count_tokens`. Tool definitions, multi-turn `tool_use` / `tool_result` history, `stop_sequences` and system prompts are all translated to whatever the loaded model's chat template expects, including each family's own tool-call syntax. A reasoning model's thinking is lifted into Anthropic `thinking` blocks rather than leaking into the text. System messages that appear mid-conversation are folded into the adjacent user turn, and unknown request fields are ignored rather than rejected.
