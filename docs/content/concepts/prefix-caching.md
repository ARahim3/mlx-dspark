---
title: Prefix caching
description: Every turn of a chat or agent session re-sends the whole conversation. The server remembers what it has already read, so each turn reads only what's new.
---

The server keeps the model's KV cache (and the drafter's context) from the previous request, and reuses the longest shared prefix instead of reading it again. On by default, for every mode and every model.

| Measured | Cold | Cached |
|---|---|---|
| A ~750-token shared context (Qwen3) | 1132 ms | **87 ms** (about 13× faster) |
| Qwen3.8-27B 4-bit, ~8k-token system prompt, identical retry | 62 s | **0.21 s** |
| Same, next turn of the conversation | 62 s | **1.05 s** |
| Same, a new conversation over the same system prompt | 62 s | **0.53 s** |
| Ornith-1.0-9B, turn 2 (2420 of 2483 tokens reused) | 6.30 s | **1.15 s** |
| Qwen3.8-27B + DFlash 2, ~4k-token prompt, identical retry | 38.3 s | **0.24 s** (159×) |

It's lossless to the same standard as everything else: a cached turn differs from a cold one only at floating-point ties, and restores are validated array for array. The cache invalidates itself on any error, so it can't drift out of sync, and a miss costs nothing extra.

The standard `usage.prompt_tokens_details.cached_tokens` field (and `cached_tokens` in [`x_mlx_dspark`](/use/server/#see-the-speedup-on-every-response)) reports how much of each prompt the cache served.

## Two modes, picked for you

**Trim mode** (dense models like Qwen3, MiniCPM5): the cache is trimmed back to the shared prefix, and only the rest is read. For Gemma-4, whose sliding-window cache rotates, trimming is exact until the window first wraps; after that it switches to checkpoints automatically.

**Checkpoint mode** (hybrid models with a recurrent state, like Qwen3.6/3.8, Ornith and Bonsai, plus Gemma-4 once its window wraps, and DFlash drafters): a recurrent state can't be rolled back like a KV cache, so the server snapshots the state at stable points and restores a snapshot when a later prompt reaches it. Two things make that hit in practice:

- **Stable boundaries.** Chat templates often re-render the end of a conversation differently on the next turn (a thinking model's `<think>` opener, for example), so an exact-boundary snapshot would miss by a few tokens every time. The server measures each template's unstable tail at runtime and snapshots just before it.
- **Rungs and anchors.** Every 8192 tokens of a long prompt, the small recurrent state is also snapshotted (a *rung*), so a request that diverges mid-prompt, like a new session over the same system prompt or a compacted history, reuses everything up to the nearest rung. A miss that shared a long prefix with a cached conversation plants an *anchor* at the exact divergence point, so the next request of that shape hits it.

## Settings

`--no-prefix-cache`
: Turn it off.

`--prefix-cache-slots N`
: How many conversations to keep (default 2, so an agent and a chat don't evict each other every turn). Very short prompts (under 1024 tokens, like a title request) don't evict conversations four or more times longer.

`--prefix-cache-rungs N`
: Rung spacing for checkpoint mode, in tokens (default 8192; 0 disables).

`--prefix-cache-dir DIR` with `--prefix-cache-max-ram-mb N`
: An optional SSD tier: once the cache exceeds N MB of RAM, older entries spill to disk instead of being dropped.

## Good to know

- **Changing `reasoning_effort` mid-conversation is a full miss**, because the template renders the prompt differently from the start.
- **A client disconnect keeps the cache.** Generation stops at the next round, and the conversation stays cached for the retry.
- **Under memory pressure** the guard first drops the shallow rungs but keeps each conversation's two deepest (the static-prefix anchors new sessions restore from), and only empties the cache at critical pressure. A partial hit inherits the rungs below its restore point, so a daily change like a date line in a system prompt doesn't force a cold read.
- **The Claude Code comparison** in [Claude Code](/use/claude-code/#which-model) was measured before checkpoint mode could hit on hybrid models. It's the reason Qwen3-8B won there.
