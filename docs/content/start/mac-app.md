---
title: The Mac app
description: A native SwiftUI app around the same engine. Chat, a model manager that knows what fits, one-click agent setup, and a lab that shows speculative decoding working on your own prompts.
---

<div class="media-frame" markdown="0">
  <video class="theme-light-only" autoplay muted loop playsinline preload="metadata" poster="/static/media/race-light.jpg" aria-label="The app's Race: plain decoding against DFlash 2 on Qwen3.8-27B, identical output, 4.58 times faster">
    <source src="/static/media/race-light.mp4" type="video/mp4">
  </video>
  <video class="theme-dark-only" autoplay muted loop playsinline preload="metadata" poster="/static/media/race-dark.jpg" aria-label="The app's Race: plain decoding against DFlash 2 on Qwen3.8-27B, identical output, 4.58 times faster">
    <source src="/static/media/race-dark.mp4" type="video/mp4">
  </video>
</div>

The Race above ran plain decoding and DFlash 2 on Qwen3.8-27B (8-bit) side by side, then compared the token ids: identical output, 4.58× faster on that prompt.

## Install

With Homebrew:

```bash
brew tap ARahim3/mlx-dspark https://github.com/ARahim3/mlx-dspark
brew trust arahim3/mlx-dspark          # Homebrew 6+: third-party taps need explicit trust
brew install --cask mlx-dspark
xattr -dr com.apple.quarantine /Applications/mlx-dspark.app
```

The app isn't notarized yet. The last line clears the quarantine flag once. Alternatively, open it, then use **System Settings › Privacy & Security › Open Anyway** after the first launch. On Homebrew 5 or older, `brew install --cask --no-quarantine mlx-dspark` replaces the `trust` and `xattr` steps.

Or download the DMG from [Releases](https://github.com/ARahim3/mlx-dspark/releases) (the `app-v*` tags) and drag it to Applications.

## First launch

The app sets up its own private engine runtime the first time, which takes about 2–4 minutes once. It doesn't touch Homebrew's Python or any virtual environment of yours, and it never uses a `pip`-installed engine. After that it:

- keeps the engine on the **latest release automatically**, so engine fixes reach you without an app update;
- tells you when a newer version of the app itself exists (`brew upgrade --cask mlx-dspark`).

## What's inside

Chat
: Saved sessions, markdown and syntax highlighting, collapsible reasoning, and per-turn timing: time to first token, decode speed, tokens accepted per round, and how much of the prompt the cache served.

Models
: The measured pairs with their speedups, anything already on disk (including LM Studio's MLX downloads and your own model folders), and any Hugging Face repo. It answers **"will this fit my Mac?" before you download**, and swaps models without restarting the server.

Agents
: One-click setup for Claude Code, Codex, OpenCode, pi and any OpenAI-compatible app: the exact config for each, and a round-trip test.

Lab
: **Race** runs several decoders on the same prompt and checks the lossless claim token by token. **Live** streams per-round acceptance. **Curves** shows this Mac's measured cost curves. **This Mac** shows your measured memory bandwidth, the plain-decoding ceiling for the loaded model, how far above it speculation runs, and macOS memory pressure and swap.

Settings
: Decoding controls, a fixed engine port, serving on your local network with an API key, and your model folders.

Menu bar
: A live gauge with tokens per second, model memory, and acceptance.

## Serve other devices from the app

**Settings › Local server** has the same two switches as `mlx-dspark serve --host 0.0.0.0 --api-key <key>`: listen on every interface, and require a key. With a key set, every route except `/health` needs `Authorization: Bearer <key>` or `x-api-key`.

## Build it yourself

The app is a SwiftPM package with no Xcode project, and builds with the Command Line Tools alone. See [`apps/MacApp/README.md`](https://github.com/ARahim3/mlx-dspark/blob/main/apps/MacApp/README.md).
