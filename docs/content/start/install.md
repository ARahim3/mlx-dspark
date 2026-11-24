---
title: Install
description: One pip install gives you the command line, the API server and the Python library. The Mac app wraps the same engine for people who'd rather click.
---

## What you need

- A Mac with Apple Silicon (M1 or newer).
- Python 3.10 or newer, for the command line and the Python API. The Mac app brings its own.
- Enough memory for the model you want. Each [model page](/models/) lists its measured peak, and [Choose a model](/start/choose-a-model/) matches models to your Mac.

## Install the engine

```bash
pip install mlx-dspark
```

Or, with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install mlx-dspark
```

That brings in MLX 0.32 or newer, `mlx-lm` and `mlx-vlm`. There's no server framework to install: the API server is built on the Python standard library. No model weights are bundled; they download from Hugging Face the first time you use a model.

Check your setup:

```bash
mlx-dspark doctor
```

`doctor` checks that you're on Apple Silicon, reports the MLX stack it found and your memory, and lists every folder it will look in for models. Add `--models` to see which supported models fit this Mac and which are already downloaded.

## Or install the Mac app

```bash
brew tap ARahim3/mlx-dspark https://github.com/ARahim3/mlx-dspark
brew trust arahim3/mlx-dspark          # Homebrew 6+: third-party taps need explicit trust
brew install --cask mlx-dspark
xattr -dr com.apple.quarantine /Applications/mlx-dspark.app   # the app isn't notarized yet
```

The app is a separate download with its own private runtime; see [The Mac app](/start/mac-app/) for the DMG and what's inside. `pip install mlx-dspark` stays engine-only.

## What happens the first time you run a model

Nothing needs configuring, but the first run does a little more than later ones:

1. **The model and its drafter download** (or load from your Hugging Face cache, LM Studio's folder, or any folder in [`MLX_DSPARK_MODEL_DIRS`](/reference/environment/)).
2. **mlx-dspark measures your Mac for that model.** It times how verification cost grows with draft width, what the drafter costs, and which of its kernels beat MLX's on this machine. That takes about 5–20 seconds, and it's how the engine picks settings for *your* chip instead of the one the published numbers came from. The result is cached per model, quantization and MLX version.
3. **The server runs a short warm-up generation**, so your first real request doesn't pay the kernel-compile cost.

Later runs skip the first two steps.

## Upgrade

```bash
pip install -U mlx-dspark
```

After an MLX upgrade, the measurement from step 2 re-runs once by itself, because the cache is keyed by MLX version. The Mac app keeps its engine on the latest release automatically. Update the app itself with `brew upgrade --cask mlx-dspark`.

## Install from source

```bash
git clone https://github.com/ARahim3/mlx-dspark
cd mlx-dspark
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
python -m pytest tests/ -q        # the model-free test suite, no downloads
```

Next: [the quickstart](/start/quickstart/).
