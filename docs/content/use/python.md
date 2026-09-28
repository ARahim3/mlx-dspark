---
title: Python
description: Load a model and its drafter in one call, generate, and read back acceptance and speed. For the full tuned stack, talk to the server instead.
---

## DSpark

```python
from mlx_dspark import load_pair, speculative_generate

target, tok, drafter, cfg = load_pair("mlx-community/Qwen3-8B-8bit")   # drafter auto-resolved
res = speculative_generate(target, tok, drafter, "Explain how rainbows form.",
                           max_new_tokens=300)
print(res.text)
print(res.mean_accept_len, res.decode_tokens_per_sec)
```

`load_pair` takes any repo or local path. For [measured models](/models/) the drafter resolves automatically; otherwise pass `drafter="…"`.

## DFlash and DFlash 2

```python
from mlx_dspark import load_dflash_pair, dflash_generate

target, tok, drafter, cfg = load_dflash_pair("mlx-community/Qwen3.8-27B-4bit")   # DFlash 2
res = dflash_generate(target, tok, drafter, "Write a binary search in Python.")
print(res.text, res.mean_accept_len)
```

`max_draft_tokens=None` (the default for `dflash_generate`) drafts the full block.

## Drafter-free lookup and plain decoding

```python
from mlx_dspark import load_target, lookup_generate, greedy_generate

target, tok = load_target("mlx-community/Qwen3-8B-8bit")
res = lookup_generate(target, tok, "Rewrite this function with type hints: …")
base = greedy_generate(target, tok, "Explain how rainbows form.")
```

## Choosing the draft cap

The library doesn't measure your Mac for you: `speculative_generate` defaults to `max_draft_tokens=2`, and `None` means the full block. To get the same measured, per-machine behaviour the CLI has, calibrate once (the result is cached on disk) and pass the controller:

```python
from mlx_dspark import calibrate

ctrl = calibrate(target, drafter, mode="dspark",
                 target_repo="mlx-community/Qwen3-8B-8bit",
                 drafter_repo="deepseek-ai/dspark_qwen3_8b_block7")
res = speculative_generate(target, tok, drafter, "…", cap_controller=ctrl)
```

The controller picks the cap each round from the measured cost curves and live acceptance. That's `--max-draft auto` on the command line. See [Draft caps](/concepts/draft-caps/).

## Sampling, stops and streaming

```python
res = speculative_generate(
    target, tok, drafter, "Write a short poem.",
    temperature=1.0, top_p=0.95, seed=0,          # lossless speculative sampling
    stop=["\n\n"],
    on_text=lambda s: print(s, end="", flush=True),  # stream text as it's committed
)
```

`temperature=0` (the default) is greedy. With a temperature, drafts are accepted by the speculative-sampling rule, so the output is an exact sample from the model at that temperature, not an approximation.

## What you get back

`GenResult` fields worth knowing:

| Field | Meaning |
|---|---|
| `text`, `token_ids` | The output. |
| `num_tokens`, `num_rounds`, `target_forwards` | Tokens produced, speculation rounds, and passes through the model. |
| `mean_accept_len` | Tokens committed per round, on average. |
| `tokens_per_sec` | End to end, including reading the prompt. |
| `decode_tokens_per_sec` | Generation only, prompt excluded. |
| `prefill_seconds`, `ttft_seconds` | Time spent reading the prompt, and time to the first text. |
| `finish_reason` | `"stop"` or `"length"`. |
| `accept_lengths` | Per-round acceptance, for your own analysis. |

## The library vs the server

The command line and server switch on extras that the library leaves off by default: the custom verify and attention kernels (each enabled per shape only after a one-time probe proves it faster on your Mac), CPU co-prefill, prefix caching, the warm-up pass and the memory guard. If you want all of that from Python, run `mlx-dspark serve` and use any OpenAI or Anthropic client. See [Run a server](/use/server/).
