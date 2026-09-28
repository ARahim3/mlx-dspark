---
title: Serving many requests
nav_title: Batching
description: Several agents or users hitting one server at once. Batching and speculation are substitutes here, not complements, and knowing why saves you tuning time.
---

```bash
mlx-dspark serve --model mlx-community/Qwen3-4B-8bit --max-batch 4
```

`--max-batch N` runs up to N concurrent requests through **one** batched forward pass, so they share a single read of the weights per step. For a few local agents hitting the server at once, that's a large aggregate win. A request running alone, or one that needs penalties, logprobs or sampled speculative decoding, takes the normal single-request path, so per-request latency never gets worse.

Batching is **continuous**: a request returns the moment it finishes, without waiting for the batch's slowest member, and its slot admits the next queued request mid-flight. Measured: a short request joining two long ones returned at 2.3 s while they ran on to 8.4 s.

## Measured

Aggregate tokens per second, 8 varied prompts of 96 tokens, M4 Pro, batched against the same prompts run one after another:

| Model | 2 at once | 4 at once | 8 at once |
|---|---|---|---|
| Qwen3-4B 8-bit (dense) | 1.70× | 3.05× | **4.00×** |
| Ornith-1.0-9B 8-bit (hybrid) | 1.94× | **3.52×** | 3.08× |
| Qwen3.6-35B-A3B 4-bit (hybrid MoE) | 1.30× | 1.73× | **2.11×** |

**The mixture of experts batches worst, not best,** which is the opposite of the intuition that a sparse model has more to gain. A dense model's batch rows share its *entire* weight read. An MoE's rows share only the non-expert parameters (about 2.8B here), and every additional row pulls in its own set of routed experts. Sparsity is what makes an MoE fast for one request, and the same property leaves it less to share across many.

## Batching and speculation are substitutes

Both work by getting more tokens out of each read of the weights. Once batching has filled the machine, extra verify width stops being cheap: on Qwen3-4B, verifying 4 rows costs 1.11× one row for a single request, but 2.10× at 16 concurrent requests. End to end at 4 concurrent requests, batched DSpark lands at **0.97×** of batched plain decoding, a wash.

So speculation's value is largest for a *single* stream, and batching's is largest for many. The drafter card's own CUDA numbers show the same shape (2.9× for one stream, 1.9× at capacity). With `--max-draft auto` the cap is calibrated per batch width, and at 4 concurrent requests it picks a longer block, worth about 5%.

## What batches

- **Dense `mlx-lm` models** batch both plain and DSpark decoding.
- **Hybrid models** (Ornith, Bonsai, Qwen3.6/3.8 and the other recurrent families) batch plain decoding. Their recurrent state is a fixed-size summary, so rows of different prompt lengths merge by plain concatenation. Batched *speculative* decoding stays dense-only: each row of a speculative round rolls back by a different amount, which a KV cache handles with per-row metadata but a recurrent state doesn't. Such a model batches its plain decoding and takes the serial path for speculation; nothing silently degrades.
- **Gemma-4** (through `mlx-vlm`, with a rotating cache) serves requests one at a time.
- **`--kv-bits`** disables batching.

A batched quantized model isn't bit-identical to one-at-a-time decoding (about 0.5% of near-tie tokens flip, because the quantized matmul takes a different path at batch width). That's inherent to any batched quantized server, not specific to speculation. Output stays greedy-correct per request.
