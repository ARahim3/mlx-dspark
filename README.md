<p align="center">
  <img src="https://raw.githubusercontent.com/ARahim3/mlx-dspark/main/mlx-dspark.png" alt="mlx-dspark" width="440">
</p>

<p align="center">
  <b>DeepSeek's DSpark <i>and</i> z-lab's DFlash speculative decoding — native on Apple Silicon via <a href="https://github.com/ml-explore/mlx">MLX</a>.</b>
  <br>Lossless drafters (same output, just faster) for the <b>Qwen3, Gemma-4, and PrismML Bonsai-27B</b> families —
  <br>plus any matched DSpark / DFlash checkpoint. Run them at the CLI, from Python, or <b>serve an OpenAI-compatible API</b> to LM Studio / any local tool.
</p>

<p align="center">
  <a href="https://pypi.org/project/mlx-dspark/"><img src="https://img.shields.io/pypi/v/mlx-dspark?color=2563eb" alt="PyPI"></a>
  <img src="https://img.shields.io/pypi/pyversions/mlx-dspark" alt="Python">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-111111?logo=apple&logoColor=white" alt="Apple Silicon">
  <a href="https://github.com/ARahim3/mlx-dspark/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/ARahim3/mlx-dspark/main/docs/demo.gif" alt="Baseline vs DSpark — same output, ~1.8x faster on Gemma-4 12B" width="840">
</p>

mlx-dspark runs two EAGLE-family speculative-decoding drafters natively on Apple Silicon: DeepSeek's
**DSpark** (semi-autoregressive, from the [DeepSpec](https://github.com/deepseek-ai/DeepSpec) codebase,
used to accelerate DeepSeek-V4) and z-lab's **DFlash** (block diffusion). Both are **lossless** — the
target verifies every token, so output is identical to normal decoding — and run under one verify loop,
so you can serve them, script them, or benchmark them head-to-head.

> **What this is *not*:** DeepSeek-V4 inference. The targets are consumer-size models (Gemma-4, Qwen3,
> PrismML's ternary Bonsai-27B) with published DSpark drafters — so this runs the real drafter method on a
> Mac, but the model producing tokens is Gemma / Qwen / Bonsai, not V4. V4 Flash/Pro (MoE, batched serving)
> is DSpark's own headline use case.

## Install

```bash
pip install mlx-dspark          # or:  uv pip install mlx-dspark
```

Apple Silicon + Python ≥ 3.10; installs mlx ≥ 0.32.0 automatically (0.32's quantized-matmul kernels are
what current speedup numbers are measured on). Model weights download from the Hugging Face cache on
first use (none bundled). No server framework is pulled in — the API server is built on the standard
library.

> **Known upstream incompatibility (worked around since 0.3.2):** mlx-vlm **0.6.4** ×
> transformers **≥ 5.12** breaks loading the gemma4 target with a misleading
> `OSError: Can't load video processor …` ([#4](https://github.com/ARahim3/mlx-dspark/issues/4),
> upstream [Blaizzy/mlx-vlm#1578](https://github.com/Blaizzy/mlx-vlm/issues/1578) — fixed on
> mlx-vlm main, unreleased). mlx-dspark ≥ 0.3.2 shims it at load time, so any mlx-vlm ≥ 0.6.3
> works; on older mlx-dspark, pin `mlx-vlm==0.6.3`. `mlx-dspark doctor` reports when the shim
> is active.

## Quickstart

You name the **target model** (`--model`, an HF repo or local path, exactly like `mlx-lm`); the matching
drafter is resolved automatically for known targets (see [Models](#models)), or pass `--drafter`.

### Serve an OpenAI-compatible API

```bash
mlx-dspark serve --model mlx-community/Qwen3-8B-8bit        # → http://127.0.0.1:8080/v1
#   --max-batch 4   continuous batching: up to 4 concurrent requests share each forward
#                   (~2.5× aggregate; a finished request returns immediately, its slot
#                   admits the next one mid-flight)
#   --kv-bits 8     quantized KV cache (long-context bandwidth saver)
#   --mode auto|dspark|dflash|lookup|baseline   ·   --no-thinking   ·   --api-key KEY
```

`--mode auto` picks the best available speculation for any target (a known DSpark drafter → else
DFlash → else drafter-free n-gram **lookup**), so *any* repo serves with some speedup and no extra flags.

Then point any OpenAI client at it — the speculative speedup is transparent:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")
print(client.chat.completions.create(
    model="Qwen3-8B-8bit",
    messages=[{"role": "user", "content": "Explain rainbows briefly."}],
).choices[0].message.content)
```

**For LM Studio / other tools:** set the OpenAI base URL to `http://127.0.0.1:8080/v1`.

The server speaks the OpenAI API: `POST /v1/chat/completions` (streaming **and** non-streaming,
multi-turn), `POST /v1/completions`, `GET /v1/models`, `GET /health`, `GET /metrics`. It supports
`temperature`, `top_p`, `top_k`, `max_tokens`, `stop`, `seed`, `presence_penalty` / `frequency_penalty`,
`logprobs` / `top_logprobs`, **tool calling** (`tools` / `tool_calls`), and a per-request thinking toggle
(`enable_thinking`). Each response carries an `x_mlx_dspark` block (accept length + tok/s) so the
spec-decode gain is visible. **Continuous batching** (`--max-batch N`) serves concurrent requests in one
batched forward for ~2.5× aggregate throughput (see [Concurrent throughput](#concurrent-throughput));
**prefix caching** (on by default) reuses the conversation prefix so multi-turn chat doesn't re-prefill
each turn (~13× faster follow-up turns on a long shared context — see [Prefix caching](#prefix-caching)).

### One-shot generation (CLI)

```bash
# downloads the drafter + instruct target on first run
mlx-dspark generate --model mlx-community/Qwen3-4B-8bit --prompt "Explain how rainbows form."

# baseline (plain target) vs dspark — same output, faster (record each, stack for a demo)
mlx-dspark generate --model mlx-community/Qwen3-4B-8bit --mode baseline --prompt "..." --max-new-tokens 400
mlx-dspark generate --model mlx-community/Qwen3-4B-8bit --mode dspark   --prompt "..." --max-new-tokens 400

# z-lab DFlash drafter (--max-draft 0 = full 16-block, its native operating point)
mlx-dspark generate --model mlx-community/gemma-4-12B-it-8bit --mode dflash --max-draft 0 --prompt "Write a binary search."

# sampled (not greedy) — lossless w.r.t. the target at temperature T (dspark and dflash)
mlx-dspark generate --model mlx-community/Qwen3-4B-8bit --prompt "Write a short poem." --temperature 1.0 --top-p 0.95 --seed 0
```

`python -m mlx_dspark …` works too, and the old flat `--prompt …` form still maps to `generate`.

### Python

```python
from mlx_dspark import load_pair, speculative_generate

target, tok, drafter, cfg = load_pair("mlx-community/Qwen3-8B-8bit")   # drafter auto-resolved
res = speculative_generate(target, tok, drafter, "Explain how rainbows form.")
print(res.text, res.mean_accept_len, res.tokens_per_sec)
```

```python
from mlx_dspark import load_dflash_pair, dflash_generate   # z-lab DFlash instead

target, tok, drafter, cfg = load_dflash_pair("mlx-community/gemma-4-12B-it-8bit")
res = dflash_generate(target, tok, drafter, "Write a binary search in Python.")  # max_draft_tokens=None = full block
print(res.text, res.mean_accept_len, res.tokens_per_sec)
```

## Models

Pass **any** target repo/path to `--model`; the matched drafter auto-resolves for the targets below
(quantization-agnostic — a `-4bit` / `-8bit` / `-bf16` of the same model resolves the same drafter). For
anything else, add `--drafter <repo>`. Run `mlx-dspark models` to print this table.

| target (`--model`) | DSpark drafter (`--mode dspark`) | DFlash drafter (`--mode dflash`) | peak RAM |
|---|---|---|---|
| `mlx-community/Qwen3-4B-8bit`        | `deepseek-ai/dspark_qwen3_4b_block7`   | `z-lab/Qwen3-4B-DFlash-b16`  | ~8 GB  |
| `mlx-community/Qwen3-8B-8bit`        | `deepseek-ai/dspark_qwen3_8b_block7`   | `z-lab/Qwen3-8B-DFlash-b16`  | ~11 GB |
| `mlx-community/gemma-4-12B-it-8bit`  | `deepseek-ai/dspark_gemma4_12b_block7` | `z-lab/gemma4-12B-it-DFlash` | ~15 GB |
| `prism-ml/Ternary-Bonsai-27B-mlx-2bit` | `Rahim/Ternary-Bonsai-27B-dspark`    | — | ~12 GB |

*Peak RAM* is measured on an M4 Pro (8-bit target + 4-bit drafter + KV cache); add headroom for macOS.
A 4-bit target (`--model …-it-4bit`) roughly halves the target's share (fits smaller Macs). **Use the
matched *instruct* target** the drafter was trained against — a base model drops acceptance sharply. The
legacy `--family qwen3|gemma4` flags still work but are deprecated in favor of `--model`.

`--drafter` lets you run **any** other matched z-lab / DeepSpec checkpoint with no code change — e.g.
**Qwen3-14B** (DSpark-only; z-lab published no 14B DFlash; ~18 GB peak; benchmarked below):

```bash
mlx-dspark generate --model mlx-community/Qwen3-14B-8bit \
  --drafter deepseek-ai/dspark_qwen3_14b_block7 --prompt "Explain how rainbows form."
```

### Bring your own drafter — what runs and what doesn't

New DSpark/DFlash drafters keep landing on HF in **three different packagings**; here is the honest
compatibility contract (loaders refuse incompatible checkpoints with an error naming the reason, never
a silent mis-load):

| checkpoint style | example | status |
|---|---|---|
| **DeepSpec-native standalone drafter** (qwen3/gemma4 backbone, any size/quant) | `deepseek-ai/dspark_qwen3_14b_block7` | ✅ runs via `--drafter` (4B/8B/14B/gemma-12B measured on an M4 Pro; larger sizes should run — reports welcome) |
| **z-lab DFlash adapter** for a qwen3/gemma4-family target | `z-lab/Qwen3-8B-DFlash-b16` | ✅ runs via `--mode dflash --drafter` |
| **PrismML dspark GGUF** (Bonsai-27B) | `prism-ml/Ternary-Bonsai-27B-gguf` → `*-dspark-bf16.gguf` | ✅ pre-converted repacks auto-resolve (`Rahim/*-dspark`); any future GGUF-only drop runs via `--drafter gguf:<repo>/<file>.gguf` (converted locally, once) |
| **vLLM "speculators" format** | `RedHatAI/GLM-5.2-speculator.dspark` | ❌ different config schema — not yet ([issue?](https://github.com/ARahim3/mlx-dspark/issues)) |
| **Full model with embedded drafter** | `deepseek-ai/DeepSeek-V4-Pro-DSpark` (893 GB, MLA+MoE) | ❌ different architecture & packaging — out of scope for consumer Macs |
| **DFlash+Markov community hybrids** | `Hikari07jp/DSpark-Gemma-4-31B-draft` | ❌ hybrid head — not yet |

Targets: any dense mlx-lm text model routes automatically; a one-time load probe verifies the
hidden-state tap reproduces the model's own forward and fails loudly if the family needs bespoke
support (drafter-free `--mode lookup` / `--mode auto` still work with **any** target). If you run a
pair we haven't measured, `mlx-dspark benchmark --json` produces a device-stamped result we can fold
into the table — please share it.

### PrismML Bonsai 27B (ternary / 1-bit Qwen3.6-27B)

[Bonsai 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-mlx-2bit) is PrismML's 1.7-bit
ternary (and 1-bit) rebuild of Qwen3.6-27B — a full 27B-class reasoning model in ~8 GB. It ships
with a DSpark drafter that PrismML publishes **GGUF-only** and, per their own docs, accelerates
their CUDA path but "not Macs yet". mlx-dspark runs it on a Mac:

```bash
mlx-dspark generate --model prism-ml/Ternary-Bonsai-27B-mlx-2bit \
  --max-draft auto --prompt "Implement binary search in Python."
# first run: downloads the target (8.5 GB) + the matched drafter (6.8 GB bf16 safetensors —
# our 1:1 repack of PrismML's GGUF-only drafter, quantized to 4-bit at load)
```

Measured on an M4 Pro 48 GB (greedy, warm, interleaved medians): baseline ~25.6 tok/s;
**1.1–1.2× on code/structured content** (acceptance ~2.8/round at cap 2). Output is lossless —
byte-identical to plain greedy decoding. Bonsai's backbone is **hybrid linear attention**
(48 of 64 layers carry recurrent state, which can't be rolled back like a KV cache), so
mlx-dspark uses a snapshot/replay verify designed for it — as far as we know the first working
speculative decoding for this model family on Apple Silicon.

Two honest caveats: speculation pays on code/structured output but is a net loss on open-ended
chat with this target (a 2-bit model's verify-width cost is steep relative to its fast plain
step), so use **`--max-draft auto`** — it tracks live acceptance and *parks* speculation
(running plain pipelined steps) whenever it would lose, matching the best fixed cap on code
while staying within ~10% of baseline on chat. And requires **mlx ≥ 0.32.0** (older mlx lacks
the multi-row 2-bit matmul path that makes verification affordable). Prefix caching and
batching remain dense-target-only. Baseline/`--mode lookup` also work for any other `qwen3_5`
(Qwen3.5/3.6-family) checkpoint.

The **1-bit** `Bonsai-27B-mlx-1bit` pack does *not* run: it is quantized to 1 bit for
PrismML's own MLX fork, and stock `mx.quantize` has no 1-bit mode (2/3/4/5/6/8 only) —
`load_target` refuses it with exactly that reason. The ternary 2-bit variant is the
stock-MLX operating point.

## How it works

- **DSpark** — a parallel backbone (5 layers) consumes the target's hidden states (EAGLE3-style) and
  proposes a 7-token block at once; a rank-256 **Markov head** adds a cheap previous-token correction that
  kills "suffix decay"; a confidence head scores each position (optional adaptive block length).
- **DFlash** (`--mode dflash`) — a *block-diffusion* drafter that denoises a whole 16-token block in one
  parallel pass and reuses the target's own embed/lm-head. Different trade-offs (see below).
- The target **verifies** every token, so output is **greedy-correct by construction** (identical to plain
  decoding up to floating-point tie-breaking). `--temperature > 0` switches to lossless speculative
  *sampling* — an exact sample from the target at temperature T (with `--top-p` / `--top-k`).

The drafter loads 1:1 from the HF checkpoint and is **4-bit quantized** by default (cheap to run each round;
quantization doesn't change acceptance — that's set by the drafter↔target match).

## Which target & drafter should I use?

Short answer on current mlx (≥ 0.32): **DSpark, everywhere** (`--mode auto` picks it for you). Measured on
an M4 Pro, warm (code prompt unless marked chat):

| target | DSpark (`--mode dspark`, cap 2) | DFlash (`--mode dflash --max-draft 0`) | pick |
|---|---|---|---|
| **Gemma-4 12B** | **2.11×** code, 1.77× chat | 1.63× code, ~0.7× chat | **DSpark** |
| **Qwen3-8B** | **1.90×** | 0.86× full block (1.47× at cap 2) | **DSpark** |
| **Qwen3-4B** | **1.64×** | modest | **DSpark** |
| **Ternary-Bonsai-27B** | **1.1–1.2×** code (`--max-draft auto`) | — | **DSpark** (auto) |

This is a *version-dependent* verdict worth knowing about: on mlx 0.31, verify cost rose steeply with the
number of tokens verified, which made DFlash's full 16-block the winner on Gemma-12B code/math (~2.1× vs
DSpark's ~1.9× then). mlx 0.32's quantized-matmul kernels made *narrow* multi-row verify disproportionately
cheaper, and DSpark's short block now wins across the board here. If your mlx/hardware differs,
`--max-draft auto` re-measures the curves on your machine, and `mlx-dspark benchmark` settles it empirically.

For target *precision*: **8-bit** is the sweet spot (best acceptance + quality); **4-bit** gives the highest
absolute throughput and fits smaller Macs but a smaller speedup ratio; bf16 is *slower* on M-series (verify
dominates). The drafter stays 4-bit either way. Full numbers and the reasoning are in
[Benchmarks & deep dive](#benchmarks--deep-dive).

## Results at a glance

**DSpark** vs plain greedy decoding of the same model, at its `cap=2` optimum (M4 Pro 48 GB, warm,
code prompt, 8-bit instruct target, 4-bit drafter, **mlx 0.32.0** — whose quantized-matmul kernels
lifted every row well past the mlx-0.31 numbers this README previously carried):

| target | accept len | baseline | mlx-dspark | speedup |
|---|---|---|---|---|
| **Gemma-4 12B** | ~2.75 | 17.0 tok/s | 35.9 tok/s | **2.11×** |
| **Qwen3-14B**   | ~2.50 | 15.5 tok/s | 29.7 tok/s | **1.92×** |
| **Qwen3-8B**    | ~2.58 | 28.7 tok/s | 54.4 tok/s | **1.90×** |
| **Qwen3-4B**    | ~2.33 | 51.3 tok/s | 84.1 tok/s | **1.64×** |
| **Ternary-Bonsai-27B** (2-bit, hybrid) | ~2.80 | 25.6 tok/s | 28.5 tok/s | **1.11×** auto (up to 1.2× fresh) |

Baselines are this harness's pipelined greedy loop, which measures at parity with `mlx_lm.generate`
(the Qwen3-4B baseline is the same 51–52 tok/s either way). All paths produce **identical** output to
plain decoding — they're just faster. Chat content accepts less than code everywhere; on the 2-bit
Bonsai target that flips speculation into a net loss, which is exactly what `--max-draft auto`'s
parking handles (see the Bonsai section). Why a Mac can't go much higher and the cost model are below.
The deep-dive's multi-prompt DSpark-vs-DFlash tables are mlx-0.31.2-era and are kept as the last full
sweep — 0.32 shifted that balance toward DSpark (spot-checked; see that section's note).

## Concurrent throughput

`--max-batch N` runs up to N concurrently-queued requests through **one** batched target forward, so
they share a single weight-read per step — the regime where speculative decoding really shines. For a
local agent swarm (a few agents hitting the server at once) this is a large aggregate win, and single
requests are unaffected: a lone request — or one using penalties / logprobs / `temperature > 0` dspark —
takes the serial path, so per-request latency never regresses.

Batching is **continuous** (dspark): a request is delivered the moment it finishes — it never waits
for the batch's slowest member — and its freed slot admits the next queued or newly-arriving request
mid-flight (measured: a short request joining two long ones returned at 2.3 s while they ran to 8.4 s).
With `--max-draft auto`, the cap is also calibrated **per batch width**: at B=4 the measured verify
curve flattens past the qmm knee (the paper's cheap-verify regime), so longer draft blocks pay again
(+5% aggregate at B=4 from the auto-picked cap on an M4 Pro).

Qwen3-4B-8bit, M4 Pro, 4 concurrent requests, mlx-0.31.2-era sweep (aggregate tokens/s vs the greedy
baseline run serially; the batched-vs-serial ratios are the durable part — absolute levels are higher
on 0.32):

| serving | aggregate tok/s | vs serialized baseline |
|---|---|---|
| greedy baseline (one at a time) | 52 | 1.00× |
| **batched baseline** (`--mode baseline --max-batch 4`) | 128 | **2.48×** |
| **batched dspark** (`--mode dspark --max-batch 4`) | 130 | **2.51×** (1.73× over serialized dspark) |

Both the target verify **and** the DSpark drafter are batched. Output stays greedy-correct per request; a
batched *quantized* target is not bit-identical to single-sequence decoding (the quantized matmul takes a
different numeric path at batch width — the same qmv→qmm knee as the cost model below — flipping ~0.5% of
near-tie tokens), which is inherent to any batched quantized server, not spec-specific. Dense mlx-lm
targets (Qwen3 / Llama / Mistral-class) only; Gemma-4 (mlx-vlm) transparently falls back to serialized.

## Prefix caching

The server keeps the target KV cache (and, for DSpark, the drafter context) from the previous turn and
reuses the shared conversation prefix instead of re-prefilling it. On a ~750-token shared context this makes
follow-up turns **~13× faster** (measured: 87 ms vs 1132 ms). It's **lossless** to the same standard as the
rest of the project (a warm turn differs from a cold one only at logit-margin≈0 ties) and invalidates itself
on any error so it can't desync.

On by default for `--mode dspark` / `baseline` on **dense** targets (Qwen3); disabled for DFlash. For
**Gemma-4** (sliding-window / rotating KV cache) reuse is exact only until the window first wraps, so entries
are reused while under the window and refused once any layer wraps — multi-turn chat under the window skips
re-prefilling like Qwen does. Flags: `--no-prefix-cache`, `--prefix-cache-slots N` (LRU slots so a chat and
an agent don't evict each other, default 2), and `--prefix-cache-dir DIR` + `--prefix-cache-max-ram-mb N`
for the optional SSD spill tier on very long contexts.

---

## Benchmarks & deep dive

*Everything below is for readers who want the numbers and the why. The sections above are enough to use it.*
Reproduce the sweep on your own Mac with `mlx-dspark benchmark --model <repo>` (warm, device-stamped, `--json`).

### The Apple-Silicon speedup ceiling

Speculative decoding amortizes a *memory-bound* single-token decode across the K tokens verified in one
forward. On a datacenter GPU that arbitrage is huge (parallel verify is nearly free, so speedup ≈ acceptance
length). On an M-series chip it's weaker — **verify cost grows with the number of tokens verified**
(multi-token verify leaves the quantized matmul's cheap few-rows path). The cost model is
`tok/s ≈ A / (drafter + overhead + slope·C)` for accept length `A` and draft cap `C`; the *slope* is a
property of (quantization × mlx version × chip), which is why `--max-draft auto` measures it on your
machine instead of trusting a constant. On mlx 0.31.2 we measured ≈ +14 ms/token for Gemma-4 12B (a
~2.2× ceiling even with a perfect drafter); mlx 0.32's kernels flattened the curve enough that Gemma-4
now measures 2.11× at cap 2 — past what the old curve allowed. The binding limiter remains acceptance
length (set by the drafter↔target match) — **not** drafter quantization (4-bit / 8-bit / bf16 give
identical acceptance; 4-bit is simply fastest).

### Long context

The speculative speedup **holds with context depth** — measured flat at ~1.6× out to 12k+ tokens on
Qwen3-4B (M4 Pro, mlx 0.31.2; absolute levels are higher on 0.32 — the flatness is the point). (Before v0.3.1 the drafter tiled its GQA/MQA KV cache redundantly every round, which
scaled with depth and made speculation go *net-negative* past a few thousand tokens on cheap-verify
targets; that's fixed — the fix is bit-for-bit identical output.) On expensive-verify targets (Gemma-12B)
speculation actually *gains* slightly with depth, since the target slows faster than the cheap drafter.

Two things do still grow with a longer prompt, for **every** decoder (baseline, `mlx-lm`, this) — not the
speculative speedup: **time-to-first-token** (reading an *L*-token prompt is inherent work) and **per-token
decode** (attention reads a longer KV cache). Soften both with prefix caching (reuse the conversation prefix
across turns, on by default) and `--kv-bits 8` (quantized KV cache — the long-context bandwidth lever).

### DSpark vs DFlash (head-to-head)

Three drafters from the same DeepSpec lineage, all EAGLE-family (a tiny drafter that consumes the *target's
hidden states*): **EAGLE3** is autoregressive (high quality, draft latency grows with block size); **DFlash**
drafts a whole block in one pass (fast, but later positions collide — "suffix decay"); **DSpark** =
DFlash's parallel backbone **+ a rank-256 Markov head** that reinjects token-to-token dependency, fixing
suffix decay for ~0.6 ms/round. This is the first MLX port of DSpark; it also runs
[z-lab](https://github.com/z-lab/dflash)'s **original** DFlash (block diffusion, Chen et al.,
[arXiv:2602.06036](https://arxiv.org/abs/2602.06036), MIT) through the same lossless loop.

> **mlx-version note:** the two multi-prompt tables below are the last full sweep, measured on
> **mlx 0.31.2**. On mlx 0.32 the balance shifted toward DSpark — narrow multi-row verify got
> disproportionately cheaper, so on the same code prompt Gemma-12B now measures DSpark cap-2 at 2.11×
> vs DFlash full-16 at 1.63×, and the 12B "DFlash wins code/math" pick no longer holds on this M4 Pro
> (the 8B "full block is a net loss" verdict still does: 0.86×). The per-domain *acceptance* numbers
> below are mlx-independent and remain the useful part; re-run `mlx-dspark benchmark` for current
> throughput on your setup.

**Gemma-4 12B** (it-8bit, M4 Pro, warm, greedy, 4 prompts/domain — accept / tok·s; greedy ≈ 17.3 tok/s):

| method | chat | code | math |
|---|---|---|---|
| **DSpark** (cap 2) | **2.45 / 28.5** | 2.78 / 32.8 | 2.86 / 32.4 |
| DFlash (cap 2) | 2.15 / 24.2 | 2.76 / 31.3 | 2.71 / 29.6 |
| DFlash (full 16) | 2.68 / 16.9 | **5.95 / 36.6** | **6.20 / 36.3** |

They're **complementary**, matching the paper's framing: DFlash's block-16 wins **structured** content on
both axes (accept ~6.0 on code/math vs DSpark's block-7 ceiling ~2.8; ~2.1× throughput) because high
acceptance amortizes the wide verify; DSpark's Markov head wins **open chat** (2.45 / 1.65×; DFlash's block
never fills on unpredictable text — full-16 chat is a slight net *loss*).

**But the winner flips on a smaller, cheap-verify target.** Qwen3-8B-8bit (warm, greedy, 3 prompts/domain;
greedy ≈ 28.8):

| method | chat | code | math |
|---|---|---|---|
| **DSpark** (cap 2) | **2.38 / 45.7** | **2.55 / 48.8** | **2.40 / 46.1** |
| DFlash (cap 2) | 1.99 / 33.8 | 2.22 / 37.0 | 2.11 / 35.7 |
| DFlash (full 16) | 2.19 / 21.1 | 2.94 / 27.6 | 2.66 / 25.5 |

Here **DSpark wins everywhere (~1.6×)** and DFlash's block advantage evaporates — full-16 is a net *loss*
(~0.9×) because the cheap verify makes the wide block cost more than it returns, and accept never climbs
(~2.9 on code vs 5.95 on the 12B). Cross-checked against z-lab's own optimized runner
[`dflash-mlx`](https://github.com/bstnxbt/dflash-mlx) on the *identical* target+drafter: its baseline matches
ours (29.3 tok/s) and its DFlash is *also* a net loss / wash at 8B (0.92× code full-block, ~1.08× adaptive) —
*even with its hand-written Metal verify kernels*. So this is DFlash at this model scale on Apple Silicon,
not an artifact of our verify loop.

Per the paper (accept length, full block, temp=1.0), DSpark beats DeepSpec's DFlash by **+16–18%** and EAGLE3
by **+27–31%**; our greedy exact-match numbers are lower than the paper's temp=1.0 speculative-sampling
numbers because greedy is the strictest possible accept rule (not a bug).

### Target precision

Since verify dominates, target precision is a speed/quality knob (mlx-0.31.2-era sweep — the 8-bit
column is higher on 0.32, see [Results at a glance](#results-at-a-glance); the qualitative trade-off
is unchanged):

| target | 8-bit (default) | 4-bit |
|---|---|---|
| Gemma-4 12B | greedy 17.5 → spec 30 tok/s (**1.73×**) | greedy 30.6 → spec 34–38 tok/s (1.1–1.25×) |
| Qwen3-4B    | greedy 49.8 → spec 73 tok/s (**1.45×**) | greedy 82 → spec 96–103 tok/s (1.17–1.26×) |

**8-bit** for the biggest spec benefit + best quality; **4-bit** for max absolute throughput or small RAM
(`--model …-it-4bit`). The drafter stays 4-bit; a bf16 target is *not* a win (verify roughly doubles).

### Tuning

- **DSpark** — `--max-draft 2` is the measured optimum for the dense presets (default): verify cost grows
  per token and the marginal draft token rarely survives. `--confidence-threshold 0.6` truncates the block
  adaptively via the confidence head instead. For **Bonsai-27B** use `--max-draft auto` (see its section).
- **`--max-draft auto`** — measures this machine + model's verify/drafter cost curves once (a few seconds,
  cached on disk) and picks the cap per round from the curves + live acceptance **and observed round
  times**, so it tracks the hardware and the mlx version instead of a hard-coded `cap=2`. It can also
  **park** speculation entirely (plain pipelined steps + periodic probe rounds) on content where
  speculation would lose — the safety net that makes it the recommended setting for Bonsai. Lossless —
  the cap only sets how many drafts get verified.
- **Hybrid n-gram drafting** (dspark, on by default) — when the current suffix already occurred earlier in the
  context (quoting, code edits, repeats), that free continuation is verified instead of running the drafter
  that round, so copy-heavy spans commit several tokens per round. Composes losslessly; `--no-lookup-drafts`
  turns it off. `--mode lookup` runs the same n-gram speculation with **no drafter at all**, for any target.
- **DFlash** — `--max-draft 0` (full 16-block) is its native point and reaches ~6 accepted tokens on
  code/math; on current mlx that still measures below DSpark cap-2 on this M4 Pro (see the pick table),
  so treat DFlash as the head-to-head benchmark option rather than the speed pick. Short caps on open
  chat; the full block never fills there.
- **Sampling** — `--temperature > 0` (+ `--top-p` / `--top-k`) is lossless w.r.t. the target at temperature T
  (the paper's §2.1 method). On M-series it's ≈ greedy speed (the extra acceptance lives in a tail a short
  cap never reaches) — it's a *sampled-output* feature, not a speed lever.

## License

MIT — see [`LICENSE`](LICENSE). An independent MLX port of the inference path of DeepSeek's DSpark drafter;
the z-lab DFlash drafter classes are vendored (MIT) with attribution in [`NOTICE`](NOTICE). No model weights
are bundled.
