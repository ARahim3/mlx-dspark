<p align="center">
  <img src="https://raw.githubusercontent.com/ARahim3/mlx-dspark/main/mlx-dspark.png" alt="mlx-dspark" width="440">
</p>

<p align="center">
  <b>DeepSeek's DSpark <i>and</i> z-lab's DFlash speculative decoding — native on Apple Silicon via <a href="https://github.com/ml-explore/mlx">MLX</a>.</b>
  <br>Lossless drafters (same output, just faster) for <b>Gemma-4, Qwen3, Ornith-1.0, Qwen3.6-27B, and Bonsai-27B</b> targets —
  <br>plus any matched DSpark / DFlash checkpoint. Run them at the CLI, from Python, serve an <b>OpenAI-compatible API</b> to LM Studio / any local tool,
  <br>or drive <b>Claude Code</b> with a model on your own Mac.
</p>

<p align="center">
  <a href="https://pypi.org/project/mlx-dspark/"><img src="https://img.shields.io/pypi/v/mlx-dspark?color=2563eb" alt="PyPI"></a>
  <img src="https://img.shields.io/pypi/pyversions/mlx-dspark" alt="Python">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-111111?logo=apple&logoColor=white" alt="Apple Silicon">
  <a href="https://github.com/ARahim3/mlx-dspark/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/ARahim3/mlx-dspark/main/docs/demo.gif" alt="Baseline vs DSpark — same output, ~2.1x faster on Gemma-4 12B" width="840">
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

## Supported models

Every row auto-resolves its drafter from `--model` (any quant of the target matches). Measured warm on
an M4 Pro with `mlx-dspark benchmark --trials 3` (median of 3, three prompts — chat/code/math); full
tables, baselines, and method in [Results at a glance](#results-at-a-glance):

| target | best measured speedup | speed |
|---|---|---|
| **Gemma-4 12B** (8-bit) | **3.09×** math · **2.63×** chat · **2.61×** code | ~49 tok/s |
| **Ornith-1.0-9B** (8-bit) | **2.53×** code · **2.48×** math · **2.21×** chat | ~64 tok/s |
| **Qwen3-14B** (8-bit) | **2.36×** math · **2.11×** code · 1.62× chat | ~31 tok/s |
| **Qwen3-8B** (8-bit) | **2.29×** math · **2.06×** code · 1.81× chat | ~58 tok/s |
| **Qwen3-4B** (8-bit) | **1.98×** math · 1.77× chat · 1.70× code | ~92 tok/s |
| **Qwen3.6-27B** (4-bit)\*\* | **1.78×** math · **1.42×** code | ~22 tok/s |
| **Ternary-Bonsai-27B** (2-bit) | **1.13×** code | ~27 tok/s |

<sub>\*\* Qwen3.6-27B works and is lossless, but it's not a speed pick yet: the only drafter
published for it so far is a community checkpoint with modest acceptance — a better-qualified
drafter would lift this row. Its numbers are the one row **not** re-measured in the 2026-07-22
sweep (cap-2 era, so likely understated — see [Results at a glance](#results-at-a-glance)).</sub>

<sub>This table is the set of pairs we have **measured and vouch for**, which is also exactly the
auto-resolve registry — that is the only thing the registry is for. It is *not* the set of models
that work: any DeepSpec-native drafter runs against any compatible target via `--drafter`, and any
target at all gets drafter-free speculation via `--mode auto`. See
[Bring your own drafter](#bring-your-own-drafter--what-runs-and-what-doesnt).</sub>

<sub>Target precision: the quants shown are each model's measured best — ratios are
*non-monotone* in bits and peak at **8-bit** on current MLX (full Ornith sweep: 4-bit 1.38× ·
8-bit 2.17× · bf16 1.54× on code; bf16 loses in both ratio *and* absolute speed because MLX's
unquantized matmul pays a ~2× cost cliff at verify width 2). Details in
[Results at a glance](#results-at-a-glance).</sub>

**Copy-heavy code editing goes further:** when the model re-emits or refactors code already in its
context (the daily agent/assistant workload), match-scaled lookup drafts reach **4.5× on Gemma-12B**
(75 tok/s) and **3.6× on Ornith-9B** (93 tok/s). Any model *not* listed still gets drafter-free
lookup speculation via `--mode auto`.

## Install

```bash
pip install mlx-dspark          # or:  uv pip install mlx-dspark
```

Apple Silicon + Python ≥ 3.10; installs mlx ≥ 0.32.0 automatically (0.32's quantized-matmul kernels are
what current speedup numbers are measured on). Model weights download from the Hugging Face cache on
first use (none bundled). No server framework is pulled in — the API server is built on the standard
library.

> **Known upstream incompatibilities (both handled):** mlx-vlm **0.6.5** moved an internal
> rope-utils module, which crashed `import mlx_dspark` on fresh installs of mlx-dspark ≤ 0.4.2 —
> fixed in **0.4.3** (both module layouts supported), so upgrade mlx-dspark rather than pinning
> mlx-vlm. Separately, mlx-vlm **0.6.4** × transformers **≥ 5.12** breaks loading the gemma4
> target with a misleading `OSError: Can't load video processor …`
> ([#4](https://github.com/ARahim3/mlx-dspark/issues/4), upstream
> [Blaizzy/mlx-vlm#1578](https://github.com/Blaizzy/mlx-vlm/issues/1578) — fixed in mlx-vlm
> 0.6.5). mlx-dspark ≥ 0.3.2 shims that one at load time, so any mlx-vlm ≥ 0.6.3 works; the
> shim self-retires on fixed releases, and `mlx-dspark doctor` reports when it is active.

## Quickstart

You name the **target model** (`--model`, an HF repo or local path, exactly like `mlx-lm`); the matching
drafter is resolved automatically for known targets (see [Models](#models)), or pass `--drafter`.

### Serve an API (OpenAI **and** Anthropic on one port)

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

### Use it from Claude Code

The same server also speaks Anthropic's **Messages API**, which is the dialect
[Claude Code](https://claude.com/claude-code) talks — so Claude Code can run entirely on a model
on your Mac. Start the server, then in another terminal:

```bash
mlx-dspark serve --model mlx-community/Qwen3-8B-8bit --no-thinking   # terminal 1
mlx-dspark claude                                                    # terminal 2
```

That's it — `mlx-dspark claude` finds the running server, points Claude Code at it, and hands over
the terminal. Anything after `--` goes to `claude` (`mlx-dspark claude -- --continue`).

**It changes nothing outside that one process.** The configuration is passed as the launched
process's environment and nowhere else: no shell profile edited, no `settings.json` written, no login
replaced. Your other Claude Code sessions — open now or started later — keep their normal account,
model, and endpoint, and this one reverts the moment it exits. Your claude.ai login stays saved and
untouched; Claude Code notes at startup that a credential variable takes precedence over it, which is
just that notice. To wire it up yourself instead, `mlx-dspark claude --print-env` prints the shell
exports and `--print-settings` prints a project-scoped `.claude/settings.local.json` block.

Endpoints: `POST /v1/messages` (streaming and non-streaming), `POST /v1/messages/count_tokens`.
Tool calling, multi-turn `tool_use`/`tool_result` history, `stop_sequences`, and system prompts are
translated to whatever the loaded model's own chat template expects — including each family's tool
syntax (Hermes JSON, Gemma-4, and the XML `<function=>` form) — and a reasoning model's `<think>` or
`<|channel>thought` output is lifted into proper Anthropic `thinking` blocks rather than leaking as
prose.

**Measured** — each of these ran a real Claude Code session that read a buggy file and fixed it with
the `Edit` tool (M4 Pro, `--no-thinking`, identical task):

| target | accept length | prefix cache | wall clock |
|---|---|---|---|
| `mlx-community/Qwen3-8B-8bit` | 3.01 | on — 2 of 3 requests, ~26k tokens reused | **~2:20** |
| `mlx-community/Ornith-1.0-9B-8bit` | **5.07** | off — hybrid target, recurrent state can't be reused | ~4:10 |
| `mlx-community/gemma-4-12B-it-8bit` | 3.68 | on, but its sliding window wraps at this prompt size | ~4:10 |

The ranking is the point: **Claude Code sends ~18–26k tokens of system prompt and tool schemas on
every request**, so prefill dominates wall clock and the target that *reuses* it wins — Qwen3-8B is
nearly 2× faster here despite the lowest accept length of the three. Ornith's 5.07 is the highest
acceptance this project has measured anywhere (tool-call JSON is very predictable), but it spends
the win on re-prefilling. Choose for prefix-cache compatibility first, drafter quality second.

### Other agent clients

The Anthropic endpoint isn't Claude Code–specific. [**pi**](https://github.com/earendil-works/pi-mono)
works out of the box against either API — add a custom provider to `~/.pi/agent/models.json`:

```json
{ "providers": { "mlx-dspark": {
    "baseUrl": "http://127.0.0.1:8080", "api": "anthropic-messages", "apiKey": "mlx-dspark",
    "models": [{ "id": "Qwen3-8B-8bit", "contextWindow": 40960, "maxTokens": 8192 }] } } }
```

then `pi --provider mlx-dspark --model Qwen3-8B-8bit`. (Swap `"api"` for `"openai-completions"`
and `"baseUrl"` for `http://127.0.0.1:8080/v1` to use the OpenAI endpoint instead — both work.)

**pi is markedly better suited to a local model than Claude Code**, for one reason: its system
prompt is **~1.5k tokens against Claude Code's ~18–26k**. Since prefill dominates, that lands
directly on the clock — the same one-bug fix takes **~6 s** instead of ~2:20, and a four-tool
task (read, two edits, read, write) finishes in **8.5 s at 24 tok/s** on Qwen3-8B. Ornith-1.0-9B
runs the same task in 18.8 s. Gemma-4-12B doesn't converge on pi's tool protocol (on either
endpoint, so it's the model, not the server) — use it with Claude Code instead.

Practical notes for a local model:

| | |
|---|---|
| **Use a tool-calling model** | These are tool-use agents first. Qwen3-8B and up handle it; smaller models flail. |
| **Agent choice moves the clock more than model choice** | The client's prompt size is the dominant cost on a local model — a lean agent like pi is an order of magnitude faster on the same hardware and the same task. |
| **`--no-thinking` is a speed knob, not a requirement** | Leaving it off works fine — reasoning is streamed as proper `thinking` blocks either way. It just costs: on Qwen3-8B the same Claude Code task ran 3:17 and 2762 output tokens with thinking vs ~2:20 and 169 without, since the model thinks before *every* tool call. A client sending `thinking: {"type": "disabled"}` gets the same effect per-request. Note it's a no-op on Gemma-4, whose template doesn't think by default. |
| **Leave prefix caching on** | It is doing most of the work (see the table). Expect the first request of a session to be the slow one regardless. |
| **Context** | An over-long request is refused with the wording Claude Code recognises as a context limit, so it compacts and retries instead of dying. `--context-window N` lowers the bar deliberately (e.g. to keep the KV cache inside your RAM budget). |

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
| `mlx-community/Qwen3.6-27B-4bit`     | `Avesed/Qwen3.6-27B-DSpark` (community) | — | ~20 GB |
| `mlx-community/Ornith-1.0-9B-8bit`   | `stanleyphoong/Ornith-1.0-9B-DSpark` (community) | — | ~13 GB |

*Peak RAM* is measured on an M4 Pro (8-bit target + 4-bit drafter + KV cache); add headroom for macOS.
Rows marked *(community)* use drafters published by the community, not by DeepSeek — quality varies more
than with the official checkpoints, and it shows up directly as acceptance length (= your speedup).
The **Ornith** drafter is the strong case: rigorously qualified by its author (17/17 gates, 95% of the
DSpark paper's reference acceptance) and it produces the best chat speedups in this table. The
**Qwen3.6-27B** drafter is solid but accepts below DeepSeek's official drafters (~2.1 vs ~2.6–2.8 at
cap 2 on code — hence the smaller speedup), is English-centric (Chinese accepts poorly, per its own
card), and was trained against a W4A16 quant — so pair it with the **4-bit** target: that's its
*matched* precision, and we measured the 8-bit target *lowering* its acceptance (any `Qwen3.6-27B-*`
quant still resolves the drafter if you want to try; see the results footnote).
A 4-bit target (`--model …-it-4bit`) roughly halves the target's share (fits smaller Macs). **Use the
matched *instruct* target** the drafter was trained against — a base model drops acceptance sharply. The
legacy `--family qwen3|gemma4` flags still work but are deprecated in favor of `--model`.

`--drafter` lets you run **any** other matched z-lab / DeepSpec checkpoint with no code change and no
registry entry — the registry only saves you from having to name the drafter:

```bash
# a pair we have measured -> the drafter auto-resolves
mlx-dspark generate --model mlx-community/Qwen3-14B-8bit --prompt "Explain how rainbows form."

# anything else -> name the drafter yourself; identical machinery from here on
mlx-dspark generate --model mlx-community/Qwen3-32B-8bit \
  --drafter deepseek-ai/dspark_qwen3_32b_block7 --prompt "Explain how rainbows form."
```

### Bring your own drafter — what runs and what doesn't

New DSpark/DFlash drafters keep landing on HF in **three different packagings**; here is the honest
compatibility contract (loaders refuse incompatible checkpoints with an error naming the reason, never
a silent mis-load):

| checkpoint style | example | status |
|---|---|---|
| **DeepSpec-native standalone drafter** (qwen3/gemma4 backbone, any size/quant) | `deepseek-ai/dspark_qwen3_32b_block7` | ✅ runs via `--drafter` — no registry entry needed (4B/8B/14B/gemma-12B are measured *and* registered, so they need no flag; larger sizes should run — reports welcome) |
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

Measured on an M4 Pro 48 GB (greedy, warm): baseline ~25.5 tok/s; **~1.15× on code/structured
content** (acceptance ~2.9/round at cap 2), ~break-even on open chat. Output is lossless —
byte-identical to plain greedy decoding. Bonsai's backbone is **hybrid linear attention**
(48 of 64 layers carry recurrent state, which can't be rolled back like a KV cache), so
mlx-dspark records each verify round's recurrence inputs and, on a partial accept, rebuilds
the state at the exact accept point (bit-exact, a few ms) — rejected drafts cost about as
little as a dense KV trim. As far as we know this is the first working speculative decoding
for this model family on Apple Silicon.

Two honest caveats. The ceiling is the 2-bit quantization itself: extra verify rows on a
2-bit model are compute-bound (they cost the same as on a 4-bit model, measured), while its
plain step is very fast — so chat-level acceptance hovers at break-even instead of the
1.6–2.1× the 8-bit presets reach. **`--max-draft auto`** stays the recommended setting: it
picks the cap from this machine's measured curves + live acceptance and can still *park*
speculation (plain pipelined steps + probe rounds) on content where it would lose. And it
requires **mlx ≥ 0.32.0** (older mlx lacks the multi-row 2-bit matmul path that makes
verification affordable at all). Prefix caching and batching remain dense-target-only.
Baseline/`--mode lookup` also work for any other `qwen3_5` (Qwen3.5/3.6-family) checkpoint.

The **1-bit** `Bonsai-27B-mlx-1bit` pack runs on stock MLX as of mlx-vlm **0.6.5** (which
ships a Python-hosted 1-bit kernel; stock `mx.quantize` still has no 1-bit mode) — but
speculative decoding measures a net **loss** on it: that kernel re-reads the full weight
stream once per verified token, so verify cost is linear in draft length, and dspark lands at
0.71–0.77× baseline at every cap (M4 Pro, healthy acceptance, losslessness intact). mlx-dspark
therefore keeps the pack unintegrated — plain generation via mlx-vlm ≥ 0.6.5 is the right tool
for it (~35 tok/s on an M4 Pro vs ~25 for the ternary), and `load_target` refuses it with a
pointer saying so. The ternary 2-bit variant remains the speculative-decoding operating point.

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

| target | DSpark (`--mode dspark`, measured cap) | DFlash (`--mode dflash --max-draft 0`) | pick |
|---|---|---|---|
| **Gemma-4 12B** | **2.61×** code, 2.63× chat *(cap 4)* | 1.63× code, ~0.7× chat | **DSpark** |
| **Qwen3-8B** | **2.06×** code *(cap 4)* | 0.86× full block (1.47× at cap 2) | **DSpark** |
| **Qwen3-4B** | **1.70×** code *(cap 4)* | modest | **DSpark** |
| **Ternary-Bonsai-27B** | **1.13×** code *(cap 2)* | — | **DSpark** |
| **Qwen3.6-27B** (4-bit, hybrid) | **1.42×** code · **1.78×** math · 1.27× chat | — | **DSpark** (auto) |
| **Ornith-1.0-9B** (8-bit, hybrid) | **2.53×** code · **2.48×** math · **2.21×** chat *(cap 4)* | — | **DSpark** |

<sub>DSpark column regenerated 2026-07-22 at each model's measured cap; the DFlash column is the
older cap-2-era sweep and was **not** re-measured, so the real DSpark margin is now wider than
the rows suggest — the verdict does not change.</sub>

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

**DSpark** vs plain greedy decoding of the same model, each at **its own measured cap** (M4 Pro 48 GB,
warm, 8-bit instruct target, 4-bit drafter, **mlx 0.32.0**). Regenerated 2026-07-22 with
`mlx-dspark benchmark --trials 3`: every number is a median of 3 runs over the harness's three
prompts, and the tok/s columns are the mean across them. Reproduce any row with that command.

The cap column is the headline change. mlx 0.32's quantized-matmul kernels widened the cheap
verify region to width 5 for 8-bit weights, moving the knee from 4 to 6 — so the old hard-coded
`cap=2` was leaving **10–35%** on the table for every 8-bit target. The cap is now derived from
each machine+model+quant's measured cost curves rather than hard-coded (see [Tuning](#tuning)),
which is why Bonsai still sits at 2 while the 8-bit rows moved to 4.

| target | cap | accept len | baseline | mlx-dspark | speedup | chat / code / math |
|---|---|---|---|---|---|---|
| **Gemma-4 12B** | 4 | 3.95 | 17.8 tok/s | 49.4 tok/s | **2.78×** | 2.63× / 2.61× / 3.09× |
| **Ornith-1.0-9B** (hybrid)² | 4 | 3.64 | 26.7 tok/s | 64.2 tok/s | **2.40×** | 2.21× / 2.53× / 2.48× |
| **Qwen3-8B** | 4 | 2.94 | 28.1 tok/s | 57.7 tok/s | **2.05×** | 1.81× / 2.06× / 2.29× |
| **Qwen3-14B**³ | 4 | 2.87 | 15.3 tok/s | 31.0 tok/s | **2.03×** | 1.62× / 2.11× / 2.36× |
| **Qwen3-4B** | 4 | 2.79 | 50.9 tok/s | 92.4 tok/s | **1.82×** | 1.77× / 1.70× / 1.98× |
| **Qwen3.6-27B** (4-bit, hybrid)²⁴ | 2 | ~2.12 | 15.2 tok/s | 21.6 tok/s | **1.42×** | — / 1.42× / 1.78× (cap 3) |
| **Ternary-Bonsai-27B** (2-bit, hybrid) | 2 | 2.60 | 25.4 tok/s | 27.2 tok/s | **1.07×** | 1.01× / 1.13× / 1.07× |

<sub>³ Qwen3-14B is not in the auto-resolve registry — pass
`--drafter deepseek-ai/dspark_qwen3_14b_block7`. ⁴ Qwen3.6-27B is the one row not re-measured in
this sweep (the 4-bit target was not on the machine); its numbers are cap-2 era and likely
understated.</sub>

Every 8-bit row peaks at **cap 4** and falls off sharply at cap 5 — the cliff sits exactly where
the measured verify curve leaves its cheap region (width 5 → 6). Bonsai is the counter-example
that shows why the cap is not a constant: its 2-bit verify cost climbs from width 2, so it peaks
at cap 2 (cap 1 = 1.00×, cap 3 = 1.06×) and there is no wide-draft regime to reach.

² Community-drafter rows. Qwen3.6-27B runs a **4-bit** target (its drafter's matched
precision — trained against a W4A16 quant); code at cap 2 shown, `--max-draft auto` settles at
cap 3; unlike 2-bit Bonsai, chat stays positive (1.27×). An 8-bit target was measured and is
NOT recommended here: acceptance *drops* (the drafter is 4-bit-native), and while `--max-draft
auto` still reaches ~2.1× against the slower 8-bit baseline, the 4-bit target is faster in
absolute tok/s everywhere. Rule of thumb: **match the target's precision to what the drafter
was trained against** — Ornith's drafter (bf16-qualified) wants 8-bit, Avesed's (W4A16) wants 4-bit. Ornith-1.0-9B (an agentic-coding
qwen3_5 hybrid, drafter qualified against the bf16 verifier) runs the **8-bit** house
sweet spot — the first target here with chat above 2× — and its acceptance is so high on code
(p≈0.96/position) that auto-cap drives the cap to the full block of 7. The **4-bit** Ornith
target trades the ratio for absolute speed: ~1.4–1.55× but 60–76 tok/s (baseline 49.3) —
pick 4-bit for peak tok/s, 8-bit for quality and the headline ratio; the same drafter
auto-resolves for both. And don't bother with a **bf16** target for speculation: we swept it
(Ornith bf16: 1.54× code at cap 3, 22.9 tok/s) — the ratio is *non-monotone* in bits and
peaks at 8-bit, because MLX's unquantized matmul pays a ~2× cost cliff at verify width 2
where the quantized kernels stay flat. bf16 is slower than 8-bit in both ratio *and*
absolute speed here.

Baselines are this harness's pipelined greedy loop, which measures at parity with `mlx_lm.generate`
(the Qwen3-4B baseline is the same 51–52 tok/s either way). All paths produce **identical** output to
plain decoding — they're just faster. Chat content accepts less than code everywhere; on the 2-bit
Bonsai target chat lands ~break-even (its verify rows are compute-bound), and `--max-draft auto`
adapts or parks where speculation would lose (see the Bonsai section). Why a Mac can't go much
higher and the cost model are below.

The table is *fresh-generation* content. On **copy-heavy editing** — the model re-emitting or
refactoring code already in its context — match-scaled lookup drafts (0.5.0, on by default) go well
past it: Gemma-12B file re-emission 3.03× → **4.51×** (75 tok/s), rename-refactor 4.33×; Ornith-9B
rename-refactor 2.79× → **3.57×** (93 tok/s), re-emission 2.45× — outputs still bit-identical, chat
and fresh code unchanged. See the hybrid-drafting bullet in
[Flags that matter](#benchmarks--deep-dive) for how it works.
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

- **DSpark** — the default cap is **measured for your machine, model and quantization**, not hard-coded:
  with no `--max-draft`, mlx-dspark benchmarks this pair's verify/drafter cost curves once (~5 s, cached
  on disk) and picks the best fixed cap. It has to be measured, because the answer moves a lot — on one
  M4 Pro under mlx 0.32 the optimum spans cap 2 to 7, and the *same model* wants cap 2 at 4-bit, 4 at
  8-bit and 6 at bf16. Pass `--max-draft N` to pin it. `--confidence-threshold 0.6` truncates the block
  adaptively via the confidence head instead. For **Bonsai-27B** use `--max-draft auto` (see its section).
- **`--wired-limit`** — off by default, and you almost certainly want to leave it that way. It raises MLX's
  wired-memory ceiling to the recommended working set (~75% of RAM) so weights can't be paged out. Wired
  pages can't be reclaimed by the OS, so on a machine already holding a large working set this can **hang
  macOS hard enough to need a power cycle** — and a 16 GB Mac, where "the model nearly fills RAM" is exactly
  the situation it was meant to help, is the most likely to wedge. It has also corrupted the verify logits on
  the gemma-4/mlx-vlm route (garbage logits can commit *wrong tokens*, not just crash); mlx-lm targets didn't
  reproduce that. It bought no measurable speed where tested (<1%, inside run-to-run noise). Reach for it only
  if you actually see paging stalls, and validate a long run before trusting the output.
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
  **Match-scaled long drafts** (`--lookup-long-draft`, default 32): a copy run whose context matches ≥8
  tokens deep earns drafts up to ~2× the matched length — verify width 16–32 is a measured plateau on
  M-series (~2.5× the cost of one step), so verbatim spans commit ~20–30 tokens per forward. Measured
  (8-bit, M4 Pro, outputs bit-identical): gemma-12B file re-emission **3.0×→4.5×** (75 tok/s), Ornith-9B
  rename-edit **2.8×→3.6×** (93 tok/s); chat unchanged. An acceptance gate parks the scaling on
  insertion-heavy edits (measured neutral there).
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
