# Changelog

All notable changes to `mlx-dspark`. Versions follow [SemVer](https://semver.org/) (pre-1.0: minor-ish features land as patch bumps).

## [Unreleased]

## [0.3.2] — 2026-07-14 — gemma4 loads on fresh installs (mlx-vlm 0.6.4 compat shim)

### Fixed
- **gemma4 target loads again on fresh installs (mlx-vlm 0.6.4 × transformers ≥ 5.12) (#3, #4).**
  mlx-vlm 0.6.4 changed its `Gemma4Processor` to hand `video_processor` through to transformers'
  `ProcessorMixin`, but left the `Gemma4UnifiedProcessor` subclass taking it via `**kwargs` — and
  transformers ≥ 5.12 validates processor kwargs against the **literal `__init__` signature**
  (`ProcessorMixin.get_attributes`), so loading the gemma4 preset raised
  `TypeError: Unexpected keyword argument video_processor`. mlx-vlm's AutoProcessor patch then
  swallowed that and fell back to transformers' own (checkpoint-incompatible) processor, which
  surfaced as an unrelated `OSError: Can't load video processor …` — the confusing traceback users
  actually saw. Root cause + fix are upstream ([Blaizzy/mlx-vlm#1578](https://github.com/Blaizzy/mlx-vlm/issues/1578),
  landed on main, unreleased as of 0.6.4); until that ships, `load_target` applies a signature shim
  that patches **only the broken 0.6.4 shape** (0.6.3 — which never passes the kwarg through — and
  fixed releases are detected and left untouched). Verified end-to-end in a fresh
  mlx-vlm 0.6.4 + transformers 5.12.1 venv: `mlx-dspark generate --model gemma4` fails before,
  generates after, and 0.6.3 behavior is byte-identical (shim no-ops). Thanks @jnyer27 for the
  upstream root-cause analysis.
- The `load_target` error message now names this known failure (with the upstream issue and the
  pin workaround) when the masked `video processor` error is detected, instead of relaying the
  red-herring `OSError` alone; `mlx-dspark doctor` reports when the shim is active.

### Added
- `_shim_gemma4_unified_processor` shape tests (broken/0.6.3/fixed/idempotent + installed-version
  consistency) in `tests/test_import_compat.py`. 141 model-free tests, ruff-clean.

## [0.3.1] — 2026-07-08 — long-context drafting fix + OpenAI structured-content messages

### Fixed
- **Speculative speedup no longer collapses at long context (cheap-verify targets).** The DSpark
  drafter's cross-attention tiled its GQA/MQA K/V up to full heads (`_repeat_kv`, n_rep 4× on
  Qwen, 16× on Gemma) across the **whole** context cache every round — `mx.fast.scaled_dot_product_attention`
  already does that broadcast internally, so it was O(n_rep · ctx_len) of pure wasted memory
  traffic that grew with depth. On cheap-verify targets (Qwen-class), where the drafter is the
  dominant share of each round, it made long-context drafting go **net-negative past a few
  thousand tokens** (measured Qwen3-4B-8bit, M4 Pro: decode speedup **0.62× at 8k, 0.57× at
  12k** — spec *slower* than baseline, while accept length stayed a healthy ~2.7). Passing the
  n_kv-head K/V straight to SDPA is **bit-for-bit identical** (same math, no redundant tiling)
  and holds the speedup **flat at ~1.6× out to 12k+** (8k 0.62×→**1.65×**, 12k 0.57×→**1.58×**).
  Validated lossless on every path: single-sequence (Qwen + Gemma, ids identical to the old code
  at 256 and 6 k ctx), batched B=4 (per-row ids identical), and the batch suite. Expensive-verify
  targets (Gemma-12B) never collapsed — the drafter is a small fraction there — so the change is
  neutral-but-harmless for them (measured 8k: 1.39× before and after). Time-to-first-token still
  grows with prompt length (that is the cost of reading the prompt, identical for baseline and
  every framework — not this bug).
- **OpenAI structured-content messages (`content` as a list of parts) no longer 400.** Coding
  agents and OpenAI SDKs commonly send `content: [{"type": "text", "text": "…"}]` instead of a
  plain string; that list reached the chat template unchanged and blew up inside it
  (`could not apply chat template: 'list object' has no attribute 'startswith'`).
  `normalize_tool_messages` now flattens the text parts to a string before templating (non-text
  parts — images/audio — are dropped; the served text models can't consume them). A plain-string
  `content` is byte-identical to before.

### Added
- `tests/test_model.py` — model-free guard that the drafter attention's native-GQA path is
  numerically identical to explicit K/V tiling (so the tiling waste can't silently return);
  `test_tools.py` gains structured-content cases. 136 model-free tests, ruff-clean.

## [0.3.0] — 2026-07-07 — dynamic batch admission, per-batch-width calibration, KV-cache quantization, checkpoint-format robustness

### Added
- **Dynamic batch admission (continuous batching M4).** `serve --max-batch N` greedy dspark
  requests now run in a **slot session** (`batch_engine.SpecSlots`): a finished request is
  delivered the instant its row retires (it no longer waits for the batch's slowest row), and the
  freed slot admits the next queued/arriving request **mid-flight**. The batch dimension is never
  resized — retirement compacts active rows to a contiguous prefix (one row-copy) and every forward
  runs at the active width, so a lone tail request verifies at serial width (B_act=1 is the
  bit-exact single-stream numeric path). Measured (Qwen3-4B-8bit, M4 Pro): a short request arriving
  1.5 s into a 2-long-request session returned at **2.3 s wall vs 8.4 s** for the long rows.
  Validated: B=1 bit-exact vs single-seq; identical prompts stay identical through staggered
  retirements; a survivor row after the batch narrows is bit-exact vs serial.
- **(B, cap) calibration grid.** `--max-draft auto` + `--max-batch N` now also measures the
  *batched* verify curves (`calibrate.measure_batch_verify_grid`); `CapController.cap_for(B)`
  picks a per-batch-width cap. Measured (Qwen3-4B-8bit, M4 Pro): at B=4 the verify curve is ~flat
  from width 2 (8+ rows are already past the qmm knee — the paper's cheap-verify regime, measured)
  so the controller picks cap 5 → **134.1 vs 128.0 tok/s aggregate (+5%)** over the single-stream
  optimum cap 2, interleaved A/B.
- **KV-cache quantization (`--kv-bits 4|8`, generate + serve).** Quantized target KV from token 0
  (mlx-lm `QuantizedKVCache`) — cuts the KV share of the per-token bandwidth bill on long
  contexts. Spec rollback trims and prefix caching work unchanged (the cache trims by pure offset
  arithmetic). Validated: kv8 spec output == kv8 baseline byte-identical; ~70 tok/s dspark on
  Qwen3-4B (no short-context regression). mlx-lm text targets only; disables `--max-batch`
  (batched path falls back to serial automatically).
- **`n` > 1 (chat + completions, non-stream).** Greedy: one generation serves all n identical
  choices; sampled: n concurrent submissions (a `BatchEngine` batches them into one weight-read).
  `n` with `stream=true` returns 400.
- **CI** (`.github/workflows/ci.yml`): model-free test suite + ruff on every push/PR, plus a
  weekly fresh-install canary (`scripts/smoke_install.sh --tests`) that catches transitive-dep
  drift (the transformers-5.13 class of breakage) before users do.

### Changed — robustness at the checkpoint-format boundary
- **Loud errors instead of silent mis-parses.** `DSparkConfig.from_json` now detects and refuses,
  with the real reason: vLLM **speculators**-format drafters (`RedHatAI/*-speculator.dspark` —
  note their `model_type` says "qwen3" too), **embedded-drafter full models**
  (`DeepSeek-V4-*-DSpark`), unknown drafter families (previously fell through to the gemma4
  branch and died with a bare KeyError), and configs missing required DeepSpec fields.
  `load_dflash` refuses DFlash+Markov community hybrids with the reason.
- **Strict-by-default drafter loading.** A tensor-name mismatch now raises (a partially-loaded
  drafter "works" with near-zero acceptance — worse than an error); `load_drafter(...,
  strict=False)` restores warn-and-load.
- **Generalized target routing.** `load_target` routes by capability, not name: multimodal
  configs (`vision_config`/`audio_config`) → mlx-vlm; any `model_type` this mlx-lm ships a module
  for (qwen3, llama, glm_moe_dsa, deepseek_v3, …, incl. mlx-lm's remap table) → mlx-lm; else
  mlx-vlm with a helpful error. Drafter modes run a one-time **tap fidelity probe**
  (`Target.verify_tap`): the replicated forward must reproduce the model's own logits on a tiny
  input, and windowed/alternating-attention families are refused structurally — a family the
  generic tap can't serve fails loudly instead of silently drafting from a wrong stream.

### Fixed
- **`BatchEngine` wedged the process at exit** (Ctrl-C'd server, scripts, tests): the scheduler
  loop occupied the one MLX executor thread forever and `concurrent.futures`' shutdown hook joins
  it. A stop sentinel + atexit-registered `close()` unblocks it (regular atexit handlers run
  before the thread join).
- Prefix caching now also accepts `QuantizedKVCache` targets (trim is offset arithmetic, same as
  `KVCache`).
- Benchmark subcommand: unused-import/`del`-vs-lambda lint traps cleaned; suite is ruff-clean.

## [0.2.0] — 2026-07-04 — continuous batching, penalties & logprobs, auto-calibration, prompt-lookup, decode-path performance

### Added
- **Continuous batching (`serve --max-batch N`, the moonshot).** Run up to N concurrently-queued
  requests through one batched target forward so they share a single weight-read per step — the
  paper's cheap-verify regime, on a Mac. New `batch_engine.py`: a general dense-mlx-lm batched
  forward (any Qwen3/Llama/Mistral-class target; gemma-4 vlm falls back to serialized) over a
  **left-aligned per-row-offset KV cache** (per-row trim = O(1) metadata, the rollback batched
  spec needs; mlx-lm's `BatchKVCache` can only trim uniformly). `BatchEngine` micro-batches
  requests with matching sampling params; a lone request / temp>0 dspark / penalized / logprobs
  request takes the serial path, so B=1 latency never regresses. Both the target verify **and** the
  DSpark drafter are batched (the drafter's ragged per-row context is padded + masked). Measured
  (Qwen3-4B-8bit, M4 Pro, 4 concurrent): **baseline B=4 2.46× aggregate**, **dspark spec B=4 ~1.67×
  over serialized spec** (129 tok/s; batching the drafter adds 1.16× over verify-only). Lossless
  per row: B=1 is bit-exact vs single-seq;
  at B>1 output is greedy-correct up to the target's batch-dependent quantized-matmul rounding (the
  same qmv→qmm knee as the perf notes; ~0.5% of tokens, inherent to any batched quantized server).
- **`presence_penalty` / `frequency_penalty` (OpenAI), lossless-wrt-penalized-target.** Penalizes
  the target logits (each verify position by the base completion counts **plus** its own draft
  prefix) so speculative/greedy output equals sequential decoding of the penalized target — for
  temp>0 too (speculative sampling stays exact wrt the penalized target `p`). Validated: penalized
  spec == penalized baseline byte-for-byte; opt-in (default path untouched, ~0.9× when active).
- **`logprobs` / `top_logprobs` (chat + completions).** Reports the raw target log-softmax at each
  committed token (chosen + top-k), gathered on-GPU only when requested (default fused path
  untouched). Validated: logprob vs a fresh forward matches to ~1e-6; spec and baseline report
  identical logprobs. Response uses OpenAI `choices[].logprobs.content[]` (and the completions shape).
- **Hardware-aware dspark-vs-DFlash signal (`calibrate.knee_width` / `drafter_recommendation`).**
  Detects the quantized-matmul knee from the calibrated verify curve: a small knee (M-series, ~4)
  → dspark wins (what `--mode auto` picks); a knee that has moved past the DFlash block width
  (M5-class) → DFlash full-block re-enters play. Surfaced in the calibration output + `/metrics.auto_cap`.
- **`--max-draft auto` (hardware-aware auto-calibration).** On Apple Silicon the verify cost is
  convex in tokens-per-round with a machine/model-dependent knee (M4 Pro + gemma-12B-8bit: knee
  at width 4 — the reason cap=2 was optimal). `auto` measures this machine+model's verify/drafter
  cost curves once (~seconds, cached in `~/.cache/mlx_dspark/`) and a live controller picks the
  cap each round from the curves + an acceptance EWMA — so the cap tracks the hardware (M1→M5),
  the model, and the content, instead of a hard-coded default. Works for `dspark` and `dflash`,
  CLI + server (`x_mlx_dspark.cap`, `/metrics.auto_cap`). Lossless by construction: the cap only
  decides how many drafted tokens are *verified*; the target still verifies every emitted token.
- **`--mode lookup` (prompt-lookup speculative decoding).** Drafter-free speculation for **any**
  target model mlx-lm/mlx-vlm can load: propose the continuation of the most recent earlier
  occurrence of the current suffix n-gram (RAG quotes, code edits, "repeat/refine" turns), verify
  with the target as usual. No draft on a miss (zero overhead — trigram-minimum matching keeps
  chat overhead ~1–4%), greedy-lossless, temperature>0 supported via one-hot-proposal speculative
  sampling (still an exact target sample). Measured (Qwen3-4B, thinking off): copy-heavy prompt
  **2.38× (119 tok/s, accept 5.9)**, code edit 1.19× — all outputs identical to greedy. New
  `lookup_generate()` API; server + CLI wired; prefix caching works (it's a plain dense-cache path).
- **Sampling defaults from the model's `generation_config.json`** (server): requests that omit
  `temperature`/`top_p`/`top_k` now get the model authors' recommended values instead of silently
  greedy. Explicit request values — including explicit 0 — always win. Shown at startup. Many
  mlx-community conversions ship no `generation_config.json` (the Qwen3 repos don't; gemma does),
  so `--default-temperature` / `--default-top-p` / `--default-top-k` flags can supply them.
- **`--default-max-tokens` (2048) / `--max-tokens-cap` (32768)** — replaces the old fixed
  512-default/8192-cap, which truncated thinking models mid-reasoning.

- **`--mode auto`** — picks the best available speculation for the target: the registry's
  DSpark drafter if known, else DFlash, else drafter-free lookup — so **any** model repo now
  serves with some speculation and no extra flags (unknown targets previously errored).
- **Hybrid drafting (dspark mode, on by default)** — when the current suffix n-gram already
  occurred in the context (quoting, code edits, repeats), the free continuation is verified
  instead of running the drafter that round; elsewhere DSpark drafts as usual. Lossless
  composition; disable with `--no-lookup-drafts`. `GenResult.lookup_rounds` /
  `x_mlx_dspark.lookup_rounds` show how often it fired.
- **Prefix caching for Gemma-4 (sliding-window targets)** — rotating caches are exact until
  they first wrap, so entries are reused while under the window and refused at store time the
  moment any layer wraps. Gemma multi-turn now skips re-prefilling like Qwen does.
- **LRU prefix-cache slots** (`--prefix-cache-slots`, default 2) — an agent process and a chat
  window no longer evict each other's conversation every turn; per-slot SSD spill retained.
- **`mlx-dspark benchmark`** — warm, reproducible sweep (baseline + chosen modes/caps,
  including `auto`) with device + mlx version, optional `--json` — for comparable numbers
  across M1→M5 machines.
- **Chunked prefill** — long prompts prefill in 2048-token pieces with `mx.clear_cache()`
  between, bounding activation memory (the `[L, vocab]` logits especially) on ≤16 GB Macs;
  identical single-forward path for prompts within one chunk. The engine also wires MLX's
  recommended working set at start (like mlx-lm's server) so weights stay resident;
  `doctor` reports/suggests `iogpu.wired_limit_mb`.

### Fixed
- **`import mlx_dspark` crashed on transformers ≥5.13** (`AttributeError: 'str' object has no
  attribute '__module__'`), which fresh installs resolve to. Root cause is upstream: mlx_lm registers
  a tokenizer by a string key and transformers 5.13 made `_LazyAutoMapping.register` assume a config
  *class*; the failure runs at mlx_lm module scope, so it took down `import mlx_dspark`. A scoped,
  idempotent compat shim in `__init__` restores the pre-5.13 behavior for non-class keys — no
  `transformers` version pin, real class keys untouched. Reported by @zboyles (#1).
- **Serving Gemma-4 (mlx-vlm targets) was broken since 0.1.0** — every request failed with
  `There is no Stream(gpu, 1) in current thread`. Root cause: mlx-vlm's model load switches the
  loading thread's default stream to a thread-local one, so models loaded on the main thread
  couldn't be run from the engine's generation thread. The engine now loads (and calibrates) on
  the same single thread that generates. (Qwen was unaffected — mlx-lm doesn't switch streams.)
- **A streaming client disconnect no longer invalidates the prefix cache.** The server converts
  a broken pipe into a graceful stop (`StopStreaming`): generation ends at the round boundary,
  the (consistent) caches are stored, and the next turn still gets prefix reuse.
- Speculative loops now stop when an accepted draft contains eos mid-block (previously they
  could generate past it).

## Decode-path performance (same release)

Output is unchanged everywhere (byte-identical token ids, streamed text, and final text validated
A/B on Qwen3-4B, 300- and 1200-token runs, greedy + dspark). Measured on an M4 Pro:

### Changed
- **Streaming detokenization is now incremental** (`_Streamer` feeds mlx-lm streaming
  detokenizers; SPM/BPE class auto-selected for plain HF fast tokenizers, full-re-decode fallback
  for anything else). Previously every round re-decoded the whole output — O(n²) over a
  generation, worst exactly on long/thinking outputs.
- **One device sync per speculative round** (greedy default path, dspark + dflash): the drafted
  tokens no longer round-trip to the CPU before verify — `verify_ids` is assembled on-GPU and the
  accepted-prefix length is computed in-graph (cumprod of positionwise matches). Drafter-context
  updates are scheduled with `mx.async_eval` instead of blocking.
- **Pipelined baseline decode** (`greedy_generate`): step t+1 is scheduled on the GPU before step
  t's token is read (mlx-lm style `async_eval`), overlapping detokenize/emit with GPU compute.
  Closes the previously-noted ~5% gap vs `mlx_lm.generate` (baseline now ~52 tok/s on Qwen3-4B-8bit,
  at parity with the official runner).
- Net: baseline **+6–7%**, dspark **+2–3%** on Qwen3-4B (larger relative effect on long streamed
  generations); 3 new model-free tests (48 total).

## [0.1.0] — serving & tooling

Turns mlx-dspark from a library + demo CLI into a usable local **tool** — serve a DSpark/DFlash
model to LM Studio, the `openai` SDK, or any OpenAI-compatible client. All additions keep the
lossless verify loop; the OpenAI surface is stdlib-only (no FastAPI/uvicorn added).

### Added
- **OpenAI-compatible API server** (`mlx_dspark.server`, `python -m mlx_dspark serve`). Point any
  OpenAI client / LM Studio / `openai` SDK at `http://host:port/v1`. Endpoints:
  `POST /v1/chat/completions` (streaming SSE **and** non-stream, **multi-turn**), `POST /v1/completions`,
  `GET /v1/models`, `GET /health`, `GET /metrics`. Serves `dspark` / `dflash` / `baseline` on one target.
  Params: `temperature`, `top_p`, `top_k`, `max_tokens`, `stop`, `seed`, `stream`, optional `--api-key`,
  CORS. Spec-decode gain surfaced in an `x_mlx_dspark` block (accept length, tok/s) + `/metrics`.
  All generation runs on one dedicated thread (MLX arrays are thread/stream-affine).
- **Prefix caching** (in-memory + optional **SSD spill**) — reuse the shared conversation prefix's KV
  across turns instead of re-prefilling it. **~13× faster turn-2** on a ~750-token shared context
  (measured). On for `dspark`/`baseline` on dense (trimmable-KVCache) targets; falls back for DFlash
  and Gemma-4's rotating/sliding-window caches. Lossless to the same fp-tie standard as the rest of
  the project; invalidated on any error so it can't desync. Flags: `--no-prefix-cache`,
  `--prefix-cache-dir`, `--prefix-cache-max-ram-mb`.
- **Tool calling** — OpenAI `tools` / `tool_calls`, parsed from both native formats (Qwen3 Hermes-JSON
  and Gemma-4 `<|tool_call>call:…`), streamed as `delta.tool_calls`; inbound history normalized so
  prior tool calls render through the chat template.
- **Lossless top-p / top-k sampling** — nucleus/top-k truncation applied to both draft and target so
  temperature sampling stays an exact sample from the (truncated) target. Validated model-free.
- **Thinking toggle** — per-request `enable_thinking` / `chat_template_kwargs` and a server `--no-thinking`
  default (silences Qwen3 `<think>` blocks for a served endpoint).
- **Model-centric interface** — name the **target** with `--model <hf-repo | local-path>` (like
  mlx-lm); the matched drafter auto-resolves from a registry (quantization-agnostic), or pass
  `--drafter`. Replaces the old 2-value `--family`. `mlx-dspark models` lists targets with a known
  drafter. `--family` / `--target` / `load_pair("qwen3")` kept as **deprecated** aliases (still work).
- **Subcommand CLI** — `serve` / `generate` / `models` / `doctor` (env + model-fit check), plus a
  `mlx-dspark` console-script entry point. The old flat `python -m mlx_dspark --prompt …` still works.
- **Test suite** (`tests/`, 35 tests) covering the server protocol, streaming, stop sequences,
  tool-call parsing, top-p losslessness, and the prefix-cache manager — all model-free (fast, CI-friendly).

### API
- `generate()` functions gained `prompt_ids=`, `cache=`/`ctx_caches=`/`reuse_len=` (prefix reuse),
  `stop=`, `top_p=`, `top_k=`, and a `finish_reason` on `GenResult`; new `encode_messages()` (multi-turn).
  Backward compatible.

## [0.0.3]

### Added
- **z-lab DFlash drafter support** (block-diffusion speculative decoding). Run z-lab's original
  DFlash checkpoints natively on Apple Silicon through the same lossless verify loop as DSpark:
  - `load_dflash()`, `load_dflash_pair()`, `DFLASH_PRESETS`, `dflash_generate()`, and a
    `python -m mlx_dspark --mode dflash` CLI path (`--max-draft 0` = full block).
  - Presets: `gemma4` (`z-lab/gemma4-12B-it-DFlash`) and `qwen3` (`z-lab/Qwen3-4B-DFlash-b16`).
    Other z-lab adapters (e.g. `Qwen3-8B-DFlash-b16`) share the arch and load via `load_dflash(repo)`.
  - DFlash reuses the **target's** embed/lm-head (bound automatically); the drafter model classes
    are vendored from [z-lab/dflash](https://github.com/z-lab/dflash) (MIT) — see `NOTICE`.
  - Greedy **and** temperature>0 (lossless speculative sampling) for DFlash.
- **DSpark vs DFlash head-to-head** in the README (same target/Mac): DFlash's block-16 wins
  code/math (accept ~6, ~2.1×); DSpark's markov head wins open chat.

## [0.0.2]
### Added / changed
- Drafter-slice speedup (compute lm_head/markov over `cap` positions only) — output-neutral +9–10%.
- `--max-draft 2` is the new default (measured M-series optimum for both families).
- Lossless temperature speculative sampling (`--temperature`, paper §2.1).
- Optional 4-bit target (`--target ...-4bit`) for max absolute throughput / ≤24 GB Macs.

## [0.0.1]
- Initial release: DSpark speculative decoding for Apple Silicon (MLX), Gemma-4 12B + Qwen3-4B.
