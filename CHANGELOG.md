# Changelog

All notable changes to `mlx-dspark`. Versions follow [SemVer](https://semver.org/) (pre-1.0: minor-ish features land as patch bumps).

## [0.17.0] — 2026-08-25 — CPU co-prefill + security hardening (#26, #27)

### Security
- **Model-supplied code is no longer executed on load (issue #26).** mlx-lm imports a checkpoint's `config.json: model_file` as Python and loads tokenizers with `trust_remote_code=True`, so a crafted repo or local path given to `/admin/load` (or `--model`) could run code as the serving user. `load_target` now scans `config.json` / `tokenizer_config.json` / processor configs (including nested `text_config`) for `model_file` / `auto_map` and refuses the checkpoint with the offending keys named; the tokenizer is loaded with `trust_remote_code=False`. Opt in per process with `--trust-remote-code` or `MLX_DSPARK_TRUST_REMOTE_CODE=1` — deliberately no per-request override. A refused hot swap of a local path is caught before the running model is released. No registry / mlx-community checkpoint carries such markers. Thanks to @cryptedx for the report.
- **Every route but `GET /health` now requires the `--api-key` when one is set (issue #27).** The admin GET routes — `/admin/integrations` in particular, which returns agent configurations containing the key — were unauthenticated. The Mac app already authenticates all its calls. Added `SECURITY.md` with the threat model and the private reporting channel.

### Added
- **CPU co-prefill** — prefill now uses the CPU's matrix units as a second GEMM engine alongside the GPU. Above a calibrated row count, every wide quantized matmul dequantizes its weight once on the GPU (the existing wide-GEMM transient) and hands a measured fraction of its rows to MLX's CPU stream, which runs concurrently in the same thread with no copy and no new dependency. Measured on an M4 Pro with Qwen3.8-27B-4bit, 2048-token prompt: **130 → 184 tok/s prefill (1.41×; 1.32× over the previous wide-GEMM path)**, +0.4 GB peak; 8k-token prompt 1.19–1.26×. Numerics are the same fp-tie class as chunked prefill (Δlogit band identical; 64-token greedy continuation token-identical), so the CLI/server enable it and the library API leaves it off (`generate.CPU_SPLIT`). The fraction is a real optimum with a cliff, so it is measured once per machine+model (`wide_gemm.measure_cpu_split`, ~15 s, cached) rather than guessed. New: `--cpu-split FRAC` on `generate` / `serve` / `benchmark` (0 disables; unset = calibrated), `/health.cpu_split`, `/admin/load {"cpu_split": 0|fraction}`, the serve banner and benchmark header state it, and `generate` prints prefill seconds. Not yet applied to bf16 targets (`nn.Linear`) or MoE expert matmuls.
- **Apple Neural Engine assessed and documented** (NOTES "CPU co-prefill … why the ANE lost the audition"): measured through CoreML on the 27B MLP shapes, its throughput tracks the weight format (fp16 2.3 TF → int4/LUT 8.4 TF), but no ANE-placeable format holds mlx's K-grouped affine weights exactly, the exact fp16 form needs a 34 GB copy, one program owns the ANE, and it has an fp16-accumulate floor. Not integrated.

## [0.16.1] — 2026-08-23 — LFM2.5 tool-calling + thinking-off

### Added
- **LFM2 tool-call parsing** — the 5th native tool-call format the server understands. LiquidAI LFM2.5 emits `<|tool_call_start|>[func(arg="v", n=3)]<|tool_call_end|>` (a Python-style call list; both tokens are non-special, so they survive detokenization). Parsed with `ast` rather than a regex — quoted commas, nested lists/dicts and mixed scalar types all parse correctly; keyword arguments go by name and a rare positional argument is mapped to the request's tool-schema parameter order; a call truncated at `max_tokens` yields no call rather than raising. Wired into `parse_tool_calls`, so every surface gets it (OpenAI streaming + non-streaming, Anthropic / Claude Code), and `<|tool_call_start|>` is held back by the streaming tool-gate.
- **`scripts/prefill.py`** — a small CLI to measure prompt-processing (prefill) throughput for any target (registry id / HF repo / local path): warms up, then reports median prefill tok/s (and decode tok/s as a cross-check) per prompt length.

### Fixed
- **`enable_thinking=false` now takes effect on LFM2.5-2.6B** (and any reasoning template that hard-prefills `<think>` and ignores the flag). LFM2.5-2.6B is a pure-reasoning model whose chat template always ends the prompt with `<think>` and has no `enable_thinking` variable — so "thinking off" (the `--no-thinking` / `/admin/load {"enable_thinking": false}` / app toggle) was silently a no-op and the model reasoned anyway. `encode_messages` now force-closes an open reasoning block with an empty `<think></think>` when thinking is requested off — the same mechanism Qwen3's own template uses. It is a no-op where the template already closes the block (Qwen3.8) or never opens one (LFM2.5-1.2B-Instruct, 8B-A1B). On-device verified: 2.6B returns a direct answer with no leaked tags.

### Docs
- README "Prompt processing (prefill)" table gains the three LFM2.5 rows (bf16, M4 Pro, median of 3): 1.2B ~3240 tok/s · 8B-A1B MoE ~2170 · 2.6B ~1420 — the MoE prefills faster than the dense 2.6B (prefill tracks *active* params). "Target precision" refreshed to mlx 0.32 (Gemma-4 12B 2.78×, Qwen3-4B 1.82×) and the stale "a bf16 target is not a win" corrected (bf16-native LFM2.5 are among the biggest wins via `gemv_wide`). Small-M kernel: the M5 hardware gate is now documented in both the ceiling and Tuning sections.

## [0.16.0] — 2026-08-22 — LiquidAI LFM2.5 family (conv+attention hybrid — first conv recurrence)

### Added
- **LiquidAI LFM2.5-DSpark support** — three community drafters + their targets, registered and
  measured on M4 Pro (decode tok/s, greedy, lossless — fp ties only). Auto-resolving registry rows
  (`lfm2.5-2.6b` / `lfm2.5-1.2b` / `lfm2.5-8b-a1b`), quant-agnostic — `--model
  LiquidAI/LFM2.5-2.6B-MLX-bf16` (or `-8bit`) picks the drafter with no `--drafter`:
  - **LFM2.5-2.6B** (bf16) — 2.62× suite / 2.79× code, 3.41× math, 2.22× chat (cap 5–6, baseline ~43 tok/s)
  - **LFM2.5-1.2B-Instruct** (bf16) — 3.30× suite / 3.70× code, 3.78× math (cap 7, ~101 → 345 tok/s)
  - **LFM2.5-8B-A1B** (MoE `lfm2_moe`) — supported + lossless with **zero extra model code**; registered on **bf16**, where the drafter pays (cap 4 = 1.26× suite / 1.67× math), matching LiquidAI's M4 Max ~1.18×. At 8-bit it's a net loss (the ~1B-active step is too cheap for the drafter) — and 8-bit-at-baseline is still the fastest way to run the model, so it's kept out of the hook/results tables.

### Fixed
- **Claude Code's `/effort` (and `--effort`) now takes effect** (issue #25). Claude Code ships the reasoning level in `output_config.effort`, not `thinking`; `_encode_anthropic` now reads it as a per-request override, clamped to what the template accepts (`high` → `medium` on Qwen3.8, from #19) and degrading to the server default on an unknown value. Skipped when thinking is disabled. Thanks to the reporter for capturing the exact request.
- **First conv-recurrence target** (`model_type` lfm2 / lfm2_moe — LiquidAI's short-conv + attention
  hybrid). New `shortconv` recurrence in `target.py` (tap + conv-window capture + rollback) — the
  cheapest of the three recurrences (a kernel-3 causal FIR, no SSM state). The 8B MoE (`lfm2_moe`)
  loaded with **zero extra model code** (structurally identical to `lfm2` for the tap).
- **Fifth checkpoint packaging** (`config.py`): LiquidAI's `Lfm2DSparkDraftModel` nests
  `target_layer_ids` / `mask_token_id` in `dflash_config` with no `projector_type` tag — hoisted.
- **`rope_traditional` knob** honoring `rope_is_neox_style`: these heads use **interleaved** rope
  (~2× acceptance vs neox, measured). Absent = family-default neox, so every existing head is
  unchanged.

## [0.15.1] — 2026-08-22 — thinking default for API clients

### Added
- **A server-side thinking default that API clients inherit** (issue #19, part 2). Clients like
  DSH, WorkBuddy, pi and Claude Code have no reasoning toggle for local models, so every request
  got the engine default — thinking on — and a simple question could spend 13k tokens reasoning.
  `/admin/load` now takes `enable_thinking` (`false` = off for requests that don't specify it, the
  same as `serve --no-thinking`; `true` = the model's own default) and `reasoning_effort`
  (low/medium/high/xhigh), both **sticky across later model swaps** like `context_window`, and
  `/health` reports `thinking_default` (`on`/`off`; the key's presence is the capability gate). A
  request that asks for thinking explicitly still gets it. The Mac app exposes it as "Thinking
  (API clients)" in Settings → Decoding.

## [0.15.0] — 2026-08-21 — roofline telemetry, memory-pressure guard, long-context SDPA split

### Fixed
- **Streamed Anthropic (Claude Code) answers lost their last 20 characters** — the tool gate's
  held-back lookahead was never released at finish for answers that reached the gate leading
  with whitespace (every Qwen3-style streamed turn). Reported, diagnosed and fixed by
  @Griffin-Thomas in #13.
- **The small-M verify kernel is gated off on M5 (`applegpu_g17`+) — it stalls there** (issue
  #19). On M5 the kernel hangs ~105 s mid-generation (reporter A/B on an M5 Pro: kernel on =
  105.1 s inter-chunk gap, off = 0.6 s; mode-independent — both dspark and dflash). It's the
  third M5 datapoint (after the #14 wedge and the #7 thread). The gate is by GPU architecture,
  before any probe or cache, because the per-shape race can't detect a stall (a microbench never
  hangs; only the sustained workload does). `MLX_DSPARK_FORCE_SMALL_M=1` overrides it for a
  paired A/B. The kernel is verified only on M4 (g16, where it was developed); a newer generation
  must re-earn it. This also removes the need for a UI opt-out — the server default is now empty
  on M5, so app users get the fix through the engine bootstrapper with no app update.
- **`reasoning_effort: "high"` no longer 400s on models that don't define it** (issue #19).
  Agent clients (WorkBuddy, pi) hardcode `"high"`, which the Qwen3.8 template (low/medium/xhigh,
  no `high`) rejected outright. An unsupported-but-valid effort now maps to the nearest value the
  loaded template accepts on the low<medium<high<xhigh scale, ties rounding down toward less
  thinking — so `"high"` becomes `medium` on Qwen3.8. (Mapping, not dropping: dropping would fall
  back to the template's own default, which is `xhigh` = the *most* thinking — the opposite of
  what a rate-limited client wants.) A value outside the union is still a clear 400.

### Added
- **Memory-pressure guard** (`serve --memory-guard`, default on; `--no-memory-guard`;
  `/admin/load {"memory_guard": bool}`; state on `/health.memory_guard`, `/machine.guard`,
  `/metrics.memory_guard`, plus a `/health.warnings` row after a shed). When macOS reports
  memory pressure (`kern.memorystatus_vm_pressure_level`), the engine gives memory back *before*
  the OS swaps the model: at **WARN** it returns the MLX allocator's retained buffers
  (`mx.clear_cache()` — ~1.3–1.5 GB after a long 27B prefill, free to re-acquire) and drops the
  prefix cache's interior rungs (~150 MB fp32 each on a hybrid), **keeping every conversation's
  boundary checkpoint**; at **CRITICAL** it empties the prefix cache. Edge-triggered with a
  120 s re-arm; the shed runs on the generation thread — immediately when idle, at the next
  round boundary when generating (WARN waits ≤60 s for the request to finish; CRITICAL takes the
  next round). A/B on Qwen3.8-27B-8bit under real WARN pressure: 1.66 GB freed with the prefix
  hit preserved (TTFT 4.2 s vs 4.7–7.1 s in the OFF arms). The first policy (drop older
  conversation slots at WARN) was measured and rejected — it cost a 36 s re-prefill to free
  0.6 GB. Decode under pressure is unchanged by design — the OS paging 29 GB of weights is what
  sets it; the guard buys headroom. See NOTES "Memory-pressure guard".
- **Roofline + machine telemetry — "is this Mac actually saturated?"** (borrowed from the
  author's unpublished `inferviz` roofline dashboard; all reporting, lossless, additive). The
  engine now measures this Mac's *achievable* memory bandwidth once (512 MB fp16 matvec, cached
  in `calibration.json` per chip × mlx; M4 Pro: 226 of 273 GB/s) and knows the loaded model's
  exact byte footprint (per-tensor loaded bytes; MoE counts routed experts at `top_k / n`, the
  embedding gather is excluded; KV bytes/token from the config), so it can report the
  single-stream decode ceiling `bandwidth ÷ bytes-per-token` and how far *above* it speculation
  is taking the live rate. New module `roofline.py` (pure, mlx-free, sysctl via ctypes — no
  psutil). Surfaces:
  - **`GET /machine`**: chip (family, GPU cores, spec bandwidth), measured bandwidth, what macOS
    sees (**memory pressure level**, swap, free %, wired limit) + the MLX allocator, the model's
    footprint, ceilings at zero / last-request / full context-window depth, **baseline MBU**
    from the calibration's already-measured width-1 step (no new measurement — Qwen3-4B-8bit:
    92% of measured bandwidth, ceiling 52.9 tok/s ≈ the known 52 tok/s baseline), and a
    data-driven **verdict** (`level · headline · findings · levers`: memory-cliff first, then the
    roofline reading, then the speculative-decoding reading, then decay / cold / context-fill).
    Answers model-less too (chip/bandwidth/memory only).
  - **`/health.warnings[]`** — `{code, level, message, action}` rows a client shows as a banner:
    live macOS memory pressure, and the engine's load-time notes (the context-window RAM
    estimate, which used to reach only stderr).
  - **`spec_info` per-request tiles** (`x_mlx_dspark`): `prompt_tokens`, `cached_tokens` (prefix-
    cache reuse — why turn 2 is fast), `completion_tokens`, `context_tokens`, `prefill_seconds`,
    `decode_seconds`, `ttft_seconds`, `prefill_tokens_per_sec` (≥16 fresh tokens), `decay_ratio`
    (late/early decode rate within the request, from the round log), `swap_delta_bytes` (the
    fits-but-swaps cliff), `cold`, `ceiling_tokens_per_sec`, `roofline_ratio`.
  - **`/metrics.system`** (pressure/swap/free) + **`/metrics.verdict`**; `/doctor` gains `chip` +
    `memory` (the `doctor` CLI prints chip, spec + measured bandwidth, and a pressure warning);
    **`/admin/models`** gains `bandwidth` (this Mac vs the reference M4 Pro, like-for-like —
    `scale` 1.0 on an M4 Pro) and per-row `weight_bytes` / `ceiling_tps` for installed targets
    (the plain-decode physics a picker can quote before loading).
  The OpenAI `usage` block is untouched (PRs #9/#24 cover `prompt_tokens_details.cached_tokens`).
  Warmup restores the verdict as it restores everything else. 3 new test files' worth of
  model-free coverage (roofline math, footprint, verdict ladder, sysctl readers, decay ratio,
  bandwidth cache, spec_info tiles, /machine); validated live on Qwen3-4B-8bit.
- **Decode-only tok/s reporting.** Every throughput surface was end-to-end (prefill + decode) —
  the pessimistic number, and not comparable to other local runtimes ((LM Studio, oMLX and others))
  which report decode-only. `GenResult` keeps `tokens_per_sec` (end-to-end) and gains
  `prefill_seconds` / `decode_seconds` / `decode_tokens_per_sec` (always ≥ end-to-end — the
  visible number goes up, never down). Exposed in `spec_info` (`decode_tokens_per_sec`),
  `/metrics` (`mean_decode_tokens_per_sec`), and the `benchmark` / `generate` CLI. Purely
  additive — lossless, existing keys unchanged.
- **SDPA verify-split for the long-context cliff** (`--sdpa-split`, default on where a per-chip
  probe finds a cliff). mlx's attention re-reads the whole KV cache once per query row at q_len
  ~6–15 (a cliff: q5 2.48 ms → q6 7.09 ms at 32k on an M4 Pro), so wide verify at agent-depth
  context is a net loss. A wide-verify SDPA is split into ≤5-row sub-calls that each stay on the
  fast path (per-row equivalent — fp-tie, the target verifies every token). The verify depth
  slope is re-measured under the split so `--max-draft auto` prices the flattened cliff. Measured
  cap-7 ~6.6k-ctx decode 13.6 → 15.2 tok/s (1.11×); on high-accept long content the auto-capper
  goes wide (cap 2 → 7) for ~1.30× at ~14k. `--no-sdpa-split` forces off; `/health` +
  `/admin/load` report/override it.
- **Hybrid n-gram copy drafting for DFlash mode** (`dflash_generate` `lookup_drafts=`, default
  OFF). Ports the dspark copy path to the DFlash block loop. Lossless; measured a net loss on the
  strong-drafter Qwen3.8 + DFlash 2 pair (copy pays for weak drafters, not strong ones), so it
  stays off and library-only.
- **On-load warmup so the first request is fast** (`serve --warmup`, default on). After a model
  loads (server startup, `/admin/load` hot-swap, or a model-less server's first load) the engine
  runs one tiny throwaway generation through the real decode path — priming the Metal kernels and
  ramping the GPU clock — before it reports `ready`. Otherwise that ~2 s cold-start lands on the
  user's first message (and shows up entirely in prefill — see the decode-only tok/s note). The
  warmup keeps nothing: prefix cache, round log (`/events`/`/rounds`), `/metrics` stats and the
  auto-cap controller are all bypassed and restored. `/health` reports `phase: "warming_up"` while
  it runs (so a client can show "Warming up…" instead of a load bar that looks stuck) and a
  `warmup` flag once ready; `--no-warmup` / `/admin/load {"warmup": false}` disable it per-load.
  Adds ~1.5 s to a load; the payoff is on a genuinely cold process (fresh boot / first-ever run /
  post-mlx-upgrade) — on an already-warm machine the first request is fast either way.

## [0.14.0] — 2026-08-20 — depth-aware verify caps: long-context decode fixed

### Fixed
- **OpenAI tool-calls streams no longer buffer the whole generation** (issue #19). With
  `tools` in a streaming request, the server used to hold back every token until the end —
  with thinking models' 4–6k-token reasoning preambles that meant minutes of dead air, and
  agent clients with inter-chunk idle timeouts (DSH/pi at 300 s) dropped the stream and lost
  the turn. Now reasoning streams live as `reasoning_content` deltas and pre-tool-call answer
  text streams as `content`; only whole `tool_calls` land atomically at the end (the same
  splitter+gate composition the Anthropic dialect has had since v0.6.0).
- **Stream keep-alives on the chat dialect are now spec-legal empty-delta chunks**, not SSE
  comments — most client SDKs never surface comments, so a comment didn't reset their idle
  timer and a long quiet stretch (a 32k prefill, a long tool tail) still timed the stream out
  client-side. An empty delta parses as a normal chunk everywhere. The Anthropic dialect
  already used native `ping` events; legacy `/v1/completions` keeps the comment.
- **Slow-round diagnostic**: any inter-round gap over 10 s (`MLX_DSPARK_SLOW_ROUND_LOG_S`)
  is logged to stderr with mode/cap/context — so "generation stalled mid-stream" reports can
  say whether the stall was inside the loop and at what depth, without a profiler.
- **Long-context decode collapse: the verify width now adapts to context depth.** Community
  report ("slowing down a lot on large context") reproduced and root-caused: with 2–8 query
  rows Metal's attention kernel re-reads the whole KV cache once per row, so verify cost picks
  up a width-proportional depth term the chat-depth calibration curves cannot see — measured
  on Qwen3.8-27B-4bit the shipped cap-7 defaults fell from 1.17–1.41× at 2k context to
  **0.53×/0.97× (net loss) at 32k** while cap 3 there still gives **1.48×**, with acceptance
  flat throughout. Calibration now also measures each pair's per-width **verify depth slope**
  (one-time ~10 s, backfilled into existing cache entries), `--max-draft auto` prices it live
  (the controller's observed-time feedback alone measurably failed to fix depth), and the
  server refines a **derived** default cap per request from the prompt length — a no-op below
  ~4k context (every measured chat-depth default, including dflash's full block, is
  untouched), shrink-only beyond it, and never applied to a cap you set explicitly.
  `spec_info.cap` reports the effective value. Full write-up: NOTES "Long-context decode".

### Added
- **`--kv-bits` now works on gated-DeltaNet hybrid targets** (qwen3_5 / qwen3_5_moe — so
  Qwen3.6/3.8/Ornith/Bonsai-class models). `make_cache` builds a MIXED cache: only the
  full-attention layers' KV is quantized (on Qwen3.8-27B those 16 of 64 layers are the
  entire 64 KB/token context growth — and the entire long-context decode slowdown, see
  NOTES "Long-context decode"); the recurrent layers keep their fixed-size state
  untouched. Spec verify + hybrid rollback + prefix caching all carry the quantized
  entries unchanged (the machinery is generic over cache state); a model-free test pins
  spec-with-rollback == committed-forward on the kv8 mixed cache. Mamba-2 hybrids
  (nemotron_h) stay refused by name until measured. The RAM-aware context warning now
  also scales its estimate by the configured kv-bits.

### Changed
- **`context_window` is now sticky across `/admin/load`s** (community report: scripts that set
  it once via a manual `/admin/load` found a later load without the field silently reverting to
  the model's 262k maximum — a RAM hazard on machines the cap was protecting). An explicit value
  persists until changed; `0` resets to the model's own maximum; omitted keeps the current
  setting. Per-pair knobs (`mode`/`max_draft`/`lookup_drafts`/`confidence_threshold`) are
  unchanged: omitted, they re-resolve to the pair's measured defaults on every swap — that IS
  their reset semantics, so they stay per-swap by design.

## [0.13.1] — 2026-08-19 — kv_bits joins the /admin/load overrides

### Added
- **`kv_bits` is now an `/admin/load` override and a `/health` field** (issue #17 — the desktop
  app had no way to set `--kv-bits`). Per-swap: `4`/`8` quantize the target's KV cache (the
  long-context RAM lever — 8-bit halves, 4-bit quarters the per-token cache), `0` is explicitly
  full precision, omitted keeps the server's startup setting. `/health` always reports the live
  value (`0` = full precision), so clients can gate their picker on the key's presence — engines
  without the override also lack the key. Validated live: load with `kv_bits: 8` → health
  reports 8 → generation runs the kv-quantized cache (accept/speedup unchanged, the v0.2.0
  kv-bits guarantees apply); bad values 400 with the accepted set named.

## [0.13.0] — 2026-08-19 — DFlash 2: the new project best, with prefix caching and auto-mode pickup

### Added
- **DFlash 2 drafters run natively** (`--mode dflash`; Inco AI's DFlash successor — a candidate
  path selector over the target head's top-16 tokens per slot plus two-tap dynamic convolutions,
  ported from the merged SGLang reference implementation). Config-gated in the existing DFlash
  loader, so DFlash 1 checkpoints behave byte-identically. Greedy stays single-sync; sampled
  decoding is lossless through the selector's own proposal distribution.
- **Qwen3.8-27B's measured best is now DFlash 2 on both quants** (`incoai/Qwen3.8-27B-DFlash2`,
  one head serves both; registered for auto-resolution). Same-session paired benchmarks at the
  identical verify width 8: **8-bit 3.63× mean** (4.06× math / 4.05× code / 2.79× chat, accept
  5.53, 8.4 → 30.5 tok/s) vs the DSpark head's 2.92×; **4-bit 2.30×** (accept 5.14,
  **33.8 tok/s — the fastest decode in the project**, ~18 GB) vs 2.01×. Lossless both quants
  (divergences from single-row greedy are fp ties at margins 0.0–0.125).
- **Prefix caching now covers `--mode dflash`** (checkpoint mode) — the one mode it skipped. The
  drafter's context is rebuilt from a bounded window of projected rows snapshotted at the stable
  prompt boundary (~21 MB on Qwen3.8-27B). Measured on the DFlash 2 pair (~4k-token system
  prompt): identical-repeat TTFT 38.3 s → **0.24 s (159×)**, next conversation turn 31.1 s →
  **1.08 s (29×)**, outputs byte-identical to cold runs and drafter acceptance preserved.
- **Registry rows can stamp a measured-best mode** (`"mode"`; the two Qwen3.8-27B rows carry
  `"dflash"`), and `--mode auto` resolves it first. `/admin/models` rows report `"mode"`.

### Changed
- **`serve` and `generate` default to `--mode auto`** (was `dspark`): a bare `--model` now runs
  each registry row's measured-best mode — which is what the README's numbers show. Side effect:
  an unknown target with no `--drafter` now runs drafter-free lookup speculation (with its
  banner) instead of erroring; pass `--mode dspark` to keep the old behavior. Explicit modes are
  unchanged, and `--mode dspark` still resolves the DSpark heads on Qwen3.8-27B for A/B.

## [0.12.4] — 2026-08-18 — Curves calibration fix + LM Studio model reuse

### Fixed
- **`GET /calibration` found no curves on any machine where the small-M verify kernel is live
  (i.e. every default setup since v0.12.0)** — the curves are cached under a `"|smm"`-tagged
  key when the kernel is applied, but the endpoint still read only the untagged key, so the
  app's Lab → Curves tab permanently showed "Not calibrated yet" on fully calibrated
  machines. The reader now prefers the variant matching the live kernel state and falls back
  to the other (`calibrate.cached_curve_entry`); the "not calibrated" hint no longer tells
  users to run `--max-draft auto` (any load without a fixed cap calibrates automatically).

### Added
- **Models LM Studio already downloaded are reused** (issue #12): `--model <publisher>/<model>`
  now loads straight from LM Studio's caches (`~/.lmstudio/models`, legacy
  `~/.cache/lm-studio/models`) when the exact directory exists — MLX layouts only (GGUF
  downloads are skipped and fall through to the hub). They also appear in the installed-models
  scan (`mlx-dspark models`, `/admin/models`, the app's "On this Mac" list) tagged
  `source: "lmstudio"`, and are excluded from the app-reclaimable disk total.

## [0.12.3] — 2026-08-18 — serve stream liveness: disconnects cancel generation, serve-side small-M toggle

### Fixed
- **A vanished streaming client no longer leaves its generation running to `max_tokens` on the
  single MLX thread** (issue #14 — the "serve wedges, `/health` stays OK" report). The OpenAI
  tool-calls stream buffers its whole generation by design (role chunk, then one delta at the
  end), so a client that timed out and disconnected mid-buffer was undetectable: the abandoned
  generation kept the one generation thread for minutes, every retry queued another one behind
  it, and from outside the server looked hard-wedged while `/health` (which never touches that
  thread) stayed green. Both stream dialects now run a keep-alive writer (SSE comments on the
  OpenAI stream, the existing `ping` on the Anthropic one, every 15 s —
  `MLX_DSPARK_STREAM_KEEPALIVE_S` overrides); a keep-alive write that fails marks the client
  gone and generation stops at the next round with the prefix cache intact. Verified live: a
  dropped 3000-token tools stream now frees the engine in ~1 s (was: the full generation), and
  the keep-alives also stop compliant clients from idle-timing-out during long buffered
  generations in the first place.
- **Mid-stream client disconnects are now logged** (one stderr line with the route and how many
  tokens the cut-short generation kept) — previously they were swallowed silently, which made a
  stalled client indistinguishable from a wedged engine in the server's own output (issue #14's
  diagnostic gap).
- **`--max-draft` help text was stale** (`generate` and `serve` both said "Defaults: dspark=2").
  The dspark default has been the calibrated `static_cap` for several versions — the help now
  says the cap is derived per machine+model+quant on first run (cached), and clarifies that
  `auto` adapts the cap per round.

### Added
- **`serve --small-m/--no-small-m`** — the small-M MMA verify kernel can finally be toggled at
  serve time (it was `generate`/`benchmark`-only, so the only serve-side A/B was downgrading the
  whole package). Unset keeps the probe-gated default; `--no-small-m` forces the stock kernel.
  `/health` reports the live state (`small_m`), the startup banner prints it, and `/admin/load`
  takes a per-swap boolean `small_m` override.
- **RAM-aware context-window warning at load** (issue #14's secondary finding): `serve` defaults
  the window to the model's own maximum (262144 on Qwen3.8-27B ⇒ ~16 GB of KV on top of ~29 GB
  of weights, which silently exhausted a 64 GB Mac). The engine now estimates the
  context-growing KV bytes/token from the model config (hybrid-aware: recurrent and
  sliding-window layers don't count; reproduces Qwen3.8-27B's measured 64 KB/token exactly) and
  warns at load — with a `--context-window` value that fits — when weights + full-window KV
  exceed ~90% of the machine's GPU working set. A warning, not a changed default.

## [0.12.2] — 2026-08-18 — better 4-bit Qwen3.8-27B drafter (DimInfer)

### Changed
- **`mlx-community/Qwen3.8-27B-4bit` now auto-resolves `DimInfer/Qwen3.8-27B-Dspark-v1`** as its
  drafter (was `RadixArk/Qwen3.8-27B-DSpark`, which stays the 8-bit row's head). DimInfer is a
  4-bit-class DeepSpec-stock head (block-15, deeper tap layers) that out-accepts RadixArk at
  every cap and content on the 4-bit target: measured **1.99× mean** at `--max-draft 7` (2.31×
  math / 2.14× code / 1.51× chat, accept up to 5.3) vs RadixArk's 1.82× the same session, and
  `static_cap` picks cap 7 unaided so a no-flag `--model …-4bit` already lands it. No
  `--confidence-threshold` needed (its acceptance is already high; the confidence head only pays
  where the drafter leaves acceptance headroom). Greedy-lossless as always. The 8-bit registry
  row is unchanged (RadixArk, trained against the FP8 verifier).

### Fixed
- **Small-M verify kernel was silently skipped when its shape cache predated the `b{bits}`
  `shape_key` format**: `apply_small_m` matched zero live modules against the stale-format shape
  strings and ran the stock kernel with no error (verify slower than intended). It now detects a
  cache whose shapes match no live module and re-measures/overwrites it
  (`small_m_qmm_shapes(refresh=True)`), self-healing without a global calibration-cache bump.
- **A hand-downloaded model in the plain-dir cache under its org-prefixed name
  (`<org>_<name>`, what `huggingface-cli`/`robust_download.py` produce) was treated as
  not-installed** — `_resolve`/`_is_local` only matched the bare basename. So a drafter like
  `DimInfer/Qwen3.8-27B-Dspark-v1` present on disk as `DimInfer_Qwen3.8-27B-Dspark-v1` was
  re-downloaded on load and its registry pair reported `ready: false` (which hid it from the
  app's ready-only model menu). All three checks — `load._resolve`, `diagnostics._is_local`,
  and `download._looks_like_repo` (the cancellable pre-fetch gate) — now match the
  `<org>_<name>` form, so a hand-downloaded copy is found instead of re-fetched.

## [0.12.1] — 2026-08-16 — race the confidence bundle

### Added
- **Per-arm `confidence` on `/admin/race`** (dspark arms; None = the server's loaded setting),
  so a measured cap+confidence bundle — Qwen3.8-27B-4bit's best is cap 7 + 0.3 — can race its
  plain-cap siblings head-to-head. `/health` advertises `race_arm_confidence: true` as the
  capability gate: clients must show a bundle arm only when it's present, because an older
  engine silently drops the field and the lane label would lie.

## [0.12.0] — 2026-08-16 — small-M verify kernel, cancellable downloads, long-context controls

### Added
- **Cancellable model downloads with live progress** (`download.py`): a first-time load now
  pre-fetches every hub repo in a killable child process before the loaders run, so the long
  phase of `/admin/load` — the download — can be stopped: **`POST /admin/load/cancel`**
  (optional `cleanup: true` also removes the partial files; the default keeps them so a
  retried load resumes instead of restarting a multi-gigabyte fetch). While fetching,
  `/health` reports `download: {repo, bytes_done, bytes_total}` for a real progress bar.
  A cancelled load unwinds like any failed load (server up, model-less), and a dying server
  takes its download child with it — no orphan quietly fetching gigabytes after quit.
- **Race arms take `cap: "auto"` and any cap 1–64** on `/admin/race`: an auto arm runs the
  per-round adaptive cap from this machine's cached cost curves (a fresh controller per arm,
  so no run biases another), raceable head-to-head against fixed caps; custom integer caps
  are validated instead of silently crashing on garbage.
- **`confidence_threshold` as an `/admin/load` override** (number in [0, 1]; 0 = off) and a
  matching `/health` report — so a client can apply a pair's measured cap+confidence bundle
  (Qwen3.8-27B-4bit's best is `cap 7 + 0.3`) without restarting the server.
- **`context_window` as an `/admin/load` override** (was serve-time-only via
  `--context-window`): a per-swap cap below the model's own maximum — the KV-cache RAM
  lever for long agent sessions; requests past it get the "prompt is too long" wording
  agent clients auto-compact on. Validated (integer ≥ 1024); `/health` already reported the
  effective window.

### Changed
- Registry `speedup` strings (what `/admin/models` badges show) refreshed to the current
  measured table — several predated the 2026-07-22 cap re-measure (Qwen3-8B 1.6→2.1×,
  gemma-4-12B 2.1→2.8×, Ornith 2.2→2.4×) and the small-M kernel results (Qwen3.8-27B
  **4-bit** 1.5→1.9× / **8-bit** 2.5→2.7×, Muse 1.5→1.7×) — these are `/admin/models` badge
  summaries; the end-to-end small-M measurements (4-bit `cap 7 + conf 0.3`, 8-bit cap 7, no
  confidence flag) are in the kernel entry below.
- **Small-M MMA verify kernel** (`small_m_qmm.py`, vendored MIT from avlp12's mlx-lm fork —
  see NOTICE): `mx.quantized_matmul` re-pays the whole weight read per row for M in 2..8
  (upstream ml-explore/mlx#4265), which is exactly the speculative verify window — the reason
  4-bit verify curves "rise steeply from width 3". An 8x8 `simdgroup_matrix` split-K kernel
  dequantizes each 4-bit weight group once and reuses it across all rows, making verify widths
  6-8 cost the same as width 5 (measured M4 Pro: 1.3-1.7x per-matmul at M=6-8, flat in M).
  Dispatched only for M in [6, 8] on shapes a one-time cached probe proves faster AND
  numerically sane on this machine (4-bit gs64, N≥4096 — the wide-GEMM doctrine); everything
  else stays on the stock kernel, at stock speed and stock numerics. On by default in the CLI
  and server (`--no-small-m` disables; library API stays off unless `calibrate.apply_small_m`
  is called). Cap calibration now measures its curves under the same kernel dispatch
  generation uses (cache schema 4, keys tagged `|smm`), so `static_cap`/auto re-derive caps.
  Output stays greedy-correct (the target verifies every token); ids can differ from the stock
  kernel at fp ties, like the batched path. Measured end-to-end (Qwen3.8-27B-4bit, M4 Pro,
  3-trial medians): **`--max-draft 7 --confidence-threshold 0.3` = 2.12x code / 1.97x math /
  1.55x chat (~1.88x mean, 27.5 tok/s)** vs the old shipped best 1.74x mean at cap 2 — the
  confidence head pays on a dense target for the first time (the wide acceptance spread the
  flat verify curve exposes is exactly its regime), plain cap 7 = 1.78x mean, `--max-draft
  auto` = 1.77x (the first hybrid pair where auto matches static — the flat curve fixed the
  controller's economics), and the no-flags default (static cap 2) keeps its old numbers plus
  2-4% from the drafter's width-8 block backbone riding the kernel. Lookup drafts re-measured
  at the flat curve: still a net loss on this pair (1.69x on vs 1.78x off at cap 7) — the
  registry's off default stands. An **8-bit unpack variant** covers 8-bit gs64 targets, where
  qmm is flat to M=5 but cliffs at 6: the kernel removes the cliff (1.20-1.62x at M=6-8), and
  on Qwen3.8-27B-8bit **static_cap moves 4 -> 7 unaided**, making the new best the zero-flag
  default: **2.72x mean (3.37x math / 2.84x code / 1.95x chat, accept 4.05) at 22.6 tok/s**
  vs the old 2.45x at cap 4 — and math acceptance reaches 5.15, the pair's highest measured.
  The confidence head does NOT pay at 8-bit (curve now flat 1-8, so truncation only costs
  acceptance) — the inverse of the 4-bit case, same mechanism.

## [0.11.0] — 2026-08-15 — reasoning effort, instant server start, streaming reasoning split

### Added
- **Reasoning-effort control** for models whose chat template supports it (Qwen3.8-class
  `reasoning_effort`, values `low`/`medium`/`xhigh` — the hint is a system-block instruction,
  ignored whenever thinking is off): serve takes `--reasoning-effort` as the server default, the
  OpenAI endpoint takes a top-level `reasoning_effort` per request (`chat_template_kwargs`
  passthrough worked before and still does), `/admin/race` takes it as a race knob, and
  `/health` reports `supports_reasoning_effort` (detected from the loaded template) plus the
  configured default so clients only show the control where it does something. Invalid values
  are a clear 400 at the boundary; templates that don't know the kwarg ignore it. Note the hint
  lands at the *head* of the prompt, so changing effort mid-conversation is a full prefix-cache
  miss — treat it as per-conversation.
- **Model-less server state** — `mlx-dspark serve --no-model` starts in ~2 s with nothing
  resident (the Mac app's fast-launch path): generation routes answer 503 with a clear "no
  model is loaded" reason until `POST /admin/load` brings one up on the same port; `/health`
  reports the new `no_model` status (distinct from `loading`: a client waits through one and
  offers a picker on the other). New **`POST /admin/unload`** releases the loaded model and
  frees its memory while the server and port survive (idempotent; `/admin/load` reverses it).
  `/doctor` and `/admin/models` now answer without a loaded model — a model picker has to work
  from exactly that state.
- **Streaming responses split reasoning into `reasoning_content`** on the OpenAI chat endpoint,
  matching the non-streaming path (which already did) and the DeepSeek-style convention clients
  expect. Covers both self-opened `<think>…</think>` and the prefilled-opener templates
  (Qwen3-2507 / Qwen3.8-class prefill the opener in the prompt, so the raw stream carries only
  a dangling `</think>` mid-text — every streaming client rendered the reasoning as answer
  text). `/v1/completions` streaming stays raw by design.

- **Qwen3.8-27B-8bit registered as its own measured pair** — model pickers (`mlx-dspark
  models`, the app) now list both quants; auto-resolution sends `…-8bit` to the new row
  (2.45× at cap 4, ~29 GB) and every other Qwen3.8 spelling to the 4-bit row (1.74× at cap 2
  but 25.3 tok/s absolute, ~18 GB); both carry the measured lookup-drafts-off default.

### Changed
- The catch-all integrations row is now "Any OpenAI-compatible app" and says the quiet part
  out loud: every tool that connects to llama.cpp's `llama-server` or Ollama connects here
  with the same base-URL setting.

## [0.10.1] — 2026-08-15 — the prefix cache actually hits: stable boundaries + hybrid partial reuse

### Fixed
- **Checkpoint-mode prefix caching (hybrid/recurrent targets, wrapped gemma-4) never hit in
  practice** (#7): the snapshot sat at the exact prompt boundary, so a byte-identical repeat
  could not hit (no token left to forward), and Qwen3.6/3.8-class chat templates re-render the
  `<think>` generation tail so real multi-turn requests missed by 1–4 tokens. The server now
  probes each template's *stable* boundary at runtime and snapshots there. Measured
  (Qwen3.8-27B-4bit, ~8k-token system prompt): TTFT 62 s → **0.21 s** on an identical repeat,
  **1.05 s** on a multi-turn extension; outputs byte-identical to the uncached run.

### Added
- **Partial prefix reuse for hybrid GDN/Mamba targets** — interior "rungs" snapshot only the
  recurrent layer state every `--prefix-cache-rungs` tokens (default 8192; the attention KV and
  drafter ctx are trimmed from the boundary snapshot), so a request that diverges mid-prompt
  (new session on the same system prompt, compacted history) partially reuses the cache instead
  of missing outright. Misses with a long shared prefix stage an **anchor** rung at the exact
  divergence point, so the next request from that fan-out hits (measured 0.53 s TTFT vs 60 s).
  Restore is bit-exact (validated array-for-array on Qwen3.8-27B). A conversation now collapses
  into one slot carrying a ladder of past boundaries instead of a chain of slots. `/metrics`
  reports `partial_hits` and per-slot `rungs`.

## [0.10.0] — 2026-08-15 — Qwen3.8-27B (SpecForge drafters) + per-pair defaults

First PyPI upload since 0.8.1 — it carries **0.9.0 below as well** (the server control plane +
telemetry behind the Mac app, prepped 2026-08-13 but never uploaded on its own).

### Added
- **Qwen3.8-27B support** — `--model mlx-community/Qwen3.8-27B-8bit` (or `-4bit`) auto-resolves
  `RadixArk/Qwen3.8-27B-DSpark`, the first **SpecForge/SGLang**-packaged head (a fourth checkpoint
  format): DFlash-backbone DSpark with nested `dflash_config`, anchor-as-pos0 sampling (the shipped
  base-class reference reads the wrong slot — measured accept 1.35 vs 3.42), a real **YaRN drafter
  rope** (honored for this packaging only; mlx-vlm's YarnRoPE matches transformers' yarn exactly),
  and target embed+lm_head reuse. Measured (M4 Pro, 3-trial medians, lossless): **8-bit cap 4 =
  2.45×** mean (3.00× math / 2.38× code / 1.96× chat, accept 3.43, 8.3 → 20.3 tok/s); 4-bit cap 2 =
  1.74× at 25.3 tok/s in ~18 GB. Both caps are the calibrated picks — no flags needed.
- **Per-pair lookup-drafts defaults** — registry pairs whose measured best runs with hybrid n-gram
  lookup drafts OFF (every MoE, the 4-bit 27B hybrids, Muse-Glimmer) now ship that default; the
  shipped configuration reproduces the vouched-for numbers with no flag. CLI flags are three-state
  (`--lookup-drafts` / `--no-lookup-drafts` / unset = pair default) on `generate`/`serve`/
  `benchmark` (which prints the setting's provenance); `POST /admin/load` accepts a boolean
  `lookup_drafts` override; a hot swap re-resolves the default for the incoming model; `/health`
  and `/admin/models` report it. Library `speculative_generate` defaults are untouched.
- **`POST /admin/race` takes an optional boolean `thinking`** — per-race chat-template override
  (the Lab's toggle), echoed in the SSE start event.

### Fixed
- The vanished-mode trap in clients driving `/admin/load`: the Mac app's Decoding picker derived
  its options from the *race* arms, so applying Baseline once removed DSpark with no way back
  (app-side fix; the endpoint was always fine).

## [0.9.0] — 2026-08-13 — the server grows a control plane + live telemetry (and a native Mac app)

The release that makes the server *observable and drivable*: everything the new Mac app renders
ships here as plain HTTP endpoints, so the CLI and any script get it too. The app itself lives in
`apps/MacApp/` (SwiftUI; installed via DMG / Homebrew cask, **not** part of the wheel — `pip
install mlx-dspark` stays engine-only).

### Added
- **Hot model swapping — `POST /admin/load`** switches the target in place (release-then-load, so
  peak memory is one model) while the server and its port stay up; `GET /admin/status` reports
  ready/loading/error, `/health` answers `status: "loading"` mid-swap, and everything else 503s
  until the new model settles. A failed load leaves the server up and recoverable.
- **Live per-round telemetry** (`telemetry.py`): an `on_round` hook in all four decode loops feeds
  a ring buffer + bounded-queue fan-out (a stalled client can never stall generation). `GET
  /events` streams every round the engine runs — engine-wide, not request-scoped, so a dashboard
  keeps updating while Claude Code is the one generating; `GET /rounds` is the polling sibling.
  Per-position acceptance (d₀, d₁, d₂ …) with honest denominators rides along in `/metrics`.
- **`GET /calibration`** — this machine's cached verify/drafter cost curves, the qmm knee, and the
  cost model's predicted tok/s per cap. Zero new measurement; pure surfacing of
  `~/.cache/mlx_dspark/`.
- **`GET /doctor` + `GET /admin/models`** — environment (chip, RAM, Metal, package versions,
  wired-limit hint) and the model inventory: every registry pair annotated with RAM feasibility
  for *this* machine, plus an **on-disk scan** (HF hub + plain-dir caches — sizes, paths,
  drafter-vs-model kind, quant-agnostic registry pairing) and total disk usage. `mlx-dspark
  doctor` renders the same payload (`--json`, `--models`).
- **`POST /admin/race`** — the same prompt through several decode strategies (dspark/dflash/
  lookup/baseline, per-arm caps), streamed with per-token timings, finished with per-arm stats and
  an **ids-identical verdict** — the losslessness claim as a checked result, not an assertion.
- **`GET /admin/integrations`** — ready-to-paste config for Claude Code, Codex, OpenCode, pi, and
  any OpenAI-compatible client, with the base URL taken from the request's own Host header.
- **Allocator memory in `GET /metrics`** (`memory`: active/peak/cache bytes) — what the loaded
  model actually holds resident.
- **Registry rows carry a measured `speedup` string** (the README table's headline ratio) so
  model pickers can answer "why this one".

### Fixed
- `knee_width` misreported the qmm knee (one flat step collapsed the baseline, so the knee read as
  4 where the measured curve is still flat; the real jump is at 6). Reporting-only — cap selection
  reads the curves directly and is unaffected.

## [0.8.1] — 2026-08-12 — causal-block drafter truncation: Muse-Glimmer +10–13%, Nemotron +2.5–3%

### Changed
- **Causal-block drafters (Muse-Glimmer, Nemotron) truncate the backbone to the rows the head
  reads — +10–13% on Muse-Glimmer, +2.5–2.8% on Nemotron, output ids unchanged.** The DSpark
  loop ran every drafter backbone at its full trained block width (15 for Muse) because a
  *bidirectional* block can't be shrunk without changing the drafting distribution — but the
  DFlash-lineage heads attend **causally** within the block, so position i never sees positions
  > i and computing only `logits_start + cap` rows is mathematically identical. Muse's 2.3B-param
  backbone was 17% of the round at width 15 (25.7 ms) and costs 10 ms at width 4. New
  `DSparkDrafter.draft_width(cap)` drives the generate loop and both calibration measurers (the
  auto-cap cost model now prices the truncated width; drafter cost rises with cap instead of
  being flat, so the calibration cache schema was bumped and cached curves re-measure). The
  drafter's block attention mask is also built once per forward instead of once per layer.
  Re-stamped on the benchmark prompts (M4 Pro, `--no-lookup-drafts`): Muse-Glimmer-30B **8-bit
  cap 4: 1.97× chat / 2.45× code / 2.99× math** (baseline 8.2 tok/s); **4-bit cap 2: 1.57× /
  1.70× / 1.94× (~25 tok/s, was ~1.47× mean)**; Nemotron-3.5-Lightning on the suite peaks at
  **cap 3 = 1.10× mean** (its 1.27×-code stamp is from higher-acceptance prompts — v0.8.0 vs
  0.8.1 verified byte-identical there, 0.8.1 +2.1% faster). Bidirectional heads (every
  DeepSpec-native drafter) keep the full-width path, byte-identical.

## [0.8.0] — 2026-08-11 — the first Mamba-2 hybrid target (NVIDIA Nemotron-3.5-Lightning), the first `muse_glimmer` target (Meta Muse-Glimmer-30B), + vLLM speculators-format heads

### Added
- **NVIDIA Nemotron-3.5-Lightning-30B-A3B runs — the first Mamba-2 hybrid target, and the
  project's first non-attention recurrence.** The target is `model_type: nemotron_h` — 52
  interleaved **Mamba-2 + latent-MoE (128 experts, top-6, +1 shared) + full-attention** blocks,
  ~3B active. Two new pieces in `target.py`: a `nemotron_h` hidden-state tap (its backbone hangs
  off `.backbone`/`.embeddings`/`.norm_f`, the cache list is *compacted* to only the M/`*` blocks,
  and the mask is chosen per block type), and an exact **Mamba-2 rollback** — the hybrid
  capture-and-rerun that made rejected drafts free on qwen3_5 gated-DeltaNet, generalized to a
  second recurrence: hook `ssm_update` + `_conv`, then on a partial accept re-run the SSD
  recurrence over the accepted prefix and reslice the conv window (attention layers KV-trim as
  before). The drafter is NVIDIA's official head — a plain qwen3 GQA backbone with DFlash-lineage
  traits (causal sliding-window-1024 block attention, a per-head attention-sink bias fed to SDPA,
  `block_size 8`, `sample_from_anchor=false`, markov-512 fixup) that reuses the target's lm_head
  (`has_lm_head=false`, bound at run time). Measured (M4 Pro, 4-bit target, greedy, warm):
  baseline ~91 tok/s → **cap 4 = 1.27× code / 1.24× math / 1.06× chat** (accept 4.55/4.41/3.74);
  lossless (the divergences from single-row greedy are floating-point / recurrent-state ties, a
  touch wider than a dense target's because the Mamba state drifts between single-row and
  multi-row forwards). Throughput peaks at cap 4 and regresses past it: the MoE verify-width cost
  bounds the ratio, not the drafter, whose acceptance is strong.
- **Meta Muse-Glimmer-30B runs — the first `muse_glimmer` target.** A multimodal, DENSE ~30B
  model (3:1 sliding/full attention, NoPE global layers, `output_multiplier` + logit softcap),
  loaded via **mlx-vlm ≥ 0.6.12** (the floor was bumped for its module). Unlike gemma4, its
  language model ships no `capture_layer_ids` hook, so `target.py` replicates its text forward for
  the hidden-state tap (per-layer sliding/full masks reproduced from the model's own mask builder,
  so the windowing is correct; `verify_tap` proves it faithful). The drafter is a community DSpark
  head by **DaoCloud** in the vLLM speculators packaging, warm-started from Meta's DFlash assistant
  — a 5-layer qwen3 GQA backbone with **causal sliding-window-2048** block attention (the
  `_translate_speculators` step now carries the causal-SWA markers when the backbone declares
  `sliding_attention` layer types) that reuses the target's `embed_tokens` **and** `lm_head`
  (`block_size 15`, `sample_from_anchor`, full 202048 vocab, markov + confidence). It is the first
  head to reuse *both* embed and head, detected from the checkpoint (the weights are simply absent);
  `has_own_embed` + `bind_embed` were added alongside the existing `has_own_lm_head` reuse. The
  reused proposal head is the **raw** `lm_head` — the target's `output_multiplier` (~0.196) + softcap
  are the verifier's output transform, not part of a draft proposal; applying them shrinks the base
  logits so the raw-scale markov bias overwhelms them and acceptance silently collapses (d0 ~85% →
  ~29%). Measured (M4 Pro, 4-bit target, greedy, warm, paired): baseline ~14 tok/s → **cap 2 (auto)
  = 1.50× code / 1.50× math / 1.40× chat** (accept ~2.5); cap 4 lifts accept to ~3.3 at ~1.4×.
  Lossless (muse's compressed logits make fp near-ties more frequent than a dense model, but every
  divergence margin is sub-ulp).
- **Muse-Glimmer-30B **8-bit** target measured — the ratio ~doubles.** An 8-bit MLX build now
  exists (`mlx-community/Muse-Glimmer-30B-8bit`, 33 GB). Its verify curve is flat to width 5 then
  knees at 6 (vs 4-bit's knee at 3), so auto-cap picks **4**, and 8-bit sits closer to the drafter's
  BF16 training verifier — together those lift baseline 8.2 tok/s → **cap 4 = 2.44× code / 2.04×
  math / 1.74× chat** (accept 3.58/3.00/2.51; decode-only, prefill removed, reaches 2.53× on code),
  vs 4-bit's 1.50×/1.50×/1.40× at cap 2. Lossless (cap 2 and cap 4 diverge from single-row greedy at
  the *same* fp-tie position). The trade-off: 8-bit decode reads ~2× the weight bytes per token, so
  *absolute* throughput is ~parity with 4-bit on code (~22 tok/s) and lower on math/chat — the better
  ratio buys 8-bit **quality at ~4-bit speed**, not raw speed, and peaks at **~40 GB RAM** (fits 48
  but tight). The registry default target stays 4-bit (~18 GB, broad-RAM); the same drafter
  auto-resolves for either quant. bf16 (~60 GB) still does not fit 48 GB.
- **Muse-Glimmer channel + tool parsing when serving Claude Code / OpenAI clients.** muse emits
  Meta's "Onyx ATEM" harmony format — recipient-tagged sub-messages (`to=self<|message|>…<|eom|>` =
  analysis/reasoning, `to=user<|message|>…` = the final answer, `to=<tool><|message|><atem:invoke
  name="X">…</atem:invoke>` = a tool call) — not the `<|channel|>` gpt-oss syntax. Left unparsed the
  raw markers leaked into the assistant text and tool calls never ran. `split_thinking` +
  `MessageStream` now recover the channels (analysis → a `thinking` block / `reasoning_content`,
  answer → text) via a new incremental `MuseChannelParser`, and `tools.py` parses the `<atem:invoke>`
  syntax (value grammar taken from the tokenizer's own `response_template`: JSON with a raw-string
  fallback). Both endpoints, streaming and non-streaming. Validated end-to-end on the 8-bit model
  (reasoning→answer and a real `get_weather` call parse clean; no marker leakage). Server-detected by
  model type (`Engine.is_muse`) so non-muse output is byte-identical; +12 model-free tests.
- **modelopt NVFP4 drafters are auto-decoded to bf16** (`nvfp4_convert.py`). NVIDIA ships the
  Nemotron head as `W4A16_NVFP4` (packed FP4/E2M1 + FP8/E4M3 block scales + an FP32 per-tensor
  scale), which mlx-lm has no loader for; `load_drafter` hand-decodes it to bf16 on first use and
  caches it, exactly as the PrismML GGUF drafters are converted. The dequantized head is published
  at `mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-DSpark-bf16` and registered against the
  `mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit` target (the MIXED_PRECISION NVFP4
  target — FP8 Mamba proj + NVFP4 experts — was converted from the BF16 source with `mlx_lm.convert`).
- **vLLM `speculators`-format DSpark heads now load.** This packaging (`speculators_config`,
  `transformer_layer_config`, `aux_hidden_state_layer_ids`) was previously refused by name. It
  turns out to be a *schema rename* rather than a different model: the tensor names are already
  the DeepSpec ones (`fc` / `hidden_norm` / `layers.N.*` / `markov_head.markov_w{1,2}` /
  `confidence_head.proj`) and the 5-tap fusion layout matches every head already shipped, so the
  config is translated on load and the existing drafter path runs unchanged. Verified against
  `makora-ai/gemma4-26b-a4b-dspark` and `mgoin/Qwen3-8B-speculator.dspark`. Only the `dspark`
  algorithm is translated; eagle/eagle3 heads are refused by name, and a head whose
  `transformer_layer_config` is too sparse to rebuild a backbone is refused with the missing
  fields listed.
- **Reduced draft vocabularies (EAGLE-3 style).** These heads emit logits over a frequent-token
  subset (`draft_vocab_size`, e.g. 32000 against a 262144-token target) and ship a `d2t` table
  mapping a draft id back to a target id. The split is asymmetric: `lm_head` and `markov_w2`
  work in draft space, while `embed_tokens` and `markov_w1` stay target-indexed because they
  consume previously *accepted* tokens. `d2t` is an offset table (`target = draft + d2t[draft]`),
  and both directions of config/weight mismatch are refused rather than silently mis-decoded.
  Temperature sampling widens the draft distribution onto the target vocabulary before the
  accept test, so the reject residual still runs over the full vocab and losslessness holds.
- **`mlx-dspark benchmark --no-lookup-drafts`.** The benchmark could not disable the 4-gram
  *hybrid* lookup draft that dspark mode runs by default — it exposed only `--modes lookup`, the
  unrelated standalone drafter-free mode — so every dspark row it had ever printed had the hybrid
  silently on, with no way to tell from the invocation. The setting is now also printed in the
  run header. This matters most on MoE targets, where it is measurably a net loss.
- **`gemma-4-26B-A4B` runs** via `--drafter makora-ai/gemma4-26b-a4b-dspark` against
  `mlx-community/gemma-4-26b-a4b-it-8bit`. Measured on an M4 Pro (median of 3, `--max-draft 2
  --no-lookup-drafts`): baseline **46.9 tok/s** → **59.5 tok/s (1.27×)**, accept 2.36 — 1.38×
  code / 1.37× math / 1.06× chat. Not registered for auto-resolution: the registry tracks pairs
  vouched for in the README table, and this ratio is still under review.

### Fixed
- **`--max-draft auto` crashed on a drafter that reuses the target's lm_head.** The auto-cap
  calibration measures the drafter's own cost curve by calling `compute_logits` directly, which
  fails on a `has_lm_head=false` head (the Nemotron drafter) because the target head is only bound
  at generation time. `calibrate()` now binds it first; no other drafter is affected.
- **DFlash-derived heads read their first prediction from the wrong block slot.**
  `DSparkSpeculatorConfig` subclasses `DFlashSpeculatorConfig`, so a `speculators` head reserves
  block slot 0 as a pure anchor and predicts from slot 1 — where DeepSpec heads use
  anchor-as-pos0. Reading slot 0 re-predicts the token already known, putting every draft one
  position late; acceptance roughly halves (**1.50 → 3.10 at cap 4**) with no error raised and no
  visible defect in the output, because the target verifies every token. Now a config field
  (`logits_start`, derived from `block_size − speculative_tokens`) applied through one helper.
  DeepSpec heads are unaffected — the gemma-4-12B pair is byte-identical before and after.
- **…and the same derivation then rejected the opposite convention.** A `speculators` head that
  proposes as many tokens as it has block slots has *no* anchor slot, so `logits_start` must be
  0 — but the guard read `0 < block_size − speculative_tokens`, so such a head silently fell
  back to 1 and drafted one position *early*, at the same invisible cost (measured on one such
  head: accept 1.77 vs 2.63, 0.88× vs 1.30× — a net slowdown rather than a working drafter). The
  newer `sample_from_anchor` field is honored as a fallback, but only when actually present: the
  pydantic class defaults it to `True` while the heads that omit it do reserve an anchor slot.

### Notes
- Two results from the gemma-4-26B-A4B work generalize past this model. **Hybrid lookup drafts
  are a net loss on MoE targets** — 1.27× off vs 1.22× on here, matching Qwen3.6-35B-A3B's
  1.27→1.21× — because a free draft still costs verify rows and each extra row pulls fresh
  routed experts. And **drafter cost is not what limits an MoE ratio**: this drafter is roughly
  8× better proportioned to its target than Qwen3.6-35B-A3B's (~6% of a target step vs ~50%),
  yet both land near 1.3×. Acceptance here climbs monotonically to 3.25 at cap 6 while throughput
  peaks at cap 2, so only something that flattens the verify curve can move the number.
- The **confidence head does not help on this target** (parity with a plain cap at best), unlike
  Qwen3.6-35B-A3B where it was a win. Adaptive truncation needs acceptance *variance* to exploit:
  that target swung 2.8→7.0 across content, this one sits in a 2.36–3.25 band where a fixed cap
  is already near optimal.

### Changed
- **The OpenAI `/v1/chat/completions` endpoint now surfaces reasoning as `reasoning_content`**
  instead of leaving the model's `<think>…</think>` (Qwen3) / thought-channel (Gemma-4) / analysis
  channel (muse) markup inline in `content`. This unifies it with the Anthropic endpoint (which
  already split reasoning into a `thinking` block) and matches the DeepSeek/vLLM convention that
  most OpenAI clients understand. Non-reasoning responses are unchanged (no reasoning → no
  `reasoning_content` field); tool calls now parse from the answer channel, not the reasoning.

## [0.7.0] — 2026-07-26 — faster prompt processing, the first MoE target, and batching for hybrids

### Added
- **Qwen3.6-35B-A3B is supported — the first Mixture-of-Experts target**, via the community
  drafter `Koopah/Qwen3.6-35B-A3B-NVFP4-DSPARK` (DeepSpec-standalone, block size 8, 8 tap
  layers). `--model mlx-community/Qwen3.6-35B-A3B-4bit` resolves it automatically. It needed
  **no model code**: the `qwen3_5_moe` route, the hybrid linear-attention tap and the
  capture-and-rerun rollback all applied unchanged, and the drafter's tensor names matched 1:1.
  Measured on an M4 Pro at 4-bit (median of 5, `--confidence-threshold 0.3 --no-lookup-drafts`):
  baseline **86.9 tok/s** → **114.5 tok/s (1.32×)**, accept 4.72 — 1.67× math / 1.24× code /
  1.05× chat. Lossless as always (divergences from sequential greedy are fp ties, margins
  0.000–0.500, and code/math land at the same position under both thresholds).
  Three results here are specific to MoE and worth reading before tuning it: the shipped-on
  hybrid **lookup drafts are a net loss** (use `--no-lookup-drafts`), the **confidence head is
  a win for the first time** in this project, and the **wide-GEMM prefill lever barely applies**
  (1.03×) because expert weights are `SwitchLinear`, not `QuantizedLinear`. The modest ratio is
  structural rather than a weak drafter: only ~3.8B parameters are active per token, so the
  baseline is the fastest in the table and a target step (~11.5 ms) is barely twice the cost of
  the 1.53B *dense* drafter round (~5.7 ms). 4-bit is the registered quant — it is what the
  drafter was trained against (NVFP4) and where an already-fast MoE belongs.
  Head-to-head against `z-lab/Qwen3.6-35B-A3B-DFlash` on the same target: DFlash **out-drafts
  DSpark on every prompt** — 9.62 accepted tokens/round on math at full block, the highest
  acceptance measured in this project — and still loses on throughput (1.11× at block 8, 0.94×
  at full 16, vs DSpark's 1.33×), because this target's verify cost rises from the very first
  extra row. Acceptance is not the objective; acceptance per unit of verify width is.
- **Hybrid targets can be batched.** `--max-batch N` previously required a model whose every
  layer holds a plain KV cache, which excluded every qwen3_5-family target (Ornith, Bonsai,
  Qwen3.6-27B, Qwen3.6-35B-A3B) because most of their layers hold recurrent linear-attention
  state. That state turns out to be the easy case: it is a fixed-size summary rather than a
  per-token buffer, so rows of **different prompt lengths merge by plain concatenation** — no
  padding, no per-row offsets, no mask — and only the minority attention layers need the
  left-aligned per-row cache that already existed. Measured aggregate throughput over 8 varied
  prompts on an M4 Pro: **Ornith-1.0-9B 3.52× at B=4**, **Qwen3.6-35B-A3B 2.11× at B=8**
  (dense Qwen3-4B, already supported, 4.00× at B=8). Batched *speculative* decoding remains
  dense-only — a spec round rolls each row back by a different amount, and recurrent state has
  no per-row trim — so a hybrid target batches its baseline and takes the serial path for
  dspark rather than silently degrading.
- **Prompt processing is 1.07–1.15× faster, bit-identically, on every target.** Every previous
  performance pass in this project was a decode pass, but a long prompt — an agent system
  prompt, a pasted file, a long chat — spends most of its wall clock in prefill. Two changes:
  (1) **stop computing prefill logits that are thrown away.** The full `lm_head` ran on every
  prefill chunk, producing `[1, chunk, vocab]`, while every caller reads only the last row —
  ~7–8% of prefill FLOPs and a 622 MB transient that was the reason prefill chunks had to be
  small. Skipping it on non-final chunks is unconditional and bit-identical.
  (2) **dequantize wide weights once, then GEMM** (new `wide_gemm.py`). `quantized_matmul`
  re-dequantizes each weight tile per output row-block, which is free at decode widths and pure
  overhead at prefill widths. Above a calibrated width mlx-dspark dequantizes once instead.
  Both are on by default; `--wide-gemm-min N` forces the width, `0` disables it.
- **Multi-turn chat can skip prefill entirely on targets that previously could not cache at
  all.** Prefix caching used to require a trimmable KV cache, which ruled out every hybrid
  target (Ornith, Bonsai, Qwen3.6-27B) and gemma-4 once its sliding window wraps. Reuse never
  actually needed trimming, only a snapshot at a boundary — so the caches are now snapshotted at
  the prompt boundary and reused whole when a later prompt extends it, which is exactly the
  agent access pattern. Measured **5.5× on turn 2** (Ornith-1.0-9B, 2420 of 2483 tokens reused,
  1.15 s vs 6.30 s), with the reused generation token-identical to a cold one. The mode is
  automatic: forced for targets that cannot trim, and latched on for gemma-4 the first time a
  store is refused, so dense targets never pay the snapshot.

### Fixed
- **`batch_generate_baseline`'s docstring promised a guarantee the code does not make.** It
  claimed each row of a batch reproduces the same prompt run at B=1 bit-for-bit. What is
  actually guaranteed is *row isolation* — rows never influence each other (an all-identical
  batch produces exactly identical rows, max spread 0.0) — but a batched matmul takes a
  different kernel path than a single-row one, so near-ties can land the other way. Measured on
  the shipped dense path: 2 of 4 rows diverge from their single-stream run within 96 tokens on
  Qwen3-4B-8bit. MoE targets amplify it roughly 4×, because top-k expert routing is a discrete
  function of the hidden state and an ulp can change which experts a token visits. The
  docstring now states the real contract, which is the one `batch_spec_generate` already
  documented.
- **DFlash drafters that nest `block_size` under `dflash_config` now load.** z-lab's newer heads
  (`Qwen3.6-35B-A3B-DFlash`) put every DFlash-specific field inside `dflash_config`, while the
  older ones (gemma-4, Qwen3-4B) put `block_size` at the top level. `load_dflash` already read
  `target_layer_ids` and `mask_token_id` from either place but not `block_size`, so a newer head
  died with a bare `KeyError: 'block_size'`. Both layouts now parse, and a config with the field
  in neither place fails with that sentence instead of a traceback.
- **A drafter config's `partial_rotary_factor` is no longer taken at face value.** DeepSpec's
  reference trainer builds the drafter config as a `deepcopy` of the *target's*, so every
  DeepSpec-stock head for a Qwen3.5/3.6-family target inherits `partial_rotary_factor`,
  `mrope_*` and gated-attention fields that describe the target and not the drafter. The
  reference drafter ropes the full `head_dim` regardless (its rope init keys off `head_dim`
  alone, and its `apply_rotary_pos_emb` multiplies `q` at full width — a quarter-width `cos`
  would not broadcast), so honoring the field roped a quarter of each head. There was no error
  anywhere — the weights key-match 1:1 — only lost acceptance: **1.18 → 3.05** on the
  Qwen3.6-27B head and **1.29 → 1.59** (code) / **1.36 → 1.78** (chat) on the Qwen3.5-0.8B one.
  Partial rotary is now honored only for qwen3_5-*native* drafters (Ornith's
  `Qwen35DSparkModel` lineage, where it is real); no previously-working checkpoint changes how
  it parses. Output was lossless before and after — this is speed, not correctness.

### Changed
- **Qwen3.6-27B now resolves `satgeze/Qwen3.6-27B-DSpark` on the 8-bit target**, replacing the
  drafter/target pair the row shipped with in 0.5.0. It is a block-15 head (vs 7 everywhere else)
  trained against the bf16 target with DeepSpec's online mode and warm-started from z-lab's
  DFlash head for the same target. Measured on an M4 Pro at cap 4 (median of 3): baseline 8.4 tok/s →
  **19.2 tok/s (2.29×)**, accept 3.15 — 2.26× chat / 1.96× code / 2.67× math. Lossless as always
  (divergences from sequential greedy are fp ties: margins 0.125/0.250, same position under both
  caps). `--model mlx-community/Qwen3.6-27B-4bit` resolves the same drafter; that pairing is not
  measured.

### Known limitations
- **Turn-2 prefix reuse depends on the model's chat template, not just its cache type.** Reuse is
  all-or-nothing at the snapshot boundary, so the next turn's prompt must extend the previous
  one exactly — which requires the template's generation-prompt suffix to be a prefix of how it
  renders a *completed* assistant turn. Templates that prefill a `<think>` opener break that by
  a few tokens and get no reuse: measured, Ornith-1.0-9B reuses under `--no-thinking` but
  Qwen3.6-27B misses by 2 tokens with thinking on and 4 with `--no-thinking`. A miss costs
  nothing (measured 1.00×, output identical) — it simply falls back to a full prefill. Targets
  in that state still get the prefill speedups above.

### Not added
- `satgeze/Qwen3.5-0.8B-DSpark` runs correctly and losslessly but is **not** registered: on a
  target this small speculation does not pay (best **0.96×**, on the 8-bit target; 0.76× at
  4-bit, 0.74× at bf16). The registry is the set of pairs we have measured *and* vouch for. It
  runs with `--drafter`, and the README says so under "bring your own drafter" so nobody spends
  the download to find out.

## [0.6.1] — 2026-07-22 — the default draft cap is measured, not hard-coded

### Changed
- **The default `--max-draft` is now derived from this machine's measured cost curves**
  instead of the hard-coded `2`. On first use of a target mlx-dspark benchmarks the
  verify/drafter curves once (~5 s, cached in `~/.cache/mlx_dspark/`) and picks the cap that
  maximizes expected tokens/second. The old constant was measured when mlx 0.31.2 put the
  quantized-matmul knee at verify width 4; mlx 0.32 moved that knee to width 6 for every
  8-bit family, so the default had been leaving **10–35%** on the table since that upgrade.
  Measured on an M4 Pro (median of 3, cap 2 → derived cap): Gemma-4 12B 2.05×→**2.78×**,
  Ornith-1.0-9B 1.83×→**2.40×**, Qwen3-8B 1.70×→**2.05×**, Qwen3-14B 1.75×→**2.03×**,
  Qwen3-4B 1.65×→**1.82×**, Ternary-Bonsai-27B 1.07× (cap 2, unchanged). No model regresses.
  Output is unaffected — the cap only sets how many drafted tokens are verified per round,
  and the target verifies every one, so this is lossless as before.
  Replacing the constant with `4` would not have worked: the optimal cap is a property of
  (model × **quantization** × chip × mlx version), and one registry row serves several quants
  — Ornith-1.0-9B wants cap 2 at 4-bit, 4 at 8-bit and 6 at bf16. `--max-draft N` still pins
  it; `--max-draft auto` still adapts per round.

- **The MLX wired-memory limit is no longer set by default** — it is now opt-in behind
  `--wired-limit` (`generate` / `serve` / `benchmark`, and `Engine(wired_limit=…)`). Wiring
  the recommended working set (~75% of RAM) was meant to stop weights being paged out on
  small Macs, but wired pages cannot be reclaimed by the OS: with the process already holding
  ~23 GiB it locked up an M4 Pro hard enough to need a power cycle. Short of that it corrupts
  the verify logits during long generations on the **gemma-4/mlx-vlm** route — a ~430-token
  run produced a garbage logits buffer, so `argmax` returned position indices instead of token
  ids. That surfaced as an `IndexError` only by luck; different garbage would have committed
  plausible *wrong tokens* and broken losslessness silently. 3/3 crashes with it, 3/3 clean
  1000-token runs without; mlx-lm-route targets did not reproduce it (Qwen3-14B 1260 tokens,
  Qwen3-8B 1501 tokens, both clean at a higher 22.8 GiB peak). It also bought no measurable
  speed where tested (<1%, inside run-to-run noise). If you are on a Mac where the model
  nearly fills RAM, turn it on deliberately and validate a long run first.

### Added
- **`mlx-dspark benchmark --trials N`** — repeats each prompt and reports the median, plus a
  per-prompt breakdown. It previously averaged its three prompts into a single number, so the
  per-content figures in the README were not reproducible with the shipped harness.
  Between-trial noise on an M4 Pro is ~14%, so single runs should not be quoted.
- **`calibrate.static_cap()`** — the resolver above, public so library users can pick the same
  cap (`speculative_generate` keeps its `max_draft_tokens=2` default: a library call should not
  silently trigger a device benchmark).
- **Qwen3-14B registered** (`mlx-community/Qwen3-14B-8bit` →
  `deepseek-ai/dspark_qwen3_14b_block7`), so `--model` resolves its drafter with no `--drafter`
  flag. It was already listed as supported but raised `ValueError`. Measured 2.03× at cap 4 and
  lossless (cap 2 and cap 4 agree bit-for-bit; both diverge from single-row greedy at one
  floating-point tie, margin 0.25 = 1–2 bf16 ulps).

### Fixed
- **`knee_width()` misreported the verify-cost knee** on any curve with a flat step — which is
  every 8-bit model under mlx 0.32. It tracked the cheap region's marginal cost as a running
  minimum, so one flat step drove the baseline to zero and every later `+1 ms` read as the
  jump (measured Qwen3-4B: reported width 4, where the curve is still flat; the real jump is
  at 6). It also structurally could not report a knee at width 2 — the bf16 shape, where an
  unquantized matmul reads the weight stream twice and is flat afterwards — so on a bf16
  target it returned the top width, implying wide verify was cheap, the opposite of the truth.
  Now the baseline is the *mean* of the cheap region's steps, a jump must also clear 15% of
  the single-row cost (a ratio test on ±1 ms of noise is meaningless), and the first step is
  judged against the rest of the curve. This is reporting only — it feeds `/metrics`,
  `mlx-dspark doctor` and the DSpark-vs-DFlash recommendation; cap selection reads the curves
  directly and was never affected.

## [0.6.0] — 2026-07-21 — Claude Code support (Anthropic Messages API)

### Added
- **`POST /v1/messages` + `POST /v1/messages/count_tokens`** — the same server now speaks
  Anthropic's Messages API alongside the OpenAI one, on the same port. This is what
  **Claude Code** talks, so a model on your Mac can drive it. New module
  `anthropic_api.py` holds the whole translation (request blocks ↔ chat-template messages,
  generated text → content blocks / SSE events) and is pure and model-free, so it's covered
  by unit tests rather than only end-to-end. Streaming is the real path: Claude Code consumes
  SSE as it arrives and stalls on a server that buffers.
- **`mlx-dspark claude`** — launches Claude Code against a running `serve`, configured for
  **that process only**. No shell profile edited, no `settings.json` written, no login
  replaced: other Claude Code sessions (running or future) are unaffected and this one
  reverts on exit. `--print-env` / `--print-settings` emit the same configuration for manual
  or project-scoped wiring. Sets the per-alias `ANTHROPIC_DEFAULT_*_MODEL` variables —
  including the `haiku` slot Claude Code uses for background work — so nothing asks for a
  model the server has never heard of, and drops a stale `ANTHROPIC_API_KEY` /
  `CLAUDE_CODE_USE_*` from the child so ambient config can't override the base URL.
- **`--context-window N`** — cap the prompt length below the target's own limit (RAM budget).
  An over-long request is refused with Anthropic's `prompt is too long` wording, which is
  what Claude Code's automatic compact-and-retry matches on — verified live: it recognised
  the limit and reported it as a context problem instead of a failed request.
- **Reasoning models get real `thinking` blocks.** Qwen3-class targets emit
  `<think>…</think>` inside the generated text; Anthropic carries that as a separate content
  block. It is now split out (streaming included, via a small state machine that defers
  opening block 0 until its type is known) instead of rendering as assistant prose, and
  `thinking: {"type": "disabled"}` maps onto the model's own `enable_thinking` switch so
  disabling it actually makes the model faster rather than just hiding the output.

### Fixed
These were found only by running more than one model and more than one client — each is a case
where one combination tolerated something another doesn't.
- **Gemma-4 no longer runs past its own turn after a tool call** (affects *every* path,
  including the OpenAI server and plain `generate` — not just the new endpoint). After emitting
  a tool call Gemma-4 does not send `<turn|>`; it sends **`<|tool_response>`** to hand back to
  the harness for the tool result, and its own response grammar terminates on either. That
  token was missing from `eos_token_ids`, so generation continued and the model **hallucinated
  the tool result and the rest of the conversation**, burning the whole `max_tokens` budget on
  fiction. A tool-calling agent hits this on every single turn: one measured request produced
  **8192 tokens instead of 14**. Adding the marker fixes it everywhere; other families resolve
  it to unk and are unaffected.
- **Mid-conversation `system` messages no longer fail the request.** Recent Claude Code sends
  operator instructions as `{"role": "system"}` entries *inside* `messages` (an Opus-4.8 API
  feature it also applies to model names it doesn't recognise, i.e. every local server). Most
  chat templates accept a system message only in first position: Qwen3 happened to tolerate it,
  Ornith-1.0-9B raised `System message must be at the beginning` and killed the session. They
  are now folded into the adjacent user turn as a `<system-reminder>` block — the documented
  fallback for models without the feature — and leading ones merge into the system prompt.
- **The XML tool-call format is parsed** (`<tool_call><function=NAME><parameter=K>v</parameter>`
  `</function></tool_call>`), used by Ornith-1.0 and several other models. Previously it fell
  through every parser and the raw markup was returned as assistant text, so no tool ever ran.
  Values are raw text with no type information, so `tools.schema_types()` feeds the request's
  own tool schemas in and each value is coerced to its declared type; without a schema only
  short single-line scalars are touched, since a multi-line value is essentially always a
  string (file contents, code) and coercing one would corrupt it. A call truncated at
  `max_tokens` is now recovered rather than dropped for a missing suffix.
- **Gemma-4's `<|channel>thought … <channel|>` reasoning no longer leaks into the response.**
  Its template prefills an empty thought channel *except* after a tool response, so the markers
  only appear mid-agent-loop — invisible in plain chat, visible in every Claude Code session.
  The pair is now split out into a `thinking` block like Qwen3's `<think>`; the marker grammar
  is taken from the model's own `response_schema.x-regex`, not guessed.
- **Routing ignored the query string**, so Claude Code's `POST /v1/messages?beta=true` would
  have 404'd. Routes now match on the path.
- **Auth accepts `x-api-key`** as well as `Authorization: Bearer` — which header carries the
  credential depends on whether the client was configured with `ANTHROPIC_AUTH_TOKEN`,
  `ANTHROPIC_API_KEY`, or an `apiKeyHelper` (a helper sends both).
- `HEAD /` (Claude Code's startup connectivity probe) returns 200 instead of 501.

### Notes
- Unknown request fields are **ignored, never rejected** (`thinking`, `context_management`,
  `output_config`, beta tool-schema fields, `metadata`, …). Claude Code's field set grows every
  release and it sends the newest fields to any endpoint whose model name it doesn't
  recognise — which is every local server — so a strict parser here would break on a Claude
  Code release that doesn't exist yet.
- **79 new model-free tests** (264 total, ruff-clean). Verified end-to-end on an M4 Pro across
  **three model families** — each ran a real Claude Code session that read a file and fixed a
  bug in it with the `Edit` tool, with zero server errors:

  | target | tool syntax | accept | prefix cache |
  |---|---|---|---|
  | `Qwen3-8B-8bit` | Hermes JSON | 3.01 | on (2/3 requests, ~26k tokens reused) |
  | `Ornith-1.0-9B-8bit` | XML `<function=>` | **5.07** | off — hybrid target, state can't be reused |
  | `gemma-4-12B-it-8bit` | Gemma `<\|tool_call>` | 3.68 | on, but the sliding window wraps at this prompt size |

  Prefill is the wall-clock cost on all three (Claude Code's prompt is ~18–26k tokens per
  request), so the target that *reuses* it finishes fastest even with a lower accept length —
  Qwen3-8B ~2:20 vs ~4:10 for the other two on the identical task.
- **Also verified with [pi](https://github.com/earendil-works/pi-mono)** (`pi-coding-agent`
  0.80.10), a second, independent agent client, against *both* the Anthropic and OpenAI
  endpoints via its `models.json` custom-provider config. pi's system prompt is ~1.5k tokens
  against Claude Code's ~18–26k, and on a local model that difference is everything: the same
  bug-fix task runs in **~6 s instead of ~2:20**, and a 4-tool task (read, two edits, read,
  write) completes in **8.5 s** at 24 tok/s on Qwen3-8B. Ornith runs it in 18.8 s. Gemma-4-12B
  does not converge on pi's tool protocol — on *both* endpoints, so it's a model/agent-protocol
  mismatch rather than a server issue; it works fine with Claude Code.

## [0.5.1] — 2026-07-21 — diagnosable server errors + dependency refresh

### Fixed
- **Server 500s are diagnosable again** (issue #5). A failed generation returned
  `generation failed: <str(e)>` and *discarded the traceback*, so an intermittent
  per-request error (the reporter saw one `list index out of range` in 461 requests)
  left no record of where it came from. The handler now logs the full traceback to
  stderr and names the exception type in the response body.
- **Prefix cache: stored slots hold exactly their token record.** A speculative round
  commits a whole block, so a generation ending on an eos that lands *mid-block* left
  the target KV cache (and drafter ctx) holding rows for tokens dropped from
  `token_ids`. Reuse trims by absolute offset, so this was harmless in practice, but it
  wasted KV and made the class's central invariant only accidentally true.
  `PrefixCache.store()` now normalizes both caches down to the record. Baseline mode
  can't reach this case (it commits one token per step).

### Changed
- **Dependencies: mlx-vlm 0.6.3→0.6.6, transformers 5.12.1→5.14.1** (dev `.venv` now
  matches latest-resolve). Both target routes verified **bit-identical** before/after —
  mlx-lm/Qwen3-4B and mlx-vlm/gemma-4-12B, greedy ids, speculative ids and accept
  lengths unchanged. The mlx-vlm 0.6.4 `Gemma4UnifiedProcessor` shim self-retires on
  0.6.6 and is kept only because the floor still admits 0.6.4.

## [0.5.0] — 2026-07-18 — community DSpark drafters: Ornith-1.0-9B (chat finally >2×) + Qwen3.6-27B + match-scaled long lookup drafts

### Added
- **Match-scaled long lookup drafts** (`--lookup-long-draft`, default 32; dspark hybrid +
  lookup mode): when the current context matches an earlier occurrence ≥8 tokens deep (a
  real copy run — re-emitting a file, mechanical edits, quoting), the free n-gram draft
  grows to ~2× the matched length instead of the flat 6, up to the ceiling. Verify width
  16–32 is a measured *plateau* on M-series (≈2.5× the cost of a single step, 8-bit,
  M4 Pro), so verbatim spans commit ~20–30 tokens per target forward. Measured (M4 Pro,
  8-bit targets, bit-identical outputs): gemma-4-12B file re-emission **3.03×→4.51×**
  (75 tok/s), rename-edit 3.03×→4.33×; Ornith-1.0-9B rename-edit **2.79×→3.57×**
  (93 tok/s), re-emission 2.18×→2.45×; chat unchanged (the scaling needs deep-match
  evidence a bare 4–5-gram hit doesn't provide). An acceptance gate parks the scaling
  when long drafts keep getting chopped early (insertion-heavy edits measured neutral)
  and probes back in every 8 lookup rounds. Inspired by llama.cpp's `ngram-mod` drafter
  (24-token match context → 48–64-token drafts) after a study of its DFlash n-gram stack.
- **DeepReinforce Ornith-1.0-9B** (qwen3_5 hybrid, agentic coding):
  `--model mlx-community/Ornith-1.0-9B-8bit` auto-resolves the community drafter
  `stanleyphoong/Ornith-1.0-9B-DSpark` (rigorously qualified by its author: 17/17 gates,
  95% of the DSpark paper's reference acceptance). Measured on an M4 Pro at cap 3:
  **2.17× code / 2.44× math / 2.11× chat** (59–69 tok/s vs a 27.9 tok/s baseline) — the
  first target here with chat above 2×; `--max-draft auto` drives the cap to the full
  block of 7 on code. The 4-bit target trades ratio for absolute speed (1.38–1.55× at
  60–76 tok/s); the same drafter resolves for any quant.
- **Qwen3.6-27B**: `--model mlx-community/Qwen3.6-27B-4bit` auto-resolves the community
  drafter `Avesed/Qwen3.6-27B-DSpark` — 1.42× code / 1.78× math / 1.27× chat (M4 Pro,
  cap 2/3). Community-drafter caveats apply: acceptance runs below DeepSeek's official
  drafters, training data is English-centric (Chinese accepts poorly), and the drafter is
  W4A16-native so the **4-bit** target is its matched precision — an 8-bit target was
  measured and *lowers* acceptance (auto-cap still reaches ~2.1× against the slower 8-bit
  baseline, but 4-bit is faster in absolute tok/s everywhere).
- **qwen3_5-flavored drafter backbones** (what Ornith needed), config-driven on the
  existing qwen3 family: gated q_proj (per-head [q ‖ gate], attention output ×
  sigmoid(gate)), partial rotary (`rope_dims`), and **offset RMSNorm weights** — qwen3_5
  checkpoints store every norm weight as an offset from one ((1+w)·x̂); `load_drafter`
  materializes the +1 at load. Loading them as plain scales silently collapses acceptance
  to ~1.25 with no error anywhere, which is exactly how it was found.

### Changed
- Drafter-precision guidance, confirmed from both directions this release: **run the
  target at the precision the drafter was trained against** (Ornith bf16-qualified →
  8-bit target; Avesed W4A16 → 4-bit target). The README pick tables reflect it.

### Fixed
- Drafter auto-resolve no longer re-downloads a model that already exists in the plain-dir
  cache (`~/.cache/mlx_dspark/models/<repo basename>`) — previously `--model` with a local
  target path still pulled the registry drafter from the Hub (6.1 GB for Ornith).

## [0.4.3] — 2026-07-17 — fresh installs work again with mlx-vlm 0.6.5

### Fixed
- **`import mlx_dspark` crashed with mlx-vlm 0.6.5**
  (`ModuleNotFoundError: No module named 'mlx_vlm.models.gemma4.rope_utils'`): 0.6.5
  consolidated per-family rope utilities into `mlx_vlm.models.rope_utils`, and `model.py`
  imported the old path at module scope. Because `mlx-vlm` is floored but not capped, every
  **fresh** `pip install mlx-dspark` since 0.6.5's release hit this at import time (existing
  environments with mlx-vlm ≤ 0.6.4 were unaffected). Both module layouts are now supported;
  the test suite passes against mlx-vlm 0.6.3 and 0.6.5.

### Changed
- The 1-bit checkpoint refusal in `load_target` now reflects the current landscape: mlx-vlm
  ≥ 0.6.5 can run 1-bit affine packs standalone via its own Python-hosted kernel (e.g.
  `prism-ml/Bonsai-27B-mlx-1bit` — measured 34.9 tok/s baseline on an M4 Pro, 1.37× the
  ternary 2-bit), but speculative decoding measures a net **loss** on that kernel (0.71–0.77×
  at any cap, healthy acceptance — its verify cost is linear in draft length), so mlx-dspark
  keeps the pack unintegrated and the error points at the ternary variant instead.
- README upstream-compat note updated: mlx-vlm 0.6.5 ships the gemma4-processor fix
  ([Blaizzy/mlx-vlm#1578](https://github.com/Blaizzy/mlx-vlm/issues/1578)), so the 0.3.2-era
  shim self-retires there (verified — the shim's gate returns False on 0.6.5).

## [0.4.2] — 2026-07-16 — exact hybrid rollback: rejected drafts no longer cost a replay

### Changed
- **Hybrid (qwen3_5/Bonsai) spec rollback is now exact — rejected drafts no longer cost a
  replay.** `Target.verify()` records references to each linear-attention layer's recurrence
  inputs (scoped pass-through hooks, zero numeric change); on a partial accept `rollback()`
  re-runs the gated-delta recurrence over just the accepted prefix from the pre-round state
  (bit-exact — the kernel consumes tokens sequentially), restores the conv window by slicing,
  and trims the KV layers by only the rejected tail. Output unchanged (ids byte-identical to
  greedy, validated on-device); speed on mid/low-acceptance content improves a lot because
  partial accepts stop re-forwarding accepted tokens through the whole model: on identical
  prompts (M4 Pro, Bonsai-27B ternary), code at acceptance 2.5/round went 0.87× → **0.99×**
  and chat at 2.2/round went 0.67× → **0.93×** at fixed cap 2; the sharp decay past cap 2 is
  gone (cap 3 now ≈ cap 2); high-acceptance code is unchanged (1.15×, within the old band).
  Applies to drafter and lookup modes alike; dense targets are untouched byte-for-byte.
- The auto-cap controller no longer prices the (removed) hybrid replay into its cap choice,
  so it stops over-penalizing higher caps on hybrid targets; observed round timings continue
  to ground it live. A new regression test guards parked-controller recovery on content
  shifts (the parked probe cadence is load-bearing — an exponential probe backoff was tried
  and reverted after it wedged the chat→code→math sweep parked).

## [0.4.1] — 2026-07-15 — version-string fix

### Fixed
- `mlx_dspark.__version__` (shown by `mlx-dspark doctor` and the smoke-install check) still said
  `0.3.2` in the published 0.4.0 wheel — the module attribute and `pyproject.toml` were bumped
  separately and drifted. The version is now **single-sourced**: hatchling reads it from
  `__init__.py` (`dynamic = ["version"]`), so the two can never disagree again. Package metadata,
  the PyPI listing, and all functionality in 0.4.0 were unaffected — the attribute was cosmetic.

## [0.4.0] — 2026-07-15 — PrismML Bonsai-27B: speculative decoding for a ternary 27B on a Mac

### Added
- **PrismML Bonsai-27B support — first speculative decoding for a hybrid linear-attention
  target on Apple Silicon.** `mlx-dspark generate --model prism-ml/Ternary-Bonsai-27B-mlx-2bit`
  auto-resolves the matching DSpark drafter, which PrismML ships
  **GGUF-only** (`*-dspark-bf16.gguf` in the `*-gguf` repos, no safetensors export exists):
  we republished a 1:1 bf16 repack in the DeepSpec layout
  ([`Rahim/Ternary-Bonsai-27B-dspark`](https://huggingface.co/Rahim/Ternary-Bonsai-27B-dspark),
  Apache 2.0, converted with the new `gguf_convert.py`); the `gguf:{repo}/{file}.gguf` drafter
  scheme converts any future GGUF-only drop locally the same way
  (cached under `~/.cache/mlx_dspark/drafters/`). Lossless (ids byte-identical
  to greedy, fp ties excepted); measured on M4 Pro 48 GB: **1.1–1.2× on code/structured**
  (~25 → 28–30 tok/s; interleaved medians 1.11×, best fresh runs 1.2×), acceptance ~2.8/round
  at cap 2 (~0.93/token — the drafter is strong; the 2-bit verify slope is the limiter). Their
  own CUDA number is 1.34×; their Metal path says "no speedup on Macs yet" — this is, as far
  as we know, the first working one.
- **Hybrid-target verify/rollback** (`Target.verify()` / `Target.rollback()`): Bonsai-27B is
  Qwen3.6 (`qwen3_5`) — 48 of 64 layers are gated-DeltaNet linear attention whose recurrent
  state cannot be trimmed like a KV cache. Each spec round runs as ONE forward over
  `[replay backlog + anchor + drafts]` with the linear caches' state arrays snapshotted **by
  reference** (MLX arrays are immutable, so the snapshot copies nothing): full accept keeps
  everything, partial accept restores the refs + trims the KV layers and re-commits the
  accepted tokens inside the next round's forward. Dense targets keep the prior trim behavior
  byte-identically. The tap replicates mlx-lm's qwen3_5 hybrid forward (per-layer fa/ssm
  masks), proven bit-faithful by the existing `verify_tap()` probe at load. Baseline and
  `--mode lookup` work for **any** qwen3_5 model (e.g. Qwen3.5-0.8B verified).
- **GIDD log-SNR conditioning** (`model.LogSnrEmbed`): the Bonsai drafters extend DeepSpec's
  DSpark architecture with a sinusoidal noise-level embedding (anchor=max, masks=min →
  fc1→silu→fc2, added to the block embeddings). The inference pattern is a pure function of
  block position, so the addend is computed once and cached. Ported 1:1 from PrismML's
  `dspark.cpp`; excluded from drafter quantization to match their packaging.
- **`--max-draft auto` can now park speculation (cap 0)** when live acceptance is too low for
  the machine's verify slope — on a 2-bit target every extra verify row costs ~40% of a full
  step, so open chat (per-token acceptance ~0.8) is a structural net loss no positive cap
  fixes. Parked stretches run as a **pipelined plain-step sprint** (greedy-loop style, tap
  riding along so the drafter context stays current) with a cap-1 probe round every 8th round
  to keep the acceptance estimate tracking. The controller now also feeds back **observed
  round times** (measured ms/token per cap overrides the synthetic cost model once a cap has
  data; a live correction factor covers the rest) and prices the hybrid replay cost —
  fixing the cold-start park (min-observations gate + running-mean warmup for the EWMA) and
  a calibration gap where the verify curve never measured width 1.
- 20 new model-free tests (GGUF converter round-trip/refusals, log-SNR numerics vs the
  reference formula, hybrid verify/rollback equivalence against sequential processing on a
  tiny real qwen3_5, controller parking/probing/observed-override, routing, registry) —
  **161 total**, ruff-clean.

### Changed
- **mlx floor raised to 0.32.0** (from 0.31.2): 0.32's quantized-matmul kernels add multi-row
  fast paths that this feature depends on — on 0.31.2, 2-bit qmm cost ~+100%/row from M=1
  (no flat region), making Bonsai speculation a wash at any cap; on 0.32 it's ~+27%/row.
  Free bonus on the existing presets (same outputs, fp ties excepted): Qwen3-4B dspark
  70–76 → **84 tok/s (1.64×)**, gemma-4-12B **1.73× → 2.11×** on this M4 Pro.
- `_route_target`: `qwen3_5`/`qwen3_5_moe` checkpoints route to mlx-lm text-only despite
  their `vision_config` (mlx-lm's module drops the vision tower in `sanitize`); only the
  mlx-lm path has the replicated-loop tap the drafters need. `Target` now detects mlx-vlm
  by module origin instead of a `language_model` attribute (mlx-lm's qwen3_5 has one too).
- README results table refreshed on mlx 0.32.0 (all rows re-measured this session, same
  M4 Pro, identical outputs): gemma-4-12B 2.11×, Qwen3-14B 1.92×, Qwen3-8B 1.90×,
  Qwen3-4B 1.64×, Ternary-Bonsai-27B 1.11× (auto).
- **The DSpark-vs-DFlash pick changed with mlx 0.32** and the README's guidance now says so:
  0.32's kernels made narrow multi-row verify disproportionately cheaper, so DSpark cap-2 now
  beats DFlash full-block even on Gemma-12B code (2.11× vs 1.63× spot-checked; on 0.31 DFlash
  led there ~2.1× vs ~1.9×). The Qwen3-8B "full block is a net loss" verdict still holds
  (0.86×). The old multi-prompt sweep tables remain in the deep dive, stamped mlx-0.31.2-era;
  `--max-draft auto` re-measures the curves per machine/mlx so the pick stays current.

### Known limitations
- Bonsai speculation is content-dependent: 1.1–1.2× on code/structured output, a net loss on
  open chat at any fixed cap (0.76× at cap 2 — the 2-bit verify slope + a 3.6B drafter leave
  no margin at ~0.8 acceptance). **`--max-draft auto` is the recommended setting**: interleaved
  medians show it matching the best fixed cap on code (1.11×) while parking on chat (0.89×,
  vs 0.76× fixed). Prefix caching and batching stay dense-only (recurrent state is not
  trimmable/shareable); `--kv-bits` is refused for hybrid targets.
- **The 1-bit `Bonsai-27B-mlx-1bit` pack does not run on stock MLX** — it is quantized to
  1 bit for PrismML's own MLX fork, and `mx.quantize` supports 2/3/4/5/6/8 bits only.
  `load_target` now refuses any unsupported-bits pack with the real reason (instead of an
  opaque crash deep in mlx-lm), and the 1-bit variant is deliberately not registered. Its
  drafter converts fine (kept out of the registry until the target is loadable).

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
