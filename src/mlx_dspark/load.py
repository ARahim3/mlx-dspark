"""Loaders for the target (Gemma-4 via mlx-vlm) and the DSpark drafter."""

from __future__ import annotations

import glob
import os

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download

from .config import DSparkConfig
from .model import DSparkDrafter
from .target import Target

# The drafter must be paired with the *instruct* target it was trained against, at decent
# precision. Presets below; pick with load_pair("gemma4") or load_pair("qwen3").
PRESETS = {
    "gemma4": {
        "target": "mlx-community/gemma-4-12B-it-8bit",
        "drafter": "deepseek-ai/dspark_gemma4_12b_block7",
    },
    "qwen3": {
        "target": "mlx-community/Qwen3-4B-8bit",
        "drafter": "deepseek-ai/dspark_qwen3_4b_block7",
    },
}
DEFAULT_TARGET = PRESETS["gemma4"]["target"]
DEFAULT_DRAFTER = PRESETS["gemma4"]["drafter"]

# z-lab's original DFlash drafters (block-diffusion; reuse the target's embed/lm_head).
# Same matched-instruct targets as DSpark so the two can be benchmarked head-to-head.
# Other z-lab adapters share the arch and load the same way, e.g.:
#   load_dflash("z-lab/Qwen3-8B-DFlash-b16")  +  load_target("mlx-community/Qwen3-8B-8bit")
DFLASH_PRESETS = {
    "gemma4": {
        "target": "mlx-community/gemma-4-12B-it-8bit",
        "drafter": "z-lab/gemma4-12B-it-DFlash",
    },
    "qwen3": {
        "target": "mlx-community/Qwen3-4B-8bit",
        "drafter": "z-lab/Qwen3-4B-DFlash-b16",
    },
}

# ---------------------------------------------------------------------------------------------
# Model registry — invisible plumbing that auto-resolves the *drafter* for a known *target*.
#
# The interface is standard (like mlx-lm): you pass a real target repo/path as `--model`. This
# table just saves you from also looking up the matched drafter for the handful of known targets;
# for anything else, pass `--drafter`. Matching is quant-agnostic (the drafter matches the *model*,
# not its quantization), so Qwen3-8B-4bit / -8bit / -bf16 all resolve the same drafter. There are
# **no user-facing nicknames** — `id` is only the substring we match against a target repo name.
# `speedup` is the measured headline ratio from the README table (M4 Pro, mlx 0.32, warm,
# vs greedy baseline) — a human string for pickers, not a promise; content and machine move it.
# `lookup_drafts: False` marks pairs whose measured-best configuration runs with hybrid
# n-gram lookup drafts OFF — the shipped default then reproduces the vouched-for numbers with
# no flag (see lookup_drafts_default; an explicit --lookup-drafts/--no-lookup-drafts wins).
REGISTRY = [
    {"id": "qwen3-4b",   "target": "mlx-community/Qwen3-4B-8bit",
     "dspark": "deepseek-ai/dspark_qwen3_4b_block7", "dflash": "z-lab/Qwen3-4B-DFlash-b16",
     "ram": "~8 GB", "speedup": "~1.8×"},
    {"id": "qwen3-8b",   "target": "mlx-community/Qwen3-8B-8bit",
     "dspark": "deepseek-ai/dspark_qwen3_8b_block7", "dflash": "z-lab/Qwen3-8B-DFlash-b16",
     "ram": "~11 GB", "speedup": "~2.1×"},
    # Same official DeepSeek drop and recipe as the 4B/8B entries above; registered
    # 2026-07-22 once it was actually benchmarked (2.03x at cap 4: 2.36x math / 2.11x code /
    # 1.62x chat, baseline 15.3 tok/s). No z-lab DFlash adapter published at this size.
    {"id": "qwen3-14b",  "target": "mlx-community/Qwen3-14B-8bit",
     "dspark": "deepseek-ai/dspark_qwen3_14b_block7",
     "ram": "~19 GB", "speedup": "~2.0×"},
    {"id": "gemma-4-12b", "target": "mlx-community/gemma-4-12B-it-8bit",
     "dspark": "deepseek-ai/dspark_gemma4_12b_block7", "dflash": "z-lab/gemma4-12B-it-DFlash",
     "ram": "~15 GB", "speedup": "~2.8×"},
    # PrismML Ternary-Bonsai 27B (ternary rebuild of Qwen3.6-27B, hybrid linear attention).
    # PrismML ships the DSpark drafter GGUF-only (no safetensors export exists); the repo
    # below is our 1:1 bf16 repack into the DeepSpec layout (converted with gguf_convert.py —
    # the "gguf:{repo}/{file}.gguf" drafter scheme converts any future drop locally the same
    # way). The 1-bit variant (prism-ml/Bonsai-27B-mlx-1bit) is NOT registered: its pack is
    # 1-bit-quantized for PrismML's own MLX fork and stock mx.quantize has no 1-bit mode
    # (load_target refuses it with the reason). Drafters are variant-specific.
    {"id": "ternary-bonsai-27b", "target": "prism-ml/Ternary-Bonsai-27B-mlx-2bit",
     "dspark": "Rahim/Ternary-Bonsai-27B-dspark",
     "ram": "~12 GB", "speedup": "~1.15× code"},
    # Qwen3.6-27B (qwen3_5 hybrid — the full-precision sibling of Ternary-Bonsai above; same
    # 64-layer/5120-hidden shape, same tap layers). Community drafter by satgeze: a
    # DeepSpec-standalone head (architectures ["Qwen3DSparkModel"]) with block_size 15 — the
    # only non-7 block here, and nothing assumes 7. Trained against the bf16 target with
    # DeepSpec online mode, warm-started from z-lab's DFlash head for the same target.
    # Its config carries partial_rotary_factor/mrope/gate fields copied from the TARGET's
    # config (DeepSpec builds the drafter config as a deepcopy); config.py treats them as the
    # noise they are — see the qwen35_native gate there and tests/test_formats.py.
    # 8-bit is the measured target (2.29x at cap 4, accept 3.15); 4-bit resolves the same
    # drafter and by the Ornith pattern should trade ratio for absolute tok/s — unmeasured.
    {"id": "qwen3.6-27b", "target": "mlx-community/Qwen3.6-27B-8bit",
     "dspark": "satgeze/Qwen3.6-27B-DSpark",
     "ram": "~32 GB", "speedup": "~2.3×"},
    # DeepReinforce Ornith-1.0-9B (qwen3_5 hybrid, agentic coding). Community drafter by
    # stanleyphoong — DeepSpec-standalone layout with a qwen3_5-flavored backbone (gated
    # q_proj + partial rotary; handled by the qwen3 config branch's gated_q_proj/rope_dims
    # knobs). Ships calibrated confidence_temperatures (unused by fixed/auto cap modes).
    # 8-bit default (house sweet spot, and this drafter was qualified against the bf16
    # verifier): measured 2.17×/2.44×/2.11× code/math/chat at cap 3 vs 1.38×/1.47×/1.23×
    # on the 4-bit target — 4-bit trades those ratios for ~10-15% more absolute tok/s.
    {"id": "ornith-1.0-9b", "target": "mlx-community/Ornith-1.0-9B-8bit",
     "dspark": "stanleyphoong/Ornith-1.0-9B-DSpark",
     "ram": "~13 GB", "speedup": "~2.4×"},
    # Qwen3.6-35B-A3B — the first **MoE** target here (qwen3_5_moe: 40 layers, 30 linear +
    # 10 full attention, 256 experts top-8, ~3.8B active). Community drafter by Koopah:
    # DeepSpec-standalone, block_size 8, 8 taps, plain dense-qwen3 backbone; loaded with zero
    # model-code change (the qwen3_5_moe route and the hybrid tap/rollback already existed).
    # **4-bit is the registered quant on purpose.** The drafter was trained on-policy against
    # unsloth/Qwen3.6-35B-A3B-NVFP4, and 4-bit is also where an MoE target belongs here: only
    # ~3.8B params are active, so the baseline is already ~87 tok/s and 8-bit would halve that
    # to buy ratio. Measured best 1.32x (see README) — modest, and the reason is structural
    # rather than a bad drafter: acceptance is high (up to 7.0/round on math), but a target
    # step is only ~11.5 ms while the 1.53B DENSE drafter costs ~5.7 ms of it. See NOTES
    # "Qwen3.6-35B-A3B: the first MoE target".
    {"id": "qwen3.6-35b-a3b", "target": "mlx-community/Qwen3.6-35B-A3B-4bit",
     "dspark": "Koopah/Qwen3.6-35B-A3B-NVFP4-DSPARK", "lookup_drafts": False,
     "ram": "~23 GB", "speedup": "~1.3×"},
    # Qwen3.8-27B (qwen3_5 hybrid — same 64-layer/5120-hidden 48-linear/16-full shape class as
    # Qwen3.6-27B, new 248320-token vocab). Two community drafters, one per quant, each matched
    # to the precision it was trained against:
    #
    # **4-bit -> DimInfer/Qwen3.8-27B-Dspark-v1** (measured 2026-08-18; beats RadixArk here).
    # A DeepSpec-stock `Qwen3DSparkModel` (NOT SpecForge — no dflash_config/projector_type):
    # 5-layer ungated qwen3 GQA (q_proj [4096,5120]; `attn_output_gate:true` in its config is
    # deepcopy-of-target noise), plain rope (no YaRN), **block_size 15** sampled anchor-as-pos0
    # (logits_start 0), tap layers [1,16,31,46,61] (deeper than RadixArk's), markov-256 +
    # confidence, reuses the target's embed AND lm_head. Trained for the Q4_K_M / 4-bit class.
    # Paired vs RadixArk (M4 Pro, warm, 3-trial medians, 200 tok, small-M kernel on, lookup off):
    # **`--max-draft 7` = 1.99x mean** (chat 1.51x / code 2.14x / math 2.31x, accept
    # 3.28/4.86/5.32; ~23/32/34 tok/s) vs RadixArk's cap7+conf0.3 = 1.82x same session — higher
    # acceptance at every cap/content. The confidence head does NOT pay here (already-high
    # acceptance -> conf-truncation just sheds accepted tokens), so no --confidence-threshold;
    # and block-15 buys nothing past cap 7 (cap 8 = 1.18x — verify width 9 exits the small-M
    # kernel window M in [6,8]). static_cap picks **7** unaided here (its block-15 backbone cost
    # amortizes at high cap where RadixArk's block-7 got 2), so a no-flag `--model` already
    # lands the 1.99x — `--max-draft 7` is just explicit. Greedy-lossless (firstdiff=-1 vs
    # single-row greedy). Loaded with zero model-code change.
    #
    # **8-bit -> RadixArk/Qwen3.8-27B-DSpark** (kept — DimInfer is 4-bit-class, not measured at
    # 8-bit). The first **SpecForge/SGLang**-packaged head here: DFlash-backbone DSpark, block_7
    # anchor-as-pos0, YaRN rope (factor 32 / orig 8192, honored), reuses embed AND lm_head,
    # trained vs the FP8 verifier so 8-bit is its matched precision (accept 2.44 -> 3.43). cap 4
    # was pre-kernel; with the small-M kernel static_cap moves to cap 7 = **2.72x mean** (math
    # 3.37x / code 2.84x / chat 1.95x, accept 4.05), 22.6 tok/s, ~29 GB. Lookup off both quants
    # (4-bit: net loss; 8-bit: a wash on the flat curve). Lossless (fp ties, margins 0.0/0.125).
    #
    # **DFlash 2 (`incoai/Qwen3.8-27B-DFlash2`) beats BOTH DSpark heads at the identical
    # verify width 8 (measured 2026-08-19, same-session pairs, 3-trial medians)** — its
    # candidate path selector + dynamic convs lift acceptance without widening the verify:
    # 8-bit cap 7 = **3.63x mean** (2.79x chat / 4.05x code / 4.06x math, accept 5.53,
    # 30.5 tok/s) vs RadixArk 2.92x; 4-bit cap 7 = **2.30x** (accept 5.14, 33.8 tok/s —
    # the absolute-speed crown) vs DimInfer 2.01x. Full block (= the dflash-mode default
    # cap) is the peak on both quants; prefix caching covers dflash since the same day.
    # So `"mode": "dflash"` makes `--mode auto` (and the app, which loads with auto) serve
    # these targets with DFlash 2; `--mode dspark` still gets the DSpark heads for A/B.
    {"id": "qwen3.8-27b", "target": "mlx-community/Qwen3.8-27B-4bit",
     "dspark": "DimInfer/Qwen3.8-27B-Dspark-v1", "dflash": "incoai/Qwen3.8-27B-DFlash2",
     "mode": "dflash", "lookup_drafts": False,
     "ram": "~18 GB", "speedup": "~2.3×"},
    # Same pair at 8-bit — best ratio (RadixArk was trained vs an FP8 verifier). Listed as its
    # own row so pickers offer both; the 4-bit row keeps the absolute-speed crown (~34 vs 30.5
    # tok/s) in ~18 GB. Resolution: the longest-id-first match sends "*-8bit" here and
    # everything else Qwen3.8 to the row above.
    {"id": "qwen3.8-27b-8bit", "target": "mlx-community/Qwen3.8-27B-8bit",
     "dspark": "RadixArk/Qwen3.8-27B-DSpark", "dflash": "incoai/Qwen3.8-27B-DFlash2",
     "mode": "dflash", "lookup_drafts": False,
     "ram": "~29 GB", "speedup": "~3.6×"},
    # NVIDIA Nemotron-3.5-Lightning-30B-A3B — a hybrid **Mamba-2 + MoE + attention** target
    # (model_type nemotron_h: 52 blocks, 128 experts top-6 + 1 shared, ~3B active, latent MoE),
    # the first non-attention recurrence here. NVIDIA's official DSpark head: a plain qwen3 GQA
    # backbone with DFlash-lineage traits (causal block, sliding-window-1024, per-head attention
    # sink, block_size 8, sample_from_anchor=false) and a markov-512 fixup head that reuses the
    # target's lm_head (has_lm_head=false). Needs the nemotron_h tap + Mamba-2 rollback in
    # target.py. Measured (M4 Pro, 4-bit target): baseline ~91 tok/s, cap 4 = 1.27x code /
    # 1.24x math / 1.06x chat (accept 4.55/4.41/3.74); lossless (fp/recurrent-state ties). The
    # MoE verify-width cost bounds the speedup, not the drafter. Drafter is bf16 (dequantized
    # from NVIDIA's modelopt NVFP4 head via nvfp4_convert.py); the raw NVFP4 repo also loads
    # (auto-decoded on first use). See NOTES "Nemotron-3.5-Lightning: the first Mamba hybrid".
    {"id": "nemotron-3.5-lightning-30b-a3b",
     "target": "mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit",
     "dspark": "mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-DSpark-bf16",
     "lookup_drafts": False,
     "ram": "~20 GB", "speedup": "~1.25× code"},
    # Meta Muse-Glimmer-30B — the first **muse_glimmer** (multimodal, 3:1 sliding/full attention,
    # NoPE globals, ~30B DENSE) target, loaded via mlx-vlm >= 0.6.12. Unlike gemma4, its language
    # model ships no capture_layer_ids hook, so target.py replicates its text forward for the tap.
    # Community DSpark drafter by DaoCloud (vLLM speculators packaging): a 5-layer qwen3 GQA backbone
    # warm-started from Meta's DFlash assistant, so its block attention is CAUSAL sliding-window-2048
    # and it reuses the target's embed_tokens AND lm_head (ships neither; the first head here to reuse
    # both). block_size 15, sample_from_anchor (logits_start 0), full 202048 vocab, markov + confidence.
    # Measured (M4 Pro, 4-bit target, warm, paired): baseline ~14 tok/s; cap 2 (auto's pick) = 1.50x
    # code / 1.50x math / 1.40x chat (accept ~2.5); cap 4 lifts accept to ~3.3 at ~1.4x. Lossless
    # (fp ties only; muse's output_multiplier 0.196 + softcap 20 compress logits, so near-ties are
    # more frequent than a dense model but every divergence margin is sub-ulp). Peak RAM ~26 GB.
    # Modest ratio is structural: a 30B DENSE 4-bit verify is expensive (curve knees at width 3, so
    # the cap stays at 2), and the drafter was trained against the BF16 verifier — the 4-bit target
    # is a precision step away (accept 4.6 on bf16 per the card -> ~3.3 here).
    #
    # **8-bit target (`mlx-community/Muse-Glimmer-30B-8bit`, 2026-08-12): the ratio ~DOUBLES.**
    # The verify curve is flat to width 5 then knees at 6 (vs 4-bit's knee at 3), so auto-cap picks
    # **4**, and 8-bit sits closer to the drafter's BF16 training verifier — both lift the payoff:
    # baseline 8.2 tok/s -> **cap 4 = 2.44x code / 2.04x math / 1.74x chat** (accept 3.58/3.00/2.51;
    # decode-only, prefill removed, hits 2.53x on code). Lossless (cap 2 & cap 4 diverge from
    # single-row greedy at the SAME fp-tie position, 39). Peak RAM ~40 GB (fits 48 but tight). The
    # catch: 8-bit decode reads ~2x the weight bytes, so ABSOLUTE throughput is ~parity with 4-bit
    # on code (~22 tok/s) and lower on math/chat — the better ratio buys 8-bit quality at ~4-bit
    # speed, not raw speed. bf16 (~60 GB) still does not fit 48 GB. Registry stays on 4-bit as the
    # broad-RAM default (~18 GB); the same drafter auto-resolves for the 8-bit target (basename
    # match). See NOTES "Muse-Glimmer-30B: the first muse_glimmer target (and reuse-both
    # DFlash-lineage head)".
    {"id": "muse-glimmer-30b", "target": "mlx-community/Muse-Glimmer-30B-4bit",
     "dspark": "DaoCloud/Muse-Glimmer-30B-DSpark", "lookup_drafts": False,
     "ram": "~26 GB (4-bit) / ~40 GB (8-bit)", "speedup": "~1.7× (8-bit: ~2.5×)"},
    # LiquidAI LFM2.5-DSpark — the first conv-recurrence targets here (model_type lfm2 / lfm2_moe:
    # Liquid AI's short-conv + attention hybrid). The drafters are plain qwen3-backbone DSpark heads
    # in a FIFTH packaging (target_layer_ids/mask_token_id nested in dflash_config, NO projector_type
    # tag) that reuse the target's tied embed AND lm_head (ship neither), block_size 9, INTERLEAVED
    # rope (rope_is_neox_style:false -> mlx traditional=True; ~2x accept vs neox, MEASURED) and
    # anchor-as-pos0. The conv state (a kernel-3 causal FIR window) is a new recurrence in target.py
    # (_capture/_rollback_shortconv) — the cheapest of the three (a pure FIR, no SSM accumulation).
    # Flat per-quant repos exist so one id resolves any quant; bf16 is the sweet spot (mlx 0.32.1's
    # gemv_wide makes wide-cap bf16 verify cheap, and these are bandwidth-light targets). Measured
    # M4 Pro, decode tok/s, 3-trial medians (see NOTES "LiquidAI LFM2.5-DSpark"):
    #   2.6B bf16: baseline ~43 tok/s; cap 5-6 = 2.79x code / 3.41x math / 2.22x chat (accept 3.75/4.55/3.03)
    #   1.2B bf16: baseline ~100 tok/s; cap 5-6 = 4.42x code / 3.14x math / 1.95x chat (accept 6.70/4.76/3.11)
    {"id": "lfm2.5-2.6b", "target": "LiquidAI/LFM2.5-2.6B-MLX-bf16",
     "dspark": "LiquidAI/LFM2.5-2.6B-DSpark",
     "ram": "~7 GB", "speedup": "~2.8×"},
    {"id": "lfm2.5-1.2b", "target": "LiquidAI/LFM2.5-1.2B-Instruct-MLX-bf16",
     "dspark": "LiquidAI/LFM2.5-1.2B-Instruct-DSpark",
     "ram": "~4 GB", "speedup": "~3.1× (4.4× code)"},
    # 8B-A1B is MoE (lfm2_moe: 32 experts, ~1B active) — loaded with ZERO extra model code (its
    # ShortConv / decoder layout matches lfm2; the MoE only swaps the FFN inside the layer, invisible
    # to the tap), lossless. **Registered on BF16, not 8-bit** — the drafter only pays where the
    # target step is expensive enough to amortize the dense 327M drafter, and this MoE's ~1B-active
    # step is very cheap. Measured (M4 Pro, greedy, benchmark suite): bf16 baseline ~65 tok/s ->
    # cap 4 = 1.26x (1.44x math / 1.30x code / 1.04x chat; per-content probe peaks 1.67x math);
    # 8bit is a NET LOSS (baseline ~114 tok/s, 0.90–0.97x every cap — the ~1B step is too cheap).
    # Confirms LiquidAI's own card (M4 Max bf16 mean 1.18x; our per-token accept ~69% matches theirs)
    # — the ratio is bounded by the MoE verify curve, not the drafter. **Absolute-speed caveat:
    # 8bit-at-baseline (~114 tok/s) is still faster than bf16+spec (~82) — so this pair is a win for
    # bf16-quality users, not the fastest way to run the model.** Kept OUT of the README hook table
    # (modest + that caveat). lookup off (MoE). The quant-agnostic id still resolves the 8bit target.
    {"id": "lfm2.5-8b-a1b", "target": "LiquidAI/LFM2.5-8B-A1B-MLX-bf16",
     "dspark": "LiquidAI/LFM2.5-8B-A1B-DSpark", "lookup_drafts": False,
     "ram": "~19 GB", "speedup": "~1.3× (MoE, bf16)"},
]

# legacy `--family` / load_pair("qwen3") values -> a concrete target repo (deprecated).
_FAMILY_ALIASES = {
    "qwen3": "mlx-community/Qwen3-4B-8bit",
    "gemma4": "mlx-community/gemma-4-12B-it-8bit",
}


def _registry_entry(target: str) -> dict | None:
    """Find the registry entry whose model id matches this target repo/path (quant-agnostic)."""
    key = os.path.basename(str(target).rstrip("/")).lower()
    key_nodash = key.replace("-", "")
    # longest id first so e.g. 'gemma-4-12b' wins over any shorter accidental match
    for entry in sorted(REGISTRY, key=lambda e: -len(e["id"])):
        eid = entry["id"]
        if eid in key or eid.replace("-", "") in key_nodash:
            return entry
    return None


def lookup_drafts_default(target: str | None) -> bool:
    """Shipped default for hybrid n-gram lookup drafts in dspark mode, per pair.

    Lookup drafts are only free where extra verify rows are cheap. On targets whose verify
    curve rises steeply from narrow widths — every MoE measured (a low-acceptance free draft
    pulls fresh routed experts per row) and the 4-bit 27B hybrids (Qwen3.8-27B: 1.74x off vs
    1.56x on at cap 2) — the free draft costs more than the drafter round it replaces, so the
    registry rows measured that way carry ``lookup_drafts: False`` and this returns it.
    Unknown targets and rows without the key keep the global default True (dense cheap-verify
    targets measure lookup as a clear win, +34% on copy content). An explicit CLI/request
    setting always wins over this default — callers only consult it when the user said
    nothing.
    """
    entry = _registry_entry(target) if target else None
    if entry is None:
        return True
    return bool(entry.get("lookup_drafts", True))


def resolve(model: str | None = None, *, mode: str = "dspark", drafter: str | None = None,
            family: str | None = None, target: str | None = None) -> tuple[str, str | None]:
    """Resolve ``(target_repo, drafter_repo)`` from a ``--model`` target and ``--mode``.

    ``model`` is a target HF repo or local path (the standard interface). The drafter is taken
    from ``drafter`` if given, else auto-resolved from :data:`REGISTRY` for a known target, else
    a helpful error. ``family`` and ``target`` are accepted as **deprecated** aliases for ``model``
    (old ``--family`` / ``--target``); a bare ``"qwen3"``/``"gemma4"`` passed as ``model`` is also
    treated as the legacy family alias. ``mode="baseline"`` / ``"lookup"`` need no drafter and
    return ``(target, None)`` — so those modes work with ANY target, registered or not.
    """
    tgt = model or target or family
    if tgt in _FAMILY_ALIASES:                     # legacy "qwen3"/"gemma4"
        tgt = _FAMILY_ALIASES[tgt]
    if not tgt:
        tgt = DEFAULT_TARGET
    if mode in ("baseline", "lookup"):
        return tgt, None
    if drafter:
        return tgt, drafter
    entry = _registry_entry(tgt)
    if entry is not None and entry.get(mode):
        return tgt, entry[mode]
    raise ValueError(
        f"no built-in {mode} drafter is registered for target {tgt!r} — pass --drafter <repo>, "
        f"use a known target (see `mlx-dspark models`), or use `--mode auto` / `--mode lookup` "
        f"(drafter-free, works with any target)."
    )


def resolve_mode(model: str | None = None, *, mode: str = "auto", drafter: str | None = None,
                 family: str | None = None, target: str | None = None
                 ) -> tuple[str, str, str | None]:
    """Like :func:`resolve` but also resolves ``mode="auto"``: pick the best available
    speculation for this target — the registry row's **measured-best mode** where one is
    stamped (``"mode"``: Qwen3.8-27B's DFlash 2 beat both DSpark heads at the same verify
    width, 2026-08-19), else its DSpark drafter (DSpark won every other M-series
    head-to-head at the short-block operating point), else its DFlash drafter, else
    drafter-free prompt-lookup — so ANY target gets some speculation.
    Returns ``(resolved_mode, target_repo, drafter_repo)``."""
    if mode != "auto":
        tgt, drf = resolve(model, mode=mode, drafter=drafter, family=family, target=target)
        return mode, tgt, drf
    if drafter:                                    # explicit drafter + auto -> DSpark adapter
        tgt, drf = resolve(model, mode="dspark", drafter=drafter, family=family, target=target)
        return "dspark", tgt, drf
    tgt, _ = resolve(model, mode="baseline", family=family, target=target)
    entry = _registry_entry(tgt)
    if entry is not None:
        best = entry.get("mode")
        if best in ("dspark", "dflash") and entry.get(best):
            return best, tgt, entry[best]
        if entry.get("dspark"):
            return "dspark", tgt, entry["dspark"]
        if entry.get("dflash"):
            return "dflash", tgt, entry["dflash"]
    return "lookup", tgt, None


def apply_wired_limit() -> None:
    """Wire MLX's recommended working set (what mlx-lm's server does) so multi-GB weights
    stay resident under memory pressure instead of getting paged mid-generation.

    **OPT-IN since 0.6.1 — this is no longer called by default, and it can hang the machine.**
    Wired pages cannot be reclaimed by the OS, so raising the ceiling to the recommended
    working set (36 GiB of 48 on an M4 Pro) while the process already holds ~23 GiB left
    macOS nothing to page out: the system locked up hard enough to need a power cycle, during
    a run that loaded Qwen3-14B and Qwen3-8B back to back. Note the shape of that risk — a
    16 GB Mac's recommended working set is ~12 GiB, so "the model nearly fills RAM", the very
    case this was added for, is also the case most likely to wedge.

    Short of hanging, it corrupts memory during long generations on the **gemma-4/mlx-vlm**
    route: a ~430-token run produced a verify-logits buffer full of garbage, so ``argmax``
    returned sequential position indices instead of token ids and the accepted-prefix length
    came back as 350 for a 4-token draft. It surfaced as ``IndexError`` only by luck —
    different garbage would have committed plausible *wrong* tokens and silently broken
    losslessness, which is the one guarantee this project makes. 3/3 crashes with it, 3/3
    clean 1000-token runs without. mlx-lm-route targets did **not** reproduce it (Qwen3-14B
    1260 tokens, Qwen3-8B 1501 tokens, both clean at a *higher* 22.8 GiB peak), so it is not
    a simple memory-pressure threshold.

    It also bought nothing measurable where it was tested: same machine, same prompts,
    baseline 18.2 vs 18.3 tok/s and dspark 47.1 vs 46.7 tok/s with vs without (<1%, well
    inside the ~14% run-to-run noise), because a 20 GiB working set on a 48 GB machine has
    no paging pressure to prevent. It earns its keep only when the model nearly fills RAM —
    a 12 GB model on a 16 GB Mac — which is the case it was added for. Turn it on there with
    ``--wired-limit`` and verify your own long runs.
    """
    try:
        if mx.metal.is_available():
            mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    except Exception:  # noqa: BLE001 — a hint, never a failure
        pass


# LM Studio's model caches, newest layout first (it moved from ~/.cache/lm-studio to
# ~/.lmstudio). Both hold plain <publisher>/<model> dirs; the MLX ones load as-is.
LMSTUDIO_ROOTS = ("~/.lmstudio/models", "~/.cache/lm-studio/models")


def is_mlx_model_dir(path: str) -> bool:
    """Whether ``path`` holds an MLX-loadable checkpoint (config + safetensors) — what
    separates an LM Studio MLX download from a GGUF one our target loaders can't read."""
    try:
        names = os.listdir(path)
    except OSError:
        return False
    return "config.json" in names and any(n.endswith(".safetensors") for n in names)


def _resolve(repo_or_path: str) -> str:
    if repo_or_path.startswith("gguf:"):
        # "gguf:{hf_repo}/{filename}.gguf" — a PrismML dspark drafter shipped as GGUF;
        # download + convert to the DeepSpec layout on first use (cached).
        from .gguf_convert import ensure_converted
        repo, filename = repo_or_path[len("gguf:"):].rsplit("/", 1)
        return ensure_converted(repo, filename)
    if os.path.isdir(repo_or_path):
        return repo_or_path
    # Prefer the plain-dir model cache (~/.cache/mlx_dspark/models/…) over a hub download:
    # models fetched by hand (e.g. around a wedged huggingface_hub) live there, and
    # auto-resolved drafters must not re-download gigabytes the user already has on disk.
    # Match BOTH naming conventions those hand-downloads use: the bare basename AND the
    # org-prefixed "<org>_<name>" form (what huggingface-cli and robust_download.py produce,
    # and what disambiguates two repos that share a basename — e.g. DimInfer's
    # Qwen3.8-27B-Dspark-v1 vs the SpecForge Qwen3.8-27B-DSpark). Basename first keeps the
    # historical preference. Must stay in lockstep with diagnostics._is_local.
    models = os.path.expanduser("~/.cache/mlx_dspark/models")
    stripped = repo_or_path.rstrip("/")
    for name in (os.path.basename(stripped), stripped.replace("/", "_")):
        local = os.path.join(models, name)
        if os.path.isdir(local):
            return local
    # Reuse models LM Studio already downloaded (issue #12): its cache is plain
    # <root>/<publisher>/<model> dirs, and its MLX downloads are exactly the safetensors
    # layout our loaders read. Exact org/name match only — no basename guessing, so two
    # publishers sharing a model name can't cross-resolve. GGUF-only dirs are skipped
    # (is_mlx_model_dir), falling through to the hub download rather than failing deep
    # inside a loader that can't read GGUF targets.
    for root in LMSTUDIO_ROOTS:
        local = os.path.join(os.path.expanduser(root), stripped)
        if os.path.isdir(local) and is_mlx_model_dir(local):
            return local
    return snapshot_download(repo_or_path)


def load_drafter(
    repo_or_path: str = DEFAULT_DRAFTER,
    *,
    quantize: bool = True,
    bits: int = 4,
    group_size: int = 64,
    strict: bool = True,
):
    """Return (drafter, config). Loads bf16 weights 1:1 by matching key names.

    The drafter is ~6.86 GB in bf16 and runs every speculative round, so by
    default it is quantized to 4-bit (~1.8 GB) — this is what makes spec
    decoding a net speedup on Apple Silicon. Output correctness is unaffected
    (the target verifies every token); only acceptance length may change.

    A checkpoint whose tensor names don't match the model raises (``strict=True``
    default) — a partially-loaded drafter "works" with near-zero acceptance, which
    is worse than an error. ``strict=False`` restores warn-and-load-anyway.

    A modelopt-NVFP4 drafter (NVIDIA Nemotron DSpark head) is transparently dequantized to a
    cached bf16 checkpoint first (mlx-lm has no modelopt loader), exactly as the GGUF Bonsai
    drafters are converted on first use.
    """
    from .nvfp4_convert import ensure_converted as _ensure_nvfp4

    path = _ensure_nvfp4(_resolve(repo_or_path))
    config = DSparkConfig.from_json(os.path.join(path, "config.json"))

    weights: dict[str, mx.array] = {}
    for st in glob.glob(os.path.join(path, "*.safetensors")):
        weights.update(mx.load(st))

    # A DFlash-warm-started head reuses the target's own embed_tokens and/or lm_head and ships
    # neither weight (DaoCloud/Muse-Glimmer-30B-DSpark reuses BOTH; the Nemotron head ships
    # embed but reuses lm_head). The checkpoint is authoritative about what's present, so read
    # the reuse flags from it — the same detection nvfp4_convert does for has_lm_head — and
    # build the drafter without the parts it will bind from the target at generation time.
    config.has_own_embed = "embed_tokens.weight" in weights
    config.has_own_lm_head = "lm_head.weight" in weights
    drafter = DSparkDrafter(config)

    # Reduced-draft-vocab heads ship two index tables next to the weights: `d2t`
    # (draft->target offsets, which inference needs) and `t2d` (a target-side boolean
    # membership mask, which only the trainer needs — previous tokens reaching the drafter
    # are already target ids). Both are index data rather than parameters, so pull them out
    # before the name check: otherwise they read as "unexpected" keys and would be quantized.
    d2t = weights.pop("d2t", None)
    weights.pop("t2d", None)
    if config.draft_vocab_size and d2t is None:
        raise ValueError(
            f"{repo_or_path}: config declares draft_vocab_size={config.draft_vocab_size} "
            f"but the checkpoint has no `d2t` table, so draft ids cannot be mapped back to "
            f"target ids. Every draft token would decode as a different word while the "
            f"drafter still appeared to run — refusing to load."
        )
    if d2t is not None and not config.draft_vocab_size:
        raise ValueError(
            f"{repo_or_path}: checkpoint ships a `d2t` table but the config declares no "
            f"draft_vocab_size — the drafter would emit draft-space ids as if they were "
            f"target ids."
        )

    # Diagnose name mismatches before loading.
    model_keys = {k for k, _ in _flatten_params(drafter)}
    ckpt_keys = set(weights.keys())
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    if missing or unexpected:
        detail = ""
        if missing:
            detail += f"\n  missing in checkpoint ({len(missing)}): {missing[:8]}"
        if unexpected:
            detail += f"\n  unexpected in checkpoint ({len(unexpected)}): {unexpected[:8]}"
        if strict:
            raise ValueError(
                f"{repo_or_path}: drafter tensor names don't match a DeepSpec-format DSpark "
                f"drafter — the checkpoint may be a different packaging or drafter variant."
                f"{detail}\n  (load_drafter(..., strict=False) force-loads the intersection, "
                f"but a partially-loaded drafter gives near-zero acceptance.)"
            )
        print(f"[load_drafter] WARNING key mismatch:{detail}")

    drafter.load_weights(list(weights.items()), strict=not (missing or unexpected))

    if config.offset_rms_norm:
        # qwen3_5-style checkpoints store RMSNorm weights as offsets from one ((1+w)·x̂).
        # Materialize the +1 so plain nn.RMSNorm computes the reference semantics (matches
        # the reference's `weight + 1.0` fused path bit-for-bit in bf16). Skipping this
        # leaves the context-fusion norm multiplying by ~0 → acceptance collapses to ~1.25.
        for _, m in drafter.named_modules():
            if isinstance(m, nn.RMSNorm):
                m.weight = m.weight + 1.0

    if d2t is not None:
        drafter.set_draft_vocab_map(d2t)

    if quantize:
        # Quantize Linear/Embedding weights; norms/scalars stay full precision. The GIDD
        # log-SNR MLP (Bonsai drafters) is excluded to match PrismML's own packaging (it
        # stays BF16 even in their Q4_1 GGUF) — it runs once per load, so size is moot.
        nn.quantize(drafter, group_size=group_size, bits=bits,
                    class_predicate=lambda p, m: (isinstance(m, (nn.Linear, nn.Embedding))
                                                  and not p.startswith("log_snr_embed")))

    mx.eval(drafter.parameters())
    return drafter, config


def _flatten_params(module) -> list[tuple[str, mx.array]]:
    from mlx.utils import tree_flatten

    return tree_flatten(module.parameters())


def load_dflash(repo_or_path: str, *, quantize: bool = True, bits: int = 4, group_size: int = 64):
    """Return (drafter, config) for a z-lab DFlash checkpoint (block-diffusion drafter).

    Unlike the DSpark drafter, DFlash has no own embed/lm_head — it reuses the target's
    (call ``drafter.bind(target.model)`` before generating; ``dflash_generate`` does this).
    Tolerant of the gemma4 config layout (rope nested under ``rope_parameters``) that
    z-lab's own ``load_draft`` assumes flat.
    """
    import json

    from .dflash_model import DFlashConfig, DFlashDraftModel

    path = _resolve(repo_or_path)
    with open(os.path.join(path, "config.json")) as f:
        cfg = json.load(f)
    if cfg.get("markov_rank"):
        # Community hybrids exist (DFlash block-16 backbone + a DSpark Markov head,
        # e.g. Hikari07jp/DSpark-Gemma-4-31B-draft) — our vendored z-lab DFlashDraftModel
        # has no Markov head, so the weights can't load. Refuse with the real reason.
        raise ValueError(
            f"{repo_or_path}: this DFlash checkpoint carries a Markov head "
            f"(markov_rank={cfg['markov_rank']}) — a DFlash+DSpark hybrid variant mlx-dspark "
            f"doesn't support yet. Open an issue: https://github.com/ARahim3/mlx-dspark/issues"
        )
    rope = cfg.get("rope_parameters") or {}
    rope_theta = cfg.get("rope_theta", rope.get("rope_theta", 1_000_000.0))
    layer_types = tuple(cfg.get("layer_types") or ["full_attention"] * cfg["num_hidden_layers"])
    # z-lab ships the DFlash-specific fields either at the top level (the gemma4 / Qwen3-4B
    # era heads) or nested under `dflash_config` (Qwen3.6-35B-A3B and later). Read every one
    # of them the same way — block_size used to be the odd one out, which made a nested-config
    # head die with a bare KeyError instead of loading.
    dfc = cfg.get("dflash_config", {})
    block_size = dfc.get("block_size", cfg.get("block_size"))
    if block_size is None:
        raise ValueError(
            f"{repo_or_path}: no block_size in the config (looked at the top level and under "
            f"'dflash_config') — this does not look like a z-lab DFlash drafter checkpoint."
        )
    # DFlash 2 (incoai/*-DFlash2): a candidate path selector + per-sublayer dynamic convs,
    # declared in dflash_config. Absent -> plain DFlash 1 (fields default to 0 / off).
    selector_rank = int(dfc.get("selector_rank") or 0)
    selector_top_k = int(dfc.get("selector_top_k") or 0)
    if "DFlash2DraftModel" in (cfg.get("architectures") or []) and not (
            selector_rank and selector_top_k):
        raise ValueError(
            f"{repo_or_path}: architectures says DFlash2DraftModel but dflash_config carries no "
            f"selector_rank/selector_top_k — refusing to run a DFlash 2 head as DFlash 1 "
            f"(the selector is where its acceptance comes from)."
        )
    config = DFlashConfig(
        hidden_size=cfg["hidden_size"], num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"], num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"], intermediate_size=cfg["intermediate_size"], vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg["rms_norm_eps"], rope_theta=rope_theta,
        max_position_embeddings=cfg["max_position_embeddings"], block_size=int(block_size),
        target_layer_ids=tuple(dfc.get("target_layer_ids") or cfg["target_layer_ids"]),
        num_target_layers=cfg["num_target_layers"],
        mask_token_id=dfc.get("mask_token_id", cfg.get("mask_token_id", 0)),
        rope_scaling=cfg.get("rope_scaling"), layer_types=layer_types,
        sliding_window=cfg.get("sliding_window"),
        # DFlash 2 nests these in dflash_config (muse); DFlash 1 gemma had softcap top-level
        final_logit_softcapping=dfc.get("final_logit_softcapping",
                                        cfg.get("final_logit_softcapping")),
        selector_rank=selector_rank, selector_top_k=selector_top_k,
        conv_kernel_size=int(dfc.get("conv_kernel_size") or 0),
        conv_group_size=int(dfc.get("conv_group_size") or 16),
        output_multiplier=float(dfc.get("output_multiplier") or 1.0),
    )
    drafter = DFlashDraftModel(config)

    weights: dict[str, mx.array] = {}
    for st in glob.glob(os.path.join(path, "*.safetensors")):
        weights.update(mx.load(st))
    model_keys = {k for k, _ in _flatten_params(drafter)}
    ckpt_keys = set(weights.keys())
    if model_keys != ckpt_keys:
        missing = sorted(model_keys - ckpt_keys)
        unexpected = sorted(ckpt_keys - model_keys)
        raise ValueError(
            f"{repo_or_path}: tensor names don't match a z-lab DFlash drafter."
            + (f"\n  missing in checkpoint ({len(missing)}): {missing[:8]}" if missing else "")
            + (f"\n  unexpected in checkpoint ({len(unexpected)}): {unexpected[:8]}"
               if unexpected else "")
        )
    drafter.load_weights(list(weights.items()))

    if quantize:
        # quantize only the backbone Linears — embed/lm_head come from the (already
        # quantized) target via bind(), so leave them untouched. DFlash 2's conv
        # kernel_projections and selector hidden_projection stay bf16 like the reference
        # (they produce multiplicative coefficients / lattice scores — semantics-sensitive
        # and tiny, ~130 MB total on the 27B head); the [vocab, rank] codebooks are raw
        # arrays (gather-only) that nn.quantize never touches.
        nn.quantize(drafter, group_size=group_size, bits=bits,
                    class_predicate=lambda p, m: isinstance(m, nn.Linear)
                    and "_conv" not in p and "candidate_selector" not in p)

    mx.eval(drafter.parameters())
    return drafter, config


def _route_target(cfg: dict) -> str:
    """Decide which loader owns a target config: ``"mlx_lm"`` or ``"mlx_vlm"``.

    Multimodal markers (``vision_config``/``audio_config`` — e.g. gemma4_unified) go to
    mlx-vlm. Otherwise any model_type mlx-lm ships a module for (qwen3, llama, glm_moe_dsa,
    deepseek_v3, …) goes to mlx-lm — mirroring mlx-lm's own model_type→module lookup incl.
    its remap table — so new text families route correctly without a code change here.
    Anything else falls back to mlx-vlm (the pre-existing behavior).

    Exception: families where mlx-lm ships a *text-only* module that digests the full
    multimodal checkpoint (its sanitize drops the vision tower) route to mlx-lm despite the
    vision_config — text generation is what this project serves, and only the mlx-lm path
    has the replicated-loop hidden-state tap the drafters need (qwen3_5 = Qwen3.6/Bonsai)."""
    _TEXT_CAPABLE_MM = {"qwen3_5", "qwen3_5_moe"}
    if cfg.get("model_type") in _TEXT_CAPABLE_MM:
        from importlib.util import find_spec
        if find_spec(f"mlx_lm.models.{cfg['model_type']}") is not None:
            return "mlx_lm"
    if "vision_config" in cfg or "audio_config" in cfg:
        return "mlx_vlm"
    model_type = cfg.get("model_type", "")
    try:
        from mlx_lm.utils import MODEL_REMAPPING
        model_type = MODEL_REMAPPING.get(model_type, model_type)
    except ImportError:
        pass
    from importlib.util import find_spec
    if model_type and find_spec(f"mlx_lm.models.{model_type}") is not None:
        return "mlx_lm"
    return "mlx_vlm"


def _shim_gemma4_unified_processor(cls=None) -> bool:
    """mlx-vlm 0.6.4 compat: make ``Gemma4UnifiedProcessor`` loadable under transformers>=5.12.

    0.6.4 changed the parent ``Gemma4Processor`` to hand ``video_processor`` through to
    ``ProcessorMixin.__init__``, but the child still takes it via ``**kwargs`` — and
    transformers>=5.12 derives a processor's valid kwargs from the literal ``__init__``
    signature (``ProcessorMixin.get_attributes``), so loading the gemma4 preset dies with
    ``TypeError: Unexpected keyword argument video_processor``. Worse, mlx-vlm's
    AutoProcessor patch swallows that TypeError and falls back to transformers' own
    checkpoint-incompatible processor, which surfaces as an unrelated ``OSError: Can't
    load video processor`` (issue #4; upstream Blaizzy/mlx-vlm#1578). Give the child the
    explicit signature upstream fixed on main. Only the broken 0.6.4 shape is patched:
    ``attributes`` names video_processor (so it reaches ProcessorMixin) while the
    signature omits it — 0.6.3 (neither) and fixed releases (both) pass through
    untouched; wrapping 0.6.3 would *introduce* an attribute-count mismatch. Returns
    whether the shim was applied. Drop once the mlx-vlm floor is above 0.6.4."""
    if cls is None:
        try:
            from mlx_vlm.models.gemma4_unified.processing_gemma4_unified import (
                Gemma4UnifiedProcessor,
            )
        except Exception:  # noqa: BLE001 — no mlx-vlm / no gemma4 module: nothing to shim
            return False
        cls = Gemma4UnifiedProcessor
    import inspect
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return False
    if "video_processor" in params or "video_processor" not in getattr(cls, "attributes", ()):
        return False
    orig = cls.__init__

    # NOTE: no functools.wraps — inspect.signature() follows __wrapped__, which would
    # resurface the old signature and defeat the patch.
    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, **kwargs):
        if video_processor is not None:
            kwargs["video_processor"] = video_processor
        orig(self, image_processor=image_processor, tokenizer=tokenizer, **kwargs)

    cls.__init__ = __init__
    return True


def load_target(repo_or_path: str = DEFAULT_TARGET, *, require_tap: bool = False,
                kv_bits: int | None = None, kv_group_size: int = 64):
    """Return (Target, tokenizer). Routes text models to mlx-lm and multimodal/unified
    models (Gemma-4) to mlx-vlm (see :func:`_route_target`), then wraps in a family-aware
    Target (hidden-state tap). ``require_tap=True`` (any drafter mode) additionally probes
    that the manual mlx-lm tap reproduces the model's own forward — a family the generic
    loop can't replicate fails loudly here instead of silently drafting from a wrong
    stream. Baseline/lookup modes skip the probe (they never tap)."""
    import json

    path = _resolve(repo_or_path)
    cfg: dict = {}
    cfg_path = os.path.join(path, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)

    # Fail with the real reason before mlx-lm dies deep inside mx.quantize: stock MLX
    # supports 2/3/4/5/6/8-bit affine quantization only. mlx-vlm >= 0.6.5 can run 1-bit
    # affine packs (e.g. prism-ml/Bonsai-27B-mlx-1bit) via its own Python-hosted kernel,
    # but that kernel's verify cost is linear in draft length (each verified token re-reads
    # the full weight stream), so speculative decoding measures net-negative on it
    # (0.71-0.77x at any cap, M4 Pro) — kept unintegrated on purpose; see NOTES.
    bits = (cfg.get("quantization") or {}).get("bits")
    if bits and int(bits) not in (2, 3, 4, 5, 6, 8):
        if int(bits) == 1:
            hint = (" mlx-vlm >= 0.6.5 runs 1-bit affine packs standalone, but speculative "
                    "decoding measures net-negative on its kernel (verify cost is linear in "
                    "draft length), so mlx-dspark does not integrate them.")
        else:
            hint = " It likely targets a vendor MLX fork with custom kernels."
        if "bonsai" in str(repo_or_path).lower():
            hint += " Use prism-ml/Ternary-Bonsai-27B-mlx-2bit instead."
        raise ValueError(
            f"{repo_or_path}: this checkpoint is quantized to {bits} bits, which stock MLX "
            f"cannot load (mx.quantize supports 2/3/4/5/6/8).{hint}"
        )

    if _route_target(cfg) == "mlx_lm":
        from mlx_lm import load as lm_load

        model, tokenizer = lm_load(path)
    else:
        from mlx_vlm import load as vlm_load

        if str(cfg.get("model_type", "")).startswith("gemma4"):
            _shim_gemma4_unified_processor()
        try:
            model, processor = vlm_load(path)
        except Exception as e:
            # mlx-vlm's AutoProcessor fallback masks its own processor failures behind
            # unrelated errors (issue #4) — name the known one instead of relaying it raw.
            hint = ""
            if "video processor" in str(e).lower() or "video_processor" in str(e):
                hint = (" This looks like the mlx-vlm 0.6.4 × transformers>=5.12 "
                        "Gemma4UnifiedProcessor incompatibility (Blaizzy/mlx-vlm#1578): "
                        "upgrade mlx-vlm past 0.6.4, or pin mlx-vlm==0.6.3.")
            raise ValueError(
                f"{repo_or_path}: target model_type {cfg.get('model_type')!r} is supported by "
                f"neither this mlx-lm ({e.__class__.__name__} from mlx-vlm fallback: {e}) — "
                f"try upgrading mlx-lm/mlx-vlm, or open an issue: "
                f"https://github.com/ARahim3/mlx-dspark/issues{hint}"
            ) from e
        tokenizer = getattr(processor, "tokenizer", processor)
    target = Target(model, tokenizer, kv_bits=kv_bits, kv_group_size=kv_group_size)
    if require_tap:
        target.verify_tap()
    return target, tokenizer


def load_pair(model: str = "gemma4", *, drafter: str | None = None):
    """Convenience: load (target, tokenizer, DSpark drafter, cfg).

    ``model`` is a target HF repo or local path (e.g. ``"mlx-community/Qwen3-8B-8bit"``); the
    matched drafter auto-resolves from the registry, or pass ``drafter=``. A legacy family alias
    (``"qwen3"`` / ``"gemma4"``) is still accepted."""
    target_repo, drafter_repo = resolve(model, mode="dspark", drafter=drafter)
    target, tok = load_target(target_repo, require_tap=True)
    drafter_m, cfg = load_drafter(drafter_repo)
    return target, tok, drafter_m, cfg


def load_dflash_pair(model: str = "gemma4", *, drafter: str | None = None):
    """Convenience: load (target, tokenizer, DFlash drafter, cfg), drafter bound to the target's
    embed/lm_head and ready for ``dflash_generate``. ``model`` as in :func:`load_pair`."""
    target_repo, drafter_repo = resolve(model, mode="dflash", drafter=drafter)
    target, tok = load_target(target_repo, require_tap=True)
    drafter_m, cfg = load_dflash(drafter_repo)
    drafter_m.bind(target.model)
    return target, tok, drafter_m, cfg
