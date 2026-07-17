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
REGISTRY = [
    {"id": "qwen3-4b",   "target": "mlx-community/Qwen3-4B-8bit",
     "dspark": "deepseek-ai/dspark_qwen3_4b_block7", "dflash": "z-lab/Qwen3-4B-DFlash-b16",
     "ram": "~8 GB"},
    {"id": "qwen3-8b",   "target": "mlx-community/Qwen3-8B-8bit",
     "dspark": "deepseek-ai/dspark_qwen3_8b_block7", "dflash": "z-lab/Qwen3-8B-DFlash-b16",
     "ram": "~11 GB"},
    {"id": "gemma-4-12b", "target": "mlx-community/gemma-4-12B-it-8bit",
     "dspark": "deepseek-ai/dspark_gemma4_12b_block7", "dflash": "z-lab/gemma4-12B-it-DFlash",
     "ram": "~15 GB"},
    # PrismML Ternary-Bonsai 27B (ternary rebuild of Qwen3.6-27B, hybrid linear attention).
    # PrismML ships the DSpark drafter GGUF-only (no safetensors export exists); the repo
    # below is our 1:1 bf16 repack into the DeepSpec layout (converted with gguf_convert.py —
    # the "gguf:{repo}/{file}.gguf" drafter scheme converts any future drop locally the same
    # way). The 1-bit variant (prism-ml/Bonsai-27B-mlx-1bit) is NOT registered: its pack is
    # 1-bit-quantized for PrismML's own MLX fork and stock mx.quantize has no 1-bit mode
    # (load_target refuses it with the reason). Drafters are variant-specific.
    {"id": "ternary-bonsai-27b", "target": "prism-ml/Ternary-Bonsai-27B-mlx-2bit",
     "dspark": "Rahim/Ternary-Bonsai-27B-dspark",
     "ram": "~12 GB"},
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
    speculation for this target — the registry's DSpark drafter if the target is known
    (DSpark won every M-series head-to-head at the short-block operating point), else its
    DFlash drafter, else drafter-free prompt-lookup — so ANY target gets some speculation.
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
        if entry.get("dspark"):
            return "dspark", tgt, entry["dspark"]
        if entry.get("dflash"):
            return "dflash", tgt, entry["dflash"]
    return "lookup", tgt, None


def apply_wired_limit() -> None:
    """Wire MLX's recommended working set (what mlx-lm's server does) so multi-GB weights
    stay resident under memory pressure instead of getting paged mid-generation."""
    try:
        if mx.metal.is_available():
            mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    except Exception:  # noqa: BLE001 — a hint, never a failure
        pass


def _resolve(repo_or_path: str) -> str:
    if repo_or_path.startswith("gguf:"):
        # "gguf:{hf_repo}/{filename}.gguf" — a PrismML dspark drafter shipped as GGUF;
        # download + convert to the DeepSpec layout on first use (cached).
        from .gguf_convert import ensure_converted
        repo, filename = repo_or_path[len("gguf:"):].rsplit("/", 1)
        return ensure_converted(repo, filename)
    if os.path.isdir(repo_or_path):
        return repo_or_path
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
    """
    path = _resolve(repo_or_path)
    config = DSparkConfig.from_json(os.path.join(path, "config.json"))
    drafter = DSparkDrafter(config)

    weights: dict[str, mx.array] = {}
    for st in glob.glob(os.path.join(path, "*.safetensors")):
        weights.update(mx.load(st))

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
    cfg = json.loads(open(os.path.join(path, "config.json")).read())
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
    dfc = cfg.get("dflash_config", {})
    config = DFlashConfig(
        hidden_size=cfg["hidden_size"], num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"], num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"], intermediate_size=cfg["intermediate_size"], vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg["rms_norm_eps"], rope_theta=rope_theta,
        max_position_embeddings=cfg["max_position_embeddings"], block_size=cfg["block_size"],
        target_layer_ids=tuple(dfc.get("target_layer_ids") or cfg["target_layer_ids"]),
        num_target_layers=cfg["num_target_layers"],
        mask_token_id=dfc.get("mask_token_id", cfg.get("mask_token_id", 0)),
        rope_scaling=cfg.get("rope_scaling"), layer_types=layer_types,
        sliding_window=cfg.get("sliding_window"),
        final_logit_softcapping=cfg.get("final_logit_softcapping"),
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
        # quantized) target via bind(), so leave them untouched.
        nn.quantize(drafter, group_size=group_size, bits=bits,
                    class_predicate=lambda p, m: isinstance(m, nn.Linear))

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
        except Exception:
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
