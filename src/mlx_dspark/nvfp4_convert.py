"""Convert a **modelopt NVFP4** DSpark drafter into the plain bf16 DeepSpec layout ``load_drafter``
reads (which then quantizes to 4-bit affine like every other drafter).

NVIDIA ships the Nemotron-3.5-Lightning DSpark head quantized with NVIDIA Model Optimizer
(``W4A16_NVFP4``): the MLP / fc / markov_w2 Linears are stored as packed FP4 (E2M1) with an
FP8 (E4M3) per-block scale and one FP32 per-tensor scale, while attention / embed_tokens /
markov_w1 stay bf16 (the ``exclude_modules`` list in ``hf_quant_config.json``). mlx-lm has no
modelopt loader, so this module hand-decodes the FP4 tensors to bf16 — the decode is exact
(the values land on the FP4 grid) and validated by coherent generation.

modelopt on-disk layout per quantized Linear ``<name>``:
  ``<name>.weight``          uint8   (out, in//2)   — two E2M1 nibbles per byte (low nibble first)
  ``<name>.weight_scale``    fp8_e4m3(out, in//16)  — per-16-element block scale (mx.load -> uint8)
  ``<name>.weight_scale_2``  fp32    ()             — per-tensor global scale
  dequant:  w[o,j] = E2M1(nibble) * E4M3(weight_scale[o, j//16]) * weight_scale_2

The result is cached under ``~/.cache/mlx_dspark/drafters/<basename>-bf16/`` (like the GGUF
converter), so the ~1 GB decode runs once per drafter.
"""
from __future__ import annotations

import glob
import json
import os
import shutil

import mlx.core as mx

_CACHE = os.path.expanduser("~/.cache/mlx_dspark/drafters")


def _e2m1_lut() -> mx.array:
    mags = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]      # E2M1: sign=bit3, exp=bits2-1, mant=bit0
    return mx.array(mags + [-m for m in mags], dtype=mx.float32)


def _e4m3_lut() -> mx.array:
    """256-entry uint8 -> f32 table for FP8 E4M3 (E4M3FN: no inf; 0x7F/0xFF = NaN)."""
    vals = []
    for b in range(256):
        s = -1.0 if (b >> 7) & 1 else 1.0
        e, m = (b >> 3) & 0xF, b & 0x7
        if e == 0:
            vals.append(s * (m / 8.0) * (2.0 ** -6))          # subnormal
        elif e == 0xF and m == 0x7:
            vals.append(float("nan"))
        else:
            vals.append(s * (1.0 + m / 8.0) * (2.0 ** (e - 7)))
    return mx.array(vals, dtype=mx.float32)


def _dequant_nvfp4(weight_u8: mx.array, scale_u8: mx.array, scale2: mx.array) -> mx.array:
    """(out, in//2) uint8 packed FP4 + (out, in//16) uint8 FP8 scale + f32 scalar -> (out, in) bf16."""
    e2m1, e4m3 = _e2m1_lut(), _e4m3_lut()
    out, half = weight_u8.shape
    lo = (weight_u8 & 0x0F).astype(mx.int32)
    hi = ((weight_u8 >> 4) & 0x0F).astype(mx.int32)
    codes = mx.stack([lo, hi], axis=-1).reshape(out, half * 2)      # elems 0,1,2,3,... (low nibble first)
    vals = e2m1[codes]
    sc = e4m3[scale_u8.astype(mx.int32)] * scale2.astype(mx.float32)
    sc = mx.repeat(sc, vals.shape[1] // sc.shape[1], axis=1)        # broadcast block scale over the group
    return (vals * sc).astype(mx.bfloat16)


def is_nvfp4_drafter(path: str) -> bool:
    """True if ``path`` is a modelopt-NVFP4 checkpoint (a ``.weight`` with a ``.weight_scale`` sibling)."""
    hq = os.path.join(path, "hf_quant_config.json")
    if os.path.exists(hq):
        try:
            with open(hq) as f:
                q = (json.load(f).get("quantization") or {})
            if "NVFP4" in str(q.get("quant_algo", "")):
                return True
        except Exception:  # noqa: BLE001
            pass
    cfg = os.path.join(path, "config.json")
    if os.path.exists(cfg):
        with open(cfg) as f:
            qc = (json.load(f).get("quantization_config") or {})
        if str(qc.get("quant_method", "")) == "modelopt" or "NVFP4" in str(qc.get("quant_algo", "")):
            return True
    return False


def ensure_converted(path: str) -> str:
    """Dequantize a modelopt-NVFP4 drafter at ``path`` to a cached bf16 checkpoint; return its dir.

    Idempotent and cached: the bf16 output lives under ``~/.cache/mlx_dspark/drafters`` keyed by
    the source basename, so it is produced once. Non-NVFP4 paths are returned unchanged."""
    if not is_nvfp4_drafter(path):
        return path
    dst = os.path.join(_CACHE, os.path.basename(path.rstrip("/")) + "-bf16")
    done = os.path.join(dst, ".mlx_dspark_converted")
    if os.path.exists(done):
        return dst
    os.makedirs(dst, exist_ok=True)

    weights: dict[str, mx.array] = {}
    for st in glob.glob(os.path.join(path, "*.safetensors")):
        weights.update(mx.load(st))
    bases = sorted({k[: -len(".weight")] for k in weights
                    if k.endswith(".weight") and (k[: -len(".weight")] + ".weight_scale") in weights})
    out: dict[str, mx.array] = {}
    for k, v in weights.items():
        if any(k == b + s for b in bases for s in (".weight", ".weight_scale", ".weight_scale_2")):
            continue
        out[k] = v                                              # bf16 pass-through (attn, embed, markov_w1, sinks)
    for b in bases:
        dq = _dequant_nvfp4(weights[b + ".weight"], weights[b + ".weight_scale"],
                            weights[b + ".weight_scale_2"])
        mx.eval(dq)
        out[b + ".weight"] = dq
    mx.save_safetensors(os.path.join(dst, "model.safetensors"), out)

    # Emit a config reconciled with the actual tensors: drop the quant blocks, and mark the
    # confidence head / lm_head by presence (the NVIDIA config omits enable_confidence_head,
    # which would otherwise default on and fail the 1:1 name check). The DFlash-lineage knobs
    # (sample_from_anchor, attention_sink_bias, sliding_window, has_lm_head) are read straight
    # from the original config by DSparkConfig.from_json, so they pass through untouched.
    with open(os.path.join(path, "config.json")) as f:
        cfg = json.load(f)
    cfg.pop("quantization_config", None)
    cfg.pop("quantization", None)
    has_conf = any(".confidence_head" in k for k in out)
    cfg["enable_confidence_head"] = has_conf
    cfg["confidence_head_with_markov"] = has_conf and cfg.get("confidence_head_with_markov", True)
    cfg["has_lm_head"] = any(k == "lm_head.weight" for k in out)
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    for extra in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                  "chat_template.jinja"):
        p = os.path.join(path, extra)
        if os.path.exists(p):
            shutil.copy(p, dst)
    open(done, "w").close()
    return dst
