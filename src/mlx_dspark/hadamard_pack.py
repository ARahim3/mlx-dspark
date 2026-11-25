"""PrismML's Hadamard-rotated 2-bit MLX packs (``model_type: prism_hadamard_qwen35``).

Ternary-Bonsai 2 / 2.5 27B (ternary rebuilds of Qwen3.8-27B) ship their MLX weights in a rotated
basis. Every packed projection stores ``W·D·H``: H is the normalized Sylvester-Walsh-Hadamard
matrix over blocks of ``block`` input features (1024 in the shipped packs), D a ±1 sign vector
for that input width. So its INPUT is rotated the same way at run time, ``x' = H·(D·x)`` per
block, before MLX's 2-bit affine matmul. The token embedding is stored rotated too and is rotated
back after the row lookup, ``D·(H·row)``. Nothing else is rotated: the norms, the gated-delta
gates (``in_proj_a`` / ``in_proj_b``), the conv and the residual stream are plain qwen3_5. The
model is therefore mlx-lm's qwen3_5 text stack with its projections swapped, which is why
:class:`~mlx_dspark.target.Target`'s hybrid tap, rollback capture and deferred commit run on it
unchanged, and why a drafter that reuses the target's embedding or head gets plain-basis values
through the modules' own calls (:meth:`HadamardLinear.dense_rows` for dense head rows).

The pack ships its own loader (``runtime/``, pinned to mlx 0.32.0 / mlx-vlm 0.6.3). This is an
independent implementation of the same transform, so nothing from the checkpoint is imported.
Numerics follow the pack's validated MLX path (``runtime/vision_artifact.py``): the rotation in
fp32 rounded once, every floating tensor cast to fp16 (the pack's declared activation dtype),
and one rotation shared by the projections that read the same activation, which changes no bit.
"""

from __future__ import annotations

import glob
import json
import math
import os
import threading

import mlx.core as mx
import mlx.nn as nn

MODEL_TYPE = "prism_hadamard_qwen35"
BASE_MODEL_TYPE = "qwen3_5"
_NS = "language_model."                       # the model's (mlx-lm qwen3_5 Model) namespace
_DROP = ("vision_tower.", "model.visual.", "visual.")   # text serving: no vision tower


def is_pack(cfg: dict) -> bool:
    return cfg.get("model_type") == MODEL_TYPE


# Each rotation is a cast up, the sign multiply, the transform and a cast down: ~1,000 small
# launches a token over the 258 rotations of a decode step. Compiled, the elementwise ops fuse
# (the same arithmetic, bit for bit): 4.8 -> 3.5 ms for a token's rotations in a dependent chain,
# x1.007 end to end on Ternary-Bonsai-2.5-27B with identical tokens (M4 Pro, mlx 0.32.5). False
# is the A/B. The transform itself stays in fp32: MLX's kernel on an fp16 input rounds inside it
# (1 ulp off), which would leave the pack's validated numerics.
COMPILE = True
_COMPILED: dict = {}


def _rotate_ops(x, signs, block: int, inverse: bool, dtype):
    shape = x.shape
    y = x.astype(mx.float32)
    if not inverse:
        y = y * signs
    y = mx.hadamard_transform(y.reshape(-1, block), scale=1.0 / math.sqrt(block)).reshape(shape)
    if inverse:
        y = y * signs
    return y.astype(dtype)


def rotate(x: mx.array, signs: mx.array, block: int, *, inverse: bool = False,
           dtype=None) -> mx.array:
    """The pack's signed block-Hadamard transform over the last axis, in fp32, rounded once to
    ``dtype`` (default ``x``'s). Forward ``H·(D·x)`` per block of ``block`` features is what a
    projection's input gets; ``inverse=True`` is ``D·(H·x)``, what a stored row gets back. H is
    orthonormal and its own inverse, so the pair round-trips up to rounding."""
    dtype = dtype or x.dtype
    if not COMPILE:
        return _rotate_ops(x, signs, block, inverse, dtype)
    key = (int(block), bool(inverse), dtype)
    fn = _COMPILED.get(key)
    if fn is None:
        fn = _COMPILED[key] = mx.compile(
            lambda v, s, _k=key: _rotate_ops(v, s, _k[0], _k[1], _k[2]))
    return fn(x, signs)


class _SharedInput:
    """One rotation for a run of projections that read the same activation, called in order on
    one thread (q/k/v, in_proj_qkv/z, gate/up). The first rotates and leaves the result for the
    next, and the last drops it, so no activation outlives its layer. A call that does not match
    (another input object, out of order) rotates its own: a caller that breaks the order costs
    one rotation, never a wrong value."""

    def __init__(self, size: int):
        self.size = int(size)
        self.local = threading.local()


class HadamardLinear(nn.Module):
    """A packed projection: ``quantized_matmul(H·(D·x), W·D·H)``, stored at ``bits`` /
    ``group_size`` (2-bit, group 128 in the shipped packs). The rotated input is rounded to the
    scales' dtype, which is a no-op on the target's own fp16 stream and keeps a bf16 caller (a
    drafter projecting through the target's head) on the fp16 kernel instead of fp32."""

    def __init__(self, weight, scales, biases, signs, block: int, group_size: int, bits: int):
        super().__init__()
        self.weight, self.scales, self.biases = weight, scales, biases
        self.block, self.group_size, self.bits = int(block), int(group_size), int(bits)
        if self.block:
            self.signs = signs
        self.__dict__["_shared"] = None           # (_SharedInput, index), link_shared_inputs

    def _rotated(self, x: mx.array) -> mx.array:
        shared = self.__dict__["_shared"]
        if shared is None:
            return rotate(x, self.signs, self.block, dtype=self.scales.dtype)
        group, index = shared
        local = group.local
        if index == 0:
            out = rotate(x, self.signs, self.block, dtype=self.scales.dtype)
            local.pending = (x, out, 1)
            return out
        pending = getattr(local, "pending", None)
        local.pending = None
        if pending is not None and pending[0] is x and pending[2] == index:
            if index + 1 < group.size:
                local.pending = (x, pending[1], index + 1)
            return pending[1]
        return rotate(x, self.signs, self.block, dtype=self.scales.dtype)

    def __call__(self, x: mx.array) -> mx.array:
        if self.block:
            x = self._rotated(x)
        return mx.quantized_matmul(x, self.weight, self.scales, self.biases, transpose=True,
                                   group_size=self.group_size, bits=self.bits)

    def dense_rows(self, ids: mx.array) -> mx.array:
        """Rows ``ids`` of the plain-basis matrix W (any int shape -> [..., in]): the stored row
        rotated back, ``D·(H·row)``. A gather, not a projection (DSpine scores candidates so)."""
        flat = ids.reshape(-1)
        rows = mx.dequantize(self.weight[flat], self.scales[flat], self.biases[flat],
                             group_size=self.group_size, bits=self.bits)
        rows = rows.reshape(*ids.shape, -1)
        return rotate(rows, self.signs, self.block, inverse=True) if self.block else rows


class HadamardEmbedding(nn.Module):
    """The packed token embedding: looked-up rows are dequantized and rotated back, so the
    residual stream starts in the plain basis."""

    def __init__(self, weight, scales, biases, signs, block: int, group_size: int, bits: int):
        super().__init__()
        self.weight, self.scales, self.biases = weight, scales, biases
        self.block, self.group_size, self.bits = int(block), int(group_size), int(bits)
        if self.block:
            self.signs = signs

    def dense_rows(self, ids: mx.array) -> mx.array:
        flat = ids.reshape(-1)
        rows = mx.dequantize(self.weight[flat], self.scales[flat], self.biases[flat],
                             group_size=self.group_size, bits=self.bits)
        rows = rows.reshape(*ids.shape, -1)
        return rotate(rows, self.signs, self.block, inverse=True) if self.block else rows

    def __call__(self, ids: mx.array) -> mx.array:
        return self.dense_rows(ids)

    def as_linear(self, x: mx.array) -> mx.array:
        """A tied head through the stored table, ``E·x = E'·(H·(D·x))`` (the forward rotation,
        then the packed matmul). The packs shipped so far keep their own lm_head."""
        if self.block:
            x = rotate(x, self.signs, self.block, dtype=self.scales.dtype)
        return mx.quantized_matmul(x, self.weight, self.scales, self.biases, transpose=True,
                                   group_size=self.group_size, bits=self.bits)


def link_shared_inputs(lm) -> int:
    """Give each layer's same-input projections one rotation (:class:`_SharedInput`) where their
    transforms match (block and signs). Returns the number of groups linked. The call orders are
    mlx-lm's qwen3_5: in_proj_qkv then in_proj_z, q then k then v, gate then up."""
    linked = 0
    for layer in lm.model.layers:
        if getattr(layer, "is_linear", False):
            groups = [(layer.linear_attn.in_proj_qkv, layer.linear_attn.in_proj_z)]
        else:
            at = layer.self_attn
            groups = [(at.q_proj, at.k_proj, at.v_proj)]
        if hasattr(layer.mlp, "gate_proj"):
            groups.append((layer.mlp.gate_proj, layer.mlp.up_proj))
        for projs in groups:
            if not all(isinstance(p, HadamardLinear) and p.block for p in projs):
                continue
            first = projs[0]
            if any(p.block != first.block or p.signs.shape != first.signs.shape
                   or not mx.array_equal(p.signs, first.signs).item() for p in projs[1:]):
                continue
            group = _SharedInput(len(projs))
            for i, p in enumerate(projs):
                p.__dict__["_shared"] = (group, i)
            linked += 1
    return linked


def unlink_shared_inputs(lm) -> None:
    """Every projection rotates its own input again (the A/B of :func:`link_shared_inputs`)."""
    for _, mod in lm.named_modules():
        if isinstance(mod, HadamardLinear):
            mod.__dict__["_shared"] = None


def _walk(root, path: str):
    parts = path.split(".")
    obj = root
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj, parts[-1]


def _check_config(cfg: dict, where: str) -> tuple[int, int]:
    base = cfg.get("base_model_type", BASE_MODEL_TYPE)
    if base != BASE_MODEL_TYPE:
        raise ValueError(f"{where}: a {MODEL_TYPE} pack over base_model_type {base!r}; only "
                         f"{BASE_MODEL_TYPE!r} (Qwen3.5-family hybrid) packs are supported")
    if not isinstance(cfg.get("text_config"), dict):
        raise ValueError(f"{where}: {MODEL_TYPE} config has no text_config")
    if not isinstance(cfg.get("modules"), list) or not cfg["modules"]:
        raise ValueError(f"{where}: {MODEL_TYPE} config lists no packed modules")
    q = cfg.get("quantization") or {}
    if q.get("mode", "affine") != "affine" or "bits" not in q or "group_size" not in q:
        raise ValueError(f"{where}: {MODEL_TYPE} quantization must be affine with bits and "
                         f"group_size (got {q})")
    return int(q["bits"]), int(q["group_size"])


def _check_module(where, rel, original, embedding, w, s, b, signs, block, bits, gs) -> None:
    kind = nn.Embedding if embedding else nn.Linear
    if type(original) is not kind or "bias" in original:
        raise ValueError(f"{where}: packed module {rel} is recorded as "
                         f"{'an embedding' if embedding else 'a projection'} but the model has "
                         f"{type(original).__name__}" + (" with a bias" if "bias" in original else ""))
    rows, width = original.weight.shape
    if width % gs or (32 * (width * bits // 32)) != width * bits:
        raise ValueError(f"{where}: {rel}: input width {width} does not fit group {gs} / {bits} bits")
    want = [(rows, width * bits // 32), (rows, width // gs), (rows, width // gs)]
    got = [tuple(a.shape) for a in (w, s, b)]
    if got != want or w.dtype != mx.uint32:
        raise ValueError(f"{where}: {rel}: packed tensors {got} ({w.dtype}) do not match a "
                         f"[{rows}, {width}] {bits}-bit group-{gs} matrix {want}")
    if block:
        if block & (block - 1) or width % block:
            raise ValueError(f"{where}: {rel}: Hadamard block {block} is not a power of two "
                             f"dividing the input width {width}")
        if signs is None or tuple(signs.shape) != (width,):
            raise ValueError(f"{where}: {rel}: missing or misshaped sign vector "
                             f"({None if signs is None else tuple(signs.shape)}, want {(width,)})")
    elif signs is not None:
        raise ValueError(f"{where}: {rel}: a sign vector on a module recorded as unrotated")


def _generation_eos(path: str) -> list[int]:
    fp = os.path.join(path, "generation_config.json")
    if not os.path.isfile(fp):
        return []
    with open(fp) as f:
        eos = json.load(f).get("eos_token_id")
    if eos is None:
        return []
    return [int(e) for e in eos] if isinstance(eos, list) else [int(eos)]


def _load_tokenizer(path: str, trust_remote_code: bool):
    from pathlib import Path

    from mlx_lm.utils import load_tokenizer
    from transformers.utils import logging as hf_logging

    # transformers reads config.json on the way and warns that "prism_hadamard_qwen35" is not
    # the generic config class it builds; nothing is instantiated from it, so the warning goes
    prev = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        # the pack's generation_config.json lists every end-of-generation id (<|im_end|> AND
        # <|endoftext|>); the tokenizer config alone names one
        return load_tokenizer(Path(path), {"trust_remote_code": trust_remote_code},
                              eos_token_ids=_generation_eos(path) or None)
    finally:
        hf_logging.set_verbosity(prev)


def load(path: str, cfg: dict | None = None, *, dtype=mx.float16, share: bool = True,
         tokenizer: bool = True, trust_remote_code: bool = False):
    """``(model, tokenizer)`` for a pack directory: mlx-lm's qwen3_5 ``Model`` built from the
    pack's ``text_config``, the recorded modules swapped for :class:`HadamardLinear` /
    :class:`HadamardEmbedding`, the weights loaded strictly as stored (they are already in
    mlx-lm's layout: conv1d moved, norms offset, so no ``sanitize``), the vision tower dropped
    (text serving, as for every qwen3_5 target here), every floating tensor cast to ``dtype``.
    Accepts the text-only (unprefixed) and the multimodal (``language_model.``) namespaces."""
    from mlx.utils import tree_map
    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    if cfg is None:
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
    bits, gs = _check_config(cfg, path)
    model = Model(ModelArgs.from_dict({"model_type": BASE_MODEL_TYPE,
                                       "text_config": dict(cfg["text_config"])}))
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise ValueError(f"{path}: no *.safetensors in the pack")
    weights: dict = {}
    for fp in files:
        for k, v in mx.load(fp).items():
            if k.startswith(_DROP) or k.startswith("mtp.") or ".mtp." in k:
                continue                      # a native MTP block is not part of the target
            weights[k if k.startswith(_NS) else _NS + k] = v

    lm = model.language_model
    seen: set[str] = set()
    signs_all = []
    for rec in cfg["modules"]:
        rel = str(rec["path"])
        if rel in seen:
            raise ValueError(f"{path}: packed module {rel} is listed twice")
        seen.add(rel)
        try:
            parent, name = _walk(lm, rel)
            original = getattr(parent, name)
        except (AttributeError, IndexError, ValueError, KeyError):
            raise ValueError(f"{path}: packed module {rel} is not in the qwen3_5 model built "
                             "from text_config") from None
        key = _NS + rel
        try:
            w, s, b = (weights[f"{key}.{t}"] for t in ("weight", "scales", "biases"))
        except KeyError as e:
            raise ValueError(f"{path}: packed module {rel} has no {e.args[0]}") from None
        block = int(rec.get("block") or 0)
        signs = weights.get(f"{key}.signs")
        embedding = bool(rec.get("embedding"))
        _check_module(path, rel, original, embedding, w, s, b, signs, block, bits, gs)
        cls = HadamardEmbedding if embedding else HadamardLinear
        setattr(parent, name, cls(w, s, b, signs, block, gs, bits))
        if block:
            signs_all.append(signs)
    if signs_all and not mx.all(mx.stack([mx.all(mx.abs(v) == 1) for v in signs_all])).item():
        raise ValueError(f"{path}: a Hadamard sign vector holds values other than +1 / -1")
    model.load_weights(list(weights.items()), strict=True)
    model.update(tree_map(
        lambda v: v.astype(dtype) if mx.issubdtype(v.dtype, mx.floating) else v,
        model.parameters()))
    model.eval()
    mx.eval(model.parameters())
    if share:
        link_shared_inputs(lm)
    tok = _load_tokenizer(path, trust_remote_code) if tokenizer else None
    return model, tok
