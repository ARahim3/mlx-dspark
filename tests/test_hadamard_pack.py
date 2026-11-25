"""Model-free tests for PrismML's Hadamard-rotated packs (hadamard_pack): the transform against
a dense Hadamard matrix, the packed projection / embedding math, a synthetic pack folded from a
random qwen3_5 model loading back onto exactly the weights it represents, the shared rotation,
the hybrid tap and rollback on it, load_target routing, and DFlash head rows."""

from __future__ import annotations

import json
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_dspark import hadamard_pack
from mlx_dspark.hadamard_pack import (
    HadamardEmbedding,
    HadamardLinear,
    link_shared_inputs,
    rotate,
    unlink_shared_inputs,
)
from mlx_dspark.target import Target

BLOCK, GS = 16, 32

TEXT_CONFIG = {
    "model_type": "qwen3_5_text",
    "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 4,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
    "vocab_size": 64, "rms_norm_eps": 1e-6,
    "linear_num_value_heads": 4, "linear_num_key_heads": 2,
    "linear_key_head_dim": 8, "linear_value_head_dim": 8,
    "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
    "tie_word_embeddings": False,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                        "partial_rotary_factor": 0.25},
}


def _hadamard(n: int) -> np.ndarray:
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def _signs(n: int, seed: int) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array(rng.choice([-1.0, 1.0], size=n).astype(np.float32))


def _float_model():
    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    mx.random.seed(3)
    model = Model(ModelArgs.from_dict({"model_type": "qwen3_5", "text_config": TEXT_CONFIG}))
    mx.eval(model.parameters())
    return model


def _packed_names(lm):
    """(name, module, is_embedding) for what Prism packs: every projection and the embedding /
    head, but not the gated-delta gates in_proj_a / in_proj_b (those ship as plain floats)."""
    for name, mod in lm.named_modules():
        if isinstance(mod, nn.Embedding):
            yield name, mod, True
        elif isinstance(mod, nn.Linear) and not name.endswith(("in_proj_a", "in_proj_b")):
            yield name, mod, False


def _write_pack(tmp_path, *, prefix="language_model.", bits=2):
    """Fold a random qwen3_5 model into a pack the way Prism's packer does (every stored row is
    the forward rotation of the float row, then quantized) and return (dir, {name: dense W})
    where dense W is what the pack represents exactly: the dequantized rows rotated back."""
    model = _float_model()
    lm = model.language_model
    signs = {32: _signs(32, 1), 64: _signs(64, 2)}
    tensors, modules, dense, packed = {}, [], {}, set()
    for name, mod, emb in _packed_names(lm):
        width = mod.weight.shape[1]
        s = signs[width]
        wq, sc, bi = mx.quantize(rotate(mod.weight, s, BLOCK), group_size=GS, bits=bits)
        sc, bi = sc.astype(mx.float16), bi.astype(mx.float16)
        key = prefix + name
        tensors.update({f"{key}.weight": wq, f"{key}.scales": sc, f"{key}.biases": bi,
                        f"{key}.signs": s})
        modules.append({"path": name, "block": BLOCK, "embedding": emb, "dtype": "float16"})
        rows = mx.dequantize(wq, sc.astype(mx.float32), bi.astype(mx.float32),
                             group_size=GS, bits=bits)
        dense[name] = rotate(rows, s, BLOCK, inverse=True)
        packed.add(name + ".weight")
    for name, v in tree_flatten(lm.parameters()):
        if name not in packed:
            tensors[prefix + name] = v
    tensors["vision_tower.patch_embed.proj.weight"] = mx.zeros((4, 4))   # dropped on load
    mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
    cfg = {"schema_version": 2, "model_type": hadamard_pack.MODEL_TYPE,
           "text_config": TEXT_CONFIG, "modules": modules,
           "quantization": {"bits": bits, "group_size": GS, "mode": "affine"},
           "base_model_type": "qwen3_5", "tensor_namespace": "mlx-vlm-qwen3_5",
           "components": {"text": True, "vision": True, "mtp": False}}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return tmp_path, dense


def _reference(dense):
    """The float model carrying exactly the weights the pack represents."""
    ref = _float_model()
    lm = ref.language_model
    for name, mod, _ in _packed_names(lm):
        mod.weight = dense[name]
    mx.eval(ref.parameters())
    return ref


def _load(path, **kw):
    """Load a synthetic pack for these tests on mlx-lm's ops path: the tiny head dims sit below
    the gated-delta Metal kernel's minimum (32), and the rest of the suite runs its tiny hybrids
    in training mode for the same reason (the loader itself leaves real packs in eval mode)."""
    model, tok = hadamard_pack.load(str(path), tokenizer=False, **kw)
    model.train()
    return model, tok


def _logits(model, ids):
    return model(mx.array([ids]), cache=model.make_cache())[0]


def test_rotate_matches_dense_hadamard():
    s = _signs(32, 5)
    x = mx.random.normal((2, 3, 32))
    h = _hadamard(BLOCK)
    sx = np.asarray(x) * np.asarray(s)
    want = np.concatenate([sx[..., :16] @ h.T, sx[..., 16:] @ h.T], axis=-1)
    np.testing.assert_allclose(np.asarray(rotate(x, s, BLOCK)), want, atol=1e-5)
    back = rotate(rotate(x, s, BLOCK), s, BLOCK, inverse=True)
    np.testing.assert_allclose(np.asarray(back), np.asarray(x), atol=1e-5)
    assert rotate(x.astype(mx.bfloat16), s, BLOCK, dtype=mx.float16).dtype == mx.float16


def test_projection_and_embedding_math():
    mx.random.seed(0)
    w = mx.random.normal((8, 32))
    s = _signs(32, 9)
    wq, sc, bi = mx.quantize(rotate(w, s, BLOCK), group_size=GS, bits=8)
    lin = HadamardLinear(wq, sc, bi, s, BLOCK, GS, 8)
    x = mx.random.normal((1, 2, 32))
    np.testing.assert_allclose(np.asarray(lin(x)), np.asarray(x @ w.T), atol=0.05, rtol=0.05)
    ids = mx.array([[1, 5], [7, 0]])
    rows = lin.dense_rows(ids)
    assert rows.shape == (2, 2, 32)
    np.testing.assert_allclose(np.asarray(rows), np.asarray(w[ids]), atol=0.02)
    np.testing.assert_allclose(np.asarray(lin(x)[0, 0, 5]),
                               np.asarray(x[0, 0] @ lin.dense_rows(mx.array([5]))[0]), atol=1e-4)

    emb = HadamardEmbedding(wq, sc, bi, s, BLOCK, GS, 8)
    np.testing.assert_allclose(np.asarray(emb(ids)), np.asarray(w[ids]), atol=0.02)
    np.testing.assert_allclose(np.asarray(emb.as_linear(x)), np.asarray(x @ w.T), atol=0.05,
                               rtol=0.05)


@pytest.mark.parametrize("prefix", ["language_model.", ""])
def test_pack_loads_onto_the_weights_it_represents(tmp_path, prefix):
    path, dense = _write_pack(tmp_path, prefix=prefix)
    model, tok = _load(path, dtype=mx.float32)
    assert tok is None
    lm = model.language_model
    assert isinstance(lm.model.embed_tokens, HadamardEmbedding)
    assert isinstance(lm.lm_head, HadamardLinear)
    la = lm.model.layers[0].linear_attn
    assert isinstance(la.in_proj_qkv, HadamardLinear) and type(la.in_proj_a) is nn.Linear
    ids = [3, 9, 1, 30, 2, 44, 5]
    got, want = _logits(model, ids), _logits(_reference(dense), ids)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), atol=2e-3, rtol=2e-3)


def test_pack_default_dtype_is_fp16(tmp_path):
    path, _ = _write_pack(tmp_path)
    model, _ = _load(path)
    floats = {v.dtype for _, v in tree_flatten(model.parameters())
              if mx.issubdtype(v.dtype, mx.floating)}
    assert floats == {mx.float16}
    assert _logits(model, [1, 2, 3]).dtype == mx.float16


def test_shared_rotation_is_exact_and_releases_its_activation(tmp_path):
    path, _ = _write_pack(tmp_path)
    model, _ = _load(path, dtype=mx.float32)
    lm = model.language_model
    groups = {id(m.__dict__["_shared"][0]) for _, m in lm.named_modules()
              if isinstance(m, HadamardLinear) and m.__dict__["_shared"] is not None}
    assert len(groups) == 2 * len(lm.model.layers)       # (q,k,v | qkv,z) and (gate, up) per layer
    ids = [4, 8, 15, 16, 23, 42]
    shared = _logits(model, ids)
    pending = [m.__dict__["_shared"][0].local.__dict__.get("pending")
               for _, m in lm.named_modules()
               if isinstance(m, HadamardLinear) and m.__dict__["_shared"] is not None]
    assert all(p is None for p in pending)                # nothing outlives its layer
    unlink_shared_inputs(lm)
    assert bool(mx.array_equal(shared, _logits(model, ids)).item())
    assert link_shared_inputs(lm) == len(groups)


def test_compiled_rotation_is_bit_identical(tmp_path, monkeypatch):
    s = _signs(64, 4)
    for dtype in (mx.float16, mx.float32):
        x = (mx.random.normal((1, 5, 64)) * 3).astype(dtype)
        for inverse in (False, True):
            monkeypatch.setattr(hadamard_pack, "COMPILE", True)
            a = rotate(x, s, BLOCK, inverse=inverse)
            monkeypatch.setattr(hadamard_pack, "COMPILE", False)
            b = rotate(x, s, BLOCK, inverse=inverse)
            assert a.dtype == b.dtype == dtype
            assert bool(mx.array_equal(a, b).item())
    path, _ = _write_pack(tmp_path)
    model, _ = _load(path)
    ids = [7, 1, 33, 2, 9]
    monkeypatch.setattr(hadamard_pack, "COMPILE", True)
    on = _logits(model, ids)
    monkeypatch.setattr(hadamard_pack, "COMPILE", False)
    assert bool(mx.array_equal(on, _logits(model, ids)).item())


def test_out_of_order_call_rotates_its_own_input(tmp_path):
    path, _ = _write_pack(tmp_path)
    model, _ = _load(path, dtype=mx.float32)
    mlp = model.language_model.model.layers[0].mlp
    x, y = mx.random.normal((1, 1, 32)), mx.random.normal((1, 1, 32))
    gate_x = mlp.gate_proj(x)                  # leaves x's rotation for up_proj ...
    up_y = mlp.up_proj(y)                      # ... which gets another input: rotates its own
    unlink_shared_inputs(model.language_model)
    assert bool(mx.array_equal(up_y, mlp.up_proj(y)).item())
    assert bool(mx.array_equal(gate_x, mlp.gate_proj(x)).item())


def test_hybrid_tap_and_rollback_on_a_pack(tmp_path):
    path, _ = _write_pack(tmp_path)
    model, _ = _load(path, dtype=mx.float32)
    t = Target(model, tokenizer=None)
    assert t.is_hybrid and t._recurrence == "gated_delta"
    t.verify_tap()                                        # replicated loop == the model's own
    prompt, committed = [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
    ref_cache = t.make_cache()
    ref, _ = t.run(mx.array([committed]), ref_cache, [0])
    cache = t.make_cache()
    t.reset_spec()
    t.run(mx.array([prompt]), cache, [0])
    t.verify(mx.array([[6, 7, 8, 9]]), cache, [0])
    t.rollback(cache, 2, [7])                             # keep 6, 7; reject 8, 9
    v2, _ = t.verify(mx.array([[10, 11, 12]]), cache, [0])
    np.testing.assert_allclose(np.asarray(v2[0]), np.asarray(ref[0, -3:]), atol=1e-4, rtol=1e-4)


def test_load_target_routes_a_pack(tmp_path, monkeypatch):
    from mlx_dspark import load as load_mod

    path, _ = _write_pack(tmp_path)
    monkeypatch.setattr(hadamard_pack, "_load_tokenizer", lambda p, trc: "tok")
    real_load = hadamard_pack.load

    def ops_path_load(*args, **kwargs):            # see _load
        model, tok = real_load(*args, **kwargs)
        model.train()
        return model, tok

    monkeypatch.setattr(hadamard_pack, "load", ops_path_load)
    target, tok = load_mod.load_target(str(path), require_tap=True)
    assert tok == "tok" and target.is_hybrid
    assert isinstance(target.model.language_model.lm_head, HadamardLinear)


def test_dflash_head_rows_rotate_a_pack_head_back():
    from mlx_dspark.dflash_model import DFlashDraftModel

    mx.random.seed(1)
    w = mx.random.normal((10, 32))
    s = _signs(32, 3)
    wq, sc, bi = mx.quantize(rotate(w, s, BLOCK), group_size=GS, bits=8)
    head = HadamardLinear(wq, sc, bi, s, BLOCK, GS, 8)
    ids = mx.array([2, 7])
    rows = DFlashDraftModel.head_rows(SimpleNamespace(_head_rows_src=head), ids)
    assert bool(mx.array_equal(rows, head.dense_rows(ids)).item())
    np.testing.assert_allclose(np.asarray(rows), np.asarray(w[ids]), atol=0.02)


def _set(key, value):
    def edit(cfg):
        cfg[key] = value
    return edit


def _set_module0(key, value=None, flip=False):
    def edit(cfg):
        rec = cfg["modules"][0]
        rec[key] = (not rec[key]) if flip else value
    return edit


@pytest.mark.parametrize("edit, message", [
    (_set("base_model_type", "llama"), "base_model_type"),
    (_set("quantization", {"bits": 2, "group_size": GS, "mode": "mxfp4"}), "affine"),
    (_set_module0("block", 12), "power of two"),
    (_set_module0("embedding", flip=True), "recorded as"),
    (_set_module0("block", 0), "sign vector"),
    (_set_module0("path", "model.layers.9.mlp.up_proj"), "not in the qwen3_5 model"),
])
def test_pack_refusals(tmp_path, edit, message):
    path, _ = _write_pack(tmp_path)
    cfg = json.loads((path / "config.json").read_text())
    edit(cfg)
    (path / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match=message):
        hadamard_pack.load(str(path), tokenizer=False)
