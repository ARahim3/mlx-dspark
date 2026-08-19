"""DFlash 2 (incoai/*-DFlash2): grouped dynamic conv + candidate path selector.

Semantics reference = the merged SGLang implementation (sgl-project/sglang
srt/models/dflash.py + srt/speculative/dflash_worker_v2.py) — the tests replicate its
math in numpy and pin our mx port against it, plus the checkpoint name/quantization
contracts for the real `incoai/Qwen3.8-27B-DFlash2` layout (verified from its
safetensors header, 2026-08-19).
"""
import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_dspark.dflash_model import (
    CandidateSelector,
    DFlashConfig,
    DFlashDraftModel,
    DFlashGroupedConv,
)
from mlx_dspark.load import _flatten_params, load_dflash

# Tiny mirror of incoai/Qwen3.8-27B-DFlash2 (5 layers there; 2 here, same key layout).
DFLASH2_MIN = {
    "model_type": "qwen3", "architectures": ["DFlash2DraftModel"], "is_causal": False,
    "hidden_size": 64, "vocab_size": 32, "num_hidden_layers": 2, "intermediate_size": 128,
    "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
    "rms_norm_eps": 1e-6, "max_position_embeddings": 512, "num_target_layers": 4,
    "rope_parameters": {"rope_theta": 1e7, "rope_type": "default"},
    "dflash_config": {
        "block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2,
        "mask_token_id": 3, "selector_rank": 8, "selector_top_k": 4,
        "target_layer_ids": [0, 2],
    },
}


def _tiny_config(**overrides) -> DFlashConfig:
    cfg = DFlashConfig(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, intermediate_size=128, vocab_size=32, rms_norm_eps=1e-6,
        rope_theta=1e7, max_position_embeddings=512, block_size=8,
        target_layer_ids=(0, 2), num_target_layers=4, mask_token_id=3,
        selector_rank=8, selector_top_k=4, conv_kernel_size=2, conv_group_size=16,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _tiny_model(cfg=None):
    """DFlash 2 model with random (non-zero) weights and a fake embed/lm_head bound."""
    mx.random.seed(0)
    model = DFlashDraftModel(cfg or _tiny_config())
    model.embed_tokens = nn.Embedding(32, 64)
    model.lm_head = nn.Linear(64, 32, bias=False)
    model.embed_scale = 1.0
    sel = model.candidate_selector
    sel.predecessor_codebook = mx.random.normal((32, 8))
    sel.successor_codebook = mx.random.normal((32, 8))
    return model


# ---------------------------------------------------------------- grouped conv

def _np_grouped_conv(x, delta, base, group_size):
    """Reference semantics of sglang's _grouped_conv for a single block [L, H]."""
    L, H = x.shape
    taps, G = delta.shape[1], delta.shape[2]
    xg = x.reshape(L, G, group_size)
    coeff = base.reshape(1, taps, G, group_size) + delta[..., None]
    out = coeff[:, 0] * xg
    for t in range(1, taps):
        shifted = np.concatenate([np.zeros((t, G, group_size)), xg[:-t]], axis=0)
        out = out + coeff[:, t] * shifted
    return out.reshape(L, H)


def test_grouped_conv_matches_numpy_reference():
    mx.random.seed(1)
    conv = DFlashGroupedConv(64, taps=2, group_size=16)
    conv.base_kernel = mx.random.normal((2, 2, 64))
    x = mx.random.normal((1, 8, 64))

    convolved, out_coeff = conv.prepare(x)
    finished = conv.finish(mx.array(np.arange(8 * 64, dtype=np.float32).reshape(1, 8, 64) / 100.0),
                           out_coeff)
    mx.eval(convolved, finished)

    xn = np.array(x[0], dtype=np.float32)
    w = np.array(conv.kernel_projection.weight, dtype=np.float32)
    base = np.array(conv.base_kernel, dtype=np.float32)
    delta = (xn @ w.T).reshape(8, 2, 2, 4)                     # [L, side, taps, groups]
    want_in = _np_grouped_conv(xn, delta[:, 0], base[0], 16)
    yn = np.arange(8 * 64, dtype=np.float32).reshape(8, 64) / 100.0
    want_out = _np_grouped_conv(yn, delta[:, 1], base[1], 16)  # out coeffs from the INPUT

    assert np.allclose(np.array(convolved[0]), want_in, atol=1e-4)
    assert np.allclose(np.array(finished[0]), want_out, atol=1e-4)


def test_grouped_conv_identity_at_init_with_zero_projection():
    # base kernel inits to identity (tap 0 = 1); with a zeroed projection the conv is a no-op —
    # the invariant the training-side init relies on.
    conv = DFlashGroupedConv(64, taps=2, group_size=16)
    conv.kernel_projection.weight = mx.zeros_like(conv.kernel_projection.weight)
    x = mx.random.normal((1, 8, 64))
    convolved, out_coeff = conv.prepare(x)
    assert mx.allclose(convolved, x).item()
    assert mx.allclose(conv.finish(x, out_coeff), x).item()


def test_grouped_conv_position_zero_has_no_predecessor_tap():
    # Block position 0 is the anchor slot: its tap-1 term must be zero (the reference masks
    # position % block_size < tap). With base tap-1 = 1 and everything else 0, the output IS
    # the shifted input — and row 0 must come out all zeros.
    conv = DFlashGroupedConv(64, taps=2, group_size=16)
    conv.kernel_projection.weight = mx.zeros_like(conv.kernel_projection.weight)
    conv.base_kernel = mx.stack([
        mx.stack([mx.zeros((64,)), mx.ones((64,))]),   # side 0: pure tap-1
        mx.stack([mx.ones((64,)), mx.zeros((64,))]),   # side 1: identity
    ])
    x = mx.random.normal((1, 8, 64))
    convolved, _ = conv.prepare(x)
    assert mx.allclose(convolved[:, 0], mx.zeros((1, 64))).item()
    assert mx.allclose(convolved[:, 1:], x[:, :-1]).item()


# ---------------------------------------------------------------- candidate selector

def _np_lattice(sel, cand, unary, hidden, anchor):
    pred_cb = np.array(sel.predecessor_codebook, dtype=np.float32)
    suc_cb = np.array(sel.successor_codebook, dtype=np.float32)
    h = np.array(hidden, dtype=np.float32) @ np.array(
        sel.hidden_projection.weight, dtype=np.float32).T
    g, k = cand.shape
    scores = np.zeros((g, k, k), dtype=np.float32)
    for s in range(g):
        for p in range(k):
            pred = anchor if s == 0 else cand[s - 1, p]
            for c in range(k):
                scores[s, p, c] = unary[s, c] + np.dot(pred_cb[pred] * h[s], suc_cb[cand[s, c]])
    return scores


def test_lattice_matches_numpy_reference_including_anchor_slot():
    mx.random.seed(2)
    sel = CandidateSelector(hidden_size=64, vocab_size=32, rank=8, top_k=4)
    sel.predecessor_codebook = mx.random.normal((32, 8))
    sel.successor_codebook = mx.random.normal((32, 8))
    cand = mx.array([[1, 5, 9, 2], [4, 5, 6, 7], [0, 3, 30, 31]], dtype=mx.int32)
    unary = mx.random.normal((3, 4))
    hidden = mx.random.normal((3, 64))

    scores = sel.lattice(cand, unary.astype(mx.float32), hidden, anchor_id=17)
    want = _np_lattice(sel, np.array(cand), np.array(unary), hidden, 17)
    assert scores.shape == (3, 4, 4)
    assert np.allclose(np.array(scores), want, atol=1e-4)
    # slot 0: every predecessor row is the anchor -> all rows identical
    assert np.allclose(want[0], want[0][0:1])


def test_walk_greedy_follows_the_chain_not_per_slot_argmax():
    sel = CandidateSelector(hidden_size=8, vocab_size=32, rank=4, top_k=3)
    cand = mx.array([[10, 11, 12], [20, 21, 22], [30, 31, 32]], dtype=mx.int32)
    # slot 0 (anchor rows identical): argmax -> index 2.
    # edge to slot 1, row 2 -> index 0  (row 0 would pick index 1 — the chain must use row 2).
    # edge to slot 2, row 0 -> index 1.
    scores = mx.array([
        [[0.0, 1.0, 5.0]] * 3,
        [[0.0, 9.0, 0.0], [0.0, 0.0, 0.0], [7.0, 1.0, 0.0]],
        [[0.0, 4.0, 1.0], [9.0, 0.0, 0.0], [0.0, 0.0, 9.0]],
    ])
    picks = sel.walk_greedy(scores, cand)
    assert picks.tolist() == [12, 20, 31]


def test_walk_sampled_near_zero_temperature_equals_greedy():
    mx.random.seed(3)
    sel = CandidateSelector(hidden_size=8, vocab_size=32, rank=4, top_k=4)
    cand = mx.array(np.random.default_rng(0).integers(0, 32, (5, 4)), dtype=mx.int32)
    scores = mx.random.normal((5, 4, 4))
    uniforms = mx.random.uniform(shape=(5,))
    greedy = sel.walk_greedy(scores, cand)
    sampled, q_rows = sel.walk_sampled(scores, cand, uniforms, temperature=1e-6)
    assert sampled.tolist() == greedy.tolist()
    # near-zero temperature: every realized q row is a point mass
    assert np.allclose(np.array(q_rows).max(axis=-1), 1.0, atol=1e-4)


def test_walk_sampled_matches_numpy_inverse_cdf_and_q_rows():
    mx.random.seed(4)
    sel = CandidateSelector(hidden_size=8, vocab_size=32, rank=4, top_k=3)
    cand = mx.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=mx.int32)
    scores = mx.random.normal((4, 3, 3))
    uniforms = mx.array([0.31, 0.77, 0.05, 0.93])
    t = 0.8
    tokens, q_rows = sel.walk_sampled(scores, cand, uniforms, temperature=t)
    mx.eval(tokens, q_rows)

    sn, un = np.array(scores, dtype=np.float32), np.array(uniforms, dtype=np.float32)

    def softmax(v):
        e = np.exp(v / t - (v / t).max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    init = softmax(sn[0, 0])
    idx = min(int((un[0] >= np.cumsum(init)).sum()), 2)
    want_tokens, want_q = [int(np.array(cand)[0, idx])], [init]
    for s in range(1, 4):
        row = softmax(sn[s, idx])
        want_q.append(row)
        idx = min(int((un[s] >= np.cumsum(row)).sum()), 2)
        want_tokens.append(int(np.array(cand)[s, idx]))
    assert tokens.tolist() == want_tokens
    assert np.allclose(np.array(q_rows), np.stack(want_q), atol=1e-5)
    assert np.allclose(np.array(q_rows).sum(axis=-1), 1.0, atol=1e-5)


def test_transform_unary_applies_multiplier_then_softcap():
    model = _tiny_model(_tiny_config(output_multiplier=0.5, final_logit_softcapping=4.0))
    out = model._transform_unary(mx.array([[1.0, -2.0, 8.0]]))
    want = 4.0 * np.tanh(0.5 * np.array([1.0, -2.0, 8.0]) / 4.0)
    assert np.allclose(np.array(out[0]), want, atol=1e-5)
    assert out.dtype == mx.float32


# ---------------------------------------------------------------- select_block

def test_select_block_greedy_shapes_and_in_graph():
    model = _tiny_model()
    cache = model.make_cache()
    fused = mx.random.normal((1, 5, 2 * 64))
    block = mx.array([[7] + [3] * 7])
    ids, cand, q = model.select_block(block, fused, cache, cap=7, anchor_id=7)
    assert cand is None and q is None
    mx.eval(ids)                                   # whole walk evaluates as one graph
    assert ids.shape == (7,)
    assert all(0 <= t < 32 for t in ids.tolist())


def test_select_block_respects_cap_truncation():
    model = _tiny_model()
    ids, _, _ = model.select_block(mx.array([[7] + [3] * 7]), mx.random.normal((1, 5, 128)),
                                   model.make_cache(), cap=3, anchor_id=7)
    assert ids.shape == (3,)


def test_select_block_sampled_returns_scatterable_q():
    model = _tiny_model()
    ids, cand, q_rows = model.select_block(
        mx.array([[7] + [3] * 7]), mx.random.normal((1, 5, 128)), model.make_cache(),
        cap=7, anchor_id=7, uniforms=mx.random.uniform(shape=(7,)), temperature=0.9)
    mx.eval(ids, cand, q_rows)
    assert ids.shape == (7,) and cand.shape == (7, 4) and q_rows.shape == (7, 4)
    dense = mx.put_along_axis(mx.zeros((7, 32)), cand, q_rows, axis=1)
    assert np.allclose(np.array(dense.sum(axis=-1)), 1.0, atol=1e-5)
    # each drafted token sits among its slot's candidates with nonzero q
    for s, tok in enumerate(ids.tolist()):
        assert tok in cand[s].tolist()
        assert float(dense[s, tok]) > 0.0


def test_select_block_candidates_are_the_lm_head_topk():
    model = _tiny_model()
    cache = model.make_cache()
    fused = mx.random.normal((1, 5, 128))
    block = mx.array([[7] + [3] * 7])
    hidden = model.forward_hidden(block, cache=model.make_cache(), target_hidden=fused,
                                  logits_start=1)[0]
    logits = np.array(model.lm_head(hidden), dtype=np.float32)
    _, cand, _ = model.select_block(block, fused, cache, cap=7, anchor_id=7,
                                    uniforms=mx.random.uniform(shape=(7,)))
    want = np.sort(np.argpartition(logits, -4, axis=-1)[:, -4:], axis=-1)
    got = np.sort(np.array(cand), axis=-1)
    assert np.array_equal(got, want)


# ---------------------------------------------------------------- load contract

# The exact tensor-name layout incoai/Qwen3.8-27B-DFlash2 ships (safetensors header,
# 2026-08-19): DFlash 1 keys + per-layer {attention,mlp}_conv + the selector triple.
_LAYER_KEYS = [
    "attention_conv.base_kernel", "attention_conv.kernel_projection.weight",
    "input_layernorm.weight",
    "mlp.down_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
    "mlp_conv.base_kernel", "mlp_conv.kernel_projection.weight",
    "post_attention_layernorm.weight",
    "self_attn.k_norm.weight", "self_attn.k_proj.weight", "self_attn.o_proj.weight",
    "self_attn.q_norm.weight", "self_attn.q_proj.weight", "self_attn.v_proj.weight",
]
_EXPECTED_KEYS = sorted(
    ["candidate_selector.hidden_projection.weight",
     "candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook",
     "fc.weight", "hidden_norm.weight", "norm.weight"]
    + [f"layers.{i}.{k}" for i in range(2) for k in _LAYER_KEYS]
)


def test_dflash2_param_names_match_the_real_checkpoint_layout():
    names = sorted(k for k, _ in _flatten_params(DFlashDraftModel(_tiny_config())))
    assert names == _EXPECTED_KEYS


def test_dflash1_config_has_no_selector_or_conv():
    cfg = _tiny_config(selector_rank=0, selector_top_k=0, conv_kernel_size=0)
    model = DFlashDraftModel(cfg)
    assert model.candidate_selector is None
    assert model.layers[0].attention_conv is None
    names = [k for k, _ in _flatten_params(model)]
    assert not any("_conv" in n or "candidate_selector" in n for n in names)


def _write_dflash2_ckpt(tmp_path, cfg_dict=DFLASH2_MIN):
    model = DFlashDraftModel(_tiny_config())
    (tmp_path / "config.json").write_text(json.dumps(cfg_dict))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), dict(_flatten_params(model)))
    return str(tmp_path)


def test_load_dflash2_roundtrip_and_config_fields(tmp_path):
    path = _write_dflash2_ckpt(tmp_path)
    drafter, cfg = load_dflash(path, quantize=False)
    assert drafter.candidate_selector is not None
    assert drafter.layers[1].mlp_conv is not None
    assert (cfg.selector_rank, cfg.selector_top_k) == (8, 4)
    assert (cfg.conv_kernel_size, cfg.conv_group_size) == (2, 16)
    assert cfg.output_multiplier == 1.0 and cfg.final_logit_softcapping is None


def test_load_dflash2_reads_muse_transform_fields_nested(tmp_path):
    cfg_dict = json.loads(json.dumps(DFLASH2_MIN))
    cfg_dict["dflash_config"]["output_multiplier"] = 0.19611613513818404
    cfg_dict["dflash_config"]["final_logit_softcapping"] = 20.0
    path = _write_dflash2_ckpt(tmp_path, cfg_dict)
    _, cfg = load_dflash(path, quantize=False)
    assert cfg.final_logit_softcapping == 20.0
    assert abs(cfg.output_multiplier - 0.19611613513818404) < 1e-9


def test_load_dflash2_quantizes_backbone_but_not_conv_or_selector(tmp_path):
    path = _write_dflash2_ckpt(tmp_path)
    drafter, _ = load_dflash(path, quantize=True, bits=4, group_size=32)
    assert isinstance(drafter.layers[0].self_attn.q_proj, nn.QuantizedLinear)
    assert isinstance(drafter.fc, nn.QuantizedLinear)
    assert isinstance(drafter.layers[0].attention_conv.kernel_projection, nn.Linear)
    assert not isinstance(drafter.layers[0].attention_conv.kernel_projection, nn.QuantizedLinear)
    assert isinstance(drafter.candidate_selector.hidden_projection, nn.Linear)
    assert not isinstance(drafter.candidate_selector.hidden_projection, nn.QuantizedLinear)


def test_dflash2_architecture_without_selector_fields_refused_with_reason(tmp_path):
    cfg_dict = json.loads(json.dumps(DFLASH2_MIN))
    for k in ("selector_rank", "selector_top_k"):
        del cfg_dict["dflash_config"][k]
    (tmp_path / "config.json").write_text(json.dumps(cfg_dict))
    with pytest.raises(ValueError, match="selector_rank"):
        load_dflash(str(tmp_path))
