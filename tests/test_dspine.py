"""DSpine (Tencent, arXiv:2609.36173): the code bridge, block-causal attention and its exact
block truncation, the last-write refinement table, the shared lattice walk, the load contract
for ``tencent/Qwen3.8-27B-DSpine-b16`` (tensor layout from its safetensors header, 2026-11-08)
— and the DFlash block-mask semantics (z-lab / SGLang ``is_causal``) the port now honors.

Semantics reference = Tencent/DSpine (Apache-2.0): ``sglang_dspine/models/dspine.py`` +
``kernels/dspine.py`` and the authors' ``dspine_mlx`` port; the tests replicate the math in
numpy and pin the mx port against it.
"""
import json
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models.base import create_causal_mask

from mlx_dspark.calibrate import dflash_cost_by_cap, measure_dflash_drafter_curve, tree_capable
from mlx_dspark.dflash_model import (
    DFlashAttention,
    DFlashConfig,
    DFlashDraftModel,
    walk_greedy,
)
from mlx_dspark.load import _flatten_params, load_dflash

H, V, C, R, K, L = 64, 32, 16, 8, 4, 2
_TINY = 1.1754943508222875e-38


def _cfg(**over) -> DFlashConfig:
    cfg = DFlashConfig(
        hidden_size=H, num_hidden_layers=L, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, intermediate_size=128, vocab_size=V, rms_norm_eps=1e-6, rope_theta=1e7,
        max_position_embeddings=512, block_size=8, target_layer_ids=(0, 2), num_target_layers=4,
        mask_token_id=3, block_causal=True, bridge_code_size=C, bridge_rank=R, bridge_eps=1e-6,
        bridge_layers=(0, 1), bridge_top_k=K,
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _randomize_bridge(bridge, seed=0, scale=0.3):
    mx.random.seed(seed)
    bridge.codebook = mx.random.normal(bridge.codebook.shape)
    bridge.read_weight = scale * mx.random.normal(bridge.read_weight.shape)
    bridge.early_read_weights = scale * mx.random.normal(bridge.early_read_weights.shape)
    bridge.feature_write_weights = scale * mx.random.normal(bridge.feature_write_weights.shape)
    bridge.gate.weight = scale * mx.random.normal(bridge.gate.weight.shape)
    bridge.gate.bias = 0.1 * mx.random.normal(bridge.gate.bias.shape)
    bridge.predecessor_gate.weight = scale * mx.random.normal(bridge.predecessor_gate.weight.shape)
    bridge.prev.weight = scale * mx.random.normal(bridge.prev.weight.shape)
    bridge.out.weight = mx.zeros_like(bridge.out.weight)
    bridge.reset()


def _model(cfg=None, seed=0) -> DFlashDraftModel:
    """A DSpine drafter with random (non-zero) weights and a fake embed / lm_head bound."""
    mx.random.seed(seed)
    model = DFlashDraftModel(cfg or _cfg())
    _randomize_bridge(model.selected_prefix_bridge, seed)
    model.embed_tokens = nn.Embedding(V, H)
    model.lm_head = nn.Linear(H, V, bias=False)
    model._head_rows_src = model.lm_head
    model.embed_scale = 1.0
    mx.eval(model.parameters())
    return model


def _np(a):
    return np.array(a, dtype=np.float32)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _np_bridge(bridge, h, ids, layer, norm_w, norm_eps):
    """The paper's Eq. 4-5 + the fused next norm, on one block [T, H]."""
    f = _np(h)[0]
    T = f.shape[0]
    rms = np.sqrt(np.maximum((f * f).mean(-1), _TINY))
    normalized = f / (rms[:, None] + bridge.eps)
    proj = normalized @ _np(bridge.read_weight_for(layer)).T
    codes = math.sqrt(C) * proj / (np.linalg.norm(proj, axis=-1, keepdims=True) + bridge.eps)
    codes[0] = math.sqrt(C) * _np(bridge.codebook)[int(ids[0, 0])]
    q = codes @ _np(bridge.gate.weight).T
    s = codes @ _np(bridge.predecessor_gate.weight).T
    v = codes @ _np(bridge.prev.weight).T
    gate = _sigmoid(q[1:] + s[:-1] + _np(bridge.gate.bias))
    delta = (gate * v[:-1]) @ _np(bridge.feature_write_weights[layer]).T
    out = f.copy()
    out[1:] += rms[1:, None] * delta
    normed = out / np.sqrt((out * out).mean(-1, keepdims=True) + norm_eps) * _np(norm_w)
    assert out.shape[0] == T
    return out, normed, {"raw": f[1:], "rms": rms[1:], "q": q[1:]}


# ---------------------------------------------------------------- the bridge write

def test_bridge_write_matches_numpy_reference_and_stashes_pre_write_state():
    model = _model()
    bridge = model.selected_prefix_bridge
    norm = nn.RMSNorm(H, eps=1e-6)
    norm.weight = mx.random.normal((H,))
    h = mx.random.normal((1, 6, H))
    ids = mx.array([[7, 3, 3, 3, 3, 3]])
    for layer in (0, 1):
        normed, out, stash = bridge(h, ids, layer, norm, want_stash=True)
        mx.eval(normed, out)
        want_out, want_normed, want_stash = _np_bridge(bridge, h, np.array(ids), layer,
                                                       norm.weight, norm.eps)
        assert np.allclose(_np(out[0]), want_out, atol=1e-4)
        assert np.allclose(_np(normed[0]), want_normed, atol=1e-4)
        assert np.allclose(_np(stash["raw"][0]), want_stash["raw"], atol=1e-5)
        assert np.allclose(_np(stash["rms"][0]), want_stash["rms"], atol=1e-5)
        assert np.allclose(_np(stash["q"][0]), want_stash["q"], atol=1e-4)
    assert bridge(h, ids, 0, norm)[2] is None               # no stash unless asked


def test_bridge_identity_with_zero_writes_and_anchor_never_written():
    model = _model()
    bridge = model.selected_prefix_bridge
    norm = nn.RMSNorm(H, eps=1e-6)
    h = mx.random.normal((1, 5, H))
    ids = mx.array([[9, 3, 3, 3, 3]])
    _, out, _ = bridge(h, ids, 0, norm)
    assert mx.allclose(out[:, 0], h[:, 0]).item()           # the anchor row is untouched
    assert not mx.allclose(out[:, 1:], h[:, 1:]).item()     # the others are written
    bridge.feature_write_weights = mx.zeros_like(bridge.feature_write_weights)
    normed, out, _ = bridge(h, ids, 1, norm)                # W = 0 -> identity (the init state)
    assert mx.allclose(out, h).item()
    assert mx.allclose(normed, norm(h), atol=1e-6).item()


def test_bridge_messages_flow_forward_only():
    """Perturbing row t changes rows t (its own gate) and t+1 (its message) and nothing before."""
    model = _model()
    bridge = model.selected_prefix_bridge
    norm = nn.RMSNorm(H, eps=1e-6)
    ids = mx.array([[9, 3, 3, 3, 3, 3]])
    h = mx.random.normal((1, 6, H))
    h2 = mx.concatenate([h[:, :3], h[:, 3:4] + 1.0, h[:, 4:]], axis=1)
    a = bridge(h, ids, 0, norm)[1]
    b = bridge(h2, ids, 0, norm)[1]
    diff = _np(mx.abs(a - b).max(axis=-1))[0]
    assert np.all(diff[:3] == 0.0)
    assert diff[3] > 0 and diff[4] > 0
    assert diff[5] == 0.0


# ---------------------------------------------------------------- block-causal truncation

def _fused(n=5, seed=1):
    mx.random.seed(seed)
    return mx.random.normal((1, n, 2 * H))


def test_causal_block_truncation_is_exact_including_the_stash():
    """Rows 0..3 of a 4-row draft equal rows 0..3 of the full 8-row draft (block-causal
    attention + forward-only messages) — what lets a DSpine round draft only cap + 1 rows."""
    model = _model()
    fused = _fused()
    full, st_full = model._forward(mx.array([[7] + [3] * 7]), fused, model.make_cache(),
                                   logits_start=1, want_stash=True)
    short, st_short = model._forward(mx.array([[7] + [3] * 3]), fused, model.make_cache(),
                                     logits_start=1, want_stash=True)
    mx.eval(full, short)
    assert short.shape == (1, 3, H)
    assert np.allclose(_np(full[:, :3]), _np(short), atol=1e-4)
    for key in ("raw", "rms", "q"):
        assert np.allclose(_np(st_full[key][:, :3]), _np(st_short[key]), atol=1e-4)
    assert model.block_causal
    assert model.draft_width(3) == 4 and model.draft_width(7) == 8 and model.draft_width(40) == 8


def test_bidirectional_block_is_not_truncatable_negative_control():
    model = _model(_cfg(block_causal=False))
    fused = _fused()
    full = model.forward_hidden(mx.array([[7] + [3] * 7]), fused, model.make_cache(), logits_start=1)
    short = model.forward_hidden(mx.array([[7] + [3] * 3]), fused, model.make_cache(), logits_start=1)
    assert not np.allclose(_np(full[:, :3]), _np(short), atol=1e-4)
    assert not model.block_causal and model.draft_width(3) == 8


def test_draft_width_follows_the_zlab_default_when_undeclared():
    full = DFlashDraftModel(_cfg(block_causal=None, bridge_rank=0))        # all full_attention
    assert not full.block_causal and full.draft_width(2) == 8
    sliding = DFlashDraftModel(_cfg(block_causal=None, bridge_rank=0,
                                    layer_types=("sliding_attention",) * L, sliding_window=64))
    assert sliding.block_causal and sliding.draft_width(2) == 3


# ---------------------------------------------------------------- the refinement table

def _np_table(bridge, raw, rms, q, cand, anchor, head_w, norm_w, norm_eps):
    k = cand.shape[1]
    sid = np.concatenate([np.full((1, k), anchor), cand[:-1]], axis=0)
    e = math.sqrt(C) * _np(bridge.codebook)[sid]                                  # [n, K, C]
    gate = _sigmoid(q[:, None, :] + e @ _np(bridge.predecessor_gate.weight).T + _np(bridge.gate.bias))
    value = e @ _np(bridge.prev.weight).T
    w = _np(bridge.feature_write_weights[-1])
    refined = raw[:, None, :] + rms[:, None, None] * ((gate * value) @ w.T)       # [n, K, H]
    normalized = refined / np.sqrt((refined ** 2).mean(-1, keepdims=True) + norm_eps) * _np(norm_w)
    rows = _np(head_w)[cand]                                                      # [n, K, H]
    return np.einsum("nsh,nch->nsc", normalized, rows), gate * value


def test_transition_table_matches_numpy_reference():
    model = _model()
    bridge = model.selected_prefix_bridge
    n = 5
    raw = mx.random.normal((n, H))
    rms = mx.abs(mx.random.normal((n,))) + 0.5
    q = mx.random.normal((n, R))
    cand = mx.array(np.random.default_rng(0).integers(0, V, (n, K)), dtype=mx.int32)
    table = bridge.transition_table(raw, rms, q, cand, 11, model.head_rows,
                                    model.norm.weight, model.norm.eps)
    mx.eval(table)
    want, _ = _np_table(bridge, _np(raw), _np(rms), _np(q), np.array(cand), 11,
                        model.lm_head.weight, model.norm.weight, model.norm.eps)
    assert table.shape == (n, K, K)
    assert np.allclose(_np(table), want, atol=1e-3)
    assert np.allclose(want[0], want[0][0:1])              # slot 0: every sender is the anchor


def test_greedy_argmax_equals_the_reference_linearized_table():
    """The reference's greedy kernel scores `<u·norm_w, E_v> + rms·<m, W^T(E_v·norm_w)>` — the
    refined state WITHOUT the final RMSNorm's per-row scale. A positive per-row scale never
    moves an argmax over the candidates, so our normalized table picks the same chain."""
    model = _model()
    bridge = model.selected_prefix_bridge
    n = 6
    rng = np.random.default_rng(1)
    raw, rms, q = rng.normal(size=(n, H)).astype(np.float32), rng.uniform(0.5, 2, n).astype(np.float32), \
        rng.normal(size=(n, R)).astype(np.float32)
    cand = rng.integers(0, V, (n, K))
    table, message = _np_table(bridge, raw, rms, q, cand, 5, model.lm_head.weight,
                               model.norm.weight, model.norm.eps)
    head = _np(model.lm_head.weight)[cand] * _np(model.norm.weight)                # [n, K, H]
    base = np.einsum("nh,nch->nc", raw, head)                                     # [n, K]
    tg = head @ _np(bridge.feature_write_weights[-1])                             # [n, K, R]
    linear = base[:, None, :] + rms[:, None, None] * np.einsum("nsr,ncr->nsc", message, tg)
    assert np.array_equal(table.argmax(-1), linear.argmax(-1))
    assert not np.allclose(table, linear)                  # the tables differ, the picks don't


# ---------------------------------------------------------------- select_block / lattice

def test_select_block_greedy_walks_the_table_in_graph():
    model = _model()
    fused = _fused()
    block = mx.array([[7] + [3] * 7])
    ids, cand, q = model.select_block(block, fused, model.make_cache(), cap=7, anchor_id=7)
    assert cand is None and q is None
    mx.eval(ids)                                            # the whole round is one graph
    assert ids.shape == (7,)
    cand_l, scores, chain = model.select_lattice(block, fused, model.make_cache(), cap=7,
                                                 anchor_id=7)
    assert cand_l.shape == (7, K) and scores.shape == (7, K, K)
    assert chain.tolist() == ids.tolist()
    assert chain.tolist() == walk_greedy(scores, cand_l).tolist()
    for s, tok in enumerate(ids.tolist()):
        assert tok in cand_l[s].tolist()


def test_select_block_truncated_cap_equals_the_full_block_prefix():
    model = _model()
    fused = _fused()
    full = model.select_block(mx.array([[7] + [3] * 7]), fused, model.make_cache(), cap=7,
                              anchor_id=7)[0]
    short = model.select_block(mx.array([[7] + [3] * 3]), fused, model.make_cache(), cap=3,
                               anchor_id=7)[0]
    assert short.shape == (3,) and short.tolist() == full.tolist()[:3]


def test_select_block_sampled_returns_scatterable_q():
    model = _model()
    ids, cand, q_rows = model.select_block(
        mx.array([[7] + [3] * 7]), _fused(), model.make_cache(), cap=7, anchor_id=7,
        uniforms=mx.random.uniform(shape=(7,)), temperature=0.9)
    mx.eval(ids, cand, q_rows)
    assert ids.shape == (7,) and cand.shape == (7, K) and q_rows.shape == (7, K)
    dense = mx.put_along_axis(mx.zeros((7, V)), cand, q_rows, axis=1)
    assert np.allclose(_np(dense.sum(axis=-1)), 1.0, atol=1e-5)
    for s, tok in enumerate(ids.tolist()):
        assert tok in cand[s].tolist() and float(dense[s, tok]) > 0.0


def test_candidates_are_the_head_topk_of_the_bridged_hidden():
    model = _model()
    fused = _fused()
    block = mx.array([[7] + [3] * 7])
    hidden = model.forward_hidden(block, fused, model.make_cache(), logits_start=1)[0]
    logits = _np(model.lm_head(hidden))
    cand, _, _ = model.select_lattice(block, fused, model.make_cache(), cap=7, anchor_id=7)
    want = np.sort(np.argpartition(logits, -K, axis=-1)[:, -K:], axis=-1)
    assert np.array_equal(np.sort(np.array(cand), axis=-1), want)


def test_head_rows_dequantizes_a_quantized_head():
    model = _model()
    qhead = nn.QuantizedLinear(H, V, bias=False, group_size=32, bits=4)
    model._head_rows_src = qhead
    ids = mx.array([[1, 5], [30, 0]], dtype=mx.int32)
    rows = model.head_rows(ids)
    full = mx.dequantize(qhead.weight, qhead.scales, qhead.biases, group_size=32, bits=4)
    assert rows.shape == (2, 2, H)
    assert mx.allclose(rows.astype(mx.float32), full[ids].astype(mx.float32), atol=1e-2).item()
    model._head_rows_src = None
    with pytest.raises(RuntimeError, match="bind"):
        model.head_rows(ids)


def test_tree_capable_and_cost_by_cap_recognize_a_dspine_head():
    model = _model()
    assert model.selects_paths and tree_capable(model) and dflash_cost_by_cap(model)
    plain = DFlashDraftModel(_cfg(bridge_rank=0, block_causal=None))
    assert not plain.selects_paths and not tree_capable(plain) and not dflash_cost_by_cap(plain)
    curve = measure_dflash_drafter_curve(model, [1, 3, 7])
    assert sorted(curve) == [1, 3, 7] and all(v > 0 for v in curve.values())


# ---------------------------------------------------------------- generate end to end

def _tiny_dspine_for(target):
    cfg = DFlashConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, intermediate_size=64, vocab_size=64, rms_norm_eps=1e-6, rope_theta=1e7,
        max_position_embeddings=512, block_size=8, target_layer_ids=(0, 1),
        num_target_layers=2, mask_token_id=3, block_causal=True, bridge_code_size=C,
        bridge_rank=R, bridge_layers=(0, 1), bridge_top_k=K,
    )
    mx.random.seed(23)
    d = DFlashDraftModel(cfg)
    _randomize_bridge(d.selected_prefix_bridge, seed=23)
    mx.eval(d.parameters())
    d.bind(target.model)
    return d


def _tree_fixtures():
    from test_tree import PROMPT, _tiny_dense, _tiny_hybrid, _Tok

    return PROMPT, _Tok, {"dense": _tiny_dense, "hybrid": _tiny_hybrid}


@pytest.mark.parametrize("kind", ["dense", "hybrid"])
def test_dflash_generate_with_a_dspine_head_reproduces_greedy(kind):
    from mlx_dspark.generate import dflash_generate, greedy_generate
    from mlx_dspark.target import Target

    prompt, tok, makers = _tree_fixtures()
    t = Target(makers[kind](), tokenizer=None)
    drafter = _tiny_dspine_for(t)
    ref = greedy_generate(t, tok(), prompt_ids=prompt, max_new_tokens=40).token_ids
    chain = dflash_generate(t, tok(), drafter, prompt_ids=prompt, max_new_tokens=40)
    assert chain.token_ids[:40] == ref[:40] and chain.tree_rounds == 0
    capped = dflash_generate(t, tok(), drafter, prompt_ids=prompt, max_new_tokens=40,
                             max_draft_tokens=3)                  # truncated 4-row blocks
    assert capped.token_ids[:40] == ref[:40]
    assert all(n <= 4 for n in capped.accept_lengths)
    for tree in (0, 11):
        res = dflash_generate(t, tok(), drafter, prompt_ids=prompt, max_new_tokens=40, tree=tree)
        assert res.token_ids[:40] == ref[:40]
    sampled = dflash_generate(t, tok(), drafter, prompt_ids=prompt, max_new_tokens=8,
                              temperature=0.7, seed=1)
    assert len(sampled.token_ids) >= 1 and sampled.tree_rounds == 0


# ---------------------------------------------------------------- block mask (is_causal)

def _attn(layer_type, block_causal, window=16):
    cfg = _cfg(bridge_rank=0, block_causal=block_causal, layer_types=(layer_type,) * L,
               sliding_window=window if layer_type == "sliding_attention" else None)
    return DFlashAttention(cfg, 0)


def _zlab_mask(L_, ctx_len, window, causal):
    query = ctx_len + np.arange(L_)[:, None]
    key = np.arange(ctx_len + L_)[None]
    context = (key < ctx_len) & (query - key < window)
    block = key >= ctx_len
    if causal:
        block = block & (key <= query)
    return context | block


def test_block_mask_semantics_follow_zlab_and_sglang():
    # full-attention layer: bidirectional unless declared causal (DSpine's 8B head)
    assert _attn("full_attention", None)._block_mask(8, 40) is None
    assert _attn("full_attention", True)._block_mask(8, 40) == "causal"
    assert _attn("full_attention", False)._block_mask(8, 40) is None
    # sliding layer, undeclared -> causal (the z-lab default = the pre-DSpine behavior)
    a = _attn("sliding_attention", None)
    assert a._block_mask(4, 8) == "causal"                            # ctx + block fits
    m = a._block_mask(4, 14)                                          # windowed, causal
    assert np.array_equal(np.array(m), np.array(create_causal_mask(4, offset=14, window_size=16)))
    assert np.array_equal(np.array(m), _zlab_mask(4, 14, 16, causal=True))
    # sliding layer, `is_causal: false` (the shipped DFlash 2 heads): bidirectional block over a
    # windowed context — the reference's mask, NOT the causal one the port used to apply
    b = _attn("sliding_attention", False)
    assert b._block_mask(4, 8) is None
    mb = b._block_mask(4, 14)
    assert np.array_equal(np.array(mb), _zlab_mask(4, 14, 16, causal=False))
    assert not np.array_equal(np.array(mb), np.array(m))
    # every block row sees every block row, and only the ctx rows inside its window
    assert np.all(np.array(mb)[:, 14:])
    assert not np.array(mb)[3, 0] and np.array(mb)[3, 2]


def test_is_causal_false_changes_the_sliding_block_output():
    """Numeric control: with is_causal False the block rows attend each other both ways."""
    mx.random.seed(5)
    causal, bidir = _attn("sliding_attention", None), _attn("sliding_attention", False)
    bidir.update(causal.parameters())
    from mlx_dspark.dflash_model import CtxRotatingKVCache, _build_rope

    rope = _build_rope(16, 1e7, 512, None)
    x = mx.random.normal((1, 4, H))
    ctx = mx.random.normal((1, 6, H))
    out_c = causal(x, ctx, rope, CtxRotatingKVCache(max_size=15))
    out_b = bidir(x, ctx, rope, CtxRotatingKVCache(max_size=15))
    assert not mx.allclose(out_c[:, 0], out_b[:, 0], atol=1e-5).item()   # row 0 now sees rows 1..3
    assert mx.allclose(out_c[:, 3], out_b[:, 3], atol=1e-5).item()       # the last row saw all anyway


# ---------------------------------------------------------------- load contract

# The exact tensor-name layout tencent/Qwen3.8-27B-DSpine-b16 ships (safetensors header,
# 2026-11-08): the DFlash 1 keys + the bridge under selected_prefix_bridge. + four int64 stamps.
_LAYER_KEYS = [
    "input_layernorm.weight",
    "mlp.down_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
    "post_attention_layernorm.weight",
    "self_attn.k_norm.weight", "self_attn.k_proj.weight", "self_attn.o_proj.weight",
    "self_attn.q_norm.weight", "self_attn.q_proj.weight", "self_attn.v_proj.weight",
]
_BRIDGE_KEYS = [
    "selected_prefix_bridge.codebook", "selected_prefix_bridge.early_read_weights",
    "selected_prefix_bridge.feature_write_weights", "selected_prefix_bridge.gate.bias",
    "selected_prefix_bridge.gate.weight", "selected_prefix_bridge.out.weight",
    "selected_prefix_bridge.predecessor_gate.weight", "selected_prefix_bridge.prev.weight",
    "selected_prefix_bridge.read_weight",
]
_STAMPS = [
    "draft_block_causal_version", "last_write_refine_version",
    "selected_prefix_bridge.independent_write_version",
    "selected_prefix_bridge.terminal_disabled_version",
]
_EXPECTED_KEYS = sorted(["fc.weight", "hidden_norm.weight", "norm.weight"] + _BRIDGE_KEYS
                        + [f"layers.{i}.{k}" for i in range(L) for k in _LAYER_KEYS])

# Tiny mirror of tencent/Qwen3.8-27B-DSpine-b16's config.json (5 layers there; 2 here).
DSPINE_MIN = {
    "architectures": ["DFlashDraftModel"], "model_type": "qwen3", "is_causal": True,
    "attention_bias": False, "hidden_act": "silu", "tie_word_embeddings": False,
    "hidden_size": H, "vocab_size": V, "draft_vocab_size": V, "num_hidden_layers": L,
    "intermediate_size": 128, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
    "rms_norm_eps": 1e-6, "max_position_embeddings": 512, "num_target_layers": 4,
    "block_size": 8, "rope_parameters": {"rope_theta": 1e7, "rope_type": "default"},
    "rope_scaling": None, "rope_theta": 1e7,
    "layer_types": ["sliding_attention"] * L, "sliding_window": 64, "use_sliding_window": True,
    "code_bridge_second_pass": {"first_exact": L - 1, "head_topk": K, "mode": "serial_table"},
    "dflash_config": {
        "context_fusion_checkpointing": True, "decoder_checkpointing": True,
        "draft_block_causal": True, "mask_token_id": 3,
        "selected_prefix_code_bridge": {"code_size": C, "eps": 1e-6, "kappa": 10.0, "rank": R},
        "selected_prefix_decode_mode": "serial", "selected_prefix_feature_gt_training": True,
        "selected_prefix_feature_layers": [1, 2], "selected_prefix_feature_teacher_layers": [1],
        "selected_prefix_feature_teacher_schedule": {"hold_fraction": 0.17, "kind": "block_bernoulli",
                                                     "peak_probability": 0.5, "zero_fraction": 0.33},
        "selected_prefix_feature_write_mode": "independent", "selected_prefix_feature_writes": True,
        "selected_prefix_last_write_refine": {"head_topk": K, "kind": "exact_sender",
                                              "version": 1, "weight": 1.0},
        "selected_prefix_terminal": False, "target_layer_ids": [0, 2],
    },
}


def test_dspine_param_names_match_the_real_checkpoint_layout():
    names = sorted(k for k, _ in _flatten_params(DFlashDraftModel(_cfg())))
    assert names == _EXPECTED_KEYS
    assert not any(n in names for n in _STAMPS)            # stamps are checked, not parameters


def _write_ckpt(tmp_path, cfg_dict=DSPINE_MIN, *, stamps=None, nonzero_out=False):
    mx.random.seed(0)
    model = DFlashDraftModel(_cfg())                       # unbound: no embed / head of its own
    _randomize_bridge(model.selected_prefix_bridge)
    mx.eval(model.parameters())
    tensors = dict(_flatten_params(model))
    if nonzero_out:
        tensors["selected_prefix_bridge.out.weight"] = mx.ones((C, R))
    for name in _STAMPS:
        tensors[name] = mx.array((stamps or {}).get(name, 1), dtype=mx.int64)
    (tmp_path / "config.json").write_text(json.dumps(cfg_dict))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
    return str(tmp_path), model


def test_load_dspine_roundtrip_and_config_fields(tmp_path):
    path, model = _write_ckpt(tmp_path)
    drafter, cfg = load_dflash(path, quantize=False)
    assert drafter.selected_prefix_bridge is not None and drafter.candidate_selector is None
    assert (cfg.bridge_code_size, cfg.bridge_rank, cfg.bridge_top_k) == (C, R, K)
    assert cfg.bridge_layers == (0, 1) and cfg.block_causal is True
    assert drafter.block_causal and drafter.selects_paths and tree_capable(drafter)
    assert cfg.sliding_window == 64 and cfg.layer_types == ("sliding_attention",) * L
    assert mx.array_equal(drafter.selected_prefix_bridge.codebook,
                          model.selected_prefix_bridge.codebook).item()
    assert mx.array_equal(drafter.layers[1].self_attn.q_proj.weight,
                          model.layers[1].self_attn.q_proj.weight).item()


def test_load_dspine_quantizes_backbone_but_not_the_bridge(tmp_path):
    path, _ = _write_ckpt(tmp_path)
    drafter, _ = load_dflash(path, quantize=True, bits=4, group_size=32)
    assert isinstance(drafter.fc, nn.QuantizedLinear)
    assert isinstance(drafter.layers[0].self_attn.q_proj, nn.QuantizedLinear)
    bridge = drafter.selected_prefix_bridge
    for lin in (bridge.prev, bridge.gate, bridge.predecessor_gate, bridge.out):
        assert isinstance(lin, nn.Linear) and not isinstance(lin, nn.QuantizedLinear)
    assert bridge.codebook.shape == (V, C)                  # raw arrays untouched


def test_load_dspine_refuses_unknown_stamp_versions(tmp_path):
    path, _ = _write_ckpt(tmp_path, stamps={"last_write_refine_version": 2})
    with pytest.raises(ValueError, match="last_write_refine_version = 2"):
        load_dflash(path, quantize=False)


def test_load_dspine_refuses_a_nonzero_terminal_projection(tmp_path):
    path, _ = _write_ckpt(tmp_path, nonzero_out=True)
    with pytest.raises(ValueError, match="terminal"):
        load_dflash(path, quantize=False)


def _variant(**dflash_over):
    cfg = json.loads(json.dumps(DSPINE_MIN))
    for k, v in dflash_over.items():
        if v is None:
            cfg["dflash_config"].pop(k, None)
        else:
            cfg["dflash_config"][k] = v
    return cfg


@pytest.mark.parametrize("cfg_dict, reason", [
    (_variant(selected_prefix_terminal=True), "terminal"),
    (_variant(selected_prefix_feature_write_mode="shared"), "write_mode"),
    (_variant(selected_prefix_feature_writes=False), "feature_writes"),
    (_variant(draft_block_causal=None), "block-causal"),
    (_variant(selected_prefix_decode_mode="parallel"), "decode_mode"),
    (_variant(selected_prefix_feature_layers=[1]), "last draft layer"),
    (_variant(selector_rank=8, selector_top_k=4), "selector"),
    (_variant(selected_prefix_last_write_refine={"head_topk": K, "kind": "predicted",
                                                 "version": 1, "weight": 1.0}), "exact_sender"),
    (_variant(selected_prefix_terminal_residual_rank=4), "residual"),
])
def test_load_dspine_refuses_unsupported_contracts_with_the_reason(tmp_path, cfg_dict, reason):
    path, _ = _write_ckpt(tmp_path, cfg_dict)
    with pytest.raises(ValueError, match=reason):
        load_dflash(path, quantize=False)


def test_load_dspine_refuses_disagreeing_refinement_widths(tmp_path):
    cfg = json.loads(json.dumps(DSPINE_MIN))
    cfg["code_bridge_second_pass"]["head_topk"] = 2
    path, _ = _write_ckpt(tmp_path, cfg)
    with pytest.raises(ValueError, match="disagree"):
        load_dflash(path, quantize=False)


def test_load_dspine_takes_the_width_from_either_declaration(tmp_path):
    cfg = json.loads(json.dumps(DSPINE_MIN))
    del cfg["code_bridge_second_pass"]                      # the Qwen3-8B head's layout
    path, _ = _write_ckpt(tmp_path, cfg)
    _, parsed = load_dflash(path, quantize=False)
    assert parsed.bridge_top_k == K
    cfg["dflash_config"].pop("selected_prefix_last_write_refine")
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="refinement width"):
        load_dflash(path, quantize=False)


def test_load_dflash2_is_causal_false_reaches_the_attention(tmp_path):
    """The shipped DFlash 2 heads declare `is_causal: false`: their sliding layers must attend
    the block bidirectionally (the reference), no longer causally."""
    from test_dflash2 import DFLASH2_MIN, _write_dflash2_ckpt

    path = _write_dflash2_ckpt(tmp_path)
    drafter, cfg = load_dflash(path, quantize=False)
    assert cfg.block_causal is False                       # declared; full layers: no-op
    assert all(not ly.self_attn.block_causal for ly in drafter.layers)
    cfg_dict = json.loads(json.dumps(DFLASH2_MIN))
    cfg_dict["layer_types"] = ["sliding_attention"] * 2
    cfg_dict["sliding_window"] = 64
    (tmp_path / "config.json").write_text(json.dumps(cfg_dict))
    drafter, cfg = load_dflash(path, quantize=False)
    assert cfg.block_causal is False
    assert all(not ly.self_attn.block_causal and ly.self_attn.is_sliding for ly in drafter.layers)
    assert not drafter.block_causal and drafter.draft_width(2) == 8
    del cfg_dict["is_causal"]                              # undeclared -> z-lab default: causal
    (tmp_path / "config.json").write_text(json.dumps(cfg_dict))
    drafter, cfg = load_dflash(path, quantize=False)
    assert cfg.block_causal is None and drafter.block_causal
