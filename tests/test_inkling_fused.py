"""Inkling 2's fused forward pieces (inkling_fused.py): each kernel against the stock arithmetic it
replaces, written out in plain MLX ops here (so these run without mlx-vlm), then the installed
pieces against mlx-vlm's own forward on a tiny random model (skipped without the looped model)."""

import math

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_dspark import inkling_fused as F

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="custom Metal kernels")
H = 256


def _bf16(*shape, scale=1.0, seed=0):
    return (mx.random.normal(shape, key=mx.random.key(seed)) * scale).astype(mx.bfloat16)


def _rel(got, ref):
    got, ref = got.astype(mx.float32), ref.astype(mx.float32)
    return float(mx.sqrt(mx.sum(mx.square(got - ref)) / mx.sum(mx.square(ref))))


# ---------------------------------------------------------------- stock arithmetic, in MLX ops
def _l2_clip(x, threshold, scale=None):                    # mlx-vlm inkling2 l2_clip
    dtype = x.dtype
    x = x.astype(mx.float32)
    if scale is not None:
        x = x * scale.astype(mx.float32)
    norm = mx.sqrt(mx.sum(mx.square(x), axis=-1, keepdims=True))
    return (x * mx.minimum(threshold / (norm + 1e-6), 1.0)).astype(dtype)


def _attention_residual(states, query):                    # mlx-vlm inkling2 _attention_residual
    states = states.astype(mx.float32)
    keys = mx.fast.rms_norm(states, None, 1e-6)
    scores = (keys @ query.reshape(-1, states.shape[-1]).T) * states.shape[-1] ** -0.5
    weights = mx.softmax(scores, axis=-2).swapaxes(-1, -2)
    return weights @ keys


def _controller(x, message, down, forget, gain, eps):       # mlx-vlm MessageController
    pair = mx.concatenate([mx.fast.rms_norm(v, None, eps) for v in (x, message)], axis=-1)
    hidden = (pair.astype(down.dtype) @ down.T).astype(mx.float32)
    f = mx.tanh(hidden @ forget.T)
    g = mx.tanh(hidden @ gain.T)
    return (1 - f) * x + g * message


def _dynamic_conv(state, static, dyn, x, heads, k=4):      # mlx-vlm DynamicShortConvolution
    b, length, _ = x.shape
    xf = x.astype(mx.float32)
    context = mx.concatenate([state, xf], axis=1)
    dynamic = dyn.astype(mx.float32).reshape(b, length, heads, 1, k)
    weights = static.T.reshape(heads, -1, k) + dynamic
    out = mx.zeros_like(xf).reshape(b, length, heads, -1)
    for lag in range(k):
        values = context[:, k - 1 - lag: k - 1 - lag + length].reshape(out.shape)
        out = out + values * weights[..., lag]
    return ((out.reshape(x.shape) + xf) / math.sqrt(2 * k + 1)).astype(x.dtype), context


# ---------------------------------------------------------------- kernels vs that arithmetic
@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("scale", [0.3, 6.0])              # the clip idle, the clip active
def test_head_prep_is_bit_identical_to_the_stock_ops(rows, scale):
    nh, nkv, hd, drel = 6, 2, 64, 16
    q, k, v, r = (_bf16(1, rows, n, scale=scale, seed=i)
                  for i, n in enumerate((nh * hd, nkv * hd, nkv * hd, nh * drel)))
    qs, rs = _bf16(nh, hd, seed=7), _bf16(nh, drel, seed=8)
    got = F.fused_head_prep(q, k, v, r, qs, rs, n_heads=nh, n_kv=nkv, head_dim=hd, d_rel=drel,
                            clip=22.627416997969522, rel_clip=8.0, eps=1e-6)
    ref = (_l2_clip(q.reshape(1, rows, nh, hd), 22.627416997969522, qs),
           _l2_clip(k.reshape(1, rows, nkv, hd), 22.627416997969522),
           mx.fast.rms_norm(v.reshape(1, rows, nkv, hd).astype(mx.float32), None, 1e-6)
           .astype(mx.bfloat16),
           _l2_clip(r.reshape(1, rows, nh, drel), 8.0, rs))
    for name, g, e in zip("qkvr", got, ref):
        assert g.dtype == mx.bfloat16 and g.shape == e.shape
        assert bool(mx.array_equal(g, e)), name


@pytest.mark.parametrize("rows", [1, 3])
def test_norm_cast_is_bit_identical_to_rms_norm_then_cast(rows):
    norm = nn.RMSNorm(H, eps=1e-6)
    norm.weight = _bf16(H, seed=3)
    x = mx.random.normal((1, rows, H), key=mx.random.key(4)) * 3
    assert bool(mx.array_equal(F.fused_norm_cast(norm, x), norm(x).astype(mx.bfloat16)))
    # a row view of a wider array (the layer's mixed[..., i, :]) reads through its row stride
    wide = mx.random.normal((1, rows, 3, H), key=mx.random.key(5))
    assert bool(mx.array_equal(F.fused_norm_cast(norm, wide[..., 1, :]),
                               norm(wide[..., 1, :]).astype(mx.bfloat16)))
    # not the fused shape (a bf16 stream): the stock pair
    xb = x.astype(mx.bfloat16)
    assert bool(mx.array_equal(F.fused_norm_cast(norm, xb), norm(xb).astype(mx.bfloat16)))


@pytest.mark.parametrize("rows", [1, 4])
def test_dynamic_conv_is_bit_identical_and_returns_the_state_window(rows):
    c, heads = 256, 2
    state = mx.random.normal((1, 3, c), key=mx.random.key(1))
    static, dyn, x = _bf16(4, c, seed=2), _bf16(1, rows, heads * 4, seed=3), _bf16(1, rows, c, seed=4)
    out, window = F._dyn_kernel()(
        inputs=[state, static, dyn, x, F._dyn_divisor(4)],
        template=[("T", x.dtype), ("W", 4), ("HEADS", heads)], grid=(rows * c, 1, 1),
        threadgroup=(256, 1, 1), output_shapes=[x.shape, (1, 3 + rows, c)],
        output_dtypes=[x.dtype, mx.float32])
    ref, context = _dynamic_conv(state, static, dyn, x, heads)
    assert bool(mx.array_equal(out, ref)) and bool(mx.array_equal(window, context))


@pytest.mark.parametrize("rows", [1, 4])
def test_short_convolution_window_and_residual_folding(rows):
    c = 256
    state = mx.random.normal((1, 3, c), key=mx.random.key(1))
    weight, x = _bf16(c, 4, 1, scale=0.3, seed=2), _bf16(1, rows, c, seed=3)
    out, window = F.fused_short_convolution(state, weight, x, 4)
    assert out.dtype == mx.bfloat16
    assert bool(mx.array_equal(window, mx.concatenate([state, x.astype(mx.float32)], axis=1)))
    ctx = mx.concatenate([state, x.astype(mx.float32)], axis=1)
    ref = sum(ctx[:, j: j + rows] * weight[:, j, 0].astype(mx.float32) for j in range(4))
    ref = (ref + x.astype(mx.float32)) / math.sqrt(5)
    assert float(mx.abs(out.astype(mx.float32) - ref).max()) < 0.02 * float(mx.abs(ref).max())
    # the layer's residual arithmetic at the stock rounding points
    res32 = mx.random.normal((1, rows, c), key=mx.random.key(5))
    as32, _ = F.fused_short_convolution(state, weight, x, 4, None, mx.float32)
    assert bool(mx.array_equal(as32, out.astype(mx.float32)))
    add32, _ = F.fused_short_convolution(state, weight, x, 4, x, mx.float32)
    assert bool(mx.array_equal(add32, out.astype(mx.float32) + x.astype(mx.float32)))
    add16, _ = F.fused_short_convolution(state, weight, x, 4, res32, mx.bfloat16)
    assert bool(mx.array_equal(add16, (res32 + out.astype(mx.float32)).astype(mx.bfloat16)))


def test_gate_is_bit_identical_to_the_stock_ops():
    out, gate = _bf16(1, 4, 384, scale=2, seed=1), _bf16(1, 4, 384, scale=2, seed=2)
    ref = (out.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))).astype(mx.bfloat16)
    assert bool(mx.array_equal(F._gated(out, gate), ref))


@pytest.mark.parametrize("n,q", [(2, 1), (5, 1), (3, 3), (2, 4)])
def test_attention_residual_within_rounding_and_row_invariant(n, q):
    rows = 4
    states = [_bf16(1, rows, H, seed=i) for i in range(n - 1)]
    states.append(mx.random.normal((1, rows, H), key=mx.random.key(9)))        # the fp32 partial
    query = mx.random.normal((H,) if q == 1 else (q, H), key=mx.random.key(10))
    got = F.fused_attention_residual(states, query)
    ref = _attention_residual(mx.stack(states, axis=-2), query)
    assert got.shape == ref.shape == (1, rows, q, H) and got.dtype == mx.float32
    assert _rel(got, ref) < 1e-5
    alone = mx.concatenate([F.fused_attention_residual([s[:, i:i + 1] for s in states], query)
                            for i in range(rows)], axis=1)
    assert bool(mx.array_equal(got, alone))


def test_controller_within_rounding_row_invariant_and_reads_row_views():
    rows, rank = 4, 16
    x = mx.random.normal((1, rows, H), key=mx.random.key(1))
    messages = mx.random.normal((1, rows, 4, H), key=mx.random.key(2))
    down, forget, gain = (_bf16(rank, 2 * H, scale=0.05, seed=3), _bf16(H, rank, scale=0.2, seed=4),
                          _bf16(H, rank, scale=0.2, seed=5))
    got = F.fused_controller(x, messages[..., 2, :], down, forget, gain, 1e-6)   # a row view
    ref = _controller(x, messages[..., 2, :], down, forget, gain, 1e-6)
    assert got.shape == ref.shape and got.dtype == mx.float32
    assert _rel(got, ref) < 2e-2          # one bf16 rounding flip in the rank code moves a row
    alone = mx.concatenate([F.fused_controller(x[:, i:i + 1], messages[:, i:i + 1, 2, :], down,
                                               forget, gain, 1e-6) for i in range(rows)], axis=1)
    assert bool(mx.array_equal(got, alone))


# ---------------------------------------------------------------- installed, against mlx-vlm
def _tiny_bf16_inkling():
    from mlx_dspark.inkling import mlx_vlm_supports_inkling2

    if not mlx_vlm_supports_inkling2("inkling2_loop"):
        pytest.skip("this mlx-vlm has no looped Inkling 2 model (0.8.0 ships the plain one only)")
    from mlx.utils import tree_map
    from mlx_vlm.models.inkling2 import Model, ModelConfig
    from mlx_vlm.models.inkling2.config import TextConfig, VisionConfig

    text = TextConfig(model_type="inkling2_loop_text", hidden_size=128, num_hidden_layers=6,
                      vocab_size=64, unpadded_vocab_size=60, num_attention_heads=4,
                      num_key_value_heads=4, head_dim=32, swa_num_attention_heads=4,
                      swa_num_key_value_heads=4, swa_head_dim=32, sliding_window_size=8,
                      local_layer_ids=[0, 2, 3, 4], d_rel=4, rel_extent=8,
                      log_scaling_n_floor=4, intermediate_size=64, n_routed_experts=4,
                      num_experts_per_tok=2, n_layers_per_attn_res_block=2,
                      loop_config={"core_start": 2, "core_end": 4, "num_passes": 2,
                                   "gate_rank": 8})
    vision = VisionConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                          projection_hidden_dim=32)
    model = Model(ModelConfig(text_config=text, vision_config=vision, model_type="inkling2_loop"))
    keys = iter(mx.random.split(mx.random.key(0), 4096))
    model.update(tree_map(lambda p: (mx.random.normal(p.shape, key=next(keys)) * 0.1)
                          .astype(mx.bfloat16), model.parameters()))
    lm = model.language_model
    # the released checkpoint keeps these in fp32
    lm.model.message_queries = lm.model.message_queries.astype(mx.float32)
    for layer in lm.model.layers:
        for res in (layer.attention_input, layer.mlp_input):
            if res is not None:
                res.query = res.query.astype(mx.float32)
        layer.mlp.e_score_correction_bias = layer.mlp.e_score_correction_bias.astype(mx.float32)
    lm.model.final_input.query = lm.model.final_input.query.astype(mx.float32)
    model.eval()                      # as a loaded checkpoint: the stock kernel paths need it too
    mx.eval(model.parameters())
    return lm


def _count_fused_calls(monkeypatch):
    hits = {}
    for name in ("_sconv_step", "fused_norm_cast", "fused_head_prep", "_dyn_sconv_step",
                 "fused_attention_residual", "fused_controller"):
        stock = getattr(F, name)

        def counted(*a, _stock=stock, _name=name, **k):
            hits[_name] = hits.get(_name, 0) + 1
            return _stock(*a, **k)

        monkeypatch.setattr(F, name, counted)
    return hits


def _walk(lm, on_kinds):
    """Prefill 12 tokens (stock), three decode steps, a 4-row verify committed at 2 rows, one
    more step — the logits of every forward after the prefill and the final cache states."""
    F.set_enabled(lm, False)
    cache = lm.make_cache()
    mx.eval(lm(inputs=mx.array([[3, 7, 11, 2, 9, 30, 41, 5, 8, 13, 21, 34]]), cache=cache).logits)
    if on_kinds:
        F.set_enabled(lm, True, on_kinds)
    outs = []
    for t in (4, 17, 23):
        outs.append(lm(inputs=mx.array([[t]]), cache=cache).logits)
        mx.eval(outs[-1])
    out = lm(inputs=mx.array([[6, 12, 19, 27]]), cache=cache, capture_layer_ids=[1, 5, 8],
             speculative_verify=True)
    outs.extend([out.logits, *out.hidden_states])
    mx.eval(outs)
    out.gdn_states.commit(2)
    outs.append(lm(inputs=mx.array([[33]]), cache=cache).logits)
    states = F._state_arrays([part for c in cache for part in c.caches])
    mx.eval(outs, states)
    F.set_enabled(lm, False)
    return outs, states


def test_installed_exact_kinds_are_bit_identical_to_mlx_vlm(monkeypatch):
    lm = _tiny_bf16_inkling()
    counts = F.install(lm, enabled=False)
    assert counts["layer"] == 6 and counts["attn"] == 6 and counts["ctrl"] == 8
    hits = _count_fused_calls(monkeypatch)
    ref, ref_states = _walk(lm, None)
    assert hits == {}                                          # off means the stock forward
    _walk(lm, F.EXACT_KINDS)
    # 8 layer visits per forward (layers 2-3 run twice): 6 sliding, 2 global; 5 forwards
    assert hits["_sconv_step"] == 5 * (8 * 2 + 6 * 2) and hits["fused_head_prep"] == 5 * 8
    assert hits["_dyn_sconv_step"] == 5 * 2 * 2 and hits["fused_norm_cast"] > 5 * 8
    for kinds in (("sconv",), ("layer",), ("attn",), F.EXACT_KINDS):
        got, got_states = _walk(lm, kinds)
        assert all(bool(mx.array_equal(a, b)) for a, b in zip(got, ref)), kinds
        assert len(got_states) == len(ref_states)
        assert all(bool(mx.array_equal(a, b)) for a, b in zip(got_states, ref_states)), kinds


def test_installed_rounding_kinds_stay_close_and_long_calls_take_the_stock_path():
    lm = _tiny_bf16_inkling()
    F.install(lm, enabled=False)
    ref, _ = _walk(lm, None)
    got, _ = _walk(lm, (*F.EXACT_KINDS, *F.ROUNDING_KINDS))
    assert all(a.shape == b.shape for a, b in zip(got, ref))
    assert max(_rel(a, b) for a, b in zip(got, ref)) < 0.2       # random weights amplify rounding
    # more positions than a verify carries: every piece defers to the stock forward
    ids = mx.array([list(range(2, 2 + F.MAX_ROWS + 4))])
    F.set_enabled(lm, False)
    stock = lm(inputs=ids, cache=lm.make_cache()).logits
    F.set_enabled(lm, True)
    fused = lm(inputs=ids, cache=lm.make_cache()).logits
    assert bool(mx.array_equal(stock, fused))


def test_selftest_passes_then_turns_off_a_body_that_no_longer_matches(monkeypatch):
    lm = _tiny_bf16_inkling()
    F.install(lm, enabled=False)
    assert F.selftest(lm) == dict.fromkeys((*F.EXACT_KINDS, *F.ROUNDING_KINDS), True)
    assert all(F._on(m) for m, _ in F._fused_modules(lm))
    # mlx-vlm changes its layer body: the copy must switch off, the other pieces stay on
    from mlx_vlm.models.inkling2 import language as L

    stock = L.DecoderLayer.__call__
    monkeypatch.setattr(L.DecoderLayer, "__call__",
                        lambda self, x, blocks, cache=None, messages=None:
                        stock(self, x, blocks, cache, messages) * 2)
    result = F.selftest(lm)
    assert result["layer"] is False and result["attn"] and result["sconv"] and result["res"]
    assert not any(F._on(m) for m, kind in F._fused_modules(lm) if kind == "layer")
    assert all(F._on(m) for m, kind in F._fused_modules(lm) if kind != "layer")
    F.set_enabled(lm, True)                                    # a rejected kind stays off
    assert not any(F._on(m) for m, kind in F._fused_modules(lm) if kind == "layer")
