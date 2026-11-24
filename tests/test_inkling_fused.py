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


# ---------------------------------------------------------------- the packed verify attention
def _packed_case(nh, nkv, hd, drel, extent, sliding, rows, keys, seed=0):
    import types

    attn = types.SimpleNamespace(rel_extent=extent, sliding=sliding, n_heads=nh, n_kv=nkv,
                                 head_dim=hd, rel_proj=mx.zeros((drel, extent)))
    s = seed * 10
    return attn, {
        "q": _bf16(1, rows, nh, hd, scale=0.6, seed=s + 1),
        "k": _bf16(1, nkv, keys, hd, scale=0.6, seed=s + 2),
        "v": _bf16(1, nkv, keys, hd, seed=s + 3),
        "r": _bf16(1, rows, nh, drel, scale=0.5, seed=s + 4),
        "rel_t": _bf16(extent, drel, scale=0.3, seed=s + 5),
        "sinks": _bf16(1, rows, nh, seed=s + 6),
        "gate": _bf16(1, rows, nh * hd, seed=s + 7),
    }


def _attention_fp32(attn, q, k, v, r, rel_t, sinks, gate):
    """mlx-vlm's attention after the cache update, with nothing rounded inside: the additive
    mask with the sink's column (bf16, as the stream's), fp32 scores, softmax and values, the
    output rounded to bf16, then the gate."""
    _, rows, nh, hd = q.shape
    g, keys = nh // k.shape[1], k.shape[2]
    zero = mx.zeros((1, k.shape[1], 1, hd), k.dtype)
    mask = F.fused_mask(r, rel_t, sinks, keys, attn.sliding, attn.rel_extent)
    kf = mx.repeat(mx.concatenate([k, zero], axis=2), g, axis=1).astype(mx.float32)
    vf = mx.repeat(mx.concatenate([v, zero], axis=2), g, axis=1).astype(mx.float32)
    scores = (q.transpose(0, 2, 1, 3).astype(mx.float32) @ kf.swapaxes(-1, -2)) * hd ** -0.5
    out = mx.softmax(scores + mask.astype(mx.float32), axis=-1) @ vf
    return F._gated(out.astype(mx.bfloat16).transpose(0, 2, 1, 3).reshape(1, rows, -1), gate)


@pytest.mark.parametrize("nh,nkv,hd,drel,extent", [
    (24, 8, 128, 16, 1024),            # Inkling-2-Mini
    (4, 4, 32, 4, 8),                  # the tiny test model: no GQA
    (8, 2, 64, 8, 40),                 # GQA 4, another head and code width
])
@pytest.mark.parametrize("windowed", [False, True])
def test_packed_attention_is_the_fp32_attention_within_rounding(nh, nkv, hd, drel, extent,
                                                                windowed):
    sliding = extent if windowed else 0
    # one tile, a partial tile, several key blocks, past the extent / at the window cap; one row
    # to more rows than a simdgroup holds for the head group
    for rows, keys in ((3, 16), (1, 5), (4, 37), (8, 300), (5, extent + 9), (2, extent + 1)):
        if sliding:
            keys = min(keys, sliding - 1 + rows)     # all a sliding call can see
        attn, t = _packed_case(nh, nkv, hd, drel, extent, sliding, rows, keys, seed=rows + keys)
        assert F._packed_ok(attn, rows)
        got = F.packed_attention(attn, **t)
        ref = _attention_fp32(attn, **t)
        assert got.shape == ref.shape == (1, rows, nh * hd) and got.dtype == mx.bfloat16
        assert _rel(got, ref) < F.PACKED_TOL, (rows, keys)       # measured: 1e-6 to 1e-4


def test_packed_attention_reads_views_and_never_looks_ahead():
    rows, keys = 4, 1040
    attn, t = _packed_case(24, 8, 128, 16, 1024, 0, rows, keys)
    ref = F.packed_attention(attn, **t)
    # K and V as views into a larger buffer (the in-place caches hand out exactly that)
    def pad(a):
        return mx.concatenate([_bf16(1, 8, 7, 128, seed=90), a, _bf16(1, 8, 300, 128, seed=91)],
                              axis=2)[..., 7: 7 + keys, :]

    viewed = F.packed_attention(attn, **{**t, "k": pad(t["k"]), "v": pad(t["v"])})
    assert bool(mx.array_equal(viewed, ref))
    # another token in the LAST row — its query, code, sink, gate and its own key / value —
    # leaves every earlier row's output the same bits
    other = dict(t)
    for name in ("q", "r", "sinks", "gate"):
        other[name] = mx.concatenate([t[name][:, :-1], t[name][:, -1:] * 1.7 + 0.3], axis=1)
    for name in ("k", "v"):
        other[name] = mx.concatenate([t[name][:, :, :-1], t[name][:, :, -1:] * -2.0], axis=2)
    moved = F.packed_attention(attn, **other)
    assert bool(mx.array_equal(moved[:, :-1], ref[:, :-1]))
    assert not bool(mx.array_equal(moved[:, -1], ref[:, -1]))
    # and a sliding layer's rows each drop the keys that left their own window
    attn, t = _packed_case(24, 8, 128, 16, 1024, 1024, rows, 1023 + rows, seed=3)
    ref = F.packed_attention(attn, **t)
    old = {**t, "k": mx.concatenate([t["k"][:, :, :1] * 5.0, t["k"][:, :, 1:]], axis=2),
           "v": mx.concatenate([t["v"][:, :, :1] * 5.0, t["v"][:, :, 1:]], axis=2)}
    moved = F.packed_attention(attn, **old)          # the oldest key: row 0 alone still sees it
    assert bool(mx.array_equal(moved[:, 1:], ref[:, 1:]))
    assert not bool(mx.array_equal(moved[:, 0], ref[:, 0]))


def test_packed_attention_refuses_shapes_it_was_not_written_for():
    ok = _packed_case(24, 8, 128, 16, 1024, 0, 4, 64)[0]
    assert F._packed_ok(ok, 8)
    for change in ({"head_dim": 100}, {"n_kv": 5}, {"rel_proj": mx.zeros((6, 1024))},
                   {"head_dim": 512}):
        attn = _packed_case(24, 8, 128, 16, 1024, 0, 4, 64)[0]
        for name, value in change.items():
            setattr(attn, name, value)
        assert not F._packed_ok(attn, 8), change
    assert not F._packed_ok(ok, 100)                 # more rows than one threadgroup holds


# ---------------------------------------------------------------- a newer mlx-vlm's call shape
# (issue #43: mlx-vlm's main gave DecoderLayer.__call__ a `speculative_verify` keyword and passed
# it on every call; the wrappers were written against the older signature and raised TypeError)
class _Layer:
    def __call__(self, x, blocks, cache=None, messages=None, speculative_verify=False):
        return ("stock", x, blocks, cache, messages, speculative_verify)


def _fused(self, x, blocks, cache=None, messages=None):
    return ("fused", x, blocks, cache, messages)


def test_wrapper_takes_a_new_argument_at_its_default_and_leaves_the_rest_to_stock():
    call = F._tolerant(_fused, _Layer)
    layer = _Layer()
    assert call(layer, 1, 2, 3, messages=4) == ("fused", 1, 2, 3, 4)
    # what the newer forward sends on an ordinary call: the argument at the stock default
    assert call(layer, 1, 2, cache=3, messages=4, speculative_verify=False) == ("fused", 1, 2, 3, 4)
    # set, by keyword or by position: only the stock method knows what it means
    assert call(layer, 1, 2, 3, None, speculative_verify=True) == ("stock", 1, 2, 3, None, True)
    assert call(layer, 1, 2, 3, None, True) == ("stock", 1, 2, 3, None, True)
    # 0 is not False: a value is the default only when it is the same kind of value
    assert call(layer, 1, 2, speculative_verify=0)[0] == "stock"
    with pytest.raises(TypeError):                   # an argument nobody knows still raises
        call(layer, 1, 2, nonsense=1)


def test_wrapper_is_the_fused_call_itself_on_the_signature_it_was_written_for():
    class Same:
        def __call__(self, x, blocks, cache=None, messages=None):
            return "stock"

    assert F._tolerant(_fused, Same) is _fused


def test_wrapper_stays_on_stock_when_the_method_is_no_longer_the_one_it_copied():
    class Renamed:
        def __call__(self, x, residuals, cache=None, messages=None):
            return ("stock", x, residuals)

    class Variadic:
        def __call__(self, x, *blocks, **options):
            return "stock"

    assert F._tolerant(_fused, Renamed)(Renamed(), 1, 2) == ("stock", 1, 2)
    assert F._tolerant(_fused, Variadic) is _fused   # nothing to compare against: as before


# ---------------------------------------------------------------- installed, against mlx-vlm
def _tiny_bf16_inkling():
    return _tiny_bf16_model().language_model


def _tiny_bf16_model():
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
    return model


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


def _walk(lm, on_kinds, make_cache=None):
    """Prefill 12 tokens (stock), three decode steps, a 4-row verify committed at 2 rows, one
    more step — the logits of every forward after the prefill and the final cache states."""
    F.set_enabled(lm, False)
    cache = (make_cache or lm.make_cache)()
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
    # 8 layer visits per forward (layers 2-3 run twice): 6 sliding, 2 global; 5 forwards. Where
    # mlx-vlm's own loop hands its layers `speculative_verify` (main since 2026-11-23), the
    # verify's layer bodies are mlx-vlm's — the fused loop, which keeps them, is off in this walk
    import inspect

    stock_layer = type(lm.model.layers[0]).__mro__[1].__call__
    bodies = 4 if "speculative_verify" in inspect.signature(stock_layer).parameters else 5
    assert hits["_sconv_step"] == bodies * 8 * 2 + 5 * 6 * 2 and hits["fused_head_prep"] == 5 * 8
    assert hits["_dyn_sconv_step"] == 5 * 2 * 2 and hits["fused_norm_cast"] > 5 * 8
    for kinds in (("sconv",), ("layer",), ("attn",), F.EXACT_KINDS):
        got, got_states = _walk(lm, kinds)
        assert all(bool(mx.array_equal(a, b)) for a, b in zip(got, ref)), kinds
        assert len(got_states) == len(ref_states)
        assert all(bool(mx.array_equal(a, b)) for a, b in zip(got_states, ref_states)), kinds


def test_in_place_kv_caches_walk_bit_identically_to_the_stock_caches():
    from mlx_dspark import inkling_cache as C

    lm = _tiny_bf16_inkling()
    F.install(lm, enabled=False)
    ref, _ = _walk(lm, None)

    def converted():
        cache = C.convert(lm.make_cache())
        assert all(C.is_sink_cache(entry.caches[0]) for entry in cache)
        for entry in cache:
            entry.caches[0].step = 2          # the 8-key window slides and is rebuilt in the walk
        return cache

    # with the fused attention (views with the sink row) and with the stock body (prefill's)
    for kinds in (F.EXACT_KINDS, ("attn",), None):
        got, _ = _walk(lm, kinds, converted)
        assert all(bool(mx.array_equal(a, b)) for a, b in zip(got, ref)), kinds


def test_one_launch_mask_is_bit_identical_and_the_walk_matches_with_it():
    lm = _tiny_bf16_inkling()                    # (skips where mlx-vlm has no looped Inkling 2)
    from mlx_vlm.models.inkling2 import language as L

    F.install(lm, enabled=False)
    for layer in (lm.model.layers[0], lm.model.layers[1]):           # sliding, global
        attn = layer.self_attn
        for rows, size in ((1, 1), (1, 6), (3, 3), (3, 9), (8, 15), (5, 40)):
            if attn.sliding and size > attn.sliding - 1 + rows:
                continue
            r = _bf16(1, rows, attn.n_heads, 4, seed=rows * 100 + size)
            sinks = _bf16(1, rows, attn.n_heads, seed=7 + size)
            ref = mx.concatenate(
                [L.banded_additive_mask(r, attn.rel_proj.astype(mx.bfloat16), size - rows, size,
                                        attn.sliding, attn.rel_extent, left_padding=None,
                                        shape_ref=mx.zeros((1, 1, size, 1), mx.bfloat16)),
                 sinks.transpose(0, 2, 1)[..., None]], axis=-1)
            got = F.fused_mask(r, F._rel_transposed(attn, mx.bfloat16), sinks, size, attn.sliding,
                               attn.rel_extent)
            assert got.shape == ref.shape and bool(mx.array_equal(got, ref)), (rows, size)
    # in the forward: the same walk with the mask from one launch and from the stock pair
    ref, ref_states = _walk(lm, None)
    for on in (True, False):
        F.set_mask(lm, on)
        got, got_states = _walk(lm, F.EXACT_KINDS)
        assert all(bool(mx.array_equal(a, b)) for a, b in zip(got, ref)), on
        assert all(bool(mx.array_equal(a, b)) for a, b in zip(got_states, ref_states)), on


def test_packed_verify_attention_runs_at_verify_widths_only_and_stays_close(monkeypatch):
    from mlx_dspark import inkling_cache as C

    lm = _tiny_bf16_inkling()
    F.install(lm, enabled=False)
    def converted():
        return C.convert(lm.make_cache())

    ref, _ = _walk(lm, F.EXACT_KINDS, converted)
    calls = []
    stock = F.packed_attention
    monkeypatch.setattr(F, "packed_attention",
                        lambda attn, q, *a: calls.append(q.shape[1]) or stock(attn, q, *a))
    monkeypatch.setattr(F, "PACKED_MIN_ROWS", 3)
    F.set_packed(lm, True, min_keys=1)
    assert F.packed_attn(lm)
    got, _ = _walk(lm, F.EXACT_KINDS, converted)
    assert calls == [4] * 8                          # the 4-row verify's 8 visits; no decode step
    # decode steps before the verify are untouched; everything after it is within rounding
    assert all(bool(mx.array_equal(a, b)) for a, b in zip(got[:3], ref[:3]))
    assert max(_rel(a, b) for a, b in zip(got, ref)) < 0.05
    # under its key threshold, on the stock caches or switched off: the stock call, bit for bit
    for setup, cache in ((lambda: F.set_packed(lm, True), converted),
                         (lambda: F.set_packed(lm, True, min_keys=1), None),
                         (lambda: F.set_packed(lm, False), converted)):
        setup()
        calls.clear()
        got, _ = _walk(lm, F.EXACT_KINDS, cache)
        assert calls == [] and all(bool(mx.array_equal(a, b)) for a, b in zip(got, ref))
    assert not F.packed_attn(lm)


def test_selftest_turns_the_packed_attention_off_when_it_drifts_or_looks_ahead(monkeypatch):
    lm = _tiny_bf16_inkling()
    F.install(lm, enabled=False)
    assert F.selftest(lm)[F.PACKED_KIND] is True and F.packed_attn(lm)
    stock = F.packed_attention
    # outside rounding
    monkeypatch.setattr(F, "packed_attention", lambda *a: stock(*a) * 1.01)
    result = F.selftest(lm)
    assert result[F.PACKED_KIND] is False and not F.packed_attn(lm)
    assert result["attn"] and result[F.KV_KIND] and result["model"]      # nothing else is touched
    assert all(F._on(m) for m, _ in F._fused_modules(lm))
    # inside any tolerance, but the first row reads the last one
    def leaky(attn, q, k, v, r, rel_t, sinks, gate):
        out = stock(attn, q, k, v, r, rel_t, sinks, gate)
        return mx.concatenate([out[:, :1] + out[:, -1:] * 1e-3, out[:, 1:]], axis=1)

    monkeypatch.setattr(F, "packed_attention", leaky)
    assert F.selftest(lm)[F.PACKED_KIND] is False
    # without the in-place caches there is nothing for it to read
    monkeypatch.setattr(F, "packed_attention", stock)
    assert F.selftest(lm)[F.PACKED_KIND] is True
    from mlx_dspark import inkling_cache as C

    monkeypatch.setattr(C.classes()[1], "sink_views", lambda self: (None, None))
    result = F.selftest(lm)
    assert result[F.KV_KIND] is False and result[F.PACKED_KIND] is False and not F.packed_attn(lm)


def _walk_wide(lm, on_kinds, width, make_cache=None):
    """Prefill (stock), a decode step, a ``width``-row verify committed at 5 rows, a decode step:
    the logits and taps of every forward after the prefill and the final cache states."""
    F.set_enabled(lm, False)
    cache = (make_cache or lm.make_cache)()
    mx.eval(lm(inputs=mx.array([[3, 7, 11, 2, 9, 30, 41, 5, 8, 13, 21, 34]]), cache=cache).logits)
    if on_kinds:
        F.set_enabled(lm, True, on_kinds)
    outs = [lm(inputs=mx.array([[4]]), cache=cache).logits]
    mx.eval(outs)
    out = lm(inputs=mx.array([[(6 + 5 * i) % 60 for i in range(width)]]), cache=cache,
             capture_layer_ids=[1, 5, 8], speculative_verify=True)
    outs.extend([out.logits, *out.hidden_states])
    mx.eval(outs)
    out.gdn_states.commit(5)
    outs.append(lm(inputs=mx.array([[33]]), cache=cache).logits)
    states = F._state_arrays([part for c in cache for part in c.caches])
    mx.eval(outs, states)
    F.set_enabled(lm, False)
    return outs, states


@pytest.mark.parametrize("width", [9, 16, 33, 63])
def test_calls_wider_than_a_drafter_round_stay_bit_identical(monkeypatch, width):
    """A lookup round verifies up to 32 copied tokens, and a short prompt is a call of up to
    MAX_ROWS rows: past 8 rows MLX's attention leaves its vector kernels, so the exact pieces
    are checked against the stock forward there too."""
    from mlx_dspark import inkling_cache as C

    assert width <= F.MAX_ROWS
    lm = _tiny_bf16_inkling()
    F.install(lm, enabled=False)
    ref, ref_states = _walk_wide(lm, None, width)
    hits = _count_fused_calls(monkeypatch)
    # until the wider calls are cleared, a verify past VECTOR_ROWS is the stock forward
    assert F.rows_limit(lm) == F.VECTOR_ROWS
    got, _ = _walk_wide(lm, (*F.EXACT_KINDS, "model"), width)
    assert all(bool(mx.array_equal(a, b)) for a, b in zip(got, ref))
    narrow = dict(hits)
    F.set_rows(lm, F.MAX_ROWS)
    for kinds in (("sconv",), ("layer",), ("attn",), (*F.EXACT_KINDS, "model")):
        got, got_states = _walk_wide(lm, kinds, width)
        assert all(bool(mx.array_equal(a, b)) for a, b in zip(got, ref)), kinds
        assert all(bool(mx.array_equal(a, b)) for a, b in zip(got_states, ref_states)), kinds
    assert hits["fused_head_prep"] > narrow["fused_head_prep"] + 8      # the wide verify too
    # ... and on the in-place caches (views into the general attention path, rolled back by trim)
    got, _ = _walk_wide(lm, (*F.EXACT_KINDS, "model"), width, lambda: C.convert(lm.make_cache()))
    assert all(bool(mx.array_equal(a, b)) for a, b in zip(got, ref))
    # the packed attention takes a wide call up to PACKED_MAX_ROWS rows (past that MLX's matmul
    # attention is the faster one, and the stock call bit for bit)
    calls = []
    stock = F.packed_attention
    monkeypatch.setattr(F, "packed_attention",
                        lambda attn, q, *a: calls.append(q.shape[1]) or stock(attn, q, *a))
    F.set_packed(lm, True, min_keys=1)
    got, _ = _walk_wide(lm, F.EXACT_KINDS, width, lambda: C.convert(lm.make_cache()))
    if width <= F.PACKED_MAX_ROWS:
        assert calls == [width] * 8
        assert max(_rel(a, b) for a, b in zip(got, ref)) < 0.05
    else:
        assert calls == [] and all(bool(mx.array_equal(a, b)) for a, b in zip(got, ref))
    F.set_rows(lm, F.VECTOR_ROWS)
    calls.clear()
    _walk_wide(lm, F.EXACT_KINDS, width, lambda: C.convert(lm.make_cache()))
    assert calls == []


def test_selftest_keeps_wide_calls_on_the_stock_path_when_they_do_not_match(monkeypatch):
    lm = _tiny_bf16_inkling()
    F.install(lm, enabled=False)
    assert F.selftest(lm)[F.WIDE_KIND] is True and F.rows_limit(lm) == F.MAX_ROWS
    # a piece that drifts only at the wide width (as a kernel picked by row count would)
    stock = F.fused_norm_cast

    def drifting(norm, x):
        out = stock(norm, x)
        return out * 2 if x.shape[1] > F.VECTOR_ROWS else out

    monkeypatch.setattr(F, "fused_norm_cast", drifting)
    result = F.selftest(lm)
    assert result[F.WIDE_KIND] is False and F.rows_limit(lm) == F.VECTOR_ROWS
    assert result[F.CHUNK_KIND] is False and F.chunk_limit(lm) == F.VECTOR_ROWS
    assert all(result[k] for k in (*F.EXACT_KINDS, F.KV_KIND, F.PACKED_KIND, "model"))
    assert all(F._on(m) for m, _ in F._fused_modules(lm))        # narrow calls keep every piece


def _walk_chunks(lm, make_cache, widths):
    """Prefill calls of ``widths`` rows, one after another on one cache (taps captured): the
    outputs of each and the cache states they leave."""
    cache = make_cache()
    outs, at = [], 0
    for width in widths:
        ids = mx.array([[(6 + 5 * (at + i)) % 60 for i in range(width)]])
        out = lm(inputs=ids, cache=cache, capture_layer_ids=[1, 5, 8])
        outs.extend([out.logits, *out.hidden_states])
        mx.eval(outs)
        at += width
    states = F._state_arrays([part for c in cache for part in c.caches])
    mx.eval(states)
    return outs, states


@pytest.mark.parametrize("widths", [(64, 64), (100, 70, 81), (F.CHUNK_ROWS, 65)])
def test_prefill_chunks_take_the_exact_pieces_and_keep_the_stock_bits(monkeypatch, widths):
    """Past MAX_ROWS a call is a prefill chunk: the bit-exact pieces run around the STOCK
    residual mixes, controllers and experts, so the whole forward — logits, taps, cache states —
    is the stock forward's, bit for bit, on the stock caches and on the in-place ones."""
    lm = _tiny_bf16_inkling()
    F.install(lm, enabled=False)
    ref, ref_states = _walk_chunks(lm, lm.make_cache, widths)
    result = F.selftest(lm)
    assert result[F.CHUNK_KIND] is True
    assert F.rows_limit(lm) == F.MAX_ROWS and F.chunk_limit(lm) == F.CHUNK_ROWS
    limits = {kind: m.__dict__[F._ROWS] for m, kind in F._fused_modules(lm)}
    assert all(limits[k] == F.CHUNK_ROWS for k in F.CHUNK_KINDS)
    assert all(limits[k] == F.MAX_ROWS for k in F.ROUNDING_KINDS)
    assert all(w > F.MAX_ROWS for w in widths)
    hits = _count_fused_calls(monkeypatch)
    scheduled = []
    stock_async = F.mx.async_eval
    monkeypatch.setattr(F.mx, "async_eval", lambda *a: scheduled.append(a) or stock_async(*a))
    for in_place in (False, True):
        hits.clear()
        scheduled.clear()
        got, got_states = _walk_chunks(
            lm, (lambda: F.make_cache(lm)) if in_place else lm.make_cache, widths)
        assert all(bool(mx.array_equal(a, b)) for a, b in zip(got, ref))
        if not in_place:      # (the in-place caches keep their rows in another layout: what
            #                    they hold is checked by the calls that read it, above)
            assert len(got_states) == len(ref_states)
            assert all(bool(mx.array_equal(a, b)) for a, b in zip(got_states, ref_states))
        assert hits["fused_head_prep"] == 8 * len(widths)        # every visit's attention body
        # ... and none of the rounding kinds: those stop at MAX_ROWS
        assert not hits.get("fused_controller") and not hits.get("fused_attention_residual")
        assert len(scheduled) == len(widths) * (8 // F.SCHEDULE_EVERY)   # the fused loop
    # the A/B: without the chunk limit those calls are the stock forward
    F.set_rows(lm, F.rows_limit(lm))
    assert F.chunk_limit(lm) == F.MAX_ROWS
    hits.clear()
    got, _ = _walk_chunks(lm, lm.make_cache, widths)
    assert all(bool(mx.array_equal(a, b)) for a, b in zip(got, ref)) and not hits
    # a call every piece takes has the rounding kinds too
    hits.clear()
    _walk_chunks(lm, lm.make_cache, (F.MAX_ROWS,))
    assert hits["fused_head_prep"] == 8 and hits["fused_controller"] and hits["fused_attention_residual"]


def test_selftest_keeps_chunks_on_the_stock_path_when_they_do_not_match(monkeypatch):
    lm = _tiny_bf16_inkling()
    F.install(lm, enabled=False)
    stock = F.fused_norm_cast

    def drifting(norm, x):                       # only at a width no verify has
        out = stock(norm, x)
        return out * 2 if x.shape[1] > F.MAX_ROWS else out

    monkeypatch.setattr(F, "fused_norm_cast", drifting)
    result = F.selftest(lm)
    assert result[F.WIDE_KIND] is True and result[F.CHUNK_KIND] is False
    assert F.rows_limit(lm) == F.MAX_ROWS and F.chunk_limit(lm) == F.MAX_ROWS
    ids = mx.array([[(3 + 7 * i) % 60 for i in range(F.MAX_ROWS + 9)]])
    got = lm(inputs=ids, cache=lm.make_cache()).logits
    F.set_enabled(lm, False)
    assert bool(mx.array_equal(got, lm(inputs=ids, cache=lm.make_cache()).logits))


def test_layer_visit_loop_is_bit_identical_and_reads_messages_once_per_pass(monkeypatch):
    lm = _tiny_bf16_inkling()
    F.install(lm, enabled=False)
    ref, ref_states = _walk(lm, None)
    scheduled = []
    monkeypatch.setattr(F.mx, "async_eval", lambda *a: scheduled.append(a))
    for kinds in (("model",), (*F.EXACT_KINDS, "model")):
        scheduled.clear()
        got, got_states = _walk(lm, kinds)
        assert all(bool(mx.array_equal(a, b)) for a, b in zip(got, ref)), kinds
        assert all(bool(mx.array_equal(a, b)) for a, b in zip(got_states, ref_states)), kinds
        # 8 visits per forward, handed over every SCHEDULE_EVERY: decode steps too (stock hands
        # over inside the speculative verify only)
        assert len(scheduled) == 5 * (8 // F.SCHEDULE_EVERY)
    # a prefill-sized call is the stock loop
    scheduled.clear()
    F.set_enabled(lm, True, ("model",))
    lm(inputs=mx.array([list(range(2, 2 + F.MAX_ROWS + 4))]), cache=lm.make_cache())
    assert scheduled == []


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
    assert F.selftest(lm) == dict.fromkeys(
        (*F.EXACT_KINDS, F.KV_KIND, F.MASK_KIND, F.PACKED_KIND, *F.ROUNDING_KINDS,
         *F.LOOP_KINDS, F.WIDE_KIND, F.CHUNK_KIND), True)
    assert F.rows_limit(lm) == F.MAX_ROWS and F.chunk_limit(lm) == F.CHUNK_ROWS
    assert all(F._on(m) for m, _ in F._fused_modules(lm))
    # the cleared in-place caches are what make_cache hands out
    from mlx_dspark import inkling_cache as C

    assert F.kv_caches(lm) and all(C.is_sink_cache(e.caches[0]) for e in F.make_cache(lm))
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


def test_selftest_keeps_the_stock_caches_when_the_in_place_ones_do_not_match(monkeypatch):
    from mlx_dspark import inkling_cache as C

    lm = _tiny_bf16_inkling()
    F.install(lm, enabled=False)
    window_cls = C.classes()[1]
    stock = window_cls.sink_views

    def off_by_one(self):                       # a window that shows the attention one key less
        k, v = stock(self)
        return k[..., 1:, :], v[..., 1:, :]

    monkeypatch.setattr(window_cls, "sink_views", off_by_one)
    result = F.selftest(lm)
    assert result[F.KV_KIND] is False and result["attn"] and result["model"]
    assert not F.kv_caches(lm)
    assert not any(C.is_sink_cache(e.caches[0]) for e in F.make_cache(lm))
    assert all(F._on(m) for m, _ in F._fused_modules(lm))      # the module kinds are untouched


def _quantize_experts(lm, mode="nvfp4"):
    """The released checkpoint's expert format on the tiny model (mlx-vlm's own quantized layer)."""
    gsz, bits = (16, 4) if mode == "nvfp4" else (32, 8)
    for layer in lm.model.layers:
        sw = layer.mlp.switch_mlp
        for name in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(sw, name)
            lin.weight = lin.weight.astype(mx.float32)
            setattr(sw, name, lin.to_quantized(group_size=gsz, bits=bits, mode=mode))
    mx.eval(lm.parameters())


@pytest.mark.parametrize("mode", ["nvfp4", "mxfp8"])
def test_installed_moe_kind_decodes_bit_identically_and_is_row_exact(mode):
    from mlx_dspark import inkling_kernels as K

    lm = _tiny_bf16_inkling()
    _quantize_experts(lm, mode)
    assert F.install(lm, enabled=False)["moe"] == 6
    K.set_expert_views(lm)

    def decode(kinds):
        F.set_enabled(lm, False)
        cache = lm.make_cache()
        mx.eval(lm(inputs=mx.array([[3, 7, 11, 2, 9, 30, 41, 5, 8, 13, 21, 34]]), cache=cache).logits)
        if kinds:
            F.set_enabled(lm, True, kinds)
        outs = []
        for t in (4, 17, 23, 6):
            outs.append(lm(inputs=mx.array([[t]]), cache=cache).logits)
            mx.eval(outs[-1])
        states = F._state_arrays([part for c in cache for part in c.caches])
        mx.eval(states)
        F.set_enabled(lm, False)
        return outs, states

    ref, ref_states = decode(None)
    for kinds in (("moe",), (*F.EXACT_KINDS, *F.ROW_EXACT_KINDS)):
        got, got_states = decode(kinds)
        assert all(bool(mx.array_equal(a, b)) for a, b in zip(got, ref)), kinds
        assert all(bool(mx.array_equal(a, b)) for a, b in zip(got_states, ref_states)), kinds
    # a verify-width call: every row is the stock module's single-row (decode) result
    moe = lm.model.layers[2].mlp
    stock = type(moe).__mro__[1].__call__
    rows = [(mx.random.normal((1, 1, 128), key=mx.random.key(i)) * 2).astype(mx.bfloat16)
            for i in (1, 2, 2, 2, 3)]
    F.set_enabled(lm, True, ("moe",))
    got = moe(mx.concatenate(rows, axis=1))
    ref_rows = mx.concatenate([stock(moe, r) for r in rows], axis=1)
    assert bool(mx.array_equal(got, ref_rows))
    # more rows than the fused path takes: the stock module
    wide = (mx.random.normal((1, K.MOE_MAX_ROWS + 1, 128)) * 2).astype(mx.bfloat16)
    assert bool(mx.array_equal(moe(wide), stock(moe, wide)))
    result = F.selftest(lm)
    assert result["moe"] is True and "moe" in F.enabled_kinds(lm)
    # cleared for a lookup round's width: the fused experts take it, each row still the stock
    # module's single-row result (more rows than one routing threadgroup, runs past one tile)
    assert result[F.WIDE_KIND] is True and F.rows_limit(lm) == F.MAX_ROWS
    rows = [(mx.random.normal((1, 1, 128), key=mx.random.key(i % 5)) * 2).astype(mx.bfloat16)
            for i in range(F.MAX_ROWS)]
    F.set_enabled(lm, True, ("moe",))
    got = moe(mx.concatenate(rows, axis=1))
    assert bool(mx.array_equal(got, mx.concatenate([stock(moe, r) for r in rows], axis=1)))
    F.disable(lm, ("moe",))
    F.set_enabled(lm, True)                                    # a disabled kind stays off
    assert "moe" not in F.enabled_kinds(lm) and "attn" in F.enabled_kinds(lm)


def test_dense_bits_quantizes_the_five_projections_and_the_logits_head():
    from mlx_dspark import inkling_kernels as K

    lm = _tiny_bf16_inkling()
    ids = mx.array([[3, 7, 11, 2, 9, 30, 41, 5, 8, 13]])
    ref = lm(inputs=ids, cache=lm.make_cache()).logits
    mx.eval(ref)
    with pytest.raises(ValueError, match="4-bit"):
        K.install_dense_bits(lm, 4)                       # measured, fails the gate: not offered
    assert K.dense_bits(lm) == 0
    assert K.install_dense_bits(lm, 8) == 6 * len(K.DENSE_PROJECTIONS) + 1
    assert K.dense_bits(lm) == 8 and K.install_dense_bits(lm, 8) == 0        # idempotent
    for layer in lm.model.layers:
        attn = layer.self_attn
        assert all(type(getattr(attn, name)).__name__ == "QuantizedLinear"
                   and getattr(attn, name).bits == 8 for name in K.DENSE_PROJECTIONS)
        assert type(attn.r_proj).__name__ == "Linear"     # position / sink projections stay
    assert type(lm.lm_head).__name__ == "Linear"          # the raw head stays: drafters bind it
    assert not K.install_fast_head(lm)                    # the logits already have their head
    got = lm(inputs=ids, cache=lm.make_cache()).logits
    assert got.dtype == mx.float32 and got.shape == ref.shape    # fp32 logits, as released
    assert 0 < _rel(got, ref) < 0.35                      # 8-bit weights; random ones amplify


def test_fused_exact_kinds_stay_bit_identical_on_the_8bit_model(monkeypatch):
    from mlx_dspark import inkling_kernels as K

    lm = _tiny_bf16_inkling()
    K.install_dense_bits(lm, 8)
    F.install(lm, enabled=False)
    ref, ref_states = _walk(lm, None)
    got, got_states = _walk(lm, F.EXACT_KINDS)
    assert all(bool(mx.array_equal(a, b)) for a, b in zip(got, ref))
    assert all(bool(mx.array_equal(a, b)) for a, b in zip(got_states, ref_states))
    assert F.selftest(lm) == dict.fromkeys(
        (*F.EXACT_KINDS, F.KV_KIND, F.MASK_KIND, F.PACKED_KIND, *F.ROUNDING_KINDS,
         *F.LOOP_KINDS, F.WIDE_KIND, F.CHUNK_KIND), True)
    assert F.rows_limit(lm) == F.MAX_ROWS and F.chunk_limit(lm) == F.CHUNK_ROWS
