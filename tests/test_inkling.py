"""Model-free tests for the Thinking Machines Inkling 2 port (inkling.py, inkling_kernels.py, the
Target's inkling branch, the server's sub-message parser).

- InklingChannelParser: reasoning / answer / tool sub-messages, markers split across stream
  pieces, tool calls re-emitted as Hermes JSON for parse_tool_calls and the tool gate;
- draft_config: the bundled head's contract from the TARGET config (mtp_config);
- Inkling2DSparkDrafter: interface shapes, the ctx-cache round trip the prefix cache relies on,
  the engine window applying to the global layer only;
- the loader: the head lives in the target repo and is routed BEFORE the NVFP4-drafter converter
  (the target's config declares modelopt NVFP4);
- the kernels (Metal only): run_qmv / fused_route / combine_routes / fused_moe bit for bit
  against the stock ops, the expert views, head_f32 vs an fp32 reference;
- the Target (needs an mlx-vlm with the inkling2 model): a transactional verify + rollback equals
  plain forwards of the kept tokens, including past the sliding-window wrap.
"""

from __future__ import annotations

import json

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_dspark import anthropic_api as A
from mlx_dspark.inkling import Inkling2DSparkDrafter, draft_config, has_bundled_head
from mlx_dspark.tools import parse_tool_calls

metal = pytest.mark.skipif(not mx.metal.is_available(), reason="custom Metal kernels")

# ---------------------------------------------------------------- the sub-message parser
# Real output shapes captured from thinkingmachines/Inkling-2-Mini (first-look runs).
_INK_MATH = ("<|message_model|><|content_thinking|>210/84 = 2.5 hours. Arrival 6:10 PM."
             "<|end_message|><|message_model|><|content_text|>The train arrives at **6:10 pm**."
             "<|end_message|>")
_INK_TOOL = ("<|message_model|><|content_thinking|>Need the weather.<|end_message|>"
             '<|message_model|>get_weather<|content_invoke_tool_json|>'
             '{"name":"get_weather","args":{"city":"Paris","days":2}}<|end_message|>')


def test_split_thinking_inkling_reasoning_then_answer():
    assert A.split_thinking(_INK_MATH) == ("210/84 = 2.5 hours. Arrival 6:10 PM.",
                                           "The train arrives at **6:10 pm**.")


def test_split_thinking_inkling_answer_only_and_unterminated_reasoning():
    assert A.split_thinking("<|message_model|><|content_text|>Hi.<|end_message|>") == ("", "Hi.")
    assert A.split_thinking("<|message_model|><|content_thinking|>still going") == (
        "still going", "")


def test_inkling_tool_call_parses_to_openai_shape():
    _, answer = A.split_thinking(_INK_TOOL)
    calls, cleaned = parse_tool_calls(answer)
    assert cleaned == ""
    assert len(calls) == 1 and calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris", "days": 2}


def test_inkling_parser_chunked_equals_whole():
    whole = A.InklingChannelParser().feed(_INK_TOOL, final=True)
    p = A.InklingChannelParser()
    got = []
    for i in range(0, len(_INK_TOOL), 5):                 # every marker straddles a cut somewhere
        got += p.feed(_INK_TOOL[i:i + 5])
    got += p.feed("", final=True)

    def join(chunks, kind):
        return "".join(t for k, t in chunks if k == kind)

    for kind in ("reasoning", "answer"):
        assert join(got, kind) == join(whole, kind)
    assert "<|" not in join(got, "reasoning") and "<|end" not in join(got, "answer")


def test_build_message_inkling_reasoning_and_tool_call():
    msg = A.build_message(_INK_TOOL, model="m", input_tokens=1, output_tokens=1,
                          finish_reason="stop")
    assert [b["type"] for b in msg["content"]] == ["thinking", "tool_use"]
    assert msg["content"][1]["input"] == {"city": "Paris", "days": 2}
    assert msg["stop_reason"] == "tool_use"


def _drive(pieces, **kw):
    st = A.MessageStream(model="m", input_tokens=7, inkling=True, **kw)
    events = list(st.start())
    for p in pieces:
        events.extend(st.delta(p))
    events.extend(st.finish(finish_reason="stop", output_tokens=5))
    return events


def test_inkling_stream_thinking_then_text_and_tool_never_leaks():
    ev = _drive([_INK_MATH[:30], _INK_MATH[30:77], _INK_MATH[77:]])
    kinds = [d["content_block"]["type"] for n, d in ev if n == "content_block_start"]
    assert kinds == ["thinking", "text"]
    text = "".join(d["delta"].get("text", "") for n, d in ev if n == "content_block_delta")
    assert text == "The train arrives at **6:10 pm**." and "<|" not in text
    ev = _drive([_INK_TOOL[:50], _INK_TOOL[50:]])
    kinds = [d["content_block"]["type"] for n, d in ev if n == "content_block_start"]
    assert kinds[-1] == "tool_use"
    streamed = "".join(d["delta"].get("text", "") for n, d in ev if n == "content_block_delta")
    assert "tool_call" not in streamed and "<|" not in streamed


# ---------------------------------------------------------------- the head's config
def _target_config(**mtp_over) -> dict:
    mtp = {"hidden_size": 64, "num_hidden_layers": 3, "num_attention_heads": 4,
           "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 128,
           "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "local_layer_ids": [0, 2],
           "sliding_window_size": 4, "block_size": 8, "markov_rank": 8,
           "markov_head_type": "rnn_hidden", "target_layer_ids": [1, 3, 5, 8]}
    mtp.update(mtp_over)
    return {"model_type": "inkling2_loop",
            "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
            "text_config": {"hidden_size": 64, "vocab_size": 96, "unpadded_vocab_size": 90,
                            "final_logit_softcapping": 30.0},
            "mtp_config": mtp}


def test_draft_config_reads_the_head_contract():
    c = draft_config(_target_config())
    assert (c.family, c.block_size, c.logits_start, c.markov_head_type) == (
        "inkling2", 8, 1, "rnn_hidden")
    assert c.target_layer_ids == [1, 3, 5, 8] and c.local_layer_ids == [0, 2]
    assert not c.has_own_lm_head and not c.has_own_embed
    assert c.unpadded_vocab_size == 90 and c.vocab_size == 96


@pytest.mark.parametrize("bad", [{"markov_head_type": "vanilla"}, {"target_layer_ids": [3, 1]},
                                 {"local_layer_ids": [5]}, {"q_bias": True}])
def test_draft_config_refuses_other_contracts(bad):
    with pytest.raises(ValueError):
        draft_config(_target_config(**bad))


# ---------------------------------------------------------------- the head
def _bound_drafter(seed=0):
    mx.random.seed(seed)
    d = Inkling2DSparkDrafter(draft_config(_target_config()))
    from mlx.utils import tree_map

    d.update(tree_map(lambda p: mx.random.normal(p.shape) * 0.2, d.parameters()))
    embed = nn.Embedding(96, 64)
    head = nn.Linear(64, 96, bias=False)
    d.bind_embed(embed)
    d.bind_lm_head(head)
    return d


def _draft(d, ctx, anchor, pos):
    noise = d.embed(mx.array([[anchor] + [0] * (d.draft_width(7) - 1)]))
    h = d.backbone(noise, pos, ctx)
    return d.sample_block(d.compute_logits(d.head_slice(h, 7))[0], anchor)


def test_drafter_shapes_and_contract():
    d = _bound_drafter()
    assert d.max_draft == 7 and d.draft_width(2) == 8           # bidirectional: full block
    ctx = d.make_ctx_cache()
    assert len(ctx) == 4                                         # 3 layers + Markov rows
    d.update_context(mx.random.normal((1, 10, 4 * 64)), ctx_offset=0, ctx_caches=ctx)
    toks = _draft(d, ctx, 5, 10)
    assert toks.shape == (7,) and int(toks.max().item()) < 90    # never a padded vocab id
    toks3 = d.sample_block(d.compute_logits(d.head_slice(
        d.backbone(d.embed(mx.array([[5] + [0] * 7])), 10, ctx), 3))[0], 5)
    assert toks3.tolist() == toks.tolist()[:3]                   # a shorter cap = a prefix


def test_drafter_ctx_cache_snapshot_restore_drafts_identically():
    # the prefix cache snapshots ctx caches as (k, v) copies and assigns them back; the Markov
    # features of the newest row live in the 4th entry, so a restore drafts exactly the same
    d = _bound_drafter()
    ctx = d.make_ctx_cache()
    fused = mx.random.normal((1, 12, 4 * 64))
    d.update_context(fused, ctx_offset=0, ctx_caches=ctx)
    want = _draft(d, ctx, 7, 12).tolist()
    snap = [(mx.array(c.k), mx.array(c.v)) for c in ctx]
    other = d.make_ctx_cache()
    d.update_context(mx.random.normal((1, 12, 4 * 64)), ctx_offset=0, ctx_caches=other)
    for c, (k, v) in zip(other, snap):
        c.k, c.v = k, v
    assert _draft(d, other, 7, 12).tolist() == want
    # trim_to (prefix-cache rungs) keeps the row cache aligned with the K/V
    for c in ctx:
        c.trim_to(8)
    assert {c.length for c in ctx} == {8}


def test_drafter_engine_window_applies_to_the_global_layer_only():
    d = _bound_drafter()
    d.ctx_window = 2
    assert [ly.self_attn.ctx_window for ly in d.layers] == [None, 2, None]
    assert [ly.self_attn.window for ly in d.layers] == [4, None, 4]
    d.ctx_window = None
    assert d.ctx_window is None


# ---------------------------------------------------------------- the loader
def _write_target_dir(tmp_path, with_head=True):
    cfg = _target_config()
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    if with_head:
        d = _bound_drafter()
        from mlx.utils import tree_flatten

        mx.save_safetensors(str(tmp_path / "mtp.safetensors"), dict(tree_flatten(d.parameters())))
    return tmp_path


def test_load_drafter_routes_the_bundled_head_before_the_nvfp4_converter(tmp_path, monkeypatch):
    from mlx_dspark import load as L
    from mlx_dspark import nvfp4_convert

    path = _write_target_dir(tmp_path)
    assert has_bundled_head(str(path))

    def boom(p):
        raise AssertionError("the 20 GB target must never reach the NVFP4 drafter converter")

    monkeypatch.setattr(nvfp4_convert, "ensure_converted", boom)
    d, cfg = L.load_drafter(str(path), quantize=True)
    assert isinstance(d, Inkling2DSparkDrafter) and cfg.family == "inkling2"
    # the Markov head stays plain (its fp32 split + fold need the weights); the layers quantize
    assert isinstance(d.markov_head.rnn_proj, nn.Linear)
    assert isinstance(d.layers[0].mlp.gate_proj, nn.QuantizedLinear)
    assert L.drafter_lineage(str(path)) == "dspark"


def test_no_head_file_is_not_a_bundled_head(tmp_path):
    assert not has_bundled_head(str(_write_target_dir(tmp_path, with_head=False)))


# ---------------------------------------------------------------- kernels
def _quant_switch(e, n, k, mode, seed=1):
    mx.random.seed(seed)
    gsz, bits = (16, 4) if mode == "nvfp4" else (32, 8)
    w = mx.random.normal((e, n, k)) * 0.05
    out = mx.quantize(w, group_size=gsz, bits=bits, mode=mode)
    m = nn.Module()
    m.weight, m.scales = out[0], out[1]
    m.mode, m.bits, m.group_size = mode, bits, gsz
    m.freeze()

    def call(x, indices, sorted_indices=False):
        return mx.gather_qmm(x, m.weight, m.scales, rhs_indices=indices, transpose=True,
                             group_size=gsz, bits=bits, mode=mode, sorted_indices=sorted_indices)
    return m, call


def _per_route(call, x, eids, src, **kw):
    """The stock arithmetic: one single-row gather_qmm product per route."""
    return call(x[src][:, None, None, :], eids[:, None].astype(mx.uint32), **kw).reshape(
        eids.shape[0], -1)


@metal
@pytest.mark.parametrize("mode,k", [("nvfp4", 512), ("nvfp4", 384), ("mxfp8", 512),
                                    ("mxfp8", 384)])       # MLX's fast and general qmv, each
def test_run_qmv_is_bit_identical_to_gather_qmm_for_every_run_length(mode, k):
    from mlx_dspark.inkling_kernels import fast_variant, run_qmv

    m, call = _quant_switch(6, 64, k, mode)
    assert fast_variant(m.bits, k) == (k == 512)
    x = mx.random.normal((5, k)).astype(mx.bfloat16)
    # runs of 1, 2, 3, 4 and 6 equal experts (6 = a pass of four and a pass of two)
    experts = mx.array([5, 0, 0, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4], mx.int32)
    src = mx.array([i % 5 for i in range(16)], mx.int32)
    ident = mx.arange(16, dtype=mx.int32)
    ref = _per_route(call, x, experts, src)
    order = mx.argsort(experts).astype(mx.int32)
    got = run_qmv(x, m, experts[order], src[order], order)      # sorted, written back in place
    assert got.dtype == mx.bfloat16 and bool(mx.array_equal(got, ref).item())
    # any route order is correct: unsorted, every route is its own run
    shuffled = mx.array([3, 12, 0, 9, 6, 15, 1, 10, 4, 13, 7, 2, 11, 5, 14, 8], mx.int32)
    assert bool(mx.array_equal(run_qmv(x, m, experts[shuffled], src[shuffled], shuffled), ref))
    assert bool(mx.array_equal(run_qmv(x, m, experts, src, ident), ref))


@metal
def test_run_qmv_applies_the_per_expert_global_scale_like_gather_qmm():
    from mlx_dspark.inkling_kernels import run_qmv

    if "global_scale" not in (mx.gather_qmm.__doc__ or ""):
        pytest.skip("this mlx has no gather_qmm(global_scale=)")
    m, _ = _quant_switch(6, 64, 384, "nvfp4")
    factor = mx.array([2688.0 * v for v in (0.5, 0.013, 1.7, 0.2, 0.09, 3.1)], mx.float32)
    m._mlx_dspark_factor = factor                 # what install_cached_global_scales leaves
    x = mx.random.normal((3, 384)).astype(mx.bfloat16)
    eids = mx.array([0, 0, 2, 3, 3, 3, 5], mx.int32)
    src = mx.array([0, 1, 2, 0, 1, 2, 1], mx.int32)
    ref = mx.gather_qmm(x[src][:, None, None, :], m.weight, m.scales,
                        rhs_indices=eids[:, None].astype(mx.uint32), transpose=True,
                        group_size=16, bits=4, mode="nvfp4", global_scale=factor).reshape(7, -1)
    assert bool(mx.array_equal(run_qmv(x, m, eids, src, mx.arange(7, dtype=mx.int32)), ref))


def _route_reference(logits, bias, top_k, route_scale):
    """mlx-vlm's ``_route`` after its score matmul, in MLX ops."""
    scores = mx.sigmoid(logits)
    indices = mx.argpartition(-(scores + bias), top_k - 1, axis=-1)[..., :top_k]
    weights = mx.take_along_axis(scores, indices, axis=-1)
    return indices, weights / (weights.sum(axis=-1, keepdims=True) + 1e-20) * route_scale


@metal
@pytest.mark.parametrize("rows", [1, 3, 8])
def test_fused_route_selects_weighs_and_sorts_like_the_stock_ops(rows):
    from mlx_dspark.inkling_kernels import fused_route

    experts, k = 128, 8
    logits = mx.random.normal((rows, experts), key=mx.random.key(rows)) * 3
    bias = mx.random.normal((experts,), key=mx.random.key(99)) * 0.1
    indices, weights = _route_reference(logits, bias, k, 8.0)
    eids, src, order, wts = fused_route(logits, bias, k, 8.0)
    assert bool(mx.array_equal(wts, weights.reshape(-1)))                  # bit for bit
    flat = indices.reshape(-1).astype(mx.int32)
    assert bool(mx.array_equal(flat[order], eids))                         # same selection ...
    assert bool(mx.array_equal(src, order // k))
    assert sorted(order.tolist()) == list(range(rows * k))                 # ... every route once
    assert eids.tolist() == sorted(eids.tolist())                          # sorted by expert


@metal
def test_fused_route_breaks_ties_like_the_stock_stable_sort():
    from mlx_dspark.inkling_kernels import fused_route

    experts, k = 128, 8
    logits = mx.zeros((2, experts))                                        # every score 0.5
    bias = mx.zeros((experts,))
    bias[100] = 0.5
    bias[7] = 0.5
    bias[50] = 0.25
    indices, weights = _route_reference(logits, bias, k, 8.0)
    assert indices[0].tolist() == [7, 100, 50, 0, 1, 2, 3, 4]              # the stock order
    eids, _, order, wts = fused_route(logits, bias, k, 8.0)
    picked = mx.zeros((2 * k,), mx.int32)
    picked[order] = eids
    assert picked.tolist() == indices.reshape(-1).tolist()
    assert bool(mx.array_equal(wts, weights.reshape(-1)))


@metal
def test_combine_routes_is_bit_identical_to_the_stock_ops():
    from mlx_dspark.inkling_kernels import combine_routes

    rows, k, hidden = 3, 8, 256
    down = (mx.random.normal((rows * k, hidden)) * 3).astype(mx.bfloat16)
    weights = mx.random.uniform(shape=(rows * k,)) * 2
    ref = ((down.reshape(rows, k, hidden).astype(mx.float32)
            * weights.reshape(rows, k)[..., None]).sum(axis=-2)).astype(mx.bfloat16)
    got = combine_routes(down, weights, k)
    assert got.shape == (rows, hidden) and bool(mx.array_equal(got, ref))


def _fake_moe(mode="nvfp4", experts=16, hidden=256, inner=64, top_k=4):
    """A SparseMoE-shaped module (what fused_moe reads) and its stock forward in MLX ops."""
    up, call_up = _quant_switch(experts, inner, hidden, mode, seed=1)
    gate, call_gate = _quant_switch(experts, inner, hidden, mode, seed=2)
    down, call_down = _quant_switch(experts, hidden, inner, mode, seed=3)
    moe, sw = nn.Module(), nn.Module()
    sw.up_proj, sw.gate_proj, sw.down_proj = up, gate, down

    def activation(u, g):                      # mlx-vlm's softcapped SwiGLU
        dtype = g.dtype
        g, u = g.astype(mx.float32), u.astype(mx.float32)
        return (mx.tanh(g / 4.0) * 4.0 * mx.sigmoid(g) * mx.tanh(u / 25.0) * 25.0).astype(dtype)

    sw.activation = activation
    moe.switch_mlp = sw
    moe.gate_weight = mx.random.normal((experts, hidden), key=mx.random.key(7)).astype(mx.bfloat16)
    moe.e_score_correction_bias = mx.random.normal((experts,), key=mx.random.key(8)) * 0.1
    moe.top_k, moe.route_scale = top_k, 8.0

    def stock_row(x):                          # x [1, 1, hidden]: the decode call
        logits = x.astype(mx.float32) @ moe.gate_weight.astype(mx.float32).T
        indices, weights = _route_reference(logits, moe.e_score_correction_bias, top_k, 8.0)
        xe = mx.expand_dims(x, (-2, -3))
        act = activation(call_up(xe, indices), call_gate(xe, indices))
        out = call_down(act, indices).squeeze(-2)
        return (out.astype(mx.float32) * weights[..., None]).sum(axis=-2).astype(out.dtype)

    return moe, stock_row


@metal
@pytest.mark.parametrize("mode", ["nvfp4", "mxfp8"])
def test_fused_moe_rows_are_bit_identical_to_single_row_stock_calls(mode):
    from mlx_dspark.inkling_kernels import fused_moe, moe_spec

    moe, stock_row = _fake_moe(mode)
    assert moe_spec(moe)
    a, b, c = (mx.random.normal((1, 1, 256), key=mx.random.key(i)).astype(mx.bfloat16)
               for i in (11, 12, 13))
    # repeated rows route alike: runs of 1..5 routes per expert in one call
    for rows in ([c], [a, b, b, b, c], [a, a, a, a, a, b, b]):
        ref = mx.concatenate([stock_row(r) for r in rows], axis=1)
        got = fused_moe(moe, mx.concatenate(rows, axis=1))
        assert got.shape == ref.shape and got.dtype == mx.bfloat16
        assert bool(mx.array_equal(got, ref)), len(rows)


@metal
def test_fused_moe_routes_in_ops_past_one_threadgroup(monkeypatch):
    from mlx_dspark import inkling_kernels as K

    moe, stock_row = _fake_moe()
    rows = [mx.random.normal((1, 1, 256), key=mx.random.key(i)).astype(mx.bfloat16) for i in range(5)]
    monkeypatch.setattr(K, "ROUTE_MAX_THREADS", 16)            # 5 rows x 16 experts no longer fit
    ref = mx.concatenate([stock_row(r) for r in rows], axis=1)
    assert bool(mx.array_equal(K.fused_moe(moe, mx.concatenate(rows, axis=1)), ref))


@metal
def test_expert_views_bind_the_same_weights_and_come_off_again():
    from types import SimpleNamespace

    from mlx_dspark import inkling_kernels as K

    moe, _ = _fake_moe()
    lm = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=moe)]))
    x = mx.random.normal((1, 3, 256), key=mx.random.key(5)).astype(mx.bfloat16)
    ref = K.fused_moe(moe, x)
    mx.eval(ref)
    assert K.set_expert_views(lm) == 3
    up = moe.switch_mlp.up_proj
    w, s = K._bound(up)
    assert w.size == 8 and s.size == 8 and w.dtype == up.weight.dtype
    assert "_mlx_dspark_views" not in up.parameters()           # not a parameter of the model
    assert bool(mx.array_equal(K.fused_moe(moe, x), ref))
    assert K.set_expert_views(lm, False) == 0 and K._bound(up)[0] is up.weight


@metal
def test_head_f32_matches_fp32_and_is_row_invariant():
    from mlx_dspark.inkling_kernels import head_f32

    mx.random.seed(3)
    w = (mx.random.normal((300, 512)) * 0.05).astype(mx.bfloat16)   # V not a multiple of 8
    h = (mx.random.normal((5, 512))).astype(mx.bfloat16)
    got = head_f32(h, w)
    ref = h.astype(mx.float32) @ w.astype(mx.float32).T
    assert got.dtype == mx.float32 and got.shape == (5, 300)
    assert mx.allclose(got, ref, atol=1e-4, rtol=1e-4)
    alone = mx.concatenate([head_f32(h[i:i + 1], w) for i in range(5)])
    assert bool(mx.array_equal(got, alone).item())


@metal
def test_head_q8_f32_matches_the_quantized_matmul_and_is_row_invariant():
    from mlx_dspark.inkling_kernels import head_q8_f32, head_q8_ok

    mx.random.seed(4)
    lin = nn.Linear(512, 300, bias=False)                           # V not a multiple of 8
    lin.weight = (mx.random.normal((300, 512)) * 0.05).astype(mx.bfloat16)
    q = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=8)
    assert not head_q8_ok(q)                                        # bf16 scales: MLX's own path
    q.scales, q.biases = q.scales.astype(mx.float32), q.biases.astype(mx.float32)
    assert head_q8_ok(q)
    h = mx.random.normal((5, 512)).astype(mx.bfloat16)
    got = head_q8_f32(h, q)
    ref = q(h.astype(mx.float32))
    assert got.dtype == mx.float32 and got.shape == (5, 300)
    assert mx.allclose(got, ref, atol=1e-4, rtol=1e-4)
    alone = mx.concatenate([head_q8_f32(h[i:i + 1], q) for i in range(5)])
    assert bool(mx.array_equal(got, alone).item())


@metal
@pytest.mark.parametrize("bits", [8, 4])
def test_fold_argmax_is_the_argmax_of_the_dequantized_markov_logits(bits):
    from mlx_dspark.inkling_kernels import FOLD_RANK, fold_argmax, quantize_fold

    mx.random.seed(5)
    vocab, vocab_out = 1000, 900                                     # not a threadgroup multiple
    fold_t = (mx.random.normal((FOLD_RANK, vocab)) * 0.1).astype(mx.bfloat16)
    fold = quantize_fold(fold_t, bits)
    rows = mx.dequantize(*fold, group_size=64, bits=bits).astype(mx.float32)      # [V, rank]
    fold_b = (mx.random.normal((vocab,)) * 0.1).astype(mx.bfloat16)
    for trial in range(6):
        latent = mx.tanh(mx.random.normal((FOLD_RANK,))).astype(mx.bfloat16)
        base = mx.random.normal((vocab,)).astype(mx.bfloat16 if trial % 2 else mx.float32)
        ref = (rows @ latent.astype(mx.float32) + fold_b.astype(mx.float32)
               + base.astype(mx.float32))
        got = fold_argmax(latent, fold, fold_b, base, bits, vocab_out)
        assert got.shape == (1,) and got.dtype == mx.int32
        assert got.item() == mx.argmax(ref[:vocab_out]).item()
    # the padded ids are never picked, and equal maxima go to the lowest id
    base = mx.zeros((vocab,), mx.float32)
    base[vocab_out + 5] = 1e4
    base[700] = 1e3
    base[300] = 1e3
    zero = mx.zeros((FOLD_RANK,), mx.bfloat16)
    flat = mx.zeros((vocab,), mx.bfloat16)
    assert fold_argmax(zero, fold, flat, base, bits, vocab_out).item() == 300


@metal
def test_drafter_greedy_block_through_the_fold_kernel_matches_the_reference_path(monkeypatch):
    # rank 256 is the kernel's layout: the same drafter with the fold kept in bf16 (the
    # reference: full-vocab bias, softcap, argmax) drafts the same block from the same context
    from mlx.utils import tree_map

    from mlx_dspark import inkling as I

    def bound(fold_bits):
        monkeypatch.setattr(I, "FOLD_BITS", fold_bits)
        mx.random.seed(7)
        d = Inkling2DSparkDrafter(draft_config(_target_config(markov_rank=256)))
        d.update(tree_map(lambda p: mx.random.normal(p.shape) * 0.2, d.parameters()))
        d.bind_embed(nn.Embedding(96, 64))
        d.bind_lm_head(nn.Linear(64, 96, bias=False))
        ctx = d.make_ctx_cache()
        d.update_context(mx.random.normal((1, 10, 4 * 64)), ctx_offset=0, ctx_caches=ctx)
        return d, ctx

    mx.random.seed(7)
    ref, ref_ctx = bound(0)
    want = _draft(ref, ref_ctx, 5, 10).tolist()
    assert ref.markov_head._fold_q is None and ref.markov_head._fold_t is not None
    from mlx_dspark import inkling as I2

    assert I2.FOLD_BITS == 0                            # (the reference arm above set it)
    got_d, got_ctx = bound(8)
    got = _draft(got_d, got_ctx, 5, 10).tolist()
    assert got_d.markov_head._fold_q is not None and got_d.markov_head._fold_t is None
    assert len(got) == 7 and max(got) < 90
    # random tiny weights leave real margins: 8-bit rounding of the fold does not move them
    assert got == want
    low_d, low_ctx = bound(4)                           # the default
    low = _draft(low_d, low_ctx, 5, 10).tolist()
    assert low_d.markov_head._fold_bits == 4 and len(low) == 7 and max(low) < 90
    # the sampling path reads the same (quantized) fold as a full-vocabulary bias
    toks, probs = got_d.sample_block_probs(
        got_d.compute_logits(got_d.head_slice(got_d.backbone(
            got_d.embed(mx.array([[5] + [0] * 7])), 10, got_ctx), 3))[0], 5, temperature=0.7)
    assert toks.shape == (3,) and probs.shape == (3, 90)
    assert mx.allclose(probs.sum(axis=-1), mx.ones((3,)), atol=1e-3)


# ---------------------------------------------------------------- the mlx-vlm gate
_INKLING_CFG = "mlx_vlm.models.inkling2.config"


@pytest.mark.parametrize("has_loop", [False, True])
def test_mlx_vlm_gate_asks_for_the_looped_model_not_the_module(monkeypatch, has_loop):
    """mlx-vlm 0.8.0 ships models/inkling2 without the looped model: presence is not support."""
    import sys
    import types

    from mlx_dspark.inkling import mlx_vlm_supports_inkling2

    fake = types.ModuleType(_INKLING_CFG)
    if has_loop:
        fake.LoopConfig = object
    monkeypatch.setitem(sys.modules, _INKLING_CFG, fake)
    assert mlx_vlm_supports_inkling2("inkling2") is True
    assert mlx_vlm_supports_inkling2("inkling2_loop") is has_loop
    assert mlx_vlm_supports_inkling2() is has_loop


def test_mlx_vlm_gate_without_the_module(monkeypatch):
    import sys

    from mlx_dspark.inkling import mlx_vlm_supports_inkling2

    monkeypatch.setitem(sys.modules, _INKLING_CFG, None)       # import raises ImportError
    assert mlx_vlm_supports_inkling2("inkling2") is False
    assert mlx_vlm_supports_inkling2("inkling2_loop") is False


def test_load_target_names_the_fix_on_an_mlx_vlm_without_the_looped_model(tmp_path, monkeypatch):
    import json
    import sys
    import types

    from mlx_dspark.inkling import MLX_VLM_HINT
    from mlx_dspark.load import load_target

    monkeypatch.setitem(sys.modules, _INKLING_CFG, types.ModuleType(_INKLING_CFG))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "inkling2_loop"}))
    with pytest.raises(ValueError) as err:
        load_target(str(tmp_path))
    assert MLX_VLM_HINT in str(err.value) and "git+https" in str(err.value)


# ---------------------------------------------------------------- the Target (mlx-vlm inkling2)
def _tiny_inkling():
    from mlx_dspark.inkling import mlx_vlm_supports_inkling2

    if not mlx_vlm_supports_inkling2("inkling2_loop"):
        pytest.skip("this mlx-vlm has no looped Inkling 2 model (0.8.0 ships the plain one only)")
    from mlx_vlm.models.inkling2 import Model, ModelConfig
    from mlx_vlm.models.inkling2.config import TextConfig, VisionConfig

    text = TextConfig(model_type="inkling2_loop_text", hidden_size=128, num_hidden_layers=6,
                      vocab_size=64, unpadded_vocab_size=60, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=32, swa_num_attention_heads=4,
                      swa_num_key_value_heads=2, swa_head_dim=32, sliding_window_size=8,
                      local_layer_ids=[0, 2, 3, 4], d_rel=4, rel_extent=8,
                      log_scaling_n_floor=4, intermediate_size=64, n_routed_experts=4,
                      num_experts_per_tok=2, n_layers_per_attn_res_block=2,
                      loop_config={"core_start": 2, "core_end": 4, "num_passes": 2,
                                   "gate_rank": 4})
    vision = VisionConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                          projection_hidden_dim=32)
    cfg = ModelConfig(text_config=text, vision_config=vision, model_type="inkling2_loop")
    mx.random.seed(0)
    model = Model(cfg)
    from mlx.utils import tree_map

    model.update(tree_map(lambda p: mx.random.normal(p.shape) * 0.1, model.parameters()))
    mx.eval(model.parameters())
    return model


def test_target_inkling_verify_rollback_equals_plain_forwards():
    from mlx_dspark.target import Target

    model = _tiny_inkling()
    t = Target(model, tokenizer=None)
    assert t._inkling and t.family == "inkling2"
    tap = [1, 3, 5, 8]
    prompt = mx.array([[1, 5, 9, 2, 7, 3, 11, 4, 6, 8, 10, 12]])   # past the 8-token window

    def fresh():
        c = t.make_cache()
        t.prefill(prompt, c, tap)
        return c

    # verify [anchor, d1, d2, d3], keep anchor + d1 (2 rejected), then one more step
    a = fresh()
    _, fused_v = t.verify(mx.array([[13, 14, 15, 16]]), a, tap)
    assert fused_v.shape == (1, 4, 4 * 128)
    t.rollback(a, n_rejected=2, accepted=[14])
    got = t.plain(mx.array([[17]]), a)
    b = fresh()
    t.plain(mx.array([[13, 14]]), b)
    want = t.plain(mx.array([[17]]), b)
    assert mx.allclose(got, want, atol=1e-3, rtol=1e-3)
    # a full accept and a width-1 step leave no transaction behind
    t.verify(mx.array([[18, 19]]), a, tap)
    t.rollback(a, n_rejected=0, accepted=[19])
    t.verify(mx.array([[20]]), a, tap)
    t.rollback(a, n_rejected=0, accepted=[])
    assert t._txn is None


def test_target_inkling_target_bits_quantizes_tags_the_curve_and_still_rolls_back():
    from mlx_dspark.target import Target

    t = Target(_tiny_inkling(), tokenizer=None, target_bits=8)
    assert t.target_bits == 8 and t.calibration_tag.endswith("|tb8")
    attn = t.model.language_model.model.layers[0].self_attn
    assert type(attn.q_proj).__name__ == "QuantizedLinear" and attn.q_proj.bits == 8
    assert Target(_tiny_inkling(), tokenizer=None).target_bits == 0
    with pytest.raises(ValueError, match="4-bit"):
        Target(_tiny_inkling(), tokenizer=None, target_bits=4)
    tap = [1, 3, 5, 8]
    prompt = mx.array([[1, 5, 9, 2, 7, 3, 11, 4, 6, 8, 10, 12]])

    def fresh():
        c = t.make_cache()
        t.prefill(prompt, c, tap)
        return c

    a = fresh()
    t.verify(mx.array([[13, 14, 15, 16]]), a, tap)
    t.rollback(a, n_rejected=2, accepted=[14])
    got = t.plain(mx.array([[17]]), a)
    b = fresh()
    t.plain(mx.array([[13, 14]]), b)
    want = t.plain(mx.array([[17]]), b)
    assert got.dtype == mx.float32 and mx.allclose(got, want, atol=1e-3, rtol=1e-3)


def test_target_bits_applies_to_inkling_configs_only():
    from mlx_dspark.load import supports_target_bits

    assert supports_target_bits({"model_type": "inkling2_loop"})
    assert supports_target_bits({"model_type": "inkling2"})
    assert not supports_target_bits({"model_type": "qwen3"}) and not supports_target_bits({})


def test_target_inkling_reset_spec_aborts_an_open_verify():
    from mlx_dspark.target import Target

    model = _tiny_inkling()
    t = Target(model, tokenizer=None)
    tap = [1, 3, 5, 8]
    prompt = mx.array([[1, 5, 9, 2, 7, 3, 11, 4, 6, 8]])
    a, b = t.make_cache(), t.make_cache()
    t.prefill(prompt, a, tap)
    t.prefill(prompt, b, tap)
    t.verify(mx.array([[13, 14, 15]]), a, tap)
    t.reset_spec()                                   # an abandoned round: back to pre-verify
    assert mx.allclose(t.plain(mx.array([[21]]), a), t.plain(mx.array([[21]]), b),
                       atol=1e-3, rtol=1e-3)


def test_bench_rows_are_natural_text_not_one_repeated_token():
    # the verify-curve harness times Inkling on consecutive natural tokens: one repeated token
    # routes every row to the same experts and under-prices the grouped verify MoE
    from mlx_dspark.target import Target

    class _Tok:
        def encode(self, text):
            return [ord(c) for c in text]

    t = Target.__new__(Target)
    t.tokenizer = _Tok()
    rows = t._bench_rows(9)
    assert rows.shape == (1, 9) and len(set(rows[0].tolist())) > 4
    t2 = Target.__new__(Target)
    t2.tokenizer = None                                   # no tokenizer: distinct fallback ids
    assert len(set(t2._bench_rows(8)[0].tolist())) == 8


def test_bind_target_builds_a_quantized_drafter_only_head(monkeypatch):
    # base logits go through a quantized COPY of the target's head (drafting-only); the Markov
    # fold keeps the exact head; rebuilt only when the head changes; 0 bits = the bf16 head
    from types import SimpleNamespace

    from mlx_dspark import inkling as I

    d = _bound_drafter()
    head = nn.Linear(64, 96, bias=False)
    tgt = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(lm_head=head)))
    d.bind_target(tgt)
    assert isinstance(d._base_head, nn.QuantizedLinear) and d._base_head.bits == 6
    first = d._base_head
    d.bind_target(tgt)
    assert d._base_head is first                                  # same head: no rebuild
    ctx = d.make_ctx_cache()
    d.update_context(mx.random.normal((1, 6, 4 * 64)), ctx_offset=0, ctx_caches=ctx)
    assert _draft(d, ctx, 3, 6).shape == (7,)
    monkeypatch.setattr(I, "DRAFT_HEAD_BITS", 0)
    d.bind_target(tgt)
    assert d._base_head is None


@metal
def test_bind_target_drafts_from_the_8bit_head_a_target_bits_target_already_holds():
    # --target-bits 8 keeps an 8-bit copy of the head for the target's logits: the drafter's
    # base logits read that one (fp32, through the same kernel) instead of building its own
    from types import SimpleNamespace

    from mlx_dspark.inkling_kernels import _DENSE_HEAD

    d = _bound_drafter()
    head = nn.Linear(64, 96, bias=False)
    head.weight = (mx.random.normal((96, 64)) * 0.2).astype(mx.bfloat16)
    shared = nn.QuantizedLinear.from_linear(head, group_size=64, bits=8)
    shared.scales, shared.biases = shared.scales.astype(mx.float32), shared.biases.astype(mx.float32)
    lm = SimpleNamespace(lm_head=head)
    tgt = SimpleNamespace(model=SimpleNamespace(language_model=lm))
    d.bind_target(tgt)
    own = d._base_head
    assert isinstance(own, nn.QuantizedLinear)
    lm.__dict__[_DENSE_HEAD] = shared                 # 64 % 256: not the kernel's layout
    d.bind_target(tgt)
    assert d._base_head is own
    # a head the kernel serves (hidden a multiple of 256)
    wide = Inkling2DSparkDrafter(draft_config(_target_config(hidden_size=256, head_dim=64)
                                              | {"text_config": {"hidden_size": 256, "vocab_size": 96,
                                                                 "unpadded_vocab_size": 90,
                                                                 "final_logit_softcapping": 30.0}}))
    from mlx.utils import tree_map

    wide.update(tree_map(lambda p: mx.random.normal(p.shape) * 0.2, wide.parameters()))
    big = nn.Linear(256, 96, bias=False)
    big.weight = (mx.random.normal((96, 256)) * 0.2).astype(mx.bfloat16)
    qbig = nn.QuantizedLinear.from_linear(big, group_size=64, bits=8)
    qbig.scales, qbig.biases = qbig.scales.astype(mx.float32), qbig.biases.astype(mx.float32)
    lm2 = SimpleNamespace(lm_head=big)
    lm2.__dict__[_DENSE_HEAD] = qbig
    wide.bind_embed(nn.Embedding(96, 256))
    wide.bind_lm_head(big)
    wide.bind_target(SimpleNamespace(model=SimpleNamespace(language_model=lm2)))
    assert callable(wide._base_head) and not isinstance(wide._base_head, nn.Module)
    assert wide._base_src is qbig
    hidden = mx.random.normal((1, 3, 256)).astype(mx.bfloat16)
    got = wide.compute_logits(hidden)
    assert got.shape == (1, 3, 96) and got.dtype == mx.float32
    assert mx.allclose(got, qbig(hidden.astype(mx.float32)), atol=1e-3, rtol=1e-3)
    ctx = wide.make_ctx_cache()
    wide.update_context(mx.random.normal((1, 6, 4 * 256)), ctx_offset=0, ctx_caches=ctx)
    assert _draft(wide, ctx, 3, 6).shape == (7,)
