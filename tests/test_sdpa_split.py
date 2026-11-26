"""Wide-verify SDPA split: correctness, gating, and the scoped patch (2026-08-21).

The split computes a wide-q attention in <=max_chunk-row sub-calls to dodge mlx's multi-row
SDPA cliff. Per query row the math is identical, so it is per-row-equivalent (fp-tie: bf16
reduction-order noise only). These tests pin equivalence across mask forms, the (q_len, kv)
gate that keeps it off prefill / short-KV, and that the monkeypatch swaps and restores.
"""
import mlx.core as mx
import pytest

from mlx_dspark.sdpa_split import (
    _ORIG_SDPA,
    SplitConfig,
    _partition,
    patchable,
    sdpa_split,
    split_sdpa,
)

Hq, Hk, D = 8, 2, 64
SCALE = 1.0 / (D ** 0.5)
# small min_kv so the split fires at a tiny test KV (correctness is L-independent)
CFG = SplitConfig(min_q=6, max_q=15, min_kv=64, max_chunk=5)


def _mk(qlen, L, mask):
    mx.random.seed(0)
    q = mx.random.normal((1, Hq, qlen, D)).astype(mx.bfloat16)
    k = mx.random.normal((1, Hk, L, D)).astype(mx.bfloat16)
    v = mx.random.normal((1, Hk, L, D)).astype(mx.bfloat16)
    if mask == "arr":
        rows = mx.arange(L - qlen, L).reshape(qlen, 1)
        cols = mx.arange(L).reshape(1, L)
        m = mx.where(cols <= rows, mx.array(0.0, mx.bfloat16), mx.array(-mx.inf, mx.bfloat16))
        m = m.reshape(1, 1, qlen, L)
    else:
        m = mask
    mx.eval(q, k, v)
    return q, k, v, m


def _maxabs(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def test_patchable():
    assert patchable() is True


def test_partition_sums_and_bounds():
    for q in range(1, 33):
        parts = _partition(q, 5)
        assert sum(parts) == q
        assert all(1 <= p <= 5 for p in parts)
        assert max(parts) - min(parts) <= 1        # balanced


@pytest.mark.parametrize("mask", ["causal", "arr", None])
@pytest.mark.parametrize("qlen", [6, 7, 8, 12])
def test_split_equals_single_when_firing(mask, qlen):
    q, k, v, m = _mk(qlen, 512, mask)
    single = _ORIG_SDPA(q, k, v, scale=SCALE, mask=m)
    split = split_sdpa(q, k, v, scale=SCALE, mask=m, cfg=CFG)
    assert split.shape == single.shape
    assert _maxabs(single, split) < 0.02          # fp-tie class


@pytest.mark.parametrize("qlen,L", [(1, 512), (4, 512), (8, 32)])
def test_gate_passthrough_is_bit_identical(qlen, L):
    # outside the (q_len window x min_kv) gate the split returns the single call verbatim
    q, k, v, m = _mk(qlen, L, "causal")
    single = _ORIG_SDPA(q, k, v, scale=SCALE, mask=m)
    out = split_sdpa(q, k, v, scale=SCALE, mask=m, cfg=CFG)
    assert _maxabs(single, out) == 0.0


def test_scope_swaps_and_restores():
    assert mx.fast.scaled_dot_product_attention is _ORIG_SDPA
    with sdpa_split(CFG):
        assert mx.fast.scaled_dot_product_attention is not _ORIG_SDPA
    assert mx.fast.scaled_dot_product_attention is _ORIG_SDPA


def test_scope_none_is_noop():
    with sdpa_split(None):
        assert mx.fast.scaled_dot_product_attention is _ORIG_SDPA


def test_patch_routes_wide_calls():
    q, k, v, m = _mk(8, 512, "causal")
    single = _ORIG_SDPA(q, k, v, scale=SCALE, mask=m)
    with sdpa_split(CFG):
        routed = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE, mask=m)
    assert _maxabs(single, routed) < 0.02


def test_library_default_off():
    # a plain speculative_generate must not silently split (library-off doctrine)
    from mlx_dspark import generate
    assert generate.SDPA_SPLIT_CFG is None


def test_masked_calls_split_from_their_own_cliff():
    # mlx 0.32.5 moved the causal cliff (q 6 -> 9 at Qwen3.8's shape) but not the masked
    # path's: a tree verify's array mask still leaves the vector kernel past max_chunk rows
    cfg = SplitConfig(min_q=9, max_q=15, min_kv=64, max_chunk=5, masked_min_q=6)
    calls = []

    def counting(q, k, v, **kw):
        calls.append(q.shape[-2])
        return _ORIG_SDPA(q, k, v, **kw)

    q, k, v, m = _mk(8, 512, "causal")
    out = split_sdpa(q, k, v, scale=SCALE, mask=m, cfg=cfg, _orig=counting)
    assert calls == [8] and _maxabs(out, _ORIG_SDPA(q, k, v, scale=SCALE, mask=m)) == 0.0
    calls.clear()
    q, k, v, m = _mk(8, 512, "arr")
    out = split_sdpa(q, k, v, scale=SCALE, mask=m, cfg=cfg, _orig=counting)
    assert calls == [4, 4] and _maxabs(out, _ORIG_SDPA(q, k, v, scale=SCALE, mask=m)) < 0.02
    calls.clear()                                  # without it, masked calls follow min_q
    split_sdpa(q, k, v, scale=SCALE, mask=m, cfg=SplitConfig(min_q=9, max_q=15, min_kv=64,
                                                             max_chunk=5), _orig=counting)
    assert calls == [8]


def test_a_probed_window_past_the_vector_limit_splits_masked_calls_from_it(monkeypatch):
    import importlib
    from types import SimpleNamespace

    from mlx_dspark import generate as G

    C = importlib.import_module("mlx_dspark.calibrate")
    target = SimpleNamespace(model=SimpleNamespace(args=SimpleNamespace(
        num_attention_heads=24, num_key_value_heads=4, head_dim=256)))
    try:
        for win, masked in (({"min_q": 9, "max_q": 15, "min_kv": 4096, "max_chunk": 5}, 6),
                            ({"min_q": 6, "max_q": 15, "min_kv": 4096, "max_chunk": 5}, None)):
            monkeypatch.setattr(C, "measure_split_window", lambda *a, w=win, **kw: dict(w))
            cfg = C.apply_sdpa_split(target, verbose=False)
            assert cfg.min_q == win["min_q"] and cfg.masked_min_q == masked
    finally:
        G.SDPA_SPLIT_CFG = None
