"""Multi-row decode-attention kernel (multirow_attn.py): numerics, gating, the scoped patch.

The kernel is fp-tie class (fp32 softmax/accumulation, input-dtype MMA operands), so the
load-bearing checks are (1) it matches an fp32 reference as closely as mlx's own SDPA does,
over every structural case it claims — causal and bidirectional, GQA 1..16, head dims 64 /
128 / 256 (the d-split path), KV lengths off the 16-key tile grid (the shifted tail tile),
strided cache views, batch > 1, bf16 and fp16, row groups past one threadgroup; (2) the gate
never routes anything outside a measured window (array masks, sinks, unmeasured shapes,
prefill-wide q, short KV) — those must be mlx's exact bits; (3) the patch composes with the
SDPA split and always restores.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_dspark import multirow_attn as mra
from mlx_dspark.multirow_attn import (
    MultirowConfig,
    Window,
    eligible,
    multirow_attention,
    multirow_sdpa,
    plan,
)
from mlx_dspark.sdpa_split import SplitConfig, sdpa_split


def _mk(B, hq, hk, q_len, L, d, dtype=mx.bfloat16, slack=0, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((B, hq, q_len, d)).astype(dtype)
    k = mx.random.normal((B, hk, L + slack, d)).astype(dtype)
    v = mx.random.normal((B, hk, L + slack, d)).astype(dtype)
    if slack:                                    # a KVCache-style view into a larger buffer
        k, v = k[:, :, :L, :], v[:, :, :L, :]
    mx.eval(q, k, v)
    return q, k, v


def _err(out, q, k, v, causal):
    scale = q.shape[-1] ** -0.5
    ref = mra._ORIG_SDPA(q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
                         scale=scale, mask="causal" if causal else None)
    return float(mx.abs(out.astype(mx.float32) - ref).max())


CASES = [
    # B, hq, hk, q_len, L, d
    (1, 24, 4, 8, 1037, 256),     # Qwen3.8 geometry, d-split, off-grid tail
    (1, 24, 4, 1, 300, 256),
    (1, 32, 8, 15, 777, 128),     # DSpark drafter block
    (1, 32, 8, 2, 64, 128),
    (1, 32, 2, 16, 500, 128),     # GQA 16 -> 256 rows: row groups
    (2, 8, 8, 5, 129, 64),        # MHA, batch 2
    (1, 16, 2, 7, 16, 128),       # KV exactly one tile
    (1, 4, 1, 3, 23, 64),         # MQA, tail shorter than a tile past one tile
]


@pytest.mark.parametrize("B,hq,hk,q_len,L,d", CASES)
@pytest.mark.parametrize("causal", [True, False])
def test_matches_reference_as_well_as_mlx(B, hq, hk, q_len, L, d, causal):
    q, k, v = _mk(B, hq, hk, q_len, L, d, slack=37)
    scale = d ** -0.5
    out = multirow_sdpa(q, k, v, scale=scale, causal=causal and q_len > 1)
    mask = "causal" if causal and q_len > 1 else None
    mine = _err(out, q, k, v, mask is not None)
    theirs = _err(mra._ORIG_SDPA(q, k, v, scale=scale, mask=mask), q, k, v, mask is not None)
    assert out.shape == q.shape and out.dtype == q.dtype
    assert mine <= 2.0 * theirs + 2e-3, (mine, theirs)


def test_fp16_inputs():
    q, k, v = _mk(1, 24, 4, 6, 900, 256, dtype=mx.float16)
    out = multirow_sdpa(q, k, v, scale=256 ** -0.5, causal=True)
    assert out.dtype == mx.float16
    assert _err(out, q, k, v, True) < 5e-3


def test_causal_rows_see_only_their_prefix():
    # the LAST query row sees every key; the first sees all but the last q_len-1 — make the
    # masked keys dominate so a mask bug would swamp the output
    q, k, v = _mk(1, 4, 4, 4, 64, 64)
    k = mx.concatenate([k[:, :, :-3], 50.0 * mx.ones_like(k[:, :, -3:])], axis=2)
    v = mx.concatenate([v[:, :, :-3], 1e3 * mx.ones_like(v[:, :, -3:])], axis=2)
    out = multirow_sdpa(q, k, v, scale=0.125, causal=True)
    # rows that may see the 1e3-valued V legitimately round at bf16 ulp ~4 there; a leak
    # into a row that must not see them would be off by ~1e3
    assert _err(out, q, k, v, True) < 50.0


def test_plan_row_groups_and_blocks():
    p = plan(32, 2, 16, 128, 32768)                # 256 rows = 32 tiles -> one full tg
    assert p["NRG"] == 1 and p["NSG"] == 32
    p = plan(64, 2, 16, 128, 32768)                # 512 rows -> 2 row groups
    assert p["NRG"] == 2 and p["RP"] >= 512
    p = plan(24, 4, 8, 256, 32768)                 # d-split at 256
    assert p["DS"] == 2 and p["NSG"] == 12
    assert p["CHUNK"] % 16 == 0 and p["NB"] * p["CHUNK"] >= 32768


CFG = MultirowConfig((Window(8, 2, 64, min_q=2, max_q=16, min_kv=64),))


def test_eligible_gate():
    q, k, v = _mk(1, 8, 2, 4, 128, 64)
    assert eligible(q, k, v, "causal", None, CFG)
    assert eligible(q, k, v, None, None, CFG)
    assert not eligible(q, k, v, mx.ones((1, 1, 4, 128), dtype=mx.bool_), None, CFG)  # array
    assert not eligible(q, k, v, "causal", mx.zeros((8,)), CFG)                        # sinks
    q1, _, _ = _mk(1, 8, 2, 1, 128, 64)
    assert not eligible(q1, k, v, "causal", None, CFG)                 # q_len 1 below window
    qw, _, _ = _mk(1, 8, 2, 17, 128, 64)
    assert not eligible(qw, k, v, "causal", None, CFG)                 # prefill-wide q
    qs, ks, vs = _mk(1, 8, 2, 4, 32, 64)
    assert not eligible(qs, ks, vs, "causal", None, CFG)               # short KV
    qo, ko, vo = _mk(1, 16, 2, 4, 128, 64)
    assert not eligible(qo, ko, vo, "causal", None, CFG)               # unmeasured shape
    assert not eligible(q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
                        "causal", None, CFG)                           # fp32


def test_scope_routes_eligible_and_passes_through_the_rest_bit_exact():
    orig = mx.fast.scaled_dot_product_attention
    q, k, v = _mk(1, 8, 2, 4, 128, 64)
    qo, ko, vo = _mk(1, 16, 2, 4, 128, 64)                             # unmeasured shape
    stock = orig(qo, ko, vo, scale=0.125, mask="causal")
    with multirow_attention(CFG):
        assert mra.active()
        routed = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125, mask="causal")
        passed = mx.fast.scaled_dot_product_attention(qo, ko, vo, scale=0.125, mask="causal")
    assert mx.fast.scaled_dot_product_attention is orig and not mra.active()
    assert _err(routed, q, k, v, True) < 5e-3
    assert float(mx.abs(passed.astype(mx.float32) - stock.astype(mx.float32)).max()) == 0.0


def test_scope_composes_over_the_split_and_restores():
    orig = mx.fast.scaled_dot_product_attention
    split = SplitConfig(min_q=6, max_q=15, min_kv=64, max_chunk=5)
    qa, ka, va = _mk(1, 8, 2, 8, 128, 64)
    arr = mx.ones((1, 1, 8, 128), dtype=mx.bool_)                      # array mask -> split
    with sdpa_split(split), multirow_attention(CFG):
        assert mra.active()
        out = mx.fast.scaled_dot_product_attention(qa, ka, va, scale=0.125, mask=arr)
    assert mx.fast.scaled_dot_product_attention is orig
    ref = orig(qa, ka, va, scale=0.125, mask=arr)
    assert float(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max()) < 0.02


def test_none_config_is_a_noop():
    orig = mx.fast.scaled_dot_product_attention
    with multirow_attention(None):
        assert mx.fast.scaled_dot_product_attention is orig


def test_config_window_lookup():
    assert CFG.window(8, 2, 64) is not None
    assert CFG.window(8, 2, 128) is None
