"""Threadgroup ceilings on M1/M2 — regression tests for the launch abort.

On Mac14,10 (M2 Pro) with mlx-dspark 0.20.1, every speculative run died while calibrating:

    ValueError: Thread group size (1024) is greater than the maximum allowed threads
                per threadgroup (448).

Both kernels that hard-coded a 1024-thread threadgroup were at fault, and the ceiling is
NOT a plain per-chip constant — it depends on the kernel's register footprint and moves as
``S`` moves. ``skinny_qmm`` now measures its own ceiling with the real kernel; ``multirow_attn``
uses the largest value observed to launch on that GPU generation.
"""

import mlx.core as mx
import pytest

from mlx_dspark import multirow_attn, skinny_qmm


@pytest.mark.parametrize("n,rt", [(48, 1), (2048, 1), (4096, 1)])
def test_skinny_qmm_geometry_respects_the_measured_cap(n, rt):
    """``geometry`` must never return an S above this GPU's probed ceiling."""
    cap = skinny_qmm._max_simdgroups()
    nt, s = skinny_qmm.geometry(n, rt)
    assert 1 <= s <= cap, f"S={s} exceeds the measured cap {cap} for n={n}"
    assert s * 32 <= 1024, "no Apple GPU admits more than 1024 threads per threadgroup"


def test_skinny_qmm_cap_is_at_least_one_and_cached():
    """The probe must return a usable value and not re-measure on every call."""
    first = skinny_qmm._max_simdgroups()
    assert first >= 1
    assert skinny_qmm._max_simdgroups() == first


def test_skinny_qmm_real_launch_at_the_reported_cap():
    """The point of the probe: the kernel actually launches at the value it reports.

    This is the assertion that failed with the stock code — ``S=32`` (1024 threads) is
    rejected on M1/M2, so a run that trusted the constant aborted before generating.
    """
    cap = skinny_qmm._max_simdgroups()
    rows, n, k = 6, 2048, 2048
    x = (mx.random.normal((rows, k)) * 0.1).astype(mx.bfloat16)
    w = mx.zeros((n, k // 8), dtype=mx.uint32)
    sc = mx.zeros((n, k // 64), dtype=mx.bfloat16)
    bi = mx.zeros((n, k // 64), dtype=mx.bfloat16)

    skinny_qmm._SIMDGROUPS_MAX = cap
    out = skinny_qmm.qmm(x, w, sc, bi, 4)
    mx.eval(out)
    assert out.shape == (rows, n)


def test_multirow_attn_plan_stays_within_the_m1_m2_ceiling():
    """``plan`` derives its threadgroup from ``MAX_SG``; keep it launchable on M1/M2."""
    p = multirow_attn.plan(hq=24, hk=4, q_len=1, d=128, kv_len=4096, batch=1)
    assert p["NSG"] * 32 <= 448, (
        f"NSG={p['NSG']} is {p['NSG'] * 32} threads — the M1/M2 ceiling for this kernel is 448"
    )
    assert multirow_attn.MAX_SG * 32 <= 448
