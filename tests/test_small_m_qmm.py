"""Model-free tests for the small-M MMA verify kernel (small_m_qmm.py).

The kernel is fp-tie class BY DESIGN (different fp32 accumulation order than
``quantized_matmul``), so unlike wide_gemm there is no bit-identity to assert; the
load-bearing checks are (1) numerics stay within a few bf16 ulps at every window width,
(2) dispatch NEVER touches anything outside the window (M outside [M_MIN, 8], non-bf16
input, unlisted instances) — those must return the stock kernel's exact bits — and
(3) the context patch composes with ``wide_matmul`` and always restores.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_dspark.small_m_qmm import (
    M_MAX,
    M_MIN,
    _orig_call,
    active,
    eligible,
    ids_for_shapes,
    measure_shapes,
    shape_key,
    small_m_matmul,
)
from mlx_dspark.wide_gemm import wide_matmul

K, N = 512, 4096  # smallest eligible shape: N >= 4096, K % 512 == 0


def _qlinear(out_features=N, in_features=K, bits=4, group_size=64, bias=False):
    lin = nn.Linear(in_features, out_features, bias=bias)
    return nn.QuantizedLinear.from_linear(lin, group_size=group_size, bits=bits)


class _Tiny(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.proj = _qlinear(**kw)


def test_eligibility_gates():
    assert eligible(_qlinear())
    assert eligible(_qlinear(bits=8))                          # 8-bit unpack variant
    assert not eligible(_qlinear(bits=2))                      # no 2-bit kernel
    assert not eligible(_qlinear(group_size=32))               # gs64 only
    assert not eligible(_qlinear(out_features=1024))           # N < 4096
    assert not eligible(_qlinear(in_features=K + 128))         # K % 512 != 0 — the
    # K-loop strides 64 values per simdgroup over a K/8 slice, so upstream's %128 gate
    # is too loose; this asserts ours stays tight
    assert not eligible(nn.Linear(K, N))                       # not quantized


def test_kernel_numerics_within_window():
    model = _Tiny()
    mod = model.proj
    ids = ids_for_shapes([shape_key(mod)], model)
    assert ids == frozenset({id(mod)})
    for m in range(M_MIN, M_MAX + 1):
        x = (mx.random.normal((1, m, K)) * 0.1).astype(mx.bfloat16)
        ref = _orig_call(mod, x).astype(mx.float32)
        with small_m_matmul(ids):
            got = mod(x).astype(mx.float32)
        diff = mx.max(mx.abs(ref - got)).item()
        scale = max(mx.max(mx.abs(ref)).item(), 1.0)
        assert diff <= 0.02 * scale, f"M={m}: |d|={diff} vs max|y|={scale}"
        assert got.shape == ref.shape


def test_kernel_numerics_8bit():
    model = _Tiny(bits=8)
    mod = model.proj
    ids = ids_for_shapes([shape_key(mod)], model)
    assert ids == frozenset({id(mod)})
    for m in (M_MIN, M_MAX):
        x = (mx.random.normal((m, K)) * 0.1).astype(mx.bfloat16)
        ref = _orig_call(mod, x).astype(mx.float32)
        with small_m_matmul(ids):
            got = mod(x).astype(mx.float32)
        diff = mx.max(mx.abs(ref - got)).item()
        scale = max(mx.max(mx.abs(ref)).item(), 1.0)
        assert diff <= 0.02 * scale, f"8-bit M={m}: |d|={diff} vs max|y|={scale}"


def test_shape_key_separates_bits():
    assert shape_key(_qlinear()) != shape_key(_qlinear(bits=8))


def test_dispatch_outside_window_is_stock_bits():
    model = _Tiny()
    ids = frozenset({id(model.proj)})
    for m in (1, 2, M_MIN - 1, M_MAX + 1, 32):
        x = (mx.random.normal((m, K)) * 0.1).astype(mx.bfloat16)
        ref = _orig_call(model.proj, x)
        with small_m_matmul(ids):
            got = model.proj(x)
        assert mx.array_equal(ref, got).item(), f"M={m} must be the stock path"


def test_non_bf16_and_unlisted_instances_fall_back():
    model = _Tiny()
    other = _qlinear()
    ids = frozenset({id(model.proj)})
    x16 = (mx.random.normal((M_MIN, K)) * 0.1).astype(mx.float16)
    with small_m_matmul(ids):
        assert mx.array_equal(_orig_call(model.proj, x16), model.proj(x16)).item()
        xbf = (mx.random.normal((M_MIN, K)) * 0.1).astype(mx.bfloat16)
        assert mx.array_equal(_orig_call(other, xbf), other(xbf)).item()


def test_bias_applies_on_kernel_path():
    mod = _qlinear(bias=True)
    model = _Tiny()
    model.proj = mod
    ids = ids_for_shapes([shape_key(mod)], model)
    x = (mx.random.normal((M_MIN, K)) * 0.1).astype(mx.bfloat16)
    ref = _orig_call(mod, x).astype(mx.float32)
    with small_m_matmul(ids):
        got = mod(x).astype(mx.float32)
    assert mx.max(mx.abs(ref - got)).item() <= 0.02 * max(mx.max(mx.abs(ref)).item(), 1.0)


def test_disabled_is_a_noop():
    with small_m_matmul(None):
        assert not active()
    with small_m_matmul(frozenset()):
        assert not active()
    assert nn.QuantizedLinear.__call__ is _orig_call


def test_patch_restores_on_exception():
    with pytest.raises(RuntimeError), small_m_matmul(frozenset({1})):
        raise RuntimeError("boom")
    assert not active()


def test_nests_with_wide_matmul():
    """The prefill wide-GEMM context runs INSIDE a decode loop that holds the small-M
    patch; on exit it must restore the small-M patch (the entry value), not the
    import-time original — otherwise the kernel silently disappears after the first
    prefill of every generation."""
    ids = frozenset({1})
    with small_m_matmul(ids):
        assert active()
        with wide_matmul(512):
            assert not active()          # wide patch holds the class during prefill
        assert active()                  # and hands it back after
    assert not active()
    assert nn.QuantizedLinear.__call__ is _orig_call


def test_measure_shapes_returns_verified_subset():
    model = _Tiny()
    shapes = measure_shapes(model)
    assert isinstance(shapes, list)
    for s in shapes:
        assert s == shape_key(model.proj)
    # an ineligible model yields nothing
    small = _Tiny(out_features=1024)
    assert measure_shapes(small) == []
