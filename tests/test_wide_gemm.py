"""Model-free tests for the prefill wide-GEMM path (wide_gemm.py).

The load-bearing one is :func:`test_bit_identical_above_threshold`: the whole reason this
optimization is allowed to be on by default is that dequantize-once + GEMM produces the
SAME bits as ``mx.quantized_matmul`` at prefill widths. That is an empirical property of
mlx's kernels, not a documented contract, so it is asserted here and will fail loudly if a
future mlx changes qmm's accumulation order (at which point the path becomes an fp-tie
change and needs a decision, not a silent regression).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_dspark.wide_gemm import (
    SAFE_MIN_ROWS,
    _eligible,
    _orig_call,
    active,
    measure_crossover,
    quantized_linears,
    verified_shapes,
    wide_matmul,
    widest_quantized_linear,
)

# Inside the bit-identical region for every dtype. The boundary is the OUTPUT tile grid,
# not K: N=1024 at M=256 disagrees in half precision (see the module docstring), which is
# what makes the runtime check load-bearing rather than decorative.
K, N, ROWS = 4096, 4096, 256


def _qlinear(out_features=N, in_features=K, bits=8, bias=False):
    lin = nn.Linear(in_features, out_features, bias=bias)
    return nn.QuantizedLinear.from_linear(lin, group_size=64, bits=bits)


def test_disabled_is_a_noop():
    with wide_matmul(None):
        assert not active()
    with wide_matmul(0):
        assert not active()
    assert nn.QuantizedLinear.__call__ is _orig_call


def test_patch_applies_and_restores():
    assert not active()
    with wide_matmul(64):
        assert active()
    assert not active()
    assert nn.QuantizedLinear.__call__ is _orig_call


def test_patch_restores_on_exception():
    with pytest.raises(RuntimeError):
        with wide_matmul(64):
            raise RuntimeError("boom")
    assert not active()
    assert nn.QuantizedLinear.__call__ is _orig_call


def test_nested_context_does_not_restore_early():
    with wide_matmul(64):
        with wide_matmul(128):        # inner sees it already active -> leaves it alone
            assert active()
        assert active()               # ...and must NOT have been unpatched by the inner exit
    assert not active()


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("bits", [4, 8])
def test_bit_identical_above_threshold(bits, dtype):
    """The property the whole optimization rests on, across every dtype and bit width
    a target can be loaded in. If this ever fails after an mlx upgrade, the wide path
    has become an fp-tie change and needs a decision — see the module docstring."""
    base = nn.Linear(K, N, bias=False)
    base.set_dtype(dtype)                 # quantize AT the dtype, as a converted repo does
    lin = nn.QuantizedLinear.from_linear(base, group_size=64, bits=bits)
    x = mx.random.normal((ROWS, K)).astype(dtype)
    mx.eval(x, lin.parameters())
    ref = lin(x)
    with wide_matmul(ROWS):
        got = lin(x)
    mx.eval(ref, got)
    assert got.shape == ref.shape
    assert float(mx.abs(got - ref).max()) == 0.0


def test_narrow_input_falls_through_unchanged():
    lin = _qlinear()
    x = mx.random.normal((2, K)).astype(mx.bfloat16)
    mx.eval(x, lin.parameters())
    ref = lin(x)
    with wide_matmul(1024):           # 2 rows is far below the threshold
        got = lin(x)
    mx.eval(ref, got)
    assert float(mx.abs(got - ref).max()) == 0.0


def test_rows_counts_leading_dims_not_just_the_first():
    """A prefill forward is [1, L, K] — the row count is L, not the batch dim 1."""
    lin = _qlinear()
    x = mx.random.normal((1, ROWS, K)).astype(mx.bfloat16)
    mx.eval(x, lin.parameters())
    ref = lin(x)
    with wide_matmul(ROWS):           # the row count is L, not the batch dim 1
        got = lin(x)
    mx.eval(ref, got)
    assert got.shape == (1, ROWS, N)
    assert float(mx.abs(got - ref).max()) == 0.0


def test_bias_is_applied_on_the_wide_path():
    lin = _qlinear(bias=True)
    x = mx.random.normal((ROWS, K)).astype(mx.bfloat16)
    mx.eval(x, lin.parameters())
    ref = lin(x)
    with wide_matmul(ROWS):
        got = lin(x)
    mx.eval(ref, got)
    assert float(mx.abs(got - ref).max()) == 0.0


def test_widest_quantized_linear_picks_the_biggest():
    model = nn.Sequential(_qlinear(512, K), _qlinear(2048, K), _qlinear(256, K))
    mx.eval(model.parameters())
    found = widest_quantized_linear(model)
    assert found is not None
    assert found["weight"].shape[0] == 2048


def test_crossover_never_returns_a_threshold_that_is_not_identical():
    """The invariant the default-on behaviour rests on: whatever threshold calibration
    hands back, the paths agree bit-for-bit AT that threshold. Here the only shape is a
    small output grid (N=1024), which differs at M=256 — so calibration must either
    refuse outright or return a wider threshold where it does agree."""
    base = nn.Linear(K, 1024, bias=False)
    base.set_dtype(mx.bfloat16)
    model = nn.Sequential(nn.QuantizedLinear.from_linear(base, group_size=64, bits=8))
    mx.eval(model.parameters())
    assert verified_shapes([model], SAFE_MIN_ROWS, widths=()) == []   # differs at 256
    got = measure_crossover(model)
    # either it refuses outright, or the shapes it hands back really are identical there
    assert got is None or verified_shapes([model], got[0], widths=()) == got[1]


def test_verify_identity_accepts_a_shape_where_they_agree():
    base = nn.Linear(K, N, bias=False)
    base.set_dtype(mx.bfloat16)
    model = nn.Sequential(nn.QuantizedLinear.from_linear(base, group_size=64, bits=8))
    mx.eval(model.parameters())
    assert len(verified_shapes([model], SAFE_MIN_ROWS, widths=())) == 1


def test_floors_are_enforced_against_the_flag():
    """A caller asking below the identity boundary must get the stock kernel there,
    not a silent numerics change."""
    lin = _qlinear()
    x = mx.random.normal((64, K)).astype(mx.bfloat16)   # 64 < SAFE_MIN_ROWS
    mx.eval(x, lin.parameters())
    ref = lin(x)
    with wide_matmul(8):                               # asks for 8; clamped to SAFE_MIN_ROWS
        got = lin(x)
    mx.eval(ref, got)
    assert float(mx.abs(got - ref).max()) == 0.0       # identical because it fell through


def test_non_affine_and_biasless_shapes_are_not_eligible():
    """Only affine quantization is covered by the identity sweep; other modes are left
    alone rather than assumed equivalent, so they never reach the wide path."""
    lin = _qlinear()
    mx.eval(lin.parameters())
    assert _eligible(lin) is True
    lin.mode = "mxfp4"
    assert _eligible(lin) is False
    assert list(quantized_linears(nn.Sequential(lin))) == []


def test_unquantized_model_has_no_crossover():
    model = nn.Sequential(nn.Linear(64, 64))
    mx.eval(model.parameters())
    assert widest_quantized_linear(model) is None
    assert measure_crossover(model) is None      # nothing to dequantize -> path stays off
