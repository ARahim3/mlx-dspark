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
    with pytest.raises(RuntimeError), wide_matmul(64):
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


# ------------------------------------------------------------------ CPU co-prefill split

from mlx_dspark.wide_gemm import (  # noqa: E402
    cpu_rows,
    frac_for,
    measure_cpu_split,
)


def _split_cfg(min_rows=64, **fracs):
    return {"min_rows": min_rows, "fracs": {int(k): v for k, v in fracs.items()}}


def test_frac_for_picks_widest_calibrated_width_not_above_rows():
    cfg = _split_cfg(512, **{"512": 0.15, "1024": 0.2, "2048": 0.3})
    assert frac_for(cfg, 256) == 0.0           # below min_rows: no split
    assert frac_for(cfg, 512) == 0.15
    assert frac_for(cfg, 1500) == 0.2
    assert frac_for(cfg, 4096) == 0.3
    assert frac_for(None, 4096) == 0.0
    assert frac_for({"min_rows": 64, "fracs": {}}, 4096) == 0.0


def test_cpu_rows_is_tile_rounded_and_never_everything():
    assert cpu_rows(2048, 0.3) == 640          # 614.4 -> nearest 64-multiple
    assert cpu_rows(2048, 0.3) % 64 == 0
    assert cpu_rows(64, 0.3) == 0              # would round to 0 tiles
    assert cpu_rows(128, 0.9) == 0             # would be everything -> no split


def test_split_matches_plain_path_within_bf16_and_restores():
    """The CPU rows are a different accumulation order (fp-tie class), so agreement is
    'within bf16 rounding', not bit-identity — that is exactly why the library leaves
    this off and the CLI turns it on, like the last-row head."""
    mx.random.seed(3)
    lin = _qlinear(bits=4)
    x = mx.random.normal((ROWS, K)).astype(mx.bfloat16)
    ref = _orig_call(lin, x)
    with wide_matmul(0, None, cpu_split=_split_cfg(64, **{"64": 0.25})):
        assert active()
        y = lin(x)
    assert not active() and nn.QuantizedLinear.__call__ is _orig_call
    mx.eval(ref, y)
    assert y.shape == ref.shape and y.dtype == ref.dtype
    d = mx.abs(y.astype(mx.float32) - ref.astype(mx.float32))
    scale = float(mx.abs(ref.astype(mx.float32)).max())
    assert float(d.max()) <= 0.02 * scale      # a few bf16 ulps of the output scale
    # the GPU rows (the first ones) are bit-identical to the dequant+GEMM path
    n_cpu = cpu_rows(ROWS, 0.25)
    wide = x @ mx.dequantize(lin["weight"], lin["scales"], lin["biases"],
                             group_size=64, bits=4).T
    mx.eval(wide)
    assert float(mx.abs(y[:-n_cpu].astype(mx.float32)
                        - wide[:-n_cpu].astype(mx.float32)).max()) == 0.0


def test_split_respects_min_rows_and_leading_dims_and_bias():
    lin = _qlinear(bits=8, bias=True)
    narrow = mx.random.normal((1, 32, K)).astype(mx.bfloat16)
    wide = mx.random.normal((2, ROWS // 2, K)).astype(mx.bfloat16)   # 256 rows over 2 dims
    with wide_matmul(0, None, cpu_split=_split_cfg(128, **{"128": 0.25})):
        y_narrow = lin(narrow)
        y_wide = lin(wide)
    mx.eval(y_narrow, y_wide)
    assert float(mx.abs(y_narrow - _orig_call(lin, narrow)).max()) == 0.0   # untouched
    assert y_wide.shape == (2, ROWS // 2, N)
    ref = _orig_call(lin, wide)
    assert float(mx.abs(y_wide.astype(mx.float32) - ref.astype(mx.float32)).max()) \
        <= 0.02 * float(mx.abs(ref.astype(mx.float32)).max())


def test_split_and_wide_path_coexist():
    """Wide path for rows in [min_rows, split.min_rows), split above."""
    lin = _qlinear(bits=8)
    mid = mx.random.normal((ROWS, K)).astype(mx.bfloat16)
    with wide_matmul(ROWS, None, cpu_split=_split_cfg(ROWS * 2, **{str(ROWS * 2): 0.25})):
        y_mid = lin(mid)
    wide = mid @ mx.dequantize(lin["weight"], lin["scales"], lin["biases"],
                               group_size=64, bits=8).T
    mx.eval(y_mid, wide)
    assert float(mx.abs(y_mid.astype(mx.float32) - wide.astype(mx.float32)).max()) == 0.0


def test_split_config_floor_and_empty_fracs_disable():
    lin = _qlinear(bits=8)
    with wide_matmul(0, None, cpu_split={"min_rows": 64, "fracs": {}}):
        assert not active()                    # nothing to split with -> no-op
    x = mx.random.normal((128, K)).astype(mx.bfloat16)
    with wide_matmul(0, None, cpu_split=_split_cfg(1, **{"1": 0.5})):
        y = lin(x)                             # floor raised to SAFE_MIN_ROWS (256)
    mx.eval(y)
    assert float(mx.abs(y - _orig_call(lin, x)).max()) == 0.0


def test_measure_cpu_split_returns_sane_structure_or_none():
    lin = _qlinear(out_features=1024, in_features=1024, bits=4)
    got = measure_cpu_split(lin, widths=(256, 512), fracs=(0.2, 0.3), iters=1)
    if got is not None:
        assert got["min_rows"] in (256, 512)
        assert all(0 < f < 1 for f in got["fracs"].values())
        assert all(int(w) >= got["min_rows"] for w in got["fracs"])
        assert all(v > 1.0 for v in got["speedup"].values())
    class _Bare(nn.Module):
        pass
    assert measure_cpu_split(_Bare()) is None
