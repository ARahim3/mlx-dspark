"""Small-M quantized matmul for the verify window: dispatch, per-shape gate, scoped patch.

``mx.quantized_matmul`` grows ~linearly in M for M in 2..~12 on Apple GPUs — the weight read
is re-paid per row until the GEMM tiling takes over (upstream ml-explore/mlx#4265; our own
#3852 is the 2-bit face of the same thing). Speculative verify lives exactly there: verify
width = cap + 1, and a drafter's block backbone runs at block width (7-16) every round.

This module routes those forwards through :mod:`.skinny_qmm` — mlx-dspark's own matrix-unit
kernel (weights as the MMA operand, loaded once per group and shared by up to 16 rows) —
over a measured window per quantization width:

- **4-bit: M ∈ [5, 16]** (M 2-3 lose to qmv/qmm, 4 ties). Whole Qwen3.8-27B-4bit verify at
  ctx 512: widths 5-8 113 -> ~96 ms, widths 9-16 214-276 -> ~165 ms.
- **8-bit: M ∈ [6, 16]** (8-bit qmm is flat to M=5, then cliffs at 6: 0.41 -> 0.66 ms on the
  Qwen3.8-27B-8bit MLP shapes; the kernel holds 0.44 through 8 and 0.56 at 12-16).

History: v0.12.0 shipped avlp12's ``qmm_mma4`` (MIT, dequant staged in threadgroup memory)
for M 6-8 — the first proof that the 4-bit curve's "rise from width 3" was a kernel dead zone,
not the hardware (cap 2 -> 7 on the 4-bit 27B). 2026-09-27 briefly vendored TensorFold's
``simd_qmm`` for 4-bit 5-16; both are replaced by ``skinny_qmm`` (1-3% faster than the
former at 4-bit, ties the latter at 8-bit 6-8, and covers 9-16 for both widths). NOTES
"The small-M MMA verify kernel" and "Engine pass 2026-09-27".

Numerics: fp32 accumulation in a different order than qmm — outputs differ by 1-2 bf16 ulps.
Same class as the batched path: per-token greedy-correct under the verify loop (the target
still verifies every token), NOT bit-identical to the stock kernel. :func:`measure_shapes`
checks both the numerics and the speed per shape at runtime and returns only shapes where
the kernel actually wins on this (chip x mlx version) — verified, never assumed.

Measurement notes (cost real time upstream, don't relearn):
- DEPENDENT chains only. Queue-batched/independent-call microbenches let kernel launches
  overlap and hide the longer critical path — avlp12 measured a kernel that looked 1.13x
  that way and made the real model 26% slower. Decode is a serial chain.
- Rotate weights across calls (katlun's lesson): a single weight stays cache-resident and
  hides the read cost the kernel exists to amortize. :func:`measure_shapes` rotates across the
  model's OWN same-shape layers, so it allocates nothing.
"""

from __future__ import annotations

import contextlib
import time

import mlx.core as mx
import mlx.nn as nn

from . import skinny_qmm as _sk

M4_MIN, M4_MAX = _sk.WINDOW[4]      # 4-bit rows served
M8_MIN, M8_MAX = _sk.WINDOW[8]      # 8-bit rows served
M_MIN, M_MAX = M8_MIN, M8_MAX       # (legacy names: the 8-bit window)

_orig_call = nn.QuantizedLinear.__call__
_active_ids: frozenset | None = None   # None = patch inactive


def _in_features(mod) -> int:
    """Affine quantization packs in_features into uint32 words (32/bits values each)."""
    return int(mod["weight"].shape[1]) * 32 // int(mod.bits)


def eligible(mod) -> bool:
    """Can this ``QuantizedLinear``'s shape/format run on the kernel at all?"""
    return isinstance(mod, nn.QuantizedLinear) and _sk.fits(mod)


def shape_key(mod) -> str:
    return f"{_in_features(mod)}x{int(mod['weight'].shape[0])}b{int(mod.bits)}"


def _mma_call(self, x):
    if _active_ids is None or id(self) not in _active_ids or x.dtype != mx.bfloat16:
        return _orig_call(self, x)
    M = 1
    for d in x.shape[:-1]:
        M *= d
    lo, hi = _sk.WINDOW[self.bits]
    if lo <= M <= hi:
        return _sk.call(self, x, M)
    return _orig_call(self, x)


@contextlib.contextmanager
def small_m_matmul(ids: frozenset | None):
    """Route the given ``QuantizedLinear`` instances (by ``id``) through the kernel for
    forwards inside their width's window. ``None``/empty disables — the context is then a
    no-op, so call sites stay unconditional.

    ``ids`` is a *pre-verified allowlist* built by :func:`calibrate.apply_small_m` from
    shapes that :func:`measure_shapes` proved faster AND numerically sane on this machine.
    An id-set (not per-call shape math) keeps the dispatch overhead off the M=1 decode path.
    Class-level patch, one-MLX-thread rule applies (see NOTES); restores to the value found
    at entry so it nests with ``wide_matmul``.
    """
    global _active_ids
    if not ids or nn.QuantizedLinear.__call__ is _mma_call:
        yield
        return
    prev = nn.QuantizedLinear.__call__
    _active_ids = frozenset(ids)
    nn.QuantizedLinear.__call__ = _mma_call
    try:
        yield
    finally:
        nn.QuantizedLinear.__call__ = prev
        _active_ids = None


def active() -> bool:
    return nn.QuantizedLinear.__call__ is _mma_call


# ------------------------------------------------------------------ calibration

_CHAIN = 12   # dependent steps per timed eval
_EVALS = 4    # first is warmup, median of the rest
_MIN_GAIN = 1.15   # a shape must beat quantized_matmul by this in the window to be worth
# the fp-tie churn; below it the flat-in-M kernel is within noise of the rising qmm curve
_EDGE_MIN = 0.97   # at the 4-bit window's bottom edge (M=5) the kernel must not LOSE
_REL_TOL = 0.02    # max|kernel - qmm| <= _REL_TOL * max|qmm| (measured 1-2 bf16 ulps,
# ~0.007 relative; 0.02 leaves headroom without admitting a broken shape)


def _time_chain(step, x0) -> float:
    times = []
    for _ in range(_EVALS):
        x = x0
        t0 = time.perf_counter()
        for t in range(_CHAIN):
            y = step(x, t)
            x = x0 + mx.mean(y).astype(x0.dtype) * 1e-20  # real dependency, ~no drift
        mx.eval(y, x)
        times.append((time.perf_counter() - t0) / _CHAIN)
    rest = sorted(times[1:])
    return rest[len(rest) // 2]


def _kernel_rows(mod, x: mx.array) -> mx.array:
    return _sk.qmm(x, mod["weight"], mod["scales"], mod["biases"], int(mod.bits))


def measure_shapes(*models, verbose: bool = False) -> list[str]:
    """Which eligible weight shapes actually WIN on the kernel, on this machine, now.

    For every distinct eligible shape in the given models: check numerics against
    ``quantized_matmul`` across the window, then race dependent chains — rotating across the
    model's own same-shape layers so nothing stays cache-resident and nothing is allocated. A
    shape must win by ``_MIN_GAIN`` at M=8 and at M=12 (the two tile regimes: one and two
    8-row tiles), and a 4-bit shape must not lose at its bottom edge M=5. Returns the shape
    keys that pass; anything else stays on the stock path at stock speed and stock numerics.
    """
    groups: dict[str, list] = {}
    for model in models:
        if model is None:
            continue
        for _, mod in model.named_modules():
            if eligible(mod):
                groups.setdefault(shape_key(mod), []).append(mod)

    won: list[str] = []
    for key, mods in groups.items():
        mod = mods[0]
        K = _in_features(mod)
        lo, hi = _sk.WINDOW[int(mod.bits)]

        ok = True
        for M in sorted({lo, 8, 12, hi}):
            x = (mx.random.normal((M, K)) * 0.1).astype(mx.bfloat16)
            ref = _orig_call(mod, x).astype(mx.float32)
            got = _kernel_rows(mod, x).astype(mx.float32)
            diff = mx.max(mx.abs(ref - got))
            scale = mx.max(mx.abs(ref))
            mx.eval(diff, scale)
            if diff.item() > _REL_TOL * max(scale.item(), 1.0):
                ok = False
                break
        if not ok:
            if verbose:
                print(f"  small-M qmm: {key} rejected (numerics)", flush=True)
            continue

        def race(M, _m=mods, _K=K):
            x = (mx.random.normal((M, _K)) * 0.1).astype(mx.bfloat16)
            mx.eval(x)

            def q_step(xx, t):
                return _orig_call(_m[t % len(_m)], xx)

            def k_step(xx, t):
                return _kernel_rows(_m[t % len(_m)], xx)

            tq = min(_time_chain(q_step, x), _time_chain(q_step, x))
            tk = min(_time_chain(k_step, x), _time_chain(k_step, x))
            return tq / tk

        g8, g12 = race(8), race(12)
        g_lo = race(lo) if lo < 6 else None
        on = g8 >= _MIN_GAIN and g12 >= _MIN_GAIN and (g_lo is None or g_lo >= _EDGE_MIN)
        if on:
            won.append(key)
        if verbose:
            edge = f", M={lo} {g_lo:.2f}x" if g_lo is not None else ""
            print(f"  small-M qmm: {key} M=8 {g8:.2f}x, M=12 {g12:.2f}x{edge} "
                  f"({'on' if on else 'off'})", flush=True)
    return won


def ids_for_shapes(shapes, *models) -> frozenset:
    """The instance allowlist for :func:`small_m_matmul`, from verified shape keys."""
    keys = set(shapes)
    out = set()
    for model in models:
        if model is None:
            continue
        for _, mod in model.named_modules():
            if eligible(mod) and shape_key(mod) in keys:
                out.add(id(mod))
    return frozenset(out)
