"""Wide-M matmul for prefill: dequantize the weight once, then a plain GEMM.

``mx.quantized_matmul`` dequantizes each weight tile *inside* its K-loop, once per output
row-block. At decode widths that costs nothing — the kernel is bandwidth-bound on the
weight read and the dequant hides under it. At **prefill** widths the same tile is
dequantized M/BM times and the redundancy becomes the bill. Doing the dequant once
(``x @ mx.dequantize(w).T``) pays one extra fp16 round-trip of the weight instead, which
a wide GEMM amortizes.

Measured (M4 Pro, mlx 0.32, warm/interleaved/median-of-3, 8-bit and 4-bit alike):

    M          256    512   1024   2048   4096
    speedup   0.92x  1.00x  1.06x  1.10x  1.13x
    max|d|      0      0      0      0      0

and end-to-end inside a real model prefill: Qwen3-8B-8bit 1.086x, Qwen3-4B-4bit 1.071x,
gemma-4-12B-8bit (mlx-vlm route) 1.085x — every one **bit-identical** to stock
(``max|Δlogit| = 0``), for +0.2-1.3 GB peak (the dequantized weight is a transient that
mlx frees during eval; they do not all stay alive across a lazy chunk graph).

It LOSES below the crossover (~M=512 here) because the one-shot dequant writes and
re-reads K*N fp16 bytes no matter how few rows there are. That crossover is a property of
(chip x mlx version x shape) — the class of constant this project measures rather than
hardcodes — so :func:`measure_crossover` derives it and the caller caches it to disk.

**Bit-identity is shape-dependent, so it is VERIFIED, not assumed.** Swept over
{fp32, fp16, bf16} x {4, 8} bits x K in {1024, 2048, 4096} x N in {1024, 4096, 12288} x
M in {64, 256, 1024, 2048}, the paths agree exactly everywhere the *output tile grid* is
big enough, and disagree by ~1 ulp where it is not: every half-precision (N=1024, M=256)
cell differs, while every N>=4096 or M>=1024 cell is exact. It is the output grid, not K —
K makes no difference at fixed (N, M). Below M=256 nothing matches, because qmm has not
reached its GEMM tiling yet. So the boundary tracks kernel tile selection, which moves
with mlx versions, dtypes and quantization modes; nothing about it is contractual.

Rather than hardcode a floor that would rot, :func:`measure_crossover` **checks every
distinct quantized weight shape in the models it is given** and returns a threshold only
if all of them reproduce ``quantized_matmul`` bit-for-bit at the widths that would be
used. Anything else — an unusual dtype, a narrow model, a future mlx that retiles qmm —
simply leaves the path off, at stock speed and stock numerics. :data:`SAFE_MIN_ROWS` is
the one hard floor kept, because below the GEMM path the idea loses on speed anyway.

That check is also what makes this safe to enable by default: it cannot silently trade
bits for throughput, and ``tests/test_wide_gemm.py`` asserts the checker actually rejects
a shape where the paths differ.
"""

from __future__ import annotations

import contextlib
import time

import mlx.core as mx
import mlx.nn as nn

_orig_call = nn.QuantizedLinear.__call__
_min_rows = 0                       # 0 = patch inactive
_shapes: frozenset | None = None    # None = every eligible shape; else an allowlist

SAFE_MIN_ROWS = 256   # qmm has not reached GEMM tiling below this: the idea loses on
# speed AND stops being bit-identical there, so no flag may go under it


def _rows(x: mx.array) -> int:
    """Rows the matmul sees: everything but the contracting dimension."""
    return x.size // x.shape[-1] if x.shape[-1] else 0


def _eligible(mod) -> bool:
    """Shapes the wide path may touch at all. ``affine`` is what every checkpoint in this
    project uses; other modes (mxfp8/nvfp4) are left alone rather than assumed."""
    return (getattr(mod, "mode", "affine") == "affine"
            and mod.get("biases") is not None)


def _wide_call(self, x):
    if _rows(x) < _min_rows or not _eligible(self):
        return _orig_call(self, x)
    if _shapes is not None and shape_key(self) not in _shapes:
        return _orig_call(self, x)      # this shape did not verify bit-identical
    w = mx.dequantize(self["weight"], self["scales"], self["biases"],
                      group_size=self.group_size, bits=self.bits)
    y = x @ w.T
    if "bias" in self:
        y = y + self["bias"]
    return y


@contextlib.contextmanager
def wide_matmul(min_rows: int | None, shapes=None):
    """Route ``nn.QuantizedLinear`` through dequantize+GEMM for forwards at least
    ``min_rows`` wide. ``None``/0 disables it — the context manager is then a no-op, so
    call sites stay unconditional.

    Patching the class (rather than the model) is deliberate: it covers the mlx-lm and
    mlx-vlm target routes and the drafter's context projections with no family code. It
    is safe because all MLX work in this project runs on one thread (see the one-thread
    rule in NOTES); do not use this from a second MLX thread.

    ``min_rows`` is raised to :data:`SAFE_MIN_ROWS` if a caller asks for less — the
    identity boundary is not negotiable by flag. ``shapes`` is the allowlist of
    :func:`shape_key` values that verified bit-identical; ``None`` allows every eligible
    shape (only correct when the caller has verified them some other way)."""
    global _min_rows, _shapes
    if not min_rows or nn.QuantizedLinear.__call__ is _wide_call:
        yield                       # disabled, or already active (nested call)
        return
    _min_rows = max(int(min_rows), SAFE_MIN_ROWS)
    _shapes = None if shapes is None else frozenset(shapes)
    nn.QuantizedLinear.__call__ = _wide_call
    try:
        yield
    finally:
        nn.QuantizedLinear.__call__ = _orig_call
        _min_rows = 0
        _shapes = None


def active() -> bool:
    return nn.QuantizedLinear.__call__ is _wide_call


# ------------------------------------------------------------------ calibration


def _in_features(mod) -> int:
    """Affine quantization packs in_features into uint32 words."""
    return int(mod["weight"].shape[1]) * 32 // int(mod.bits)


def quantized_linears(*models):
    """Every eligible ``QuantizedLinear`` in the given models (target, drafter, …). The
    drafter matters too: its context projections run at prefill width."""
    for model in models:
        if model is None:
            continue
        inner = getattr(model, "model", model)
        for _, mod in inner.named_modules():
            if isinstance(mod, nn.QuantizedLinear) and _eligible(mod):
                yield mod


def widest_quantized_linear(*models) -> nn.QuantizedLinear | None:
    """The largest eligible ``QuantizedLinear`` — a stand-in for the MLP matmuls that
    dominate prefill. None for an unquantized (bf16) model: no dequant to skip."""
    best, best_size = None, 0
    for mod in quantized_linears(*models):
        size = mod["weight"].size
        if size > best_size:
            best, best_size = mod, size
    return best


def shape_key(mod) -> str:
    """Identity of a weight for verification purposes: the two paths agree or disagree per
    (in, out, bits, dtype) — see the module docstring."""
    return (f"{_in_features(mod)}x{int(mod['weight'].shape[0])}"
            f"x{int(mod.bits)}x{mod['scales'].dtype}")


def _matches(mod, M: int) -> bool:
    """Do the two paths agree bit-for-bit for this weight at ``M`` rows?"""
    K = _in_features(mod)
    for _ in range(2):                       # two draws: one lucky match proves little
        x = mx.random.normal((M, K)).astype(mod["scales"].dtype)
        mx.eval(x)
        a = mx.quantized_matmul(x, mod["weight"], scales=mod["scales"],
                                biases=mod["biases"], transpose=True,
                                group_size=mod.group_size, bits=mod.bits)
        c = x @ mx.dequantize(mod["weight"], mod["scales"], mod["biases"],
                              group_size=mod.group_size, bits=mod.bits).T
        mx.eval(a, c)
        same = float(mx.abs(a - c).max()) == 0.0
        del x, a, c
        mx.clear_cache()
        if not same:
            return False
    return True


def verified_shapes(models, min_rows: int,
                    widths: tuple[int, ...] = (512, 1024, 2048)) -> list[str]:
    """The shape keys across ``models`` that reproduce ``quantized_matmul`` exactly at
    ``min_rows`` and at each of ``widths``. This per-shape allowlist is the gate that lets
    the path be on by default; a shape that does not verify keeps the stock kernel.

    Per-shape rather than all-or-nothing because that is how the divergence actually
    distributes: on Qwen3-8B only k/v_proj (N=1024) disagree, and disqualifying the whole
    model for them would throw away the win on the MLP matmuls that dominate prefill.

    Sampled, not proved: a prefill's last chunk can be any width >= min_rows. The sample
    is weighted to the small end because that is where the paths were observed to diverge
    (a small output tile grid), and ``min_rows`` — the narrowest width the path can ever
    see — is always checked."""
    ok: list[str] = []
    seen: set = set()
    for mod in quantized_linears(*models):
        key = shape_key(mod)
        if key in seen:
            continue
        seen.add(key)
        if all(_matches(mod, M) for M in (min_rows, *widths)):
            ok.append(key)
    return ok


def _bench(fn, iters: int = 5, warmup: int = 2) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    ts = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def measure_crossover(*models, widths: tuple[int, ...] = (256, 512, 1024, 2048),
                      margin: float = 0.02) -> tuple[int, list[str]] | None:
    """``(min_rows, verified shape keys)`` for this machine+models, or ``None`` when the
    path should stay off (nothing quantized, never faster, or no shape verifies).

    ``min_rows`` is the smallest measured width from which dequant+GEMM wins by ``margin``
    and keeps winning at every wider point. The shape list is then the subset of the
    models' weights that reproduce qmm bit-for-bit there — per shape, because the
    divergence is per shape: on Qwen3-8B only k/v_proj (N=1024) differ, and disqualifying
    the model for them would throw away the win on the MLP matmuls that dominate prefill.

    Costs ~2-3 s once per (machine, mlx version, model); the caller caches it."""
    lin = widest_quantized_linear(*models)
    if lin is None:
        return None
    w, s, b = lin["weight"], lin["scales"], lin["biases"]
    K, dtype = _in_features(lin), s.dtype
    wins: dict[int, bool] = {}
    for M in widths:
        if M < SAFE_MIN_ROWS:
            wins[M] = False
            continue
        x = mx.random.normal((M, K)).astype(dtype)
        mx.eval(x)
        t_q = _bench(lambda a=x: mx.quantized_matmul(
            a, w, scales=s, biases=b, transpose=True,
            group_size=lin.group_size, bits=lin.bits))
        t_w = _bench(lambda a=x: a @ mx.dequantize(
            w, s, b, group_size=lin.group_size, bits=lin.bits).T)
        wins[M] = t_q > t_w * (1.0 + margin)
        del x
        mx.clear_cache()
    for i, M in enumerate(widths):
        if not all(wins[m] for m in widths[i:]):
            continue
        keys = verified_shapes(models, int(M), widths=widths[i + 1:])
        return (int(M), keys) if keys else None
    return None
