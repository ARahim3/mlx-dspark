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

**CPU co-prefill (the row split).** Prefill runs the GPU at ~85% of its GEMM peak, so
the only way to make it faster is a second engine. The Apple Neural Engine measured out
(see NOTES "CPU co-prefill"): its throughput tracks the *weight format* (fp16 2.3 TF,
int8 4.5, int4/LUT 8.4 on an M4 Pro) and none of the fast formats can hold our
K-grouped affine weights exactly, while the exact fp16 form needs a 34 GB copy of a 27B
MLP. The CPU can: MLX's own CPU stream runs a bf16 GEMM at ~3.2 TF on the M4 Pro's
matrix units, **it overlaps with the GPU** (unified memory, no copy, no second Python
thread, no dependency), and it consumes the SAME dequantized transient the wide path
already makes. So above a measured width every eligible ``QuantizedLinear`` hands the
LAST ``frac`` of its rows to the CPU stream and the GPU takes the rest; the two run
concurrently and ``concatenate`` joins them. Measured on Qwen3.8-27B-4bit, 2048-token
chunk: 130 -> 184 tok/s (**1.41x** over stock, 1.32x over the wide path alone), +0.4 GB
peak, 64-token greedy continuation token-identical.

Not bit-identical: the CPU rows are a different accumulation order (fp-tie class —
max|Δlogit| 2.0 on the 27B's final row vs 1.55 for chunked prefill and 2.0 for the
unverified wide path, i.e. the same band). Hence the CLI/server turn it on (like the
last-row head slice and the small-M kernel) and the library API leaves it off. The
fraction and the row floor are (chip x mlx x model) constants: ``measure_cpu_split``
finds the best fraction per width — it falls off a cliff past the balance point (M4
Pro: 0.3 at 2048 rows, 1.41x; 0.45 is 1.08x) — and the caller caches it.
"""

from __future__ import annotations

import contextlib
import time

import mlx.core as mx
import mlx.nn as nn

_orig_call = nn.QuantizedLinear.__call__
_min_rows = 0                       # 0 = wide (dequant+GEMM) path inactive
_shapes: frozenset | None = None    # None = every eligible shape; else an allowlist
_cpu_split: dict | None = None      # {"min_rows": N, "fracs": {width: frac}} or None = off
_ROUND = 64                         # CPU row slice granularity (GEMM tile friendly)

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


def frac_for(cfg: dict | None, rows: int) -> float:
    """The CPU row fraction to use at ``rows`` (0.0 = no split): the entry measured at
    the widest calibrated width not above ``rows`` — the balance point moves with width
    because the CPU GEMM's efficiency does (M4 Pro: 0.2 at 1024, 0.3 at 2048)."""
    if not cfg or rows < int(cfg.get("min_rows") or 0):
        return 0.0
    best_w, best_f = 0, 0.0
    for w, f in cfg.get("fracs", {}).items():
        w = int(w)
        if best_w < w <= rows:
            best_w, best_f = w, float(f)
    return best_f


def cpu_rows(rows: int, frac: float) -> int:
    """Rows handed to the CPU stream: ``frac`` of ``rows`` rounded to :data:`_ROUND`,
    never all of them and never fewer than one tile."""
    n = round(rows * frac / _ROUND) * _ROUND
    return 0 if n >= rows else n


def _split_call(self, x, rows: int, frac: float):
    """GPU takes rows[:a], the CPU stream rows[a:], both from ONE dequantized transient.
    The CPU work is scheduled by mlx's own scheduler on its CPU stream — no second Python
    thread, so the one-MLX-thread rule holds — and unified memory means no copy."""
    n_cpu = cpu_rows(rows, frac)
    if n_cpu == 0:
        return None
    w = mx.dequantize(self["weight"], self["scales"], self["biases"],
                      group_size=self.group_size, bits=self.bits)
    lead = x.shape[:-1]
    x2 = x.reshape(rows, x.shape[-1])
    a = rows - n_cpu
    y_gpu = x2[:a] @ w.T
    with mx.stream(mx.cpu):
        y_cpu = x2[a:] @ w.T
    y = mx.concatenate([y_gpu, y_cpu], axis=0)
    if "bias" in self:
        y = y + self["bias"]
    return y.reshape(*lead, y.shape[-1])


def _wide_call(self, x):
    if not _eligible(self):
        return _orig_call(self, x)
    rows = _rows(x)
    frac = frac_for(_cpu_split, rows)
    if frac:
        y = _split_call(self, x, rows, frac)
        if y is not None:
            return y
    if not _min_rows or rows < _min_rows:
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
def wide_matmul(min_rows: int | None, shapes=None, cpu_split: dict | None = None):
    """Route ``nn.QuantizedLinear`` through dequantize+GEMM for forwards at least
    ``min_rows`` wide, and — when ``cpu_split`` is given (see :func:`measure_cpu_split`)
    — hand a measured fraction of the rows to the CPU stream from its ``min_rows`` up.
    ``None``/0 for both disables it — the context manager is then a no-op, so call sites
    stay unconditional.

    Patching the class (rather than the model) is deliberate: it covers the mlx-lm and
    mlx-vlm target routes and the drafter's context projections with no family code. It
    is safe because all MLX work in this project runs on one thread (see the one-thread
    rule in NOTES); do not use this from a second MLX thread.

    ``min_rows`` is raised to :data:`SAFE_MIN_ROWS` if a caller asks for less — the
    identity boundary is not negotiable by flag. ``shapes`` is the allowlist of
    :func:`shape_key` values that verified bit-identical; ``None`` allows every eligible
    shape (only correct when the caller has verified them some other way)."""
    global _min_rows, _shapes, _cpu_split
    split = cpu_split if (cpu_split and cpu_split.get("fracs")) else None
    if (not min_rows and split is None) or nn.QuantizedLinear.__call__ is _wide_call:
        yield                       # disabled, or already active (nested call)
        return
    _min_rows = max(int(min_rows), SAFE_MIN_ROWS) if min_rows else 0
    _shapes = None if shapes is None else frozenset(shapes)
    if split is not None:
        split = {"min_rows": max(int(split.get("min_rows") or SAFE_MIN_ROWS), SAFE_MIN_ROWS),
                 "fracs": {int(k): float(v) for k, v in split["fracs"].items()}}
    _cpu_split = split
    prev = nn.QuantizedLinear.__call__   # restore to ENTRY value, not the import-time
    # original — this context nests inside small_m_matmul (prefill inside a decode
    # loop), and restoring _orig_call would silently strip that patch mid-generation.
    nn.QuantizedLinear.__call__ = _wide_call
    try:
        yield
    finally:
        nn.QuantizedLinear.__call__ = prev
        _min_rows = 0
        _shapes = None
        _cpu_split = None


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


def pick_fraction(times: dict, *, tol: float = 0.03) -> tuple[float, float]:
    """``(frac, seconds)`` — the SMALLEST fraction whose time is within ``tol`` of the best.

    The split's cost curve is asymmetric: below the balance point a smaller CPU share loses
    a little, past it the CPU becomes the critical path and the whole prefill loses a lot
    (M4 Pro, 27B, 2048 rows: 0.30 = 1.41x, 0.45 = 1.08x). Microbench noise of a few percent
    is enough to make the raw argmin land one step past balance — measured 2026-08-25: two
    calibrations of the same machine picked 0.30 and 0.35 at width 2048, and 0.35 cost 7.5%
    end-to-end (173 vs 188 tok/s). So ties within ``tol`` resolve toward the safe side."""
    if not times:
        return 0.0, float("inf")
    best_t = min(times.values())
    for f in sorted(times):
        if times[f] <= best_t * (1.0 + tol):
            return f, times[f]
    return min(times, key=times.get), best_t


def measure_cpu_split(*models, widths: tuple[int, ...] = (512, 1024, 2048),
                      fracs: tuple[float, ...] = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4),
                      margin: float = 0.05, iters: int = 5) -> dict | None:
    """``{"min_rows": N, "fracs": {width: frac}}`` for this machine+models, or ``None``
    when the CPU split should stay off (nothing quantized, or it never wins by ``margin``
    at the widest width).

    Per width, the split is raced against the better of the two GPU-only paths (stock
    qmm and dequant+GEMM) on the widest quantized weight, and the best fraction kept;
    ``min_rows`` is the smallest width from which the split wins and keeps winning at
    every wider width. The balance point is a real optimum, not a monotone knob — past
    it the CPU becomes the critical path and the whole thing loses (M4 Pro, 27B, 2048
    rows: 0.3 -> 1.41x, 0.45 -> 1.08x) — which is why it is measured and cached rather
    than guessed from a core count. Costs ~10-20 s once for a 27B; the caller caches it."""
    lin = widest_quantized_linear(*models)
    if lin is None:
        return None
    w, s, b = lin["weight"], lin["scales"], lin["biases"]
    K, dtype = _in_features(lin), s.dtype
    gs, bits = lin.group_size, lin.bits
    best: dict[int, tuple[float, float]] = {}      # width -> (frac, speedup)
    for M in widths:
        if M < SAFE_MIN_ROWS:
            continue
        x = mx.random.normal((M, K)).astype(dtype)
        mx.eval(x)
        t_q = _bench(lambda a=x: mx.quantized_matmul(
            a, w, scales=s, biases=b, transpose=True, group_size=gs, bits=bits), iters)
        t_w = _bench(lambda a=x: a @ mx.dequantize(w, s, b, group_size=gs, bits=bits).T,
                     iters)
        base = min(t_q, t_w)

        def split(a, frac):
            wd = mx.dequantize(w, s, b, group_size=gs, bits=bits)
            n = cpu_rows(a.shape[0], frac)
            y0 = a[:-n] @ wd.T
            with mx.stream(mx.cpu):
                y1 = a[-n:] @ wd.T
            return mx.concatenate([y0, y1], axis=0)

        times = {}
        for f in fracs:
            if cpu_rows(M, f) == 0:
                continue
            times[f] = _bench(lambda a=x, f=f: split(a, f), iters)
        best_f, best_t = pick_fraction(times)          # ties resolve toward the smaller share
        if best_f and base > best_t * (1.0 + margin):
            best[M] = (best_f, base / best_t)
        del x
        mx.clear_cache()
    if not best:
        return None
    for i, M in enumerate(widths):
        if all(m in best for m in widths[i:] if m >= SAFE_MIN_ROWS):
            keep = {m: best[m][0] for m in widths[i:] if m in best}
            return {"min_rows": int(M), "fracs": keep,
                    "speedup": {m: round(best[m][1], 3) for m in keep}}
    return None
