"""Skinny quantized matmul: mlx-dspark's matrix-unit kernel for the verify window (5-16 rows).

``quantized_matmul`` re-pays the weight read per row for M in 2..~12 on Apple GPUs (the dead
zone of ml-explore/mlx#4265) — exactly the rows speculative verify and a drafter's block run.
This kernel computes ``y = x·dequant(W)ᵀ`` transposed on the 8×8 ``simdgroup_matrix`` units,
``C[n][m] += A[n][k]·B[k][m]``, with the WEIGHTS as the A operand: every lane loads its two
packed words of a weight row straight into registers (no threadgroup staging, no barriers in
the K loop, no zero-padded copy of x), and up to two 8-row tiles share one weight read.

Inside a 64-input quantization group, MMA step ``s`` takes field ``s`` of 8 consecutive packed
words (4-bit: one slice of 8 words × 8 nibbles; 8-bit: two slices of 8 words × 4 bytes), so a
lane's A pair is field ``s`` of its two adjacent words and its B pair is input ``s`` of a
contiguous 8/4-input run it loaded once per group. Fields are masked in place
(``word & (mask << bits·s)``) against inputs pre-scaled by ``2^-(bits·s)`` — the products are
exact in fp32 — and scale/bias factor out per group (``acc += scale·P + bias·Σx``). Groups are
interleaved over the simdgroups (split-K: 8 on wide outputs, more on narrow ones — see
:func:`geometry`) and reduced through threadgroup memory at the end.

Credits (ideas, no code): weights-as-the-matrix-operand for skinny batches with in-place
fields against pre-scaled inputs is the layout of TensorFold's ``simd_qmm``
(github.com/ashhart/TensorFold, MIT) — this project vendored that kernel for a day
(2026-09-27) and replaced it with this independent implementation after the review; the
pre-scaled-input / unshifted-mask dot product is a long-standing llama.cpp Metal technique;
the first small-M kernel here was avlp12's ``qmm_mma4`` (dequant staged in threadgroup
memory), which this replaces for both widths.

Measured (M4 Pro, mlx 0.32.2, dependent chains over 8 rotated weight copies, ms):

    4-bit 17408x5120   M5 0.379→0.313  M8 0.580→0.314  M12 0.863→0.531  M16 0.939→0.533
    4-bit 5120x17408   M5 0.391→0.320  M8 0.603→0.319  M12 0.885→0.532  M16 0.943→0.535
    8-bit 17408x5120   M6 0.552→0.440  M8 0.664→0.443  M12 0.988→0.554  M16 0.950→0.556

(stock → this kernel). Versus the vendored kernels it replaces: 1-3% faster than TensorFold's
at every 4-bit width; ties avlp12's at 8-bit 6-8 (+8% on 5120x17408). A load-only probe of the
same access pattern streams at 212-217 GB/s (contiguous streaming: 241-244), so what remains
is ALU/memory overlap, not the addresses; fp16/bf16 fragments are no faster than fp32 here
(7.5 vs 7.0 TFLOP/s MMA), and an exponent-bias dequant (no int→float convert) was slower.

Windows: 4-bit M ∈ [5, 16] (M 2-3 lose to qmv/qmm, 4 ties); 8-bit M ∈ [6, 16] (M 5 ties).
Numerics: fp32 accumulation in a fixed order — fp-tie class vs ``quantized_matmul``
(≤ ~0.005 relative), and row-count invariant (a row's result does not depend on how many
rows ride with it). The dispatch, per-shape race and numerics gate live in
:mod:`.small_m_qmm`.
"""

from __future__ import annotations

import math

import mlx.core as mx

GROUP = 64
RT_MAX = 2      # 8-row tiles per threadgroup (16 rows)
WINDOW = {4: (5, 16), 8: (6, 16)}   # rows served per quantization width

_HEADER = r"""
// the bf16 at position e of a packed word, as fp32 (bf16 -> fp32 is a 16-bit shift)
inline float bf_at(uint w, int e) { return as_type<float>((e & 1) ? (w & 0xFFFF0000u) : (w << 16)); }
"""

_SRC = r"""
  constexpr int G = K / 64;                        // quant groups along K
  constexpr int SLICES = BITS / 4;                 // 8-word slices per group
  constexpr int FIELDS = 32 / BITS;                // fields per word = MMA steps per slice
  constexpr int WPR = K * BITS / 32;               // packed words per weight row
  const int lane = int(thread_index_in_simdgroup);
  const int c = int(simdgroup_index_in_threadgroup);
  const int fm = ((lane >> 4) << 2) + ((lane >> 1) & 3);       // fragment row of this lane
  const int fn = (((lane >> 3) & 1) << 2) + ((lane & 1) << 1);  // fragment column pair
  const int R = X_shape[0];
  const int n0 = int(threadgroup_position_in_grid.x) * (8 * NT);
  const int m0 = int(threadgroup_position_in_grid.y) * (8 * RT);

  int nrow[NT];                                    // this lane's weight rows (clamped)
  for (int t = 0; t < NT; ++t) nrow[t] = min(n0 + 8 * t + fm, N - 1);
  const device T* x0[RT];                          // this lane's two input rows per tile
  const device T* x1[RT];
  for (int r = 0; r < RT; ++r) {
    x0[r] = X + size_t(min(m0 + 8 * r + fn, R - 1)) * K;
    x1[r] = X + size_t(min(m0 + 8 * r + fn + 1, R - 1)) * K;
  }
  float acc[RT][NT][2];
  for (int r = 0; r < RT; ++r)
    for (int t = 0; t < NT; ++t) { acc[r][t][0] = 0.0f; acc[r][t][1] = 0.0f; }

  for (int g = c; g < G; g += S) {                 // quant groups, interleaved over simdgroups
    simdgroup_matrix<float, 8, 8> P[RT][NT];
    float sum0[RT], sum1[RT];
    for (int r = 0; r < RT; ++r) {
      sum0[r] = 0.0f; sum1[r] = 0.0f;
      for (int t = 0; t < NT; ++t) P[r][t] = simdgroup_matrix<float, 8, 8>(0.0f);
    }
    #pragma clang loop unroll(full)
    for (int sl = 0; sl < SLICES; ++sl) {
      uint2 w[NT];
      for (int t = 0; t < NT; ++t)
        w[t] = *(const device uint2*)((const device uint*)Wq + size_t(nrow[t]) * WPR
                                      + 8 * (SLICES * g + sl) + fn);
      const int xo = 64 * g + 32 * sl * (SLICES - 1) + FIELDS * fm;   // this lane's input run
      uint4 xa[RT], xb[RT];
      for (int r = 0; r < RT; ++r) {
        if (BITS == 4) {
          xa[r] = *(const device uint4*)(x0[r] + xo);
          xb[r] = *(const device uint4*)(x1[r] + xo);
        } else {
          xa[r] = uint4(*(const device uint2*)(x0[r] + xo), 0u, 0u);
          xb[r] = uint4(*(const device uint2*)(x1[r] + xo), 0u, 0u);
        }
        for (int e = 0; e < FIELDS; ++e) {
          sum0[r] += bf_at(xa[r][e / 2], e);
          sum1[r] += bf_at(xb[r][e / 2], e);
        }
      }
      #pragma clang loop unroll(full)
      for (int s = 0; s < FIELDS; ++s) {
        const uint mask = ((1u << BITS) - 1u) << (BITS * s);
        const float pre = as_type<float>(uint(127 - BITS * s) << 23);    // 2^-(BITS*s)
        simdgroup_matrix<float, 8, 8> b[RT];
        for (int r = 0; r < RT; ++r) {
          b[r].thread_elements()[0] = bf_at(xa[r][s / 2], s) * pre;
          b[r].thread_elements()[1] = bf_at(xb[r][s / 2], s) * pre;
        }
        for (int t = 0; t < NT; ++t) {
          simdgroup_matrix<float, 8, 8> a;
          a.thread_elements()[0] = float(w[t].x & mask);
          a.thread_elements()[1] = float(w[t].y & mask);
          for (int r = 0; r < RT; ++r) simdgroup_multiply_accumulate(P[r][t], a, b[r], P[r][t]);
        }
      }
    }
    for (int r = 0; r < RT; ++r) {                 // group input sums over the 8 fm lanes
      for (int d = 2; d <= 16; d <<= 1) {
        if (d == 8) continue;
        sum0[r] += simd_shuffle_xor(sum0[r], ushort(d));
        sum1[r] += simd_shuffle_xor(sum1[r], ushort(d));
      }
    }
    for (int t = 0; t < NT; ++t) {
      const float sc = float(SC[size_t(nrow[t]) * G + g]);
      const float bi = float(BI[size_t(nrow[t]) * G + g]);
      for (int r = 0; r < RT; ++r) {
        acc[r][t][0] = fma(bi, sum0[r], fma(sc, P[r][t].thread_elements()[0], acc[r][t][0]));
        acc[r][t][1] = fma(bi, sum1[r], fma(sc, P[r][t].thread_elements()[1], acc[r][t][1]));
      }
    }
  }

  threadgroup float part[S * RT * NT * 64];        // split-K partials
  for (int r = 0; r < RT; ++r)
    for (int t = 0; t < NT; ++t)
      for (int e = 0; e < 2; ++e) part[((c * RT + r) * NT + t) * 64 + fm * 8 + fn + e] = acc[r][t][e];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = c * 32 + lane; i < RT * NT * 64; i += S * 32) {
    float v = 0.0f;
    for (int q = 0; q < S; ++q) v += part[q * (RT * NT * 64) + i];
    const int r = i / (NT * 64), t = (i / 64) % NT, e = i % 64;
    const int row = m0 + 8 * r + (e % 8), n = n0 + 8 * t + (e / 8);
    if (row < R && n < N) OUT[size_t(row) * N + n] = T(v);
  }
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="dspark_skinny_qmm", input_names=["X", "Wq", "SC", "BI"], output_names=["OUT"],
            source=_SRC, header=_HEADER)
    return _KERNEL


def fits(mod) -> bool:
    """The layout the kernel reads: 4- or 8-bit affine, groups of 64, K % 64 == 0, N % 8 == 0
    (scales/biases in any float dtype — read as ``float(SC[i])``)."""
    w = mod["weight"]
    return (int(mod.bits) in WINDOW and int(mod.group_size) == GROUP
            and getattr(mod, "mode", "affine") == "affine" and "biases" in mod
            and mod["scales"].dtype in (mx.bfloat16, mx.float16, mx.float32) and w.ndim == 2
            and (int(w.shape[1]) * 32 // int(mod.bits)) % GROUP == 0 and int(w.shape[0]) % 8 == 0)


def _probe_cap() -> int:
    """Largest ``S`` this GPU will actually launch for THE REAL qmm kernel, measured once.

    ``maxTotalThreadsPerThreadgroup`` on Apple Silicon is **not** one number per chip: it
    depends on the kernel's register footprint, and it moves as ``S`` moves (more split-K
    chunks ⇒ more live state ⇒ fewer threads admitted). Measured on Mac14,10 (M2 Pro) with
    mlx-dspark 0.20.1: the stock ``S=32`` was rejected with
    ``Thread group size (1024) is greater than the maximum allowed threads per threadgroup
    (448)`` — and every fixed guess (832 → 26, 704 → 22, 512 → 16) was rejected in turn,
    because the ceiling fell alongside the value being tested.

    A separate 1-thread probe kernel is no help either: it compiles a different occupancy
    profile and happily reports 1024 on the same machine.

    So the only reliable measurement is the real kernel with a real shape, binary-searched
    once per process. Costs ~0.5 s and is cached in ``_SIMDGROUPS_MAX``.
    """
    # Shapes mirror what the calibration and decode paths actually run: a wide-N expert
    # tile, a wide-K projection, and a narrow hybrid gate. The ceiling is shape-dependent,
    # so the probe keeps the smallest surviving value — that one is safe for all of them.
    shapes = [(248320, 2048, 6), (2048, 512, 6), (512, 2048, 6)]
    try:
        samples = []
        for n, k, rows in shapes:
            samples.append((
                (mx.random.normal((rows, k)) * 0.1).astype(mx.bfloat16),
                mx.zeros((n, k // 8), dtype=mx.uint32),
                mx.zeros((n, k // 64), dtype=mx.bfloat16),
                mx.zeros((n, k // 64), dtype=mx.bfloat16),
                n, k, rows,
            ))
    except Exception:
        return 8

    def launches(s: int) -> bool:
        for x, w, sc, bi, n, k, rows in samples:
            try:
                nt = 2 if n % 16 == 0 else 1
                template = [("T", mx.bfloat16), ("K", k), ("N", n), ("BITS", 4),
                            ("NT", nt), ("RT", 1), ("S", s)]
                grid = (math.ceil(n / (8 * nt)) * s * 32, math.ceil(rows / 8), 1)
                out = _kernel()(inputs=[x, w, sc, bi], template=template, grid=grid,
                                 threadgroup=(s * 32, 1, 1),
                                 output_shapes=[(rows, n)], output_dtypes=[mx.bfloat16])
                mx.eval(out[0])          # force THIS launch to execute — not just to be queued
            except Exception:
                return False
        return True

    # 32 is the highest the stock code uses; below 8 the kernel has no split-K left to give.
    lo, hi, best = 1, 32, 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if launches(mid):
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return max(1, best)


def _max_simdgroups() -> int:
    """This GPU's measured simdgroup ceiling for the qmm kernel (probed once, cached).

    Falls back to a conservative 8 (256 threads) if the probe cannot run at all, so an
    unknown chip still launches the kernel instead of aborting the whole generation.
    """
    global _SIMDGROUPS_MAX
    try:
        return _SIMDGROUPS_MAX  # type: ignore[name-defined]
    except NameError:
        pass
    try:
        cap = _probe_cap()
    except Exception:
        cap = 8
    _SIMDGROUPS_MAX = max(1, min(32, int(cap)))  # type: ignore[assignment]
    return _SIMDGROUPS_MAX  # type: ignore[return-value]


def geometry(n: int, rt: int = 1) -> tuple[int, int]:
    """``(NT, S)``: 8-output tiles per simdgroup and simdgroups (split-K chunks) per
    threadgroup. Wide outputs take 4 tiles × 8 chunks (the measured best on every
    4096+-output shape); narrow ones trade tiles for K-splits so the grid still fills the GPU
    — at N=48 (a hybrid's gate projections) a fixed 4×8 launches 2 threadgroups and ran 0.92×
    of stock. The split-K partials (S·RT·NT·64 floats) stay within 16 KB of threadgroup
    memory for two row tiles and 16-32 KB for one.

    ``S`` is clamped to what this GPU actually admits for THIS kernel (see
    :func:`_max_simdgroups`). A hard-coded 32 aborts the launch on M1/M2, which cap well
    below the 1024 threads that M3/M4 allow."""
    cap = _max_simdgroups()
    if n >= 4096:
        return 4, min(8, cap)
    if n >= 1024:
        return (4 if rt == 1 else 2), min(16, cap)
    return (2 if rt == 1 and n % 16 == 0 else 1), min(32, cap)


def plan(rows: int, n: int, k: int, bits: int, dtype=mx.bfloat16):
    """``(template, grid, threadgroup)`` for ``rows`` × [n, k] (cached per module per rows)."""
    rt = min(RT_MAX, (rows + 7) // 8)
    nt, s = geometry(n, rt)
    template = [("T", dtype), ("K", k), ("N", n), ("BITS", bits), ("NT", nt), ("RT", rt), ("S", s)]
    grid = (math.ceil(n / (8 * nt)) * s * 32, math.ceil(rows / (8 * rt)), 1)
    return template, grid, (s * 32, 1, 1)


def qmm(x2: mx.array, weight: mx.array, scales: mx.array, biases: mx.array, bits: int,
        _plan=None) -> mx.array:
    """``x2 [rows, K] @ dequant(W)ᵀ`` for gs64 affine 4/8-bit ``W`` -> ``[rows, N]``."""
    rows, k = int(x2.shape[0]), int(x2.shape[1])
    n = int(weight.shape[0])
    template, grid, tg = _plan or plan(rows, n, k, bits, x2.dtype)
    return _kernel()(inputs=[x2, weight, scales, biases], template=template, grid=grid,
                     threadgroup=tg, output_shapes=[(rows, n)], output_dtypes=[x2.dtype])[0]


def call(mod, x: mx.array, rows: int) -> mx.array:
    """``QuantizedLinear(x)`` through the kernel, launch plan cached on the module (per-call
    Python cost matters: ~370 linears a 27B verify forward)."""
    plans = mod.__dict__.get("_dspark_skinny")
    if plans is None:
        plans = {}
        object.__setattr__(mod, "_dspark_skinny", plans)    # not a parameter
    k = x.shape[-1]
    p = plans.get(rows)
    if p is None:
        p = plans[rows] = plan(rows, int(mod["weight"].shape[0]), int(k), int(mod.bits), x.dtype)
    y = qmm(x.reshape(rows, k), mod["weight"], mod["scales"], mod["biases"], int(mod.bits), _plan=p)
    y = y.reshape(*x.shape[:-1], y.shape[-1])
    if "bias" in mod:
        y = y + mod["bias"]
    return y
