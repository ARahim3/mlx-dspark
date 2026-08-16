"""Small-M quantized matmul for the verify window: an MMA kernel that amortizes the
weight read across rows.

``mx.quantized_matmul`` grows ~linearly in M for M in 2..8 on Apple GPUs — the weight
read is re-paid per row until the GEMM tiling takes over around M=13+ (upstream
ml-explore/mlx#4265; our own #3852 is the 2-bit face of the same thing). Speculative
verify lives exactly there: verify width = cap + 1, and the drafter's bidirectional
block backbone runs at block width every round. So on 4-bit targets the verify curve
"rises steeply from width 3" (the measured shape that pins Qwen3.8-27B-4bit at cap 2)
for no fundamental reason.

The kernel here is avlp12's ``qmm_mma4`` (github.com/avlp12/mlx-lm ``fast_qmm.py``,
MIT — see NOTICE), vendored with the dispatch rewritten to this project's conventions:
an 8x8 ``simdgroup_matrix`` MMA tile covers M <= 8 exactly, so each quantized weight
group is read and dequantized ONCE and reused by every row; K is split across the 8
simdgroups of a threadgroup (split-K) so the serial loop per simdgroup is short.

Measured (M4 Pro, mlx 0.32.0, dependent chain, rotated weights — both are load-bearing,
see "Measurement notes" below), real Qwen3.8-27B-4bit shapes:

    shape              M=4     M=5     M=6     M=7     M=8
    17408x5120        0.85x   1.05x   1.31x   1.58x   1.61x
    5120x17408        0.87x   1.05x   1.32x   1.61x   1.63x
    10240x5120        0.84x   1.03x   1.28x   1.55x   1.56x
    248320x5120 (head) 0.85x  1.07x   1.36x   1.68x   1.70x

The kernel is FLAT in M (one tile). ``quantized_matmul`` wins below M=5 (its GEMV path
is at roofline), so this is a supplement dispatched only inside M in [M_MIN, 8] — never
a replacement. End-to-end it turned Qwen3.8-27B-4bit code cap 5 from 23.7 to 29.2 tok/s
(2.06x vs baseline, past cap 2's 1.89x), output ids identical.

An **8-bit variant** (same kernel, different unpack — 4 values per uint32 instead of 8)
covers 8-bit gs64 targets, where the story is the mirror image: 8-bit qmm is already
flat to M=5 (why cap 4 was its optimum) but cliffs at M=6 (0.41 -> 0.66 ms on the
Qwen3.8-27B-8bit MLP shapes, 5.7 -> 9.3 on its lm_head); the kernel holds the flat level
through M=8 (1.20-1.62x at M=6-8), so caps 5-7 stop being priced out there too.

Numerics: fp32 accumulation in a different order than qmm — outputs differ by 1-2 bf16
ulps (max|d| 0.125-0.375 at max|y| ~30-58). Same class as the batched path: per-token
greedy-correct under the verify loop (the target still verifies every token), NOT
bit-identical to the stock kernel. :func:`measure_shapes` checks both the numerics and
the speed per shape at runtime and returns only shapes where the kernel actually wins
on this (chip x mlx version) — the wide_gemm doctrine: verified, never assumed.

Measurement notes (cost real time upstream, don't relearn):
- DEPENDENT chains only. Queue-batched/independent-call microbenches let kernel launches
  overlap and hide the longer critical path — avlp12 measured a kernel that looked 1.13x
  that way and made the real model 26% slower. Decode is a serial chain.
- Rotate weights across calls (katlun's lesson): a single weight stays cache-resident
  and hides the read cost the kernel exists to amortize. :func:`measure_shapes` rotates
  across the model's OWN same-shape layers, so it allocates nothing.

Eligibility is tighter than upstream's: the K-loop strides 64 values per simdgroup over
a K/8 slice, so K must be a multiple of 512 (upstream gates on 128, which would
mis-handle K in {128,256,384} mod 512 — all real shapes here are %512 anyway).
"""

from __future__ import annotations

import contextlib
import time

import mlx.core as mx
import mlx.nn as nn

M_MIN = 6   # measured crossover on a dependent chain: M=5 is 1.03-1.07x (noise band,
# and upstream measured it net-negative in-model), M=6 is the first clear win. Constant
# of (chip x mlx version) — revisit after any mlx upgrade, like every knee here.
# The same window holds for the 8-bit variant from the other side: 8-bit qmm is FLAT to
# M=5 (why cap 4 was its measured optimum) and cliffs at 6 (0.41 -> 0.66 ms on the MLP
# shapes); the kernel sits at the flat level through M=8 (1.20-1.62x at M=6-8).
M_MAX = 8   # one MMA tile
N_MIN = 4096  # below this the grid is too few threadgroups to occupy the GPU

# The dequant unpack is the only part that differs by quantization width: 4-bit packs 8
# values per uint32 (16 values = 2 uints per lane-slice), 8-bit packs 4 (16 values = 4
# uints). Everything else — split-K, staging layout, MMA, reduction — is identical.
_UNPACK = {
    4: r"""
            const device uint* wr = w + (size_t)n * (K / 8) + (ka >> 3) + kq * 2;
            uint p0 = wr[0], p1 = wr[1];
            for (int t = 0; t < 8; ++t)
                bt[(kq * 16 + t) * 8 + j] = (bfloat16_t)((float)((p0 >> (4 * t)) & 15u) * s + bb);
            for (int t = 0; t < 8; ++t)
                bt[(kq * 16 + 8 + t) * 8 + j] = (bfloat16_t)((float)((p1 >> (4 * t)) & 15u) * s + bb);
""",
    8: r"""
            const device uint* wr = w + (size_t)n * (K / 4) + (ka >> 2) + kq * 4;
            for (int u = 0; u < 4; ++u) {
                uint p = wr[u];
                for (int t = 0; t < 4; ++t)
                    bt[(kq * 16 + u * 4 + t) * 8 + j] =
                        (bfloat16_t)((float)((p >> (8 * t)) & 255u) * s + bb);
            }
""",
}

_SRC = r"""
    const int K = KD, N = ND, M = MD;
    const int KPS = KD / 8;                 // K-span per simdgroup (split-K)

    uint tid  = thread_position_in_threadgroup.x;
    uint tgid = threadgroup_position_in_grid.x;
    uint sg   = tid >> 5;
    uint lane = tid & 31;

    int n0 = (int)tgid * 8;                 // one threadgroup -> 8 output columns

    // x is read straight from device into the MMA (bf16 in, fp32 accumulate): no
    // staging keeps threadgroup memory small and, more importantly, the critical
    // path short.
    threadgroup bfloat16_t bs[8 * 512];     // per-simdgroup 64k x 8n dequant stage
    threadgroup float red[8 * 64];          // cross-simdgroup reduction

    simdgroup_matrix<float, 8, 8> C = simdgroup_matrix<float, 8, 8>(0);
    threadgroup bfloat16_t* bt = bs + sg * 512;

    // Split-K: the 8 simdgroups each walk 1/8 of K in short serial loops. (A prior
    // revision walked all of K per threadgroup and put ~160 barrier pairs on the
    // critical path.)
    int kbeg = (int)sg * KPS;
    for (int kk = 0; kk < KPS; kk += 64) {
        int ka = kbeg + kk;
        int j  = (int)(lane & 7);
        int kq = (int)(lane >> 3);
        int n  = n0 + j;
        if (n < N) {
            // dequantize per quantization group (64 values), not per MMA tile (8):
            // one scale/bias load serves the whole group and barriers drop 8x.
            int g = ka >> 6;
            float s  = (float)sc[(size_t)n * (K / 64) + g];
            float bb = (float)bi[(size_t)n * (K / 64) + g];
__UNPACK__
        } else {
            for (int t = 0; t < 16; ++t) bt[(kq * 16 + t) * 8 + j] = (bfloat16_t)0;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<bfloat16_t, 8, 8> A, B;
        for (int kt = 0; kt < 8; ++kt) {
            simdgroup_load(A, x + ka + kt * 8, K);   // x rows 0..7 (padded to 8)
            simdgroup_load(B, bt + kt * 64, 8);
            simdgroup_multiply_accumulate(C, A, B, C);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(C, red + sg * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // sum the 8 split-K partials and write out
    for (int i = (int)tid; i < 64; i += 256) {
        int m = i >> 3, j = i & 7;
        int n = n0 + j;
        if (m < M && n < N) {
            float v = 0.0f;
            for (int q = 0; q < 8; ++q) v += red[q * 64 + i];
            out[(size_t)m * N + n] = (bfloat16_t)v;
        }
    }
"""

_TG = 256  # 8 simdgroups
_KERNELS = {
    bits: mx.fast.metal_kernel(
        name=f"dspark_qmm_mma{bits}",
        input_names=["x", "w", "sc", "bi"],
        output_names=["out"],
        source=_SRC.replace("__UNPACK__", unpack),
    )
    for bits, unpack in _UNPACK.items()
}

_orig_call = nn.QuantizedLinear.__call__
_active_ids: frozenset | None = None   # None = patch inactive


def _in_features(mod) -> int:
    """Affine quantization packs in_features into uint32 words (32/bits values each)."""
    return int(mod["weight"].shape[1]) * 32 // int(mod.bits)


def eligible(mod) -> bool:
    """Can this ``QuantizedLinear``'s shape/format run on the kernel at all?"""
    if not isinstance(mod, nn.QuantizedLinear):
        return False
    if getattr(mod, "mode", "affine") != "affine" or "biases" not in mod:
        return False
    if int(mod.bits) not in _KERNELS or int(mod.group_size) != 64:
        return False
    N = int(mod["weight"].shape[0])
    return N >= N_MIN and _in_features(mod) % 512 == 0


def shape_key(mod) -> str:
    return f"{_in_features(mod)}x{int(mod['weight'].shape[0])}b{int(mod.bits)}"


def _mma(x8: mx.array, wq, sc, bi, M: int, N: int, K: int, bits: int) -> mx.array:
    (out,) = _KERNELS[bits](
        inputs=[x8, wq, sc, bi],
        template=[("KD", K), ("ND", N), ("MD", M)],
        output_shapes=[(M, N)], output_dtypes=[mx.bfloat16],
        grid=(((N + 7) // 8) * _TG, 1, 1), threadgroup=(_TG, 1, 1),
    )
    return out


def _mma_call(self, x):
    M = 1
    for d in x.shape[:-1]:
        M *= d
    if (M_MIN <= M <= M_MAX and x.dtype == mx.bfloat16
            and _active_ids is not None and id(self) in _active_ids):
        K = x.shape[-1]
        N = self["weight"].shape[0]
        flat = x.reshape(M, K)
        if M < 8:  # the kernel reads a full 8-row MMA tile from device
            flat = mx.concatenate(
                [flat, mx.zeros((8 - M, K), dtype=flat.dtype)], axis=0)
        y = _mma(flat, self["weight"], self["scales"], self["biases"],
                 M, N, K, int(self.bits))
        y = y.reshape(*x.shape[:-1], N)
        if "bias" in self:
            y = y + self["bias"]
        return y
    return _orig_call(self, x)


@contextlib.contextmanager
def small_m_matmul(ids: frozenset | None):
    """Route the given ``QuantizedLinear`` instances (by ``id``) through the MMA kernel
    for forwards of M in [M_MIN, M_MAX] rows. ``None``/empty disables — the context is
    then a no-op, so call sites stay unconditional.

    ``ids`` is a *pre-verified allowlist* built by :func:`calibrate.apply_small_m` from
    shapes that :func:`measure_shapes` proved faster AND numerically sane on this
    machine. An id-set (not per-call shape math) keeps the dispatch overhead off the
    M=1 decode path. Class-level patch, one-MLX-thread rule applies (see NOTES);
    restores to the value found at entry so it nests with ``wide_matmul``.
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
_MIN_GAIN = 1.15   # a shape must beat quantized_matmul by this at M=8 to be worth the
# fp-tie churn; below it the flat-in-M kernel is within noise of the rising qmm curve.
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


def measure_shapes(*models, verbose: bool = False) -> list[str]:
    """Which eligible weight shapes actually WIN on the kernel, on this machine, now.

    For every distinct eligible shape in the given models: check numerics against
    ``quantized_matmul`` at each window width, then race a dependent chain at M=8 —
    rotating across the model's own same-shape layers so nothing stays cache-resident
    and nothing is allocated. Returns the shape keys that pass both gates; anything
    else stays on the stock path at stock speed and stock numerics.
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
        N = int(mod["weight"].shape[0])
        K = _in_features(mod)
        bits = int(mod.bits)

        ok = True
        for M in range(M_MIN, M_MAX + 1):
            x = (mx.random.normal((M, K)) * 0.1).astype(mx.bfloat16)
            x8 = x if M == 8 else mx.concatenate(
                [x, mx.zeros((8 - M, K), dtype=x.dtype)], axis=0)
            ref = _orig_call(mod, x).astype(mx.float32)
            got = _mma(x8, mod["weight"], mod["scales"], mod["biases"],
                       M, N, K, bits).astype(mx.float32)
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

        x = (mx.random.normal((8, K)) * 0.1).astype(mx.bfloat16)
        mx.eval(x)

        def q_step(xx, t, _m=mods):
            w = _m[t % len(_m)]
            return _orig_call(w, xx)

        def k_step(xx, t, _m=mods, _N=N, _K=K, _b=bits):
            w = _m[t % len(_m)]
            return _mma(xx, w["weight"], w["scales"], w["biases"], 8, _N, _K, _b)

        tq = min(_time_chain(q_step, x), _time_chain(q_step, x))
        tk = min(_time_chain(k_step, x), _time_chain(k_step, x))
        if tq / tk >= _MIN_GAIN:
            won.append(key)
        if verbose:
            print(f"  small-M qmm: {key} M=8 {tq/tk:.2f}x "
                  f"({'on' if tq / tk >= _MIN_GAIN else 'off'})", flush=True)
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
