"""Multi-row decode attention: one Metal kernel for every few-row SDPA speculative decoding runs.

Speculative verify attends 2-16 query rows (verify width = cap + 1) over the whole KV
cache, and a DSpark drafter attends its 7-16-row block over the whole drafter context.
mlx's ``scaled_dot_product_attention`` serves those shapes badly at depth:

- the *vector* kernel (``q_len * gqa <= 32``) gives every (query row, head) its own
  simdgroup and pays one ``simd_sum`` reduction per key per row, so cost grows linearly
  with rows even though the KV bytes are shared (Qwen3.8 24/4/256 @32k: q1 0.72 ms,
  q4 1.94 ms);
- past that gate it falls to the *unfused* graph (materialized scores, the KV re-read per
  query head): q6 7.0 ms, q8 7.5 ms at 32k — the cliff ``sdpa_split`` dodges by cutting
  the rows into sub-calls, which still pays the per-row cost (q8 split 3.9 ms).

This kernel packs ALL ``gqa * q_len`` rows of a KV head into 8-row ``simdgroup_matrix``
tiles and runs Q·Kᵀ and P·V as 8x8 MMAs over 16-key tiles, so each K/V tile is read from
DRAM once per threadgroup and consumed by every row on the matrix units. The key range is
split across threadgroups (flash-decoding split-K) for occupancy, and a second pass merges
the per-block ``(max, sum, acc)`` partials. For head_dim 256 the d axis is split across a
pair of simdgroups (register pressure: 32 Q + 32 O fragments per lane spills and ran 4x
slower than mlx), exchanging partial scores through threadgroup memory once per tile.

Measured (M4 Pro, mlx 0.32.2, dependent chains over 4 rotated KV "layers"):

    Qwen3.8 24/4/256 @32k   q1 0.73->0.65  q4 1.95->0.87  q8 7.5 (split 3.9)->1.56  q16 8.0->3.3 ms
    Qwen3-8B 32/8/128 @32k  q1 0.82->0.62  q4 1.64->0.68  q8 3.05->0.96  q16 4.7->2.0 ms
    GQA-16 32/2/128 @32k    q4 6.8->0.44   q8 7.2->0.89  (the split's window is 2 rows there)
    and 1.5-3x at 2k-8k for q >= 2.

Whole-forward (real Qwen3.8-27B-4bit verify, with the tap, vs the better of stock/split):
width 8 at 16k 206->187 ms, at 32k 238->198 ms (1.20x); width 4 at 32k 124->107 ms.

Numerics: fp32 softmax and accumulation, bf16/fp16 MMA inputs (P rounded to the input
dtype for the P·V MMA — the unfused path mlx takes past the cliff rounds P too). Max |Δ|
vs an fp32 reference equals or beats mlx's own kernels at every probed shape: fp-tie
class, like the split and the small-M kernel. The gate is MEASURED per (chip x mlx x
attention shape) by :func:`measure_window` and numerics-checked there; M5 (g17+) is gated
off until verified end to end there (the small-M kernel's M5 stall is the precedent).
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass

import mlx.core as mx

# The real kernel, captured at import (the patch must never re-enter itself).
_ORIG_SDPA = mx.fast.scaled_dot_product_attention

_P1 = r"""
    constexpr int DT = DH / 8 / DS;            // 8-wide d tiles owned by this simdgroup
    const int L   = k_shape[2];
    const int Hk  = k_shape[1];
    const int hk  = (int)threadgroup_position_in_grid.x / NRG;
    const int rg  = (int)threadgroup_position_in_grid.x % NRG;
    const int b   = (int)threadgroup_position_in_grid.y;
    const int blk = (int)threadgroup_position_in_grid.z;
    const uint tid  = thread_position_in_threadgroup.x;
    const uint sg   = tid >> 5;
    const uint lane = tid & 31;
    const int rt = rg * (NSG / DS) + (int)sg / DS;     // global 8-row tile
    const int dh = (int)sg % DS;
    const int d0 = dh * (DH / DS);

    const int64_t ksb = k_strides[0], ksh = k_strides[1], kss = k_strides[2];
    const int64_t vsb = v_strides[0], vsh = v_strides[1], vss = v_strides[2];
    const device T* kb = k + b * ksb + hk * ksh + d0;
    const device T* vb = v + b * vsb + hk * vsh + d0;
    const device T* qb = q + ((int64_t)(b * Hk + hk) * RP + rt * 8) * DH + d0;

    simdgroup_matrix<T, 8, 8> Qf[DT];
    for (int dt = 0; dt < DT; ++dt) simdgroup_load(Qf[dt], qb + dt * 8, DH);
    simdgroup_matrix<float, 8, 8> Of[DT];
    for (int dt = 0; dt < DT; ++dt) Of[dt] = simdgroup_matrix<float, 8, 8>(0);

    threadgroup float sbuf[DS > 1 ? 2 * NSG * 128 : 1];

    // lane -> (row fm, cols fn, fn+1) of an 8x8 fragment (Apple's simdgroup layout)
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    const int row = rt * 8 + fm;
    const int qi = row % QL;                   // rows are (gqa head)-major, then q position
    const int kmax = CAUSAL ? (L - QL + qi) : (L - 1);
    const float sc = scale[0] * 1.4426950408889634f;   // exp2 domain

    float m = -1e30f, l = 0.0f;
    const int k0 = blk * CHUNK;
    const int k1 = min(L, k0 + CHUNK);
    int it = 0;
    for (int kt = k0; kt < k1; kt += 16, ++it) {
        // the tail tile shifts back in bounds; keys it re-covers are masked below
        const int ld = (kt + 16 <= L) ? kt : max(0, L - 16);
        simdgroup_matrix<float, 8, 8> S0 = simdgroup_matrix<float, 8, 8>(0);
        simdgroup_matrix<float, 8, 8> S1 = simdgroup_matrix<float, 8, 8>(0);
        const device T* kp = kb + (int64_t)ld * kss;
        for (int dt = 0; dt < DT; ++dt) {
            simdgroup_matrix<T, 8, 8> K0, K1;
            simdgroup_load(K0, kp + dt * 8, kss, ulong2(0, 0), true);
            simdgroup_load(K1, kp + 8 * kss + dt * 8, kss, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(S0, Qf[dt], K0, S0);
            simdgroup_multiply_accumulate(S1, Qf[dt], K1, S1);
        }
        float s[4] = {S0.thread_elements()[0], S0.thread_elements()[1],
                      S1.thread_elements()[0], S1.thread_elements()[1]};
        if (DS > 1) {                          // sum the d-slice partial scores
            threadgroup float* mine = sbuf + ((it & 1) * NSG + sg) * 128;
            simdgroup_store(S0, mine, 8);
            simdgroup_store(S1, mine + 64, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int o = 1; o < DS; ++o) {
                const int partner = ((int)sg / DS) * DS + ((dh + o) % DS);
                threadgroup float* th = sbuf + ((it & 1) * NSG + partner) * 128;
                s[0] += th[fm * 8 + fn];      s[1] += th[fm * 8 + fn + 1];
                s[2] += th[64 + fm * 8 + fn]; s[3] += th[64 + fm * 8 + fn + 1];
            }
        }
        float mx_ = -INFINITY;
        for (int j = 0; j < 4; ++j) {
            const int key = ld + (j >> 1) * 8 + fn + (j & 1);
            s[j] = (key < kt || key > kmax || key >= k1) ? -INFINITY : s[j] * sc;
            mx_ = max(mx_, s[j]);
        }
        mx_ = max(mx_, simd_shuffle_xor(mx_, 1));
        mx_ = max(mx_, simd_shuffle_xor(mx_, 8));
        const float mn = max(m, mx_);
        const float f = fast::exp2(m - mn);
        float ps = 0.0f;
        for (int j = 0; j < 4; ++j) { s[j] = fast::exp2(s[j] - mn); ps += s[j]; }
        ps += simd_shuffle_xor(ps, 1);
        ps += simd_shuffle_xor(ps, 8);
        l = l * f + ps;
        m = mn;
        simdgroup_matrix<T, 8, 8> P0, P1;
        P0.thread_elements()[0] = (T)s[0];
        P0.thread_elements()[1] = (T)s[1];
        P1.thread_elements()[0] = (T)s[2];
        P1.thread_elements()[1] = (T)s[3];
        const device T* vp = vb + (int64_t)ld * vss;
        for (int dt = 0; dt < DT; ++dt) {
            Of[dt].thread_elements()[0] *= f;
            Of[dt].thread_elements()[1] *= f;
            simdgroup_matrix<T, 8, 8> V0, V1;
            simdgroup_load(V0, vp + dt * 8, vss);
            simdgroup_load(V1, vp + 8 * vss + dt * 8, vss);
            simdgroup_multiply_accumulate(Of[dt], P0, V0, Of[dt]);
            simdgroup_multiply_accumulate(Of[dt], P1, V1, Of[dt]);
        }
    }
    const int NB = (int)threadgroups_per_grid.z;
    const int64_t pbase = ((int64_t)(b * Hk + hk) * NB + blk) * RP;
    device float* op = opart + (pbase + rt * 8) * DH + d0;
    for (int dt = 0; dt < DT; ++dt) simdgroup_store(Of[dt], op + dt * 8, DH);
    if (fn == 0 && dh == 0) {
        mpart[pbase + row] = m;
        lpart[pbase + row] = l;
    }
"""

_P2 = r"""
    const int d  = (int)thread_position_in_grid.x;
    const int r  = (int)thread_position_in_grid.y;
    const int bh = (int)thread_position_in_grid.z;
    const device float* mp = mpart + (int64_t)bh * NB * RP + r;
    const device float* lp = lpart + (int64_t)bh * NB * RP + r;
    const device float* op = opart + ((int64_t)bh * NB * RP + r) * DH + d;
    float M = -1e30f;
    for (int j = 0; j < NB; ++j) M = max(M, mp[j * RP]);
    float den = 0.0f, num = 0.0f;
    for (int j = 0; j < NB; ++j) {
        const float w = fast::exp2(mp[j * RP] - M);
        den += w * lp[j * RP];
        num += w * op[(int64_t)j * RP * DH];
    }
    out[((int64_t)bh * R + r) * DH + d] = (T)(num / den);
"""

_K1 = mx.fast.metal_kernel(name="dspark_mra_p1", input_names=["q", "k", "v", "scale"],
                           output_names=["opart", "mpart", "lpart"], source=_P1,
                           ensure_row_contiguous=False)   # K/V are strided KV-cache views
_K2 = mx.fast.metal_kernel(name="dspark_mra_p2", input_names=["opart", "mpart", "lpart"],
                           output_names=["out"], source=_P2)

MAX_SG = 32      # simdgroups per threadgroup (1024 threads)
MAX_D = 256
_DTYPES = (mx.bfloat16, mx.float16)


def plan(hq: int, hk: int, q_len: int, d: int, kv_len: int, batch: int = 1) -> dict:
    """Launch geometry. ``DS``: d-split factor (2 at head_dim 256 — register pressure);
    row tiles beyond one threadgroup's 32 simdgroups spill into row groups (``NRG``); the
    key range is cut into ``NB`` blocks of ``CHUNK`` keys so the grid fills the GPU (more
    blocks on short KV, fewer on long where each block amortizes the combine). Tuned on
    M4 Pro over ds in {1,2,4} x grid in {128..1024}; the win is broad, not a knife edge."""
    rows = (hq // hk) * q_len
    ds = 2 if d >= 256 else 1
    tiles = -(-rows // 8)
    per_tg = max(1, MAX_SG // ds)
    nrg = -(-tiles // per_tg)
    per_tg = -(-tiles // nrg)
    target = 1024 if kv_len <= 4096 else (256 if q_len <= 4 else 128)
    nb = max(1, min(max(1, kv_len // 128), target // max(1, hk * batch * nrg)))
    chunk = math.ceil(math.ceil(kv_len / nb) / 16) * 16
    return {"R": rows, "RP": nrg * per_tg * 8, "DS": ds, "NSG": per_tg * ds, "NRG": nrg,
            "CHUNK": chunk, "NB": math.ceil(kv_len / chunk)}


def multirow_sdpa(q: mx.array, k: mx.array, v: mx.array, *, scale: float,
                  causal: bool) -> mx.array:
    """``scaled_dot_product_attention(q, k, v, scale, mask="causal" if causal else None)``
    for the few-row shapes (see :func:`eligible`). ``k``/``v`` may be strided cache views."""
    B, Hq, qL, D = q.shape
    Hk, L = k.shape[1], k.shape[2]
    p = plan(Hq, Hk, qL, D, L, B)
    R, RP, nb = p["R"], p["RP"], p["NB"]
    qq = q.reshape(B, Hk, R, D)
    if RP != R:
        qq = mx.pad(qq, [(0, 0), (0, 0), (0, RP - R), (0, 0)])
    qq = mx.contiguous(qq)
    opart, mpart, lpart = _K1(
        inputs=[qq, k, v, mx.array([scale], dtype=mx.float32)],
        template=[("T", q.dtype), ("DH", D), ("RP", RP), ("QL", qL), ("CHUNK", p["CHUNK"]),
                  ("CAUSAL", bool(causal)), ("DS", p["DS"]), ("NSG", p["NSG"]),
                  ("NRG", p["NRG"])],
        grid=(Hk * p["NRG"] * 32 * p["NSG"], B, nb),
        threadgroup=(32 * p["NSG"], 1, 1),
        output_shapes=[(B, Hk, nb, RP, D), (B, Hk, nb, RP), (B, Hk, nb, RP)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    (out,) = _K2(
        inputs=[opart, mpart, lpart],
        template=[("T", q.dtype), ("DH", D), ("RP", RP), ("R", R), ("NB", nb)],
        grid=(D, R, B * Hk),
        threadgroup=(min(D, 256), 1, 1),
        output_shapes=[(B, Hq, qL, D)],
        output_dtypes=[q.dtype],
    )
    return out


@dataclass(frozen=True)
class Window:
    """Measured gate for one attention shape ``(hq, hk, d)``: the kernel serves its calls
    with ``min_q <= q_len <= max_q`` and ``kv_len >= min_kv``."""
    hq: int
    hk: int
    d: int
    min_q: int = 2
    max_q: int = 16
    min_kv: int = 1024


@dataclass(frozen=True)
class MultirowConfig:
    """One :class:`Window` per MEASURED attention shape (the target's full-attention layers,
    a DSpark drafter's context attention). A call whose shape was never raced on this
    machine passes through to mlx — the kernel only runs where it was measured to win."""
    windows: tuple[Window, ...] = ()

    def window(self, hq: int, hk: int, d: int) -> Window | None:
        for w in self.windows:
            if w.hq == hq and w.hk == hk and w.d == d:
                return w
        return None


def eligible(q, k, v, mask, sinks, cfg: MultirowConfig) -> bool:
    """Structural + gate check. Only unmasked or string-"causal" attention without sinks,
    4-D bf16/fp16 tensors of a MEASURED shape, matching K/V/Q head dims (a multiple of 8,
    <= 256), whole GQA groups, and q_len / KV length inside that shape's measured window.
    Everything else (prefill's
    wide q, array masks — sliding-window drafters, batched rows — sinks, quantized KV which
    never reaches this function) passes through to mlx unchanged."""
    if sinks is not None or q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return False
    if not (mask is None or (isinstance(mask, str) and mask == "causal")):
        return False
    B, Hq, qL, D = q.shape
    Hk, L = k.shape[1], k.shape[2]
    w = cfg.window(Hq, Hk, D)
    if w is None or not (w.min_q <= qL <= w.max_q and qL <= L) or max(w.min_kv, 16) > L:
        return False
    if q.dtype not in _DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    if k.shape[-1] != D or v.shape[-1] != D or D % 8 or D > MAX_D or v.shape[2] != L:
        return False
    # Precondition, not checked (mx.array exposes no strides to Python): K/V are contiguous
    # along head_dim. True of every caller the patch can see — mlx-lm KV caches and the
    # drafter context hand out views sliced along the SEQUENCE axis only; the kernel reads
    # their real batch/head/seq strides (ensure_row_contiguous=False), never copying.
    return not (Hk == 0 or Hq % Hk or v.shape[1] != Hk or k.shape[0] != B or v.shape[0] != B)


# ---- scoped patch (mirrors sdpa_split / small_m_qmm) --------------------------------

_ACTIVE: MultirowConfig | None = None
_PREV = None     # what the patch falls through to (the split patch, or mlx's own)


def _patched_call(q, k, v, *, scale, mask=None, sinks=None, **kw):
    cfg = _ACTIVE
    if cfg is not None and not kw and eligible(q, k, v, mask, sinks, cfg):
        return multirow_sdpa(q, k, v, scale=scale,
                             causal=isinstance(mask, str) and q.shape[2] > 1)
    prev = _PREV or _ORIG_SDPA
    return prev(q, k, v, scale=scale, mask=mask, sinks=sinks, **kw)


@contextlib.contextmanager
def multirow_attention(cfg: MultirowConfig | None):
    """Route eligible ``mx.fast.scaled_dot_product_attention`` calls through the kernel
    while active; everything else falls through to whatever was installed before (so it
    composes with :func:`~mlx_dspark.sdpa_split.sdpa_split` installed first). No-op when
    ``cfg`` is None or the patch is already installed (nests cleanly)."""
    global _ACTIVE, _PREV
    if cfg is None or mx.fast.scaled_dot_product_attention is _patched_call:
        yield
        return
    prev = mx.fast.scaled_dot_product_attention
    prev_cfg, prev_prev = _ACTIVE, _PREV
    _ACTIVE, _PREV = cfg, prev
    mx.fast.scaled_dot_product_attention = _patched_call
    try:
        yield
    finally:
        mx.fast.scaled_dot_product_attention = prev
        _ACTIVE, _PREV = prev_cfg, prev_prev


def active() -> bool:
    return mx.fast.scaled_dot_product_attention is _patched_call


# ---- measurement ---------------------------------------------------------------------

PROBE_Q = (2, 4, 6, 8, 12, 16)
PROBE_KV = (1024, 4096, 16384)
WIN = 1.05         # the kernel must beat mlx by >5% (the noise band) at a probed point


def _chain_ms(fn, q0, kvs, iters: int = 4) -> float:
    """ms per call over a DEPENDENT chain (each output is the next query) across rotated
    KV sets — independent calls overlap on the GPU and a single KV stays cache-resident,
    both of which flatter a kernel (the small-M measurement notes)."""
    import time

    def run():
        q = q0
        for _ in range(iters):
            for k, v in kvs:
                q = fn(q, k, v)
        return q
    mx.eval(run())
    mx.synchronize()
    t = time.perf_counter()
    mx.eval(run())
    mx.synchronize()
    return (time.perf_counter() - t) * 1e3 / (iters * len(kvs))


def measure_window(hq: int, hk: int, d: int, *, dtype=mx.bfloat16,
                   verbose: bool = False) -> tuple[dict | None, dict]:
    """Race the kernel against mlx on this (chip x mlx x attention shape) and return
    ``(window, report)``: ``window`` is ``{min_q, max_q, min_kv}`` (a :class:`Window` for
    this shape) or ``None`` (disabled: the kernel failed to build, mis-computed, or never won).

    ``max_q`` is the widest probed q such that the kernel beats mlx by >5% at every probed
    q <= it at the deepest KV; ``min_kv`` the shortest probed KV from which that whole q
    range wins at every deeper probe too (the win grows with depth). Numerics are
    checked at every probe against an fp32 reference: the kernel's max |Δ| must stay within
    2x mlx's own (+ a small floor), or the whole window is refused."""
    if d % 8 or d > MAX_D or hk == 0 or hq % hk:
        return None, {"reason": "shape"}
    scale = 1.0 / math.sqrt(d)
    report: dict = {"shape": f"{hq}x{hk}x{d}", "points": {}}
    wins_at: dict[int, list[int]] = {}
    try:
        for kv in PROBE_KV:
            kvs = [(mx.random.normal((1, hk, kv, d)).astype(dtype),
                    mx.random.normal((1, hk, kv, d)).astype(dtype)) for _ in range(3)]
            mx.eval(kvs)
            ok = []
            for ql in PROBE_Q:
                q = mx.random.normal((1, hq, ql, d)).astype(dtype)
                k0, v0 = kvs[0]
                ref = _ORIG_SDPA(q.astype(mx.float32), k0.astype(mx.float32),
                                 v0.astype(mx.float32), scale=scale, mask="causal")
                o_m = multirow_sdpa(q, k0, v0, scale=scale, causal=True)
                o_x = _ORIG_SDPA(q, k0, v0, scale=scale, mask="causal")
                e_m = float(mx.abs(o_m.astype(mx.float32) - ref).max())
                e_x = float(mx.abs(o_x.astype(mx.float32) - ref).max())
                if math.isnan(e_m) or e_m > 2.0 * e_x + 2e-3:     # NaN or off
                    report["reason"] = f"numerics q{ql} kv{kv}: {e_m:.4g} vs mlx {e_x:.4g}"
                    if verbose:
                        print(f"  multirow-attn: REFUSED — {report['reason']}", flush=True)
                    return None, report
                t_x = _chain_ms(lambda q, k, v: _ORIG_SDPA(q, k, v, scale=scale, mask="causal"),
                                q, kvs)
                t_m = _chain_ms(lambda q, k, v: multirow_sdpa(q, k, v, scale=scale, causal=True),
                                q, kvs)
                report["points"][f"q{ql}@{kv}"] = round(t_x / t_m, 2)
                if t_x > WIN * t_m:
                    ok.append(ql)
            wins_at[kv] = ok
            del kvs
        mx.clear_cache()
    except Exception as e:  # noqa: BLE001 — a kernel that won't build here is simply off
        report["reason"] = f"kernel unavailable: {type(e).__name__}: {e}"
        if verbose:
            print(f"  multirow-attn: OFF — {report['reason']}", flush=True)
        return None, report
    window = None
    deep = wins_at[PROBE_KV[-1]]
    max_q = 0
    for ql in PROBE_Q:                   # contiguous q range winning at the deepest probe
        if ql not in deep:
            break
        max_q = ql
    if max_q:
        qs = [ql for ql in PROBE_Q if ql <= max_q]
        min_kv = PROBE_KV[-1]
        for kv in reversed(PROBE_KV):    # extend toward shorter KV while the range holds
            if all(ql in wins_at[kv] for ql in qs):
                min_kv = kv
            else:
                break
        window = {"min_q": PROBE_Q[0], "max_q": max_q, "min_kv": min_kv}
    else:
        report["reason"] = "no win on this chip/mlx"
    return window, report
