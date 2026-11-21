"""Metal kernels for Inkling 2's routed experts and logits (Thinking Machines Inkling-2-Mini).

``run_qmv`` — the routed-expert projections of a decode step or a short verify: MLX's own
single-row quantized matvec (``fp_qmv_impl`` / ``fp_qmv_fast_impl``), route by route, with each
expert's packed weights read and decoded ONCE for every route sent to it. A verify of ``w`` rows
routes ``8·w`` (row, expert) pairs and consecutive real tokens share experts heavily
(Inkling-2-Mini: 75% / 67% / 46% distinct at w = 2 / 4 / 8), so the routes are sorted by expert
and the first route of each run computes the whole run. Bit-identical to ``mx.gather_qmm`` per
route (same lanes, same K blocks, the same fp32 running sum), so a verify row's expert outputs
ARE the same token's decode outputs.

``fused_moe`` — an Inkling 2 ``SparseMoE`` call of 1–16 rows around it: the routing scores as
single-row products (the stock matmul rounds a verify's scores differently from decode's); top-k,
the route weights and the expert sort in one launch (``fused_route``), the route combine in
another (``combine_routes``). 8 launches per layer visit against the stock 16 (one row) or ~22 (a
verify), and every row bit-identical to the stock module's decode call for that token. Installed
and self-tested by ``inkling_fused`` (kind ``moe``).

``head_f32`` — fp32 logits from the bf16 output head without the stock 1.85 GB fp32 copy of it.

``install_dense_bits`` — opt-in (``--target-bits 8``): the attention projections and the output
head, which the released checkpoint keeps in bf16 and which are two thirds of the bytes a token
reads, quantized to 8-bit at load. NOT the released weights; see its docstring for what was measured.

Plus an exact per-instance install: the expert layers' constant global scale factors computed once
instead of per call (``install_cached_global_scales``). See NOTES "Inkling expert path" for the
cost model (a first route of an expert is memory-bound, a repeat is decode-bound) and the kernel
variants that lost (archived in ``drafts/inkling/``).

The fp4/fp8 types and the qmv tile are adapted from MLX
(``mlx/backend/metal/kernels/fp_quantized.h``, MIT, © 2023 Apple Inc.).
"""

from __future__ import annotations

import types
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

_HEADER = r"""
constant constexpr float F8E4M3_MAX = 448.0f;
constant constexpr float F4E2M1_MAX = 6.0f;

struct fp4_e2m1 {
  operator float16_t() thread {
    half converted = as_type<half>(ushort((bits & 7) << 9));
    converted *= 16384.0;
    return bits & 8 ? -converted : converted;
  }
  operator float() thread { return static_cast<float>(this->operator float16_t()); }
  uint8_t bits;
};

struct fp8_e4m3 {
  operator float16_t() thread {
    uint16_t v = bits & 127;
    uint16_t sign_bit = ((uint16_t)((bits >> 7) & 1)) << 15;
    uint16_t u = (v << 7) | (((v + 1) >> 7) << 14) | sign_bit;
    half converted = as_type<half>(u);
    return converted * 256.0;
  }
  operator float() thread { return static_cast<float>(this->operator float16_t()); }
  uint8_t bits;
};

struct fp8_e8m0 {
  operator float() thread {
    uint32_t out = (bits == 0)
        ? 0x400000
        : (bits == 0xFF ? 0x7FC00000 : (static_cast<uint16_t>(bits) << 23));
    return as_type<float>(out);
  }
  uint8_t bits;
};

template <typename T, int group_size>
static inline T dequantize_scale(uint8_t s) {
  if constexpr (group_size == 16) {
    return T(*(thread fp8_e4m3*)(&s));
  } else {
    return T(*(thread fp8_e8m0*)(&s));
  }
}

template <int bits, typename U = float>
struct Dequantize {
  U operator()(uint8_t x) thread {
    if constexpr (bits == 8) {
      return U(*(thread fp8_e4m3*)(&x));
    } else {
      return U(*(thread fp4_e2m1*)(&x));
    }
  }
};

// One tile of MLX's single-row qmv (2 simdgroups x 4 output rows; a lane owns VPT values of
// every BLOCK-wide slice of K) for V routes that share an expert. Each row's packed values are
// read and decoded once per slice and reused for the V routes; per route the arithmetic is
// fp_qmv_impl's (FAST: fp_qmv_fast_impl's) operation for operation. The products are exact in
// fp32 (a bf16 input times a 2- or 4-bit-mantissa weight), so only two things can round: the
// per-slice sum, kept in MLX's order, and `scale * accum`, which MLX multiplies and THEN adds
// (qdot returns the product) — written as two statements so it cannot contract to an fma.
// (`rs` / `rd`: the V routes' input and output rows; `inv_scale_enc`: the expert's global scale —
// read by the caller, whose index arrays may live in either address space.)
template <typename T, int BITS, int GS, bool FAST, bool HAS_GS, int OUT_N, int V>
inline void run_tile(
    const device T* x, const device uint32_t* w, const device uint8_t* s, device T* out,
    const thread int* rs, const thread int* rd, const float inv_scale_enc,
    const int K, const int expert, const int out_row, const uint lane) {
  constexpr int PACKS = FAST ? 2 : 1;
  constexpr int VPT = (32 / BITS) * PACKS;      // values per lane per slice
  constexpr int BLOCK = VPT * 32;               // slice width
  constexpr int SSTEP = GS / VPT;               // lanes sharing one group scale
  constexpr int LANE_BYTES = VPT * BITS / 8;
  const int kw = K * BITS / 8;                  // packed bytes per weight row
  const int kg = K / GS;                        // scales per weight row
  const device uint8_t* ws = (const device uint8_t*)w
      + size_t(expert) * OUT_N * kw + size_t(out_row) * kw + lane * LANE_BYTES;
  const device uint8_t* sc = s + size_t(expert) * OUT_N * kg + size_t(out_row) * kg + lane / SSTEP;
  const device T* xp[V];
  float result[V][4];
  #pragma unroll
  for (int v = 0; v < V; v++) {
    xp[v] = x + size_t(rs[v]) * K + lane * VPT;
    #pragma unroll
    for (int r = 0; r < 4; r++) result[v][r] = 0.0f;
  }
  for (int k = 0; k < K; k += BLOCK) {
    if (K - k - int(lane) * VPT <= 0) break;    // the tail slice of a K that is not a multiple
    float xt[V][VPT];
    #pragma unroll
    for (int v = 0; v < V; v++) {
      #pragma unroll
      for (int i = 0; i < VPT; i++) xt[v][i] = static_cast<float>(xp[v][k + i]);
    }
    #pragma unroll
    for (int r = 0; r < 4; r++) {
      const float sv = dequantize_scale<float, GS>(sc[r * kg]);
      const device uint8_t* wl = ws + r * kw;
      float d[VPT];
      if constexpr (BITS == 4) {
        const device uint16_t* wq = (const device uint16_t*)wl;
        #pragma unroll
        for (int i = 0; i < VPT / 4; i++) {
          const uint q = wq[i];
          d[4 * i] = Dequantize<4>{}(uint8_t(q));
          d[4 * i + 1] = Dequantize<4>{}(uint8_t(q >> 4));
          d[4 * i + 2] = Dequantize<4>{}(uint8_t(q >> 8));
          d[4 * i + 3] = Dequantize<4>{}(uint8_t(q >> 12));
        }
      } else {
        #pragma unroll
        for (int i = 0; i < VPT; i++) d[i] = Dequantize<8>{}(wl[i]);
      }
      #pragma unroll
      for (int v = 0; v < V; v++) {
        float accum = 0;
        if constexpr (BITS == 4) {
          #pragma unroll
          for (int i = 0; i < VPT / 4; i++) {
            accum += (xt[v][4 * i] * d[4 * i] + xt[v][4 * i + 1] * d[4 * i + 1]
                      + xt[v][4 * i + 2] * d[4 * i + 2] + xt[v][4 * i + 3] * d[4 * i + 3]);
          }
        } else {
          #pragma unroll
          for (int i = 0; i < VPT; i++) accum += xt[v][i] * d[i];
        }
        const float p = sv * accum;
        result[v][r] = result[v][r] + p;
      }
    }
    ws += BLOCK * BITS / 8;
    sc += BLOCK / GS;
  }
  #pragma unroll
  for (int v = 0; v < V; v++) {
    device T* y = out + size_t(rd[v]) * OUT_N + out_row;
    #pragma unroll
    for (int r = 0; r < 4; r++) {
      const float acc = simd_sum(result[v][r]);
      if (lane == 0) {
        if constexpr (HAS_GS) {
          y[r] = static_cast<T>(acc * inv_scale_enc);
        } else {
          y[r] = static_cast<T>(acc);
        }
      }
    }
  }
}
"""

# Grid z = one slot per route, the routes sorted by expert. The first slot of a run of equal
# experts (its leader) computes the whole run, up to RUN_V routes per pass over the weights (a
# longer run takes another pass, from cache); the other slots of the run exit at once. Each run
# length has its own straight-line tile: a tile sized for more routes than it serves costs up
# to 45% (measured), so nothing is padded.
_SOURCE_RUN = r"""
  const uint first = threadgroup_position_in_grid.z;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int R = int(eids_shape[0]);
  const int expert = int(eids[first]);
  if (first > 0 && int(eids[first - 1]) == expert) return;      // not its run's leader
  int end = int(first) + 1;
  while (end < R && int(eids[end]) == expert) ++end;
  const int K = x_shape[x_ndim - 1];
  const int out_row = int(threadgroup_position_in_grid.y) * 8 + int(sg) * 4;
  float inv_scale_enc = 1.0f;
  if (HAS_GS) {
    inv_scale_enc = g[expert] / (F8E4M3_MAX * F4E2M1_MAX);
  }
  int begin = int(first);
  while (begin < end) {
    const int nv = end - begin;
    thread int rs[4];
    thread int rd[4];
    #pragma unroll
    for (int v = 0; v < 4; v++) {
      const int r = min(begin + v, end - 1);
      rs[v] = int(src[r]);
      rd[v] = int(dst[r]);
    }
    if (nv >= 4) {
      run_tile<T, BITS, GS, FAST, HAS_GS, OUT_N, 4>(x, w, s, out, rs, rd, inv_scale_enc, K, expert, out_row, lane);
      begin += 4;
    } else if (nv == 3) {
      run_tile<T, BITS, GS, FAST, HAS_GS, OUT_N, 3>(x, w, s, out, rs, rd, inv_scale_enc, K, expert, out_row, lane);
      begin += 3;
    } else if (nv == 2) {
      run_tile<T, BITS, GS, FAST, HAS_GS, OUT_N, 2>(x, w, s, out, rs, rd, inv_scale_enc, K, expert, out_row, lane);
      begin += 2;
    } else {
      run_tile<T, BITS, GS, FAST, HAS_GS, OUT_N, 1>(x, w, s, out, rs, rd, inv_scale_enc, K, expert, out_row, lane);
      begin += 1;
    }
  }
"""

# M4 Pro, one projection launch (K 2304 x N 1152, 1.49 MB per expert), us: a route whose expert is
# new costs ~6.6 (memory-bound), a repeat through a per-route kernel ~4.4 (decode-bound) and here
# ~0 up to 4 routes a run — 16 pairs 164 -> 117, 8 quads 160 -> 79, 32 distinct 219 -> 209.


@lru_cache(maxsize=1)
def _run_kernel():
    return mx.fast.metal_kernel(
        name="mlx_dspark_inkling_run_qmv",
        input_names=["x", "w", "s", "g", "eids", "src", "dst"],
        output_names=["out"],
        header=_HEADER,
        source=_SOURCE_RUN,
        compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=1)
def _no_factor() -> mx.array:
    z = mx.zeros((8,), mx.float32)
    mx.eval(z)
    return z


_VIEW = "_mlx_dspark_views"


def _bound(linear):
    """The arrays the kernel binds for a layer's packed weights and scales: 8-element views of
    them when :func:`set_expert_views` installed some, else the tensors themselves."""
    return linear.__dict__.get(_VIEW) or (linear.weight, linear.scales)


def set_expert_views(language_model, enabled: bool = True) -> int:
    """Bind every routed-expert tensor to the kernels as an 8-element VIEW of itself (the same
    Metal buffer at offset 0; the kernel indexes the whole tensor from it, as it does anyway).

    Why: MLX commits the command buffer once the arrays bound in it exceed 40 M ELEMENTS
    (``MLX_MAX_MB_PER_BUFFER``; ``mlx/backend/metal/device.cpp``), and one stacked expert tensor
    is 42.5 M (NVFP4) or 85 M (MXFP8) — so every expert launch ended its buffer: 144 commits a
    token, ~1.4 ms of a 31 ms decode step. A view is charged its own 8 elements. Nothing
    process-wide changes (the env knob would also apply to every other model in the server), and
    the weights are persistent, so the limit's purpose — bounding the temporaries one buffer
    holds — is untouched. The views share the tensors' buffers and live on the layer, so they die
    with it. ``enabled=False`` removes them (the A/B). Returns how many projections have one.

    Relies on a basic slice being a view of the same buffer; ``inkling_fused.selftest`` compares
    the kernel's output with ``mx.gather_qmm`` bit for bit AFTER this ran, so if that ever stops
    holding the expert kernels stay off."""
    n = 0
    for layer in language_model.model.layers:
        sw = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
        for name in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(sw, name, None) if sw is not None else None
            if lin is None or layer_spec(lin) is None:
                continue
            if not enabled:
                lin.__dict__.pop(_VIEW, None)
                continue
            if _VIEW not in lin.__dict__:
                # (8, not fewer: a kernel input under 8 elements is passed as a constant)
                views = (lin.weight.reshape(-1)[:8], lin.scales.reshape(-1)[:8])
                mx.eval(views)
                lin.__dict__[_VIEW] = views
            n += 1
    return n


def fast_variant(bits: int, k: int) -> bool:
    """Whether MLX runs its fast single-row qmv for this input width (two packs per lane: K a
    multiple of 512 at 4 bits, 256 at 8), else the general one. The two reduce in different
    orders; ``inkling_fused.selftest`` checks the choice against ``mx.gather_qmm`` at load."""
    return k % ((32 // bits) * 2 * 32) == 0


def run_qmv(x: mx.array, linear, eids: mx.array, src: mx.array, dst: mx.array) -> mx.array:
    """``out[dst[r]] = x[src[r]] @ W[eids[r]]ᵀ`` for every route ``r``, bit-identical to
    ``mx.gather_qmm``'s single-row product per route. Routes with equal ``eids`` next to each
    other share one read of that expert (sort by expert to get it; any order is correct).
    ``x``: ``[rows, K]``; ``eids`` / ``src`` / ``dst``: int32 ``[R]``; returns ``[R, N]`` in
    ``x.dtype``."""
    bits, gsz, _ = layer_spec(linear)
    n = int(linear.weight.shape[-2])
    k = int(linear.scales.shape[-1]) * gsz
    factor = global_factor(linear)
    has_gs = factor is not None
    r = int(eids.shape[0])
    w, sc = _bound(linear)
    return _run_kernel()(
        inputs=[x, w, sc, factor if has_gs else _no_factor(), eids, src, dst],
        template=[("T", x.dtype), ("BITS", bits), ("GS", gsz), ("FAST", fast_variant(bits, k)),
                  ("HAS_GS", has_gs), ("OUT_N", n)],
        grid=(32, (n // 8) * 2, r),
        threadgroup=(32, 2, 1),
        output_shapes=[(r, n)],
        output_dtypes=[x.dtype],
    )[0]


def global_factor(linear) -> mx.array | None:
    """The per-expert ``global_scale`` MLX's gather_qmm receives for this layer (mlx-vlm's
    ``ScaledQuantizedSwitchLinear``: 2688/scale or 2688·scale), or None when it has none. A
    layer given a precomputed factor (:func:`install_cached_global_scales`) returns that."""
    cached = getattr(linear, "_mlx_dspark_factor", None)
    if cached is not None:
        return cached
    gs = getattr(linear, "global_scale", None)
    if gs is None:
        return None
    scale = gs.astype(mx.float32)
    factor = (2688.0 / scale if getattr(linear, "_scale_name", "") == "weight_global_scale"
              else 2688.0 * scale)
    return mx.broadcast_to(factor.reshape(-1), (linear.weight.shape[0],))


def layer_spec(linear) -> tuple | None:
    """``(bits, group_size, mode)`` for a routed-expert layer this kernel can serve, else None."""
    mode = getattr(linear, "mode", None)
    bits, gsz = getattr(linear, "bits", None), getattr(linear, "group_size", None)
    if (mode, bits, gsz) not in (("nvfp4", 4, 16), ("mxfp8", 8, 32)):
        return None
    if ("bias" in linear or linear.get("biases") is not None or "pre_quant_scale" in linear
            or linear.weight.dtype != mx.uint32 or linear.scales.dtype != mx.uint8
            or linear.weight.ndim != 3):
        return None
    k = int(linear.scales.shape[-1]) * gsz
    n = int(linear.weight.shape[-2])
    if n % 8 or k % 32:
        return None
    return bits, gsz, mode


# ---- routing: top-k, route weights and the expert sort in one launch -------------------------------
# mlx-vlm's _route after its score matmul, plus the argsort the run kernel wants:
#   score = sigmoid(logit)                       MLX's Sigmoid: 1 / (1 + exp(|x|)), mirrored
#   top-k of -(score + bias)                     MLX's sort is a stable merge sort, NaN last
#   weight = score / (sum(scores) + 1e-20) * route_scale     the sum in rank order, as the stock
#   routes sorted by expert (ties by route)      row reduce adds them
# One threadgroup: a thread per (row, expert). Every step keeps the stock op's rounding points.
def _route_source(experts: int, top_k: int, rows: int, route_scale: float) -> str:
    return f"""
    constexpr int E = {experts};
    constexpr int TOPK = {top_k};
    constexpr int NR = {rows};
    constexpr int R = NR * TOPK;
    const uint t = thread_position_in_threadgroup.x;
    const uint row = t / E;
    const uint e = t % E;
    threadgroup float tg_key[NR * E];
    threadgroup float tg_score[R];
    threadgroup int tg_expert[R];
    const float logit = logits[t];
    const float y = 1 / (1 + metal::precise::exp(metal::abs(logit)));
    const float score = (logit < 0) ? y : 1 - y;
    const float biased = score + bias[e];
    const float key = -biased;
    tg_key[t] = key;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool kn = metal::isnan(key);
    int rank = 0;
    for (int j = 0; j < E; ++j) {{
        const float other = tg_key[row * E + j];
        const bool on = metal::isnan(other);
        const bool before = (kn | on) ? ((!on) & kn) : (other < key);
        const bool after = (kn | on) ? ((!kn) & on) : (key < other);
        rank += (before || (!after && uint(j) < e)) ? 1 : 0;
    }}
    if (rank < TOPK) {{
        tg_expert[row * TOPK + rank] = int(e);
        tg_score[row * TOPK + rank] = score;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < uint(R)) {{
        const uint rrow = t / TOPK;
        float total = 0;
        for (int j = 0; j < TOPK; ++j) total = tg_score[rrow * TOPK + j] + total;
        const float denom = total + 1e-20f;
        const float share = tg_score[t] / denom;
        wts[t] = share * {route_scale!r}f;
        const int mine = tg_expert[t];
        int pos = 0;
        for (int r = 0; r < R; ++r) {{
            const int other = tg_expert[r];
            pos += (other < mine || (other == mine && uint(r) < t)) ? 1 : 0;
        }}
        eids[pos] = mine;
        src[pos] = int(rrow);
        order[pos] = int(t);
    }}
"""


@lru_cache(maxsize=32)
def _route_kernel(experts: int, top_k: int, rows: int, route_scale: float):
    return mx.fast.metal_kernel(
        name=f"mlx_dspark_inkling_route_{experts}_{top_k}_{rows}",
        input_names=["logits", "bias"],
        output_names=["eids", "src", "order", "wts"],
        compile_options={"math_mode": "safe"},
        source=_route_source(experts, top_k, rows, route_scale),
    )


ROUTE_MAX_THREADS = 1024      # one threadgroup holds every (row, expert) of a call


def fused_route(logits: mx.array, bias: mx.array, top_k: int, route_scale: float):
    """``(eids, src, order, weights)`` from the routing logits ``[rows, E]`` (fp32) and the
    score bias ``[E]`` (fp32): the routes sorted by expert (``eids``), the row each reads
    (``src``), its place in mlx-vlm's ``[row, k]`` route order (``order``), and the route weights
    in that order (``[rows * top_k]``, fp32). The selection and weights are bit-identical to
    mlx-vlm's ``_route``."""
    rows, experts = int(logits.shape[0]), int(logits.shape[1])
    r = rows * top_k
    return _route_kernel(experts, top_k, rows, float(route_scale))(
        inputs=[logits, bias],
        grid=(rows * experts, 1, 1),
        threadgroup=(rows * experts, 1, 1),
        output_shapes=[(r,), (r,), (r,), (r,)],
        output_dtypes=[mx.int32, mx.int32, mx.int32, mx.float32],
    )


# ---- the route combine ------------------------------------------------------------------------------
# mlx-vlm's _combine_routes: (out.astype(fp32) * weight) summed over a row's routes in order, cast
# back (stock: a multiply, a column reduce, a cast).
_SOURCE_COMBINE = r"""
    const uint i = thread_position_in_grid.x;
    const int H = d_shape[1];
    const uint row = i / H;
    const int h = int(i % H);
    if (row * TOPK >= uint(d_shape[0])) return;
    float acc = 0;
    for (int j = 0; j < TOPK; ++j) {
        const size_t r = size_t(row) * TOPK + j;
        const float p = static_cast<float>(d[r * H + h]) * wts[r];
        acc = p + acc;
    }
    out[i] = static_cast<T>(acc);
"""


@lru_cache(maxsize=1)
def _combine_kernel():
    return mx.fast.metal_kernel(
        name="mlx_dspark_inkling_combine_routes",
        input_names=["d", "wts"],
        output_names=["out"],
        compile_options={"math_mode": "safe"},
        source=_SOURCE_COMBINE,
    )


def combine_routes(down: mx.array, weights: mx.array, top_k: int) -> mx.array:
    """``[rows, H]`` from the route outputs ``[rows * top_k, H]`` (mlx-vlm's route order) and
    their weights ``[rows * top_k]``; bit-identical to mlx-vlm's ``_combine_routes``."""
    r, hidden = int(down.shape[0]), int(down.shape[1])
    rows = r // top_k
    return _combine_kernel()(
        inputs=[down, weights],
        template=[("T", down.dtype), ("TOPK", top_k)],
        grid=(rows * hidden, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, hidden)],
        output_dtypes=[down.dtype],
    )[0]


# ---- the MoE ----------------------------------------------------------------------------------------
# Rows a fused call accepts: a decode step, a verify, a prefill's short tail. Longer calls are
# prefill chunks, where mlx-vlm's sorted gather already batches experts.
MOE_MAX_ROWS = 16
_GATE_F32 = "_mlx_dspark_gate_f32"


def moe_spec(moe) -> bool:
    """Whether :func:`fused_moe` can serve this ``SparseMoE`` (checked once, at install)."""
    sw = getattr(moe, "switch_mlp", None)
    projs = [getattr(sw, p, None) for p in ("gate_proj", "up_proj", "down_proj")]
    if sw is None or any(q is None or layer_spec(q) is None for q in projs):
        return False
    gate, bias = getattr(moe, "gate_weight", None), getattr(moe, "e_score_correction_bias", None)
    return (gate is not None and bias is not None and gate.ndim == 2
            and bias.dtype == mx.float32 and bias.shape == (gate.shape[0],)
            and isinstance(getattr(moe, "top_k", None), int)
            and 0 < moe.top_k <= gate.shape[0] <= ROUTE_MAX_THREADS)


def _gate_f32(moe) -> mx.array:
    """The routing matrix in fp32, cast once (stock casts its 128 x H bf16 rows every call)."""
    cached = moe.get(_GATE_F32)
    if cached is None:
        cached = moe.gate_weight.astype(mx.float32)
        mx.eval(cached)
        moe[_GATE_F32] = cached
    return cached


def fused_moe(moe, x: mx.array) -> mx.array:
    """An Inkling 2 ``SparseMoE`` forward for one batch row of 1..``MOE_MAX_ROWS`` positions,
    bit-identical ROW BY ROW to the stock module's single-row (decode) call: the routing scores
    as one single-row product per position, :func:`fused_route`, three :func:`run_qmv`, the stock
    activation, :func:`combine_routes`.

    The stock module is not row-invariant itself: at 2+ rows MLX computes the fp32 score matmul
    with its matrix kernel, whose sums round differently from the single-row kernel's (78% of a
    verify's routing logits differ in the last bit from the same tokens' decode logits). Writing
    the product as matrix @ a batch of columns keeps every row on the single-row kernel, in one
    launch — so a verify row routes, and weighs its routes, exactly as its decode step does."""
    b, n, h = x.shape
    k = moe.top_k
    sw = moe.switch_mlp
    gate = _gate_f32(moe)
    experts = int(gate.shape[0])
    logits = (gate @ x.astype(mx.float32).reshape(n, h, 1)).reshape(n, experts)
    if n * experts <= ROUTE_MAX_THREADS:
        eids, src, order, weights = fused_route(logits, moe.e_score_correction_bias, k,
                                                moe.route_scale)
    else:                 # more rows than one threadgroup holds: the same routing in stock ops
        eids, src, order, weights = _route_ops(logits, moe.e_score_correction_bias, k,
                                               moe.route_scale)
    ident = _arange(n * k)
    xs = x.reshape(n, h)
    up = run_qmv(xs, sw.up_proj, eids, src, ident)
    gate_out = run_qmv(xs, sw.gate_proj, eids, src, ident)
    down = run_qmv(sw.activation(up, gate_out), sw.down_proj, eids, ident, order)
    return combine_routes(down, weights, k).reshape(b, n, h)


def _route_ops(logits: mx.array, bias: mx.array, top_k: int, route_scale: float):
    """:func:`fused_route` in MLX ops (mlx-vlm's ``_route`` after its score matmul, then the
    expert sort), for calls wider than the kernel's one threadgroup."""
    scores = mx.sigmoid(logits)
    indices = mx.argpartition(-(scores + bias), top_k - 1, axis=-1)[..., :top_k]
    weights = mx.take_along_axis(scores, indices, axis=-1)
    weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20) * route_scale
    flat = indices.reshape(-1).astype(mx.int32)
    order = mx.argsort(flat).astype(mx.int32)
    return flat[order], order // top_k, order, weights.reshape(-1)


@lru_cache(maxsize=32)
def _arange(n: int) -> mx.array:
    a = mx.arange(n, dtype=mx.int32)
    mx.eval(a)
    return a


# ---- the output head ----------------------------------------------------------------------------
# Inkling 2's head is a bf16 [V, H] matrix whose logits the reference requires in fp32
# (hf_quant_config: fwd_output_dtype float32). mlx-vlm gets fp32 by promoting: for <= 15 rows it
# keeps a 1.85 GB fp32 copy of the head and reads THAT every step (7.6 ms/step on an M4 Pro, 10 ms
# at 4-8 rows), above it MLX converts the whole matrix per call. This kernel reads the bf16 matrix
# once per call and accumulates in fp32: ~half the traffic, no fp32 copy resident, and — the
# reduction order is per (row, vocab) and independent of how many rows the call carries — a
# verify row's logits are bit-identical to the same token's decode logits.
_SOURCE_HEAD = r"""
  // threadgroup: NSG simdgroups x 4 vocab rows; lane owns VPT=8 consecutive K per 256-wide block
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int K = h_shape[1];
  const int V = w_shape[0];
  const int row0 = int(threadgroup_position_in_grid.y) * (4 * NSG) + int(sg) * 4;
  if (row0 >= V) return;
  float acc[R][4];
  #pragma unroll
  for (int r = 0; r < R; r++) {
    for (int i = 0; i < 4; i++) acc[r][i] = 0.0f;
  }
  const device bfloat16_t* wp[4];
  #pragma unroll
  for (int i = 0; i < 4; i++) wp[i] = w + size_t(min(row0 + i, V - 1)) * K + lane * 8;
  for (int k = 0; k < K; k += 256) {
    float wv[4][8];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      #pragma unroll
      for (int j = 0; j < 8; j++) wv[i][j] = float(wp[i][k + j]);
    }
    #pragma unroll
    for (int r = 0; r < R; r++) {
      const device bfloat16_t* hp = h + size_t(r) * K + k + lane * 8;
      float xv[8];
      #pragma unroll
      for (int j = 0; j < 8; j++) xv[j] = float(hp[j]);
      #pragma unroll
      for (int i = 0; i < 4; i++) {
        float s = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; j++) s = fma(wv[i][j], xv[j], s);
        acc[r][i] += s;
      }
    }
  }
  #pragma unroll
  for (int r = 0; r < R; r++) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      float t = simd_sum(acc[r][i]);
      if (lane == 0 && row0 + i < V) out[size_t(r) * V + row0 + i] = t;
    }
  }
"""

HEAD_MAX_ROWS = 16
HEAD_NSG = 2


@lru_cache(maxsize=1)
def _head_kernel():
    return mx.fast.metal_kernel(
        name="mlx_dspark_inkling_head_f32",
        input_names=["h", "w"],
        output_names=["out"],
        source=_SOURCE_HEAD,
        compile_options={"math_mode": "safe"},
    )


def head_f32(h: mx.array, weight: mx.array) -> mx.array:
    """``h [R, H] (bf16) @ weight[V, H] (bf16)ᵀ`` with fp32 accumulation, fp32 out, ``R <= 16``,
    every row reduced identically whatever ``R`` is. ``H`` must be a multiple of 256."""
    r = int(h.shape[0])
    v = int(weight.shape[0])
    return _head_kernel()(
        inputs=[h, weight],
        template=[("R", r), ("NSG", HEAD_NSG)],
        grid=(32, (-(-v // (4 * HEAD_NSG))) * HEAD_NSG, 1),
        threadgroup=(32, HEAD_NSG, 1),
        output_shapes=[(r, v)],
        output_dtypes=[mx.float32],
    )[0]


def install_fast_head(language_model) -> bool:
    """Route an Inkling 2 language model's fp32 logits (``_logits_from_norm``) through
    :func:`head_f32` for calls of <= 16 rows (decode, verify, a prefill's last row); longer calls
    keep the stock path. Per instance; never builds the stock path's fp32 head copy for short
    calls. Returns whether it was installed."""
    import types

    lm = language_model
    head = getattr(lm, "lm_head", None)
    if lm.__dict__.get(_DENSE_HEAD) is not None:       # the logits go through the 8-bit head
        return False
    if (head is None or type(head).__name__ != "Linear" or "bias" in head
            or head.weight.dtype != mx.bfloat16 or head.weight.shape[1] % 256
            or not mx.metal.is_available()):
        return False
    stock = type(lm)._logits_from_norm

    def _logits_from_norm(self, h):
        dtype = self.model.embed_norm.weight.dtype
        rows = 1
        for d in h.shape[:-1]:
            rows *= int(d)
        if rows > HEAD_MAX_ROWS or self.training or dtype != mx.bfloat16:
            return stock(self, h)
        flat = h.astype(dtype).reshape(rows, h.shape[-1])
        logits = head_f32(flat, self.lm_head.weight).reshape(*h.shape[:-1], -1)
        cap = self.config.final_logit_softcapping
        if cap:
            logits = mx.tanh(logits / cap) * cap
        return logits[..., : self.config.unpadded_vocab_size]

    lm._logits_from_norm = types.MethodType(_logits_from_norm, lm)
    return True


# ---- the bf16 parts at 8-bit (opt-in) -----------------------------------------------------------
# The released checkpoint quantizes only the experts. Per token (Inkling-2-Mini) the rest is the
# five large attention projections of every layer visit, 2.49 GB in bf16, and the output head,
# 0.93 GB, against 1.84 GB of experts — two thirds of the traffic of a decode step.
DENSE_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "gate_proj", "o_proj")
DENSE_BITS = (8,)          # 4 was measured and is not offered (see install_dense_bits)
_DENSE = "_mlx_dspark_dense_bits"
_DENSE_HEAD = "_mlx_dspark_dense_head"


def install_dense_bits(language_model, bits: int = 8, group_size: int = 64) -> int:
    """Quantize IN PLACE (affine, like any ``-8bit`` MLX model) each layer's five large attention
    projections and give the logits an 8-bit copy of the output head. The experts, the small
    position / sink projections, the embeddings and the drafter are untouched, and ``lm_head``
    itself stays (drafters bind the raw head). Returns how many matrices were quantized.

    These are NOT the released weights. Measured on Inkling-2-Mini (M4 Pro, mlx 0.32.4, paired in
    one process; NOTES "Inkling ... quantized in memory"): plain decode 1.26x, a 4-row verify
    48.6 -> 45.4 ms (a quantized product costs per row; the bf16 one is flat), prompt processing
    and DSpark acceptance unchanged. Step by step on the same tokens (4,620 positions) the next
    token differs from the released model's in 1.6%, at released margins of 0.10 median / 0.74
    max (the released model disagrees with itself in 0.8% between a decode step and a verify
    pass). The 8-bit model's own greedy output, judged by the released weights, has 1.56%
    near-tie mismatches against the released model's own 1.06% (4,608 positions): inside the
    token-forcing gate's 0.5-point tolerance, at its edge — close, measurably not identical. 4-bit
    fails that gate (3.65%, 8 large-margin events), so only 8 is accepted.

    The head copy keeps fp32 scales and takes an fp32 input, so the logits stay fp32 as the
    checkpoint's ``fwd_output_dtype`` requires."""
    if bits not in DENSE_BITS:
        raise ValueError(f"target bits must be one of {DENSE_BITS} (got {bits}): 4-bit attention "
                         "measurably changes this model's output")
    lm = language_model
    if lm.__dict__.get(_DENSE):
        if lm.__dict__[_DENSE] != bits:
            raise ValueError("this model's dense weights are already quantized to "
                             f"{lm.__dict__[_DENSE]} bits")
        return 0
    n = 0
    for layer in lm.model.layers:
        attn = getattr(layer, "self_attn", None)
        for name in DENSE_PROJECTIONS:
            lin = getattr(attn, name, None) if attn is not None else None
            if type(lin) is not nn.Linear or lin.weight.shape[1] % group_size:
                continue
            q = nn.QuantizedLinear.from_linear(lin, group_size=group_size, bits=bits)
            mx.eval(q.parameters())
            setattr(attn, name, q)
            n += 1
    head = getattr(lm, "lm_head", None)
    if (type(head) is nn.Linear and "bias" not in head
            and head.weight.shape[1] % group_size == 0):
        qh = nn.QuantizedLinear.from_linear(head, group_size=group_size, bits=bits)
        qh.scales = qh.scales.astype(mx.float32)
        qh.biases = qh.biases.astype(mx.float32)
        mx.eval(qh.parameters())
        lm.__dict__[_DENSE_HEAD] = qh           # the model's own lm_head stays as released

        def _logits_from_norm(self, h):
            dtype = self.model.embed_norm.weight.dtype
            logits = self.__dict__[_DENSE_HEAD](h.astype(dtype).astype(mx.float32))
            cap = self.config.final_logit_softcapping
            if cap:
                logits = mx.tanh(logits / cap) * cap
            return logits[..., : self.config.unpadded_vocab_size]

        lm._logits_from_norm = types.MethodType(_logits_from_norm, lm)
        n += 1
    lm.__dict__[_DENSE] = bits
    mx.clear_cache()                            # hand the replaced bf16 matrices back
    return n


def dense_bits(language_model) -> int:
    """The bits :func:`install_dense_bits` left this model's dense weights at (0 = as released)."""
    return int(language_model.__dict__.get(_DENSE, 0))


# ---- constant per-expert scale factors ------------------------------------------------------------
# mlx-vlm's ScaledQuantizedSwitchLinear recomputes ``2688·global_scale`` (cast, multiply, reshape,
# broadcast) on EVERY call — ~400 of a decode step's ~13k graph primitives, for a constant. Computed
# once and reused here: same values, same gather_qmm call. (Bit-exact; no measurable speed change —
# see NOTES "Inkling-2-Mini ported" on why removing cheap primitives alone does not move decode.)
_CACHED_SCALE_CLASSES: dict[type, type] = {}


def _cached_scale_class(base: type) -> type:
    cls = _CACHED_SCALE_CLASSES.get(base)
    if cls is None:
        def __call__(self, x, indices, sorted_indices=False):
            factor = self._mlx_dspark_factor
            if factor is None or mx.default_device() != mx.gpu:
                return base.__call__(self, x, indices, sorted_indices=sorted_indices)
            out = mx.gather_qmm(x, self.weight, self.scales, rhs_indices=indices, transpose=True,
                                group_size=self.group_size, bits=self.bits, mode=self.mode,
                                global_scale=factor, sorted_indices=sorted_indices)
            if "bias" in self:
                out = out + self.bias[indices][..., None, :]
            return out

        cls = type(f"CachedScale{base.__name__}", (base,), {"__call__": __call__})
        _CACHED_SCALE_CLASSES[base] = cls
    return cls


def install_cached_global_scales(language_model) -> int:
    """Precompute each routed-expert layer's global-scale factor once (per instance; only the
    GPU branch of mlx-vlm's ScaledQuantizedSwitchLinear, whose conditions are checked here once).
    Returns how many projections took it."""
    n = 0
    for layer in language_model.model.layers:
        sw = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
        for name in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(sw, name, None) if sw is not None else None
            gs = getattr(lin, "global_scale", None)
            if (lin is None or gs is None or "pre_quant_scale" in lin
                    or gs.size not in (1, lin.weight.shape[0])
                    or getattr(lin, "_mlx_dspark_factor", None) is not None):
                continue
            factor = global_factor(lin)
            mx.eval(factor)
            lin._mlx_dspark_factor = factor
            base = type(lin)
            lin.__class__ = _cached_scale_class(base)
            n += 1
    return n
