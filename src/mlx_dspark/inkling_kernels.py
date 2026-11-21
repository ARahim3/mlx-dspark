"""Metal kernels for Inkling 2's verify and logits (Thinking Machines Inkling-2-Mini).

``grouped_qmv`` — a batch-1 verify's routed-expert projections with each expert's quant groups
dequantized ONCE for every draft row routed to it. A verify of ``w`` rows routes ``8·w`` (row,
expert) pairs; MLX's ``gather_qmm`` runs one matrix-vector product per pair below its matrix-kernel
threshold (>= 4 rows per expert on average, i.e. >= 512 rows here). Consecutive real tokens share
experts heavily (Inkling-2-Mini: 79% / 62% / 47% distinct at w = 2 / 4 / 8), so the routes are sorted
by expert and each run of equal experts is computed together in MLX's own small-batch layout
(``fp_qmv_wide_impl``). Row-invariant: a route's result does not depend on its run.

``head_f32`` — fp32 logits from the bf16 output head without the stock 1.85 GB fp32 copy of it.

Plus an exact per-instance install: the expert layers' constant global scale factors computed once
instead of per call (``install_cached_global_scales``). See NOTES "Inkling-2-Mini ported" for the
measurements and the exact-but-slower kernel variants that lost (archived in
``drafts/inkling/inkling_kernels_with_exact_dedup.py``).

The fp4/fp8 types are adapted from MLX (``mlx/backend/metal/kernels/fp_quantized.h``, MIT,
© 2023 Apple Inc.) via mlx-vlm's ``masked_quantized.py`` (MIT).
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

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
"""

# Row-invariant grouped "wide" qmv (MLX's fp_qmv_wide_impl layout, adapted to expert runs).
# Each thread owns one output row whose K reduction is split over KL lanes, lane j taking quant
# groups j, j+KL, ...; a quant group is dequantized once into float4 registers and dot()'d against
# every route of the run (streamed V at a time); the KL partials combine by a fixed shuffle
# ladder; the per-expert global scale and the cast come last. A route's arithmetic never depends
# on which other routes share its expert, so a verify row is bit-identical to the same token's
# decode row when decode runs this kernel too (one route per run).
_SOURCE_WIDE = r"""
  const uint first = threadgroup_position_in_grid.z;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int R = int(eids_shape[0]);
  const int expert = int(eids[first]);
  if (first > 0 && int(eids[first - 1]) == expert) return;      // not its run's leader
  int end = int(first) + 1;
  while (end < R && int(eids[end]) == expert) ++end;

  const int K = x_shape[x_ndim - 1];
  constexpr int ROWS_SG = 32 / KL;           // output rows per simdgroup
  constexpr int NF4 = GS / 4;                // float4 lanes per quant group
  const short k_lane = lane % KL;
  const short sg_row = lane / KL;
  const int out_row = int(threadgroup_position_in_grid.y) * (ROWS_SG * NSG) + ROWS_SG * int(sg) + sg_row;
  const int kw = K * BITS / 8;
  const int kg = K / GS;
  const int wr = min(out_row, OUT_N - 1);     // a tile past N reads a valid row, never writes
  const device uint8_t* wrow = (const device uint8_t*)w + size_t(expert) * OUT_N * kw + size_t(wr) * kw;
  const device uint8_t* srow = s + size_t(expert) * OUT_N * kg + size_t(wr) * kg;
  float inv_scale_enc = 1.0f;
  if (HAS_GS) {
    inv_scale_enc = g[expert] / (F8E4M3_MAX * F4E2M1_MAX);
  }

  for (int begin = int(first); begin < end; begin += V) {
    const int nv = min(V, end - begin);
    const device T* xv[V];
    #pragma unroll
    for (int v = 0; v < V; v++) {
      xv[v] = x + size_t(src[begin + min(v, nv - 1)]) * K;
    }
    float result[V];
    #pragma unroll
    for (int v = 0; v < V; v++) result[v] = 0.0f;

    for (int gi = k_lane; gi < kg; gi += KL) {
      const int k0 = gi * GS;
      float sc = dequantize_scale<float, GS>(srow[gi]);
      const device uint8_t* wg = wrow + k0 * BITS / 8;
      float4 w4[NF4];
      if (BITS == 4) {
        const device uint16_t* wq = (const device uint16_t*)wg;
        #pragma unroll
        for (int i = 0; i < NF4; i++) {
          w4[i] = float4(Dequantize<4>{}(wq[i]), Dequantize<4>{}(wq[i] >> 4),
                         Dequantize<4>{}(wq[i] >> 8), Dequantize<4>{}(wq[i] >> 12));
        }
      } else {
        #pragma unroll
        for (int i = 0; i < NF4; i++) {
          w4[i] = float4(Dequantize<8>{}(wg[4 * i]), Dequantize<8>{}(wg[4 * i + 1]),
                         Dequantize<8>{}(wg[4 * i + 2]), Dequantize<8>{}(wg[4 * i + 3]));
        }
      }
      #pragma unroll
      for (int v = 0; v < V; v++) {
        if (v < nv) {
          const device vec<T, 4>* x4 = (const device vec<T, 4>*)(xv[v] + k0);
          float acc = 0;
          #pragma unroll
          for (int j = 0; j < NF4; j++) {
            acc += dot(w4[j], float4(x4[j]));
          }
          result[v] += sc * acc;
        }
      }
    }
    #pragma unroll
    for (int v = 0; v < V; v++) {
      if (KL >= 32) result[v] += simd_shuffle_down(result[v], 16);
      if (KL >= 16) result[v] += simd_shuffle_down(result[v], 8);
      if (KL >= 8) result[v] += simd_shuffle_down(result[v], 4);
      if (KL >= 4) result[v] += simd_shuffle_down(result[v], 2);
      if (KL >= 2) result[v] += simd_shuffle_down(result[v], 1);
    }
    if (k_lane == 0 && out_row < OUT_N) {
      for (int v = 0; v < nv; v++) {
        out[size_t(dst[begin + v]) * OUT_N + out_row] = static_cast<T>(result[v] * inv_scale_enc);
      }
    }
  }
"""

V_MAX = 8   # routes computed per pass over an expert's weights (a longer run takes a 2nd pass)
# Tuned on an M4 Pro, Inkling-2-Mini, 48 chained MoE visits of real hidden states (ms, stock →
# grouped): w=2 24.0 → 21.7, w=4 43.3 → 32.8, w=8 77.1 → 51.7; KL 4 beat 8 / 16, NSG 4 ≈ 2.
# At w=1 the stock qmv wins (12.1 vs 15.4), so a single row keeps the stock path.
WIDE_KL = 4     # lanes per output row's K reduction
WIDE_NSG = 2    # simdgroups per threadgroup


@lru_cache(maxsize=1)
def _wide_kernel():
    return mx.fast.metal_kernel(
        name="mlx_dspark_inkling_grouped_wide_qmv",
        input_names=["x", "w", "s", "g", "eids", "src", "dst"],
        output_names=["out"],
        header=_HEADER,
        source=_SOURCE_WIDE,
        compile_options={"math_mode": "safe"},
    )


def grouped_qmv(x: mx.array, linear, eids: mx.array, src: mx.array, dst: mx.array,
                kl: int = WIDE_KL, nsg: int = WIDE_NSG, v: int = V_MAX) -> mx.array:
    """Row-invariant grouped projection: ``out[dst[r]] = x[src[r]] @ W[eids[r]]ᵀ`` for every
    route ``r`` (``eids`` sorted ascending), each expert's quant groups dequantized once per run
    of equal ``eids`` and reused for every route in it. ``x``: ``[rows, K]``; returns
    ``[len(eids), N]`` in ``x.dtype``. MLX's wide-qmv arithmetic: within rounding of the stock
    qmv, and every route's result independent of its run."""
    bits, gsz, _ = layer_spec(linear)
    n = int(linear.weight.shape[-2])
    factor = global_factor(linear)
    has_gs = factor is not None
    g = factor if has_gs else mx.zeros((1,), mx.float32)
    r = int(eids.shape[0])
    rows_tg = (32 // kl) * nsg
    return _wide_kernel()(
        inputs=[x, linear.weight, linear.scales, g, eids, src, dst],
        template=[("T", x.dtype), ("BITS", bits), ("GS", gsz), ("HAS_GS", has_gs),
                  ("OUT_N", n), ("V", v), ("KL", kl), ("NSG", nsg)],
        grid=(32, (-(-n // rows_tg)) * nsg, r),
        threadgroup=(32, nsg, 1),
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


# ---- the verify MoE ---------------------------------------------------------------------------
# Rows from which (and up to which) a SparseMoE call takes the grouped path: a verify (2..16
# rows). One row is a decode step — the stock qmv is faster there; longer calls are prefill
# chunks, where mlx-vlm's sorted gather already batches experts.
GROUPED_MOE_MIN_ROWS = 2
GROUPED_MOE_MAX_ROWS = 16


def grouped_moe(moe, x: mx.array) -> mx.array:
    """An Inkling 2 ``SparseMoE`` forward for a short multi-row call: the reference routing and
    route combine (mlx-vlm's own compiled ``_route`` / ``_combine_routes``), with the three expert
    projections through :func:`grouped_qmv` over the expert-sorted routes."""
    import sys

    lang = sys.modules[type(moe).__mro__[1].__module__]
    indices, weights = lang._route(x, moe.gate_weight, moe.e_score_correction_bias, moe.top_k,
                                   moe.route_scale)
    b, n, h = x.shape
    k = moe.top_k
    flat = indices.reshape(-1).astype(mx.int32)
    order = mx.argsort(flat).astype(mx.int32)
    eids = flat[order]
    src = order // k
    ident = mx.arange(n * k, dtype=mx.int32)
    xs = x.reshape(n, h)
    sw = moe.switch_mlp
    up = grouped_qmv(xs, sw.up_proj, eids, src, ident)
    gate = grouped_qmv(xs, sw.gate_proj, eids, src, ident)
    down = grouped_qmv(sw.activation(up, gate), sw.down_proj, eids, ident, order)
    return lang._combine_routes(down.reshape(b, n, k, h), weights)


_GROUPED_CLASSES: dict[type, type] = {}


def _grouped_class(base: type) -> type:
    cls = _GROUPED_CLASSES.get(base)
    if cls is None:
        def __call__(self, x):
            if (self._mlx_dspark_grouped and x.ndim == 3 and x.shape[0] == 1
                    and GROUPED_MOE_MIN_ROWS <= x.shape[1] <= GROUPED_MOE_MAX_ROWS
                    and getattr(self, "sharding_group", None) is None):
                return grouped_moe(self, x)
            return base.__call__(self, x)

        cls = type(f"Grouped{base.__name__}", (base,), {"__call__": __call__})
        _GROUPED_CLASSES[base] = cls
    return cls


def install_grouped_moe(language_model, enabled: bool = True) -> int:
    """Give every routed-expert layer of an Inkling 2 language model the grouped verify path
    (a per-INSTANCE class swap: nothing global is patched, and other models are untouched).
    ``enabled=False`` keeps the swap but routes every call to the stock forward (the A/B).
    Returns the number of layers that took it."""
    n = 0
    for layer in language_model.model.layers:
        moe = getattr(layer, "mlp", None)
        sw = getattr(moe, "switch_mlp", None)
        projs = [getattr(sw, p, None) for p in ("gate_proj", "up_proj", "down_proj")]
        if sw is None or any(q is None or layer_spec(q) is None for q in projs):
            continue
        base = type(moe)
        if base.__name__.startswith("Grouped"):
            base = base.__mro__[1]
        moe.__class__ = _grouped_class(base)
        moe._mlx_dspark_grouped = bool(enabled)
        n += 1
    return n


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
