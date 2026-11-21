"""Fused forward pieces for Inkling 2 (Thinking Machines Inkling-2-Mini) at decode / verify widths.

One decode step of the stock mlx-vlm forward launches ~7,000 GPU kernels for 48 layer visits, most
of them a few thousand elements wide: about half of a step's time is the launches, not the weights
(NOTES "Inkling fused forward"). Each piece here replaces a run of small stock operations with one
Metal kernel, for calls of one batch row and at most ``MAX_ROWS`` positions (a decode step or a
speculative verify); anything else — prefill chunks, the CPU, training — takes the stock path.

Installed per INSTANCE (a class swap on the model's own modules, like ``inkling_kernels``): nothing
global is patched and other models are untouched. ``set_enabled(lm, False)`` routes every call back
to the stock forward (the A/B); ``MLX_DSPARK_INKLING_FUSED=0`` keeps it off from the start.

``fused_short_convolution`` — the depthwise 4-tap convolution AND its next cache state in one
launch (stock: a kernel, a zeros fill, a cast and a concatenate). Bit-identical to stock: the
output expression is mlx-vlm's own (``models/inkling2/kernels.py``, MIT), the state is a copy.

``fused_attention_residual`` — the learned softmax mix over the residual blocks (stock: stack,
cast, RMS norm, two matmuls, a softmax: ~11 launches, and ~49 us at 4 rows because the tiny
matmuls fall on MLX's tiled GEMM). ``fused_controller`` — a loop pass's message controller
(stock: two RMS norms, a concatenate, three matmuls, casts, tanh, blend: 16 launches; here two). Both keep
the stock arithmetic's rounding POINTS: the squared sums reduce in the order of MLX's own
``rms_single_row`` (``mlx/backend/metal/kernels/rms_norm.metal``, MIT, © 2024 Apple Inc.), the
controller's rank-64 code is accumulated in fp32 and rounded to bf16 exactly where MLX's bf16
matmul rounds it. What differs is the summation order of fp32 dot products — the fp-tie class.

``moe`` — the routed experts (``inkling_kernels.fused_moe``): top-k, route weights and the expert
sort in one launch, MLX's own single-row quantized matvec per route with each expert's weights
read once per run of routes, the combine in one launch. For 1–16 positions, each row bit-identical
to the stock module's single-row call (the stock multi-row call rounds its routing scores
differently from its own single-row call; this one does not).
"""

from __future__ import annotations

import math
from functools import lru_cache

import mlx.core as mx

MAX_ROWS = 8          # positions per call a fused piece accepts (the widest verify: cap 7 + anchor)
_FLAG = "_mlx_dspark_fused"


def _gpu() -> bool:
    return mx.metal.is_available() and mx.default_device() == mx.gpu


def _on(module) -> bool:
    return bool(module.__dict__.get(_FLAG, False))


# ---- short convolution + next state -------------------------------------------------------------
# out[t, c]    = (sum_j ctx[t + j, c] * w[c, j] + x[t, c]) / sqrt(W + 1), ctx = [state ‖ x] (virtual)
# window[s, c] = ctx[s, c] in fp32 — the stock path's concatenate; its last W-1 rows are the next
#                state, and a speculative verify records the whole window for rollback.
def _sconv_source(res: bool) -> str:
    add = "conv = conv + float(res[t * res_strides[1] + c * res_strides[2]]);" if res else ""
    return r"""
    uint i = thread_position_in_grid.x;
    int L = x_shape[1], C = x_shape[2];
    if (i >= uint(L * C)) return;
    int c = i % C, t = i / C;
    float value = 0;
    for (int j = 0; j < W; ++j) {
        int p = t + j - W + 1;
        float xv = p < 0
            ? float(state[(p + W - 1) * state_strides[1] + c * state_strides[2]])
            : float(x[p * x_strides[1] + c * x_strides[2]]);
        float w = weight[c * weight_strides[0] + j * weight_strides[1]];
        value = value + xv * w;
    }
    float xt = float(x[t * x_strides[1] + c * x_strides[2]]);
    float conv = float(T((value + xt) / divisor));
    """ + add + r"""
    out[i] = OT(conv);
    window[(W - 1 + t) * C + c] = xt;
    if (t == 0) {
        for (int s = 0; s < W - 1; ++s) {
            window[s * C + c] = float(state[s * state_strides[1] + c * state_strides[2]]);
        }
    }
"""


@lru_cache(maxsize=2)
def _sconv_kernel(res: bool):
    return mx.fast.metal_kernel(
        name="mlx_dspark_inkling_sconv_state" + ("_res" if res else ""),
        input_names=["state", "weight", "x", "divisor"] + (["res"] if res else []),
        output_names=["out", "window"],
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
        source=_sconv_source(res),
    )


@lru_cache(maxsize=8)
def _divisor(width: int) -> mx.array:
    d = mx.array(math.sqrt(width + 1), mx.float32)
    mx.eval(d)
    return d


def fused_short_convolution(state: mx.array, weight: mx.array, x: mx.array, width: int,
                            res: mx.array | None = None, out_dtype=None):
    """``(out, window)`` for one batch row: the convolution output (``x``'s dtype, as stock) and
    the fp32 ``[1, width - 1 + L, C]`` concatenation ``[state ‖ x]`` whose tail is the next state.

    ``res`` / ``out_dtype`` fold the decoder layer's residual arithmetic into the launch, at the
    stock rounding points: the convolution is rounded to ``x``'s dtype, widened to fp32, ``res``
    is added in fp32, and the sum is stored as ``out_dtype`` (default: ``x``'s dtype)."""
    _, length, channels = x.shape
    out_dtype = out_dtype or x.dtype
    inputs = [state, weight, x, _divisor(width)] + ([res] if res is not None else [])
    return _sconv_kernel(res is not None)(
        inputs=inputs,
        template=[("T", x.dtype), ("OT", out_dtype), ("W", width)],
        grid=(length * channels, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape, (1, width - 1 + length, channels)],
        output_dtypes=[out_dtype, mx.float32],
    )


def _sconv_ok(conv, x, cache, mask) -> bool:
    return (cache is not None and mask is None and not conv.training and conv.kernel_size == 4
            and x.ndim == 3 and x.shape[0] == 1 and x.shape[1] <= MAX_ROWS
            and x.dtype == mx.bfloat16 and x.shape[-1] % 128 == 0 and _gpu())


def _sconv_step(conv, x, cache, res=None, out_dtype=None):
    """One fused convolution on the layer's conv cache (the stock ``_update_conv_cache`` rules:
    the next state is the window's tail; a speculative verify records the whole window)."""
    k, idx = conv.kernel_size, conv.conv_idx
    state = cache[idx]
    if state is None:
        state = mx.zeros((1, k - 1, x.shape[-1]))
    out, window = fused_short_convolution(state, conv.conv.weight, x, k, res, out_dtype)
    if getattr(cache, "is_speculating", False):
        cache.record_speculative_window(idx, window, k - 1)
    cache[idx] = window[:, -(k - 1):]
    return out


def _sconv_call(base):
    def __call__(self, x, cache=None, mask=None):
        if not (_on(self) and _sconv_ok(self, x, cache, mask)):
            return base.__call__(self, x, cache, mask)
        return _sconv_step(self, x, cache)

    return __call__


# ---- attention residual -------------------------------------------------------------------------
# One threadgroup per position, one thread per NR = 4 hidden elements (MLX's RMS_N_READS, so the
# sum of squares reduces exactly as mx.fast.rms_norm's):
#   keys_i = x_i * rsqrt(mean(x_i^2) + eps)                 (fp32, per element — as rms_norm)
#   w_qi   = softmax_i(keys_i . query_q * H^-0.5)
#   out_q  = sum_i w_qi * keys_i
_NR = 4
_RES_EPS = 1e-6           # mlx-vlm's _attention_residual passes this literal


def _res_source(n: int, q: int, hidden: int) -> str:
    load = "\n".join(f"    for (int j = 0; j < NR; ++j) xs[{i}][j] = float(s{i}[base + j]);"
                     for i in range(n))
    return f"""
    constexpr int NR = {_NR};
    constexpr int N = {n};
    constexpr int Q = {q};
    constexpr int H = {hidden};
    constexpr int NSG = H / (NR * 32);
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint row = threadgroup_position_in_grid.x;
    const size_t base = size_t(row) * H + lid * NR;
    threadgroup float tg_part[N * Q * 32];
    threadgroup float tg_norm[N];
    threadgroup float tg_w[N * Q];
    float xs[N][NR];
{load}
    // 1. per-state sum of squares (rms_single_row's reduction)
    float ss[N];
    for (int i = 0; i < N; ++i) {{
        float acc = 0;
        for (int j = 0; j < NR; ++j) acc += xs[i][j] * xs[i][j];
        ss[i] = simd_sum(acc);
    }}
    if (sg == 0) {{
        for (int i = 0; i < N; ++i) tg_part[i * 32 + lane] = 0;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {{
        for (int i = 0; i < N; ++i) tg_part[i * 32 + sg] = ss[i];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {{
        for (int i = 0; i < N; ++i) {{
            float t = simd_sum(tg_part[i * 32 + lane]);
            if (lane == 0) tg_norm[i] = metal::precise::rsqrt(t / float(H) + {_RES_EPS!r}f);
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 2. scores and their softmax over the states
    float keys[N][NR];
    for (int i = 0; i < N; ++i) {{
        const float inv = tg_norm[i];
        for (int j = 0; j < NR; ++j) keys[i][j] = xs[i][j] * inv;
    }}
    float d[N * Q];
    for (int qi = 0; qi < Q; ++qi) {{
        float qv[NR];
        for (int j = 0; j < NR; ++j) qv[j] = float(query[size_t(qi) * H + lid * NR + j]);
        for (int i = 0; i < N; ++i) {{
            float acc = 0;
            for (int j = 0; j < NR; ++j) acc += keys[i][j] * qv[j];
            d[i * Q + qi] = simd_sum(acc);
        }}
    }}
    if (sg == 0) {{
        for (int k = 0; k < N * Q; ++k) tg_part[k * 32 + lane] = 0;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {{
        for (int k = 0; k < N * Q; ++k) tg_part[k * 32 + sg] = d[k];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {{
        float sc[N * Q];
        for (int k = 0; k < N * Q; ++k) sc[k] = simd_sum(tg_part[k * 32 + lane]) * {hidden ** -0.5!r}f;
        if (lane == 0) {{
            for (int qi = 0; qi < Q; ++qi) {{
                float top = sc[qi];
                for (int i = 1; i < N; ++i) top = max(top, sc[i * Q + qi]);
                float e[N];
                float z = 0;
                for (int i = 0; i < N; ++i) {{
                    e[i] = metal::precise::exp(sc[i * Q + qi] - top);
                    z += e[i];
                }}
                for (int i = 0; i < N; ++i) tg_w[i * Q + qi] = e[i] / z;
            }}
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 3. the mix
    for (int qi = 0; qi < Q; ++qi) {{
        for (int j = 0; j < NR; ++j) {{
            float acc = 0;
            for (int i = 0; i < N; ++i) acc += tg_w[i * Q + qi] * keys[i][j];
            out[(size_t(row) * Q + qi) * H + lid * NR + j] = acc;
        }}
    }}
"""


@lru_cache(maxsize=64)
def _res_kernel(n: int, q: int, hidden: int):
    return mx.fast.metal_kernel(
        name=f"mlx_dspark_inkling_attn_residual_{n}_{q}_{hidden}",
        input_names=[f"s{i}" for i in range(n)] + ["query"],
        output_names=["out"],
        compile_options={"math_mode": "safe"},
        source=_res_source(n, q, hidden),
    )


def _row_group(hidden: int) -> int | None:
    """Threads of the one-threadgroup-per-position layout, or None when ``hidden`` does not fit it."""
    group = hidden // _NR
    return group if hidden % (_NR * 32) == 0 and group <= 1024 else None


def fused_attention_residual(states: list[mx.array], query: mx.array) -> mx.array:
    """mlx-vlm's ``_attention_residual`` with ``normalize_values``: ``states`` are ``[1, R, H]``
    (bf16 or fp32, oldest block first, the partial block last), ``query`` is ``[H]`` or
    ``[Q, H]``; returns fp32 ``[1, R, Q, H]``."""
    _, rows, hidden = states[-1].shape
    q = 1 if query.ndim == 1 else int(query.shape[0])
    group = _row_group(hidden)
    return _res_kernel(len(states), q, hidden)(
        inputs=[*states, query],
        grid=(rows * group, 1, 1),
        threadgroup=(group, 1, 1),
        output_shapes=[(1, rows, q, hidden)],
        output_dtypes=[mx.float32],
    )[0]


def _res_call(base):
    def __call__(self, blocks, partial):
        if not (_on(self) and partial.ndim == 3 and partial.shape[0] == 1
                and partial.shape[1] <= MAX_ROWS and len(blocks) < 8 and not self.training
                and _row_group(partial.shape[-1]) is not None and _gpu()
                and all(s.shape == partial.shape and s.dtype in (mx.bfloat16, mx.float32)
                        for s in (*blocks, partial))):
            return base.__call__(self, blocks, partial)
        return fused_attention_residual([*blocks, partial], self.query)

    return __call__


# ---- message controller -------------------------------------------------------------------------
#   pair   = bf16([rms(x) ‖ rms(message)])
#   hidden = bf16(pair @ down^T)            fp32 accumulate, ONE rounding: MLX's bf16 matmul result
#   out    = (1 - tanh(hidden @ forget^T)) * x + tanh(hidden @ gain^T) * message          (fp32)
# Two launches, each spread over many threadgroups (one threadgroup per position keeps the whole
# controller on a single GPU core: 89 us against the stock chain's 38). The code kernel gives every
# CODE_GROUP hidden units their own threadgroup, each recomputing the two norms (cheap, and
# identical everywhere); the blend kernel is one thread per output element. bf16 weights are read
# four at a time as raw bits (a bf16 is the top half of its fp32).
_CODE_GROUP = 8


def _ctrl_code_source(hidden: int, rank: int, eps: float) -> str:
    return f"""
    constexpr int NR = {_NR};
    constexpr int H = {hidden};
    constexpr int RANK = {rank};
    constexpr int G = {_CODE_GROUP};
    constexpr int NSG = H / (NR * 32);
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tg = threadgroup_position_in_grid.x;
    const uint row = tg / (RANK / G);
    const int r0 = int(tg % (RANK / G)) * G;
    const int h0 = int(lid) * NR;
    threadgroup float tg_sq[2 * 32];
    threadgroup float tg_norm[2];
    threadgroup float tg_part[G * 32];
    const float4 xv = *(const device packed_float4*)(x + size_t(row) * x_strides[x_ndim - 2] + h0);
    const float4 mv = *(const device packed_float4*)(message + size_t(row) * message_strides[message_ndim - 2] + h0);
    // 1. the two sums of squares (rms_single_row's reduction)
    float ax = 0, am = 0;
    for (int j = 0; j < NR; ++j) ax += xv[j] * xv[j];
    for (int j = 0; j < NR; ++j) am += mv[j] * mv[j];
    ax = simd_sum(ax);
    am = simd_sum(am);
    if (sg == 0) {{
        tg_sq[lane] = 0;
        tg_sq[32 + lane] = 0;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {{
        tg_sq[sg] = ax;
        tg_sq[32 + sg] = am;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {{
        float tx = simd_sum(tg_sq[lane]);
        float tm = simd_sum(tg_sq[32 + lane]);
        if (lane == 0) {{
            tg_norm[0] = metal::precise::rsqrt(tx / float(H) + {eps!r}f);
            tg_norm[1] = metal::precise::rsqrt(tm / float(H) + {eps!r}f);
        }}
        if (lane >= NSG) {{
            for (int g = 0; g < G; ++g) tg_part[g * 32 + lane] = 0;
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 2. G units of the code
    float px[NR], pm[NR];
    for (int j = 0; j < NR; ++j) {{
        px[j] = float(bfloat16_t(xv[j] * tg_norm[0]));
        pm[j] = float(bfloat16_t(mv[j] * tg_norm[1]));
    }}
    const device uint16_t* dbits = (const device uint16_t*)down;
    for (int g = 0; g < G; ++g) {{
        const device uint16_t* dr = dbits + size_t(r0 + g) * (2 * H) + h0;
        const float4 wx = as_type<float4>(uint4(*(const device packed_ushort4*)dr) << 16);
        const float4 wm = as_type<float4>(uint4(*(const device packed_ushort4*)(dr + H)) << 16);
        float acc = 0;
        for (int j = 0; j < NR; ++j) acc += px[j] * wx[j];
        for (int j = 0; j < NR; ++j) acc += pm[j] * wm[j];
        acc = simd_sum(acc);
        if (lane == 0) tg_part[g * 32 + sg] = acc;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {{
        for (int g = 0; g < G; ++g) {{
            float t = simd_sum(tg_part[g * 32 + lane]);
            if (lane == 0) hid[size_t(row) * RANK + r0 + g] = float(bfloat16_t(t));
        }}
    }}
"""


def _ctrl_blend_source(hidden: int, rank: int) -> str:
    return f"""
    constexpr int H = {hidden};
    constexpr int RANK = {rank};
    const uint i = thread_position_in_grid.x;
    const uint row = i / H;
    const int h = int(i % H);
    if (row >= uint(hid_shape[0])) return;
    const float xv = x[size_t(row) * x_strides[x_ndim - 2] + h];
    const float mv = message[size_t(row) * message_strides[message_ndim - 2] + h];
    const device float* hr = hid + size_t(row) * RANK;
    const device packed_ushort4* fp = (const device packed_ushort4*)((const device uint16_t*)forget + size_t(h) * RANK);
    const device packed_ushort4* gp = (const device packed_ushort4*)((const device uint16_t*)gain + size_t(h) * RANK);
    float4 fa = 0, ga = 0;
    for (int r = 0; r < RANK / 4; ++r) {{
        const float4 hv = *(const device packed_float4*)(hr + 4 * r);
        fa += hv * as_type<float4>(uint4(fp[r]) << 16);
        ga += hv * as_type<float4>(uint4(gp[r]) << 16);
    }}
    const float fo = metal::precise::tanh((fa.x + fa.y) + (fa.z + fa.w));
    const float gg = metal::precise::tanh((ga.x + ga.y) + (ga.z + ga.w));
    const float keep = 1.0f - fo;
    const float a = keep * xv;
    const float b = gg * mv;
    out[i] = a + b;
"""


@lru_cache(maxsize=64)
def _ctrl_kernels(hidden: int, rank: int, eps: float):
    code = mx.fast.metal_kernel(
        name=f"mlx_dspark_inkling_controller_code_{hidden}_{rank}",
        input_names=["x", "message", "down"],
        output_names=["hid"],
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
        source=_ctrl_code_source(hidden, rank, eps),
    )
    blend = mx.fast.metal_kernel(
        name=f"mlx_dspark_inkling_controller_blend_{hidden}_{rank}",
        input_names=["x", "message", "hid", "forget", "gain"],
        output_names=["out"],
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
        source=_ctrl_blend_source(hidden, rank),
    )
    return code, blend


def fused_controller(x: mx.array, message: mx.array, down: mx.array, forget: mx.array,
                     gain: mx.array, eps: float) -> mx.array:
    """mlx-vlm's ``MessageController`` for fp32 ``x`` / ``message`` of ``[1, R, H]`` (row views
    are fine: read through their row stride), bf16 weights ``down [rank, 2H]``, ``forget`` /
    ``gain`` ``[H, rank]``; returns fp32 ``[1, R, H]``."""
    rows, hidden, rank = int(x.shape[-2]), int(x.shape[-1]), int(down.shape[0])
    group = _row_group(hidden)
    code, blend = _ctrl_kernels(hidden, rank, float(eps))
    hid = code(
        inputs=[x, message, down],
        grid=(rows * (rank // _CODE_GROUP) * group, 1, 1),
        threadgroup=(group, 1, 1),
        output_shapes=[(rows, rank)],
        output_dtypes=[mx.float32],
    )[0]
    return blend(
        inputs=[x, message, hid, forget, gain],
        grid=(rows * hidden, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, rows, hidden)],
        output_dtypes=[mx.float32],
    )[0]


def _ctrl_call(base):
    def __call__(self, x, message):
        if not (_on(self) and x.ndim == 3 and x.shape[0] == 1 and x.shape[1] <= MAX_ROWS
                and message.shape == x.shape and x.dtype == mx.float32
                and message.dtype == mx.float32 and not self.training
                and self.down_weight.dtype == mx.bfloat16
                and self.forget_weight.dtype == mx.bfloat16
                and self.gain_weight.dtype == mx.bfloat16
                and self.down_weight.shape[1] == 2 * x.shape[-1]
                and self.down_weight.shape[0] % _CODE_GROUP == 0
                and _row_group(x.shape[-1]) is not None and _gpu()):
            return base.__call__(self, x, message)
        return fused_controller(x, message, self.down_weight, self.forget_weight,
                                self.gain_weight, self.eps)

    return __call__


# ---- RMS norm + cast ----------------------------------------------------------------------------
# nn.RMSNorm on an fp32 stream with a bf16 weight, then the cast to bf16 the decoder layer applies
# (stock: a weight cast, the norm, the cast). MLX's rms_single_row arithmetic, rounded once at the
# end: bit-identical to the stock pair.
def _norm_source(hidden: int, eps: float) -> str:
    return f"""
    constexpr int NR = {_NR};
    constexpr int H = {hidden};
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint row = threadgroup_position_in_grid.x;
    const int h0 = int(lid) * NR;
    threadgroup float tg_sq[32];
    threadgroup float tg_inv[1];
    const float4 xv = *(const device packed_float4*)(x + size_t(row) * x_strides[x_ndim - 2] + h0);
    float acc = 0;
    for (int j = 0; j < NR; ++j) acc += xv[j] * xv[j];
    acc = simd_sum(acc);
    if (sg == 0) tg_sq[lane] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) tg_sq[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {{
        acc = simd_sum(tg_sq[lane]);
        if (lane == 0) tg_inv[0] = metal::precise::rsqrt(acc / float(H) + {eps!r}f);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = tg_inv[0];
    for (int j = 0; j < NR; ++j) {{
        out[size_t(row) * H + h0 + j] = bfloat16_t(float(weight[h0 + j]) * (xv[j] * inv));
    }}
"""


@lru_cache(maxsize=16)
def _norm_kernel(hidden: int, eps: float):
    return mx.fast.metal_kernel(
        name=f"mlx_dspark_inkling_rms_norm_bf16_{hidden}",
        input_names=["x", "weight"],
        output_names=["out"],
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
        source=_norm_source(hidden, eps),
    )


def fused_norm_cast(norm, x: mx.array) -> mx.array:
    """``norm(x).astype(bf16)`` for an ``nn.RMSNorm`` — one launch when ``x`` is fp32 ``[1, R, H]``
    (a row view is fine) and the weight bf16; the stock pair otherwise."""
    hidden = int(x.shape[-1])
    group = _row_group(hidden)
    weight = norm.weight
    if (x.dtype != mx.float32 or weight.dtype != mx.bfloat16 or weight.ndim != 1 or x.ndim != 3
            or group is None):
        return norm(x).astype(mx.bfloat16)
    rows = int(x.shape[-2])
    return _norm_kernel(hidden, float(norm.eps))(
        inputs=[x, weight],
        grid=(rows * group, 1, 1),
        threadgroup=(group, 1, 1),
        output_shapes=[(1, rows, hidden)],
        output_dtypes=[mx.bfloat16],
    )[0]


# ---- attention: per-head clip / normalize of q, k, v, r --------------------------------------------
# One simdgroup per (position, head): q and r are scaled and L2-clipped, k is L2-clipped, v is RMS
# normalized (mlx-vlm's l2_clip x3 and rms_norm: 15 launches). Every reduction follows the stock
# kernel's order — rms_single_row for v (one simdgroup at 128 wide), MLX's row_reduce_simple for
# the q / k squared sums (four reads per lane, then a simd sum) and row_reduce_small for r's 16
# (one sequential sum) — and squares are rounded before they are added, as separate Square and Sum
# launches round them (one expression would contract to an fma): bit-identical to stock.
def _prep_source(nh: int, nkv: int, hd: int, drel: int, thr: float, rel_thr: float,
                 eps: float) -> str:
    return f"""
    constexpr int NH = {nh};
    constexpr int NKV = {nkv};
    constexpr int HD = {hd};
    constexpr int DREL = {drel};
    constexpr int PER = HD / 32;
    constexpr int SLOTS = 2 * NH + 2 * NKV;
    const uint lane = thread_index_in_simdgroup;
    const uint sid = threadgroup_position_in_grid.y;
    const uint row = sid / SLOTS;
    const int slot = int(sid % SLOTS);
    if (slot < NH) {{
        const size_t off = size_t(slot) * HD + lane * PER;
        const size_t base = size_t(row) * (NH * HD) + off;
        float xs[PER];
        float acc = 0;
        for (int j = 0; j < PER; ++j) {{
            xs[j] = float(qraw[base + j]) * float(qscale[off + j]);
            const float sq = xs[j] * xs[j];
            acc = sq + acc;
        }}
        acc = simd_sum(acc);
        const float f = min({thr!r}f / (metal::precise::sqrt(acc) + 1e-06f), 1.0f);
        for (int j = 0; j < PER; ++j) qo[base + j] = bfloat16_t(xs[j] * f);
    }} else if (slot < NH + NKV) {{
        const size_t base = size_t(row) * (NKV * HD) + size_t(slot - NH) * HD + lane * PER;
        float xs[PER];
        float acc = 0;
        for (int j = 0; j < PER; ++j) {{
            xs[j] = float(kraw[base + j]);
            const float sq = xs[j] * xs[j];
            acc = sq + acc;
        }}
        acc = simd_sum(acc);
        const float f = min({thr!r}f / (metal::precise::sqrt(acc) + 1e-06f), 1.0f);
        for (int j = 0; j < PER; ++j) ko[base + j] = bfloat16_t(xs[j] * f);
    }} else if (slot < NH + 2 * NKV) {{
        const size_t base = size_t(row) * (NKV * HD) + size_t(slot - NH - NKV) * HD + lane * PER;
        float xs[PER];
        float acc = 0;
        for (int j = 0; j < PER; ++j) {{
            xs[j] = float(vraw[base + j]);
            acc += xs[j] * xs[j];
        }}
        acc = simd_sum(acc);
        const float inv = metal::precise::rsqrt(acc / float(HD) + {eps!r}f);
        for (int j = 0; j < PER; ++j) vo[base + j] = bfloat16_t(xs[j] * inv);
    }} else {{
        if (lane >= uint(DREL)) return;
        const int head = slot - NH - 2 * NKV;
        const size_t hbase = size_t(row) * (NH * DREL) + size_t(head) * DREL;
        float acc = 0;
        float mine = 0;
        for (int j = 0; j < DREL; ++j) {{
            const float xj = float(rraw[hbase + j]) * float(rscale[size_t(head) * DREL + j]);
            const float sq = xj * xj;
            acc = sq + acc;
            if (uint(j) == lane) mine = xj;
        }}
        const float f = min({rel_thr!r}f / (metal::precise::sqrt(acc) + 1e-06f), 1.0f);
        ro[hbase + lane] = bfloat16_t(mine * f);
    }}
"""


@lru_cache(maxsize=16)
def _prep_kernel(nh: int, nkv: int, hd: int, drel: int, thr: float, rel_thr: float, eps: float):
    return mx.fast.metal_kernel(
        name=f"mlx_dspark_inkling_head_prep_{nh}_{nkv}_{hd}_{drel}",
        input_names=["qraw", "kraw", "vraw", "rraw", "qscale", "rscale"],
        output_names=["qo", "ko", "vo", "ro"],
        compile_options={"math_mode": "safe"},
        source=_prep_source(nh, nkv, hd, drel, thr, rel_thr, eps),
    )


def fused_head_prep(qraw, kraw, vraw, rraw, q_scale, rel_scale, *, n_heads: int, n_kv: int,
                    head_dim: int, d_rel: int, clip: float, rel_clip: float, eps: float):
    """``(q, k, v, r)`` as mlx-vlm's attention prepares them, each bf16 ``[1, R, heads, dim]``:
    ``l2_clip(q * q_scale)``, ``l2_clip(k)``, ``rms_norm(v)``, ``l2_clip(r * rel_scale)``."""
    rows = int(qraw.shape[1])
    slots = 2 * n_heads + 2 * n_kv
    return _prep_kernel(n_heads, n_kv, head_dim, d_rel, float(clip), float(rel_clip),
                        float(eps))(
        inputs=[qraw, kraw, vraw, rraw, q_scale, rel_scale],
        grid=(32, rows * slots, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, rows, n_heads, head_dim), (1, rows, n_kv, head_dim),
                       (1, rows, n_kv, head_dim), (1, rows, n_heads, d_rel)],
        output_dtypes=[mx.bfloat16] * 4,
    )


# ---- attention: the global layers' dynamic short convolution --------------------------------------
# out[t, c] = (sum_lag ctx[t - lag, c] * (static[lag, c] + dyn[t, head(c), lag]) + x[t, c]) / 3
# (stock: ~40 launches per call — a concatenate, four slice-multiply-adds, casts).
_DYN_SOURCE = r"""
    uint i = thread_position_in_grid.x;
    int L = x_shape[1], C = x_shape[2];
    if (i >= uint(L * C)) return;
    int c = i % C, t = i / C;
    const int head = c / (C / HEADS);
    float acc = 0;
    for (int lag = 0; lag < W; ++lag) {
        int p = t - lag;
        float v = p < 0
            ? float(state[(W - 1 + p) * state_strides[1] + c * state_strides[2]])
            : float(x[p * x_strides[1] + c * x_strides[2]]);
        float w = float(stat[lag * C + c]) + float(dyn[(t * HEADS + head) * W + lag]);
        float prod = v * w;
        acc = lag == 0 ? prod : acc + prod;
    }
    float xt = float(x[t * x_strides[1] + c * x_strides[2]]);
    out[i] = T((acc + xt) / divisor);
    window[(W - 1 + t) * C + c] = xt;
    if (t == 0) {
        for (int s = 0; s < W - 1; ++s) {
            window[s * C + c] = float(state[s * state_strides[1] + c * state_strides[2]]);
        }
    }
"""


@lru_cache(maxsize=1)
def _dyn_kernel():
    return mx.fast.metal_kernel(
        name="mlx_dspark_inkling_dynamic_sconv_state",
        input_names=["state", "stat", "dyn", "x", "divisor"],
        output_names=["out", "window"],
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
        source=_DYN_SOURCE,
    )


@lru_cache(maxsize=8)
def _dyn_divisor(width: int) -> mx.array:
    d = mx.array(math.sqrt(2 * width + 1), mx.float32)
    mx.eval(d)
    return d


def _dyn_sconv_step(conv, x, dynamic_input, cache):
    """mlx-vlm's ``DynamicShortConvolution`` on the layer's conv cache: the per-head dynamic taps
    stay a stock projection, everything after it is one launch."""
    k, idx = conv.kernel_size, conv.conv_idx
    _, length, channels = x.shape
    state = cache[idx]
    if state is None:
        state = mx.zeros((1, k - 1, channels))
    out, window = _dyn_kernel()(
        inputs=[state, conv.static_weight, conv.weight_proj(dynamic_input), x, _dyn_divisor(k)],
        template=[("T", x.dtype), ("W", k), ("HEADS", conv.num_heads)],
        grid=(length * channels, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape, (1, k - 1 + length, channels)],
        output_dtypes=[x.dtype, mx.float32],
    )
    if getattr(cache, "is_speculating", False):
        cache.record_speculative_window(idx, window, k - 1)
    cache[idx] = window[:, -(k - 1):]
    return out


@mx.compile
def _gated(out, gate):
    return (out.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))).astype(out.dtype)


@lru_cache(maxsize=16)
def _zero_row(batch: int, heads: int, head_dim: int, dtype) -> mx.array:
    """The attention sink's key / value row (stock fills one per call)."""
    z = mx.zeros((batch, heads, 1, head_dim), dtype)
    mx.eval(z)
    return z


def _temporal_parts(kv, v: mx.array) -> list[mx.array]:
    """The pieces mlx-vlm's ``RotatingKVCache._temporal_order`` returns or concatenates, so the
    caller can add the sink row in the same copy. The self-test runs a wrapped ring against the
    stock forward (``ring_run``)."""
    if kv._idx == v.shape[2]:
        return [v]
    if kv._idx < kv.offset:
        return [v[..., : kv.keep, :], v[..., kv._idx:, :], v[..., kv.keep: kv._idx, :]]
    return [v[..., : kv._idx, :]]


# ---- the attention body -------------------------------------------------------------------------
# mlx-vlm's Attention.__call__ (models/inkling2/language.py, Apache-2.0) for one batch row: the
# projections, the KV cache update, the relative-position mask and the attention itself are the
# stock calls; the pieces between them are the launches above.
def _attn_call(base):
    import sys

    lang = sys.modules[base.__module__]

    def __call__(self, x, key_input=None, value_input=None, cache=None, conv_mask=None):
        cfg = self.config
        kv, conv = cache if cache is not None else (None, None)
        if not (_on(self) and kv is not None and conv is not None and conv_mask is None
                and x.ndim == 3 and x.shape[0] == 1 and x.shape[1] <= MAX_ROWS
                and x.dtype == mx.bfloat16 and not self.training
                and self.sharding_group is None and cfg.use_value_norm_no_gamma
                and self.head_dim % 32 == 0 and cfg.d_rel <= 32 and _gpu()
                and not isinstance(kv, lang.BatchRotatingKVCache)
                and getattr(kv, "left_padding", None) is None
                and self.k_sconv.kernel_size == 4 and self.v_sconv.kernel_size == 4
                and (self.n_kv * self.head_dim) % 128 == 0
                and self.q_scale.dtype == mx.bfloat16 and self.rel_scale.dtype == mx.bfloat16):
            return base.__call__(self, x, key_input, value_input, cache, conv_mask)
        b, length, _ = x.shape
        key_input = x if key_input is None else key_input
        value_input = x if value_input is None else value_input
        k = self.k_proj(key_input)
        v = self.v_proj(value_input)
        if self.is_sliding:
            k = _sconv_step(self.k_sconv, k, conv)
            v = _sconv_step(self.v_sconv, v, conv)
        else:
            k = _dyn_sconv_step(self.k_sconv, k, key_input, conv)
            v = _dyn_sconv_step(self.v_sconv, v, value_input, conv)
        q, k, v, r = fused_head_prep(
            self.q_proj(x), k, v, self.r_proj(x), self.q_scale, self.rel_scale,
            n_heads=self.n_heads, n_kv=self.n_kv, head_dim=self.head_dim, d_rel=cfg.d_rel,
            clip=cfg.qk_l2_clip_threshold, rel_clip=cfg.rel_l2_clip_threshold,
            eps=cfg.rms_norm_eps)
        if not self.is_sliding and cfg.log_scaling_n_floor is not None:
            offset = kv.offset
            # tau = 1 + alpha * log(max(position / n_floor, 1)) is exactly 1 up to n_floor
            if offset + length > cfg.log_scaling_n_floor:
                positions = mx.arange(length) + mx.array(offset)[..., None] + 1
                tau = 1 + cfg.log_scaling_alpha * mx.log(
                    mx.maximum(positions / cfg.log_scaling_n_floor, 1))
                q = (q.astype(mx.float32) * tau[..., None, None]).astype(x.dtype)
                r = (r.astype(mx.float32) * tau[..., None, None]).astype(x.dtype)
        q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
        k, v = kv.update_and_fetch(k, v)
        # [keys in temporal order ‖ the sink's zero row] as ONE concatenate (stock: the ring's
        # reorder, a zeros fill, then another copy of the whole K and V for the sink row)
        rotating = isinstance(kv, lang.RotatingKVCache)
        k_parts = _temporal_parts(kv, k) if rotating else [k]
        v_parts = _temporal_parts(kv, v) if rotating else [v]
        size = sum(part.shape[2] for part in k_parts)
        mask = lang.banded_additive_mask(r, self.rel_proj.astype(x.dtype), size - length, size,
                                         self.sliding, self.rel_extent, left_padding=None,
                                         shape_ref=k if k.shape[2] == size else k_parts[0])
        sinks = self.sink_proj(x).transpose(0, 2, 1)[..., None]
        mask = mx.concatenate([mask, sinks], axis=-1)
        k = mx.concatenate([*k_parts, _zero_row(b, self.n_kv, self.head_dim, k.dtype)], axis=2)
        v = mx.concatenate([*v_parts, _zero_row(b, self.n_kv, self.head_dim, v.dtype)], axis=2)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(b, length, -1)
        return self.o_proj(_gated(out, self.gate_proj(x)))

    return __call__


# ---- the decoder layer body ---------------------------------------------------------------------
# mlx-vlm's DecoderLayer.__call__ (models/inkling2/language.py, Apache-2.0) with the residual
# arithmetic folded into the two convolution launches and each norm + cast into one.
def _layer_call(base):
    import sys

    lang = sys.modules[base.__module__]

    def __call__(self, x, blocks, cache=None, messages=None):
        conv = cache[1] if cache is not None else None
        if not (_on(self) and conv is not None and x.ndim == 3
                and _sconv_ok(self.attn_sconv, x, conv, lang._cache_padding_mask(conv, x.shape[1]))
                and self.mlp_sconv.kernel_size == 4 and not self.training
                and _row_group(x.shape[-1]) is not None):
            return base.__call__(self, x, blocks, cache, messages)
        mixed = (self.attention_input(blocks, x) if self.attention_input is not None
                 else x[..., None, :])
        if self.is_sliding and messages is None:
            attn = self.self_attn(fused_norm_cast(self.input_layernorm, mixed[..., 0, :]),
                                  cache=cache, conv_mask=None)
        else:
            parts = [mixed[..., min(i, mixed.shape[-2] - 1), :] for i in range(3)]
            if messages is not None:
                parts = [self.controllers[name](part, messages[..., i, :])
                         for i, (name, part) in enumerate(zip(("attn", "k", "v"), parts))]
            norms = ((self.input_layernorm,) * 3 if self.is_sliding
                     else (self.q_input_layernorm, self.k_input_layernorm, self.v_input_layernorm))
            parts = [fused_norm_cast(norm, part) for norm, part in zip(norms, parts)]
            attn = self.self_attn(*parts, cache=cache, conv_mask=None)
        # residual = attn_sconv(attn).astype(fp32) [+ x.astype(fp32)]
        residual = _sconv_step(self.attn_sconv, attn, conv,
                               None if self.is_block_boundary else x, mx.float32)
        if self.is_block_boundary:
            blocks.append(x)
        hidden = self.mlp_input(blocks, residual)[..., 0, :]
        if messages is not None:
            hidden = self.controllers["mlp"](hidden, messages[..., 3, :])
        hidden = self.mlp(fused_norm_cast(self.post_attention_layernorm, hidden))
        # out = (residual + mlp_sconv(hidden).astype(fp32)).astype(bf16)
        out = _sconv_step(self.mlp_sconv, hidden, conv, residual, x.dtype)
        conv.advance(x.shape[1])
        return out

    return __call__


# ---- the routed experts --------------------------------------------------------------------------
def _moe_call(base):
    from . import inkling_kernels as K

    def __call__(self, x):
        if not (_on(self) and x.ndim == 3 and x.shape[0] == 1
                and 1 <= x.shape[1] <= K.MOE_MAX_ROWS and x.dtype == mx.bfloat16
                and not self.training and getattr(self, "sharding_group", None) is None
                and _gpu()):
            return base.__call__(self, x)
        return K.fused_moe(self, x)

    return __call__


# ---- install --------------------------------------------------------------------------------------
_CLASSES: dict[tuple[type, str], type] = {}


def _swap(module, make_call, tag: str) -> None:
    """Give ``module`` (one instance) a subclass of its own class whose ``__call__`` is fused."""
    base = type(module)
    if base.__name__.startswith(tag):
        return
    cls = _CLASSES.get((base, tag))
    if cls is None:
        cls = type(f"{tag}{base.__name__}", (base,), {"__call__": make_call(base)})
        _CLASSES[(base, tag)] = cls
    module.__class__ = cls


def _fused_modules(language_model):
    from .inkling_kernels import moe_spec

    model = language_model.model
    final = getattr(model, "final_input", None)
    if final is not None and getattr(final, "normalize_values", False):
        yield final, "res"
    for layer in model.layers:
        if hasattr(layer, "attn_sconv") and hasattr(layer, "mlp_sconv"):
            yield layer, "layer"
        if getattr(layer, "mlp", None) is not None and moe_spec(layer.mlp):
            yield layer.mlp, "moe"
        if getattr(layer, "self_attn", None) is not None and hasattr(layer.self_attn, "q_scale"):
            yield layer.self_attn, "attn"
        for name in ("attention_input", "mlp_input"):
            res = getattr(layer, name, None)
            if res is not None and getattr(res, "normalize_values", False):
                yield res, "res"
        for ctrl in (getattr(layer, "controllers", None) or {}).values():
            yield ctrl, "ctrl"
        for conv in (getattr(layer, "attn_sconv", None), getattr(layer, "mlp_sconv", None)):
            if conv is not None:
                yield conv, "sconv"
        attn = getattr(layer, "self_attn", None)
        for name in ("k_sconv", "v_sconv"):
            conv = getattr(attn, name, None) if attn is not None else None
            if conv is not None and hasattr(conv, "conv"):          # static (sliding layers) only
                yield conv, "sconv"


_MAKERS = {"sconv": _sconv_call, "res": _res_call, "ctrl": _ctrl_call, "layer": _layer_call,
           "attn": _attn_call, "moe": _moe_call}


_OFF = "_mlx_dspark_fused_off"      # kinds the self-test switched off for this model
EXACT_KINDS = ("sconv", "layer", "attn")       # bit-identical to the stock forward
ROW_EXACT_KINDS = ("moe",)                     # each row bit-identical to the stock DECODE call
ROUNDING_KINDS = ("res", "ctrl")               # fp32 dot products summed in another order
ALL_KINDS = EXACT_KINDS + ROW_EXACT_KINDS + ROUNDING_KINDS


def install(language_model, enabled: bool = True) -> dict[str, int]:
    """Swap the fused pieces into an Inkling 2 language model. Returns how many modules of each
    kind took one. Idempotent."""
    counts: dict[str, int] = {}
    for module, kind in _fused_modules(language_model):
        _swap(module, _MAKERS[kind], "Fused")
        counts[kind] = counts.get(kind, 0) + 1
    set_enabled(language_model, enabled)
    return counts


def set_enabled(language_model, enabled: bool, kinds=None) -> None:
    """Route every installed piece to the fused path (``True``) or the stock forward (the A/B).
    ``kinds`` limits the switch to some pieces (``"sconv"``, ``"res"``, ``"ctrl"``, ``"layer"``,
    ``"attn"``, ``"moe"``). A kind the self-test rejected (or :func:`disable` named) stays off."""
    off = language_model.__dict__.get(_OFF, ())
    for module, kind in _fused_modules(language_model):
        if kinds is None or kind in kinds:
            module.__dict__[_FLAG] = bool(enabled) and kind not in off


def disable(language_model, kinds) -> None:
    """Keep ``kinds`` on the stock path from now on (an A/B switch; survives ``set_enabled``)."""
    off = set(language_model.__dict__.get(_OFF, ())) | set(kinds)
    language_model.__dict__[_OFF] = tuple(sorted(off))
    set_enabled(language_model, False, kinds)


def enabled_kinds(language_model) -> tuple[str, ...]:
    """The kinds currently routed to their fused path (sorted)."""
    return tuple(sorted({kind for module, kind in _fused_modules(language_model) if _on(module)}))


def _moe_selftest(language_model, rand) -> bool:
    """The fused MoE against the stock module ROW BY ROW (what a decode step computes for each
    token), on one layer per expert format, with rows repeated so that runs of 1, 2, 3, 4 and
    more routes per expert all occur — each run length is its own kernel path. The first call
    is the widest verify (8 rows): the router's largest threadgroup, so a GPU that cannot launch
    it fails here, at load."""
    from .inkling_kernels import fused_moe, moe_spec

    seen = set()
    for layer in language_model.model.layers:
        moe = getattr(layer, "mlp", None)
        if moe is None or not moe_spec(moe):
            continue
        fmt = tuple((p.mode, p.bits, tuple(p.weight.shape[1:]))
                    for p in (moe.switch_mlp.up_proj, moe.switch_mlp.down_proj))
        if fmt in seen:
            continue
        seen.add(fmt)
        stock = type(moe).__mro__[1].__call__
        hidden = int(moe.gate_weight.shape[1])
        a, b, c = (rand((1, 1, hidden), mx.bfloat16) for _ in range(3))
        for rows in ([a, a, a, a, a, b, b, c], [a, b, b, b, c], [c]):
            ref = mx.concatenate([stock(moe, row) for row in rows], axis=1)
            if not bool(mx.array_equal(fused_moe(moe, mx.concatenate(rows, axis=1)), ref)):
                return False
    return bool(seen)


def _rel_l2(got: mx.array, ref: mx.array) -> float:
    got, ref = got.astype(mx.float32), ref.astype(mx.float32)
    return float(mx.sqrt(mx.sum(mx.square(got - ref)) / (mx.sum(mx.square(ref)) + 1e-30)))


def _state_arrays(cache) -> list[mx.array]:
    out = []
    for part in cache:
        state = getattr(part, "state", None)
        for a in (state if isinstance(state, (list, tuple)) else [state]):
            if isinstance(a, mx.array):
                out.append(a)
    return out


def selftest(language_model, rows: int = 2, seed: int = 0) -> dict[str, bool]:
    """Compare every installed kind with the stock forward of THIS mlx-vlm on random inputs, and
    switch off any kind that does not hold: the layer and attention bodies are copies of
    mlx-vlm's, so an upstream change must turn them off, not change the model. ``sconv`` /
    ``layer`` / ``attn`` must match bit for bit (outputs and the cache states they leave);
    ``moe`` must match a single-row layer call bit for bit and, at more rows, the stock module's
    single-row call row by row (:func:`_moe_selftest`); ``res`` / ``ctrl`` within rounding. A
    kernel that does not build on this GPU (threadgroup limits) fails its kind the same way.
    Leaves the model with every passing kind on. Returns ``{kind: passed}``."""
    lm = language_model
    model = lm.model
    layers = model.layers
    visits = list(model.layer_indices)
    hidden = int(model.embed_norm.weight.shape[0])
    key = mx.random.key(seed)
    keys = iter(mx.random.split(key, 128))

    def rand(shape, dtype=mx.float32):
        return mx.random.normal(shape, key=next(keys)).astype(dtype)

    def layer_run(index, on_kinds, messages=None, rows=rows):
        x = rand((1, rows, hidden), mx.bfloat16)
        blocks = [rand((1, rows, hidden), mx.bfloat16)] if layers[index].attention_input is not None else []
        outs = []
        for on in (False, True):
            set_enabled(lm, False)
            if on:
                set_enabled(lm, True, on_kinds)
            cache = lm.make_cache()[visits.index(index)]
            out = layers[index](x, list(blocks), cache, messages)
            st = _state_arrays(cache)
            mx.eval(out, st)
            outs.append((out, st))
        (a, sa), (b, sb) = outs
        return (bool(mx.array_equal(a, b)) and len(sa) == len(sb)
                and all(bool(mx.array_equal(u, v)) for u, v in zip(sa, sb)))

    def ring_run(index, steps=6, window=3):
        """A sliding layer on a ``window``-key ring for ``steps`` single-row calls — it fills,
        wraps and is read back in temporal order (the attention body reorders the ring and adds
        the sink row in one copy; a load-time probe on a fresh 1024-key ring never wraps)."""
        kv = lm.make_cache()[visits.index(index)][0]
        if not hasattr(kv, "max_size") or not hasattr(kv, "_temporal_order"):
            return True
        xs = [rand((1, 1, hidden), mx.bfloat16) for _ in range(steps)]
        blocks = [rand((1, 1, hidden), mx.bfloat16)] if layers[index].attention_input is not None else []
        runs = []
        for on in (False, True):
            set_enabled(lm, False)
            if on:
                set_enabled(lm, True, ("attn",))
            cache = lm.make_cache()[visits.index(index)]
            cache.caches = (type(kv)(max_size=window, keep=0), *cache.caches[1:])
            outs = [layers[index](x, list(blocks), cache, None) for x in xs]
            st = _state_arrays(cache)
            mx.eval(outs, st)
            runs.append((outs, st))
        (a, sa), (b, sb) = runs
        return (all(bool(mx.array_equal(u, v)) for u, v in zip(a, b)) and len(sa) == len(sb)
                and all(bool(mx.array_equal(u, v)) for u, v in zip(sa, sb)))

    sliding = next((i for i, layer in enumerate(layers) if layer.is_sliding and i > 0), None)
    globl = next((i for i, layer in enumerate(layers) if not layer.is_sliding), None)
    core = next((i for i, layer in enumerate(layers) if getattr(layer, "controllers", None)), None)
    probes = [i for i in (sliding, globl) if i is not None]
    result: dict[str, bool] = {}
    lm.__dict__[_OFF] = ()
    installed = {kind for _, kind in _fused_modules(lm)}
    for kind in EXACT_KINDS:
        if kind not in installed:
            continue
        try:
            ok = all(layer_run(i, (kind,)) for i in probes)
            if ok and core is not None:
                ok = layer_run(core, (kind,), rand((1, rows, 4, hidden)))
            if ok and kind == "attn" and sliding is not None:
                ok = ring_run(sliding)
        except Exception:  # noqa: BLE001 — a kernel this GPU cannot launch: keep the stock path
            ok = False
        result[kind] = ok
    if "moe" in installed:
        # a decode step through the whole layer (bit for bit, states included), then the
        # multi-row call against single-row stock calls
        try:
            ok = all(layer_run(i, ("moe",), rows=1) for i in probes)
            set_enabled(lm, False)
            ok = ok and _moe_selftest(lm, rand)
        except Exception:  # noqa: BLE001
            ok = False
        result["moe"] = ok
    try:
        res = layers[probes[0]].mlp_input if probes else None
        ok = res is not None
        if ok:
            blocks = [rand((1, rows, hidden), mx.bfloat16) for _ in range(2)]
            partial = rand((1, rows, hidden))
            ref = type(res).__mro__[1].__call__(res, blocks, partial)
            ok = _rel_l2(fused_attention_residual([*blocks, partial], res.query), ref) < 1e-4
    except Exception:  # noqa: BLE001
        ok = False
    result["res"] = ok
    try:
        ctrl = next(iter(layers[core].controllers.values())) if core is not None else None
        ok = ctrl is not None and ctrl.down_weight.shape[0] % _CODE_GROUP == 0
        if ok:
            x, message = rand((1, rows, hidden)), rand((1, rows, hidden))
            ref = type(ctrl).__mro__[1].__call__(ctrl, x, message)
            got = fused_controller(x, message, ctrl.down_weight, ctrl.forget_weight,
                                   ctrl.gain_weight, ctrl.eps)
            ok = _rel_l2(got, ref) < 2e-2
    except Exception:  # noqa: BLE001
        ok = False
    result["ctrl"] = ok
    lm.__dict__[_OFF] = tuple(kind for kind, passed in result.items() if not passed)
    set_enabled(lm, True)
    return result
