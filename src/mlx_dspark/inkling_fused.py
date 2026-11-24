"""Fused forward pieces for Inkling 2 (Thinking Machines Inkling-2-Mini) at decode / verify widths.

One decode step of the stock mlx-vlm forward launches ~7,000 GPU kernels for 48 layer visits, most
of them a few thousand elements wide: about half of a step's time is the launches, not the weights
(NOTES "Inkling fused forward"). Each piece here replaces a run of small stock operations with one
Metal kernel, for calls of one batch row: every piece up to ``MAX_ROWS`` positions (a decode step,
a speculative verify, a short prompt or a follow-up turn's new tokens), and the bit-exact pieces
alone up to ``CHUNK_ROWS`` (a prefill chunk — a forward bit-identical to the stock one, without
most of its launches); anything else — the CPU, training, a batch — takes the stock path.

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
read once per run of routes, the combine in one launch. At any width a fused call has, each row
is bit-identical to the stock module's single-row call (the stock multi-row call rounds its
routing scores differently from its own single-row call; this one does not).

Calls of up to ``VECTOR_ROWS`` positions are a drafter round (MLX's attention still runs its
vector kernels there); up to ``MAX_ROWS`` they are a lookup round's long copy draft or a short
prompt, taken once the self-test has matched every exact piece to the stock forward at that width
(``set_rows``); up to ``CHUNK_ROWS`` they are prefill chunks, which ``CHUNK_KINDS`` take once the
self-test's ``chunk`` stage has matched them there.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from functools import lru_cache

import mlx.core as mx

from .inkling_cache import is_sink_cache, small_window, windowed

MAX_ROWS = 63         # positions per call EVERY fused piece accepts: a verify (a lookup round is
#                       32 copied tokens + the anchor, a drafter round cap 7 + anchor) and a short
#                       prompt. 63, not more: from 64 rows MLX's sorted expert gather is out of
#                       its 17-63-row dead zone and the stock experts are the cheaper ones (M4 Pro,
#                       one forward on a fresh cache: 56 rows 409 ms stock / 226 fused, 64 rows
#                       283 / 252, 80 rows 322 / 307 against 302 for CHUNK_KINDS alone)
CHUNK_ROWS = 512      # positions per call the bit-exact pieces (CHUNK_KINDS) accept: a prefill
#                       chunk (Target.prefill_chunk), where the launches they save are a tenth of
#                       the forward and the experts / residual mixes are the stock modules
VECTOR_ROWS = 8       # where MLX's SDPA still runs its vector kernels: the widest drafter round,
#                       and the limit until the self-test clears the wider calls (set_rows)
_FLAG = "_mlx_dspark_fused"
_ROWS = "_mlx_dspark_rows"      # per fused module: the positions per call it accepts
_CHUNK = "_mlx_dspark_chunk_rows"   # per model: what CHUNK_KINDS accept (>= the model's _ROWS)


def _gpu() -> bool:
    return mx.metal.is_available() and mx.default_device() == mx.gpu


def _on(module) -> bool:
    return bool(module.__dict__.get(_FLAG, False))


def _rows(module) -> int:
    return module.__dict__.get(_ROWS, VECTOR_ROWS)


# ---- tree verify --------------------------------------------------------------------------------
# A verify whose rows are a draft TREE (tree.py: row 0 the anchor, each drafted row after its
# parent, the main path first) instead of a chain. Three things in this forward follow a token's
# position in time, and each takes a row's own root path instead:
#   the short convolutions   a row's 3 earlier taps are its 3 nearest ANCESTORS (the committed
#                            state where it has fewer) — `conv_idx`, read by the same kernels;
#   the attention            of the tree's rows a row sees itself and its ancestors, and its
#                            relative-position bias, sliding window and log scaling are taken
#                            at the anchor's position plus its DEPTH — the mask kernel and the
#                            packed attention read `depth` / `bits`;
#   the caches               hold the rows in tree order until the commit moves the accepted
#                            path into place (a counter when it is the main path's prefix).
# Everything else is per row (experts, residual mixes, messages, controllers, the head). On a
# chain-shaped tree every kernel evaluates the chain's own expression: bit-identical, which
# Target.tree_selftest checks before a loop may use trees.
_TREE = None            # the TreeRound whose graph is being built (the MLX thread is the only one)


class TreeRound:
    """One tree verify on Inkling 2: what the fused pieces read while its graph is built, what
    its commit needs, and the transaction itself (``Target._txn``: ``active`` / ``abort`` as
    mlx-vlm's, ``commit`` along a path).

    ``convs``: per short convolution ``(conv cache, slot, the fp32 window [state ‖ rows])``;
    ``kvs``: per attention ``(kv cache, the K rows, the V rows)`` as written; ``masks``:
    attention calls that took the tree's mask. A visit that left the fused path recorded less
    (:meth:`complete`). ``caches`` (the model's layer-visit caches): what :meth:`abort`
    restores — every conv state and KV position as the round found them, whatever ran."""

    def __init__(self, tree, caches=None):
        self.depth, self.bits, self.conv_idx = tree.depth, tree.bits, tree.conv_idx
        self.width = int(tree.width)
        self.convs: list = []
        self.kvs: list = []
        self.masks = 0
        self.active = True
        self._held = [(entry.caches[1], list(entry.caches[1].cache), entry.caches[0],
                       int(entry.caches[0].offset)) for entry in caches or ()]

    def complete(self, visits: int) -> bool:
        """Whether every one of ``visits`` layer visits ran its tree path: four convolutions,
        one KV append and one tree mask each."""
        return (len(self.convs) == 4 * visits and len(self.kvs) == visits
                and self.masks == visits)

    def commit(self, path) -> None:
        """Leave the caches as a chain verify of ``path`` (accepted rows, root first) committed
        in full would: each conv state the last taps along the path, each KV cache the path's
        rows after the committed ones."""
        if not self.active:
            raise RuntimeError("this tree verify was already committed or aborted")
        path = [int(r) for r in path]
        keep = len(path)
        if not path or path[0] != 0 or keep > self.width:
            raise ValueError(f"not a root path of this tree: {path}")
        contiguous = path == list(range(keep))
        pidx = None if contiguous else mx.array(path, dtype=mx.int32)
        picks: dict[int, mx.array] = {}
        for cache, slot, window in self.convs:
            k1 = int(window.shape[1]) - self.width           # the state's rows
            if contiguous:
                cache[slot] = mx.contiguous(window[:, keep: keep + k1])
                continue
            pick = picks.get(k1)
            if pick is None:
                taps = list(range(k1)) + [k1 + r for r in path]
                pick = picks[k1] = mx.array(taps[len(taps) - k1:], dtype=mx.int32)
            cache[slot] = mx.contiguous(window[:, pick])
        for kv, keys, values in self.kvs:
            if contiguous:                                   # the common case: a counter
                kv.trim(self.width - keep)
            else:
                kv.trim(self.width)
                kv.update_and_fetch(keys[..., pidx, :], values[..., pidx, :])
        self._close()

    def abort(self) -> None:
        """Put the caches back as they were before the verify."""
        if not self.active:
            return
        for conv, states, kv, offset in self._held:
            conv.cache = list(states)
            if kv.offset > offset:
                kv.trim(kv.offset - offset)
        self._close()

    def _close(self) -> None:
        self.convs, self.kvs, self._held, self.active = [], [], [], False


@contextmanager
def tree_round(tree, caches):
    """Build one forward over ``caches`` as a tree verify of ``tree`` (a ``tree.DeviceTree``):
    the fused pieces called inside take its structure. Yields the :class:`TreeRound`; a
    forward that raises is aborted (the caches as before it)."""
    global _TREE
    if _TREE is not None:
        raise RuntimeError("a tree verify is already being built")
    rnd = _TREE = TreeRound(tree, caches)
    try:
        yield rnd
    except BaseException:
        rnd.abort()
        raise
    finally:
        _TREE = None


def tree_supported(language_model) -> bool:
    """Whether this model's forward can run a tree verify: the pieces that carry a row's
    position — the layer and attention bodies, the short convolutions, the one-launch mask —
    on their fused path, and the in-place KV caches (a commit is their counter, or a
    re-append)."""
    lm = language_model
    return (kv_caches(lm) and {"layer", "attn", "sconv"} <= set(enabled_kinds(lm))
            and all(layer.self_attn.__dict__.get(_MASK) for layer in lm.model.layers
                    if getattr(layer, "self_attn", None) is not None))


# ---- short convolution + next state -------------------------------------------------------------
# out[t, c]    = (sum_j ctx[t + j, c] * w[c, j] + x[t, c]) / sqrt(W + 1), ctx = [state ‖ x] (virtual)
# window[s, c] = ctx[s, c] in fp32 — the stock path's concatenate; its last W-1 rows are the next
#                state, and a speculative verify records the whole window for rollback.
def _sconv_source(res: bool, tree: bool = False) -> str:
    add = "conv = conv + float(res[t * res_strides[1] + c * res_strides[2]]);" if res else ""
    # a tree verify reads each row's taps along its own root path: cidx[t, j] indexes
    # [state ‖ x] (tree.DraftTree.conv_index) — t + j on a chain, the same sum in the same order
    tap = ("int p = int(cidx[t * cidx_strides[0] + j * cidx_strides[1]]) - (W - 1);" if tree
           else "int p = t + j - W + 1;")
    return r"""
    uint i = thread_position_in_grid.x;
    int L = x_shape[1], C = x_shape[2];
    if (i >= uint(L * C)) return;
    int c = i % C, t = i / C;
    float value = 0;
    for (int j = 0; j < W; ++j) {
        """ + tap + r"""
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


@lru_cache(maxsize=4)
def _sconv_kernel(res: bool, tree: bool = False):
    return mx.fast.metal_kernel(
        name="mlx_dspark_inkling_sconv_state" + ("_res" if res else "") + ("_tree" if tree else ""),
        input_names=(["state", "weight", "x", "divisor"] + (["res"] if res else [])
                     + (["cidx"] if tree else [])),
        output_names=["out", "window"],
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
        source=_sconv_source(res, tree),
    )


@lru_cache(maxsize=8)
def _divisor(width: int) -> mx.array:
    d = mx.array(math.sqrt(width + 1), mx.float32)
    mx.eval(d)
    return d


def fused_short_convolution(state: mx.array, weight: mx.array, x: mx.array, width: int,
                            res: mx.array | None = None, out_dtype=None,
                            cidx: mx.array | None = None):
    """``(out, window)`` for one batch row: the convolution output (``x``'s dtype, as stock) and
    the fp32 ``[1, width - 1 + L, C]`` concatenation ``[state ‖ x]`` whose tail is the next state.

    ``res`` / ``out_dtype`` fold the decoder layer's residual arithmetic into the launch, at the
    stock rounding points: the convolution is rounded to ``x``'s dtype, widened to fp32, ``res``
    is added in fp32, and the sum is stored as ``out_dtype`` (default: ``x``'s dtype).

    ``cidx`` (a tree verify): int32 ``[L, width]``, row ``t``'s taps as indices into
    ``[state ‖ x]`` — its ``width - 1`` nearest ancestors and itself."""
    _, length, channels = x.shape
    out_dtype = out_dtype or x.dtype
    inputs = ([state, weight, x, _divisor(width)] + ([res] if res is not None else [])
              + ([cidx] if cidx is not None else []))
    return _sconv_kernel(res is not None, cidx is not None)(
        inputs=inputs,
        template=[("T", x.dtype), ("OT", out_dtype), ("W", width)],
        grid=(length * channels, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape, (1, width - 1 + length, channels)],
        output_dtypes=[out_dtype, mx.float32],
    )


def _sconv_ok(conv, x, cache, mask) -> bool:
    return (cache is not None and mask is None and not conv.training and conv.kernel_size == 4
            and x.ndim == 3 and x.shape[0] == 1 and x.shape[1] <= _rows(conv)
            and x.dtype == mx.bfloat16 and x.shape[-1] % 128 == 0 and _gpu())


def _sconv_step(conv, x, cache, res=None, out_dtype=None):
    """One fused convolution on the layer's conv cache (the stock ``_update_conv_cache`` rules:
    the next state is the window's tail; a speculative verify records the whole window)."""
    k, idx = conv.kernel_size, conv.conv_idx
    state = cache[idx]
    if state is None:
        state = mx.zeros((1, k - 1, x.shape[-1]))
    tree = _TREE
    if tree is not None:
        out, window = fused_short_convolution(state, conv.conv.weight, x, k, res, out_dtype,
                                              tree.conv_idx)
        tree.convs.append((cache, idx, window))
        cache[idx] = window[:, -(k - 1):]            # a placeholder: commit or abort sets it
        return out
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
                and partial.shape[1] <= _rows(self) and len(blocks) < 8 and not self.training
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
        if not (_on(self) and x.ndim == 3 and x.shape[0] == 1 and x.shape[1] <= _rows(self)
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
def _dyn_source(tree: bool = False) -> str:
    # a tree verify: lag `lag` of row t is its lag-th ancestor (see _sconv_source)
    tap = ("int p = int(cidx[t * cidx_strides[0] + (W - 1 - lag) * cidx_strides[1]]) - (W - 1);"
           if tree else "int p = t - lag;")
    return r"""
    uint i = thread_position_in_grid.x;
    int L = x_shape[1], C = x_shape[2];
    if (i >= uint(L * C)) return;
    int c = i % C, t = i / C;
    const int head = c / (C / HEADS);
    float acc = 0;
    for (int lag = 0; lag < W; ++lag) {
        """ + tap + r"""
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


@lru_cache(maxsize=2)
def _dyn_kernel(tree: bool = False):
    return mx.fast.metal_kernel(
        name="mlx_dspark_inkling_dynamic_sconv_state" + ("_tree" if tree else ""),
        input_names=["state", "stat", "dyn", "x", "divisor"] + (["cidx"] if tree else []),
        output_names=["out", "window"],
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
        source=_dyn_source(tree),
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
    tree = _TREE
    out, window = _dyn_kernel(tree is not None)(
        inputs=([state, conv.static_weight, conv.weight_proj(dynamic_input), x, _dyn_divisor(k)]
                + ([tree.conv_idx] if tree is not None else [])),
        template=[("T", x.dtype), ("W", k), ("HEADS", conv.num_heads)],
        grid=(length * channels, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape, (1, k - 1 + length, channels)],
        output_dtypes=[x.dtype, mx.float32],
    )
    if tree is not None:
        tree.convs.append((cache, idx, window))
        cache[idx] = window[:, -(k - 1):]            # a placeholder: commit or abort sets it
        return out
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


# ---- attention: the relative-position mask and the sink's column, one launch ----------------------
# mlx-vlm builds the additive mask [1, H, L, S] with a kernel of 8 x 8 threadgroups over
# (key, query row) — at one query row seven of every eight threads have nothing to do — that
# reads its [d_rel, extent] projection down a column (2 KB between the 16 values of one key
# distance), and then concatenates the sink's logit column: a second copy of the whole mask.
# Here: one thread per output element, the projection transposed once per layer so a distance's
# values sit together, and the sink's column written by the same launch. The value expression is
# mlx-vlm's own (`_MASK_V2_SRC`, models/inkling/language.py, MIT): the same 16 products added in
# the same order — bit-identical, which the self-test checks against the stock pair at load.
# A tree verify (the last LQ keys are the tree's rows, main path first): a row's position is the
# anchor's plus its DEPTH, so its distance to a committed key is taken from there, and of the
# tree's own rows it sees itself and its ancestors (`bits`: one uint32 of them per row), each at
# their difference in depth. On a chain that is the expression above: the same mask, bit for bit.
_MASK_DIST = "const int dist = (i + (S - LQ)) - j;              // backward distance"
_MASK_DIST_TREE = """const int C = S - LQ;                             // committed keys
    int dist;                                         // backward distance; < 0 = not visible
    if (j < C) {
        dist = C + depth[i] - j;
    } else {
        const int row = j - C;
        dist = ((bits[i] >> uint(row)) & 1u) ? depth[i] - depth[row] : -1;
    }"""
_MASK_SOURCE = r"""
    const int j = int(thread_position_in_grid.x);     // key; j == S is the sink's column
    const int i = int(thread_position_in_grid.y);     // query row
    const int h = int(thread_position_in_grid.z);     // head
    const int S = dims[0];
    const int LQ = rel_shape[1];
    const int H = rel_shape[2];
    const size_t at = (size_t(h) * LQ + i) * (S + 1) + j;
    if (j == S) {
        out[at] = sinks[size_t(i) * H + h];
        return;
    }
    /*DIST*/
    T val;
    if (dist < 0) {
        val = (T)(-1e30f);                            // causal
    } else if (SLIDING > 0 && dist >= SLIDING) {
        val = (T)(-1e30f);                            // sliding-window cap
    } else if (dist < REL_EXTENT) {
        float acc = 0.0f;
        const size_t rbase = (size_t(i) * H + h) * D_REL;
        const size_t pbase = size_t(dist) * D_REL;
        for (uint d = 0; d < D_REL; ++d)
            acc += (float)rel[rbase + d] * (float)proj[pbase + d];
        val = (T)acc;
    } else {
        val = (T)0;                                   // in context, outside the band
    }
    out[at] = val;
"""
_MASK = "_mlx_dspark_mask"          # per attention module: the one-launch mask is on
_REL_T = "_mlx_dspark_rel_t"        # per attention module: (rel_proj it was built from, its transpose)


@lru_cache(maxsize=2)
def _mask_kernel(tree: bool = False):
    return mx.fast.metal_kernel(
        name="mlx_dspark_inkling_mask_sink" + ("_tree" if tree else ""),
        input_names=["rel", "proj", "sinks", "dims"] + (["depth", "bits"] if tree else []),
        output_names=["out"],
        source=_MASK_SOURCE.replace("/*DIST*/", _MASK_DIST_TREE if tree else _MASK_DIST),
    )


@lru_cache(maxsize=4096)
def _dims(size: int) -> mx.array:
    d = mx.array([size], mx.int32)
    mx.eval(d)
    return d


def _rel_transposed(attn, dtype) -> mx.array:
    """``rel_proj`` as ``[extent, d_rel]`` in the stream's dtype, built once per layer."""
    held = attn.__dict__.get(_REL_T)
    if held is None or held[0] is not attn.rel_proj or held[1].dtype != dtype:
        t = mx.contiguous(attn.rel_proj.astype(dtype).T)
        mx.eval(t)
        held = attn.__dict__[_REL_T] = (attn.rel_proj, t)
    return held[1]


def fused_mask(rel: mx.array, proj_t: mx.array, sinks: mx.array, size: int, sliding: int,
               rel_extent: int, tree=None) -> mx.array:
    """mlx-vlm's ``banded_additive_mask`` for the last ``L`` of ``size`` keys with the sink's
    logit as column ``size``: ``[1, H, L, size + 1]`` from ``rel [1, L, H, d_rel]``, ``proj_t
    [extent, d_rel]`` and ``sinks [1, L, H]`` (all one dtype). ``tree`` (a :class:`TreeRound`):
    the ``L`` rows are a verify tree's — positions by depth, of each other only ancestors."""
    _, rows, heads, d_rel = rel.shape
    return _mask_kernel(tree is not None)(
        inputs=([rel, proj_t, sinks, _dims(int(size))]
                + ([tree.depth, tree.bits] if tree is not None else [])),
        template=[("T", rel.dtype), ("D_REL", int(d_rel)), ("REL_EXTENT", int(rel_extent)),
                  ("SLIDING", int(sliding))],
        grid=(size + 1, rows, heads),
        threadgroup=(min(256, size + 1), 1, 1),
        output_shapes=[(1, heads, rows, size + 1)],
        output_dtypes=[rel.dtype],
    )[0]


def _attention_mask(attn, lang, r, sinks, size, length, shape_ref, tree=None):
    """The attention's additive mask with the sink's column: one launch when the self-test
    cleared it for this model, the stock kernel + concatenate otherwise (a tree verify has
    only the launch: the stock mask is a chain's)."""
    if attn.__dict__.get(_MASK) and sinks.dtype == r.dtype and r.shape[0] == 1:
        if tree is not None:
            tree.masks += 1
        return fused_mask(r, _rel_transposed(attn, r.dtype), sinks, size, attn.sliding,
                          attn.rel_extent, tree)
    if tree is not None:
        raise RuntimeError("Inkling tree verify needs the one-launch attention mask")
    mask = lang.banded_additive_mask(r, attn.rel_proj.astype(r.dtype), size - length, size,
                                     attn.sliding, attn.rel_extent, left_padding=None,
                                     shape_ref=shape_ref)
    return mx.concatenate([mask, sinks.transpose(0, 2, 1)[..., None]], axis=-1)


def set_mask(language_model, enabled: bool) -> None:
    """Switch the one-launch mask on or off for every attention module (the A/B; the self-test
    decides at load)."""
    for layer in language_model.model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "rel_proj"):
            attn.__dict__[_MASK] = bool(enabled)


# ---- attention at verify widths: K and V read once per KV head -------------------------------------
# At L query rows MLX's vector SDPA gives every (query head, row) its own pass over the keys: with
# G query heads per KV head that is G * L reads of the same K and V, plus a mask array of
# [heads, L, S] built first. Here all G * L rows of a KV head share one pass: the rows sit on
# 8 x 8 simdgroup matrices (8 rows a simdgroup), the keys arrive 16 at a time straight from the
# cache's view, the relative-position bias is computed where the mask would have been read, and
# the sink's logit and the output gate are folded into the second pass. Two launches (a partial
# softmax per block of keys, then the combine) for the stock path's mask launch, SDPA and gate.
#
# NOT bit-identical to the stock call: the same products (scores from bf16 q and k, exact in
# fp32; the bias rounded to the stream's dtype as the mask array is; fp32 weights times the
# values) reduced over the keys in another grouping, and the softmax's running maximum taken per
# tile. The fp-tie class of ``res`` / ``ctrl``. Used for calls of PACKED_MIN_ROWS to
# PACKED_MAX_ROWS rows over PACKED_MIN_KEYS+ keys (where it wins); a decode step, and everything
# else, keeps the stock, bit-identical call.
# Derived from MLX's own two-pass sdpa_vector (mlx/backend/metal/kernels/sdpa_vector.h, MIT,
# © 2024 Apple Inc.).
# Where it wins (M4 Pro, paired in the verify loop): nothing or a small loss under 1,024 keys,
# +3% of a 3-row verify and +4-5% of a 4-row one from there on, a tie at 2 rows. 1,024 keys with
# the sink is also where MLX's SDPA switches to its two-pass kernel, whose per-block partial
# outputs are rounded to the stream's dtype: from there the stock call is ~3e-3 (relative) from
# the fp32 arithmetic and this kernel ~1e-6; under it the two agree to ~1e-6.
# At 9-12 rows it still leads MLX's attention there (the matmul path: -3 ms of a verify at 2,600
# tokens); from ~16 rows MLX's GEMMs are the faster way to do this arithmetic (+5 to +10 ms for
# this kernel at 16-33 rows, and x0.5-0.8 at a prefill chunk's 512), so it stops at 12.
PACKED_MIN_ROWS = 3
PACKED_MAX_ROWS = 12
PACKED_MIN_KEYS = 1024                  # keys counting the sink's
_PACKED = "_mlx_dspark_packed_attn"     # per attention module: the key count it starts at (0 = off)
_PACKED_TILE = 16                       # keys per step: two 8 x 8 score matrices
_PACKED_BLOCKS = 16                     # key blocks per KV head (threadgroups working one head)


def _packed_p1_source(nh: int, nkv: int, hd: int, drel: int, scale: float, rel_extent: int,
                      sliding: int, tree: bool = False) -> str:
    g = nh // nkv
    # a tree verify: the row's position is the anchor's plus its depth, and of the tree's own
    # rows (the last QL keys) it sees itself and its ancestors (see _MASK_DIST_TREE)
    q_abs = ("""const int n_ctx = TOTAL_S - QL;                    // committed keys
    const int q_depth = depth[qi];
    const uint q_bits = bits[qi];""" if tree else "const int q_abs = (TOTAL_S - QL) + qi;")
    dist = ("""int dist;                                           // backward distance
            if (key < n_ctx) {
                dist = n_ctx + q_depth - key;
            } else {
                const int trow = key - n_ctx;
                dist = (trow < QL && ((q_bits >> uint(trow)) & 1u)) ? q_depth - depth[trow] : -1;
            }""" if tree else "const int dist = q_abs - key;                       // backward distance")
    bias = " + ".join(f"dot(rv{i}, as_type<float4>(uint4(*(const device packed_ushort4*)"
                      f"(p + {4 * i})) << 16))" for i in range(drel // 4))
    rload = "\n".join(
        f"    const float4 rv{i} = as_type<float4>(uint4(*(const device packed_ushort4*)"
        f"(rp + {4 * i})) << 16);" for i in range(drel // 4))
    return f"""
    constexpr int DH = {hd};
    constexpr int DT = DH / 8;
    constexpr int DREL = {drel};
    constexpr int NH = {nh};
    constexpr int G = {g};
    constexpr int SLIDING = {sliding};
    constexpr int REL_EXTENT = {rel_extent};
    const int hk = int(threadgroup_position_in_grid.x);
    const int blk = int(threadgroup_position_in_grid.z);
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int NB = meta[0];
    const int TOTAL_S = meta[1];
    const int chunk = meta[2];
    const int k0 = blk * chunk;
    const int k1 = min(TOTAL_S, k0 + chunk);
    // this thread's two elements of an 8 x 8 simdgroup matrix: row fm, columns fn and fn + 1
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    // a matrix row is one (query head of this KV head, query row); spare rows repeat the last
    const int row = int(sg) * 8 + fm;
    const int valid_row = min(row, G * QL - 1);
    const int qi = valid_row % QL;
    const int hq = hk * G + valid_row / QL;
    {q_abs}

    const device bfloat16_t* qrow = q + (size_t(qi) * NH + hq) * DH;
    simdgroup_matrix<bfloat16_t, 8, 8> Qf[DT];
    #pragma unroll
    for (int dt = 0; dt < DT; ++dt) {{
        Qf[dt].thread_elements()[0] = qrow[dt * 8 + fn];
        Qf[dt].thread_elements()[1] = qrow[dt * 8 + fn + 1];
    }}
    simdgroup_matrix<float, 8, 8> Of[DT];
    #pragma unroll
    for (int dt = 0; dt < DT; ++dt) Of[dt] = simdgroup_matrix<float, 8, 8>(0.0f);

    // the row's relative-position code (bf16 read as raw bits: a bf16 is the top half of its fp32)
    const device uint16_t* rp = (const device uint16_t*)r + (size_t(qi) * NH + hq) * DREL;
{rload}
    const device uint16_t* relp = (const device uint16_t*)rel_t;

    const int64_t kss = k_buf_strides[2], vss = v_buf_strides[2];
    const device bfloat16_t* kb = k_buf + int64_t(hk) * k_buf_strides[1];
    const device bfloat16_t* vb = v_buf + int64_t(hk) * v_buf_strides[1];
    threadgroup bfloat16_t kstage[16 * DH];
    threadgroup bfloat16_t vstage[16 * DH];

    float m = -1e30f, l = 0.0f;
    for (int kt = k0; kt < k1; kt += 16) {{
        const bool full = k1 - kt >= 16;
        simdgroup_matrix<float, 8, 8> S0 = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> S1 = simdgroup_matrix<float, 8, 8>(0.0f);
        if (full) {{
            const device bfloat16_t* kp = kb + int64_t(kt) * kss;
            for (int dt = 0; dt < DT; ++dt) {{
                simdgroup_matrix<bfloat16_t, 8, 8> K0, K1;
                simdgroup_load(K0, kp + dt * 8, kss, ulong2(0, 0), true);
                simdgroup_load(K1, kp + 8 * kss + dt * 8, kss, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S0, Qf[dt], K0, S0);
                simdgroup_multiply_accumulate(S1, Qf[dt], K1, S1);
            }}
        }} else {{
            // the block's last keys: staged in threadgroup memory, zero padded to a tile
            const uint tid = thread_position_in_threadgroup.x;
            const uint nthr = threads_per_threadgroup.x;
            for (uint idx = tid; idx < 16 * (DH / 4); idx += nthr) {{
                const int j = int(idx) / (DH / 4);
                const int d4 = (int(idx) % (DH / 4)) * 4;
                if (kt + j < k1) {{
                    *(threadgroup packed_ushort4*)(kstage + j * DH + d4) = *(const device packed_ushort4*)
                        ((const device uint16_t*)(kb + int64_t(kt + j) * kss) + d4);
                    *(threadgroup packed_ushort4*)(vstage + j * DH + d4) = *(const device packed_ushort4*)
                        ((const device uint16_t*)(vb + int64_t(kt + j) * vss) + d4);
                }} else {{
                    *(threadgroup packed_ushort4*)(kstage + j * DH + d4) = packed_ushort4(0);
                    *(threadgroup packed_ushort4*)(vstage + j * DH + d4) = packed_ushort4(0);
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int dt = 0; dt < DT; ++dt) {{
                simdgroup_matrix<bfloat16_t, 8, 8> K0, K1;
                simdgroup_load(K0, kstage + dt * 8, DH, ulong2(0, 0), true);
                simdgroup_load(K1, kstage + 8 * DH + dt * 8, DH, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S0, Qf[dt], K0, S0);
                simdgroup_multiply_accumulate(S1, Qf[dt], K1, S1);
            }}
        }}

        // this thread's four scores: mask, scale, bias — then the row's running softmax
        float s[4] = {{S0.thread_elements()[0], S0.thread_elements()[1],
                      S1.thread_elements()[0], S1.thread_elements()[1]}};
        float top = -INFINITY;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {{
            const int key = kt + (j >> 1) * 8 + fn + (j & 1);
            {dist}
            if (key >= k1 || dist < 0 || (SLIDING > 0 && dist >= SLIDING)) {{
                s[j] = -INFINITY;                               // padding, causal, window cap
            }} else {{
                float bias = 0.0f;
                if (dist < REL_EXTENT) {{
                    const device uint16_t* p = relp + size_t(dist) * DREL;
                    bias = float(bfloat16_t({bias}));
                }}
                s[j] = s[j] * {scale!r}f + bias;
            }}
            top = max(top, s[j]);
        }}
        top = max(top, simd_shuffle_xor(top, 1));               // the row's four lanes
        top = max(top, simd_shuffle_xor(top, 8));
        const float mn = max(m, top);
        const float f = metal::fast::exp(m - mn);
        float ps = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {{
            s[j] = metal::fast::exp(s[j] - mn);
            ps += s[j];
        }}
        ps += simd_shuffle_xor(ps, 1);
        ps += simd_shuffle_xor(ps, 8);
        l = l * f + ps;
        m = mn;
        // fp32 weights times the values, as the stock kernel multiplies them: each thread
        // widens its own two elements of a value tile on load
        simdgroup_matrix<float, 8, 8> P0, P1;
        P0.thread_elements()[0] = s[0];
        P0.thread_elements()[1] = s[1];
        P1.thread_elements()[0] = s[2];
        P1.thread_elements()[1] = s[3];
        const device bfloat16_t* vp = vb + int64_t(kt + fm) * vss + fn;
        const threadgroup bfloat16_t* vs = vstage + int(fm) * DH + fn;
        for (int dt = 0; dt < DT; ++dt) {{
            Of[dt].thread_elements()[0] *= f;
            Of[dt].thread_elements()[1] *= f;
            simdgroup_matrix<float, 8, 8> V0, V1;
            if (full) {{
                V0.thread_elements()[0] = float(vp[dt * 8]);
                V0.thread_elements()[1] = float(vp[dt * 8 + 1]);
                V1.thread_elements()[0] = float(vp[8 * vss + dt * 8]);
                V1.thread_elements()[1] = float(vp[8 * vss + dt * 8 + 1]);
            }} else {{
                V0.thread_elements()[0] = float(vs[dt * 8]);
                V0.thread_elements()[1] = float(vs[dt * 8 + 1]);
                V1.thread_elements()[0] = float(vs[8 * DH + dt * 8]);
                V1.thread_elements()[1] = float(vs[8 * DH + dt * 8 + 1]);
            }}
            simdgroup_multiply_accumulate(Of[dt], P0, V0, Of[dt]);
            simdgroup_multiply_accumulate(Of[dt], P1, V1, Of[dt]);
        }}
        if (!full) threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    const int64_t pbase = (int64_t(hk) * NB + blk) * RP;
    device float* op = opart + (pbase + int(sg) * 8) * DH;
    for (int dt = 0; dt < DT; ++dt) simdgroup_store(Of[dt], op + dt * 8, DH);
    if (fn == 0) {{
        mpart[pbase + row] = m;
        lpart[pbase + row] = l;
    }}
"""


def _packed_p2_source(nh: int, nkv: int, hd: int) -> str:
    return f"""
    constexpr int DH = {hd};
    constexpr int NH = {nh};
    constexpr int G = {nh // nkv};
    const int d = int(thread_position_in_grid.x);
    const int r = int(thread_position_in_grid.y);
    const int hk = int(thread_position_in_grid.z);
    const int NB = meta[0];
    const int qi = r % QL;
    const int hq = hk * G + r / QL;
    const device float* mp = mpart + int64_t(hk) * NB * RP + r;
    const device float* lp = lpart + int64_t(hk) * NB * RP + r;
    const device float* op = opart + (int64_t(hk) * NB * RP + r) * DH + d;
    // the sink: a key of zeros whose logit is the row's own — it only adds to the denominator
    const float sk = float(sinks[size_t(qi) * NH + hq]);
    float M = sk;
    for (int j = 0; j < NB; ++j) M = max(M, mp[j * RP]);
    float den = metal::fast::exp(sk - M);
    float num = 0.0f;
    for (int j = 0; j < NB; ++j) {{
        const float w = metal::fast::exp(mp[j * RP] - M);
        den += w * lp[j * RP];
        num += w * op[int64_t(j) * RP * DH];
    }}
    // the attention's output in the stream's dtype, then the gate (mlx-vlm: out * sigmoid(gate))
    const float a = float(bfloat16_t(num / den));
    const size_t at = (size_t(qi) * NH + hq) * DH + d;
    const float gv = float(gate[at]);
    const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(gv)));
    out[at] = bfloat16_t(a * ((gv < 0.0f) ? y : 1.0f - y));
"""


@lru_cache(maxsize=32)
def _packed_kernels(nh: int, nkv: int, hd: int, drel: int, scale: float, rel_extent: int,
                    sliding: int, tree: bool = False):
    p1 = mx.fast.metal_kernel(
        name=(f"mlx_dspark_inkling_packed_attn_p1_{nh}_{nkv}_{hd}_{drel}_{rel_extent}_{sliding}"
              + ("_tree" if tree else "")),
        input_names=["q", "k_buf", "v_buf", "r", "rel_t", "meta"] + (["depth", "bits"] if tree else []),
        output_names=["opart", "mpart", "lpart"],
        ensure_row_contiguous=False,
        source=_packed_p1_source(nh, nkv, hd, drel, scale, rel_extent, sliding, tree),
    )
    p2 = mx.fast.metal_kernel(
        name=f"mlx_dspark_inkling_packed_attn_p2_{nh}_{nkv}_{hd}",
        input_names=["opart", "mpart", "lpart", "sinks", "gate", "meta"],
        output_names=["out"],
        source=_packed_p2_source(nh, nkv, hd),
    )
    return p1, p2


@lru_cache(maxsize=8192)
def _packed_meta(size: int) -> tuple[mx.array, int]:
    """``([blocks, keys, keys per block], blocks)`` for a call over ``size`` keys."""
    blocks = _PACKED_BLOCKS // 2 if size <= 256 else _PACKED_BLOCKS
    per_block = math.ceil(size / blocks)
    chunk = max(_PACKED_TILE, math.ceil(per_block / _PACKED_TILE) * _PACKED_TILE)   # whole tiles
    nb = math.ceil(size / chunk)
    meta = mx.array([nb, size, chunk], mx.int32)
    mx.eval(meta)
    return meta, nb


def _packed_ok(attn, length: int) -> bool:
    """Whether the packed kernel was written for this attention's shape: whole 8-wide tiles of
    the head dimension (and two staged 16-key tiles of it inside a threadgroup's 32 KB), the
    relative code in float4s, and the rows of a KV head (``length`` per query head) in one
    threadgroup."""
    d_rel = int(attn.rel_proj.shape[0])
    return (attn.n_heads % attn.n_kv == 0 and attn.head_dim % 8 == 0 and attn.head_dim <= 256
            and d_rel % 4 == 0 and length >= 1
            and math.ceil(attn.n_heads // attn.n_kv * length / 8) * 32 <= 1024)


def packed_attention(attn, q: mx.array, k: mx.array, v: mx.array, r: mx.array, rel_t: mx.array,
                     sinks: mx.array, gate: mx.array, tree=None) -> mx.array:
    """The gated attention output ``[1, L, heads * head_dim]`` (what mlx-vlm hands to ``o_proj``)
    for ``q [1, L, heads, head_dim]``, the cache's ``k`` / ``v [1, kv_heads, S, head_dim]`` in
    temporal order WITHOUT the sink row (views are read where they lie), ``r [1, L, heads,
    d_rel]``, ``rel_t [extent, d_rel]``, ``sinks [1, L, heads]`` and the gate projection
    ``[1, L, heads * head_dim]`` — all bf16. ``tree`` (a :class:`TreeRound`): the ``L`` rows are
    a verify tree's, the last ``L`` keys theirs."""
    _, length, nh, hd = q.shape
    nkv, size = int(k.shape[1]), int(k.shape[2])
    rows = (nh // nkv) * length
    nsg = math.ceil(rows / 8)
    rp = nsg * 8
    p1, p2 = _packed_kernels(nh, nkv, hd, int(r.shape[-1]), float(hd) ** -0.5,
                             int(attn.rel_extent), int(attn.sliding or 0), tree is not None)
    meta, nb = _packed_meta(size)
    template = [("QL", length), ("RP", rp)]
    opart, mpart, lpart = p1(
        inputs=[q, k, v, r, rel_t, meta] + ([tree.depth, tree.bits] if tree is not None else []),
        template=template,
        grid=(nkv * 32 * nsg, 1, nb),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(nkv, nb, rp, hd), (nkv, nb, rp), (nkv, nb, rp)],
        output_dtypes=[mx.float32] * 3,
    )
    return p2(
        inputs=[opart, mpart, lpart, sinks, gate, meta],
        template=template,
        grid=(hd, rows, nkv),
        threadgroup=(hd, 1, 1),
        output_shapes=[(1, length, nh * hd)],
        output_dtypes=[mx.bfloat16],
    )[0]


def _packed_reference(attn, q, k, v, r, rel_t, sinks, gate, tree=None) -> mx.array:
    """What :func:`packed_attention` computes, in plain fp32 MLX ops with nothing rounded
    inside: the additive mask with the sink's column (in the stream's dtype, as stock builds
    it), fp32 scores, softmax and values, the output rounded once, then the gate. The
    self-test's yardstick — the stock call itself rounds more (bf16 partials in MLX's two-pass
    kernel from 1,024 keys, a bf16 matmul where no vector kernel covers the head width)."""
    _, rows, nh, hd = q.shape
    g = nh // k.shape[1]
    zero = _zero_row(1, int(k.shape[1]), hd, k.dtype)
    mask = fused_mask(r, rel_t, sinks, int(k.shape[2]), attn.sliding, attn.rel_extent, tree)
    kf = mx.repeat(mx.concatenate([k, zero], axis=2), g, axis=1).astype(mx.float32)
    vf = mx.repeat(mx.concatenate([v, zero], axis=2), g, axis=1).astype(mx.float32)
    scores = (q.transpose(0, 2, 1, 3).astype(mx.float32) @ kf.swapaxes(-1, -2)) * hd ** -0.5
    out = mx.softmax(scores + mask.astype(mx.float32), axis=-1) @ vf
    return _gated(out.astype(q.dtype).transpose(0, 2, 1, 3).reshape(1, rows, -1), gate)


def set_packed(language_model, enabled: bool, min_keys: int | None = None) -> None:
    """Switch the packed verify attention on or off for every attention module it was written
    for (the A/B; the self-test decides at load). ``min_keys``: the key count, sink included,
    it starts at (default ``PACKED_MIN_KEYS``)."""
    start = int(PACKED_MIN_KEYS if min_keys is None else min_keys) if enabled else 0
    for layer in language_model.model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "rel_proj"):
            attn.__dict__[_PACKED] = start if _packed_ok(attn, PACKED_MAX_ROWS) else 0


def packed_attn(language_model) -> bool:
    """Whether the packed verify attention is on for this model."""
    return any(layer.self_attn.__dict__.get(_PACKED) for layer in language_model.model.layers
               if getattr(layer, "self_attn", None) is not None)


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
                and x.ndim == 3 and x.shape[0] == 1 and x.shape[1] <= _rows(self)
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
        tree = _TREE                    # a tree verify's round (TreeRound), else None
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
                steps = mx.arange(length) if tree is None else tree.depth
                positions = steps + mx.array(offset)[..., None] + 1
                tau = 1 + cfg.log_scaling_alpha * mx.log(
                    mx.maximum(positions / cfg.log_scaling_n_floor, 1))
                q = (q.astype(mx.float32) * tau[..., None, None]).astype(x.dtype)
                r = (r.astype(mx.float32) * tau[..., None, None]).astype(x.dtype)
        k, v = k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)
        if tree is not None:
            if not is_sink_cache(kv):
                raise RuntimeError("Inkling tree verify needs the in-place KV caches")
            tree.kvs.append((kv, k, v))          # the rows as written: a commit re-appends them
        k, v = kv.update_and_fetch(k, v)
        sinks = self.sink_proj(x)
        packed_from = self.__dict__.get(_PACKED)
        if (packed_from and PACKED_MIN_ROWS <= length <= PACKED_MAX_ROWS
                and k.shape[2] + 1 >= packed_from and is_sink_cache(kv)
                and sinks.dtype == r.dtype == k.dtype == v.dtype == mx.bfloat16):
            # a verify over a long window: all rows of a KV head share one read of K and V
            # (NOT bit-identical to the stock call below — see packed_attention)
            if tree is not None:
                tree.masks += 1                  # the kernel takes the tree's mask itself
            return self.o_proj(packed_attention(
                self, q, k, v, r, _rel_transposed(self, r.dtype), sinks, self.gate_proj(x), tree))
        q = q.transpose(0, 2, 1, 3)
        if is_sink_cache(kv):
            # inkling_cache: the rows are in temporal order and the sink's zero row sits in the
            # buffer behind them, so K and V are VIEWS — the same keys in the same order into
            # the same SDPA kernel, nothing copied
            size = k.shape[2]
            mask = _attention_mask(self, lang, r, sinks, size, length, k, tree)
            k, v = kv.sink_views()
        elif callable(getattr(kv, "with_sink", None)):
            # mlx-vlm's own sink cache (its looped model's since 2026-11-22): its views
            size = k.shape[2]
            mask = _attention_mask(self, lang, r, sinks, size, length, k)
            k, v = kv.with_sink(size)
        else:
            # [keys in temporal order ‖ the sink's zero row] as ONE concatenate (stock: the
            # ring's reorder, a zeros fill, then another copy of the whole K and V)
            rotating = isinstance(kv, lang.RotatingKVCache)
            k_parts = _temporal_parts(kv, k) if rotating else [k]
            v_parts = _temporal_parts(kv, v) if rotating else [v]
            size = sum(part.shape[2] for part in k_parts)
            mask = _attention_mask(self, lang, r, sinks, size, length,
                                   k if k.shape[2] == size else k_parts[0])
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


# ---- the layer-visit loop -------------------------------------------------------------------------
# mlx-vlm's TextModel.__call__ (models/inkling2/language.py, Apache-2.0) for decode / verify
# widths. Nothing numeric changes; two things about how the graph reaches the GPU do:
#   * A loop pass's messages for ALL its layers come from one read. The bank a pass reads is
#     fixed when the pass starts, so the stock loop's per-layer read (two matmuls and a softmax,
#     24 times a token) is the same rows of one larger product — bit-identical on the kernels
#     MLX picks here, which the self-test checks on every load.
#   * The graph is handed to the GPU every few visits at decode width too. Stock does that only
#     inside a speculative verify (every 4 visits); a single-row forward was encoded whole before
#     the GPU saw any of it, which a pipelined decode loop hides and an unpipelined step (a
#     verify of one row, a parked round's first step, calibration) pays in full.
# Visits between hand-offs (stock's value inside a verify). Measured on an M4 Pro, decode width:
# pipelined 29.6 ms/step with the stock loop, 28.4-28.5 at every 4 / 8 / 12 / 16 / 24; a step
# built and evaluated alone 36.7 -> 30.4 / 30.7 / 31.6 / 32.1 / 33.5.
SCHEDULE_EVERY = 4


def _model_call(base):
    def __call__(self, inputs, cache=None, input_embeddings=None, skip_final_norm=False,
                 capture_layer_ids=None, speculative_verify=False):
        ref = inputs if input_embeddings is None else input_embeddings
        visits = len(self.layer_indices)
        ok = (_on(self) and bool(cache) and len(cache) == visits and not self.training
              and ref.shape[0] == 1 and ref.shape[1] <= _rows(self) and _gpu())
        if ok and capture_layer_ids is not None:
            ok = (len(capture_layer_ids) == len(set(capture_layer_ids))
                  and all(type(i) is int and 0 <= i <= visits for i in capture_layer_ids))
        if not ok:
            return base.__call__(self, inputs, cache, input_embeddings, skip_final_norm,
                                 capture_layer_ids, speculative_verify)
        requested = set(capture_layer_ids or ())
        captures = {}
        x = self.embed(inputs) if input_embeddings is None else input_embeddings
        blocks = []
        loop = self.loop
        pass_messages = None
        per_layer = int(self.message_queries.shape[1]) if loop is not None else 0
        every = SCHEDULE_EVERY
        for visit, index in enumerate(self.layer_indices):
            if loop is not None and index == loop.core_start:
                if visit == index:
                    prelude, prelude_blocks = x, blocks.copy()
                else:
                    # The first block appended in the core is the prelude partial.
                    # Only completed core blocks and the last partial supply messages.
                    states = mx.stack([*blocks[len(prelude_blocks) + 1:], x], axis=-2)
                    bank = mx.fast.rms_norm(states.astype(mx.float32), None, 1e-6)
                    x, blocks = prelude, prelude_blocks.copy()
                    queries = self.message_queries
                    scores = (queries.reshape(-1, queries.shape[-1]) @ bank.swapaxes(-1, -2)
                              ) * x.shape[-1] ** -0.5
                    pass_messages = mx.softmax(scores, axis=-1) @ bank
            messages = None
            if pass_messages is not None and index < loop.core_end:
                at = (index - loop.core_start) * per_layer
                messages = pass_messages[..., at: at + per_layer, :]
            if visit in requested:
                captures[visit] = x
            x = self.layers[index](x, blocks, cache[visit], messages)
            if every and (visit + 1) % every == 0:
                mx.async_eval(x)
        h = self.final_input(blocks, x)[..., 0, :]
        normalized = self.norm(h).astype(x.dtype)
        if visits in requested:
            captures[visits] = normalized
        result = h if skip_final_norm else normalized
        if capture_layer_ids is not None:
            return result, [captures[i] for i in capture_layer_ids]
        return result

    return __call__


# ---- the routed experts --------------------------------------------------------------------------
def _moe_call(base):
    from . import inkling_kernels as K

    def __call__(self, x):
        if not (_on(self) and x.ndim == 3 and x.shape[0] == 1
                and 1 <= x.shape[1] <= max(K.MOE_MAX_ROWS, _rows(self)) and x.dtype == mx.bfloat16
                and not self.training and getattr(self, "sharding_group", None) is None
                and _gpu()):
            return base.__call__(self, x)
        return K.fused_moe(self, x)

    return __call__


# ---- install --------------------------------------------------------------------------------------
_CLASSES: dict[tuple[type, str], type] = {}


def _same_default(value, default) -> bool:
    return value is default or (type(value) is type(default)
                                and type(value) in (bool, int, float, str) and value == default)


def _tolerant(call, base):
    """``call`` (a fused ``__call__`` written against one mlx-vlm) behind the signature of the
    INSTALLED one. An argument ``call`` does not know is dropped when it carries the stock
    method's own default — that is the call ``call`` was written for — and sends the call to the
    stock method otherwise, as does anything else that cannot be matched. A newer mlx-vlm that
    grows a parameter then runs, on the stock path where it must, instead of raising."""
    import inspect

    try:
        known = inspect.signature(call).parameters
        stock = inspect.signature(base.__call__).parameters
    except (TypeError, ValueError):
        return call
    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    if any(p.kind not in kinds for p in (*known.values(), *stock.values())):
        return call
    names, stock_names = list(known), list(stock)
    if stock_names[:len(names)] != names:
        # renamed or reordered parameters: not the method ``call`` was copied from
        def __call__(self, *args, **kwargs):
            return base.__call__(self, *args, **kwargs)

        return __call__
    if len(stock_names) == len(names):
        return call
    limit = len(names) - 1                       # positional arguments ``call`` takes after self
    defaults = {name: p.default for name, p in stock.items() if name not in known}

    def __call__(self, *args, **kwargs):
        extra = [name for name in kwargs if name not in known]
        if len(args) > limit or not all(
                name in defaults and _same_default(kwargs[name], defaults[name]) for name in extra):
            return base.__call__(self, *args, **kwargs)
        for name in extra:
            del kwargs[name]
        return call(self, *args, **kwargs)

    return __call__


def _swap(module, make_call, tag: str) -> None:
    """Give ``module`` (one instance) a subclass of its own class whose ``__call__`` is fused."""
    base = type(module)
    if base.__name__.startswith(tag):
        return
    cls = _CLASSES.get((base, tag))
    if cls is None:
        cls = type(f"{tag}{base.__name__}", (base,), {"__call__": _tolerant(make_call(base), base)})
        _CLASSES[(base, tag)] = cls
    module.__class__ = cls


def _fused_modules(language_model):
    from .inkling_kernels import moe_spec

    model = language_model.model
    if hasattr(model, "layer_indices") and hasattr(model, "final_input"):
        yield model, "model"
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
           "attn": _attn_call, "moe": _moe_call, "model": _model_call}


_OFF = "_mlx_dspark_fused_off"      # kinds the self-test switched off for this model
EXACT_KINDS = ("sconv", "layer", "attn")       # bit-identical to the stock forward
LOOP_KINDS = ("model",)                        # the layer-visit loop: bit-identical, whole forward
ROW_EXACT_KINDS = ("moe",)                     # each row bit-identical to the stock DECODE call
ROUNDING_KINDS = ("res", "ctrl")               # fp32 dot products summed in another order
ALL_KINDS = EXACT_KINDS + LOOP_KINDS + ROW_EXACT_KINDS + ROUNDING_KINDS
# Not module kinds: whether new caches are the in-place ones of inkling_cache, and whether the
# attention's mask and sink column come from one launch (both bit-identical, both self-tested)
KV_KIND = "kv"
MASK_KIND = "mask"
# ... whether the pieces also hold for calls wider than a drafter round (up to MAX_ROWS: a lookup
# round's long copy draft, a short prompt), each against the stock forward at that width
WIDE_KIND = "wide"
# ... and whether the bit-exact ones hold at a prefill chunk's width (CHUNK_ROWS), where they run
# around the STOCK experts, residual mixes and controllers: the stock forward's bits
CHUNK_KIND = "chunk"
CHUNK_KINDS = EXACT_KINDS + LOOP_KINDS
# ... and whether a verify over a long window takes the packed attention (within rounding of the
# stock call; needs the in-place caches)
PACKED_KIND = "vattn"
PACKED_TOL = 2e-3          # vs the fp32 arithmetic (measured: 1e-7 to 3e-4)
PACKED_TOL_STOCK = 2e-2    # vs the stock module, which rounds more inside (measured: up to 4e-3)
_KV = "_mlx_dspark_kv_caches"


def set_rows(language_model, rows: int, chunk: int | None = None) -> None:
    """The positions per call the fused pieces accept. ``rows``, every piece: ``VECTOR_ROWS`` (a
    drafter round) or ``MAX_ROWS`` (a lookup round and a short prompt too). ``chunk``, the
    bit-exact pieces alone (``CHUNK_KINDS``): up to ``CHUNK_ROWS``, a prefill chunk; ``None`` =
    no wider than ``rows``. The self-test decides both at load; this is the A/B."""
    rows = max(1, min(int(rows), MAX_ROWS))
    chunk = max(rows, min(int(chunk or 0), CHUNK_ROWS))
    for module, kind in _fused_modules(language_model):
        module.__dict__[_ROWS] = chunk if kind in CHUNK_KINDS else rows
    language_model.__dict__[_ROWS] = rows
    language_model.__dict__[_CHUNK] = chunk


def rows_limit(language_model) -> int:
    """The widest call EVERY fused piece takes on this model (a verify, a short prompt)."""
    return language_model.__dict__.get(_ROWS, VECTOR_ROWS)


def chunk_limit(language_model) -> int:
    """The widest call the bit-exact pieces take on this model (a prefill chunk where the
    self-test cleared it; else :func:`rows_limit`)."""
    return max(rows_limit(language_model), language_model.__dict__.get(_CHUNK, 0))


def kv_caches(language_model) -> bool:
    """Whether the self-test cleared the in-place KV caches for this model."""
    return bool(language_model.__dict__.get(_KV, False))


def make_cache(language_model) -> list:
    """The model's layer-visit caches, with the in-place KV caches where the self-test cleared
    them (``inkling_cache``); the stock mlx-vlm caches otherwise."""
    cache = language_model.make_cache()
    if kv_caches(language_model):
        from .inkling_cache import convert

        convert(cache)
    return cache


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
    ``"attn"``, ``"moe"``, ``"model"``). A kind the self-test rejected (or :func:`disable`
    named) stays off."""
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


@contextmanager
def stock(language_model):
    """Run what is inside on mlx-vlm's STOCK forward — every fused piece, the in-place KV
    caches, the packed attention and the head kernels off — and put all of it back on exit,
    as it was (what the self-test rejected stays rejected). What a comparison against the
    stock runtime runs (the server's ``stock`` race arm). Caches made inside are mlx-vlm's
    own: a cache must not cross the boundary in either direction."""
    from . import inkling_kernels as K

    lm = language_model
    missing = object()
    flags = [(module, module.__dict__.get(_FLAG, missing)) for module, _ in _fused_modules(lm)]
    attns = [layer.self_attn for layer in lm.model.layers
             if getattr(layer, "self_attn", None) is not None]
    packed = [(attn, attn.__dict__.get(_PACKED, missing)) for attn in attns]
    held = {name: lm.__dict__.get(name, missing)
            for name in (_KV, K._FAST_HEAD, K._DENSE_HEAD_KERNEL)}
    try:
        for module, _ in flags:
            module.__dict__[_FLAG] = False
        for attn in attns:
            attn.__dict__[_PACKED] = 0
        for name in held:
            lm.__dict__[name] = False
        yield
    finally:
        for owner, name, value in ([(m, _FLAG, v) for m, v in flags]
                                   + [(a, _PACKED, v) for a, v in packed]
                                   + [(lm, n, v) for n, v in held.items()]):
            if value is missing:
                owner.__dict__.pop(name, None)
            else:
                owner.__dict__[name] = value


def _moe_selftest(language_model, rand, wide: bool = False) -> bool:
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
        plans = [[a, a, a, a, a, b, b, c], [a, b, b, b, c], [c]]
        if wide:        # a lookup round: more rows than one routing threadgroup, runs past a tile
            plans = [[a] * 6 + [b] * 9 + [c] * (MAX_ROWS - 15)]
        for rows in plans:
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


def tree_selftest(language_model, seed: int = 0) -> bool:
    """The tree variants of the kernels on this model's own shapes and weights (one sliding and
    one global layer), before a model-level check (``Target.tree_selftest``) runs whole
    forwards. Two exact properties each:

    * on a CHAIN-shaped tree a tree kernel returns the chain kernel's bits — it evaluates the
      same expression;
    * on a branched tree, row ``i`` holds the bits the chain kernel gives the LAST row of a call
      over ``i``'s root path — the convolutions (static with the folded residual, dynamic), and
      the mask over committed keys on both sides of the sliding cap and of the
      relative-position extent, with every row outside the path masked.

    The packed attention is not exact against anything (see :func:`packed_attention`): on the
    chain-shaped tree it must return the chain kernel's bits, on the branched one stay within
    ``PACKED_TOL`` of the same arithmetic in fp32 ops under the tree's mask."""
    from .tree import DraftTree, chain_tree, to_device

    lm = language_model
    layers = lm.model.layers
    keys = iter(mx.random.split(mx.random.key(seed), 256))

    def rand(shape, dtype=mx.bfloat16):
        return mx.random.normal(shape, key=next(keys)).astype(dtype)

    def same(a, b) -> bool:
        return a.shape == b.shape and bool(mx.array_equal(a, b))

    #                                  rows:  0  1  2  3  4  5  6  7
    host = DraftTree(tokens=list(range(8)), parent=[-1, 0, 1, 2, 0, 4, 1, 6])
    n = host.width
    tree = TreeRound(to_device(host, 4))
    chain = TreeRound(to_device(chain_tree(0, range(1, n)), 4))
    paths = [mx.array(host.path_to(i), dtype=mx.int32) for i in range(n)]

    def convs(layer) -> bool:
        attn = layer.self_attn
        hidden = int(layer.attn_sconv.conv.weight.shape[0])
        x, res = rand((1, n, hidden)), rand((1, n, hidden), mx.float32)
        state = rand((1, 3, hidden), mx.float32)
        weight = layer.attn_sconv.conv.weight

        def static(xs, rs, cidx=None):
            return fused_short_convolution(state, weight, xs, 4, rs, mx.float32, cidx)[0]

        got = static(x, res, tree.conv_idx)
        if not same(static(x, res, chain.conv_idx), static(x, res)):
            return False
        if not all(same(got[:, i], static(x[:, p], res[:, p])[:, -1])
                   for i, p in enumerate(paths)):
            return False
        conv = attn.k_sconv
        if hasattr(conv, "conv"):                # a sliding layer: its K / V convs are static
            return True
        channels = int(conv.static_weight.shape[1])
        xk, dyn_in = rand((1, n, channels)), rand((1, n, int(conv.weight_proj.weight.shape[1])))
        kstate = rand((1, 3, channels), mx.float32)

        def dynamic(xs, ds, cidx=None):
            length = int(xs.shape[1])
            return _dyn_kernel(cidx is not None)(
                inputs=([kstate, conv.static_weight, conv.weight_proj(ds), xs, _dyn_divisor(4)]
                        + ([cidx] if cidx is not None else [])),
                template=[("T", xs.dtype), ("W", 4), ("HEADS", conv.num_heads)],
                grid=(length * channels, 1, 1), threadgroup=(256, 1, 1),
                output_shapes=[xs.shape, (1, 3 + length, channels)],
                output_dtypes=[xs.dtype, mx.float32])[0]

        got = dynamic(xk, dyn_in, tree.conv_idx)
        return (same(dynamic(xk, dyn_in, chain.conv_idx), dynamic(xk, dyn_in))
                and all(same(got[:, i], dynamic(xk[:, p], dyn_in[:, p])[:, -1])
                        for i, p in enumerate(paths)))

    def masks(attn) -> bool:
        d_rel = int(attn.rel_proj.shape[0])
        rel_t = _rel_transposed(attn, mx.bfloat16)
        edge = attn.sliding or attn.rel_extent
        for ctx in sorted({5, edge - 3, edge + 4}):
            if ctx < 1:
                continue
            size = ctx + n
            r, sinks = rand((1, n, attn.n_heads, d_rel)), rand((1, n, attn.n_heads))

            def mask(rs, ss, keys_, t=None):
                return fused_mask(rs, rel_t, ss, keys_, attn.sliding, attn.rel_extent, t)

            if not same(mask(r, sinks, size, chain), mask(r, sinks, size)):
                return False
            got = mask(r, sinks, size, tree)
            for i, p in enumerate(paths):
                d = int(p.shape[0])
                ref = mask(r[:, p], sinks[:, p], ctx + d)[:, :, -1]      # [1, H, ctx + d + 1]
                row = got[:, :, i]                                       # [1, H, size + 1]
                on = mx.concatenate([mx.arange(ctx), ctx + p, mx.array([size])])
                off = mx.array([ctx + j for j in range(n) if j not in host.path_to(i)] or [0],
                               dtype=mx.int32)
                if not same(row[..., on], ref):
                    return False
                if len(host.path_to(i)) < n and float(row[..., off].astype(mx.float32).max()) > -1e29:
                    return False
        return True

    def packed(attn) -> bool:
        if not attn.__dict__.get(_PACKED):
            return True                          # off for this model: trees take the SDPA call
        d_rel = int(attn.rel_proj.shape[0])
        rel_t = _rel_transposed(attn, mx.bfloat16)
        edge = attn.sliding or attn.rel_extent
        size = (min(edge + 6, attn.sliding - 1) if attn.sliding else edge + 6) + n
        args = (rand((1, n, attn.n_heads, attn.head_dim)),
                rand((1, attn.n_kv, size, attn.head_dim)),
                rand((1, attn.n_kv, size, attn.head_dim)),
                rand((1, n, attn.n_heads, d_rel)), rel_t, rand((1, n, attn.n_heads)),
                rand((1, n, attn.n_heads * attn.head_dim)))
        if not same(packed_attention(attn, *args, chain), packed_attention(attn, *args)):
            return False
        return _rel_l2(packed_attention(attn, *args, tree),
                       _packed_reference(attn, *args, tree)) < PACKED_TOL

    sliding = next((layer for i, layer in enumerate(layers) if layer.is_sliding and i > 0), None)
    globl = next((layer for layer in layers if not layer.is_sliding), None)
    probes = [layer for layer in (sliding, globl) if layer is not None]
    try:
        return bool(probes) and all(
            convs(layer) and masks(layer.self_attn) and packed(layer.self_attn)
            for layer in probes)
    except Exception:  # noqa: BLE001 — a kernel this GPU cannot build: chains
        return False


def selftest(language_model, rows: int = 2, seed: int = 0) -> dict[str, bool]:
    """Compare every installed kind with the stock forward of THIS mlx-vlm on random inputs, and
    switch off any kind that does not hold: the layer and attention bodies are copies of
    mlx-vlm's, so an upstream change must turn them off, not change the model. ``sconv`` /
    ``layer`` / ``attn`` must match bit for bit (outputs and the cache states they leave);
    ``moe`` must match a single-row layer call bit for bit and, at more rows, the stock module's
    single-row call row by row (:func:`_moe_selftest`); ``res`` / ``ctrl`` within rounding. A
    kernel that does not build on this GPU (threadgroup limits) fails its kind the same way.
    ``kv`` and ``mask`` are not module kinds: ``kv`` says whether the in-place KV caches
    (``inkling_cache``) reproduce the stock caches bit for bit here, and :func:`make_cache` hands
    them out only then; ``mask`` whether the one-launch mask + sink column equals mlx-vlm's
    kernel and concatenate (:func:`set_mask`); ``vattn`` whether the packed verify attention
    (:func:`packed_attention`) stays within rounding of the stock call and leaks nothing from a
    later row into an earlier one (:func:`set_packed`); ``wide`` whether all of it also holds
    for calls of up to ``MAX_ROWS`` positions — a lookup round, a short prompt — which then take
    the fused pieces too (:func:`set_rows`; otherwise they stop at ``VECTOR_ROWS``, a drafter
    round); ``chunk`` whether the bit-exact pieces also match the stock forward at a prefill
    chunk's width (``CHUNK_ROWS``), which they then take around the stock experts.
    Leaves the model with every passing kind on. Returns ``{kind: passed}``."""
    lm = language_model
    model = lm.model
    layers = model.layers
    visits = list(model.layer_indices)
    hidden = int(model.embed_norm.weight.shape[0])
    key = mx.random.key(seed)
    keys = iter(mx.random.split(key, 512))

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
        if not windowed(kv):
            return True
        xs = [rand((1, 1, hidden), mx.bfloat16) for _ in range(steps)]
        blocks = [rand((1, 1, hidden), mx.bfloat16)] if layers[index].attention_input is not None else []
        runs = []
        for on in (False, True):
            set_enabled(lm, False)
            if on:
                set_enabled(lm, True, ("attn",))
            cache = lm.make_cache()[visits.index(index)]
            cache.caches = (small_window(kv, window), *cache.caches[1:])
            outs = [layers[index](x, list(blocks), cache, None) for x in xs]
            st = _state_arrays(cache)
            mx.eval(outs, st)
            runs.append((outs, st))
        (a, sa), (b, sb) = runs
        return (all(bool(mx.array_equal(u, v)) for u, v in zip(a, b)) and len(sa) == len(sb)
                and all(bool(mx.array_equal(u, v)) for u, v in zip(sa, sb)))

    def kv_run(index, window=5, plan=None, arms=((True, True), (True, False))):
        """The in-place caches (``inkling_cache``) against the stock ones on one layer, through
        the calls a generation makes: decode steps, a prefill-sized chunk, and verifies inside
        mlx-vlm's speculative transaction committed in part, in full and at one row — on a
        ``window``-key window with 2 spare rows, so it fills, slides and is rebuilt. Checked
        with the fused attention body (views with the sink row) and with the stock body (what a
        call past the fused limits takes). Outputs must match bit for bit."""
        from mlx_vlm.speculative.cache_state import start_speculative_cache

        from .inkling_cache import convert

        # (rows, rows kept): kept None = an ordinary call, else a verify committed there
        plan = plan or [(1, None), (1, None), (4, 2), (1, None), (3, 3), (4, 1), (1, None),
                        (6, None), (2, 1), (1, None), (5, 4), (3, 1), (1, None), (1, None)]
        xs = [rand((1, n, hidden), mx.bfloat16) for n, _ in plan]
        blocks = [rand((1, 1, hidden), mx.bfloat16)] if layers[index].attention_input is not None else []

        def run(new, fused):
            set_enabled(lm, False)
            if fused:
                set_enabled(lm, True, ("attn",))
            entry = lm.make_cache()[visits.index(index)]
            kv = entry.caches[0]
            if windowed(kv):
                entry.caches = (small_window(kv, window), *entry.caches[1:])
            if new:
                convert([entry])
                if not is_sink_cache(entry.caches[0]):
                    raise RuntimeError("no in-place cache for this layer's KV cache")
                entry.caches[0].step = 2
            outs = []
            for x, (n, keep) in zip(xs, plan):
                bl = [mx.broadcast_to(b, (1, n, hidden)) for b in blocks]
                if keep is None:
                    outs.append(layers[index](x, bl, entry, None))
                    mx.eval(outs[-1])
                    continue
                txn = start_speculative_cache([entry], n)
                outs.append(layers[index](x, bl, entry, None))
                mx.eval(outs[-1])
                txn.commit(keep)
            return outs

        ref = run(False, False)
        return all(bool(mx.array_equal(a, b))
                   for arm in arms
                   for a, b in zip(run(*arm), ref))

    def model_run(rows=rows, first=True):
        taps = [1, len(visits)]
        ids = ([mx.array([[3]])] if first else []) + [
            mx.array([[5, 7][:rows] + [11 + i for i in range(max(0, rows - 2))]])]
        runs = []
        for on in (False, True):
            set_enabled(lm, True)
            set_enabled(lm, on, ("model",))
            cache = make_cache(lm)
            outs = []
            for x in ids:
                out, caps = model(x, cache=cache, capture_layer_ids=taps)
                outs.extend([out, *caps])
                mx.eval(outs)
            st = _state_arrays([part for entry in cache for part in entry.caches])
            mx.eval(st)
            runs.append((outs, st))
        (a, sa), (b, sb) = runs
        return (all(bool(mx.array_equal(u, v)) for u, v in zip(a, b)) and len(sa) == len(sb)
                and all(bool(mx.array_equal(u, v)) for u, v in zip(sa, sb)))

    def mask_run(index):
        """The one-launch mask + sink column against mlx-vlm's kernel and concatenate on this
        layer's own projection: one row and several, key counts on both sides of the sliding
        cap and of the relative-position extent. Bit for bit."""
        import sys

        attn = layers[index].self_attn
        lang = sys.modules[type(attn).__mro__[1].__module__]
        d_rel = int(attn.rel_proj.shape[0])
        edge = attn.sliding or attn.rel_extent
        for n in (1, rows, MAX_ROWS):
            for size in sorted({n, n + 7, edge - 1 + n, edge + 5 + n}):
                if attn.sliding and size > attn.sliding - 1 + n:
                    continue                     # more keys than a sliding call can see
                r = rand((1, n, attn.n_heads, d_rel), mx.bfloat16)
                sinks = rand((1, n, attn.n_heads), mx.bfloat16)
                ref = mx.concatenate(
                    [lang.banded_additive_mask(r, attn.rel_proj.astype(r.dtype), size - n, size,
                                               attn.sliding, attn.rel_extent, left_padding=None,
                                               shape_ref=mx.zeros((1, 1, size, 1), r.dtype)),
                     sinks.transpose(0, 2, 1)[..., None]], axis=-1)
                got = fused_mask(r, _rel_transposed(attn, r.dtype), sinks, size, attn.sliding,
                                 attn.rel_extent)
                if got.shape != ref.shape or not bool(mx.array_equal(got, ref)):
                    return False
        return True

    def packed_run(index):
        """The packed verify attention on this layer's shape and weights, the widest verify past
        the sliding cap / the relative-position extent (several key blocks, a partial tile, the
        window cap) and a short window:
        the kernel against the same arithmetic in fp32 ops, closely (the stock call rounds more
        inside, so it cannot be the yardstick for the numerics); the attention module with the
        kernel against the stock module on the in-place caches, loosely — an upstream change
        of the attention is far outside it; and causality, which no tolerance covers: with
        another token in the LAST row, every earlier row's output must be the same bits."""
        from .inkling_cache import convert

        attn = layers[index].self_attn
        if not _packed_ok(attn, PACKED_MAX_ROWS):
            return False
        edge = attn.sliding or attn.rel_extent
        d_rel = int(attn.rel_proj.shape[0])
        rel_t = _rel_transposed(attn, mx.bfloat16)
        for n, keys in ((PACKED_MAX_ROWS, edge + 6), (VECTOR_ROWS, edge + 6), (PACKED_MIN_ROWS, 12)):
            size = min(keys, attn.sliding - 1) + n if attn.sliding else keys + n
            args = (rand((1, n, attn.n_heads, attn.head_dim), mx.bfloat16),
                    rand((1, attn.n_kv, size, attn.head_dim), mx.bfloat16),
                    rand((1, attn.n_kv, size, attn.head_dim), mx.bfloat16),
                    rand((1, n, attn.n_heads, d_rel), mx.bfloat16), rel_t,
                    rand((1, n, attn.n_heads), mx.bfloat16),
                    rand((1, n, attn.n_heads * attn.head_dim), mx.bfloat16))
            if not _rel_l2(packed_attention(attn, *args),
                           _packed_reference(attn, *args)) < PACKED_TOL:
                return False

        def run(prefix, x, start):
            entry = lm.make_cache()[visits.index(index)]
            convert([entry])
            if not is_sink_cache(entry.caches[0]):
                raise RuntimeError("no in-place cache for this layer's KV cache")
            attn.__dict__[_PACKED] = 0
            attn(prefix, cache=entry)            # more rows than a verify: the stock body
            attn.__dict__[_PACKED] = start
            out = attn(x, cache=entry)
            mx.eval(out)
            return out

        set_enabled(lm, False)
        set_enabled(lm, True, ("attn",))
        held = attn.__dict__.get(_ROWS)
        attn.__dict__[_ROWS] = MAX_ROWS
        try:
            # a drafter round over a long window; the widest call it takes over a short one
            for n, keys in ((VECTOR_ROWS, edge + 6), (PACKED_MAX_ROWS, MAX_ROWS + 4)):
                prefix = rand((1, keys, hidden), mx.bfloat16)
                x = rand((1, n, hidden), mx.bfloat16)
                ref, got = run(prefix, x, 0), run(prefix, x, 1)
                if got.shape != ref.shape or not _rel_l2(got, ref) < PACKED_TOL_STOCK:
                    return False
                other = mx.concatenate([x[:, :-1], rand((1, 1, hidden), mx.bfloat16)], axis=1)
                if not bool(mx.array_equal(run(prefix, other, 1)[:, :-1], got[:, :-1])):
                    return False
            return True
        finally:
            attn.__dict__[_PACKED] = 0
            attn.__dict__[_ROWS] = held if held is not None else VECTOR_ROWS

    def wide_run():
        """Calls wider than a drafter round (a lookup round: up to ``MAX_ROWS`` positions), where
        MLX's attention leaves its vector kernels and its matrix products may pick other
        kernels: every exact piece against the stock forward at that width (states included),
        the in-place caches through a wide verify committed in part and in full, the experts
        row by row, and the layer-visit loop around them."""
        set_rows(lm, MAX_ROWS)
        # the exact kinds that passed, together (a mismatch switches the wide calls off whichever
        # piece it is, so there is nothing to tell apart here)
        exact = tuple(kind for kind in EXACT_KINDS if result.get(kind))
        if exact:
            if not all(layer_run(i, exact, rows=MAX_ROWS) for i in probes):
                return False
            if core is not None and not layer_run(core, exact, rand((1, MAX_ROWS, 4, hidden)),
                                                  rows=MAX_ROWS):
                return False
        if result.get(KV_KIND) and not all(
                kv_run(i, plan=[(1, None), (MAX_ROWS, 9), (1, None), (MAX_ROWS - 1, MAX_ROWS - 1),
                                (1, None)]) for i in probes):
            return False
        if result.get("moe"):
            set_enabled(lm, False)
            if not _moe_selftest(lm, rand, wide=True):
                return False
        if result.get("model"):
            set_enabled(lm, True)
            if not model_run(rows=MAX_ROWS, first=False):
                return False
        return True

    def bank_run(n):
        """The loop's ONE read of a pass's messages against the stock loop's read per layer
        (mlx-vlm's expression) at ``n`` positions, for banks of 1 to 4 states: the same rows of
        one larger product — on the kernels MLX picks at that width."""
        if model.loop is None:
            return True
        queries = model.message_queries
        per_layer, scale = int(queries.shape[1]), hidden ** -0.5
        for states in (1, 2, 3, 4):
            bank = mx.fast.rms_norm(rand((1, n, states, hidden), mx.bfloat16).astype(mx.float32),
                                    None, 1e-6)
            scores = (queries.reshape(-1, queries.shape[-1]) @ bank.swapaxes(-1, -2)) * scale
            once = mx.softmax(scores, axis=-1) @ bank
            for j in range(int(queries.shape[0])):
                ref = mx.softmax((queries[j] @ bank.swapaxes(-1, -2)) * scale, axis=-1) @ bank
                if not bool(mx.array_equal(once[..., j * per_layer: (j + 1) * per_layer, :], ref)):
                    return False
        return True

    def chunk_run():
        """A prefill chunk (``CHUNK_ROWS`` positions): the bit-exact pieces that passed, alone —
        the experts, residual mixes and controllers are the stock modules at this width —
        against the stock forward on the sliding, global and core probes, outputs and cache
        states; the in-place caches under the fused attention body through a chunk that fills
        the window, a verify that slides it and a step; and the loop's one message read."""
        exact = tuple(kind for kind in EXACT_KINDS if result.get(kind))
        if not exact:
            return False
        set_rows(lm, rows_limit(lm), CHUNK_ROWS)
        if not all(layer_run(i, exact, rows=CHUNK_ROWS) for i in probes):
            return False
        if core is not None and not layer_run(core, exact, rand((1, CHUNK_ROWS, 4, hidden)),
                                              rows=CHUNK_ROWS):
            return False
        if result.get(KV_KIND) and result.get("attn") and not all(
                kv_run(i, window=CHUNK_ROWS + 2, arms=((True, True),),
                       plan=[(1, None), (CHUNK_ROWS, None), (4, 2), (1, None)]) for i in probes):
            return False
        return not result.get("model") or bank_run(CHUNK_ROWS)

    sliding = next((i for i, layer in enumerate(layers) if layer.is_sliding and i > 0), None)
    globl = next((i for i, layer in enumerate(layers) if not layer.is_sliding), None)
    core = next((i for i, layer in enumerate(layers) if getattr(layer, "controllers", None)), None)
    probes = [i for i in (sliding, globl) if i is not None]
    result: dict[str, bool] = {}
    lm.__dict__[_OFF] = ()
    set_packed(lm, False)               # the exact checks compare the stock attention call
    set_rows(lm, VECTOR_ROWS)
    installed = {kind for _, kind in _fused_modules(lm)}
    if "attn" in installed:
        # first, so the attention and cache checks below run on the mask they will ship with
        try:
            ok = bool(probes) and all(mask_run(i) for i in probes)
        except Exception:  # noqa: BLE001 — the kernel does not build here: keep the stock pair
            ok = False
        result[MASK_KIND] = ok
        set_mask(lm, ok)
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
    if "attn" in installed:
        try:
            ok = bool(probes) and all(kv_run(i) for i in probes)
        except Exception:  # noqa: BLE001 — another mlx-vlm cache layout: keep the stock caches
            ok = False
        result[KV_KIND] = ok
        # the packed verify attention reads the in-place caches' views: only with them
        try:
            ok = ok and result.get("attn", False) and all(packed_run(i) for i in probes)
        except Exception:  # noqa: BLE001 — the kernel does not build here: keep the stock call
            ok = False
        result[PACKED_KIND] = ok
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
    lm.__dict__[_OFF] = tuple(kind for kind, passed in result.items()
                              if not passed and kind not in (KV_KIND, MASK_KIND, PACKED_KIND))
    lm.__dict__[_KV] = bool(result.get(KV_KIND))
    if "model" in installed:
        # the loop last, around the pieces that passed: whole forwards (a decode step, then a
        # wider call on the same caches) with and without it — outputs, taps and cache states
        try:
            ok = model_run()
        except Exception:  # noqa: BLE001
            ok = False
        result["model"] = ok
        if not ok:
            lm.__dict__[_OFF] += ("model",)
    # last: the wider calls, with every verdict above in force
    try:
        ok = bool(probes) and wide_run()
    except Exception:  # noqa: BLE001 — a kernel this GPU cannot launch at that width
        ok = False
    result[WIDE_KIND] = ok
    set_rows(lm, MAX_ROWS if ok else VECTOR_ROWS)
    # ... and a prefill chunk, for the bit-exact pieces alone
    try:
        chunk = bool(probes) and chunk_run()
    except Exception:  # noqa: BLE001 — a kernel this GPU cannot launch at that width
        chunk = False
    result[CHUNK_KIND] = chunk
    set_rows(lm, MAX_ROWS if ok else VECTOR_ROWS, CHUNK_ROWS if chunk else None)
    set_packed(lm, bool(result.get(PACKED_KIND)))
    set_enabled(lm, True)
    return result
