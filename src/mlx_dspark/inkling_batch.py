"""Batched plain decoding for Inkling 2: one token per REQUEST through one forward.

A decode step reads the same weights whatever it decodes, so ``B`` requests that each need one
token can share the read: their rows ride ``[1, B, H]`` through every row-wise piece of the fused
forward (``inkling_fused``: the projections, the routed experts, the residual mixes, the loop
controllers, the output head), exactly as a verify's rows do. What a verify's rows share and
requests do not is the SEQUENCE: each request has its own keys and values, its own position and
its own short-convolution history. So, per layer visit:

- the K / V caches stay one per request (the in-place caches of ``inkling_cache``, the ones the
  single-stream path uses), and each request's attention is the single-row decode call over its
  own cache — the same kernel on the same shapes, so prompts of different lengths need no
  padding and a request joins or leaves between steps with nothing to copy;
- the four short-convolution states are a fixed ``[1, 3, C]`` per request whatever its length, so
  they are held STACKED on the batch axis (``[B, 3, C]``) and each convolution is one launch for
  all requests (``slot_short_convolution`` / ``slot_dynamic_convolution``: the single-row
  kernels' expressions with the time index replaced by the request index).

Row for row this is the decode step's arithmetic; the bf16 projections are the one place a row's
bits can depend on its neighbours (MLX picks its matmul kernel by row count), the same fp-tie
class as a verify row. ``selftest`` holds a two-request step against two single-row steps.
"""

from __future__ import annotations

import time
from functools import lru_cache

import mlx.core as mx

from . import inkling_fused as F
from .inkling_cache import is_sink_cache

MAX_SLOTS = 8                 # requests per step: F.VECTOR_ROWS, the width every fused piece takes


@lru_cache(maxsize=1)
def _lang():
    """mlx-vlm's Inkling 2 language module (its mask builder is the fallback F._attention_mask
    takes when the one-launch mask is off)."""
    from importlib import import_module

    return import_module("mlx_vlm.models.inkling2.language")


# ---- short convolution, one launch for B requests -----------------------------------------------
# F._sconv_source at t = 0 with the batch row where the time index was: the same four products
# added in the same order, the same rounding points.
def _slot_sconv_source(res: bool) -> str:
    add = "conv = conv + float(res[b * res_strides[1] + c * res_strides[2]]);" if res else ""
    return r"""
    uint i = thread_position_in_grid.x;
    int B = x_shape[1], C = x_shape[2];
    if (i >= uint(B * C)) return;
    int c = i % C, b = i / C;
    size_t sb = size_t(b) * state_strides[0] + size_t(c) * state_strides[2];
    float value = 0;
    for (int j = 0; j < W; ++j) {
        float xv = j < W - 1
            ? float(state[sb + j * state_strides[1]])
            : float(x[b * x_strides[1] + c * x_strides[2]]);
        float w = weight[c * weight_strides[0] + j * weight_strides[1]];
        value = value + xv * w;
    }
    float xt = float(x[b * x_strides[1] + c * x_strides[2]]);
    float conv = float(T((value + xt) / divisor));
    """ + add + r"""
    out[i] = OT(conv);
    size_t nb = (size_t(b) * (W - 1)) * C + c;
    for (int s = 0; s < W - 2; ++s) {
        next[nb + s * C] = float(state[sb + (s + 1) * state_strides[1]]);
    }
    next[nb + (W - 2) * C] = xt;
"""


@lru_cache(maxsize=2)
def _slot_sconv_kernel(res: bool):
    return mx.fast.metal_kernel(
        name="mlx_dspark_inkling_slot_sconv" + ("_res" if res else ""),
        input_names=["state", "weight", "x", "divisor"] + (["res"] if res else []),
        output_names=["out", "next"],
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
        source=_slot_sconv_source(res),
    )


def slot_short_convolution(state: mx.array, weight: mx.array, x: mx.array, width: int,
                           res: mx.array | None = None, out_dtype=None):
    """``(out, next_state)`` for ``B`` requests: ``x [1, B, C]`` is one new input per request,
    ``state [B, width - 1, C]`` (fp32) each request's last inputs. ``out [1, B, C]`` is what
    ``inkling_fused.fused_short_convolution`` gives each request alone (``res`` / ``out_dtype``
    as there), ``next_state`` the states one input later."""
    _, rows, channels = x.shape
    out_dtype = out_dtype or x.dtype
    inputs = [state, weight, x, F._divisor(width)] + ([res] if res is not None else [])
    return _slot_sconv_kernel(res is not None)(
        inputs=inputs,
        template=[("T", x.dtype), ("OT", out_dtype), ("W", width)],
        grid=(rows * channels, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape, (rows, width - 1, channels)],
        output_dtypes=[out_dtype, mx.float32],
    )


# F._DYN_SOURCE at t = 0, per request (the global layers' K / V convolution)
_SLOT_DYN_SOURCE = r"""
    uint i = thread_position_in_grid.x;
    int B = x_shape[1], C = x_shape[2];
    if (i >= uint(B * C)) return;
    int c = i % C, b = i / C;
    const int head = c / (C / HEADS);
    size_t sb = size_t(b) * state_strides[0] + size_t(c) * state_strides[2];
    float xt = float(x[b * x_strides[1] + c * x_strides[2]]);
    float acc = 0;
    for (int lag = 0; lag < W; ++lag) {
        float v = lag > 0 ? float(state[sb + (W - 1 - lag) * state_strides[1]]) : xt;
        float w = float(stat[lag * C + c]) + float(dyn[(b * HEADS + head) * W + lag]);
        float prod = v * w;
        acc = lag == 0 ? prod : acc + prod;
    }
    out[i] = T((acc + xt) / divisor);
    size_t nb = (size_t(b) * (W - 1)) * C + c;
    for (int s = 0; s < W - 2; ++s) {
        next[nb + s * C] = float(state[sb + (s + 1) * state_strides[1]]);
    }
    next[nb + (W - 2) * C] = xt;
"""


@lru_cache(maxsize=1)
def _slot_dyn_kernel():
    return mx.fast.metal_kernel(
        name="mlx_dspark_inkling_slot_dynamic_sconv",
        input_names=["state", "stat", "dyn", "x", "divisor"],
        output_names=["out", "next"],
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
        source=_SLOT_DYN_SOURCE,
    )


def slot_dynamic_convolution(state: mx.array, static_weight: mx.array, dyn: mx.array,
                             x: mx.array, width: int, heads: int):
    """mlx-vlm's ``DynamicShortConvolution`` for ``B`` requests (``dyn [1, B, heads * width]``
    is its ``weight_proj`` of each request's input): ``(out [1, B, C], next_state)``."""
    _, rows, channels = x.shape
    return _slot_dyn_kernel()(
        inputs=[state, static_weight, dyn, x, F._dyn_divisor(width)],
        template=[("T", x.dtype), ("W", width), ("HEADS", heads)],
        grid=(rows * channels, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape, (rows, width - 1, channels)],
        output_dtypes=[x.dtype, mx.float32],
    )


def _sconv(conv, x, states, res=None, out_dtype=None):
    out, states[conv.conv_idx] = slot_short_convolution(
        states[conv.conv_idx], conv.conv.weight, x, conv.kernel_size, res, out_dtype)
    return out


def _dyn_sconv(conv, x, dynamic_input, states):
    out, states[conv.conv_idx] = slot_dynamic_convolution(
        states[conv.conv_idx], conv.static_weight, conv.weight_proj(dynamic_input), x,
        conv.kernel_size, conv.num_heads)
    return out


# ---- the stacked convolution states --------------------------------------------------------------
def stack_states(lm, caches: list) -> list[list[mx.array]]:
    """``[visit][conv index] -> [B, 3, C]`` (fp32) from each request's own layer-visit caches."""
    model = lm.model
    stacked = []
    for visit, index in enumerate(model.layer_indices):
        layer = model.layers[index]
        attn = layer.self_attn
        kv_channels = attn.n_kv * attn.head_dim
        hidden = int(layer.attn_sconv.conv.weight.shape[0])
        widths = (kv_channels, kv_channels, hidden, hidden)
        row = []
        for idx, channels in enumerate(widths):
            parts = []
            for cache in caches:
                state = cache[visit][1][idx]
                if state is None:
                    state = mx.zeros((1, layer.attn_sconv.kernel_size - 1, channels), mx.float32)
                parts.append(state.astype(mx.float32))
            row.append(parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0))
        stacked.append(row)
    return stacked


def unstack_states(stacked: list[list[mx.array]], caches: list, rows=None) -> None:
    """Give each request's caches their own convolution states back (row ``b`` of the stack to
    ``caches[b]``; ``rows`` names the stack rows when the caches are a subset)."""
    rows = range(len(caches)) if rows is None else rows
    for visit, row in enumerate(stacked):
        for idx, state in enumerate(row):
            for cache, b in zip(caches, rows):
                cache[visit][1][idx] = state[b: b + 1]


def take_states(stacked: list[list[mx.array]], rows: list[int]) -> list[list[mx.array]]:
    """The stack restricted to ``rows`` (requests that stay), in that order."""
    pick = mx.array(rows)
    return [[state[pick] for state in row] for row in stacked]


def join_states(a: list[list[mx.array]], b: list[list[mx.array]]) -> list[list[mx.array]]:
    """Two stacks as one (requests joining a running batch)."""
    return [[mx.concatenate([sa, sb], axis=0) for sa, sb in zip(ra, rb)] for ra, rb in zip(a, b)]


# ---- the attention body, one cache per request ----------------------------------------------------
def _attention(attn, lang, x, key_input, value_input, kvs, states):
    """mlx-vlm's ``Attention.__call__`` for ``x [1, B, H]`` = one new position of ``B`` sequences.
    The projections and the per-head preparation run once for all rows; each row then attends
    over its own cache with the single-row decode call."""
    cfg = attn.config
    key_input = x if key_input is None else key_input
    value_input = x if value_input is None else value_input
    k = attn.k_proj(key_input)
    v = attn.v_proj(value_input)
    if attn.is_sliding:
        k = _sconv(attn.k_sconv, k, states)
        v = _sconv(attn.v_sconv, v, states)
    else:
        k = _dyn_sconv(attn.k_sconv, k, key_input, states)
        v = _dyn_sconv(attn.v_sconv, v, value_input, states)
    q, k, v, r = F.fused_head_prep(
        attn.q_proj(x), k, v, attn.r_proj(x), attn.q_scale, attn.rel_scale,
        n_heads=attn.n_heads, n_kv=attn.n_kv, head_dim=attn.head_dim, d_rel=cfg.d_rel,
        clip=cfg.qk_l2_clip_threshold, rel_clip=cfg.rel_l2_clip_threshold,
        eps=cfg.rms_norm_eps)
    if not attn.is_sliding and cfg.log_scaling_n_floor is not None:
        offsets = [kv.offset for kv in kvs]
        # tau = 1 + alpha * log(max(position / n_floor, 1)): exactly 1 up to n_floor
        if max(offsets) + 1 > cfg.log_scaling_n_floor:
            positions = mx.array([o + 1 for o in offsets])
            tau = 1 + cfg.log_scaling_alpha * mx.log(
                mx.maximum(positions / cfg.log_scaling_n_floor, 1))
            q = (q.astype(mx.float32) * tau[..., None, None]).astype(x.dtype)
            r = (r.astype(mx.float32) * tau[..., None, None]).astype(x.dtype)
    sinks = attn.sink_proj(x)
    scale = attn.head_dim ** -0.5
    outs = []
    for b, kv in enumerate(kvs):
        kb, _ = kv.update_and_fetch(k[:, b: b + 1].transpose(0, 2, 1, 3),
                                    v[:, b: b + 1].transpose(0, 2, 1, 3))
        mask = F._attention_mask(attn, lang, r[:, b: b + 1], sinks[:, b: b + 1],
                                 kb.shape[2], 1, kb)
        kb, vb = kv.sink_views()
        out = mx.fast.scaled_dot_product_attention(
            q[:, b: b + 1].transpose(0, 2, 1, 3), kb, vb, scale=scale, mask=mask)
        outs.append(out.transpose(0, 2, 1, 3).reshape(1, 1, -1))
    out = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)
    return attn.o_proj(F._gated(out, attn.gate_proj(x)))


# ---- the decoder layer body -------------------------------------------------------------------------
def _layer(layer, lang, x, blocks, kvs, states, messages):
    """``inkling_fused._layer_call`` with the sequence state per request."""
    mixed = (layer.attention_input(blocks, x) if layer.attention_input is not None
             else x[..., None, :])
    if layer.is_sliding and messages is None:
        attn = _attention(layer.self_attn, lang,
                          F.fused_norm_cast(layer.input_layernorm, mixed[..., 0, :]),
                          None, None, kvs, states)
    else:
        parts = [mixed[..., min(i, mixed.shape[-2] - 1), :] for i in range(3)]
        if messages is not None:
            parts = [layer.controllers[name](part, messages[..., i, :])
                     for i, (name, part) in enumerate(zip(("attn", "k", "v"), parts))]
        norms = ((layer.input_layernorm,) * 3 if layer.is_sliding
                 else (layer.q_input_layernorm, layer.k_input_layernorm, layer.v_input_layernorm))
        parts = [F.fused_norm_cast(norm, part) for norm, part in zip(norms, parts)]
        attn = _attention(layer.self_attn, lang, *parts, kvs, states)
    residual = _sconv(layer.attn_sconv, attn, states,
                      None if layer.is_block_boundary else x, mx.float32)
    if layer.is_block_boundary:
        blocks.append(x)
    hidden = layer.mlp_input(blocks, residual)[..., 0, :]
    if messages is not None:
        hidden = layer.controllers["mlp"](hidden, messages[..., 3, :])
    hidden = layer.mlp(F.fused_norm_cast(layer.post_attention_layernorm, hidden))
    return _sconv(layer.mlp_sconv, hidden, states, residual, x.dtype)


# ---- one step ---------------------------------------------------------------------------------------
def step(lm, tokens: mx.array, caches: list, states: list[list[mx.array]], tap=None):
    """One token for each of ``B`` requests: ``tokens [B]`` (each request's last token),
    ``caches[b]`` its layer-visit caches, ``states`` the stacked convolution states
    (:func:`stack_states`; updated in place). Returns the logits ``[B, vocab]`` — and, with
    ``tap`` (layer visits, as ``Target.prefill`` takes them), the tapped hidden states
    ``[1, B, len(tap) * H]`` a DSpark drafter keeps as context.

    ``inkling_fused._model_call``'s loop at ``B`` rows: the loop passes' messages and the
    residual mixes are per row already."""
    model = lm.model
    lang = _lang()
    rows = int(tokens.shape[0])
    visits = len(model.layer_indices)
    requested = set(tap or ())
    captures = {}
    x = model.embed(tokens.reshape(1, rows))
    blocks: list = []
    loop = model.loop
    pass_messages = None
    per_layer = int(model.message_queries.shape[1]) if loop is not None else 0
    every = F.SCHEDULE_EVERY
    for visit, index in enumerate(model.layer_indices):
        if loop is not None and index == loop.core_start:
            if visit == index:
                prelude, prelude_blocks = x, blocks.copy()
            else:
                states_ = mx.stack([*blocks[len(prelude_blocks) + 1:], x], axis=-2)
                bank = mx.fast.rms_norm(states_.astype(mx.float32), None, 1e-6)
                x, blocks = prelude, prelude_blocks.copy()
                queries = model.message_queries
                scores = (queries.reshape(-1, queries.shape[-1]) @ bank.swapaxes(-1, -2)
                          ) * x.shape[-1] ** -0.5
                pass_messages = mx.softmax(scores, axis=-1) @ bank
        messages = None
        if pass_messages is not None and index < loop.core_end:
            at = (index - loop.core_start) * per_layer
            messages = pass_messages[..., at: at + per_layer, :]
        if visit in requested:
            captures[visit] = x
        x = _layer(model.layers[index], lang, x, blocks, [c[visit][0] for c in caches],
                   states[visit], messages)
        if every and (visit + 1) % every == 0:
            mx.async_eval(x)
    h = model.final_input(blocks, x)[..., 0, :]
    normalized = model.norm(h)
    logits = lm._logits_from_norm(normalized)[0]
    if tap is None:
        return logits
    if visits in requested:
        captures[visits] = normalized.astype(x.dtype)
    return logits, mx.concatenate([captures[i] for i in tap], axis=-1)


def supported(lm) -> bool:
    """Whether this model can take batched steps: the fused pieces and the in-place KV caches
    the per-request attention reads (``inkling_fused.selftest`` cleared both at load)."""
    on = set(F.enabled_kinds(lm))
    return (F.kv_caches(lm) and {"layer", "attn", "model"} <= on
            and all(is_sink_cache(c[0]) for c in F.make_cache(lm)[:1]))


# ---- a session: requests that decode together -------------------------------------------------------
PREFILL_SHARE = 0.5           # while a request reads a prompt longer than one chunk, the share of
#                               the time it gets; the requests already decoding get the rest (a
#                               chunk, then steps for as long as the chunk took)
TAP_FLUSH = 512               # buffered context rows per request before they go to the drafter


class Request:
    """One generation, from admission to its result. ``cache`` / ``ctx`` / ``reuse_len`` and the
    prefill hooks are what the single-stream loops take (the server's prefix-cache plan);
    ``meta`` is the caller's handle."""

    def __init__(self, prompt_ids, *, max_new_tokens, temperature=0.0, top_p=1.0, top_k=0,
                 stop=None, on_text=None, on_round=None, cache=None, ctx=None, reuse_len=0,
                 prefill_marks=None, on_prefill=None, on_prefill_chunk=None, meta=None,
                 prepare=None):
        self.prepare = prepare            # called once, right before the prompt is read: the
        #                                   place to fill in cache / ctx / reuse_len / hooks
        self.prompt_ids = list(prompt_ids)
        self.max_new_tokens = int(max_new_tokens)
        self.sampling = (float(temperature or 0.0), top_p, top_k)
        self.stop = stop
        self.on_text = on_text
        self.on_round = on_round
        self.cache, self.ctx, self.reuse_len = cache, ctx, int(reuse_len) if cache is not None else 0
        self.prefill_marks, self.on_prefill = prefill_marks, on_prefill
        self.on_prefill_chunk = on_prefill_chunk
        self.meta = meta
        self.out_ids: list[int] = []      # every token produced so far
        self.pending_known = False        # out_ids[-1] is the token the next step forwards
        self.filled = self.reuse_len      # prompt tokens in the caches
        self.ctx_len = self.reuse_len     # positions in the drafter's context
        self.taps: list = []              # context rows of batched steps, not yet in ctx
        self.streamer = None
        self.accept_lengths: list[int] = []
        self.rounds = self.forwards = self.lookup_rounds = self.tree_rounds = 0
        self.solo_tokens = self.solo_rounds = self.solo_stretches = 0   # single-stream stretches
        self.shared = False               # it has decoded alongside another request
        self.batched_tokens = 0           # ... for this many of its tokens
        self.t0 = self.t_prefill = None
        self.finish = None                # "stop" | "length" once it is over


def _session_streamer():
    from .generate import _Streamer

    class SessionStreamer(_Streamer):
        """A streamer whose loop can be asked to stop at a round boundary and be resumed:
        ``stopped`` also turns true when ``waiting()`` does (another request wants in), with
        ``yielded`` telling that apart from a stop string or a client that went away."""

        waiting = None
        yielded = False

        def update(self, out_ids):
            super().update(out_ids)
            if not self.stopped and self.waiting is not None and self.waiting():
                self.yielded = self.stopped = True

        def resume(self, fed: int) -> None:
            """Carry on after a yield, with ``fed`` tokens of the next list already seen."""
            if self.yielded:
                self.yielded = self.stopped = False
            self.n_fed = fed

    return SessionStreamer


class Session:
    """Requests sharing one Inkling 2 target. Two or more decode together, one token each per
    :func:`step`; a request that is alone runs ``solo`` — the single-stream speculative loop,
    on the same caches — and goes back to the shared step the moment another request is
    waiting. Requests join and leave between steps.

    ``drafter`` (dspark mode) makes every step keep the tapped hidden states, so a request's
    drafter context stays complete through batched stretches and ``solo`` can pick it up;
    without one (baseline) a lone request simply decodes at one row.

    ``solo(request, prompt_ids, reuse_len, max_new_tokens, fresh)`` must run the loop with
    ``request.streamer`` and return its ``GenResult`` (``fresh``: the prompt has not been read
    yet — the loop prefills it, with the request's prefill hooks). ``waiting()`` says whether
    someone else wants in.

    Drive it with :meth:`admit` and :meth:`tick`; everything runs on the caller's thread."""

    def __init__(self, target, tokenizer, drafter=None, *, capacity: int = MAX_SLOTS,
                 solo=None, waiting=None, prefill_share: float = PREFILL_SHARE):
        from .generate import eos_token_ids

        self.target, self.tokenizer, self.drafter = target, tokenizer, drafter
        self.lm = target.model.language_model
        self.tap = list(drafter.config.target_layer_ids) if drafter is not None else None
        self.capacity = max(1, min(int(capacity), MAX_SLOTS))
        self.solo = solo if drafter is not None else None
        self.waiting = waiting or (lambda: False)
        self.prefill_share = min(1.0, max(0.05, float(prefill_share)))
        self.eos = eos_token_ids(tokenizer)
        self.rows: list[Request] = []         # decoding together
        self.y = None                         # their pending tokens, [B] (lazy)
        self.states = None                    # their stacked convolution states
        self.filling: list[Request] = []      # admitted, prompt not read yet
        self.steps = self.step_rows = self.peak = 0
        self.prefill_seconds = 0.0            # wall time spent reading prompts
        self._streamer_cls = _session_streamer()

    # -- capacity ---------------------------------------------------------------------------------
    @property
    def n_active(self) -> int:
        return len(self.rows) + len(self.filling)

    @property
    def has_free_slot(self) -> bool:
        return self.n_active < self.capacity

    # -- admission --------------------------------------------------------------------------------
    def admit(self, req: Request) -> None:
        req.t0 = time.time()
        self.filling.append(req)

    def _open(self, req: Request) -> None:
        """``req`` starts reading its prompt: its caches (the caller's plan, or fresh ones) and
        its streamer."""
        if req.streamer is not None:
            return
        if req.prepare is not None:
            req.prepare(req)
        if req.cache is None:
            req.cache = self.target.make_cache()
            req.reuse_len = 0
        if req.ctx is None and self.drafter is not None:
            req.ctx = self.drafter.make_ctx_cache()
            req.reuse_len = 0
        req.filled = req.ctx_len = req.reuse_len = int(req.reuse_len)
        req.streamer = self._streamer_cls(self.tokenizer, self.eos, req.on_text, req.stop)

    def _read_chunk(self, req: Request) -> bool:
        """One prefill chunk of ``req``'s prompt; True once the whole prompt is in its caches
        (its first token is then pending and it has joined the rows)."""
        from .generate import (
            _cache_arrays,
            _prefill_plain,
            _prefill_tapped,
            _release_prefill_scratch,
            _sample_arr,
            _target_chunk,
        )

        self._open(req)
        t_chunk = time.time()
        ids = req.prompt_ids
        start = req.filled
        end = min(start + _target_chunk(self.target), len(ids))
        on_mark = ((lambda p: req.on_prefill(req.cache, req.ctx, p))
                   if req.on_prefill is not None else None)
        if self.tap is not None:
            logits, _ = _prefill_tapped(
                self.target, ids[start:end], req.cache, self.tap, drafter=self.drafter,
                ctx_caches=req.ctx, ctx_offset=start, marks=req.prefill_marks, on_mark=on_mark)
        else:
            logits = _prefill_plain(self.target, ids[start:end], req.cache, base=start,
                                    marks=req.prefill_marks, on_mark=on_mark)
        req.filled = req.ctx_len = end
        if end < len(ids):
            mx.eval(_cache_arrays(req.cache), [c.k for c in req.ctx] if req.ctx else [])
            mx.clear_cache()
            if req.on_prefill_chunk is not None:
                req.on_prefill_chunk(end)
            self.prefill_seconds += time.time() - t_chunk
            return False
        if req.on_prefill is not None:
            req.on_prefill(req.cache, req.ctx, len(ids))
        first = _sample_arr(logits[0, -1], *req.sampling).reshape(1).astype(mx.int32)
        mx.eval(first, [c.k for c in req.ctx] if req.ctx else [])
        _release_prefill_scratch(self.target, len(ids) - req.reuse_len)
        req.t_prefill = time.time()
        self.prefill_seconds += req.t_prefill - t_chunk
        self._join(req, first)
        return True

    def _join(self, req: Request, pending: mx.array) -> None:
        mine = stack_states(self.lm, [req.cache])
        self.states = mine if not self.rows else join_states(self.states, mine)
        self.y = pending if not self.rows else mx.concatenate([self.y, pending])
        self.rows.append(req)
        self.peak = max(self.peak, len(self.rows))

    def _drop(self, b: int) -> Request:
        req = self.rows.pop(b)
        if self.rows:
            keep = [i for i in range(len(self.rows) + 1) if i != b]
            self.y = self.y[mx.array(keep)]
            self.states = take_states(self.states, keep)
        else:
            self.y = self.states = None
        return req

    # -- tokens -----------------------------------------------------------------------------------
    def _push(self, req: Request, token: int, shared: bool = False) -> None:
        """A token of ``req`` became known (it has just been forwarded)."""
        if req.pending_known:                 # the single-stream loop already reported it
            req.pending_known = False
            return
        req.out_ids.append(token)
        req.batched_tokens += shared
        req.accept_lengths.append(1)
        req.rounds += 1
        req.forwards += 1
        if req.on_round is not None:
            req.on_round(drafted=0, accepted=0, committed=1, cap=0, source="plain")
        req.streamer.update(req.out_ids)
        self._settle(req, token)

    def _settle(self, req: Request, last: int) -> None:
        st = req.streamer
        if last in self.eos or (st.stopped and not st.yielded):
            req.finish = "stop"
        elif len(req.out_ids) >= req.max_new_tokens:
            req.finish = "length"

    def _result(self, req: Request):
        from .generate import GenResult

        st = req.streamer
        cut = st.stopped and not st.yielded
        st.flush()
        text = st.text if cut else self.tokenizer.decode(
            [t for t in req.out_ids if t not in self.eos])
        now = time.time()
        return GenResult(
            text=text, token_ids=req.out_ids, num_tokens=len(req.out_ids), num_rounds=req.rounds,
            accept_lengths=req.accept_lengths, target_forwards=req.forwards,
            seconds=now - req.t0, finish_reason=req.finish, lookup_rounds=req.lookup_rounds,
            tree_rounds=req.tree_rounds,
            prefill_seconds=(req.t_prefill or now) - req.t0,
            batched_tokens=req.batched_tokens)

    # -- the shared step ----------------------------------------------------------------------------
    def _step(self, finished: list) -> None:
        from .batch_engine import _sample_rows
        from .generate import _sample_arr

        rows = self.rows
        out = step(self.lm, self.y, [r.cache for r in rows], self.states, self.tap)
        logits, fused = out if self.tap is not None else (out, None)
        params = {r.sampling for r in rows}
        if len(params) == 1:
            y_next = _sample_rows(logits, *rows[0].sampling)
        else:
            y_next = mx.stack([_sample_arr(logits[b], *r.sampling) for b, r in enumerate(rows)])
        y_next = y_next.astype(mx.int32)
        mx.async_eval(y_next, *([fused] if fused is not None else []))
        tokens = self.y.tolist()              # forwarded just now; computed by the last step
        self.y = y_next
        self.steps += 1
        self.step_rows += len(rows)
        for b, req in enumerate(rows):
            req.shared = req.shared or len(rows) > 1
            if fused is not None:
                req.taps.append(fused[:, b: b + 1])
            self._push(req, tokens[b], len(rows) > 1)
        for b in range(len(rows) - 1, -1, -1):
            req = rows[b]
            if req.finish is not None:
                finished.append((self._drop(b), None))
            elif len(req.taps) >= TAP_FLUSH:
                self._flush_taps(req)

    def _flush_taps(self, req: Request) -> None:
        """Hand the context rows of ``req``'s batched steps to the drafter."""
        if not req.taps:
            return
        rows = req.taps[0] if len(req.taps) == 1 else mx.concatenate(req.taps, axis=1)
        self.drafter.update_context(rows, ctx_offset=req.ctx_len, ctx_caches=req.ctx)
        req.ctx_len += len(req.taps)
        req.taps = []
        mx.async_eval([c.k for c in req.ctx])

    # -- alone: the single-stream loop ----------------------------------------------------------------
    def _leave(self, b: int) -> Request:
        """Take row ``b`` out of the shared step with its caches whole: its pending token read,
        its convolution states back in its own caches, its context rows with the drafter."""
        req = self.rows[b]
        self._push(req, int(self.y[b].item()))        # the pending token: not forwarded yet
        if req.finish is None:
            unstack_states(self.states, [req.cache], rows=[b])
            self._flush_taps(req)
        return self._drop(b)

    def _run_solo(self, req: Request, fresh: bool, finished: list) -> None:
        self._open(req)
        started = time.time()
        st = req.streamer
        st.waiting = self.waiting
        st.resume(0)                                  # the loop's token list starts empty
        prompt = req.prompt_ids if fresh else req.prompt_ids + req.out_ids
        res = self.solo(req, prompt, req.reuse_len if fresh else len(prompt) - 1,
                        req.max_new_tokens - len(req.out_ids), fresh)
        st.waiting = None
        if fresh:
            req.t_prefill = started + res.prefill_seconds
            self.prefill_seconds += res.prefill_seconds
        req.out_ids.extend(res.token_ids)
        req.accept_lengths.extend(res.accept_lengths)
        req.rounds += res.num_rounds
        req.forwards += res.target_forwards
        req.lookup_rounds += res.lookup_rounds
        req.tree_rounds += res.tree_rounds
        req.solo_tokens += len(res.token_ids)
        req.solo_rounds += res.num_rounds
        req.solo_stretches += 1
        st.n_fed = len(req.out_ids)
        if res.token_ids:
            self._settle(req, res.token_ids[-1])
        if req.finish is None and not st.yielded:     # the loop ended on its own account
            req.finish = res.finish_reason
        if req.finish is not None:
            finished.append((req, None))
            return
        st.resume(len(req.out_ids))
        # back to the shared step: the caches hold the prompt and every token but the last
        held = len(req.prompt_ids) + len(req.out_ids) - 1
        for c in req.ctx or ():
            c.trim_to(held)
        req.filled = len(req.prompt_ids)
        req.ctx_len = held
        req.pending_known = True
        self._join(req, mx.array([req.out_ids[-1]], dtype=mx.int32))
        if req.t_prefill is None:
            req.t_prefill = time.time()

    # -- one unit of work ------------------------------------------------------------------------------
    def tick(self) -> list:
        """Advance by one unit — a chunk of a prompt (and, while a long prompt is being read, a
        few steps for the requests already decoding), a single-stream stretch, or one shared
        step — and return ``[(request, GenResult)]`` for the requests that finished."""
        finished: list = []
        if self.filling:
            req = self.filling[0]
            if self.solo is not None and self.n_active == 1 and not self.waiting():
                self.filling.pop(0)
                self._run_solo(req, True, finished)
            else:
                started = time.time()
                if self._read_chunk(req):
                    self.filling.pop(0)
                elif self.rows:
                    # a long prompt, and others already decoding: they get their share of the
                    # time before its next chunk, so a newcomer cannot freeze running streams
                    spent = time.time() - started
                    until = time.time() + spent * (1.0 - self.prefill_share) / self.prefill_share
                    while self.rows:
                        self._step(finished)
                        if time.time() >= until:
                            break
        elif len(self.rows) == 1 and self.solo is not None and not self.waiting():
            req = self._leave(0)
            if req.finish is not None:
                finished.append((req, None))
            else:
                self._run_solo(req, False, finished)
        elif self.rows:
            self._step(finished)
        return [(req, self._result(req)) for req, _ in finished]
