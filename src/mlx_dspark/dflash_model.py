"""DFlash block-diffusion drafter (MLX) — vendored from z-lab/dflash (MIT).

Upstream: https://github.com/z-lab/dflash  (file: dflash/model_mlx.py)
Paper:    DFlash: Block Diffusion for Flash Speculative Decoding — Chen et al.,
          arXiv:2602.06036.

MIT License. Copyright (c) 2026 Z Lab.

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.

Only the *drafter model* classes are vendored here (verbatim), so mlx-dspark can run
z-lab's published DFlash checkpoints natively on Apple Silicon and benchmark them
head-to-head against the DeepSeek DSpark drafter under one lossless verify loop. The
generation/verification loop is mlx-dspark's own (see ``generate.dflash_generate``);
z-lab's ``stream_generate`` / gated-delta rollback paths are intentionally not vendored.

**DFlash 2** (Inco AI, 2026-08 — https://inco.ai/blog/dflash2/; checkpoints
``incoai/*-DFlash2``, Apache-2.0; no paper yet) adds two trained components on the same
backbone, ported here from the merged SGLang reference implementation
(sgl-project/sglang ``srt/models/dflash.py`` + ``srt/speculative/dflash_worker_v2.py``,
Apache-2.0 — the authoritative semantics, per the standing shipped-serving-patches rule):

- ``DFlashGroupedConv`` — a two-tap dynamic depthwise convolution wrapping each attention
  and MLP sublayer (input and output convolved, both coefficient sets from ONE projection
  of the sublayer input). Counters within-block suffix decay; block position 0 (the
  anchor) has no tap-1 predecessor, so the zero-padded shift IS the block-boundary mask.
- ``CandidateSelector`` — keeps the target head's top-K token candidates per mask slot and
  scores the K x K transitions between adjacent slots
  (``score = unary_logit(cand) + <A[pred] * proj(h_slot), B[cand]>``, predecessors at
  slot 0 = the verified anchor token), then walks one coherent path: greedy follows the
  best successor; T>0 samples the walk and returns q over the K candidates for the
  standard lossless spec-sampling accept. The selector consumes the *post-final-norm*
  backbone hidden — the same rows the lm_head reads.

Both are config-gated (``dflash_config.selector_rank`` / ``conv_kernel_size``); DFlash 1
checkpoints load and run byte-identically to before.

**DSpine** (Tencent, 2026-09/11 — Zhang et al., "Draft in Parallel, Condition Through Depth:
Adjacent Causal Injection for Speculative Decoding", arXiv:2609.36173; code
https://github.com/Tencent/DSpine, Apache-2.0; checkpoints ``tencent/*-DSpine-b16``) keeps the
DFlash backbone (block-CAUSAL attention, ``dflash_config.draft_block_causal``) and adds one
trained module, ported here from the authors' reference (``sglang_dspine/models/dspine.py`` +
``kernels/dspine.py`` and their ``dspine_mlx`` port — the authoritative semantics):

- :class:`DSpineCodeBridge` — after EVERY draft layer each block position receives a gated
  message built from its immediate predecessor's *predicted feature* (the layer output
  projected into a fixed "transfer space" made of the target lm_head's whitened rows), so the
  causal conditioning chain unfolds over depth while the block is still drafted in one pass.
  The anchor sends its exact token code. The write replaces the next layer's input RMSNorm.
- the **last-write refinement**: after the backbone, each slot's top-K head candidates are
  rescored conditioned on the predecessor's *decided* token — the final write redone with the
  exact code of each candidate sender — giving a ``[slot, sender, candidate]`` transition table
  that the same greedy / sampled walk DFlash 2's selector uses (and the tree builder) turns into
  a chain. No selector weights: the table is the backbone's own head read through the bridge.

Because every layer is block-causal and the bridge only passes messages forward, a round can
draft just ``cap + 1`` rows and get exactly the full block's first rows (:meth:`draft_width`),
so the drafter's cost tracks the cap instead of the trained block (16 on the published heads).

Architecture (differs from the DSpark drafter in ``model.py``): a Qwen3-style backbone
(silu MLP, separate v_proj, per-head q/k RMSNorm, default RoPE, sliding-window attention
on some layers) that **reuses the target model's embed_tokens + lm_head** (tied), consumes
a multi-layer fused target-hidden context via EAGLE-style KV injection, and predicts a
whole block of mask positions in a single parallel (block-diffusion) pass.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.qwen3 import MLP
from mlx_lm.models.rope_utils import initialize_rope


class CtxWindowCache:
    """One draft layer's ctx K / V rows, kept in TEMPORAL ORDER in one buffer that the block
    attention reads in place.

    :meth:`put` writes a round's ``[new ctx rows ‖ block rows]`` right after the valid rows and
    returns views ``[window ‖ block rows]``. The ctx rows join the window — its last ``max_size``
    rows (``sliding_window - 1``; ``None`` = every row, a full-attention layer) — and the block
    rows are scratch the next call overwrites, so a round copies nothing: when the buffer runs
    out, the rows still visible move to the front of a fresh one, once per ``step`` ctx rows.

    Temporal order is load-bearing: :meth:`DFlashAttention._block_mask` is POSITIONAL (a
    sliding layer hides each block row's oldest ctx rows by index). mlx-lm's rotating cache,
    which this replaced, handed back its ring as stored after a one-row append — the newest ctx
    row at index 0, hidden from every draft slot after each round that accepted nothing.

    ``offset`` is the ABSOLUTE position of the next ctx row (what rope reads) and may run ahead
    of the rows held: a prefix-cache restore presets it to the window's start, and the prefill
    bound / the sliding skip advance it past rows that were never written (:meth:`skip`). The
    buffer's own bookkeeping never reads it (issue #33: sizing a buffer from an absolute
    offset)."""

    step = 256                  # spare ctx rows a fresh buffer gets: one compaction per `step`

    def __init__(self, max_size: int | None = None):
        self.max_size = max_size
        self.keys = None
        self.values = None
        self.offset = 0
        self._start = 0             # valid ctx rows are [_start, _end) of the buffer
        self._end = 0

    @property
    def rows(self) -> int:
        """Ctx rows the window holds."""
        return self._end - self._start

    def skip(self, n: int) -> None:
        """Advance the absolute position past ``n`` rows that will never be appended."""
        self.offset += n

    def put(self, keys, values, n_ctx: int):
        """Write ``keys`` / ``values`` ``[B, kv_heads, n_ctx + scratch, D]`` after the valid
        rows; the first ``n_ctx`` rows join the window. -> ``(K, V, ctx_len)``: views of
        ``[the window's ctx rows ‖ the scratch rows]`` and how many of them are ctx."""
        s = int(keys.shape[2])
        end = self._end
        new_end = end + n_ctx
        first = self._start
        if self.max_size is not None:
            first = max(first, new_end - self.max_size)     # oldest row still in the window
        if self.keys is None or end + s > int(self.keys.shape[2]):
            # no room: [rows still visible ‖ new rows ‖ spare] as one fresh buffer
            old = self.keys is not None and first < end
            b, h, _, d = keys.shape
            spare = self.step + (s - n_ctx)
            self.keys = mx.concatenate(
                ([self.keys[..., first:end, :]] if old else [])
                + [keys, mx.zeros((b, h, spare, d), keys.dtype)], axis=2)
            self.values = mx.concatenate(
                ([self.values[..., first:end, :]] if old else [])
                + [values, mx.zeros((b, h, spare, values.shape[3]), values.dtype)], axis=2)
            self._end = (end - first if old else 0) + n_ctx
            self._start = 0 if self.max_size is None else max(0, self._end - self.max_size)
        else:
            self.keys[..., end: end + s, :] = keys
            self.values[..., end: end + s, :] = values
            self._start, self._end = first, new_end
        self.offset += n_ctx
        hi = self._end + (s - n_ctx)
        return (self.keys[..., self._start: hi, :], self.values[..., self._start: hi, :],
                self._end - self._start)

    def update_and_fetch(self, keys, values):
        """Append ctx rows only (no scratch) -> the window's ``(keys, values)``."""
        k, v, _ = self.put(keys, values, int(keys.shape[2]))
        return k, v


def skip_ctx(cache, n: int) -> None:
    """Advance a drafter ctx cache's absolute position past ``n`` never-appended rows. Only a
    sliding layer's bounded window may run ahead of its rows (:meth:`CtxWindowCache.skip`); a
    full-attention layer's cache only ever sees ``n == 0`` — its window is unbounded, so
    nothing is dropped ahead of it."""
    if n <= 0:
        return
    if getattr(cache, "max_size", None) is not None and hasattr(cache, "skip"):
        cache.skip(n)
    else:
        raise ValueError(f"{type(cache).__name__} cannot skip {n} ctx rows: only a sliding-"
                         "window ctx cache may run ahead of its rows")


@dataclass
class DFlashConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    block_size: int
    target_layer_ids: tuple[int, ...]
    num_target_layers: int
    mask_token_id: int = 0
    rope_scaling: dict[str, Any] | None = None
    layer_types: tuple[str, ...] = field(default_factory=tuple)
    sliding_window: int | None = None
    final_logit_softcapping: float | None = None
    # DFlash 2 (0 = absent -> DFlash 1 behavior, byte-identical)
    selector_rank: int = 0
    selector_top_k: int = 0
    conv_kernel_size: int = 0        # taps; 2 for every published DFlash 2 head
    conv_group_size: int = 16        # channels sharing one dynamic coefficient correction
    output_multiplier: float = 1.0   # scales the selector's unary logits (muse_glimmer); the
    # bilinear term is trained against the TRANSFORMED logit scale, so unlike the DSpark
    # reuse-head rule (raw lm_head only) the multiplier/softcap must be applied here.
    # Where the selector's softcap applies (dflash_config.selector_softcap_mode, the reference
    # patch shipped with incoai/Inkling-2-Mini-DFlash2): "before_correction" (the default, every
    # earlier head) caps the unary logits and adds the bilinear term to them; "after_correction"
    # (the Inkling 2 head) adds the bilinear term to the RAW unary logits and caps the sum.
    selector_softcap_mode: str = "before_correction"
    # Block attention direction — z-lab's / SGLang's top-level `is_causal`, DSpine's
    # `dflash_config.draft_block_causal`. None = the z-lab default: sliding-window layers attend
    # the block causally, full-attention layers bidirectionally (block diffusion). True / False
    # force every layer. The shipped DFlash 2 heads declare `is_causal: false` (bidirectional
    # block over a windowed context); DSpine heads declare causal.
    block_causal: bool | None = None
    # DSpine code bridge (0 = absent -> plain DFlash / DFlash 2)
    bridge_code_size: int = 0        # C: the transfer space (codebook columns)
    bridge_rank: int = 0             # R: the message dimension
    bridge_eps: float = 1e-6
    bridge_layers: tuple[int, ...] = ()   # 0-based draft layers carrying a write (all, in practice)
    bridge_top_k: int = 0            # K: candidates per slot in the last-write refinement


def _build_rope(head_dim, rope_theta, max_position_embeddings, rope_scaling):
    return initialize_rope(
        dims=head_dim,
        base=rope_theta,
        traditional=False,
        scaling_config=rope_scaling,
        max_position_embeddings=max_position_embeddings,
    )


class DFlashAttention(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.scale = config.head_dim ** -0.5
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None
        # z-lab / SGLang: `is_causal` when declared, else sliding layers causal, full layers not
        self.block_causal = (self.is_sliding if config.block_causal is None
                             else bool(config.block_causal))
        self.head_dim = config.head_dim
        # A sliding layer's block rows each have their own window (row i sees the last
        # ``window - 1 - i`` ctx rows): an array mask once ctx + block pass the window. True =
        # the whole block reads the LAST row's window instead — each row a subset of its own,
        # short of it by at most ``L - 1`` of its oldest rows — so the call needs no array mask
        # and is one the multi-row attention kernel serves (multirow_attn.eligible). Per head,
        # where measured: :meth:`DFlashDraftModel.share_block_window`.
        self.block_window = False
        self.q_proj = nn.Linear(dim, n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * config.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def _window_rows(self, x_ctx, cache):
        """The ctx rows a sliding layer can keep — the last ``window - 1`` — with the cache's
        absolute position advanced past the rest."""
        S = x_ctx.shape[1]
        if self.is_sliding:
            keep_ctx = self.sliding_window - 1
            if keep_ctx < S:
                skip_ctx(cache, S - keep_ctx)
                x_ctx = x_ctx[:, S - keep_ctx:]
        return x_ctx

    def _kv(self, rows, rope, offset: int):
        """Rows ``[B, S, H]`` at absolute positions ``offset..`` -> their roped keys and
        values ``[B, kv_heads, S, head_dim]``."""
        B, S, _ = rows.shape
        keys = self.k_norm(self.k_proj(rows).reshape(B, S, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        values = self.v_proj(rows).reshape(B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        return rope(keys, offset=offset), values

    def append_ctx(self, x_ctx, rope, cache):
        """Project + rope + append ctx rows into this layer's cache, returning the window's
        (keys, values). The ctx half of ``__call__``, split out so a prefix-cache restore can
        rebuild the drafter context without running a draft block (kv-proj-only, the same path
        SGLang's worker feeds context through). A sliding layer keeps only the last
        ``window - 1`` rows and advances ``cache.offset`` past the skipped ones."""
        x_ctx = self._window_rows(x_ctx, cache)
        keys, values = self._kv(x_ctx, rope, cache.offset)
        return cache.update_and_fetch(keys, values)

    def __call__(self, x, x_ctx, rope, cache):
        B, L, _ = x.shape
        x_ctx = self._window_rows(x_ctx, cache)
        S = x_ctx.shape[1]
        pos = cache.offset                       # the ctx rows rope from here, the block after
        ctx_keys, ctx_values = self._kv(x_ctx, rope, pos)
        prop_keys, prop_values = self._kv(x, rope, pos + S)
        queries = self.q_norm(self.q_proj(x).reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        queries = rope(queries, offset=pos + S)
        # one write of [ctx ‖ block] into the layer's buffer; the attention reads a view of it
        keys, values, ctx_len = cache.put(mx.concatenate([ctx_keys, prop_keys], axis=2),
                                          mx.concatenate([ctx_values, prop_values], axis=2), S)
        if self.block_window and self.is_sliding and ctx_len + L > self.sliding_window:
            drop = min(ctx_len, ctx_len + L - self.sliding_window)
            keys, values, ctx_len = keys[..., drop:, :], values[..., drop:, :], ctx_len - drop
        mask = self._block_mask(L, ctx_len)
        output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=mask)
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))

    def _block_mask(self, L: int, ctx_len: int):
        """The ``L`` block rows' mask over ``[ctx rows | block rows]`` (keys), z-lab's semantics:
        a causal block sees ctx + block rows ``0..t``; a bidirectional block sees every block row;
        a sliding layer sees only the ctx rows within its window either way. ``None`` = all keys,
        ``"causal"`` = SDPA's bottom-right-aligned causal mask, else a boolean [L, ctx + L]."""
        window = self.sliding_window
        fits = not self.is_sliding or ctx_len + L <= window
        if self.block_causal:
            return "causal" if fits else create_causal_mask(L, offset=ctx_len, window_size=window)
        if fits:
            return None
        query = ctx_len + mx.arange(L)[:, None]
        key = mx.arange(ctx_len + L)[None]
        return ((key < ctx_len) & (query - key < window)) | (key >= ctx_len)


class DFlashGroupedConv(nn.Module):
    """DFlash 2: grouped dynamic depthwise K-tap convolution across one draft block.

    Wraps one sublayer: ``prepare`` convolves its input and returns the coefficient set for
    ``finish`` to convolve its output — BOTH sets come from one projection of the input.
    Each coefficient is a learned per-channel base plus a content-adaptive per-group
    correction (``group_size`` channels share one correction). Applied only to the block
    rows (the ctx path never sees it), so the zero-padded tap shift is exactly the
    reference's block-boundary mask: position 0 (the anchor) has no in-block predecessor,
    position 1 reads the anchor's representation.
    """

    def __init__(self, hidden_size: int, taps: int, group_size: int):
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(f"conv_group_size={group_size} must divide hidden_size={hidden_size}")
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        # [side (in/out), tap, channel] — identity at init (tap 0 = 1), the layout training exports
        self.base_kernel = mx.concatenate(
            [mx.ones((2, 1, hidden_size)), mx.zeros((2, taps - 1, hidden_size))], axis=1)
        self.kernel_projection = nn.Linear(hidden_size, 2 * taps * self.num_groups, bias=False)

    def _convolve(self, x, delta, side: int):
        # x [B, L, H]; delta [B, L, taps, num_groups]
        B, L, H = x.shape
        xg = x.reshape(B, L, self.num_groups, self.group_size)
        coeff = (self.base_kernel[side].reshape(1, 1, self.taps, self.num_groups, self.group_size)
                 + delta[..., None])
        out = coeff[:, :, 0] * xg
        for t in range(1, self.taps):
            shifted = mx.pad(xg[:, :-t], ((0, 0), (t, 0), (0, 0), (0, 0)))
            out = out + coeff[:, :, t] * shifted
        return out.reshape(B, L, H)

    def prepare(self, x):
        coeff = self.kernel_projection(x).reshape(*x.shape[:-1], 2, self.taps, self.num_groups)
        return self._convolve(x, coeff[..., 0, :, :], 0), coeff[..., 1, :, :]

    def finish(self, y, delta):
        return self._convolve(y, delta, 1)


class CandidateSelector(nn.Module):
    """DFlash 2: scores the K x K transitions between adjacent mask slots, then walks them.

    ``lattice`` builds ``scores[slot, p, c] = unary[slot, c] + <A[pred] * proj(h_slot), B[c]>``
    where the predecessors of slot 0 are all the verified anchor token and of slot s are
    slot s-1's candidates. ``walk_greedy`` follows the best successor from the anchor;
    ``walk_sampled`` draws the walk by inverse CDF (one uniform per slot, shared across
    predecessor rows, matching the reference) and returns the realized q rows over the K
    candidates for the spec-sampling accept. Scores are fp32 like the reference (bf16
    bilinear + fp32 unary).
    """

    def __init__(self, hidden_size: int, vocab_size: int, rank: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.predecessor_codebook = mx.zeros((vocab_size, rank))
        self.successor_codebook = mx.zeros((vocab_size, rank))
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)

    def lattice(self, candidate_ids, unary_logits, hidden, anchor_id: int):
        # candidate_ids [g, K] int; unary_logits [g, K] fp32; hidden [g, H] (post-final-norm)
        k = candidate_ids.shape[1]
        h = self.hidden_projection(hidden)                                    # [g, r]
        anchor = mx.full((1, k), anchor_id, dtype=candidate_ids.dtype)
        pred_ids = mx.concatenate([anchor, candidate_ids[:-1]], axis=0)       # [g, K]
        pre = self.predecessor_codebook[pred_ids] * h[:, None, :]             # [g, K, r]
        suc = self.successor_codebook[candidate_ids]                          # [g, K, r]
        bilinear = pre @ suc.transpose(0, 2, 1)                               # [g, Kpred, Kcand]
        return unary_logits[:, None, :].astype(mx.float32) + bilinear.astype(mx.float32)

    def walk_greedy(self, scores, candidate_ids):
        """Chain-argmax path -> drafted token ids [g], fully in-graph (no sync)."""
        return walk_greedy(scores, candidate_ids)

    def walk_sampled(self, scores, candidate_ids, uniforms, temperature: float):
        """Sampled walk -> (drafted ids [g], q rows [g, K] the tokens were drawn from)."""
        return walk_sampled(scores, candidate_ids, uniforms, temperature)


TOPK_BLOCK = 256


def topk_ids(logits, k: int, block: int = TOPK_BLOCK):
    """Ids ``[R, k]`` (int32, unordered) of the ``k`` largest values in each row of ``logits
    [R, V]``: the set ``mx.argpartition`` over the whole row gives, in two cheap stages. Every
    element of a row's top-k lies in a ``block`` whose maximum is at least the k-th largest
    value, and at most ``k`` blocks qualify — so the top-k of the ``k`` blocks with the largest
    maxima IS the row's top-k (an exact tie at the k-th value may resolve to another member of
    the tie). A draft head's candidates over a 200k vocabulary: 7 rows 0.9 -> ~0.15 ms on an M4
    Pro (NOTES "DFlash2 on Inkling-2-Mini"). Small vocabularies take one argpartition."""
    rows, v = int(logits.shape[0]), int(logits.shape[-1])
    nb = -(-v // block)
    if nb < 4 * k:
        return mx.argpartition(logits, kth=-k, axis=-1)[:, -k:].astype(mx.int32)
    pad = nb * block - v
    if pad:
        logits = mx.concatenate([logits, mx.full((rows, pad), -float("inf"), logits.dtype)], axis=1)
    blocks = logits.reshape(rows, nb, block)
    best = mx.argpartition(blocks.max(axis=-1), kth=-k, axis=-1)[:, -k:]          # [R, k] blocks
    cand = mx.take_along_axis(blocks, best[..., None], axis=1).reshape(rows, k * block)
    pos = mx.argpartition(cand, kth=-k, axis=-1)[:, -k:]                          # [R, k]
    return (mx.take_along_axis(best, pos // block, axis=1) * block + pos % block).astype(mx.int32)


def walk_greedy(scores, candidate_ids):
    """Chain-argmax over a ``[slots, K predecessor, K candidate]`` transition lattice (DFlash 2's
    selector scores or DSpine's refinement table): slot 0's predecessor rows are all the anchor,
    each later slot's row is the chosen predecessor's. -> drafted ids [slots], in-graph."""
    g = scores.shape[0]
    idx = mx.argmax(scores[0, 0])           # slot 0: every predecessor row is the anchor
    picks = [candidate_ids[0, idx]]
    for s in range(1, g):
        idx = mx.argmax(scores[s, idx])
        picks.append(candidate_ids[s, idx])
    return mx.stack(picks)


def walk_sampled(scores, candidate_ids, uniforms, temperature: float):
    """Sampled walk over the same lattice (one uniform per slot, shared across predecessor rows,
    inverse CDF of the softmaxed row) -> (drafted ids [g], q rows [g, K] the tokens were drawn
    from — the proposal the spec-sampling accept needs)."""
    g, k = candidate_ids.shape
    t = max(float(temperature), 1e-5)
    init = mx.softmax(scores[0, 0] / t, axis=-1)                          # [K]
    idx = mx.minimum((uniforms[0] >= mx.cumsum(init)).sum(), k - 1)
    picks, q_rows = [candidate_ids[0, idx]], [init]
    if g > 1:
        trans = mx.softmax(scores[1:] / t, axis=-1)                       # [g-1, K, K]
        maps = mx.minimum(
            (uniforms[1:, None, None] >= mx.cumsum(trans, axis=-1)).sum(axis=-1), k - 1)
        for s in range(1, g):
            q_rows.append(trans[s - 1, idx])     # realized row BEFORE stepping
            idx = maps[s - 1, idx]
            picks.append(candidate_ids[s, idx])
    return mx.stack(picks), mx.stack(q_rows)


_F32_TINY = 1.1754943508222875e-38      # torch.finfo(float32).tiny — the reference's RMS floor


class DSpineCodeBridge(nn.Module):
    """DSpine's adjacent causal injection: one gated write after every draft layer.

    Parameter names and shapes match the checkpoint under ``selected_prefix_bridge.`` so it
    loads 1:1. With ``u`` = a layer's output rows [B, T, H] (the pre-write residual stream) and
    ``C[v]`` the fixed codebook (the target lm_head's rows, l2-normalized, PCA-whitened to ``C``
    dims, normalized again — paper App. A; shipped in the checkpoint):

        z_t = sqrt(C) · normalize(R_ℓ · u_t / (RMS(u_t) + eps))   the predicted feature; the
                                                                   anchor (t = 0) sends sqrt(C)·C[x]
        g_t = sigmoid(G_q z_t + G_p z_{t-1} + b)                   receiver + sender gate
        m_t = g_t ⊙ A z_{t-1}                                      the predecessor's message
        h_t = u_t + RMS(u_t) · W_ℓ m_t          for t >= 1; the anchor row is never written

    then the NEXT layer's input RMSNorm (the final norm after the last layer) is applied to
    ``h`` — fused into the write in the reference, so the following layer takes its input
    pre-normed (:meth:`DFlashDecoderLayer.__call__` ``x_normed``). ``G_q`` (``gate``, with the
    bias ``b``), ``G_p`` (``predecessor_gate``) and ``A`` (``prev``) are shared across layers;
    ``R_ℓ`` (``early_read_weights[ℓ]``, ``read_weight`` for the last layer) and ``W_ℓ``
    (``feature_write_weights[ℓ]``) are per layer. ``out`` is the retired "terminal" projection
    the checkpoints ship zeroed (``selected_prefix_terminal: false``): loaded, never used.

    Math follows ``sglang_dspine/kernels/dspine.py`` operation for operation; the rounding points
    are the backbone's dtype (bf16 on the published heads, fp32 in the tests), not the reference
    kernels' — the target verifies every token, so only acceptance could differ, and only in the
    fp-tie class.
    """

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.code_size = int(config.bridge_code_size)
        self.rank = int(config.bridge_rank)
        self.num_layers = int(config.num_hidden_layers)
        self.eps = float(config.bridge_eps)
        self.layers = frozenset(int(i) for i in config.bridge_layers)
        self.sqrt_c = math.sqrt(self.code_size)
        H, C, R, L = config.hidden_size, self.code_size, self.rank, self.num_layers
        self.prev = nn.Linear(C, R, bias=False)               # A
        self.gate = nn.Linear(C, R, bias=True)                # G_q (+ b)
        self.predecessor_gate = nn.Linear(C, R, bias=False)   # G_p
        self.out = nn.Linear(R, C, bias=False)                # terminal projection: zero, unused
        self.codebook = mx.zeros((config.vocab_size, C))
        self.read_weight = mx.zeros((C, H))
        self.early_read_weights = mx.zeros((max(L - 1, 1), C, H))
        self.feature_write_weights = mx.zeros((L, H, R))
        self._packed = None                                   # [3R, C] = [G_q; G_p; A], derived

    def read_weight_for(self, layer: int):
        return self.read_weight if layer == self.num_layers - 1 else self.early_read_weights[layer]

    def packed_weight(self):
        """The three shared projections as one ``[3R, C]`` matrix (built on first use; call
        :meth:`reset` after replacing ``gate`` / ``predecessor_gate`` / ``prev`` weights)."""
        if self._packed is None:
            self._packed = mx.concatenate(
                [self.gate.weight, self.predecessor_gate.weight, self.prev.weight], axis=0)
        return self._packed

    def reset(self) -> None:
        self._packed = None

    def __call__(self, hidden, block_ids, layer: int, next_norm, want_stash: bool = False):
        """One write after ``layer``. ``hidden`` [B, T, H] = the layer's output rows; ``block_ids``
        [B, T] (column 0 = the anchor token). Returns ``(next_norm(h), h, stash)`` where ``stash``
        (``want_stash``, the last layer) = the pre-write state of rows 1.. the refinement redoes
        the write from — ``{"raw": u, "rms": RMS(u), "q": G_q z}``, fp32, aligned with
        ``logits_start=1``."""
        dt = hidden.dtype
        f = hidden.astype(mx.float32)
        rms = mx.sqrt(mx.maximum(mx.mean(f * f, axis=-1), _F32_TINY))              # [B, T]
        normalized = (f / (rms[..., None] + self.eps)).astype(dt)
        projected = (normalized @ self.read_weight_for(layer).T).astype(mx.float32)  # [B, T, C]
        norm = mx.sqrt(mx.sum(projected * projected, axis=-1, keepdims=True))
        codes = self.sqrt_c * projected / (norm + self.eps)
        anchor_codes = self.sqrt_c * self.codebook[block_ids[:, 0]].astype(mx.float32)
        codes = mx.concatenate([anchor_codes[:, None, :], codes[:, 1:, :]], axis=1)
        packed = (codes.astype(dt) @ self.packed_weight().T).astype(mx.float32)     # [B, T, 3R]
        q, s, v = mx.split(packed, [self.rank, 2 * self.rank], axis=-1)
        gate = mx.sigmoid(q[:, 1:] + s[:, :-1] + self.gate.bias.astype(mx.float32))
        message = (gate * v[:, :-1]).astype(dt)                                     # [B, T-1, R]
        delta = (message @ self.feature_write_weights[layer].T).astype(mx.float32)  # [B, T-1, H]
        h = mx.concatenate([f[:, :1], f[:, 1:] + rms[:, 1:, None] * delta], axis=1).astype(dt)
        stash = {"raw": f[:, 1:], "rms": rms[:, 1:], "q": q[:, 1:]} if want_stash else None
        return next_norm(h), h, stash

    def transition_table(self, raw, rms, q, candidate_ids, anchor_id: int, head_rows,
                         norm_weight, norm_eps: float):
        """The last-write refinement as a ``[slots, K sender, K candidate]`` fp32 table: for slot
        t and each candidate sender s of slot t-1 (slot 0's senders are all the anchor), redo the
        final write from the stashed ``raw`` / ``rms`` / ``q`` rows with the EXACT code
        ``sqrt(C)·C[s]`` in place of the predicted feature, apply the final norm, and score slot
        t's own candidates with their target-head rows (``head_rows(ids) -> [..., H]``). Rows of
        the table are conditional logits over the K candidates (the reference's sampling table;
        its greedy kernel drops the per-row norm scale, which leaves every argmax unchanged)."""
        k = candidate_ids.shape[1]
        anchor = mx.full((1, k), int(anchor_id), dtype=candidate_ids.dtype)
        sid = mx.concatenate([anchor, candidate_ids[:-1]], axis=0)                  # [n, K]
        exact = self.sqrt_c * self.codebook[sid].astype(mx.float32)                  # [n, K, C]
        gate = mx.sigmoid(q[:, None, :] + exact @ self.predecessor_gate.weight.astype(mx.float32).T
                          + self.gate.bias.astype(mx.float32))                       # [n, K, R]
        value = exact @ self.prev.weight.astype(mx.float32).T
        w_last = self.feature_write_weights[self.num_layers - 1].astype(mx.float32)  # [H, R]
        refined = raw[:, None, :] + rms[:, None, None] * ((gate * value) @ w_last.T)  # [n, K, H]
        normalized = (refined * mx.rsqrt(mx.mean(refined * refined, axis=-1, keepdims=True)
                                         + float(norm_eps)) * norm_weight.astype(mx.float32))
        rows = head_rows(candidate_ids).astype(mx.float32)                           # [n, K, H]
        return normalized @ rows.swapaxes(1, 2)                                      # [n, Ks, Kc]


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.self_attn = DFlashAttention(config, layer_idx)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.conv_kernel_size:
            self.attention_conv = DFlashGroupedConv(
                config.hidden_size, config.conv_kernel_size, config.conv_group_size)
            self.mlp_conv = DFlashGroupedConv(
                config.hidden_size, config.conv_kernel_size, config.conv_group_size)
        else:
            self.attention_conv = None
            self.mlp_conv = None

    def __call__(self, x, x_ctx, rope, cache, x_normed=None):
        # ``x_normed``: this layer's input already normalized by a fused DSpine write (the
        # bridge applies the next layer's input_layernorm); ``x`` stays the residual stream.
        a = self.input_layernorm(x) if x_normed is None else x_normed
        if self.attention_conv is not None:
            a, out_coeff = self.attention_conv.prepare(a)
        attn = self.self_attn(a, x_ctx, rope, cache)
        if self.attention_conv is not None:
            attn = self.attention_conv.finish(attn, out_coeff)
        h = x + attn
        m = self.post_attention_layernorm(h)
        if self.mlp_conv is not None:
            m, out_coeff = self.mlp_conv.prepare(m)
        mlp = self.mlp(m)
        if self.mlp_conv is not None:
            mlp = self.mlp_conv.finish(mlp, out_coeff)
        return h + mlp


def _block_window_default(measured: bool) -> bool:
    """Whether a head's block shares one ctx window: ``measured`` (the pair was measured with
    it) unless ``MLX_DSPARK_DFLASH_BLOCK_WINDOW`` says 1 / 0."""
    env = os.environ.get("MLX_DSPARK_DFLASH_BLOCK_WINDOW", "").strip().lower()
    if env in ("1", "true", "on"):
        return True
    if env in ("0", "false", "off"):
        return False
    return measured


class DFlashDraftModel(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        if not self.config.layer_types:
            self.config.layer_types = ("full_attention",) * self.config.num_hidden_layers
        concat_dim = len(config.target_layer_ids) * config.hidden_size
        self.fc = nn.Linear(concat_dim, config.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [DFlashDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope = _build_rope(
            config.head_dim, config.rope_theta, config.max_position_embeddings, config.rope_scaling
        )
        self.candidate_selector = (
            CandidateSelector(config.hidden_size, config.vocab_size,
                              config.selector_rank, config.selector_top_k)
            if config.selector_rank else None
        )
        self.selected_prefix_bridge = DSpineCodeBridge(config) if config.bridge_rank else None
        self.embed_tokens = None
        self.lm_head = None
        self._head_rows_src = None       # the module whose rows ARE the head (see head_rows)
        self.embed_scale = 1.0
        self.valid_vocab = None          # the bound target's real vocabulary (see vocab_out)

    @property
    def vocab_out(self) -> int:
        """Token ids a draft may take and the width of a sampled round's q: the bound target's
        REAL vocabulary when its head is padded past it (Inkling 2: 200,058 of 201,024 rows —
        the target's logits stop there, and the reference patch limits the candidates to it),
        else the config's."""
        return int(self.valid_vocab or self.config.vocab_size)

    @property
    def _softcap_after(self) -> bool:
        return (self.config.selector_softcap_mode == "after_correction"
                and self.config.final_logit_softcapping is not None)

    @property
    def selects_paths(self) -> bool:
        """Drafts one coherent chain through a candidate lattice (DFlash 2's selector or
        DSpine's refinement) via :meth:`select_block` / :meth:`select_lattice`, instead of
        per-slot argmax — what the generate loop, the tree builder and the calibrator key on."""
        return self.candidate_selector is not None or self.selected_prefix_bridge is not None

    @property
    def block_causal(self) -> bool:
        """Every layer attends the block causally (rows 0..t never see rows > t)."""
        return all(layer.self_attn.block_causal for layer in self.layers)

    def share_block_window(self, on: bool = True) -> None:
        """Every sliding layer's block reads its LAST row's ctx window (no array mask: see
        ``DFlashAttention.block_window``) instead of a window per row. Changes what is drafted
        by the oldest ``L - 1`` ctx rows of ~``sliding_window``, so it is on only for heads
        whose acceptance was measured with it: :meth:`bind` turns it on for Inkling 2's DFlash 2
        head (NOTES "The DFlash drafter's ctx window …"); ``MLX_DSPARK_DFLASH_BLOCK_WINDOW``
        (1 / 0) forces it for any head — the A/B."""
        for layer in self.layers:
            layer.self_attn.block_window = bool(on)

    def draft_width(self, cap: int) -> int:
        """Block rows to feed for a round verifying ``cap`` drafts: the anchor + ``cap`` masks on
        a block-causal head (exact truncation — attention and the bridge's messages both flow
        forward only, so the first rows equal the full block's; DSpine), the trained block for
        block diffusion (a bidirectional block depends on all of its rows)."""
        bs = int(self.config.block_size)
        return max(2, min(bs, int(cap) + 1)) if self.block_causal else bs

    def bind(self, target_model):
        """Wire the drafter to the *target's* embed_tokens + lm_head (DFlash reuses them).

        Inkling 2 binds what its own draft heads read (SGLang's ``get_input_embeddings`` for the
        model, which ``dflash_worker_v2`` embeds the block with): the FULL input embedding
        ``embed_norm ∘ embed_tokens``, the head's fp32 logits, and its real vocabulary
        (:func:`~mlx_dspark.inkling.dflash_bindings`)."""
        from .inkling import dflash_bindings

        inkling = dflash_bindings(target_model)
        self.share_block_window(_block_window_default(measured=inkling is not None))
        if inkling is not None:
            embed, head, head_src, vocab = inkling
            self.embed_tokens, self.embed_scale = embed, 1.0
            self.lm_head, self._head_rows_src = head, head_src
            self.valid_vocab = vocab
            return self
        self.valid_vocab = None
        if hasattr(target_model, "embed_tokens"):
            inner = target_model
        elif hasattr(target_model, "model") and hasattr(target_model.model, "embed_tokens"):
            inner = target_model.model
        elif (hasattr(target_model, "language_model") and
              hasattr(target_model.language_model, "model") and
              hasattr(target_model.language_model.model, "embed_tokens")):
            inner = target_model.language_model.model
        else:
            raise AttributeError(f"Cannot find embed_tokens in {type(target_model).__name__}")
        self.embed_tokens = inner.embed_tokens
        self.embed_scale = getattr(self.embed_tokens, "embed_scale", getattr(inner, "embed_scale", 1.0))
        lm = getattr(target_model, "language_model", target_model)
        head = getattr(target_model, "lm_head", None) or getattr(lm, "lm_head", None)
        if head is not None:
            self.lm_head = head
            self._head_rows_src = head
        else:                                   # tied: the embedding table IS the head
            self.lm_head = self.embed_tokens.as_linear
            self._head_rows_src = self.embed_tokens
        return self

    def head_rows(self, ids):
        """Dense rows of the bound target head for token ``ids`` (any int shape -> [..., H]),
        dequantized from a quantized head's packed rows (what DSpine's refinement scores its
        candidates with; a gather, not a vocab-wide projection)."""
        src = self._head_rows_src
        if src is None:
            raise RuntimeError("bind() the drafter to a target before reading head rows")
        if hasattr(src, "scales"):              # nn.QuantizedLinear / nn.QuantizedEmbedding
            biases = getattr(src, "biases", None)
            return mx.dequantize(src.weight[ids], src.scales[ids],
                                 None if biases is None else biases[ids],
                                 group_size=src.group_size, bits=src.bits,
                                 mode=getattr(src, "mode", "affine"))
        weight = getattr(src, "weight", None)
        if weight is None:
            raise TypeError(f"{type(src).__name__} exposes no head weight rows")
        return weight[ids]

    # -- the drafter's whole state as a window of projected ctx rows: the prefix cache's
    # checkpoints and the batched session (inkling_batch.Session) hold it this way ----------
    @property
    def ctx_window_rows(self) -> int | None:
        """Projected ctx rows the drafter can ever attend: ``sliding_window - 1`` when every
        layer is sliding, unbounded (None) when any layer attends the whole context."""
        cfg = self.config
        if cfg.sliding_window and cfg.layer_types and all(
                t == "sliding_attention" for t in cfg.layer_types):
            return int(cfg.sliding_window) - 1
        return None

    def make_ctx_cache(self) -> list:
        """``[DFlashCtxWindow]``: what :func:`~mlx_dspark.generate.dflash_generate` takes as
        ``ctx_caches`` to restore the drafter from, and keeps current when it yields."""
        return [DFlashCtxWindow(self.ctx_window_rows)]

    def update_context(self, target_hidden, ctx_offset: int = 0, ctx_caches=None) -> None:
        """Add the fused target rows of positions ``[ctx_offset, ctx_offset + n)`` to the
        window in ``ctx_caches`` (projected; only the rows the window can keep are projected).
        The DSpark drafters' interface: chunked prefill and a batched session's steps feed it."""
        n = int(target_hidden.shape[1])
        keep = self.ctx_window_rows
        if keep is not None and n > keep:
            target_hidden, ctx_offset = target_hidden[:, n - keep:], int(ctx_offset) + n - keep
        ctx_caches[0].extend(self.project_ctx(target_hidden), start=int(ctx_offset))

    def make_cache(self):
        caches = []
        for layer_type in self.config.layer_types:
            if layer_type == "sliding_attention":
                if self.config.sliding_window is None:
                    raise ValueError("Draft config must define sliding_window for sliding_attention layers.")
                caches.append(CtxWindowCache(self.config.sliding_window - 1))
            else:
                caches.append(CtxWindowCache())
        return caches

    def project_ctx(self, target_hidden):
        """Fused target hidden [B, S, taps*H] -> the drafter's context rows [B, S, H]
        (``hidden_norm(fc(.))``). Per-position, so computing it over any slice matches the
        full-width result up to qmm width ulps (fp-tie class)."""
        return self.hidden_norm(self.fc(target_hidden))

    def append_ctx(self, h_ctx, cache):
        """Append PROJECTED context rows (from :meth:`project_ctx`) to every layer's ctx
        cache without drafting — used by a prefix-cache restore to rebuild the drafter
        context from a stored window (:class:`DFlashCtxWindow`). Advance each layer cache
        to the rows' start position first (:func:`skip_ctx`); the fresh path instead appends
        ctx inside the first draft call, same math."""
        for layer, c in zip(self.layers, cache):
            layer.self_attn.append_ctx(h_ctx, self.rope, c)

    def forward_hidden(self, inputs, target_hidden, cache, logits_start: int = 0):
        """Backbone forward -> post-final-norm hidden [B, L - logits_start, H]. (Slicing
        before or after the per-token norm is identical; sliced first so DFlash 2's
        selector never norms rows nobody reads.)"""
        return self._forward(inputs, target_hidden, cache, logits_start)[0]

    def _forward(self, inputs, target_hidden, cache, logits_start: int = 0,
                 want_stash: bool = False):
        """:meth:`forward_hidden` plus, on a DSpine head, the last write's pre-write stash
        (rows 1.., see :meth:`DSpineCodeBridge.__call__`) its refinement redoes; ``None`` else."""
        h = self.embed_tokens(inputs) * self.embed_scale
        h_ctx = self.project_ctx(target_hidden)
        # the ctx rows this call appends, projected: a loop that must hand the drafter's state
        # on (dflash_generate under a batched session) keeps them for its DFlashCtxWindow
        object.__setattr__(self, "_ctx_rows", h_ctx)
        bridge = self.selected_prefix_bridge
        last = len(self.layers) - 1
        normed, stash = None, None
        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            h = layer(h, h_ctx, self.rope, c, x_normed=normed)
            normed = None
            if bridge is not None and i in bridge.layers:
                next_norm = self.layers[i + 1].input_layernorm if i < last else self.norm
                normed, h, st = bridge(h, inputs, i, next_norm,
                                       want_stash=want_stash and i == last)
                stash = st if st is not None else stash
        if normed is None:                       # no fused final write: norm the rows read
            if logits_start:
                h = h[:, logits_start:]
            return self.norm(h), stash
        return (normed[:, logits_start:] if logits_start else normed), stash

    def __call__(self, inputs, target_hidden, cache, logits_start: int = 0):
        logits = self.lm_head(self.forward_hidden(inputs, target_hidden, cache, logits_start))
        if self.valid_vocab is not None:
            logits = logits[..., : self.vocab_out]
        if self.config.final_logit_softcapping is not None:
            cap = self.config.final_logit_softcapping
            logits = mx.tanh(logits / cap) * cap
        return logits

    def _transform_unary(self, logits):
        """DFlash 2 unary-logit transform (reference ``_transform_unary_logits``): fp32,
        then the target's output multiplier + softcap. The bilinear selector term is
        trained against THIS scale — feeding raw logits would misweight it (the inverse
        of the DSpark reuse-head raw-lm_head rule; the reference code is the proof).
        ``selector_softcap_mode: after_correction`` moves the softcap past the correction
        (:meth:`_lattice`): the unary logits stay raw here."""
        logits = logits.astype(mx.float32)
        if self.config.output_multiplier != 1.0:
            logits = logits * self.config.output_multiplier
        if self.config.final_logit_softcapping is not None and not self._softcap_after:
            cap = self.config.final_logit_softcapping
            logits = mx.tanh(logits / cap) * cap
        return logits

    def select_block(self, block, target_hidden, cache, *, cap: int, anchor_id: int,
                     uniforms=None, temperature: float = 1.0):
        """DFlash 2 drafting: backbone forward, target-head top-K per mask slot, selector
        walk from the anchor. Greedy (``uniforms`` None) returns
        ``(draft_ids [cap], None, None)`` fully in-graph; sampled returns
        ``(draft_ids [cap], candidate_ids [cap, K], q_rows [cap, K])`` — scatter q_rows at
        candidate_ids for the dense q the spec-sampling accept consumes."""
        candidate_ids, scores, _ = self._lattice(block, target_hidden, cache, cap, anchor_id)
        if uniforms is None:
            return walk_greedy(scores, candidate_ids), None, None
        draft_ids, q_rows = walk_sampled(scores, candidate_ids, uniforms, temperature)
        return draft_ids, candidate_ids, q_rows

    def select_lattice(self, block, target_hidden, cache, *, cap: int, anchor_id: int):
        """DFlash 2 drafting for TREE verify: ``(candidate_ids [cap, K], scores [cap, K, K],
        chain [cap])`` — the lattice :meth:`select_block` walks (same backbone, head top-K and
        selector, one shared code path) plus the greedy chain it would have returned. Lazy: the
        loop builds the tree from it on the GPU (``tree.device_tree``) inside the same graph."""
        candidate_ids, scores, _ = self._lattice(block, target_hidden, cache, cap, anchor_id)
        return candidate_ids, scores, walk_greedy(scores, candidate_ids)

    def _lattice(self, block, target_hidden, cache, cap: int, anchor_id: int):
        """Backbone -> post-norm hidden of the ``cap`` mask slots -> target-head top-K ->
        the ``[cap, K, K]`` transition scores: DFlash 2's selector lattice, or DSpine's
        last-write refinement table. Returns ``(candidate_ids, scores, hidden)``."""
        if self.selected_prefix_bridge is not None:
            return self._bridge_lattice(block, target_hidden, cache, cap, anchor_id)
        sel = self.candidate_selector
        hidden = self.forward_hidden(block, target_hidden, cache, logits_start=1)[0][:cap]
        # trim a padded target head to the real vocab (reference: weight[:org_vocab_size],
        # and the Inkling 2 patch's valid_vocab_size) — a candidate id beyond it would gather
        # garbage codebook rows, or propose a token the target's logits do not cover
        logits = self.lm_head(hidden)[..., : self.vocab_out]                  # [cap, V]
        candidate_ids = topk_ids(logits, sel.top_k)
        unary = self._transform_unary(mx.take_along_axis(logits, candidate_ids, axis=-1))
        scores = sel.lattice(candidate_ids, unary, hidden, anchor_id)
        if self._softcap_after:
            # the reference patch's CandidateSelector.post_softcap: the trained selector caps
            # unary + correction together (monotonic per row, so the greedy walk is the raw
            # sum's; the sampled walk's softmax is not)
            cap_v = self.config.final_logit_softcapping
            scores = mx.tanh(scores / cap_v) * cap_v
        return candidate_ids, scores, hidden

    def _bridge_lattice(self, block, target_hidden, cache, cap: int, anchor_id: int):
        """DSpine: backbone (``cap + 1`` rows on a causal block) -> head top-K per slot ->
        the refinement table over them (raw head logits: the write replaces the head's INPUT,
        so the scores live on the head's own scale)."""
        bridge = self.selected_prefix_bridge
        hidden, stash = self._forward(block, target_hidden, cache, logits_start=1, want_stash=True)
        hidden = hidden[0][:cap]
        logits = self.lm_head(hidden)[..., : self.vocab_out]                  # [cap, V]
        candidate_ids = topk_ids(logits, self.config.bridge_top_k)
        scores = bridge.transition_table(
            stash["raw"][0][:cap], stash["rms"][0][:cap], stash["q"][0][:cap],
            candidate_ids, anchor_id, self.head_rows, self.norm.weight, self.norm.eps)
        return candidate_ids, scores, hidden


class DFlashCtxWindow:
    """Prefix-cache holder for the DFlash drafter's context state.

    The drafter's ctx caches are per-layer projections of one shared row stream
    (``project_ctx`` of the fused target hidden), and a sliding-window head only ever
    attends the last ``window - 1`` of those rows — so the whole drafter state is
    recoverable from a bounded window of projected rows, which is what this holds:
    ``.k`` = the last ``cap`` rows [1, W, H] (None when empty), ``.v`` = a [1] int32
    carrying the window's absolute START position. Those two attribute names are the
    drafter-ctx contract :mod:`~mlx_dspark.prefix_cache` snapshots, restores and trims
    (``.k``/``.v`` arrays + :meth:`trim_to`), so checkpoint slots carry it unchanged.

    A restore rebuilds fresh drafter caches from the rows (``append_ctx`` with each layer
    cache skipped ahead to ``start`` via :func:`skip_ctx`). A window trimmed too deep to cover its rung
    goes EMPTY: drafting then starts context-bare and self-heals as rounds append — less
    acceptance for a while, never a correctness issue (the target verifies every token).

    ``cap`` = ``sliding_window - 1`` when EVERY layer is sliding; None (unbounded — the
    whole prompt's rows) when any layer has full attention, which needs them all.
    """

    def __init__(self, cap: int | None = None):
        self.cap = cap
        self.k = None
        self.v = mx.array([0], dtype=mx.int32)

    @property
    def start(self) -> int:
        return int(self.v[0])

    @property
    def rows(self) -> int:
        return 0 if self.k is None else int(self.k.shape[1])

    @property
    def end(self) -> int:
        return self.start + self.rows

    def set(self, h_ctx, end: int) -> None:
        """Hold the last <= cap of these projected rows, which end at position ``end``."""
        if self.cap is not None and h_ctx.shape[1] > self.cap:
            h_ctx = h_ctx[:, -self.cap:]
        self.k = h_ctx
        self.v = mx.array([end - h_ctx.shape[1]], dtype=mx.int32)

    def extend(self, h_ctx, start: int) -> None:
        """Add projected rows beginning at position ``start``: appended when they continue the
        window (``start == end``), else they replace it (a gap leaves nothing to continue)."""
        if self.k is not None and start == self.end:
            h_ctx, start = mx.concatenate([self.k, h_ctx], axis=1), self.start
        self.set(h_ctx, end=int(start) + int(h_ctx.shape[1]))

    def trim_to(self, n: int) -> None:
        """Drop rows at positions >= n (prefix-cache rung/store trim). Rows before the
        window's start are gone for good, so trimming past it empties the holder."""
        if self.k is None or n >= self.end:
            return
        keep = n - self.start
        if keep > 0:
            self.k = self.k[:, :keep]
        else:
            self.k = None
            self.v = mx.array([n], dtype=mx.int32)
