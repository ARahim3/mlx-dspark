"""Continuous batching for mlx-dspark — run B requests through the target in one
forward so they share a single weight-read per step (verify cost is memory-bound; B
sequences amortize it). This is the "cheap-verify" regime the paper gets from batched GPU
serving, brought to a Mac.

**Model contract (why this is general, not qwen-specific).** The batched forward is the
*generic* mlx-lm dense-model loop — ``model.model.{embed_tokens, layers, norm}`` with
``layer(h, mask, cache)`` and attention that reads ``cache.offset`` for RoPE — the exact
shape ``target.py::_run_mlxlm`` already taps. Any standard mlx-lm dense target (Qwen3-4B/8B/
14B, Llama, Mistral, Phi, …) satisfies it; :func:`batchable` gates on it so a *new dense model*
works automatically and anything unproven (gemma-4 vlm + rotating/sliding cache) cleanly falls
back to the serialized engine instead of decoding wrong. The only batching-specific piece is
:class:`BatchCache`, which duck-types the KV-cache interface the layers call — so it slots into
any of those models unchanged.

Batched *spec* decoding needs each row to roll its KV back by a different amount every round
(rows accept different numbers of drafted tokens). mlx-lm's own ``BatchKVCache`` shares one
write cursor and can only trim uniformly, so this module carries its own **left-aligned,
per-row-offset** cache: row b's real tokens live in columns ``0..offset[b]``, writes scatter to
each row's own offset, and ``trim`` is pure per-row metadata (the same O(1) trick the single-seq
``KVCache`` uses). Per-row RoPE rides ``mx.fast.rope``'s array-offset support (also model-agnostic).
"""

from __future__ import annotations

import time

import mlx.core as mx

from .generate import (
    GenResult,
    _prefill_plain,
    _Streamer,
    eos_token_ids,
)

STEP = 256  # KV buffer grows in chunks of this many columns (like mlx-lm's caches)


# --------------------------------------------------------------------------- capability


def batch_spec_supported(target) -> bool:
    """True iff **batched speculative** decoding works here, which is stricter than
    :func:`batchable`.

    A spec round rolls each row back by a different amount (rows accept different numbers of
    drafted tokens). For an attention layer that is per-row metadata on :class:`BatchCache`.
    For a *recurrent* layer there is no trim at all — the single-row path rebuilds the state by
    re-running the recurrence over the accepted prefix (``Target.rollback``), and doing that at
    a **different ``keep`` per row** needs a masked batched re-run that does not exist yet. So
    hybrids batch their baseline but not their spec rounds; they still get full single-stream
    speculation. See NOTES "Batching hybrid targets"."""
    return batchable(target) and all(
        type(c).__name__ == "KVCache" for c in target.make_cache())


def _is_trimmable(c) -> bool:
    fn = getattr(c, "is_trimmable", None)
    return bool(fn()) if callable(fn) else hasattr(c, "trim")


# Recurrent (linear-attention / SSM) layer caches we know how to batch. Named explicitly
# rather than duck-typed: a *wrapped* RotatingKVCache also looks like "not trimmable, holds
# arrays", and merging one by concatenation would silently decode garbage past the window.
_RECURRENT_CACHES = ("ArraysCache", "MambaCache")


def _recurrent_mergeable(c) -> bool:
    """True for a linear-attention (gated-DeltaNet) layer cache we can batch.

    The enabling property, and the reason hybrids turn out to be *easy* to batch: a linear
    layer's cache is **length-independent**. It holds a fixed-size recurrent state plus a
    conv window of ``kernel-1`` columns — a summary of the sequence so far, not a per-token
    buffer. So rows with different prompt lengths merge with a plain concatenate on the batch
    axis, with no padding, no per-row offsets and no mask. Only the (minority) full-attention
    layers need the left-aligned :class:`BatchCache`.

    Note a *fresh* cache holds ``[None, ...]`` — the states are allocated by the first forward
    — so this is a check on shape-of-cache, not on contents; it has to answer before any
    prefill has run (:func:`batchable` is a capability query).
    """
    st = getattr(c, "state", None)
    return (type(c).__name__ in _RECURRENT_CACHES and not _is_trimmable(c)
            and isinstance(st, (list, tuple)) and len(st) > 0
            and all(s is None or isinstance(s, mx.array) for s in st))


def batchable(target) -> bool:
    """True iff this target supports the batched path.

    Two cache shapes qualify, per layer: the standard ``KVCache`` (batched as a left-aligned
    per-row :class:`BatchCache`) and a length-independent recurrent cache (batched by
    concatenation — see :func:`_recurrent_mergeable`). A model may mix them, which is exactly
    the qwen3_5 / qwen3_5_moe hybrids (Ornith, Bonsai, Qwen3.6-27B, Qwen3.6-35B-A3B): 30 of 40
    layers recurrent, 10 attention. VLM/gemma-4 (rotating / sliding cache, mlx-vlm wrapper)
    returns False -> caller falls back to the serialized engine. Anything else unproven falls
    back too, rather than decoding wrong."""
    if getattr(target, "is_vlm", False):
        return False
    try:
        cache = target.make_cache()
    except Exception:  # noqa: BLE001
        return False
    if not cache:
        return False
    return all(type(c).__name__ == "KVCache" or _recurrent_mergeable(c) for c in cache)


# --------------------------------------------------------------------------- batched cache


class BatchCache:
    """Left-aligned per-row KV cache for one layer of a batched decode.

    ``keys``/``values`` are ``[B, H, Lbuf, D]``; row b's real content occupies columns
    ``0..offsets[b]`` (contiguous, no gaps — trimmed tail garbage is overwritten by the next
    write). ``offsets[b]`` is both the row's logical length and the RoPE start position of its
    next token. Duck-types what mlx-lm attention calls: ``.offset`` (array, for rope) and
    ``.update_and_fetch``. Masking is supplied externally (see :func:`build_batch_mask`) because
    the per-row causal+padding mask depends only on the offsets the caller already holds.
    """

    def __init__(self, keys: mx.array, values: mx.array, offsets: list[int]):
        self.keys = keys
        self.values = values
        self.offsets = list(offsets)  # python list: cheap per-row indexing for writes/trim/mask
        self.rows: int | None = None  # active-prefix view width (None = all rows); see SpecSlots

    # -- construction --
    @classmethod
    def from_rows(cls, kv_pairs: list[tuple[mx.array, mx.array]]) -> BatchCache:
        """Merge per-row single-seq (keys, values) — each ``[1, H, len_b, D]`` — into one
        left-aligned batched buffer (row b at columns ``0..len_b``)."""
        lens = [k.shape[2] for k, _ in kv_pairs]
        B = len(kv_pairs)
        H = kv_pairs[0][0].shape[1]
        Dk = kv_pairs[0][0].shape[3]
        Dv = kv_pairs[0][1].shape[3]
        dt = kv_pairs[0][0].dtype
        Lbuf = _round_step(max(lens))
        keys = mx.zeros((B, H, Lbuf, Dk), dtype=dt)
        values = mx.zeros((B, H, Lbuf, Dv), dtype=dt)
        for b, (k, v) in enumerate(kv_pairs):
            n = lens[b]
            keys[b : b + 1, :, :n, :] = k
            values[b : b + 1, :, :n, :] = v
        return cls(keys, values, lens)

    @classmethod
    def empty(cls, capacity: int, n_heads: int, dk: int, dv: int, dtype) -> BatchCache:
        """A fixed-capacity cache with every row empty — the slot-reuse form (SpecSlots):
        rows are filled by :meth:`set_row` on admission and vacated by trimming to 0."""
        return cls(mx.zeros((capacity, n_heads, STEP, dk), dtype=dtype),
                   mx.zeros((capacity, n_heads, STEP, dv), dtype=dtype), [0] * capacity)

    # -- slot reuse (dynamic admission) --
    def set_row(self, b: int, keys: mx.array, values: mx.array, offset: int) -> None:
        """Overwrite row b's KV with a single-seq row (``[1, H, offset, D]``) and reset its
        offset — how a freed slot admits a new request without resizing the batch dim."""
        if offset > self.keys.shape[2]:
            self._grow(_round_step(offset))
        if offset:
            self.keys[b : b + 1, :, :offset, :] = keys
            self.values[b : b + 1, :, :offset, :] = values
        self.offsets[b] = offset

    def move_row(self, src: int, dst: int) -> None:
        """Copy row src's live columns onto row dst and vacate src — retirement compaction,
        keeping active rows a contiguous prefix so forwards can run at the active width."""
        n = self.offsets[src]
        if n:
            self.keys[dst : dst + 1, :, :n, :] = self.keys[src : src + 1, :, :n, :]
            self.values[dst : dst + 1, :, :n, :] = self.values[src : src + 1, :, :n, :]
        self.offsets[dst] = n
        self.offsets[src] = 0

    # -- interface the attention layers use --
    @property
    def offset(self) -> mx.array:
        off = self.offsets if self.rows is None else self.offsets[: self.rows]
        return mx.array(off, dtype=mx.int32)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Write each row's ``T`` new tokens at its own offset, advance offsets, and return the
        full ``[B, H, Lcur, D]`` key/value tensors (``Lcur = max(new offsets)``). Rows shorter
        than ``Lcur`` carry stale/other-row content past their offset — the mask hides it.
        ``keys`` may cover only the first B rows of the buffer (the active-prefix view)."""
        B, _H, T, _ = keys.shape
        need = max(o + T for o in self.offsets[:B])
        if need > self.keys.shape[2]:
            self._grow(_round_step(need))
        for b in range(B):
            o = self.offsets[b]
            self.keys[b : b + 1, :, o : o + T, :] = keys[b : b + 1]
            self.values[b : b + 1, :, o : o + T, :] = values[b : b + 1]
            self.offsets[b] = o + T
        Lcur = max(self.offsets[:B])
        return self.keys[:B, :, :Lcur, :], self.values[:B, :, :Lcur, :]

    def _grow(self, new_lbuf: int) -> None:
        B, H, _, Dk = self.keys.shape
        Dv = self.values.shape[3]
        nk = mx.zeros((B, H, new_lbuf, Dk), dtype=self.keys.dtype)
        nv = mx.zeros((B, H, new_lbuf, Dv), dtype=self.values.dtype)
        L = self.keys.shape[2]
        nk[:, :, :L, :] = self.keys
        nv[:, :, :L, :] = self.values
        self.keys, self.values = nk, nv

    def trim(self, ns: list[int]) -> None:
        """Roll each row back by ``ns[b]`` tokens — pure metadata (the trimmed tail is
        overwritten by the next write). ``ns[b] == 0`` leaves row b untouched."""
        for b, n in enumerate(ns):
            self.offsets[b] = max(0, self.offsets[b] - n)


def _round_step(n: int) -> int:
    return ((n + STEP - 1) // STEP) * STEP


def build_batch_mask(offsets: list[int], T: int) -> mx.array:
    """Boolean attention mask ``[B, 1, T, Lcur]`` (Lcur = max(offsets)+T): row b query position
    i (absolute offset[b]+i) attends key column j iff ``j <= offset[b] + i`` — causal within the
    new block *and* padding-safe against shorter rows / stale tail. Broadcasts over heads."""
    off = mx.array(offsets, dtype=mx.int32)[:, None, None, None]  # [B,1,1,1]
    Lcur = max(offsets) + T
    i = mx.arange(T, dtype=mx.int32)[None, None, :, None]         # [1,1,T,1]
    j = mx.arange(Lcur, dtype=mx.int32)[None, None, None, :]      # [1,1,1,Lcur]
    return j <= (off + i)


# --------------------------------------------------------------------------- batched forward


def batched_forward(target, ids: mx.array, caches: list, tap: list[int] | None = None,
                    mask=None):
    """One batched forward through an mlx-lm target with per-row caches.

    ``ids`` is ``[B, T]`` (T=1 for baseline decode, cap+1 for verify). Returns
    ``(logits [B, T, V], fused [B, T, n_tap*H] | None)``. The generic layer loop mirrors
    ``target.py::_run_mlxlm`` / ``_body_hybrid`` (embed -> layers(h, mask, cache) -> norm ->
    head), so it is not tied to any one model; ``tap`` captures the residual stream after the
    given layer ids for the DSpark drafter (None = plain forward, the baseline path).

    Hybrid targets take the same loop with a per-layer mask choice, exactly as the single-row
    path does. Their recurrent layers get ``mask=None``: ``create_ssm_mask`` returns None for
    this cache type, and the batched rows all advance by the same ``T`` tokens per step, so
    there is nothing to mask — the differing prompt *lengths* already live in each row's own
    recurrent state, which is length-independent."""
    mm = target._inner.model
    if mask is None:
        mask = build_batch_mask(_row_offsets(caches), ids.shape[1])
    h = mm.embed_tokens(ids)
    tapset = set(tap or [])
    captured = []
    hybrid = getattr(target, "is_hybrid", False)
    for i, (layer, c) in enumerate(zip(mm.layers, caches)):
        if hybrid:
            h = layer(h, mask=(None if getattr(layer, "is_linear", False) else mask), cache=c)
        else:
            h = layer(h, mask, c)
        if i in tapset:
            captured.append(h)
    logits = target._head_mlxlm(mm.norm(h))
    fused = mx.concatenate(captured, axis=-1) if captured else None
    return logits, fused


def _prefill_rows(target, prompts_ids: list[list[int]], tap: list[int] | None = None):
    """Prefill each row's prompt individually (one-at-a-time bounds memory and keeps the code
    on the existing single-seq prefill), returning per-row (last_logits, cache, fused|None)."""
    rows = []
    for ids in prompts_ids:
        cache = target.make_cache()
        if tap is None:
            logits = _prefill_plain(target, ids, cache)
            fused = None
        else:
            from .generate import _prefill_tapped

            logits, fused = _prefill_tapped(target, ids, cache, tap)
        rows.append((logits, cache, fused))
    return rows


def _merge_row_caches(row_caches: list[list]) -> list:
    """Per-layer batched cache list from a list of per-row single-seq cache lists.

    Attention layers merge into a left-aligned :class:`BatchCache` (rows keep their own
    offsets). Recurrent layers merge by concatenating their state on the batch axis — no
    padding and no offsets, because that state is a fixed-size summary rather than a per-token
    buffer (see :func:`_recurrent_mergeable`). A hybrid model simply gets a mix.
    """
    n_layers = len(row_caches[0])
    out = []
    for l in range(n_layers):
        proto = row_caches[0][l]
        if _recurrent_mergeable(proto):
            merged = type(proto)(len(proto.state))
            for j in range(len(proto.state)):
                merged[j] = mx.contiguous(
                    mx.concatenate([c[l][j] for c in row_caches], axis=0))
            out.append(merged)
            continue
        pairs = [(c[l].keys[..., : c[l].offset, :], c[l].values[..., : c[l].offset, :])
                 for c in row_caches]
        out.append(BatchCache.from_rows(pairs))
    return out


def _row_offsets(caches: list) -> list[int]:
    """The per-row token counts, read off whichever layer actually tracks length. A hybrid's
    recurrent layers do not know the length at all, so the attention layers are the authority."""
    for c in caches:
        if isinstance(c, BatchCache):
            return c.offsets
    raise ValueError("no attention-layer cache in this model — cannot recover row offsets")


def _sample_rows(logits_last: mx.array, temperature: float, top_p: float, top_k: int) -> mx.array:
    """Next token per row from ``[B, V]`` logits — argmax (temp 0) or truncated sample."""
    if temperature > 0.0:
        from .sampling import sample_probs, truncate_probs

        probs = truncate_probs(mx.softmax(logits_last / temperature, axis=-1), top_p, top_k)
        return sample_probs(probs)
    return mx.argmax(logits_last, axis=-1)


# --------------------------------------------------------------------------- per-row bookkeeping


def _as_list(x, B):
    return list(x) if isinstance(x, (list, tuple)) else [x] * B


class _RowTracker:
    """Per-row state for a batched generation: streaming detok + stop strings + eos + per-row
    ``max_new_tokens`` + result assembly. One shared abstraction so the batched baseline and spec
    loops both get streaming / stop-sequences / independent per-row lengths (what a server needs)
    without duplicating the single-seq loop's :class:`~mlx_dspark.generate._Streamer` handling.

    ``first_tokens`` are the prefill-sampled tokens (counted as each row's token 1, and streamed).
    ``commit(b, toks)`` appends a round's committed tokens for row b; ``finished[b]`` flips on
    eos / length / stop-string. ``results(secs)`` returns one :class:`GenResult` per row."""

    def __init__(self, tokenizer, first_tokens, *, max_new_tokens, on_texts=None, stops=None):
        self.tok = tokenizer
        self.eos = eos_token_ids(tokenizer)
        B = len(first_tokens)
        self.B = B
        self.mnt = _as_list(max_new_tokens, B)
        ot = on_texts or [None] * B
        sp = stops or [None] * B
        self.streamers = [_Streamer(tokenizer, self.eos, ot[b], sp[b]) for b in range(B)]
        self.out_ids = [[int(first_tokens[b])] for b in range(B)]
        self.accept_lengths: list[list[int]] = [[] for _ in range(B)]
        self.finished = [False] * B
        for b in range(B):
            self.streamers[b].update(self.out_ids[b])
            self._check(b)

    def _check(self, b):
        ids = self.out_ids[b]
        if ids[-1] in self.eos or len(ids) >= self.mnt[b] or self.streamers[b].stopped:
            self.finished[b] = True

    def commit(self, b, tokens, accept_len=None):
        if self.finished[b]:
            return
        if accept_len is not None:
            self.accept_lengths[b].append(accept_len)
        for tok in tokens:
            self.out_ids[b].append(tok)
            if tok in self.eos:
                break
        self.streamers[b].update(self.out_ids[b])
        self._check(b)

    def all_done(self):
        return all(self.finished)

    def results(self, secs):
        out = []
        for b in range(self.B):
            self.streamers[b].flush()
            ids = self.out_ids[b]
            stopped = self.streamers[b].stopped
            reason = "stop" if (ids[-1] in self.eos or stopped) else "length"
            text = (self.streamers[b].text if stopped
                    else self.tok.decode([t for t in ids if t not in self.eos]))
            spec = bool(self.accept_lengths[b])
            al = self.accept_lengths[b] if spec else [1] * len(ids)
            out.append(GenResult(
                text=text, token_ids=ids, num_tokens=len(ids), num_rounds=len(al),
                accept_lengths=al, target_forwards=(len(al) + 1) if spec else len(ids),
                seconds=secs, finish_reason=reason,
            ))
        return out


# --------------------------------------------------------------------------- baseline (M1)


def batch_generate_baseline(
    target,
    tokenizer,
    prompts_ids: list[list[int]],
    *,
    max_new_tokens=128,                    # int, or a per-row list[int]
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    on_texts=None,                          # per-row on_text callbacks (streaming)
    stops=None,                             # per-row stop-string lists
    seed: int | None = None,
) -> list[GenResult]:
    """Static batched plain decode of B prompts (no drafter) — the batched baseline. Every row
    advances one token per step through one shared forward; a finished row keeps stepping (its
    output ignored) until all rows finish. Returns one :class:`GenResult` per row.

    **What is guaranteed is row isolation, not bit-identity with B=1.** Rows never influence
    each other: an all-identical batch produces exactly identical rows (measured, max spread
    0.0). But a batched matmul takes a different kernel path than a single-row one, so a row's
    logits differ from the same prompt run alone by ~1 bf16 ulp, and a near-tie can land the
    other way — measured on the shipped dense path too (Qwen3-4B-8bit, 2 of 4 rows diverge
    within 96 tokens). This is the same contract :func:`batch_spec_generate` documents, and it
    is a property of batched quantized matmul, not of this loop.

    **MoE targets amplify it**: top-k expert routing is a *discrete* function of the hidden
    state, so an ulp-level difference can send a token to a different expert. Batched-vs-single
    argmax flips measured ~4x more often on Qwen3.6-35B-A3B than on a dense target of the same
    quantization. Batching is a throughput knob; if you need a specific single-stream
    transcript reproduced exactly, run that request unbatched."""
    if seed is not None:
        mx.random.seed(seed)
    B = len(prompts_ids)
    t0 = time.time()

    rows = _prefill_rows(target, prompts_ids)
    caches = _merge_row_caches([c for _, c, _ in rows])
    first = _sample_rows(mx.stack([lg[0, -1] for lg, _, _ in rows]), temperature, top_p, top_k)
    mx.eval(first)
    trk = _RowTracker(tokenizer, [int(x) for x in first.tolist()],
                      max_new_tokens=max_new_tokens, on_texts=on_texts, stops=stops)
    cur = first.reshape(B, 1)

    while not trk.all_done():
        logits, _ = batched_forward(target, cur, caches)
        ynext = _sample_rows(logits[:, -1], temperature, top_p, top_k)
        mx.eval(ynext)
        toks = ynext.tolist()
        for b in range(B):
            trk.commit(b, [int(toks[b])])
        cur = ynext.reshape(B, 1)

    return trk.results(time.time() - t0)


# --------------------------------------------------------------------------- spec (M2, Stage A)


class _BatchCtx:
    """Batched context K/V for one drafter layer — duck-types the ``.k`` / ``.v`` that
    :meth:`DSparkAttention.attend` reads, so ``drafter.backbone`` runs batched unchanged."""
    __slots__ = ("k", "v")

    def __init__(self, k, v):
        self.k = k
        self.v = v


def _batched_ctx(ctx_rows: list[list]) -> tuple[list[_BatchCtx], list[int]]:
    """Pad the per-row drafter :class:`CtxCache`s into batched ``[B, n_kv, Lctx_max, D]`` buffers
    (one :class:`_BatchCtx` per layer) + the per-row context lengths. ``ctx_rows`` is a list over
    rows of the per-layer cache list; context length is identical across layers (update_context
    appends to every layer), so one length vector serves the block mask and the RoPE offset."""
    B = len(ctx_rows)
    n_layers = len(ctx_rows[0])
    lens = [ctx_rows[b][0].length for b in range(B)]
    Lmax = max(lens)
    out = []
    for l in range(n_layers):
        c0 = ctx_rows[0][l]
        n_kv, Dk, Dv = c0.k.shape[1], c0.k.shape[3], c0.v.shape[3]
        bk = mx.zeros((B, n_kv, Lmax, Dk), dtype=c0.k.dtype)
        bv = mx.zeros((B, n_kv, Lmax, Dv), dtype=c0.v.dtype)
        for b in range(B):
            cb = ctx_rows[b][l]
            bk[b : b + 1, :, : lens[b], :] = cb.k
            bv[b : b + 1, :, : lens[b], :] = cb.v
        out.append(_BatchCtx(bk, bv))
    return out, lens


def _ctx_block_mask(lens: list[int], k: int) -> mx.array:
    """Boolean ``[B, 1, k, Lmax+k]`` mask for the batched block attention: each block position
    attends its row's valid context (columns ``< lens[b]``) plus every block position (columns
    ``>= Lmax``, bidirectional) — hiding shorter rows' context padding. Matches the single-seq
    full-attention exactly per row."""
    B = len(lens)
    Lmax = max(lens)
    lens_a = mx.array(lens)[:, None, None, None]                 # [B,1,1,1]
    j = mx.arange(Lmax + k)[None, None, None, :]                 # [1,1,1,Lmax+k]
    m = (j >= Lmax) | (j < lens_a)                              # block region OR valid context
    return mx.broadcast_to(m, (B, 1, k, Lmax + k))


def _batched_sample_block(drafter, base_logits: mx.array, first_prev: list[int]) -> mx.array:
    """Batched greedy DSpark block sampling: ``base_logits [B, cap, V]``, ``first_prev [B]`` ->
    ``[B, cap]``. The Markov head is sequential over the cap positions (position i's bias depends
    on the token at i-1) but vectorized across rows. Returns TARGET-vocabulary ids — on a
    reduced-vocab head the argmax is a draft id and has to be mapped before it is emitted or
    fed back as ``prev`` (markov_w1 is indexed by target ids). Mirrors
    ``DSparkDrafter.sample_block``; keep the two in step."""
    _B, cap, _ = base_logits.shape
    if drafter.markov_head is None:
        return drafter.to_target_ids(mx.argmax(base_logits, axis=-1))
    prev = mx.array(first_prev)                                  # [B]
    toks = []
    for i in range(cap):
        step = base_logits[:, i, :] + drafter.markov_head.step_bias(prev)   # [B, V]
        nxt = drafter.to_target_ids(mx.argmax(step, axis=-1))   # [B]
        toks.append(nxt)
        prev = nxt
    return mx.stack(toks, axis=1)                               # [B, cap]


class SpecSlots:
    """Fixed-capacity slot-based continuous batched **DSpark** spec decoding (greedy) — the
    dynamic-admission engine (M4). ``capacity`` slots share fixed batched KV buffers; a request
    is :meth:`admit`-ted into a free slot (single-seq prefill → :meth:`BatchCache.set_row`),
    :meth:`step` runs one spec round over all active rows, and a finished row is *retired the
    moment it finishes* — its result returned immediately, its slot free for the next request.
    The batch dimension is never resized; instead retirement **compacts** the active rows into a
    contiguous prefix (:meth:`BatchCache.move_row`, a one-off row copy ≈ a bandwidth blip) and
    every forward runs at the *active* width — so a lone tail request verifies at serial width
    (no pad waste, and B_act=1 is the bit-exact single-stream numeric path).

    Per-row output is the target's greedy decoding under the usual batched-quantized contract
    (greedy-correct per row; bit-identical to single-seq only at width 1 — see NOTES). Drafting
    is Stage B (one batched backbone forward) or Stage A (per-row) via ``batched_drafter``."""

    def __init__(self, target, tokenizer, drafter, *, capacity: int,
                 max_draft_tokens: int = 2, batched_drafter: bool = True,
                 cap_controller=None):
        cfg = drafter.config
        self.target, self.tokenizer, self.drafter = target, tokenizer, drafter
        self.tap = list(cfg.target_layer_ids)
        self.k = int(cfg.block_size)
        self.mask_id = int(cfg.mask_token_id)
        self.cap = max(1, min(int(max_draft_tokens), self.k))
        self.ctrl = cap_controller     # optional CapController: per-batch-width cap + EWMA
        self.batched_drafter = batched_drafter
        self.capacity = int(capacity)
        self.n_active = 0
        self.caches: list[BatchCache] | None = None   # built from the first admission's shapes
        C = self.capacity
        self.ctx = [None] * C          # per-slot drafter CtxCache list
        self.n_cached = [0] * C        # per-slot target tokens cached (drafter ctx offset)
        self.pending = [0] * C         # per-slot next token to verify
        self.trk = [None] * C          # per-slot single-row _RowTracker
        self.meta = [None] * C         # caller's job handle, returned on retirement
        self.t0 = [0.0] * C

    @property
    def has_free_slot(self) -> bool:
        return self.n_active < self.capacity

    def admit(self, prompt_ids: list[int], *, max_new_tokens: int,
              on_text=None, stop=None, meta=None) -> None:
        """Prefill one request (single-seq, seeding its drafter ctx) into the next free slot."""
        if not self.has_free_slot:
            raise RuntimeError("SpecSlots.admit: no free slot")
        from .generate import _prefill_tapped

        cache = self.target.make_cache()
        ctx = self.drafter.make_ctx_cache()
        logits, _ = _prefill_tapped(self.target, list(prompt_ids), cache, self.tap,
                                    drafter=self.drafter, ctx_caches=ctx)
        first = int(mx.argmax(logits[0, -1]).item())
        if self.caches is None:
            self.caches = [BatchCache.empty(self.capacity, c.keys.shape[1], c.keys.shape[3],
                                            c.values.shape[3], c.keys.dtype) for c in cache]
        b = self.n_active
        for bc, c in zip(self.caches, cache):
            bc.set_row(b, c.keys[..., : c.offset, :], c.values[..., : c.offset, :], c.offset)
        self.ctx[b] = ctx
        self.n_cached[b] = len(prompt_ids)
        self.pending[b] = first
        self.trk[b] = _RowTracker(self.tokenizer, [first], max_new_tokens=[max_new_tokens],
                                  on_texts=[on_text], stops=[stop])
        self.meta[b] = meta
        self.t0[b] = time.time()
        self.n_active += 1

    def _retire(self, b: int):
        """Deliver slot b's result and compact the last active row into its place."""
        meta, res = self.meta[b], self.trk[b].results(time.time() - self.t0[b])[0]
        last = self.n_active - 1
        if b != last:
            for c in self.caches:
                c.move_row(last, b)
            for state in (self.ctx, self.n_cached, self.pending, self.trk, self.meta, self.t0):
                state[b] = state[last]
        for c in self.caches:
            c.offsets[last] = 0
        self.ctx[last], self.trk[last], self.meta[last] = None, None, None
        self.pending[last], self.n_cached[last], self.t0[last] = 0, 0, 0.0
        self.n_active = last
        return meta, res

    def _sweep_finished(self) -> list:
        out = []
        b = 0
        while b < self.n_active:
            if self.trk[b].finished[0]:
                out.append(self._retire(b))   # compaction refills slot b — recheck the same b
            else:
                b += 1
        return out

    def step(self) -> list:
        """One spec round over the active rows. Returns ``[(meta, GenResult), …]`` for every
        request that finished this round (already retired; their slots are free again)."""
        finished = self._sweep_finished()      # e.g. eos / token limit hit on the prefill token
        B = self.n_active
        if B == 0:
            return finished
        drafter, target = self.drafter, self.target
        k, mask_id = self.k, self.mask_id
        # cap for this round: calibrated per batch width when a controller is attached
        # (B multiplies into the qmm-knee arithmetic, so wider batches want shorter caps);
        # per-round cap changes are lossless — the cap only picks how many drafts get verified
        cap = self.cap if self.ctrl is None else max(
            1, min(self.ctrl.cap_for(B) if B > 1 else self.ctrl.cap, self.k))
        pend = self.pending[:B]
        for c in self.caches:
            c.rows = B                          # attention reads the active-width rope offsets

        # ---- 1. draft ----
        if self.batched_drafter:
            # Stage B: one batched backbone forward over [B, k, H]. The ragged part is the
            # per-row context — pad it to a rectangle + a mask that hides each row's padding,
            # and feed a per-row RoPE offset (rows sit at different context lengths).
            block_ids = mx.array([[pend[b]] + [mask_id] * (k - 1) for b in range(B)])   # [B,k]
            noise = drafter.embed(block_ids)                                    # [B, k, H]
            bctx, ctx_lens = _batched_ctx([self.ctx[b] for b in range(B)])
            mask = _ctx_block_mask(ctx_lens, k)
            block_hidden = drafter.backbone(noise, mx.array(ctx_lens), bctx, mask)  # [B, k, H]
            base_logits = drafter.compute_logits(drafter.head_slice(block_hidden, cap))  # [B,cap,V]
            drafts_arr = _batched_sample_block(drafter, base_logits, pend)      # [B, cap]
            drafts = [drafts_arr[b] for b in range(B)]
        else:
            # Stage A: run the drafter sequentially per row (batched verify only).
            drafts = []
            for b in range(B):
                block_ids = [pend[b]] + [mask_id] * (k - 1)
                noise = drafter.embed(mx.array([block_ids]))
                block_hidden = drafter.backbone(noise, self.n_cached[b], self.ctx[b])
                base_logits = drafter.compute_logits(drafter.head_slice(block_hidden, cap))[0]  # [cap,V]
                drafts.append(drafter.sample_block(base_logits, first_prev_token=pend[b]))

        # ---- 2. batched verify (the shared weight-read), at the active width ----
        verify_ids = mx.stack([mx.concatenate([mx.array([pend[b]], dtype=drafts[b].dtype),
                                               drafts[b]]) for b in range(B)])      # [B, cap+1]
        vmask = build_batch_mask(self.caches[0].offsets[:B], cap + 1)
        v_logits, v_fused = batched_forward(target, verify_ids, self.caches, self.tap,
                                            mask=vmask)

        # ---- 3. accept per row (row-wise cumprod of the argmax match) ----
        tt = mx.argmax(v_logits, axis=-1)                                           # [B, cap+1]
        match = (mx.stack(drafts) == tt[:, :cap]).astype(mx.int32)
        n_arr = mx.cumprod(match, axis=1).sum(axis=1)                               # [B]
        mx.eval(tt, n_arr, v_fused)
        n_list = n_arr.tolist()
        tt_list = tt.tolist()

        # ---- 4. commit + per-row trim + per-row drafter-ctx update ----
        trims = []
        for b in range(B):
            n = int(n_list[b])
            committed = [int(x) for x in drafts[b].tolist()[:n]] + [int(tt_list[b][n])]
            trims.append(cap - n)
            drafter.update_context(v_fused[b : b + 1, : n + 1, :],
                                   ctx_offset=self.n_cached[b], ctx_caches=self.ctx[b])
            self.n_cached[b] += n + 1
            self.trk[b].commit(0, committed, accept_len=len(committed))
            self.pending[b] = self.trk[b].out_ids[0][-1]
            if self.ctrl is not None:          # same drafter acceptance process as serial rounds
                self.ctrl.update(n, cap)
        for c in self.caches:                              # per-row rollback, every layer
            c.trim(trims)

        finished += self._sweep_finished()
        return finished


def batch_spec_generate(
    target,
    tokenizer,
    drafter,
    prompts_ids: list[list[int]],
    *,
    max_new_tokens=128,                    # int, or a per-row list[int]
    max_draft_tokens: int = 2,
    batched_drafter: bool = True,           # Stage B: batch the drafter too (False = Stage A)
    on_texts=None,                          # per-row on_text callbacks (streaming)
    stops=None,                             # per-row stop-string lists
    seed: int | None = None,
) -> list[GenResult]:
    """Static batched **DSpark** speculative decode (greedy): admit all B prompts into a
    :class:`SpecSlots` of capacity B and step until done. Rows that finish early retire
    immediately (their timing reflects their own finish; the remaining rows verify at the
    narrower active width). Per row the output is the target's greedy decoding — B=1 is
    bit-exact vs single-seq; at B>1 greedy-correct up to the target's batch-dependent
    quantized-matmul rounding (see NOTES). Returns one GenResult per row, in prompt order."""
    if seed is not None:
        mx.random.seed(seed)
    B = len(prompts_ids)
    if B == 0:
        return []
    mnt = _as_list(max_new_tokens, B)
    ot = on_texts or [None] * B
    sp = stops or [None] * B
    slots = SpecSlots(target, tokenizer, drafter, capacity=B,
                      max_draft_tokens=max_draft_tokens, batched_drafter=batched_drafter)
    for i, ids in enumerate(prompts_ids):
        slots.admit(ids, max_new_tokens=mnt[i], on_text=ot[i], stop=sp[i], meta=i)
    results: list[GenResult | None] = [None] * B
    while slots.n_active:
        for meta, res in slots.step():
            results[meta] = res
    return results
