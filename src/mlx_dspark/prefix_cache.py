"""Prefix KV caching for the OpenAI server — skip re-prefilling a shared conversation prefix.

A multi-turn request's prompt is (almost) the previous prompt + the assistant's reply + the new
user turn, so the bulk of it was already computed last turn. This keeps the target KV cache (and,
for DSpark, the drafter's context cache) from recent conversations and, on the next request,
reuses the entry with the longest common prefix: trim the caches back to it and prefill only the
new suffix.

Two reuse modes, because "can this cache be rolled back?" and "can this cache be reused?" are
different questions:

  * **trim** (default) — trim the stored caches back to an arbitrary earlier position and
    prefill the rest. Exact for a plain ``KVCache`` (Qwen3). A ``RotatingKVCache`` (Gemma-4's
    sliding-window layers) is linear — identical to a plain cache — until it first wraps at
    ``max_size``; mlx-lm's own ``is_trimmable()`` encodes exactly this.
  * **checkpoint** — snapshot the caches at a **boundary** and reuse them only at exactly
    that length. This needs no rollback at all, so it works for the caches trim mode has to
    refuse: a hybrid target's linear-attention state (recurrent, no trim) and a wrapped
    rotating window. It is exact for the same reason ``Target.rollback`` is: the state after
    k tokens is a function of the first k tokens only, so a snapshot taken there is precisely
    what a cold prefill of that prefix would produce.

    Reuse is all-or-nothing at a snapshot position, so *where* the snapshots sit is the whole
    game. Three kinds of position, all chosen by the server (see ``Engine``):

      - the **stable prompt boundary** — the prompt boundary minus the few trailing
        generation-prompt tokens that do not survive into the next turn's re-rendered
        transcript (Qwen3.6-class templates prefill a ``<think>`` opener that the completed
        turn renders differently; snapshotting at the full boundary made every multi-turn
        request miss by 2–4 tokens). Snapshotting at least one token *below* the prompt
        boundary also makes a byte-identical repeat (regenerate/retry) a hit: the loop
        re-forwards the unstable tail and gets its logits, which a snapshot at the exact
        boundary cannot provide.
      - **rungs** — interior positions (chunk-spaced during prefill, plus the boundaries of
        superseded slots) at which only the *non-trimmable* layer states are kept. The
        trimmable layers (a hybrid's attention KV, the drafter ctx) reuse the slot's full
        snapshot trimmed back to the rung — KV rows are position-local, so that is exact.
        This is what makes reuse *partial* for hybrid/recurrent targets: a request that
        diverges from a cached conversation mid-way (new session on the same system prompt,
        compacted history) reuses the longest rung under the divergence instead of missing
        outright.
      - an **anchor** — when a request misses (or lands on a shallow rung) but shared a long
        prefix with some slot, ``acquire`` remembers that LCP and the server snapshots a rung
        there during this request's prefill, so the *next* request from that fan-out pattern
        hits at the exact divergence point.

Mode is not a user choice: checkpoint is forced on for targets whose caches are not trimmable
at all, and switched on adaptively for a rotating-window target the first time a store is
refused because the window wrapped (so gemma-4 pays the snapshot only once it needs it).
  * **dspark + lookup + baseline modes.** DFlash's drafter keeps its own cache that can't roll
    back, so it isn't reused here.
  * **A small LRU of conversations** (default 2 slots) — so an agent process and a chat window
    hitting the same server don't evict each other every turn. Still not a multi-tenant KV pool.

Losslessness: reuse is lossless to the same standard as the rest of mlx-dspark — the output is a
valid decoding of the target, differing from a cold run only at logit-margin≈0 ties (fp
nondeterminism between chunked and single-pass prefill). An in-flight entry is only re-validated
by ``store()``, so a generation error can never leave a cache desynced from the token record it
claims to represent.

Optional **L2 SSD spill**: when the slots' total RAM exceeds a byte budget, least-recent slots
are serialized to a directory (target KV via mlx-lm's ``save_prompt_cache``, drafter ctx as
safetensors) and dropped from RAM, reloaded on their next reuse.
"""

from __future__ import annotations

import contextlib
import glob
import json
import os

import mlx.core as mx


def _lcp(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _trim_target(cache, to_len: int) -> None:
    for c in cache:
        off = getattr(c, "offset", None)
        if off is None or not hasattr(c, "trim"):
            continue
        n = int(off) - to_len
        if n > 0:
            c.trim(n)


def target_cache_reusable(cache) -> bool:
    """True if every layer cache can, at least while in its linear regime, be rolled back to
    any earlier position: plain ``KVCache`` and ``QuantizedKVCache`` always (both trim by pure
    offset arithmetic); ``RotatingKVCache`` (sliding-window) only counts if it exposes the
    mlx-lm rotation machinery (``max_size``/``is_trimmable``) — its wrap is then caught
    per-entry at store time. Anything else (exotic) is rejected."""
    def ok(c) -> bool:
        name = type(c).__name__
        if not (hasattr(c, "trim") and hasattr(c, "offset")):
            return False
        if name in ("KVCache", "QuantizedKVCache"):
            return True
        if name == "RotatingKVCache":
            return hasattr(c, "max_size") and hasattr(c, "is_trimmable")
        return False

    return all(ok(c) for c in cache)


def _storable(cache) -> bool:
    """A finished generation's cache may only be stored if every layer can still be trimmed
    to an arbitrary earlier position — i.e. no RotatingKVCache has wrapped its window."""
    for c in cache:
        fn = getattr(c, "is_trimmable", None)
        if callable(fn) and not fn():
            return False
    return True


def _layer_trimmable(c) -> bool:
    """Can THIS layer cache be rolled back to an arbitrary earlier position right now?
    mlx-lm's ``is_trimmable()`` where present (ArraysCache/recurrent -> False, wrapped
    RotatingKVCache -> False); otherwise the same structural test ``_trim_target`` uses."""
    fn = getattr(c, "is_trimmable", None)
    if callable(fn):
        return bool(fn())
    return hasattr(c, "trim") and hasattr(c, "offset")


def _cache_ram_bytes(cache) -> int:
    total = 0
    for c in cache:
        st = getattr(c, "state", None)
        if isinstance(st, (list, tuple)):
            for a in st:
                total += getattr(a, "nbytes", 0)
    return total


# -- checkpoint snapshots ---------------------------------------------------------------
#
# A snapshot is (state, meta_state) per layer cache — mlx-lm's own serialization contract,
# the same pair save_prompt_cache/load_prompt_cache round-trips — held as COPIES. The copies
# are load-bearing: a wrapped RotatingKVCache writes its ring in place, so handing out the
# stored arrays would let one request corrupt the snapshot for every later one.


def _copy_tree(v):
    if isinstance(v, mx.array):
        return mx.array(v)
    if isinstance(v, (list, tuple)):
        return type(v)(_copy_tree(x) for x in v)
    return v


def _eval_tree(v, out=None):
    """Materialize every mx.array in a copied tree now — a bounded memcpy at capture time
    beats a deferred one that would otherwise land inside a later request's TTFT (and keep
    the source buffers pinned in the meantime)."""
    arrs = []
    def walk(x):
        if isinstance(x, mx.array):
            arrs.append(x)
        elif isinstance(x, (list, tuple)):
            for y in x:
                walk(y)
    walk(v)
    if arrs:
        mx.eval(arrs)
    return v


def _snapshot(cache, ctx):
    snap = [(_copy_tree(c.state), c.meta_state) for c in cache]
    snap_ctx = None if ctx is None else [(_copy_tree(c.k), _copy_tree(c.v)) for c in ctx]
    _eval_tree([s for s, _ in snap])
    if snap_ctx is not None:
        _eval_tree(snap_ctx)
    return snap, snap_ctx


def _rung_capture(cache) -> list[tuple[int, object, object]]:
    """Copies of the NON-trimmable layer states only — everything a later partial restore
    can't get by trimming the slot's full snapshot. ``[(layer_idx, state, meta_state)]``."""
    cap = [(i, _copy_tree(c.state), c.meta_state)
           for i, c in enumerate(cache) if not _layer_trimmable(c)]
    _eval_tree([s for _, s, _ in cap])
    return cap


def _tree_bytes(v) -> int:
    if isinstance(v, mx.array):
        return v.nbytes
    if isinstance(v, (list, tuple)):
        return sum(_tree_bytes(x) for x in v)
    return 0


def _rung_ram_bytes(rungs: dict) -> int:
    return sum(_tree_bytes(state) for cap in rungs.values() for _, state, _ in cap)


def _restore(make_cache, make_ctx, snapshot):
    snap, snap_ctx = snapshot
    cache = make_cache()
    for c, (state, meta) in zip(cache, snap):
        c.state = _copy_tree(state)     # KVCache derives offset from this; Rotating doesn't
        if meta:                        # ...so a rotating cache's offset/_idx ride here
            c.meta_state = meta
    ctx = None
    if snap_ctx is not None and make_ctx is not None:
        ctx = make_ctx()
        for c, (k, v) in zip(ctx, snap_ctx):
            c.k, c.v = _copy_tree(k), _copy_tree(v)
    return cache, ctx


def _snapshot_ram_bytes(snapshot) -> int:
    snap, snap_ctx = snapshot
    total = sum(getattr(v, "nbytes", 0)
                for state, _ in snap
                for v in (state if isinstance(state, (list, tuple)) else [state]))
    if snap_ctx:
        total += sum(getattr(a, "nbytes", 0) for pair in snap_ctx for a in pair)
    return total


class _Slot:
    """One cached conversation: its token record + the caches holding that prefix's KV.

    ``snapshot`` is set instead of ``cache``/``ctx`` for a checkpoint slot — immutable
    state reusable only at exactly ``len(tokens)``, and never checked out (so a failed
    generation cannot invalidate it). A checkpoint slot may also carry ``rungs``: interior
    positions where the non-trimmable layer states were captured, enabling partial reuse
    below the boundary (trimmable layers come from ``snapshot`` trimmed back)."""

    __slots__ = ("cache", "ctx", "nontrim", "rungs", "sid", "snapshot", "spilled", "tokens")

    def __init__(self, tokens, cache, ctx, sid: int, snapshot=None, rungs=None,
                 nontrim=None):
        self.tokens: list[int] = tokens
        self.cache = cache            # None while spilled to disk, or for a checkpoint slot
        self.ctx = ctx
        self.spilled = False
        self.sid = sid                # unique id -> distinct spill filenames
        self.snapshot = snapshot      # not None => checkpoint slot
        self.rungs: dict = rungs or {}   # pos -> [(layer_idx, state, meta_state)]
        self.nontrim: list[int] = nontrim or []   # layer idxs non-trimmable at snapshot time


class PrefixCache:
    def __init__(self, make_cache, make_ctx=None, *, min_reuse: int = 16,
                 l2_dir: str | None = None, max_ram_bytes: int = 0, slots: int = 2,
                 checkpoint: bool = False, max_rungs: int = 8):
        self.make_cache = make_cache          # () -> list[target layer cache]
        self.make_ctx = make_ctx              # () -> list[CtxCache] | None (None for baseline)
        self.min_reuse = max(1, min_reuse)
        self.checkpoint_mode = bool(checkpoint)   # see the module docstring; can latch on
        self.l2_dir = l2_dir
        self.max_ram_bytes = max_ram_bytes    # 0 = never spill (pure in-memory)
        self.max_slots = max(1, slots)
        self.max_rungs = max(1, max_rungs)    # per-slot ladder cap (recurrent state is ~MBs
        #                                       to ~100s of MB per rung on big hybrids)
        self.hits = 0
        self.partial_hits = 0                 # subset of hits that restored at a rung
        self.reused_tokens = 0
        self._slots: list[_Slot] = []         # most-recently-used first
        self._next_sid = 0
        self._pending_rungs: dict = {}        # pos -> capture, staged during this prefill
        self._anchor = 0                      # LCP the server should snapshot at (see acquire)
        if l2_dir:
            os.makedirs(l2_dir, exist_ok=True)

    # -- public API (engine calls these under its generation lock) --
    def acquire(self, prompt_ids: list[int]):
        """Return ``(cache, ctx, reuse_len)`` for this request — the best-matching slot's
        caches trimmed to the shared prefix, or fresh ones. A trim slot is checked out
        (removed) until ``store()`` re-validates it; checkpoint slots are never checked out
        (every restore is a copy). Also stages the anchor suggestion (:meth:`take_anchor`)
        and drops any rungs a failed previous generation left pending."""
        self._pending_rungs = {}
        self._anchor = 0
        best, best_len, best_rung, best_lcp = None, 0, None, 0
        for slot in self._slots:
            lcp = _lcp(slot.tokens, prompt_ids)
            best_lcp = max(best_lcp, lcp)
            rung = None
            if slot.snapshot is not None:
                if lcp >= len(slot.tokens) and len(prompt_ids) > len(slot.tokens):
                    reuse = len(slot.tokens)     # the boundary itself
                else:
                    # partial: deepest rung under the divergence point
                    rung = max((r for r in slot.rungs
                                if self.min_reuse <= r <= lcp and r < len(prompt_ids)),
                               default=0) or None
                    reuse = rung or 0
            else:
                reuse = max(0, min(lcp, len(slot.tokens), len(prompt_ids) - 1))
            if reuse > best_len:
                best, best_len, best_rung = slot, reuse, rung
        if (self.checkpoint_mode and best_lcp >= self.min_reuse
                and best_lcp - best_len >= self.min_reuse and best_lcp < len(prompt_ids)):
            # a future request from the same fan-out will diverge here too — worth a rung
            self._anchor = best_lcp
        if best is not None and best_len >= self.min_reuse:
            if best.snapshot is not None:
                if best_rung is None:
                    cache, ctx = _restore(self.make_cache, self.make_ctx, best.snapshot)
                else:
                    cache, ctx = self._restore_rung(best, best_rung)
                    if cache is None:            # a non-captured layer can't roll back
                        return (self.make_cache(),
                                (self.make_ctx() if self.make_ctx else None), 0)
                    self.partial_hits += 1
                self._slots.remove(best)
                self._slots.insert(0, best)          # LRU touch
                self.hits += 1
                self.reused_tokens += best_len
                return cache, ctx, best_len
            self._slots.remove(best)          # in flight; store() re-validates
            cache, ctx = self._materialize(best)
            if cache is not None:
                _trim_target(cache, best_len)
                if ctx is not None:
                    for c in ctx:
                        c.trim_to(best_len)
                self.hits += 1
                self.reused_tokens += best_len
                return cache, ctx, best_len
        return self.make_cache(), (self.make_ctx() if self.make_ctx else None), 0

    def wants_checkpoint(self) -> bool:
        """True when the generate loops should call :meth:`checkpoint` after prefill."""
        return self.checkpoint_mode

    def take_anchor(self) -> int:
        """Position the server should add to this request's prefill marks (0 = none): the
        LCP of the request :meth:`acquire` just served against the best-matching slot, when
        that request missed (or only reached a shallow rung). One-shot."""
        a, self._anchor = self._anchor, 0
        return a

    def rung(self, cache, pos: int) -> None:
        """Stage an interior snapshot at ``pos`` (mid-prefill, caches hold exactly the first
        ``pos`` tokens). Only non-trimmable layer states are captured — the rest of a partial
        restore comes from the owning slot's boundary snapshot, trimmed. Attached to the slot
        :meth:`checkpoint` creates at the end of this prefill."""
        if not self.checkpoint_mode or pos < self.min_reuse:
            return
        self._pending_rungs[pos] = _rung_capture(cache)

    def checkpoint(self, cache, ctx, n_prompt: int, prompt_ids: list[int]) -> None:
        """Snapshot the caches at ``n_prompt`` — the ``on_prefill`` hook at the stable
        boundary. Only the first ``n_prompt`` tokens are in the caches at this point, by
        construction: the loops split prefill at the server's marks. Absorbs this prefill's
        staged rungs, and collapses any older slot whose boundary this prompt extends into a
        rung of the new slot (one slot per conversation, with a ladder of past boundaries)."""
        pending, self._pending_rungs = self._pending_rungs, {}
        if not self.checkpoint_mode or n_prompt < self.min_reuse:
            return
        tokens = list(prompt_ids[:n_prompt])
        rungs = {r: cap for r, cap in pending.items() if self.min_reuse <= r < n_prompt}
        for slot in self._slots:                 # already have this exact boundary
            if slot.snapshot is not None and slot.tokens == tokens:
                slot.rungs.update(rungs)
                self._cap_rungs(slot)
                return
        nontrim = [i for i, c in enumerate(cache) if not _layer_trimmable(c)]
        for old in [s for s in self._slots
                    if s.snapshot is not None and len(s.tokens) < n_prompt
                    and tokens[:len(s.tokens)] == s.tokens]:
            # this prompt extends `old`'s boundary: keep its ladder, demote its boundary to
            # a rung (its non-trimmable states are already in its snapshot), free the rest
            for r, cap in old.rungs.items():
                rungs.setdefault(r, cap)
            b = len(old.tokens)
            if b >= self.min_reuse and b not in rungs:
                snap, _ = old.snapshot
                rungs[b] = [(i, snap[i][0], snap[i][1]) for i in old.nontrim]
            self._slots.remove(old)
        slot = _Slot(tokens, None, None, self._next_sid, snapshot=_snapshot(cache, ctx),
                     rungs=rungs, nontrim=nontrim)
        self._next_sid += 1
        self._slots.insert(0, slot)
        self._cap_rungs(slot)
        while len(self._slots) > self.max_slots:
            self._evict(self._slots.pop())

    def store(self, cache, ctx, prompt_ids: list[int], token_ids: list[int]) -> None:
        # the cache holds KV for the prompt + every generated token EXCEPT the last (that one is
        # the pending token, not yet fed through the target) — see the generate loops.
        if not _storable(cache):              # e.g. a RotatingKVCache wrapped mid-generation
            # This target's caches can no longer be rolled back to an arbitrary position,
            # so trim-mode reuse is over for it — latch checkpoint mode on and let the next
            # request snapshot at its prompt boundary instead. (Before this, gemma-4 simply
            # lost prefix caching for the rest of the process the first time it wrapped —
            # which at Claude-Code prompt sizes is immediately.)
            self.checkpoint_mode = True
            return
        tokens = list(prompt_ids) + list(token_ids[:-1])
        # ...except that a speculative round commits a whole block at once, and the loops stop
        # at an eos landing MID-block: the tokens after it are dropped from token_ids, but the
        # verify forward already wrote their KV rows (and their drafter ctx). So normalize the
        # caches down to the token record before storing, or the slot would claim a prefix
        # shorter than the KV it actually holds. Reuse trims by absolute offset, so the excess
        # was harmless in practice — but it wasted KV and left this class's central invariant
        # (caches hold exactly `slot.tokens`) only accidentally true, which is not something a
        # future reader should have to re-derive. Baseline can't reach this: it commits one
        # token per step, so eos is always the last token appended.
        _trim_target(cache, len(tokens))
        if ctx is not None:
            for c in ctx:
                c.trim_to(len(tokens))
        slot = _Slot(tokens, cache, ctx, self._next_sid)
        self._next_sid += 1
        self._slots.insert(0, slot)
        while len(self._slots) > self.max_slots:
            self._evict(self._slots.pop())
        self._maybe_spill()

    def reset(self) -> None:
        self._slots = []
        self._pending_rungs = {}
        self._anchor = 0
        self._clear_spill_files()

    def info(self) -> dict:
        newest = self._slots[0] if self._slots else None
        return {"enabled": True,
                "mode": "checkpoint" if self.checkpoint_mode else "trim",
                "cached_tokens": len(newest.tokens) if newest else 0,
                "slots": [{"tokens": len(s.tokens), "spilled": s.spilled,
                           "kind": "checkpoint" if s.snapshot is not None else "trim",
                           **({"rungs": sorted(s.rungs)} if s.rungs else {})}
                          for s in self._slots],
                "hits": self.hits, "partial_hits": self.partial_hits,
                "reused_tokens": self.reused_tokens,
                "l2": bool(self.l2_dir)}

    # -- internals --
    def _restore_rung(self, slot: _Slot, r: int):
        """Fresh caches holding exactly the first ``r`` tokens of ``slot``: captured layers
        from the rung, everything else from the boundary snapshot trimmed back to ``r``
        (exact — KV rows are position-local). Returns ``(None, None)`` if a non-captured
        layer can't be trimmed (e.g. a rotating window that wrapped after the rung)."""
        cache, ctx = _restore(self.make_cache, self.make_ctx, slot.snapshot)
        captured = {i: (st, meta) for i, st, meta in slot.rungs[r]}
        for i, c in enumerate(cache):
            if i in captured:
                st, meta = captured[i]
                c.state = _copy_tree(st)
                if meta:
                    c.meta_state = meta
            elif not _layer_trimmable(c):
                return None, None
        _trim_target(cache, r)
        if ctx is not None:
            for c in ctx:
                c.trim_to(r)
        return cache, ctx

    def _cap_rungs(self, slot: _Slot) -> None:
        """Bound the ladder's RAM: drop the rung with the smallest gap to its lower
        neighbor until at most ``max_rungs`` remain (keeps the ladder roughly spread —
        the deep rungs near a shared system prompt are the reusable ones)."""
        while len(slot.rungs) > self.max_rungs:
            pos = sorted(slot.rungs)
            gaps = [(pos[i] - (pos[i - 1] if i else 0), pos[i]) for i in range(len(pos))]
            slot.rungs.pop(min(gaps)[1])

    def _materialize(self, slot: _Slot):
        if slot.cache is not None:
            return slot.cache, slot.ctx
        if slot.spilled and self.l2_dir:
            return self._load_spill(slot)
        return None, None

    def _evict(self, slot: _Slot) -> None:
        if not self.l2_dir:
            return
        for p in self._spill_paths(slot.sid):
            with contextlib.suppress(OSError):
                os.remove(p)

    def _maybe_spill(self) -> None:
        if self.max_ram_bytes <= 0 or not self.l2_dir:
            return
        # spill least-recent in-RAM slots until the total fits the budget (never the newest —
        # it's the one most likely reused next turn, unless it alone exceeds the budget).
        # Checkpoint slots count toward the budget but are not spillable yet (their state
        # isn't in live cache objects, which is what save_prompt_cache wants) — so they are
        # the floor the trim slots get spilled around. Worth doing later: an SSD-persisted
        # checkpoint would survive restarts, which for a fixed ~20k agent system prompt is
        # the single most reusable thing this cache ever holds.
        def total() -> int:
            return sum((_snapshot_ram_bytes(s.snapshot) + _rung_ram_bytes(s.rungs))
                       if s.snapshot is not None else _cache_ram_bytes(s.cache)
                       for s in self._slots if s.cache is not None or s.snapshot is not None)

        for slot in reversed(self._slots):
            if total() <= self.max_ram_bytes:
                return
            if slot.snapshot is None and slot.cache is not None:
                self._save_spill(slot)
                slot.cache = slot.ctx = None
                slot.spilled = True

    # -- L2 SSD spill --
    def _spill_paths(self, sid: int) -> tuple[str, str]:
        return (os.path.join(self.l2_dir, f"target_cache_{sid}.safetensors"),
                os.path.join(self.l2_dir, f"ctx_cache_{sid}.safetensors"))

    def _save_spill(self, slot: _Slot) -> None:
        from mlx_lm.models.cache import save_prompt_cache

        tpath, cpath = self._spill_paths(slot.sid)
        save_prompt_cache(tpath, slot.cache, metadata={"tokens": json.dumps(slot.tokens)})
        if slot.ctx is not None:
            arrs = {}
            for i, c in enumerate(slot.ctx):
                if c.k is not None:
                    arrs[f"{i}.k"] = c.k
                    arrs[f"{i}.v"] = c.v
            mx.save_safetensors(cpath, arrs)

    def _load_spill(self, slot: _Slot):
        from mlx_lm.models.cache import load_prompt_cache

        tpath, cpath = self._spill_paths(slot.sid)
        cache, _meta = load_prompt_cache(tpath, return_metadata=True)
        ctx = None
        if self.make_ctx is not None and os.path.exists(cpath):
            ctx = self.make_ctx()
            arrs = mx.load(cpath)
            for i, c in enumerate(ctx):
                if f"{i}.k" in arrs:
                    c.k, c.v = arrs[f"{i}.k"], arrs[f"{i}.v"]
        slot.cache, slot.ctx = cache, ctx     # back in RAM
        slot.spilled = False
        return cache, ctx

    def _clear_spill_files(self) -> None:
        if not self.l2_dir:
            return
        for p in glob.glob(os.path.join(self.l2_dir, "*_cache_*.safetensors")):
            with contextlib.suppress(OSError):
                os.remove(p)
