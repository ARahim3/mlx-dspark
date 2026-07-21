"""Prefix KV caching for the OpenAI server — skip re-prefilling a shared conversation prefix.

A multi-turn request's prompt is (almost) the previous prompt + the assistant's reply + the new
user turn, so the bulk of it was already computed last turn. This keeps the target KV cache (and,
for DSpark, the drafter's context cache) from recent conversations and, on the next request,
reuses the entry with the longest common prefix: trim the caches back to it and prefill only the
new suffix.

Scope (chosen for correctness):
  * **Dense targets, plus sliding-window targets while under the window.** Reuse trims the KV
    cache back to an arbitrary earlier position. That's always exact for a plain ``KVCache``
    (Qwen3). A ``RotatingKVCache`` (Gemma-4's sliding-window layers) is linear — identical to a
    plain cache — until it first wraps at ``max_size``; mlx-lm's own ``is_trimmable()`` encodes
    exactly this. So rotating caches are reused too, and an entry is refused at store time the
    moment any layer has wrapped (typical chats stay well under the window).
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


def _cache_ram_bytes(cache) -> int:
    total = 0
    for c in cache:
        st = getattr(c, "state", None)
        if isinstance(st, (list, tuple)):
            for a in st:
                total += getattr(a, "nbytes", 0)
    return total


class _Slot:
    """One cached conversation: its token record + the caches holding that prefix's KV."""

    __slots__ = ("tokens", "cache", "ctx", "spilled", "sid")

    def __init__(self, tokens, cache, ctx, sid: int):
        self.tokens: list[int] = tokens
        self.cache = cache            # None while spilled to disk
        self.ctx = ctx
        self.spilled = False
        self.sid = sid                # unique id -> distinct spill filenames


class PrefixCache:
    def __init__(self, make_cache, make_ctx=None, *, min_reuse: int = 16,
                 l2_dir: str | None = None, max_ram_bytes: int = 0, slots: int = 2):
        self.make_cache = make_cache          # () -> list[target layer cache]
        self.make_ctx = make_ctx              # () -> list[CtxCache] | None (None for baseline)
        self.min_reuse = max(1, min_reuse)
        self.l2_dir = l2_dir
        self.max_ram_bytes = max_ram_bytes    # 0 = never spill (pure in-memory)
        self.max_slots = max(1, slots)
        self.hits = 0
        self.reused_tokens = 0
        self._slots: list[_Slot] = []         # most-recently-used first
        self._next_sid = 0
        if l2_dir:
            os.makedirs(l2_dir, exist_ok=True)

    # -- public API (engine calls these under its generation lock) --
    def acquire(self, prompt_ids: list[int]):
        """Return ``(cache, ctx, reuse_len)`` for this request — the best-matching slot's
        caches trimmed to the shared prefix, or fresh ones. The chosen slot is checked out
        (removed) until ``store()`` re-validates it; other slots are untouched."""
        best, best_len = None, 0
        for slot in self._slots:
            lcp = _lcp(slot.tokens, prompt_ids)
            reuse = max(0, min(lcp, len(slot.tokens), len(prompt_ids) - 1))
            if reuse > best_len:
                best, best_len = slot, reuse
        if best is not None and best_len >= self.min_reuse:
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

    def store(self, cache, ctx, prompt_ids: list[int], token_ids: list[int]) -> None:
        # the cache holds KV for the prompt + every generated token EXCEPT the last (that one is
        # the pending token, not yet fed through the target) — see the generate loops.
        if not _storable(cache):              # e.g. a RotatingKVCache wrapped mid-generation
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
        self._clear_spill_files()

    def info(self) -> dict:
        newest = self._slots[0] if self._slots else None
        return {"enabled": True,
                "cached_tokens": len(newest.tokens) if newest else 0,
                "slots": [{"tokens": len(s.tokens), "spilled": s.spilled} for s in self._slots],
                "hits": self.hits, "reused_tokens": self.reused_tokens,
                "l2": bool(self.l2_dir)}

    # -- internals --
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
            try:
                os.remove(p)
            except OSError:
                pass

    def _maybe_spill(self) -> None:
        if self.max_ram_bytes <= 0 or not self.l2_dir:
            return
        # spill least-recent in-RAM slots until the total fits the budget (never the newest —
        # it's the one most likely reused next turn, unless it alone exceeds the budget)
        def total() -> int:
            return sum(_cache_ram_bytes(s.cache) for s in self._slots if s.cache is not None)

        for slot in reversed(self._slots):
            if total() <= self.max_ram_bytes:
                return
            if slot.cache is not None:
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
            try:
                os.remove(p)
            except OSError:
                pass
