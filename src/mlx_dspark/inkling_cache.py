"""KV caches for Inkling 2's attention that the stock SDPA reads IN PLACE.

mlx-vlm's Inkling 2 attention ends every layer visit with two concatenates that copy the whole K
and the whole V: the attention sink is a zero key / value row appended to them, and a wrapped
``RotatingKVCache`` has to be put back in temporal order first (its relative-position logits index
backward distance). A multi-row verify copies the ring twice more (``_update_concat``) and, on a
partial accept, its rollback restores the ring and replays the kept rows — another copy. At a full
1024-key window that is 4 MB per sliding visit and copy, 40 visits a token; on the 8 global visits
it grows with the context (270 MB a token at 8k).

Both caches here keep their rows in TEMPORAL ORDER in one buffer and keep the sink's zero row
INSIDE it, right after the newest key. ``mx.fast.scaled_dot_product_attention`` takes a strided
view at decode / verify widths (its vector kernels address K and V through their strides), so the
attention gets ``buffer[first : last + 2]`` — the same keys in the same order as stock, the same
kernel, bit-identical output — and nothing is copied (measured: 48 calls at a full window 3.8 ->
2.3 ms, at 8k keys 21.9 -> 9.4).

``SinkKVCache`` — mlx-vlm's ``KVCache`` (the global layers) plus the spare zero row.

``WindowKVCache`` — replaces ``RotatingKVCache`` (the sliding layers): a linear window instead of
a ring. Rows are appended in place; when the buffer is full, the rows still visible move to the
front of a fresh one (one copy per ``step`` tokens instead of several per token). Rolling back a
verify's rejected rows is a counter (``trim``), so mlx-vlm's speculative transaction handles it as
an ordinary append-only cache: no ring snapshot, no replay.

What a call sees is exactly what the stock caches show it — the last ``max_size - 1`` committed
keys, then the call's own rows — on the fused attention path (``sink_views``) and on the stock one
(``update_and_fetch``, which returns the same rows without the sink; prefill chunks take it).
``inkling_fused.selftest`` runs both against the stock caches through fill, wrap and rollback.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


def _zeros(like: mx.array, rows: int) -> mx.array:
    b, h, _, d = like.shape
    return mx.zeros((b, h, rows, d), like.dtype)


@lru_cache(maxsize=1)
def classes():
    """``(SinkKVCache, WindowKVCache)``, built on first use (mlx-vlm is an optional dependency)."""
    from mlx_vlm.models import cache as vlm_cache

    KVCache, base = vlm_cache.KVCache, vlm_cache._BaseCache

    class SinkKVCache(KVCache):
        """``KVCache`` whose buffer holds a zero row right after the valid ones (the attention
        sink), so ``sink_views`` is a view. ``_dirty``: rows below it may hold data — a verify's
        rejected rows stay in the buffer after ``trim`` — so the sink row is rewritten only when
        it falls inside them."""

        def __init__(self):
            super().__init__()
            self._dirty = 0
            self._buf = None            # the keys object this cache built (see _capacity)

        def _capacity(self) -> int:
            if self.keys is None:
                return 0
            if self.keys is not self._buf:      # assigned from outside: trust nothing past offset
                self._buf, self._dirty = self.keys, int(self.keys.shape[2])
            return int(self.keys.shape[2])

        def update_and_fetch(self, keys, values):
            prev, s = self.offset, int(keys.shape[2])
            end = prev + s
            if end + 1 > self._capacity():
                # [valid rows ‖ new rows ‖ zeros] as one fresh buffer, rounded up to `step`
                pad = -(end + 1) % self.step + 1
                old = self.keys is not None and prev > 0
                self.keys = mx.concatenate(
                    ([self.keys[..., :prev, :]] if old else []) + [keys, _zeros(keys, pad)], axis=2)
                self.values = mx.concatenate(
                    ([self.values[..., :prev, :]] if old else []) + [values, _zeros(values, pad)],
                    axis=2)
                self._buf, self._dirty = self.keys, end
            else:
                if end < self._dirty:           # the sink slot holds a stale row: rewrite it too
                    keys = mx.concatenate([keys, _zeros(keys, 1)], axis=2)
                    values = mx.concatenate([values, _zeros(values, 1)], axis=2)
                self.keys[..., prev: prev + keys.shape[2], :] = keys
                self.values[..., prev: prev + values.shape[2], :] = values
                self._dirty = max(self._dirty, end)
            self.offset = end
            return self.keys[..., :end, :], self.values[..., :end, :]

        def sink_views(self):
            """``(K, V)`` of the last ``update_and_fetch`` with the sink's zero row appended."""
            end = self.offset + 1
            return self.keys[..., :end, :], self.values[..., :end, :]

    class WindowKVCache(base):
        """A sliding window of the last ``max_size`` keys, stored in temporal order.

        Valid rows are ``[_start, _end)`` of the buffer; a call of ``s`` rows sees the last
        ``max_size - 1`` of them plus its own (what ``RotatingKVCache`` shows a decode step and a
        multi-row update alike). ``offset`` counts the tokens seen, as on the stock cache."""

        step = 256                  # spare rows a fresh buffer gets: one compaction per `step` tokens

        def __init__(self, max_size: int):
            self.max_size = int(max_size)
            self.keys = None
            self.values = None
            self.offset = 0
            self._start = 0
            self._end = 0
            self._dirty = 0
            self._buf = None

        def _capacity(self) -> int:
            if self.keys is None:
                return 0
            if self.keys is not self._buf:      # assigned from outside: all of it is valid rows
                self._buf = self.keys
                self._start, self._end = 0, int(self.keys.shape[2])
                self._dirty = self._end
            return int(self.keys.shape[2])

        def update_and_fetch(self, keys, values):
            s = int(keys.shape[2])
            cap = self._capacity()
            end = self._end
            first = max(self._start, end - (self.max_size - 1))     # oldest row this call sees
            if end + s + 1 > cap:
                # no room: [rows still visible ‖ new rows ‖ zeros] as one fresh buffer
                old = end > first
                self.keys = mx.concatenate(
                    ([self.keys[..., first:end, :]] if old else [])
                    + [keys, _zeros(keys, 1 + self.step)], axis=2)
                self.values = mx.concatenate(
                    ([self.values[..., first:end, :]] if old else [])
                    + [values, _zeros(values, 1 + self.step)], axis=2)
                self._buf = self.keys
                self._start, self._end = 0, end - first + s
                self._dirty = self._end
            else:
                new_end = end + s
                if new_end < self._dirty:       # the sink slot holds a stale row: rewrite it too
                    keys = mx.concatenate([keys, _zeros(keys, 1)], axis=2)
                    values = mx.concatenate([values, _zeros(values, 1)], axis=2)
                self.keys[..., end: end + keys.shape[2], :] = keys
                self.values[..., end: end + values.shape[2], :] = values
                self._start, self._end = first, new_end
                self._dirty = max(self._dirty, new_end)
            self.offset += s
            return (self.keys[..., self._start: self._end, :],
                    self.values[..., self._start: self._end, :])

        def sink_views(self):
            """``(K, V)`` of the last ``update_and_fetch`` with the sink's zero row appended."""
            end = self._end + 1
            return self.keys[..., self._start: end, :], self.values[..., self._start: end, :]

        # -- the cache contract (mlx-vlm's _BaseCache; our prefix cache and memory accounting) --
        def trim(self, n):
            """Drop the newest ``n`` rows (a verify's rejected drafts). Only rows still in the
            buffer can go; the rows a longer history would need were never kept."""
            n = max(0, min(int(n), self._end - self._start))
            self._end -= n
            self.offset -= n
            return n

        def is_trimmable(self):
            return self.offset < self.max_size

        def size(self):
            return min(self.offset, self.max_size)

        def empty(self):
            return self.keys is None

        @property
        def nbytes(self):
            return 0 if self.keys is None else self.keys.nbytes + self.values.nbytes

        @property
        def state(self):
            if self.keys is None:
                return None, None
            self._capacity()
            return (self.keys[..., self._start: self._end, :],
                    self.values[..., self._start: self._end, :])

        @state.setter
        def state(self, v):
            self.keys, self.values = v
            self._buf = self.keys
            self._start = 0
            self._end = self._dirty = 0 if self.keys is None else int(self.keys.shape[2])
            self.offset = self._end

        @property
        def meta_state(self):
            return (str(self.max_size), str(self.offset))

        @meta_state.setter
        def meta_state(self, v):
            if len(v) != 2:
                raise ValueError("WindowKVCache: this state was saved by another cache class "
                                 f"(meta {tuple(v)!r})")
            self.max_size, self.offset = int(v[0]), int(v[1])

    profile = getattr(vlm_cache, "_windowed_memory_profile", None)
    if profile is not None:             # mlx-vlm's RAM estimate for a bounded window
        WindowKVCache.memory_profile = profile
    return SinkKVCache, WindowKVCache


def is_sink_cache(kv) -> bool:
    """Whether ``kv`` is one of this module's caches (the attention body takes ``sink_views``)."""
    return callable(getattr(kv, "sink_views", None))


def convert(caches: list) -> list:
    """Swap the KV half of each Inkling 2 layer-visit cache (``CacheList(kv, conv)``) for its
    in-place counterpart: ``RotatingKVCache(keep=0)`` -> ``WindowKVCache``, ``KVCache`` ->
    ``SinkKVCache``, and mlx-vlm's own ``AttentionSinkKVCache`` (what its looped model makes
    since 2026-11-22: the same layout, without the views the packed attention and a batched
    step read) to the one its ``max_size`` names. Fresh (empty) caches only; anything else is
    left as it is."""
    from mlx_vlm.models import cache as vlm_cache

    sink_cls, window_cls = classes()
    upstream = getattr(vlm_cache, "AttentionSinkKVCache", None)
    for entry in caches:
        parts = getattr(entry, "caches", None)
        if not parts:
            continue
        kv = parts[0]
        if getattr(kv, "keys", None) is not None:
            continue
        if type(kv) is vlm_cache.RotatingKVCache and kv.keep == 0:
            new = window_cls(kv.max_size)
        elif type(kv) is vlm_cache.KVCache:
            new = sink_cls()
        elif upstream is not None and type(kv) is upstream:
            new = sink_cls() if kv.max_size is None else window_cls(kv.max_size)
        else:
            continue
        entry.caches = (new, *parts[1:])
    return caches


def windowed(kv) -> bool:
    """Whether ``kv`` (a stock or an in-place attention cache) is a sliding window."""
    return getattr(kv, "max_size", None) is not None


def small_window(kv, window: int):
    """A fresh STOCK cache of ``kv``'s class over a ``window``-key window (the self-test fills,
    wraps and rolls back a few keys instead of a thousand)."""
    from mlx_vlm.models import cache as vlm_cache

    if isinstance(kv, vlm_cache.RotatingKVCache):
        return type(kv)(max_size=window, keep=0)
    return type(kv)(window)
