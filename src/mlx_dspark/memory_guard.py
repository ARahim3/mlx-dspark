"""Memory-pressure guard: shed what we can before macOS swaps the model out from under us.

The fits-but-swaps cliff: a big model plus its KV cache, prefix-cache snapshots (a hybrid
checkpoint copies whole recurrent states — ~150 MB fp32 per rung on Qwen3.8-27B, up to 8 per
slot, 2 slots) and the MLX allocator's *retained* free buffers can creep past what macOS will
keep resident. The OS compresses, then swaps, and decode collapses ~10x (issue #14's 45-on-48 GB
shape). Nothing in the engine noticed — until :mod:`roofline` started
reading ``kern.memorystatus_vm_pressure_level``. This module acts on it.

Design (edge-triggered, threaded through the generation thread, never racing it):

- A daemon thread polls the pressure level every few seconds (a microsecond sysctl). A
  transition *into* WARN, or *up to* CRITICAL, requests a shed. Steady pressure does not
  re-trigger; a second shed at the same level needs ``rearm_s`` to pass (120 s) — after the
  first one there is little left to give, and clearing the allocator cache every tick would
  kill freed-buffer reuse for nothing.
- The shed itself runs **on the MLX generation thread**: immediately (via the engine's
  single-worker executor) when the engine is idle, or — when a generation is in flight — at the
  next **round boundary** through the engine's per-round hook. WARN waits up to ``defer_s``
  (60 s) for the request to finish first (a mid-run ``mx.clear_cache()`` forfeits buffer reuse
  for the rest of that generation); CRITICAL takes the very next round. A round is the finest
  granularity there is — a Metal forward can't be interrupted from Python.
- **What WARN frees:** ``mx.clear_cache()`` — the allocator's *retained* buffers, often GBs
  after a long prefill, which macOS counts as ours and which cost nothing to re-acquire — plus
  the prefix cache's interior rungs (~150 MB fp32 each on a 27B hybrid). Every conversation
  KEEPS its boundary checkpoint: the A/B (NOTES "Memory-pressure guard") measured dropping one
  as a 36 s re-prefill under pressure to free 0.6 GB, while the cache clear freed 1.3 GB for
  free. **CRITICAL** empties the prefix cache outright (a critical Mac is about to thrash or
  kill; the re-prefill is the lesser cost). Weights are never touched; the model keeps serving.
- Every shed is recorded (level, bytes freed, when) for ``/health.warnings``, ``/machine`` and
  the log, so a user can see *why* a turn re-prefilled.

Pure enough to test model-free: the poller, clock, and the two actions are injectable.
"""

from __future__ import annotations

import contextlib
import sys
import threading
import time

from .roofline import memory_pressure

LEVEL_ORDER = {"normal": 0, "unknown": 0, "warn": 1, "critical": 2}
DEFAULT_INTERVAL_S = 3.0
DEFAULT_REARM_S = 120.0
DEFAULT_DEFER_S = 60.0


def _allocator_bytes() -> int:
    try:
        import mlx.core as mx

        return int(mx.get_active_memory()) + int(mx.get_cache_memory())
    except Exception:  # noqa: BLE001 — no mlx (tests)
        return 0


def _clear_allocator_cache() -> None:
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:  # noqa: BLE001
        pass


class MemoryGuard:
    """One per :class:`~mlx_dspark.server.Engine`; stopped on close/swap.

    ``prefix`` is the engine's :class:`PrefixCache` (or None); ``submit`` runs a zero-arg
    callable on the MLX thread (the engine's executor); ``is_busy`` says whether a generation
    is in flight; ``poll`` returns ``{"label": ...}`` (:func:`roofline.memory_pressure`).
    """

    def __init__(self, *, prefix=None, submit=None, is_busy=None, poll=memory_pressure,
                 clock=time.monotonic, interval_s: float = DEFAULT_INTERVAL_S,
                 rearm_s: float = DEFAULT_REARM_S, defer_s: float = DEFAULT_DEFER_S,
                 clear_cache=_clear_allocator_cache, allocator_bytes=_allocator_bytes,
                 log=None):
        self.prefix = prefix
        self._submit = submit
        self._is_busy = is_busy or (lambda: False)
        self._poll = poll
        self._clock = clock
        self.interval_s = interval_s
        self.rearm_s = rearm_s
        self.defer_s = defer_s
        self._clear_cache = clear_cache
        self._allocator_bytes = allocator_bytes
        self._log = log or (lambda msg: print(f"[serve] {msg}", file=sys.stderr, flush=True))
        self._lock = threading.Lock()
        self._pending: str | None = None         # level of a shed waiting for the MLX thread
        self._pending_since: float = 0.0
        self._last_level = "normal"              # last polled level (edge detection)
        self._armed: str | None = None           # a suppressed rise, retried while elevated
        self._last_shed_at: float | None = None
        self._last_shed_level: str | None = None
        self.events: list[dict] = []             # shed records, newest last (bounded)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> MemoryGuard:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="memory-guard", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    @property
    def level(self) -> str:
        return self._last_level

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            with contextlib.suppress(Exception):  # a failed tick must never kill the guard
                self.observe(self._poll().get("label", "unknown"))

    # ------------------------------------------------------------------ edge detection

    def observe(self, level: str) -> bool:
        """Feed one pressure reading; returns True when it requested a shed.

        Edge-triggered: a rising edge (normal→warn, anything→critical) requests a shed. A
        rise that the re-arm window suppressed stays *armed* while pressure remains elevated,
        so sustained pressure sheds again once the window passes (at most every ``rearm_s``);
        dropping back to normal disarms it."""
        prev, self._last_level = self._last_level, level
        if LEVEL_ORDER.get(level, 0) == 0:
            self._armed = None
            return False
        rising = LEVEL_ORDER.get(level, 0) > LEVEL_ORDER.get(prev, 0)
        if not rising and self._armed != level:
            return False                      # steady pressure already acted on
        return self.request(level)

    def request(self, level: str) -> bool:
        """Ask for a shed at ``level``; honours the re-arm window; runs now if idle."""
        now = self._clock()
        with self._lock:
            if (self._last_shed_at is not None and now - self._last_shed_at < self.rearm_s
                    and LEVEL_ORDER.get(level, 0) <= LEVEL_ORDER.get(self._last_shed_level, 0)):
                self._armed = level           # retry on later ticks while it stays elevated
                return False                  # recently shed at this level or higher
            self._armed = None
            if self._pending is not None and LEVEL_ORDER[self._pending] >= LEVEL_ORDER[level]:
                return True                   # already waiting at this level or higher
            self._pending, self._pending_since = level, now
        if not self._is_busy() and self._submit is not None:
            # Idle: do it now, on the MLX thread. If a request sneaks in first, the executor
            # serializes us behind it, which is still the right thread and still bounded.
            self._submit(self._run_pending)
        return True

    # ------------------------------------------------------------------ the shed

    def on_round(self) -> None:
        """Called by the engine's per-round hook, on the generation thread, each round."""
        pending = self._pending
        if pending is None:
            return
        if pending == "critical" or self._clock() - self._pending_since >= self.defer_s:
            self._run_pending()

    def _run_pending(self) -> dict | None:
        with self._lock:
            level, self._pending = self._pending, None
            if level is None:
                return None
        return self.shed(level)

    def shed(self, level: str) -> dict:
        """Free what ``level`` calls for. Runs on the MLX thread by construction."""
        before = self._allocator_bytes()
        dropped = {}
        if self.prefix is not None:
            try:
                dropped = self.prefix.shed(level)
            except Exception as e:  # noqa: BLE001 — never let a shed take the engine down
                dropped = {"error": f"{type(e).__name__}: {e}"}
        self._clear_cache()
        after = self._allocator_bytes()
        now = self._clock()
        event = {"level": level, "at": time.time(), "freed_bytes": max(before - after, 0),
                 "allocator_before": before, "allocator_after": after, **dropped}
        with self._lock:
            self._last_shed_at, self._last_shed_level = now, level
            self.events.append(event)
            del self.events[:-16]
        gb = 1024 ** 3
        self._log(f"memory guard: pressure {level.upper()} — freed "
                  f"{event['freed_bytes'] / gb:.2f} GB (prefix cache: "
                  f"{dropped.get('action', 'none')}; allocator {before / gb:.1f} → "
                  f"{after / gb:.1f} GB)")
        return event

    # ------------------------------------------------------------------ reporting

    def info(self) -> dict:
        with self._lock:
            last = self.events[-1] if self.events else None
            pending = self._pending
        return {"enabled": True, "level": self._last_level, "pending": pending,
                "last_shed": last, "sheds": len(self.events),
                "rearm_s": self.rearm_s, "defer_s": self.defer_s}

    def warning(self, within_s: float = 600.0) -> dict | None:
        """A ``/health.warnings`` row for a recent shed, or None."""
        with self._lock:
            last = self.events[-1] if self.events else None
        if last is None or time.time() - last["at"] > within_s:
            return None
        gb = 1024 ** 3
        return {
            "code": "memory_guard",
            "level": "attention",
            "message": (f"Memory guard freed {last['freed_bytes'] / gb:.1f} GB when macOS "
                        f"reported {last['level'].upper()} pressure — the prefix cache was "
                        f"{'emptied, so the next turn re-prefills' if last['level'] == 'critical' else 'trimmed (rungs dropped; conversations kept)'}."),
            "action": "Free memory (close apps, lower the context window, smaller quant) so "
                      "it stops recurring.",
        }
