"""Per-round speculative-decoding telemetry.

The engine already *knows* everything interesting about a round — how many tokens were
drafted, how many the target accepted, which cap was used, whether the draft came free from
the n-gram lookup — but until now that only survived as an average in ``GenResult``. This
module keeps the per-round detail so it can be streamed live (``GET /events``) and aggregated
(``GET /metrics``).

The one aggregate worth calling out is **per-position acceptance** (d₀, d₁, d₂ …): the
probability that the drafter's i-th proposed token survives verification. It is the quantity
the whole project's speedup rests on — the README's "d_0 ~85%" — and it is derivable for free
from what each round already reports.

Design notes:
- ``record`` runs on the generation thread, inside the decode loop. It must stay cheap: a dict
  append, a couple of counter bumps, and a non-blocking put per subscriber. No I/O, no locks
  held across a callback.
- Subscribers are bounded queues that **drop** when full rather than block. A slow HTTP client
  must never be able to stall token generation.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time

# A round is a few milliseconds at best, so a client polling once a second still wants a
# decent window of history; 512 rounds is a few seconds of the fastest observed decoding.
DEFAULT_CAPACITY = 512

# Per-subscriber backlog. Deliberately small: if a consumer falls this far behind, the useful
# thing is fresh events, not a growing pile of stale ones.
SUBSCRIBER_BACKLOG = 256


class RoundLog:
    """Thread-safe ring buffer + fan-out for per-round events.

    Written from the single generation thread, read from HTTP handler threads.
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY):
        self.capacity = max(1, int(capacity))
        self._events: list[dict] = []
        self._subscribers: set[queue.Queue] = set()
        self._lock = threading.Lock()
        self._seq = 0
        self._t0 = time.perf_counter()
        # per-position acceptance: offered[i] = drafts proposed at position i,
        # accepted[i] = how many of those the target kept.
        self._offered: list[int] = []
        self._accepted: list[int] = []
        # accept-length histogram, committed-tokens -> rounds
        self._accept_hist: dict[int, int] = {}
        self._totals = {"rounds": 0, "committed": 0, "drafted": 0,
                        "accepted": 0, "lookup_rounds": 0}

    # ------------------------------------------------------------------ writing

    def record(self, event: dict) -> None:
        """Append one round. Called from the decode loop — keep it cheap."""
        event["seq"] = self._seq = self._seq + 1
        event.setdefault("t", round((time.perf_counter() - self._t0) * 1e3, 1))

        drafted = int(event.get("drafted", 0))
        accepted = int(event.get("accepted", 0))
        committed = int(event.get("committed", 0))

        with self._lock:
            self._events.append(event)
            if len(self._events) > self.capacity:
                del self._events[: len(self._events) - self.capacity]

            self._totals["rounds"] += 1
            self._totals["committed"] += committed
            self._totals["drafted"] += drafted
            self._totals["accepted"] += accepted
            if event.get("source") == "lookup":
                self._totals["lookup_rounds"] += 1

            self._accept_hist[committed] = self._accept_hist.get(committed, 0) + 1

            if drafted:
                if len(self._offered) < drafted:
                    grow = drafted - len(self._offered)
                    self._offered.extend([0] * grow)
                    self._accepted.extend([0] * grow)
                for i in range(drafted):
                    self._offered[i] += 1
                for i in range(accepted):
                    self._accepted[i] += 1

            subscribers = tuple(self._subscribers)

        for q in subscribers:
            with contextlib.suppress(queue.Full):   # a slow client must never stall generation
                q.put_nowait(event)

    def publish(self, event_name: str, event: dict) -> None:
        """Fan out a named live event without adding it to round history or aggregates."""
        with self._lock:
            subscribers = tuple(self._subscribers)
        for q in subscribers:
            with contextlib.suppress(queue.Full):
                q.put_nowait((event_name, event))

    # ------------------------------------------------------------------ reading

    def snapshot(self, limit: int | None = None) -> list[dict]:
        with self._lock:
            events = list(self._events)
        return events[-limit:] if limit else events

    def decay_ratio(self, request_id: str, n: int = 16) -> float | None:
        """Late-run / early-run decode rate for one request: the committed-tokens-per-ms of
        its last ``n`` rounds over its first ``n``. < 0.85 means the generation slowed
        noticeably within itself (growing-context cost or thermals). None with fewer than
        ``2n`` rounds — too short to say. Round 0 is skipped: its ``ms`` includes prefill."""
        with self._lock:
            evs = [e for e in self._events if e.get("req") == request_id and e.get("i", 0) > 0]
        if len(evs) < 2 * n:
            return None

        def rate(chunk):
            ms = sum(float(e.get("ms", 0.0)) for e in chunk)
            return (sum(int(e.get("committed", 0)) for e in chunk) / ms) if ms > 0 else None

        early, late = rate(evs[:n]), rate(evs[-n:])
        if not early or late is None:
            return None
        return round(late / early, 4)

    def position_acceptance(self) -> list[float]:
        """d₀, d₁, d₂ … — the fraction of drafts accepted at each block position."""
        with self._lock:
            return [round(a / o, 4) if o else 0.0
                    for a, o in zip(self._accepted, self._offered)]

    def stats(self) -> dict:
        with self._lock:
            totals = dict(self._totals)
            hist = dict(self._accept_hist)
            offered = list(self._offered)
            accepted = list(self._accepted)
        rounds = totals["rounds"]
        return {
            "rounds": rounds,
            "committed_tokens": totals["committed"],
            "mean_accept_len": round(totals["committed"] / rounds, 3) if rounds else 0.0,
            "draft_acceptance": (round(totals["accepted"] / totals["drafted"], 4)
                                 if totals["drafted"] else 0.0),
            "lookup_rounds": totals["lookup_rounds"],
            # d_0, d_1, … — the drafter-quality curve the speedup rests on
            "position_acceptance": [round(a / o, 4) if o else 0.0
                                    for a, o in zip(accepted, offered)],
            "position_offered": offered,
            "accept_histogram": {str(k): hist[k] for k in sorted(hist)},
        }

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._offered.clear()
            self._accepted.clear()
            self._accept_hist.clear()
            for k in self._totals:
                self._totals[k] = 0

    # ------------------------------------------------------------------ pub/sub

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=SUBSCRIBER_BACKLOG)
        with self._lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._subscribers.discard(q)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)


class RoundRecorder:
    """The ``on_round`` callable handed to a generation loop.

    Holds the per-request identity and round index so the loops themselves stay ignorant of
    telemetry beyond "call this with the numbers you already computed".
    """

    __slots__ = ("_i", "_t", "log", "mode", "request_id")

    def __init__(self, log: RoundLog, request_id: str, mode: str):
        self.log = log
        self.request_id = request_id
        self.mode = mode
        self._i = 0
        self._t = time.perf_counter()

    def __call__(self, *, drafted: int, accepted: int, committed: int,
                 cap: int | None = None, source: str = "drafter") -> None:
        now = time.perf_counter()
        self.log.record({
            "req": self.request_id,
            "mode": self.mode,
            "i": self._i,
            "drafted": drafted,
            "accepted": accepted,
            "committed": committed,
            "cap": cap,
            "source": source,
            "ms": round((now - self._t) * 1e3, 2),
        })
        self._i += 1
        self._t = now
