"""Per-round telemetry: ring buffer, aggregates, and fan-out.

Model-free — the whole point of RoundLog is that it's a plain data structure the decode loops
call into, so it can be exercised without weights.
"""

import contextlib
import queue
import threading

import pytest

from mlx_dspark.telemetry import RoundLog, RoundRecorder


def _round(drafted=2, accepted=2, committed=3, cap=2, source="drafter"):
    return {"drafted": drafted, "accepted": accepted, "committed": committed,
            "cap": cap, "source": source}


class TestRingBuffer:
    def test_records_and_returns_events(self):
        log = RoundLog()
        log.record(_round())
        log.record(_round(accepted=1, committed=2))
        events = log.snapshot()
        assert len(events) == 2
        assert [e["committed"] for e in events] == [3, 2]

    def test_sequence_numbers_are_monotonic(self):
        log = RoundLog()
        for _ in range(5):
            log.record(_round())
        assert [e["seq"] for e in log.snapshot()] == [1, 2, 3, 4, 5]

    def test_ring_evicts_oldest(self):
        log = RoundLog(capacity=3)
        for i in range(10):
            log.record(_round(committed=i))
        events = log.snapshot()
        assert len(events) == 3
        assert [e["committed"] for e in events] == [7, 8, 9]

    def test_snapshot_limit(self):
        log = RoundLog()
        for i in range(10):
            log.record(_round(committed=i))
        assert [e["committed"] for e in log.snapshot(3)] == [7, 8, 9]

    def test_reset_clears_events_and_aggregates(self):
        log = RoundLog()
        log.record(_round())
        log.reset()
        assert log.snapshot() == []
        assert log.stats()["rounds"] == 0


class TestPositionAcceptance:
    """d_0, d_1, ... — the drafter-quality curve the project's speedup rests on."""

    def test_all_accepted_gives_one_everywhere(self):
        log = RoundLog()
        for _ in range(4):
            log.record(_round(drafted=3, accepted=3, committed=4))
        assert log.position_acceptance() == [1.0, 1.0, 1.0]

    def test_decay_across_positions(self):
        log = RoundLog()
        # 4 rounds, each drafting 3; position 0 always accepted, 1 accepted half, 2 never.
        for accepted in (1, 2, 1, 2):
            log.record(_round(drafted=3, accepted=accepted, committed=accepted + 1))
        d = log.position_acceptance()
        assert d[0] == 1.0                     # 4/4
        assert d[1] == 0.5                     # 2/4
        assert d[2] == 0.0                     # 0/4

    def test_curve_grows_when_a_later_round_drafts_deeper(self):
        log = RoundLog()
        log.record(_round(drafted=2, accepted=2, committed=3))
        log.record(_round(drafted=5, accepted=4, committed=5))
        d = log.position_acceptance()
        assert len(d) == 5
        assert d[0] == 1.0                     # accepted in both rounds
        assert d[4] == 0.0                     # offered once, rejected

    def test_rounds_with_no_draft_do_not_pollute_the_curve(self):
        """Baseline/parked rounds draft nothing — they must not count as position-0 misses."""
        log = RoundLog()
        log.record(_round(drafted=2, accepted=2, committed=3))
        for _ in range(10):
            log.record(_round(drafted=0, accepted=0, committed=1, cap=0, source="plain"))
        assert log.position_acceptance() == [1.0, 1.0]


class TestStats:
    def test_mean_accept_len(self):
        log = RoundLog()
        log.record(_round(committed=3))
        log.record(_round(committed=1))
        assert log.stats()["mean_accept_len"] == 2.0

    def test_accept_histogram(self):
        log = RoundLog()
        for committed in (1, 3, 3, 2):
            log.record(_round(committed=committed))
        assert log.stats()["accept_histogram"] == {"1": 1, "2": 1, "3": 2}

    def test_lookup_rounds_counted_separately(self):
        log = RoundLog()
        log.record(_round(source="lookup"))
        log.record(_round(source="drafter"))
        log.record(_round(source="lookup"))
        assert log.stats()["lookup_rounds"] == 2

    def test_draft_acceptance_ratio(self):
        log = RoundLog()
        log.record(_round(drafted=4, accepted=1, committed=2))
        log.record(_round(drafted=4, accepted=3, committed=4))
        assert log.stats()["draft_acceptance"] == 0.5      # 4 of 8

    def test_empty_log_reports_zeros_not_errors(self):
        stats = RoundLog().stats()
        assert stats["rounds"] == 0
        assert stats["mean_accept_len"] == 0.0
        assert stats["draft_acceptance"] == 0.0
        assert stats["position_acceptance"] == []


class TestFanOut:
    def test_subscriber_receives_events(self):
        log = RoundLog()
        q = log.subscribe()
        log.record(_round(committed=7))
        assert q.get_nowait()["committed"] == 7

    def test_unsubscribe_stops_delivery(self):
        log = RoundLog()
        q = log.subscribe()
        log.unsubscribe(q)
        log.record(_round())
        with pytest.raises(queue.Empty):
            q.get_nowait()

    def test_multiple_subscribers_all_get_the_event(self):
        log = RoundLog()
        qs = [log.subscribe() for _ in range(3)]
        log.record(_round(committed=5))
        assert all(q.get_nowait()["committed"] == 5 for q in qs)

    def test_named_event_is_live_only(self):
        log = RoundLog()
        q = log.subscribe()
        event = {"req": "abc", "processed": 2048, "total": 22000, "active": True}
        log.publish("prefill", event)
        assert q.get_nowait() == ("prefill", event)
        assert log.snapshot() == []
        assert log.stats()["rounds"] == 0

    def test_slow_subscriber_never_blocks_the_writer(self):
        """A stalled HTTP client must not be able to stall token generation."""
        from mlx_dspark.telemetry import SUBSCRIBER_BACKLOG

        log = RoundLog(capacity=4)
        log.subscribe()                                    # never drained
        for _ in range(SUBSCRIBER_BACKLOG + 50):
            log.record(_round())                           # must not raise or hang
        assert log.stats()["rounds"] == SUBSCRIBER_BACKLOG + 50

    def test_concurrent_writers_and_readers(self):
        log = RoundLog(capacity=1000)
        q = log.subscribe()
        stop = threading.Event()

        def drain():
            while not stop.is_set():
                with contextlib.suppress(queue.Empty):
                    q.get(timeout=0.05)

        reader = threading.Thread(target=drain)
        reader.start()
        writers = [threading.Thread(target=lambda: [log.record(_round()) for _ in range(100)])
                   for _ in range(4)]
        for w in writers:
            w.start()
        for w in writers:
            w.join()
        stop.set()
        reader.join()
        assert log.stats()["rounds"] == 400


class TestRoundRecorder:
    def test_records_with_request_identity_and_index(self):
        log = RoundLog()
        rec = RoundRecorder(log, "abc123", "dspark")
        rec(drafted=2, accepted=2, committed=3, cap=2)
        rec(drafted=2, accepted=0, committed=1, cap=2)
        events = log.snapshot()
        assert [e["req"] for e in events] == ["abc123", "abc123"]
        assert [e["i"] for e in events] == [0, 1]
        assert events[0]["mode"] == "dspark"

    def test_measures_per_round_wall_time(self):
        log = RoundLog()
        rec = RoundRecorder(log, "r", "dspark")
        rec(drafted=1, accepted=1, committed=2)
        assert log.snapshot()[0]["ms"] >= 0.0

    def test_source_is_carried_through(self):
        log = RoundLog()
        rec = RoundRecorder(log, "r", "dspark")
        rec(drafted=6, accepted=6, committed=7, cap=6, source="lookup")
        assert log.snapshot()[0]["source"] == "lookup"


class TestDecayRatio:
    def _log_with(self, early_ms, late_ms, n=16):
        log = RoundLog()
        rec = RoundRecorder(log, "reqA", "dspark")
        # round 0 carries prefill in its ms and must be skipped — give it an absurd value
        log.record({"req": "reqA", "i": 0, "drafted": 2, "accepted": 2, "committed": 3, "ms": 5000.0})
        for i in range(1, 2 * n + 1):
            ms = early_ms if i <= n else late_ms
            log.record({"req": "reqA", "i": i, "drafted": 2, "accepted": 2, "committed": 3,
                        "ms": ms})
        assert rec.request_id == "reqA"
        return log

    def test_steady_run_is_one(self):
        assert self._log_with(10.0, 10.0).decay_ratio("reqA") == pytest.approx(1.0)

    def test_slowing_run_is_below_one_and_ignores_round_zero(self):
        assert self._log_with(10.0, 20.0).decay_ratio("reqA") == pytest.approx(0.5)

    def test_too_short_is_none(self):
        log = RoundLog()
        for i in range(10):
            log.record({"req": "r", "i": i, "drafted": 1, "accepted": 1, "committed": 2, "ms": 1.0})
        assert log.decay_ratio("r") is None

    def test_scoped_to_the_request(self):
        log = self._log_with(10.0, 20.0)
        assert log.decay_ratio("other") is None
