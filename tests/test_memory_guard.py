"""Model-free tests for the memory-pressure guard: edge detection, the re-arm window, the
idle-vs-generating dispatch (immediate vs round boundary), WARN deferral vs CRITICAL, the shed
accounting, and the /health warning row."""

from __future__ import annotations

from mlx_dspark.memory_guard import MemoryGuard


class _Prefix:
    def __init__(self):
        self.calls = []

    def shed(self, level):
        self.calls.append(level)
        return {"action": f"shed-{level}", "prefix_bytes_freed": 100, "slots_dropped": 1,
                "rungs_dropped": 3}


class _Rig:
    def __init__(self, busy=False):
        self.clock = 1000.0
        self.busy = busy
        self.submitted = []
        self.cleared = 0
        self.logs = []
        self.prefix = _Prefix()
        self.guard = MemoryGuard(
            prefix=self.prefix, submit=self.submitted.append, is_busy=lambda: self.busy,
            poll=lambda: {"label": "normal"}, clock=lambda: self.clock,
            clear_cache=self._clear, allocator_bytes=lambda: 1000 if not self.cleared else 400,
            log=self.logs.append, rearm_s=120.0, defer_s=60.0)

    def _clear(self):
        self.cleared += 1

    def run_submitted(self):
        while self.submitted:
            self.submitted.pop(0)()


def test_rising_edge_sheds_immediately_when_idle():
    rig = _Rig()
    assert rig.guard.observe("normal") is False
    assert rig.guard.observe("warn") is True
    assert len(rig.submitted) == 1                 # idle -> straight onto the MLX thread
    rig.run_submitted()
    assert rig.prefix.calls == ["warn"] and rig.cleared == 1
    info = rig.guard.info()
    assert info["sheds"] == 1 and info["pending"] is None
    assert info["last_shed"]["freed_bytes"] == 600 and info["last_shed"]["action"] == "shed-warn"
    assert "memory guard: pressure WARN" in rig.logs[0]


def test_steady_pressure_does_not_retrigger_and_rearm_window_holds():
    rig = _Rig()
    rig.guard.observe("warn")
    rig.run_submitted()
    assert rig.guard.observe("warn") is False      # steady, not an edge
    rig.guard.observe("normal")
    rig.clock += 30
    assert rig.guard.observe("warn") is False      # a second WARN edge inside the re-arm window
    rig.clock += 100                               # 130 s after the shed, still WARN
    assert rig.guard.observe("warn") is True       # the suppressed rise stayed armed
    rig.run_submitted()
    assert rig.prefix.calls == ["warn", "warn"]
    assert rig.guard.observe("warn") is False      # acted on; steady again


def test_disarms_when_pressure_returns_to_normal():
    rig = _Rig()
    rig.guard.observe("warn")
    rig.run_submitted()
    rig.clock += 10
    rig.guard.observe("normal")
    assert rig.guard.observe("warn") is False      # suppressed (re-arm) and armed
    rig.guard.observe("normal")                    # …then pressure cleared: disarmed
    rig.clock += 200
    assert rig.guard.observe("normal") is False
    assert rig.prefix.calls == ["warn"]


def test_escalation_to_critical_bypasses_the_rearm_window():
    rig = _Rig()
    rig.guard.observe("warn")
    rig.run_submitted()
    rig.clock += 5
    assert rig.guard.observe("critical") is True
    rig.run_submitted()
    assert rig.prefix.calls == ["warn", "critical"]


def test_warn_defers_to_a_round_boundary_after_the_defer_window_while_generating():
    rig = _Rig(busy=True)
    assert rig.guard.observe("warn") is True
    assert rig.submitted == []                     # generating: nothing submitted
    rig.guard.on_round()
    assert rig.prefix.calls == []                  # too early — let the request finish
    rig.clock += 61
    rig.guard.on_round()
    assert rig.prefix.calls == ["warn"] and rig.cleared == 1
    rig.guard.on_round()                           # nothing pending any more
    assert rig.prefix.calls == ["warn"]


def test_critical_takes_the_very_next_round():
    rig = _Rig(busy=True)
    rig.guard.observe("critical")
    rig.guard.on_round()
    assert rig.prefix.calls == ["critical"]


def test_pending_does_not_double_shed_when_both_paths_fire():
    rig = _Rig()
    rig.guard.observe("warn")                      # idle -> submitted
    rig.guard.on_round()                           # a round boundary races the submission…
    rig.run_submitted()                            # …and the queued job finds nothing pending
    assert rig.prefix.calls == ["warn"]


def test_unknown_and_normal_levels_never_shed():
    rig = _Rig()
    assert rig.guard.observe("unknown") is False
    assert rig.guard.observe("normal") is False
    assert rig.submitted == []


def test_shed_survives_a_failing_prefix_and_no_prefix():
    rig = _Rig()
    rig.guard.prefix = None
    rig.guard.shed("warn")
    assert rig.cleared == 1

    class Boom:
        def shed(self, level):
            raise RuntimeError("nope")

    rig.guard.prefix = Boom()
    event = rig.guard.shed("critical")
    assert "error" in event and rig.cleared == 2


def test_warning_row_appears_after_a_shed():
    rig = _Rig()
    assert rig.guard.warning() is None
    rig.guard.shed("warn")
    row = rig.guard.warning()
    assert row["code"] == "memory_guard" and "trimmed" in row["message"] and row["action"]
    assert "conversations kept" in row["message"]
    rig.guard.shed("critical")
    assert "emptied" in rig.guard.warning()["message"]


def test_polling_thread_feeds_observe(monkeypatch):
    levels = iter(["normal", "warn", "warn"])
    seen = []
    guard = MemoryGuard(poll=lambda: {"label": next(levels, "warn")}, interval_s=0.01,
                        submit=lambda fn: seen.append(fn), is_busy=lambda: False,
                        clear_cache=lambda: None, allocator_bytes=lambda: 0,
                        log=lambda m: None)
    guard.start()
    import time
    deadline = time.time() + 2
    while not seen and time.time() < deadline:
        time.sleep(0.01)
    guard.stop()
    assert len(seen) == 1 and guard.level == "warn"
