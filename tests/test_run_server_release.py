"""run_server() must not keep the engine's MemoryGuard alive for the life of the process.

After the first ``POST /admin/load`` on a server that was started with a model, the *initial*
engine's weights used to stay resident: ``run_server`` kept ``guard = getattr(engine,
"memory_guard", None)`` in a local (for the startup banner) while ``httpd.serve_forever()``
blocked, and

    run_server frame local -> MemoryGuard.prefix -> PrefixCache.make_cache (bound) -> target

kept the whole target reachable after ``Engine.close()``.

Model-free: a plain object stands in for the weights (anything that still references it is
"resident"), the real ``Engine.close()`` runs, and the HTTP server is stubbed so
``serve_forever`` runs a hook instead of blocking.
"""

import contextlib
import io
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest

import mlx_dspark.server as S
from mlx_dspark.memory_guard import MemoryGuard
from mlx_dspark.prefix_cache import PrefixCache
from mlx_dspark.server import EngineHolder


class _Target:
    """Stands in for the loaded weights."""

    def make_cache(self):
        return []


def _engine(with_guard=True):
    """A real Engine shell (no model) with a prefix cache and, optionally, a MemoryGuard."""
    target = _Target()
    prefix = PrefixCache(target.make_cache, None, checkpoint=True, slots=2)
    eng = S.Engine.__new__(S.Engine)
    eng.mode, eng.model_id, eng.target_repo, eng.drafter_repo = "dflash", "m", "/t", "/d"
    eng.small_m = eng.sdpa_split = eng.multirow_attn = False
    eng.cpu_split, eng.cap_controller, eng.sampling_defaults = None, None, {}
    eng.drafter_window = None
    eng.prefix, eng.target, eng.drafter = prefix, target, object()
    eng._executor = ThreadPoolExecutor(max_workers=1)
    eng.memory_guard = (
        MemoryGuard(prefix=prefix, submit=lambda fn: None, is_busy=lambda: False)
        if with_guard else None
    )
    return eng


def _serve(monkeypatch, holder, hook):
    """Run run_server(holder) with ThreadingHTTPServer stubbed; ``hook(frame)`` runs in
    place of serve_forever() with run_server's own frame. Returns the printed banner."""

    class _HTTPD:
        def __init__(self, address, handler):
            self.daemon_threads = False

        def serve_forever(self):
            hook(sys._getframe(1))

        def server_close(self):
            pass

    monkeypatch.setattr(S, "ThreadingHTTPServer", _HTTPD)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        S.run_server(holder, host="127.0.0.1", port=0)
    return out.getvalue()


class TestRunServerDoesNotPinTheGuard:
    def test_frame_holds_no_reference_to_the_memory_guard(self, monkeypatch):
        eng = _engine()
        guard_id = id(eng.memory_guard)
        holder = EngineHolder(eng, load_kwargs={})
        del eng
        seen = {}

        def hook(frame):
            seen["pinned"] = [k for k, v in frame.f_locals.items() if id(v) == guard_id]

        _serve(monkeypatch, holder, hook)
        assert seen["pinned"] == []

    def test_initial_target_is_released_by_a_swap_while_serving(self, monkeypatch):
        holder = EngineHolder(_engine(), load_kwargs={})
        target_ref = weakref.ref(holder._engine.target)
        seen = {}

        def hook(_frame):
            # what EngineHolder.swap() does before it loads the replacement
            old = holder._engine
            holder._engine = None
            old.close()
            del old
            # no gc.collect(): the release must not depend on the cyclic collector
            seen["alive"] = target_ref() is not None

        _serve(monkeypatch, holder, hook)
        assert seen["alive"] is False

    @pytest.mark.parametrize(("with_guard", "expected"), [
        (True, "  memory guard: on (sheds caches under macOS memory pressure)"),
        (False, "  memory guard: off"),
    ])
    def test_banner_line_is_unchanged(self, monkeypatch, with_guard, expected):
        holder = EngineHolder(_engine(with_guard), load_kwargs={})
        banner = _serve(monkeypatch, holder, lambda frame: None)
        assert expected in banner.splitlines()
