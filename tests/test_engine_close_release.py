"""Engine.close() must break every reference chain from the engine's helpers to the model, not
just drop the engine's own attributes. A PrefixCache holds the target's bound ``make_cache``,
and a MemoryGuard holds the PrefixCache; anything that still referenced either after a swap
(the serve banner's frame local in PR #42, or a library user's variable) kept a stale copy of
the weights resident. Model-free: a plain object stands in for the weights; nothing uses
``gc.collect()``, since a strong reference from a live frame is not cyclic garbage."""

from __future__ import annotations

import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest

import mlx_dspark.server as S
from mlx_dspark.memory_guard import MemoryGuard
from mlx_dspark.prefix_cache import PrefixCache


class _Target:
    def make_cache(self):
        return []


def _shell_engine(target, prefix, guard):
    """A real Engine shell (no model) with exactly what close() touches."""
    eng = S.Engine.__new__(S.Engine)
    eng.prefix, eng.target, eng.drafter, eng.cap_controller = prefix, target, object(), None
    eng.memory_guard = guard
    eng._executor = ThreadPoolExecutor(max_workers=1)
    return eng


def test_prefix_cache_release_drops_the_factories_and_the_target_with_them():
    target = _Target()
    ref = weakref.ref(target)
    prefix = PrefixCache(target.make_cache, None, checkpoint=True, slots=2)
    del target
    assert ref() is not None                     # the bound make_cache keeps it alive
    prefix.release()
    assert ref() is None and prefix.released
    with pytest.raises(RuntimeError, match="released"):
        prefix.make_cache()
    with pytest.raises(RuntimeError, match="released"):
        prefix.acquire([1, 2, 3])
    prefix.release()                             # idempotent


def test_memory_guard_stop_lets_go_of_the_prefix_cache():
    target = _Target()
    ref = weakref.ref(target)
    prefix = PrefixCache(target.make_cache, None, checkpoint=True, slots=2)
    guard = MemoryGuard(prefix=prefix, submit=lambda fn: None, is_busy=lambda: False)
    del target
    guard.stop()
    assert guard.prefix is None
    assert isinstance(guard.shed("warn"), dict)       # a stopped guard sheds nothing, quietly
    prefix.release()
    del prefix
    assert ref() is None                         # guard alone no longer reaches the weights


def test_engine_close_releases_the_target_despite_stray_references():
    target = _Target()
    ref = weakref.ref(target)
    prefix = PrefixCache(target.make_cache, None, checkpoint=True, slots=2)
    guard = MemoryGuard(prefix=prefix, submit=lambda fn: None, is_busy=lambda: False)
    eng = _shell_engine(target, prefix, guard)
    del target
    # the stray references PR #42 found in run_server()'s frame, and the ones a library user
    # might keep: both the guard and the cache stay referenced here across the close
    eng.close()
    del eng
    assert ref() is None
    assert guard.prefix is None and prefix.released


def test_engine_close_is_idempotent_and_leaves_nothing_reachable():
    target = _Target()
    ref = weakref.ref(target)
    prefix = PrefixCache(target.make_cache, None, checkpoint=True, slots=2)
    eng = _shell_engine(target, prefix, None)
    del target
    eng.close()
    eng.close()
    assert eng.prefix is None and eng.target is None and eng.memory_guard is None
    assert ref() is None
