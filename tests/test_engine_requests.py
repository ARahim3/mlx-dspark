"""Engine._begin_request / _finish_request: a request's prefix-cache plan and its bookkeeping,
on real (tiny) models — the split a batched session shares with the single-stream path."""
import gc
import weakref

import mlx.core as mx
import pytest
from test_dspark_tree import _tiny_dspark
from test_tree import PROMPT, _tiny_dense, _tiny_hybrid, _Tok

from mlx_dspark import server as S
from mlx_dspark.target import Target

LONG = [(5 * i + 3) % 60 for i in range(40)]


def _engine(mode, **kw):
    target = Target(_tiny_dense(), tokenizer=_Tok())
    drafter = _tiny_dspark(32) if mode == "dspark" else None
    return S.Engine(target, _Tok(), drafter, mode=mode, model_id="tiny", target_repo="tiny",
                    drafter_repo=None, max_draft_tokens=2 if drafter is not None else None,
                    lookup_drafts=False, **kw)


def _ask(engine, ids, n=6):
    return engine.generate(ids, max_tokens=n, temperature=0.0, stop=None, seed=None)


@pytest.mark.parametrize("mode", ["baseline", "dspark"])
def test_a_follow_up_request_reuses_the_prefix_and_decodes_the_same_tokens(mode):
    """Turn 1, then a turn whose prompt extends turn 1's prompt + answer: the second reads
    only what is new, and writes what an engine without a prefix cache writes."""
    cached, plain = _engine(mode), _engine(mode, prefix_cache=False)
    assert cached.prefix is not None and plain.prefix is None
    first = _ask(cached, LONG)
    follow_up = LONG + first.token_ids + PROMPT
    second = _ask(cached, follow_up)
    assert first.reused_tokens == 0 and first.prompt_tokens == len(LONG)
    assert second.reused_tokens >= len(LONG)                 # the whole first turn was held
    assert second.prompt_tokens == len(follow_up)
    want = _ask(plain, follow_up)
    assert second.token_ids == want.token_ids and want.reused_tokens == 0
    assert cached.stats["requests"] == 2
    assert cached.stats["prompt_tokens"] == len(LONG) + len(follow_up)
    assert cached.stats["completion_tokens"] == first.num_tokens + second.num_tokens
    assert cached.stats["generation_seconds"] > 0 and cached.prefix.hits == 1


def test_a_failed_generation_resets_the_prefix_cache(monkeypatch):
    engine = _engine("baseline")
    _ask(engine, LONG)
    assert engine.prefix.info()["cached_tokens"] > 0

    def boom(*a, **k):
        raise RuntimeError("mid-generation")

    monkeypatch.setattr(S, "greedy_generate", boom)
    with pytest.raises(RuntimeError):
        _ask(engine, LONG + PROMPT)
    assert engine.prefix.info()["cached_tokens"] == 0        # never a desynced cache
    assert engine.stats["requests"] == 1


def test_cap_at_follows_the_configured_cap_without_a_depth_model():
    engine = _engine("dspark")
    assert engine._cap_at(10) == engine._cap_at(100_000) == 2
    mx.eval(mx.zeros(1))


def test_begin_request_creates_no_reference_cycle_pinning_cache():
    """``run.on_prefill_chunk`` must not capture ``run`` back into a cycle: CPython refcounting
    has to reclaim ``run`` and its ``run.cache`` as soon as a request finishes instead of pinning
    multi-GB KV buffers in active Metal memory until a generation-2 GC happens to run."""
    class _CacheObj:
        pass

    engine = _engine("baseline", prefix_cache=False)
    q = engine.rounds.subscribe()
    gc.disable()
    try:
        run = engine._begin_request(LONG, on_text=None)
        sentinel = _CacheObj()
        ref = weakref.ref(sentinel)
        run.cache = sentinel
        del sentinel
        run.on_prefill_chunk(16)
        engine._end_prefill_progress(run)
        engine._end_prefill_progress(run)                     # idempotent after clearing flag
        del run
        assert ref() is None
    finally:
        gc.enable()
        engine.rounds.unsubscribe(q)
        engine.close()
    seen = [q.get_nowait() for _ in range(q.qsize())]
    assert [(e["processed"], e.get("done")) for e in seen] == [(16, None), (len(LONG), True)]


def test_checkpoint_mode_generate_releases_request_cache_without_gc():
    """End-to-end on a tiny hybrid (checkpoint-mode prefix cache): each turn checks out a copy
    into ``run.cache`` and never re-stores it, so returning from ``generate()`` must drop that
    per-request cache immediately via refcounting even when cyclic GC is disabled."""
    class _TrackedCache(list):
        pass

    target = Target(_tiny_hybrid(), tokenizer=_Tok())
    engine = S.Engine(target, _Tok(), None, mode="baseline", model_id="tiny-hybrid",
                      target_repo="tiny-hybrid", drafter_repo=None, max_draft_tokens=None,
                      lookup_drafts=False)
    assert engine.prefix is not None and engine.prefix.checkpoint_mode
    orig_make = engine.prefix.make_cache
    refs = []

    def tracked_make():
        c = _TrackedCache(orig_make())
        refs.append(weakref.ref(c))
        return c

    engine.prefix.make_cache = tracked_make
    gc.disable()
    try:
        first = _ask(engine, LONG, n=4)
        follow_up = LONG + first.token_ids + PROMPT
        _ask(engine, follow_up, n=4)
        assert len(refs) >= 2, f"expected >=2 per-request caches, got {len(refs)}"
        alive = sum(r() is not None for r in refs)
        assert alive == 0, f"{alive}/{len(refs)} per-request caches still alive without GC"
    finally:
        gc.enable()
        engine.close()
