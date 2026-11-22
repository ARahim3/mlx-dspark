"""Engine._begin_request / _finish_request: a request's prefix-cache plan and its bookkeeping,
on real (tiny) models — the split a batched session shares with the single-stream path."""
import mlx.core as mx
import pytest
from test_dspark_tree import _tiny_dspark
from test_tree import PROMPT, _tiny_dense, _Tok

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
