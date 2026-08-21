"""Decode-only vs end-to-end tok/s reporting (2026-08-21).

The split is purely additive: `tokens_per_sec` keeps its end-to-end (prefill+decode)
meaning, and `decode_tokens_per_sec` reports the decode-only rate other local runtimes
show. Decode rate is always >= end-to-end and collapses to it when prefill is unmeasured,
so no existing caller or ratio changes.
"""
from mlx_dspark.generate import GenResult


def _mk(n_tokens, seconds, prefill=0.0):
    return GenResult(
        text="x" * n_tokens,
        token_ids=list(range(n_tokens)),
        num_tokens=n_tokens,
        num_rounds=n_tokens,
        accept_lengths=[1] * n_tokens,
        target_forwards=n_tokens,
        seconds=seconds,
        prefill_seconds=prefill,
    )


def test_default_is_backward_compatible():
    # Any caller that doesn't set prefill_seconds gets decode == end-to-end (no surprise).
    r = _mk(100, 2.0)
    assert r.prefill_seconds == 0.0
    assert r.decode_seconds == 2.0
    assert r.decode_tokens_per_sec == r.tokens_per_sec == 50.0


def test_decode_rate_is_higher_never_lower():
    # 100 tokens in 2.0s total, 0.5s of it prefill -> decode is faster, not slower.
    r = _mk(100, 2.0, prefill=0.5)
    assert r.tokens_per_sec == 50.0                       # end-to-end unchanged
    assert r.decode_seconds == 1.5
    assert r.decode_tokens_per_sec == 100 / 1.5           # ~66.7
    assert r.decode_tokens_per_sec > r.tokens_per_sec


def test_simple_prompt_barely_moves():
    # A simple prompt has tiny prefill -> decode ~= end-to-end (a few percent higher at most).
    r = _mk(200, 4.0, prefill=0.04)   # 1% of the wall in prefill
    assert r.decode_tokens_per_sec > r.tokens_per_sec
    assert r.decode_tokens_per_sec / r.tokens_per_sec < 1.02


def test_all_prefill_does_not_divide_by_zero():
    r = _mk(10, 1.0, prefill=1.0)     # degenerate: nothing left for decode
    assert r.decode_seconds == 1e-9
    assert r.decode_tokens_per_sec == 10 / 1e-9   # finite, not a ZeroDivisionError
