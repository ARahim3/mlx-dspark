"""Model-free tests for token forcing (``mlx_dspark.token_force``): the pooled statistics,
the two gate rules (rate: one-sided two-proportion z-test with a 0.5pp margin; events:
Garwood Poisson upper limit, rule of three at zero reference events), and the
teacher-forced scorer itself on tiny random mlx-lm models (dense qwen3 and the qwen3_5
gated-DeltaNet hybrid)."""

from __future__ import annotations

import math

import mlx.core as mx
import pytest

from mlx_dspark.generate import greedy_generate
from mlx_dspark.lookup import lookup_generate
from mlx_dspark.target import Target
from mlx_dspark.token_force import (
    DEFAULT_DELTA,
    ForcedScores,
    ForceStats,
    Mismatch,
    event_rule,
    force_stats,
    gate,
    poisson_upper,
    rate_rule,
    score_forced,
)

# ---------------------------------------------------------------- statistics


def _scores(margins, committed=None):
    """ForcedScores where position i mismatches iff margins[i] is not None."""
    n = len(margins)
    committed = committed or list(range(n))
    argmax = [c if m is None else c + 1 for c, m in zip(committed, margins)]
    return ForcedScores(committed, argmax, [0.0 if m is None else m for m in margins])


def test_zero_mismatches():
    st = force_stats(_scores([None] * 50))
    assert st.positions == 50 and st.n_mismatches == 0
    assert st.mismatch_rate == 0.0 and st.max_margin == 0.0
    assert st.near_ties == st.gray == st.large_events == 0
    assert st.to_dict()["list"] == []


def test_margin_classes_and_boundaries():
    # delta 0.25: <=0.25 near-tie, (0.25, 0.5] gray, >0.5 large
    st = force_stats(_scores([0.0, 0.1, 0.25, 0.3, 0.5, 0.6, None, None, None, None]))
    assert st.positions == 10 and st.n_mismatches == 6
    assert st.mismatch_rate == pytest.approx(0.6)
    assert st.near_ties == 3            # 0.0 (exact tie, different id), 0.1, 0.25
    assert st.gray == 2                 # 0.3, 0.5 (2*delta is not > 2*delta)
    assert st.large_events == 1         # 0.6
    assert st.max_margin == pytest.approx(0.6)


def test_custom_delta():
    st = force_stats(_scores([0.3, 0.7, None]), delta=0.4)
    assert st.delta == 0.4 and st.near_ties == 1 and st.gray == 1 and st.large_events == 0
    with pytest.raises(ValueError):
        force_stats(_scores([None]), delta=0.0)


def test_pooling_records_sample_and_position():
    a = _scores([None, 0.1, None])
    b = _scores([0.9, None], committed=[7, 8])
    st = force_stats([a, b])
    assert st.positions == 5
    assert st.mismatches == [Mismatch(0, 1, 1, 2, 0.1), Mismatch(1, 0, 7, 8, 0.9)]
    assert st.large_events == 1


def test_forced_scores_validates_lengths():
    with pytest.raises(ValueError):
        ForcedScores([1, 2], [1], [0.0, 0.0])
    assert ForcedScores([1, 2, 3], [1, 5, 3], [0.0, 1.0, 0.0]).mismatch_positions() == [1]


# ---------------------------------------------------------------- rate rule


def test_rate_rule_identical_zero_passes():
    r = rate_rule(0, 1000, 0, 1000)
    assert r.passed and r.observed == 0.0 and r.threshold == 0.005


def test_rate_rule_small_excess_below_tolerance_passes():
    assert rate_rule(3, 1000, 0, 1000).passed           # 0.3pp < 0.5pp observed


def test_rate_rule_excess_not_significant_passes():
    # 0.9pp observed excess, but the 95% lower bound is below 0.5pp
    r = rate_rule(9, 1000, 0, 1000)
    assert r.passed and r.observed < 0.005


def test_rate_rule_significant_excess_fails():
    r = rate_rule(20, 1000, 0, 1000)
    se = math.sqrt(0.02 * 0.98 / 1000)
    assert not r.passed
    assert r.observed == pytest.approx(0.02 - 1.6448536 * se, rel=1e-6)


def test_rate_rule_equal_high_rates_pass_and_better_candidate_passes():
    assert rate_rule(50, 1000, 50, 1000).passed
    assert rate_rule(0, 1000, 40, 1000).passed


def test_rate_rule_zero_se_edge():
    assert not rate_rule(1000, 1000, 0, 1000).passed    # both degenerate, diff 1.0
    assert rate_rule(1000, 1000, 1000, 1000).passed     # diff 0


def test_rate_rule_rejects_bad_input():
    with pytest.raises(ValueError):
        rate_rule(0, 0, 0, 10)
    with pytest.raises(ValueError):
        rate_rule(11, 10, 0, 10)


# ---------------------------------------------------------------- event rule


def test_poisson_upper_garwood_values():
    assert poisson_upper(0) == pytest.approx(-math.log(0.05))      # 2.9957
    assert poisson_upper(1) == pytest.approx(4.7439, abs=1e-3)
    assert poisson_upper(2) == pytest.approx(6.2958, abs=1e-3)
    assert poisson_upper(10) == pytest.approx(16.962, abs=1e-2)
    with pytest.raises(ValueError):
        poisson_upper(-1)


def test_event_rule_zero_reference_rule_of_three():
    for x in (0, 1, 2):
        assert event_rule(x, 1000, 0, 1000).passed
    assert not event_rule(3, 1000, 0, 1000).passed
    assert not event_rule(10, 1000, 0, 1000).passed


def test_event_rule_longer_candidate_earns_no_extra_allowance():
    # 3 events still fail when the candidate scored twice the positions
    assert not event_rule(3, 2000, 0, 1000).passed


def test_event_rule_shorter_candidate_gets_less_allowance():
    r = event_rule(2, 500, 0, 1000)                      # allowance 1.498
    assert not r.passed and r.threshold == pytest.approx(-math.log(0.05) / 2)


def test_event_rule_nonzero_reference():
    assert event_rule(4, 1000, 1, 1000).passed            # allowance 4.74
    assert not event_rule(5, 1000, 1, 1000).passed


def test_event_rule_rejects_bad_input():
    with pytest.raises(ValueError):
        event_rule(0, 0, 0, 10)
    with pytest.raises(ValueError):
        event_rule(-1, 10, 0, 10)


# ---------------------------------------------------------------- gate


def test_gate_combines_both_rules():
    clean = ForceStats(1000, [Mismatch(0, i, 1, 2, 0.01) for i in range(2)])
    v = gate(clean, ForceStats(1000, []))
    assert v.passed and v.rate.passed and v.events.passed
    # 3 large-margin events: rate is fine (0.3pp) but the event rule fails
    big = ForceStats(1000, [Mismatch(0, i, 1, 2, 5.0) for i in range(3)])
    v = gate(big, ForceStats(1000, []))
    assert v.rate.passed and not v.events.passed and not v.passed
    d = v.to_dict()
    assert d["passed"] is False and d["events"]["name"] == "events"


def test_gate_requires_same_delta():
    with pytest.raises(ValueError):
        gate(ForceStats(10, [], 0.25), ForceStats(10, [], 0.5))


# ---------------------------------------------------------------- the scorer on tiny models


class _Tok:
    """Just enough tokenizer for the generate loops: no eos ids, char-decode."""

    def decode(self, ids):
        return "".join(chr(65 + (i % 26)) for i in ids)


def _tiny_dense():
    """A real (tiny, random) mlx-lm qwen3 model, fp32."""
    from mlx_lm.models.qwen3 import Model, ModelArgs

    args = ModelArgs(model_type="qwen3", hidden_size=32, num_hidden_layers=2,
                     intermediate_size=64, num_attention_heads=4, rms_norm_eps=1e-6,
                     vocab_size=64, num_key_value_heads=2, max_position_embeddings=512,
                     rope_theta=10000.0, head_dim=8, tie_word_embeddings=False)
    mx.random.seed(11)
    model = Model(args)
    mx.eval(model.parameters())
    return model


def _tiny_hybrid():
    """A real (tiny, random) mlx-lm qwen3_5 hybrid: 3 gated-DeltaNet + 1 full-attn layers."""
    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    args = ModelArgs.from_dict({
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 4,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "vocab_size": 64, "rms_norm_eps": 1e-6,
            "linear_num_value_heads": 4, "linear_num_key_heads": 2,
            "linear_key_head_dim": 8, "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
            "tie_word_embeddings": False,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                "partial_rotary_factor": 0.25},
        },
    })
    mx.random.seed(3)
    model = Model(args)
    mx.eval(model.parameters())
    return model


PROMPT = [3, 17, 42, 5, 9, 33, 21]


def _greedy(t, n=24):
    return greedy_generate(t, _Tok(), prompt_ids=PROMPT, max_new_tokens=n).token_ids


@pytest.mark.parametrize("make", [_tiny_dense, _tiny_hybrid], ids=["dense", "hybrid"])
def test_scorer_greedy_output_has_no_mismatches(make):
    t = Target(make(), tokenizer=None)
    out = _greedy(t)
    assert len(out) == 24
    s = score_forced(t, PROMPT, out)
    assert len(s) == 24 and s.committed == out
    assert s.mismatch_positions() == [] and s.argmax == out
    assert all(m == 0.0 for m in s.margin)
    st = force_stats(s)
    assert st.mismatch_rate == 0.0 and st.max_margin == 0.0


@pytest.mark.parametrize("make", [_tiny_dense, _tiny_hybrid], ids=["dense", "hybrid"])
def test_scorer_chunking_is_invariant(make):
    t = Target(make(), tokenizer=None)
    out = _greedy(t)
    out[10] = (out[10] + 1) % 64                          # give it a nonzero margin to compare
    whole = score_forced(t, PROMPT, out)
    # prompt split across 3 body-only chunks, outputs across 5 scoring chunks (ragged tail)
    pieces = score_forced(t, PROMPT, out, chunk=5, prefill_chunk=2)
    assert pieces.argmax == whole.argmax
    assert pieces.margin == pytest.approx(whole.margin, abs=1e-4)


@pytest.mark.parametrize("make", [_tiny_dense, _tiny_hybrid], ids=["dense", "hybrid"])
def test_scorer_reports_corrupted_last_token(make):
    t = Target(make(), tokenizer=None)
    out = _greedy(t)
    bad = list(out)
    bad[-1] = (out[-1] + 1) % 64
    s = score_forced(t, PROMPT, bad)
    assert s.mismatch_positions() == [len(out) - 1]
    assert s.argmax[-1] == out[-1] and s.committed[-1] == bad[-1]
    assert s.margin[-1] > 0.0
    st = force_stats(s)
    assert st.mismatches == [Mismatch(0, len(out) - 1, bad[-1], out[-1], s.margin[-1])]


def test_scorer_corrupted_middle_token_position_and_margin_value():
    t = Target(_tiny_dense(), tokenizer=None)
    out = _greedy(t)
    k = 9
    bad = list(out)
    bad[k] = (out[k] + 5) % 64
    s = score_forced(t, PROMPT, bad, chunk=4)
    # the first mismatch is exactly the corrupted position, and the target still wanted the
    # original token there (later positions condition on the corrupted prefix and may differ)
    assert s.mismatch_positions()[0] == k
    assert s.argmax[k] == out[k]
    # the margin is the logit gap from one un-chunked committed forward
    logits, _ = t.run(mx.array([PROMPT + bad[:-1]]), t.make_cache(), [0])
    row = logits[0, len(PROMPT) - 1 + k].astype(mx.float32)
    want = (row[out[k]] - row[bad[k]]).item()
    assert want > 0.0 and s.margin[k] == pytest.approx(want, abs=1e-4)


def test_scorer_edge_cases():
    t = Target(_tiny_dense(), tokenizer=None)
    assert len(score_forced(t, PROMPT, [])) == 0
    with pytest.raises(ValueError):
        score_forced(t, [], [1, 2])
    with pytest.raises(ValueError):
        score_forced(t, PROMPT, [1, 999])                 # out of the 64-token vocab
    # a one-token prompt: no body-only rows, the first scored row is the prompt itself
    out = greedy_generate(t, _Tok(), prompt_ids=[5], max_new_tokens=8).token_ids
    assert score_forced(t, [5], out).mismatch_positions() == []


def test_gate_end_to_end_lookup_vs_greedy():
    """A lossless spec loop (drafter-free lookup) scored against plain greedy passes the
    gate; a candidate with corrupted tokens fails it."""
    t = Target(_tiny_dense(), tokenizer=None)
    prompt = PROMPT * 3                                   # repetition gives lookup hits
    ref = greedy_generate(t, _Tok(), prompt_ids=prompt, max_new_tokens=32).token_ids
    cand = lookup_generate(t, _Tok(), prompt_ids=prompt, max_new_tokens=32).token_ids
    st_ref = force_stats(score_forced(t, prompt, ref))
    st_cand = force_stats(score_forced(t, prompt, cand))
    assert st_ref.n_mismatches == 0 and st_cand.n_mismatches == 0
    assert gate(st_cand, st_ref).passed
    bad = list(cand)
    for k in (4, 12, 20, 28):                             # 4 forced wrong tokens
        bad[k] = (bad[k] + 7) % 64
    st_bad = force_stats(score_forced(t, prompt, bad), delta=DEFAULT_DELTA)
    assert st_bad.n_mismatches >= 4
    assert not gate(st_bad, st_ref).passed
