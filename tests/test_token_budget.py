"""`max_new_tokens` is a hard cap in every mode. A speculative round commits a whole block
(accepted drafts + the bonus token), so the loops used to return up to a round MORE than asked:
the server reported 201 and 205 completion tokens for `max_tokens: 200` (found in the TensorFold
head-to-head, 2026-11-08). Plain decoding was always exact. The KV the verify wrote for the dropped
tail is harmless: the prefix cache normalizes caches down to the token record on store (the same
path an eos landing mid-block takes)."""
from __future__ import annotations

import mlx.core as mx
import pytest
from test_dspark_tree import _tiny_dspark
from test_tree import PROMPT, _tiny_dense, _tiny_dflash2_for, _Tok

from mlx_dspark.batch_engine import _RowTracker
from mlx_dspark.generate import dflash_generate, greedy_generate, speculative_generate
from mlx_dspark.lookup import lookup_generate
from mlx_dspark.target import Target


@pytest.mark.parametrize("budget", [1, 2, 5, 11])
def test_dflash_generate_returns_exactly_the_budget(budget):
    t = Target(_tiny_dense(), tokenizer=None)
    d = _tiny_dflash2_for(t)
    ref = greedy_generate(t, _Tok(), prompt_ids=PROMPT, max_new_tokens=budget).token_ids
    for tree in (None, 0):
        res = dflash_generate(t, _Tok(), d, prompt_ids=PROMPT, max_new_tokens=budget,
                              max_draft_tokens=7, tree=tree)
        assert len(res.token_ids) == budget == res.num_tokens
        assert res.token_ids == ref                      # the prefix of greedy, cut at the budget
        assert res.finish_reason == "length"


@pytest.mark.parametrize("budget", [1, 3, 6])
def test_speculative_generate_returns_exactly_the_budget(budget):
    t = Target(_tiny_dense(), tokenizer=None)
    d = _tiny_dspark(32)
    ref = greedy_generate(t, _Tok(), prompt_ids=PROMPT, max_new_tokens=budget).token_ids
    for tree in (None, 0):
        res = speculative_generate(t, _Tok(), d, prompt_ids=PROMPT, max_new_tokens=budget,
                                   max_draft_tokens=4, lookup_drafts=False, tree=tree)
        assert len(res.token_ids) == budget == res.num_tokens
        assert res.token_ids == ref
        assert res.finish_reason == "length"


class _ParkedController:
    """Parked from the first round, a cap-1 probe every ``probe_every`` rounds."""

    probe_every = 3

    def __init__(self, sync_clock):
        self.parked_sync_clock = sync_clock
        self.cap = 0
        self.rounds = 0
        self.seen = []

    def set_depth(self, n):
        pass

    def update(self, accepted_n, cap_used, *, round_ms=None, committed=None):
        assert round_ms is not None and round_ms >= 0 and committed >= 1
        self.seen.append((cap_used, committed))
        self.rounds += 1
        self.cap = 1 if self.rounds % self.probe_every == 0 else 0


def test_parked_sprint_clock_changes_no_token_and_no_round():
    # CapController.parked_sync_clock only moves WHEN a parked step's clock is read (at
    # scheduling, or once its token is on the host): same tokens, same rounds, same budget
    t = Target(_tiny_dense(), tokenizer=None)
    d = _tiny_dspark(32)
    budget = 14
    ref = greedy_generate(t, _Tok(), prompt_ids=PROMPT, max_new_tokens=budget).token_ids
    runs = []
    for sync in (False, True):
        ctrl = _ParkedController(sync)
        res = speculative_generate(t, _Tok(), d, prompt_ids=PROMPT, max_new_tokens=budget,
                                   max_draft_tokens=None, cap_controller=ctrl,
                                   lookup_drafts=False)
        assert res.token_ids == ref and len(res.token_ids) == budget
        assert any(cap == 0 for cap, _ in ctrl.seen) and any(cap == 1 for cap, _ in ctrl.seen)
        runs.append(ctrl.seen)
    assert runs[0] == runs[1]


def test_lookup_generate_returns_exactly_the_budget():
    t = Target(_tiny_dense(), tokenizer=None)
    # a repetitive prompt so the n-gram index fires copy drafts longer than the budget
    prompt = [3, 17, 42, 5, 9, 33, 21] * 4 + [3, 17, 42, 5]
    for budget in (1, 2, 4):
        ref = greedy_generate(t, _Tok(), prompt_ids=prompt, max_new_tokens=budget).token_ids
        res = lookup_generate(t, _Tok(), prompt_ids=prompt, max_new_tokens=budget, max_draft_tokens=6)
        assert len(res.token_ids) == budget == res.num_tokens
        assert res.token_ids == ref


def test_row_tracker_commit_stops_at_each_rows_budget():
    tr = _RowTracker(_Tok(), first_tokens=[5, 6], max_new_tokens=[3, 10])
    tr.commit(0, [1, 2, 3, 4, 5])                    # 1 first token + 5 committed > budget 3
    tr.commit(1, [1, 2, 3, 4, 5])
    assert tr.out_ids[0] == [5, 1, 2] and tr.finished[0]
    assert tr.out_ids[1] == [6, 1, 2, 3, 4, 5] and not tr.finished[1]
    res = tr.results(1.0)
    assert res[0].num_tokens == 3 and res[0].finish_reason == "length"
    mx.eval(mx.zeros(1))                              # keep the test's device use trivial
