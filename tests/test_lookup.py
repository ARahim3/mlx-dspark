"""Model-free tests for prompt-lookup drafting: the n-gram index semantics."""

from __future__ import annotations

from mlx_dspark.lookup import LongDraftGate, NGramIndex


def _idx(tokens, **kw):
    ix = NGramIndex(**kw)
    ix.extend(list(tokens))
    return ix


def test_default_minimum_is_trigram():
    # bigram-only repetition must NOT fire at the defaults (bigrams are spurious on chat
    # text and rejected drafts cost wider forwards — measured net-negative on M-series)
    ix = _idx([5, 6, 7, 0, 1, 2, 5, 6])
    assert ix.propose() == []
    # a trigram match does fire
    ix = _idx([4, 5, 6, 7, 8, 0, 4, 5, 6], max_draft=2)
    assert ix.propose() == [7, 8]


def test_no_match_returns_empty():
    assert _idx([1, 2, 3, 4]).propose() == []


def test_simple_repeat_proposes_continuation():
    # ... 5 6 7 8 ... 5 6 -> propose [7, 8, ...] (bigram matching enabled explicitly)
    ix = _idx([5, 6, 7, 8, 9, 1, 5, 6], min_n=2, max_n=3, max_draft=3)
    assert ix.propose() == [7, 8, 9]


def test_self_occurrence_is_skipped():
    # the current suffix IS in the index (it was just inserted) — must not match itself
    ix = _idx([1, 2, 3, 4, 5], min_n=2, max_n=3)   # suffix (4,5) occurs only as the suffix
    assert ix.propose() == []


def test_latest_earlier_occurrence_wins():
    # (5,6) appears twice earlier with different continuations; the most recent wins
    ix = _idx([5, 6, 7, 0, 5, 6, 8, 0, 5, 6], min_n=2, max_n=3, max_draft=1)
    assert ix.propose() == [8]


def test_longer_ngram_preferred():
    # trigram (4,5,6) -> 9 is more specific than bigram (5,6) -> 7
    ix = _idx([4, 5, 6, 9, 0, 5, 6, 7, 0, 4, 5, 6], min_n=2, max_n=3, max_draft=1)
    assert ix.propose() == [9]


def test_max_draft_and_tail_truncation():
    ix = _idx([5, 6, 7, 8, 9, 10, 11, 0, 5, 6], min_n=2, max_n=3, max_draft=4)
    assert ix.propose() == [7, 8, 9, 10]
    # continuation runs into the end of the sequence -> shorter draft, never empty-pads
    ix2 = _idx([1, 9, 9, 2, 9, 9], min_n=2, max_n=3, max_draft=4)
    assert ix2.propose() == [2, 9, 9]


def test_incremental_extend_matches_bulk():
    toks = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 9]
    bulk = _idx(toks, min_n=2, max_n=3, max_draft=2)
    inc = NGramIndex(min_n=2, max_n=3, max_draft=2)
    for t in toks:
        inc.extend([t])
    assert bulk.propose() == inc.propose()


def test_weak_hit_stays_at_max_draft_even_with_long_draft():
    # a bare n-gram hit with no backward extension (matched context < 8) must NOT earn a
    # long draft — this is the llama.cpp ngram-mod lesson: long drafts need long evidence
    ix = _idx([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 5, 6],
              min_n=2, max_n=3, max_draft=2)
    assert ix.propose() == [7, 8]
    assert ix.propose(long_draft=32) == [7, 8]    # match is only the 2-gram itself


def test_deep_match_earns_scaled_draft():
    # earlier occurrence: A = [1..12] followed by continuation [50..]; suffix repeats the
    # last 8 tokens of A -> backward match m=8 -> draft 2*m=16 (clipped by history/ceiling)
    a = list(range(1, 13))                        # 12 tokens
    cont = list(range(50, 90))                    # 40-token continuation
    toks = a + cont + [0] + a[-8:]                # suffix = last 8 tokens of A
    ix = _idx(toks, min_n=4, max_n=5, max_draft=6)
    assert ix.propose() == cont[:6]               # no long_draft: base behavior
    assert ix.propose(long_draft=32) == cont[:16]         # m=8 -> 16
    assert ix.propose(long_draft=12) == cont[:12]         # ceiling clips
    assert ix.propose(long_draft=6) == cont[:6]           # ceiling == base: disabled


def test_full_backward_match_hits_ceiling():
    # suffix repeats a long earlier span verbatim -> m reaches the ceiling walk cap and
    # the draft is clipped to the ceiling
    a = list(range(1, 25))                        # 24 tokens
    cont = list(range(100, 164))                  # 64-token continuation
    toks = a + cont + [0] + a                     # the whole of A repeats
    ix = _idx(toks, min_n=4, max_n=5, max_draft=6)
    assert ix.propose(long_draft=32) == cont[:32]

    # the backward walk must stop at the start of the sequence (j == -1) without error
    ix2 = _idx(a + cont + a[:0] + a, min_n=4, max_n=5, max_draft=6)
    assert ix2.propose(long_draft=64) == cont[:48]  # m capped by sequence start (m=24) -> 48


def test_gate_parks_after_two_chopped_long_drafts_and_probes_back():
    g = LongDraftGate(probe_every=3)
    assert g.allowed
    g.update(drafted=6, accepted=1, base=6)       # base-length rounds never count as fails
    assert g.allowed
    g.update(drafted=24, accepted=5, base=6)      # long draft chopped early (< half)
    assert g.allowed                              # one strike is forgiven
    g.update(drafted=20, accepted=2, base=6)      # second strike -> parked
    assert not g.allowed
    for _ in range(2):                            # base rounds tick toward the probe
        g.update(drafted=6, accepted=6, base=6)
    assert not g.allowed
    g.update(drafted=6, accepted=6, base=6)
    assert g.allowed                              # probe window open
    g.update(drafted=28, accepted=28, base=6)     # probe succeeds -> unparked
    assert g.allowed
    g.update(drafted=16, accepted=8, base=6)      # >= half accepted counts as success
    assert g.allowed
