"""Speculative-sampling rounds as one graph with one sync (``generate.SAMPLED_ONE_SYNC``): the
in-graph accept (``_spec_sample_accept_arr``) makes the random draws of the host-side accept in
the same order over the same values, so a seed gives the same tokens either way — against the
historical accept, and through both generate loops with the switch on and off."""

from __future__ import annotations

import mlx.core as mx
import pytest
from test_dspark_tree import _tiny_dspark
from test_tree import PROMPT, _tiny_dense, _tiny_dflash2_for, _tiny_hybrid, _Tok

from mlx_dspark import generate as g
from mlx_dspark.sampling import sample_probs, truncate_probs
from mlx_dspark.target import Target


def _historical_accept(v_logits, draft, q_probs, temperature, top_p=1.0, top_k=0):
    """The accept as it ran before the in-graph version: the accepted length synced first, then
    the residual or the bonus drawn on the host's branch."""
    L = len(draft)
    p = truncate_probs(mx.softmax(v_logits / temperature, axis=-1), top_p, top_k)
    rows = mx.arange(L)
    idx = mx.array(draft)
    pd = p[rows, idx]
    qd = q_probs[rows, idx]
    u = mx.random.uniform(shape=(L,))
    accepted = u < mx.minimum(1.0, pd / mx.maximum(qd, 1e-9))
    n = int(mx.cumprod(accepted.astype(mx.int32)).sum().item())
    if n < L:
        resid = mx.maximum(p[n] - q_probs[n], 0.0)
        resid = resid / mx.maximum(resid.sum(), 1e-9)
        repl = int(mx.random.categorical(mx.log(resid + 1e-20)).item())
    else:
        repl = int(sample_probs(p[L]).item())
    return n, repl


@pytest.mark.parametrize("T,top_p,top_k", [(1.0, 1.0, 0), (0.7, 0.9, 0), (1.3, 1.0, 5)])
def test_in_graph_accept_equals_the_host_accept_draw_for_draw(T, top_p, top_k):
    V, seen = 48, set()
    for case in range(60):
        L = 1 + case % 5
        mx.random.seed(1000 + case)
        v_logits = mx.random.normal((L + 1, V)) * 2
        # a proposal close to the target (long accepted runs) or far from it (early rejects)
        noise = mx.random.normal((L + 2, V)) * (0.2 if case % 2 else 3.0)
        q = truncate_probs(mx.softmax((mx.concatenate([v_logits, v_logits[:1]]) + noise) / T,
                                      axis=-1), top_p, top_k)       # more rows than drafts
        draft = [int(x) for x in sample_probs(q[:L]).tolist()]
        mx.eval(v_logits, q)
        mx.random.seed(case)
        ref = _historical_accept(v_logits, draft, q, T, top_p, top_k)
        mx.random.seed(case)
        got = g._spec_sample_accept(v_logits, draft, q, T, top_p, top_k)
        mx.random.seed(case)
        n_arr, repl_arr = g._spec_sample_accept_arr(v_logits, mx.array(draft), q, T, top_p, top_k)
        assert got == ref and (int(n_arr.item()), int(repl_arr.item())) == ref
        seen.add("all" if ref[0] == L else "reject")
    assert seen == {"all", "reject"}                 # both the bonus and the residual draw ran


def _both(monkeypatch, run):
    """``run()`` with the two-phase round on the historical accept, then the one-sync round."""
    with monkeypatch.context() as m:                 # its own scope: other patches stay
        m.setattr(g, "SAMPLED_ONE_SYNC", False)
        m.setattr(g, "_spec_sample_accept", _historical_accept)
        two_phase = run()
    assert g.SAMPLED_ONE_SYNC
    return two_phase, run()


def _tiny_dflash1_for(target):
    """A plain DFlash head (no selector): the loop samples from its softmax rows."""
    from mlx_dspark.dflash_model import DFlashConfig, DFlashDraftModel

    cfg = DFlashConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, intermediate_size=64, vocab_size=64, rms_norm_eps=1e-6, rope_theta=1e7,
        max_position_embeddings=512, block_size=8, target_layer_ids=(0, 1),
        num_target_layers=2, mask_token_id=3, selector_rank=0, selector_top_k=0,
        conv_kernel_size=0, conv_group_size=16,
    )
    mx.random.seed(22)
    d = DFlashDraftModel(cfg)
    mx.eval(d.parameters())
    d.bind(target.model)
    return d


@pytest.fixture
def copy_hits(monkeypatch):
    """Every third lookup proposal is a 3-token draft (a random model never copies its prompt,
    so the n-gram index alone would not fire)."""
    from mlx_dspark.lookup import NGramIndex

    def propose(self, long_draft=None):
        # counted on the index itself: each generation builds its own, and an id() can be
        # reused by the next one
        n = self.__dict__["_proposals"] = self.__dict__.get("_proposals", 0) + 1
        return [5, 9, 33] if n % 3 == 0 else []

    monkeypatch.setattr(NGramIndex, "propose", propose)


@pytest.mark.parametrize("make,hidden", [(_tiny_dense, 32), (_tiny_hybrid, 32)])
@pytest.mark.parametrize("kw", [
    {"temperature": 1.0},
    {"temperature": 0.7, "top_p": 0.9},
    {"temperature": 1.2, "top_k": 8},
    {"temperature": 1.0, "lookup_drafts": True},
    {"temperature": 1.0, "lookup_drafts": True, "presence_penalty": 0.5},   # stays two-phase
])
def test_sampled_dspark_round_gives_the_same_tokens_in_one_sync(monkeypatch, copy_hits, make,
                                                                hidden, kw):
    t = Target(make(), tokenizer=None)
    d = _tiny_dspark(hidden)
    args = {"prompt_ids": PROMPT, "max_new_tokens": 48, "max_draft_tokens": 4,
            "lookup_drafts": False, "seed": 7, **kw}
    a, b = _both(monkeypatch, lambda: g.speculative_generate(t, _Tok(), d, **args))
    assert a.token_ids == b.token_ids and a.accept_lengths == b.accept_lengths
    assert len(a.token_ids) > 1
    assert a.lookup_rounds == b.lookup_rounds and bool(a.lookup_rounds) == bool(
        kw.get("lookup_drafts"))
    lens = set(a.accept_lengths)
    assert 1 in lens and max(lens) > 1               # early rejects and accepted runs both ran


class _Parked:
    """A cap controller that parks speculation: every round is one plain sampled step."""

    cap = 0
    probe_every = 8

    def set_depth(self, depth):
        pass

    def update(self, *a, **k):
        pass


def test_parked_sampled_steps_give_the_same_tokens(monkeypatch):
    t = Target(_tiny_dense(), tokenizer=None)
    d = _tiny_dspark(32)
    args = {"prompt_ids": PROMPT, "max_new_tokens": 24, "max_draft_tokens": None,
            "lookup_drafts": False, "seed": 3, "temperature": 0.9, "top_p": 0.95}
    ref = g.greedy_generate(t, _Tok(), prompt_ids=PROMPT, max_new_tokens=24, seed=3,
                            temperature=0.9, top_p=0.95)
    parked = g.speculative_generate(t, _Tok(), d, cap_controller=_Parked(), **args)
    assert parked.accept_lengths == [1] * (len(parked.token_ids) - 1)
    # one categorical draw per token, from the same rows as plain sampling
    assert parked.token_ids == ref.token_ids


@pytest.mark.parametrize("make", [_tiny_dense, _tiny_hybrid])
@pytest.mark.parametrize("head", [_tiny_dflash2_for, _tiny_dflash1_for])   # selector walk; softmax
@pytest.mark.parametrize("kw", [
    {"temperature": 1.0},
    {"temperature": 0.7, "top_p": 0.9, "max_draft_tokens": 3},
    {"temperature": 1.0, "lookup_drafts": True},
])
def test_sampled_dflash_round_gives_the_same_tokens_in_one_sync(monkeypatch, copy_hits, make,
                                                                head, kw):
    t = Target(make(), tokenizer=None)
    drafter = head(t)
    args = {"prompt_ids": PROMPT, "max_new_tokens": 48, "seed": 5, **kw}
    a, b = _both(monkeypatch, lambda: g.dflash_generate(t, _Tok(), drafter, **args))
    assert a.token_ids == b.token_ids and a.accept_lengths == b.accept_lengths
    assert len(a.token_ids) > 1
    assert a.lookup_rounds == b.lookup_rounds and bool(a.lookup_rounds) == bool(
        kw.get("lookup_drafts"))
