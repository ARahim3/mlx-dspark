"""Model-free tests for the native-MTP drafter.

The properties worth pinning here are not "does it run" — they are the two that decide
whether MTP can share the prefix cache at all:

  1. the context is a **pure function of the committed prefix** (same tokens in, same
     rows out, regardless of how the calls were chunked), and
  2. the context **trims exactly**, so a restore at an earlier position is the state a
     cold run would have produced.

Both are what ``prefix_cache`` already assumes of the DSpark drafter context. If either
breaks, MTP still generates correct output (the target verifies every token) — it just
silently stops reusing prompts, which is the whole reason to prefer it over a runtime
that restores from a block boundary. So these are the tests that would catch the
regression that matters.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_dspark.model import CtxCache
from mlx_dspark.mtp_model import MTPConfig, MTPDrafter

H = 32
VOCAB = 61


def _cfg() -> MTPConfig:
    return MTPConfig(hidden_size=H, num_attention_heads=4, num_key_value_heads=2,
                     head_dim=8, intermediate_size=64, max_depth=3)


def _drafter(seed: int = 0) -> MTPDrafter:
    mx.random.seed(seed)
    d = MTPDrafter(_cfg())
    mx.eval(d.parameters())
    # stand-ins for the target's embed_tokens / lm_head, which MTP does not ship
    emb = mx.random.normal((VOCAB, H))
    head = mx.random.normal((H, VOCAB))
    mx.eval(emb, head)
    d.bind_embed(lambda ids: emb[ids])
    d.bind_lm_head(lambda h: h @ head)
    return d


def _hiddens(n: int, seed: int = 7) -> mx.array:
    mx.random.seed(seed)
    h = mx.random.normal((1, n, H))
    mx.eval(h)
    return h


def test_context_length_is_one_behind_the_committed_tokens():
    """MTP row ``t`` fuses ``(hidden_t, embed(x_{t+1}))``, so the newest hidden has no
    partner yet. Pinning the ``N-1`` invariant matters because the prefix cache stores
    the context next to a token count: if the two ever disagree by a variable amount, a
    restore silently reuses rows that belong to a different position."""
    d = _drafter()
    ctx = d.make_ctx_cache()
    tokens = [3, 9, 14, 2, 55]
    d.update_context(_hiddens(len(tokens)), 0, ctx, token_ids=tokens)
    assert ctx[0].length == len(tokens) - 1 == d.ctx_len_for(len(tokens))


def test_chunked_prefill_matches_one_shot():
    """The prefill feeds the drafter in chunks so long prompts never materialise every
    fused state at once, and a resumed request feeds the suffix alone. Both must land on
    the byte-identical context a single call would have built — otherwise reuse changes
    the draft distribution, which shows up as a quiet acceptance drop rather than a
    failure."""
    tokens = [5, 12, 40, 7, 33, 21, 8]
    hid = _hiddens(len(tokens))

    one = _drafter()
    ctx_one = one.make_ctx_cache()
    one.update_context(hid, 0, ctx_one, token_ids=tokens)

    split = _drafter()
    ctx_split = split.make_ctx_cache()
    cut = 3
    split.update_context(hid[:, :cut, :], 0, ctx_split, token_ids=tokens[:cut])
    split.update_context(hid[:, cut:, :], cut, ctx_split, token_ids=tokens[cut:])

    assert ctx_split[0].length == ctx_one[0].length
    mx.eval(ctx_one[0].k, ctx_split[0].k)
    assert mx.allclose(ctx_one[0].k, ctx_split[0].k, atol=1e-5)
    assert mx.allclose(ctx_one[0].v, ctx_split[0].v, atol=1e-5)


def test_trim_to_equals_a_shorter_prefill():
    """Trimming back to a shared prefix must equal having only ever seen that prefix —
    this is the exactness claim ``prefix_cache`` makes when it reuses a conversation, and
    it holds only because the MTP layer is full attention (position-local K/V rows) and
    not one of the trunk's recurrent layers."""
    tokens = [4, 18, 27, 6, 11, 44]
    hid = _hiddens(len(tokens))
    keep = 3

    full = _drafter()
    ctx_full = full.make_ctx_cache()
    full.update_context(hid, 0, ctx_full, token_ids=tokens)
    ctx_full[0].trim_to(keep)

    short = _drafter()
    ctx_short = short.make_ctx_cache()
    short.update_context(hid[:, : keep + 1, :], 0, ctx_short, token_ids=tokens[: keep + 1])

    assert ctx_short[0].length == keep
    mx.eval(ctx_full[0].k, ctx_short[0].k)
    assert mx.allclose(ctx_full[0].k, ctx_short[0].k, atol=1e-5)
    assert mx.allclose(ctx_full[0].v, ctx_short[0].v, atol=1e-5)


def test_context_snapshot_shape_matches_what_prefix_cache_stores():
    """``prefix_cache._snapshot`` stores a drafter context as ``[(c.k, c.v), …]`` and
    restores by trimming. Assert the MTP context is that same shape so it rides the
    existing machinery instead of needing a second cache format — the difference between
    a drafter that joins prefix reuse and one that has to disable it."""
    d = _drafter()
    ctx = d.make_ctx_cache()
    d.update_context(_hiddens(4), 0, ctx, token_ids=[1, 2, 3, 4])
    assert all(isinstance(c, CtxCache) for c in ctx)
    snap = [(c.k, c.v) for c in ctx]
    assert len(snap) == 1 and all(k is not None and v is not None for k, v in snap)


def test_drafting_leaves_the_committed_context_untouched():
    """Draft rows are speculative: they are appended to the same cache so each depth can
    attend to the ones before it, then trimmed. If they survived, the context would
    depend on tokens the target never accepted — the state would stop being a function
    of the committed prefix and every later restore would be wrong."""
    d = _drafter()
    ctx = d.make_ctx_cache()
    tokens = [2, 19, 31, 5]
    d.update_context(_hiddens(len(tokens)), 0, ctx, token_ids=tokens)
    before_len = ctx[0].length
    before_k = ctx[0].k

    draft, q = d.draft_block(pending=7, n_cached=len(tokens), ctx_caches=ctx, cap=3)
    mx.eval(draft)

    assert draft.shape == (3,)
    assert q is None                       # greedy path proposes deterministically
    assert ctx[0].length == before_len
    assert mx.allclose(ctx[0].k, before_k, atol=1e-6)


def test_draft_respects_cap_and_max_depth():
    """The cap is the loop's per-round budget and ``max_depth`` the head's ceiling;
    drafting past either wastes verify width on tokens measured to accept ~18% of the
    time. Speed-only — the target still verifies whatever is proposed."""
    d = _drafter()
    ctx = d.make_ctx_cache()
    d.update_context(_hiddens(3), 0, ctx, token_ids=[1, 2, 3])
    assert d.draft_block(4, 3, ctx, cap=1)[0].shape == (1,)
    assert d.draft_block(4, 3, ctx, cap=99)[0].shape == (d.config.max_depth,)


def test_sampled_drafts_report_their_proposal_distribution():
    """``temperature > 0`` uses the paper's accept rule ``min(1, p/q)``, so the loop needs
    the same ``q`` the token was drawn from. Returning ``q`` from the rollout — rather
    than letting the caller re-derive it — is what keeps sampling lossless at depth."""
    d = _drafter()
    ctx = d.make_ctx_cache()
    d.update_context(_hiddens(3), 0, ctx, token_ids=[1, 2, 3])
    draft, q = d.draft_block(4, 3, ctx, cap=3, temperature=0.8)
    mx.eval(draft, q)
    assert q.shape == (3, VOCAB)
    sums = q.sum(axis=-1)
    assert mx.allclose(sums, mx.ones_like(sums), atol=1e-4)


def test_context_requires_tokens():
    """The DSpark drafter's context is built from hidden states alone; MTP additionally
    needs the next token at each position. Failing loudly beats silently fusing against
    a zero embedding, which would draft plausibly and accept far less."""
    d = _drafter()
    ctx = d.make_ctx_cache()
    with pytest.raises(ValueError):
        d.update_context(_hiddens(3), 0, ctx, token_ids=None)
