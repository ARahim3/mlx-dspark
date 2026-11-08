"""DSpark trees: ``DSparkDrafter.tree_lattice`` (the drafter's own chain plus Markov-scored side
candidates, in ``device_tree``'s shape) and the tree round in ``speculative_generate``. Lossless
like the DFlash 2 tree — the target decides every token, so the output is greedy's — and the
chain never loses a token to the lattice (candidate 0 of every slot IS the chain)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from test_tree import PROMPT, _tiny_dense, _tiny_hybrid, _tiny_hybrid_kernel, _Tok

from mlx_dspark.calibrate import static_tree, tree_capable
from mlx_dspark.config import DSparkConfig
from mlx_dspark.generate import greedy_generate, speculative_generate
from mlx_dspark.model import DSparkDrafter
from mlx_dspark.target import Target
from mlx_dspark.tree import best_first_tree, device_tree


def _tiny_dspark(hidden: int, markov_rank: int = 4, seed: int = 23) -> DSparkDrafter:
    cfg = DSparkConfig(family="qwen3", hidden_size=hidden, vocab_size=64, num_hidden_layers=1,
                       intermediate_size=2 * hidden, num_attention_heads=2,
                       num_key_value_heads=1, head_dim=8, attention_k_eq_v=False, rope_theta=1e4,
                       rope_type="default", block_size=5, mask_token_id=63,
                       target_layer_ids=[0, 1], markov_rank=markov_rank,
                       enable_confidence_head=False, final_logit_softcapping=None,
                       mlp_activation="silu", norm_style="qwen", use_v_norm=False)
    mx.random.seed(seed)
    d = DSparkDrafter(cfg)
    mx.eval(d.parameters())
    return d


# ---------------------------------------------------------------- the lattice


@pytest.mark.parametrize("markov_rank", [4, 0])
def test_tree_lattice_is_base_plus_markov_bias_with_the_chain_as_candidate_zero(markov_rank):
    d = _tiny_dspark(32, markov_rank=markov_rank)
    mx.random.seed(1)
    base = mx.random.normal((4, 64)) * 2
    anchor = 7
    chain = d.sample_block(base, first_prev_token=anchor)
    cand, scores = d.tree_lattice(base, chain, anchor, k=6)
    mx.eval(cand, scores)
    assert cand.shape == (4, 6) and scores.shape == (4, 6, 6) and cand.dtype == mx.int32
    assert cand[:, 0].tolist() == chain.tolist()                 # the chain leads every slot
    for s in range(4):
        row = cand[s].tolist()
        assert len(set(row)) == 6                                 # distinct candidates
        parents = [anchor] if s == 0 else cand[s - 1].tolist()    # slot 0: row 0 = the anchor
        for j, p in enumerate(parents):
            bias = (d.markov_head.step_bias(mx.array([p]))[0] if d.markov_head is not None
                    else mx.zeros((64,)))
            ref = (base[s] + bias)[mx.array(row)]
            assert mx.allclose(scores[s, j], ref, atol=1e-4, rtol=1e-4).item()
        # the chain token is its parent's argmax over the full vocab, hence over the lattice
        assert int(mx.argmax(scores[s, 0])) == 0
    # so the best-first tree lays the chain out as its main path (rows 1..d), host and device
    host = best_first_tree(np.array(cand), np.array(scores), 4, anchor)
    main = [host.tokens[r] for r in range(1, host.width) if host.parent[r] == r - 1]
    assert main and main == chain.tolist()[: len(main)]
    dev = device_tree(cand, scores, 4, anchor, 4)
    assert dev.tokens.tolist() == host.tokens and dev.parent.tolist() == host.parent


def test_tree_lattice_maps_a_reduced_vocab_head_both_ways():
    d = _tiny_dspark(32)
    rng = np.random.default_rng(3)
    perm = rng.permutation(64).astype(np.int32)                 # draft id i -> target id perm[i]
    d.set_draft_vocab_map(mx.array(perm - np.arange(64, dtype=np.int32)))
    ids = mx.array([5, 17, 63, 0], dtype=mx.int32)
    assert d._to_draft_ids(d.to_target_ids(ids)).tolist() == ids.tolist()
    mx.random.seed(2)
    base = mx.random.normal((3, 64))
    chain = d.sample_block(base, first_prev_token=int(perm[9]))   # target ids
    cand, scores = d.tree_lattice(base, chain, int(perm[9]), k=5)
    mx.eval(cand, scores)
    assert cand[:, 0].tolist() == chain.tolist()
    assert all(len(set(cand[s].tolist())) == 5 for s in range(3))


# ---------------------------------------------------------------- the loop


@pytest.mark.parametrize("make,hidden",
                         [(_tiny_hybrid_kernel, 64), (_tiny_hybrid, 32), (_tiny_dense, 32)])
def test_speculative_generate_with_trees_reproduces_greedy(make, hidden):
    t = Target(make(), tokenizer=None)
    d = _tiny_dspark(hidden)
    kw = {"prompt_ids": PROMPT, "max_new_tokens": 40, "max_draft_tokens": 4,
          "lookup_drafts": False}
    ref = greedy_generate(t, _Tok(), prompt_ids=PROMPT, max_new_tokens=40).token_ids
    chain = speculative_generate(t, _Tok(), d, **kw)
    assert chain.token_ids[:40] == ref[:40] and chain.tree_rounds == 0
    branched = 0
    for tree in (0, 11):                            # equal width; a wide tree
        res = speculative_generate(t, _Tok(), d, tree=tree, **kw)
        assert res.token_ids[:40] == ref[:40]
        branched += res.tree_rounds
    assert branched > 0                             # a random head is unsure: trees branch
    assert t._pending is None                       # deferred commit flushed at the exit
    # sampling and logprobs keep chains (tree rounds are greedy-only, chain positions only)
    s = speculative_generate(t, _Tok(), d, tree=0, temperature=0.7, seed=1,
                             prompt_ids=PROMPT, max_new_tokens=8, max_draft_tokens=4,
                             lookup_drafts=False)
    assert s.tree_rounds == 0
    lp = speculative_generate(t, _Tok(), d, tree=0, logprobs=1, prompt_ids=PROMPT,
                              max_new_tokens=8, max_draft_tokens=4, lookup_drafts=False)
    assert lp.tree_rounds == 0 and lp.logprobs is not None


def test_tree_rounds_report_chain_positions_and_the_tree_source():
    t = Target(_tiny_hybrid_kernel(), tokenizer=None)
    d = _tiny_dspark(64)
    seen = []
    speculative_generate(t, _Tok(), d, prompt_ids=PROMPT, max_new_tokens=24, max_draft_tokens=4,
                         lookup_drafts=False, tree=11,
                         on_round=lambda **r: seen.append(r))
    assert seen and all(r["drafted"] <= 4 for r in seen)        # nodes != chain positions
    assert {r["source"] for r in seen} <= {"tree", "drafter"}


def test_static_tree_takes_dspark_heads(tmp_path):
    t = Target(_tiny_hybrid_kernel(), tokenizer=None)
    d = _tiny_dspark(64)
    assert tree_capable(d)
    got = static_tree(t, d, setting="auto", mode="dspark", target_repo="tiny/t",
                      drafter_repo="tiny/d", cache_dir=str(tmp_path))
    assert got == 0 and not any(tmp_path.iterdir())
    assert static_tree(t, d, setting="off", mode="dspark", target_repo="x",
                       drafter_repo="y") is None
