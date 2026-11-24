"""Tree verify on Inkling 2's fused forward (inkling_fused "tree verify", Target.verify_tree /
commit_tree). Model-free: the tiny looped Inkling 2 with random bf16 weights (the fused pieces'
own fixture: an 8-token sliding window, relative positions to 8, log scaling from position 4)."""
import mlx.core as mx
import pytest
from test_inkling_batch import TAP, _target, _tiny_dflash2

from mlx_dspark import inkling_fused as F
from mlx_dspark.tree import DraftTree, chain_tree

metal = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal kernels")
PROMPT = mx.array([[3, 7, 11, 2, 9, 30, 41, 5, 8, 13, 21, 34]])        # past the 8-token window
#                                             rows:  0   1   2   3   4   5   6   7
TREE = DraftTree(tokens=[14, 22, 35, 40, 44, 51, 17, 6], parent=[-1, 0, 1, 2, 0, 4, 1, 6])


def _fresh(t):
    c = t.make_cache()
    t.prefill(PROMPT, c, TAP)
    return c


def _settled(t):
    """A prefilled cache after one aborted CHAIN verify: a sliding window stops showing the rows
    no later call can see once a verify has run over it (aborted or not), so cache states are
    compared from here on."""
    c = _fresh(t)
    mx.eval(t.verify(mx.array([[50, 51, 52, 53, 54, 55, 56, 57]]), c, TAP)[0])
    t.reset_spec()
    return c


def _states(cache):
    out = [mx.contiguous(a) for a in F._state_arrays([part for entry in cache
                                                      for part in entry.caches])]
    mx.eval(out)
    return out


def _same(xs, ys) -> bool:
    return len(xs) == len(ys) and all(
        x.shape == y.shape and bool(mx.array_equal(x, y)) for x, y in zip(xs, ys))


@metal
def test_the_tree_kernels_hold_on_the_models_own_layers_and_the_target_takes_trees():
    t, lm = _target()
    assert t.supports_tree and F.tree_supported(lm)
    assert F.tree_selftest(lm)
    assert t.tree_ready() and t._tree_selftest_worst < 1e-2
    assert t._tree_conv_k() == 4 and t.tree_max_nodes == 31          # tree.TREE_MAX_NODES
    F.set_rows(lm, F.VECTOR_ROWS)
    assert t.tree_max_nodes == F.VECTOR_ROWS - 1                     # no wider than a call takes
    assert t._txn is None and F._TREE is None


@metal
def test_tree_rows_are_the_chain_rows_of_their_root_paths():
    t, _ = _target()
    c = _settled(t)
    before = _states(c)
    logits, fused = t.verify_tree(TREE, c, TAP)
    assert logits.shape[:2] == (1, 8) and fused.shape == (1, 8, len(TAP) * 128)
    mx.eval(logits, fused)
    # every row against the chain verify of its own root path (that chain's last row)
    for row in range(1, 8):
        path = TREE.path_to(row)
        ref_c = _fresh(t)
        ref, ref_f = t.verify(mx.array([[TREE.tokens[r] for r in path]]), ref_c, TAP)
        mx.eval(ref, ref_f)
        t.reset_spec()
        assert mx.allclose(logits[0, row], ref[0, -1], atol=2e-2, rtol=2e-2), row
        assert mx.allclose(fused[0, row], ref_f[0, -1], atol=2e-2, rtol=2e-2), row
    # an abort puts every cache back, bit for bit
    t.abort_tree(c, TREE.width)
    assert t._txn is None and _same(_states(c), before)
    # a chain as a tree IS the chain verify
    toks = [14, 22, 35, 40, 44]
    got = t.verify_tree(chain_tree(toks[0], toks[1:]), c, TAP)
    mx.eval(*got)
    t.abort_tree(c, 5)
    ref = t.verify(mx.array([toks]), c, TAP)
    mx.eval(*ref)
    t.reset_spec()
    assert bool(mx.array_equal(got[0], ref[0])) and bool(mx.array_equal(got[1], ref[1]))
    assert _same(_states(c), before)


@metal
def test_a_row_outside_the_root_path_never_moves_a_row():
    t, _ = _target()
    c = _fresh(t)
    logits = t.verify_tree(TREE, c, TAP)[0]
    mx.eval(logits)
    t.abort_tree(c, TREE.width)
    # another token in row 3 (a leaf under 0-1-2): every other row keeps its bits
    other = DraftTree(tokens=TREE.tokens[:3] + [50] + TREE.tokens[4:], parent=TREE.parent)
    moved = t.verify_tree(other, c, TAP)[0]
    mx.eval(moved)
    t.abort_tree(c, TREE.width)
    keep = mx.array([0, 1, 2, 4, 5, 6, 7])
    assert bool(mx.array_equal(moved[0, keep], logits[0, keep]))
    assert not bool(mx.array_equal(moved[0, 3], logits[0, 3]))
    # ... and in row 1 (rows 2, 3, 6, 7 hang under it): rows 0, 4, 5 keep theirs
    other = DraftTree(tokens=[TREE.tokens[0], 50, *TREE.tokens[2:]], parent=TREE.parent)
    moved = t.verify_tree(other, c, TAP)[0]
    mx.eval(moved)
    t.abort_tree(c, TREE.width)
    keep = mx.array([0, 4, 5])
    assert bool(mx.array_equal(moved[0, keep], logits[0, keep]))


@metal
@pytest.mark.parametrize("path", [[0, 1, 2], [0, 4, 5], [0, 1, 6, 7], [0], [0, 1, 2, 3]])
def test_a_tree_commit_leaves_the_chain_committed_caches(path):
    t, _ = _target()
    a = _fresh(t)
    t.verify_tree(TREE, a, TAP)
    t.commit_tree(a, path)
    assert t._txn is None
    b = _fresh(t)
    toks = [TREE.tokens[r] for r in path]
    t.verify(mx.array([toks]), b, TAP)
    t.rollback(b, n_rejected=0, accepted=toks[1:])
    sa, sb = _states(a), _states(b)
    assert len(sa) == len(sb)
    for x, y in zip(sa, sb):
        assert x.shape == y.shape and mx.allclose(x, y, atol=2e-2, rtol=2e-2)
    # ... and decoding goes on from them alike: more committed steps, past the window again
    for token in (16, 23, 31, 45, 52):
        got, want = t.plain(mx.array([[token]]), a), t.plain(mx.array([[token]]), b)
        assert mx.allclose(got, want, atol=2e-2, rtol=2e-2)


@metal
def test_a_tree_verify_refuses_a_forward_that_leaves_the_fused_path():
    t, lm = _target()
    c = _settled(t)
    before = _states(c)
    tree = DraftTree(tokens=[14, 22, 35, 40], parent=[-1, 0, 0, 2])
    # one attention back on mlx-vlm's stock mask (a chain's): the round must not go through
    attn = lm.model.layers[3].self_attn
    attn.__dict__[F._MASK] = False
    assert not t.supports_tree
    with pytest.raises(RuntimeError, match="tree verify"):
        t.verify_tree(tree, c, TAP)
    assert t._txn is None and F._TREE is None and _same(_states(c), before)
    attn.__dict__[F._MASK] = True
    # one attention body on mlx-vlm's stock path (a chain's mask, its K / V rows unrecorded):
    # caught by the round's count, whatever else that visit recorded
    attn.__dict__[F._FLAG] = False
    with pytest.raises(RuntimeError, match="left the fused tree path"):
        t.verify_tree(tree, c, TAP)
    assert t._txn is None and _same(_states(c), before)
    attn.__dict__[F._FLAG] = True
    F.set_enabled(lm, False, ("layer",))
    assert not t.supports_tree
    F.set_enabled(lm, True, ("layer",))
    assert t.supports_tree
    t.verify_tree(tree, c, TAP)
    with pytest.raises(ValueError, match="root path"):
        t._txn.commit([1, 2])
    t.commit_tree(c, [0, 2, 3])
    with pytest.raises(RuntimeError, match="without a preceding verify_tree"):
        t.commit_tree(c, [0])
    # reset_spec aborts a tree verify left open, like a chain's
    c = _settled(t)
    before = _states(c)
    t.verify_tree(tree, c, TAP)
    t.reset_spec()
    assert t._txn is None and _same(_states(c), before)


@metal
def test_dflash_generate_with_trees_reproduces_greedy_decoding():
    from mlx_dspark.generate import dflash_generate, greedy_generate

    t, _ = _target()
    d = _tiny_dflash2(t)
    prompt = [4, 6, 12, 19, 27, 33, 1, 14, 22, 35, 40, 44, 51, 17]
    ref = greedy_generate(t, t.tokenizer, prompt_ids=prompt, max_new_tokens=40).token_ids
    branched = 0
    for cap in (None, 6, 3):
        chain = dflash_generate(t, t.tokenizer, d, prompt_ids=prompt, max_new_tokens=40,
                                max_draft_tokens=cap)
        tree = dflash_generate(t, t.tokenizer, d, prompt_ids=prompt, max_new_tokens=40,
                               max_draft_tokens=cap, tree=0)
        assert chain.token_ids == ref[: len(chain.token_ids)] and chain.tree_rounds == 0, cap
        assert tree.token_ids == ref[: len(tree.token_ids)], cap
        assert len(tree.token_ids) == len(chain.token_ids)
        assert tree.num_rounds <= chain.num_rounds, cap      # same rows a round, never fewer tokens
        branched += tree.tree_rounds
    assert branched > 0                      # random selector scores: the trees do branch
    assert t._txn is None


@metal
def test_without_its_fused_pieces_the_loop_keeps_chains():
    from mlx_dspark.generate import dflash_generate, greedy_generate

    t, lm = _target()
    F.disable(lm, ("attn",))
    assert not t.supports_tree and not t.tree_ready()
    d = _tiny_dflash2(t)
    prompt = [4, 6, 12, 19, 27, 33, 1, 14, 22, 35, 40, 44, 51, 17]
    ref = greedy_generate(t, t.tokenizer, prompt_ids=prompt, max_new_tokens=16).token_ids
    res = dflash_generate(t, t.tokenizer, d, prompt_ids=prompt, max_new_tokens=16, tree=0)
    assert res.token_ids == ref[: len(res.token_ids)] and res.tree_rounds == 0
