"""Best-first tree verify: tree.py, Target.verify_tree/commit_tree/tree_selftest, the tree
gated-delta kernel, the multirow tree mask, calibrate's tree pricing, and dflash_generate's
tree rounds end to end.

The load-bearing property is losslessness: a tree's every row must compute exactly what a
chain verify along that row's root path computes, and committing an accepted path must leave
the caches a chain commit of it would leave — so a tree round commits exactly the target's
greedy tokens. On the tiny fp32 models here those equalities are tight; the real-model
fp-tie class is covered by tree_selftest's tolerance and the token-forcing gate.
"""

from __future__ import annotations

import contextlib
import importlib
import itertools

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_dspark import tree_gdn
from mlx_dspark.calibrate import (
    measure_tree_curve,
    measure_tree_fallback_ratio,
    parse_tree_setting,
    static_tree,
    wide_tree_nodes,
)
from mlx_dspark.target import Target
from mlx_dspark.tree import (
    TREE_GAIN_PRIOR,
    TREE_LEAVES_PRIOR,
    DraftTree,
    best_first_tree,
    chain_tree,
    device_tree,
    prior_leaves,
    synthetic_tree,
    to_device,
    tree_gain,
)

_CAL = importlib.import_module("mlx_dspark.calibrate")   # the package attribute is the function

# ---------------------------------------------------------------- fixtures


class _Tok:
    """Just enough tokenizer for the generate loops: no eos ids, char-decode."""

    def decode(self, ids):
        return "".join(chr(65 + (i % 26)) for i in ids)


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


def _tiny_hybrid_kernel():
    """The same tiny hybrid in EVAL mode with 32-dim linear heads: mlx-lm runs the gated-delta
    Metal kernel (training-mode modules take the ops path) and tree_gdn.eligible accepts it —
    the path production takes."""
    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    args = ModelArgs.from_dict({
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "vocab_size": 64, "rms_norm_eps": 1e-6,
            "linear_num_value_heads": 4, "linear_num_key_heads": 2,
            "linear_key_head_dim": 32, "linear_value_head_dim": 32,
            "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
            "tie_word_embeddings": False,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                "partial_rotary_factor": 0.25},
        },
    })
    mx.random.seed(4)
    model = Model(args)
    model.eval()
    mx.eval(model.parameters())
    return model


def _tiny_dense():
    """A real (tiny, random) mlx-lm qwen3 model."""
    from mlx_lm.models.qwen3 import Model, ModelArgs

    args = ModelArgs(model_type="qwen3", hidden_size=32, num_hidden_layers=2,
                     intermediate_size=64, num_attention_heads=4, rms_norm_eps=1e-6,
                     vocab_size=64, num_key_value_heads=2, max_position_embeddings=512,
                     rope_theta=10000.0, head_dim=8, tie_word_embeddings=False)
    mx.random.seed(11)
    model = Model(args)
    mx.eval(model.parameters())
    return model


PROMPT = [3, 17, 42, 5, 9, 33, 21]
#               rows:  0  1  2   3   4   5          (0 = anchor)
BRANCHED = DraftTree(tokens=[7, 8, 9, 10, 11, 12], parent=[-1, 0, 1, 0, 3, 1])


def _prefilled(t, prompt=PROMPT):
    c = t.make_cache()
    t.run(mx.array([prompt]), c, [0])
    return c


# ---------------------------------------------------------------- DraftTree


def test_draft_tree_structure():
    t = BRANCHED
    assert t.depth == [0, 1, 2, 1, 2, 2]
    assert t.children == [[1, 3], [2, 5], [], [4], [], []]
    assert t.leaves() == [2, 4, 5]
    assert t.leaf_paths() == [[0, 1, 2], [0, 3, 4], [0, 1, 5]]
    assert t.path_to(4) == [0, 3, 4]
    assert not t.is_chain() and chain_tree(7, [8, 9]).is_chain()
    anc = t.ancestor_matrix()
    assert anc[4].tolist() == [True, False, False, True, True, False]
    assert anc[5].tolist() == [True, True, False, False, False, True]
    p_of, d_of = t.row_slots()
    for r in range(t.width):                       # each row lies on its slot's path
        assert t.leaf_paths()[p_of[r]][d_of[r]] == r


def test_draft_tree_rejects_non_topological_rows():
    with pytest.raises(ValueError):
        DraftTree(tokens=[1, 2, 3], parent=[-1, 2, 0])


def test_conv_index_is_the_sliding_window_on_a_chain_and_follows_paths_on_a_tree():
    chain = chain_tree(5, [1, 2, 3])
    assert chain.conv_index(4).tolist() == [[r, r + 1, r + 2, r + 3] for r in range(4)]
    # row 4 = path [0, 3, 4]: last 4 of [win0, win1, win2, 3+0, 3+3, 3+4]
    assert BRANCHED.conv_index(4)[4].tolist() == [2, 3, 6, 7]
    assert BRANCHED.conv_index(4)[0].tolist() == [0, 1, 2, 3]


def test_walk_accepts_along_the_matching_branch_and_takes_the_targets_bonus():
    t = BRANCHED
    nxt = [10, 0, 0, 11, 99, 0]               # root -> 10 (row 3) -> 11 (row 4) -> 99
    assert t.walk(nxt) == ([0, 3, 4], 99)
    assert t.walk([42, 0, 0, 0, 0, 0]) == ([0], 42)          # no child matches the root
    assert t.walk([8, 12, 0, 0, 0, 77]) == ([0, 1, 5], 77)   # second child of row 1


# ---------------------------------------------------------------- best-first builder


def _lsm(x):
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def _brute_top(cand, scores, n):
    """Every lattice path, scored by summed edge log-softmax; the top n as token tuples."""
    S, K = cand.shape
    lp0, lps = _lsm(scores[0, 0]), _lsm(scores)
    out = []
    for d in range(1, S + 1):
        for path in itertools.product(range(K), repeat=d):
            s = lp0[path[0]] + sum(lps[i, path[i - 1], path[i]] for i in range(1, d))
            out.append((s, tuple(int(cand[i, path[i]]) for i in range(d))))
    out.sort(key=lambda x: -x[0])
    return {p for _, p in out[:n]}


def test_best_first_equals_the_brute_force_top_n_and_is_prefix_closed():
    rng = np.random.default_rng(0)
    for _ in range(20):
        cand = rng.permutation(100)[:9].reshape(3, 3)
        scores = rng.normal(size=(3, 3, 3)) * 2
        for n in range(1, 13):
            t = best_first_tree(cand, scores, n, anchor=99)
            got = {tuple(t.tokens[r] for r in t.path_to(row)[1:]) for row in range(1, t.width)}
            assert got == _brute_top(cand, scores, n)
            assert all(t.parent[r] < r for r in range(1, t.width))     # topological


def test_best_first_lays_the_greedy_chain_out_first():
    rng = np.random.default_rng(1)
    for _ in range(30):
        cand = rng.permutation(1000)[:7 * 16].reshape(7, 16)
        scores = rng.normal(size=(7, 16, 16)) * 3
        t = best_first_tree(cand, scores, 7, anchor=5)
        # the greedy walk (argmax at each slot along the chain)
        idx, chain = int(np.argmax(scores[0, 0])), []
        chain.append(int(cand[0, idx]))
        for s in range(1, 7):
            idx = int(np.argmax(scores[s, idx]))
            chain.append(int(cand[s, idx]))
        main = []                                  # rows 1.. while each is the previous's child
        for r in range(1, t.width):
            if t.parent[r] != r - 1:
                break
            main.append(t.tokens[r])
        assert main and main == chain[:len(main)]


def test_device_tree_matches_the_host_builder():
    rng = np.random.default_rng(2)
    for _ in range(25):
        cand = rng.permutation(5000)[:7 * 16].reshape(7, 16).astype(np.int32)
        scores = (rng.normal(size=(7, 16, 16)) * rng.uniform(0.5, 4)).astype(np.float32)
        for n in (3, 7, 11, 15):
            host = best_first_tree(cand, scores, n, 99)
            dev = device_tree(mx.array(cand), mx.array(scores), n, 99, conv_k=4)
            ref = to_device(host, 4)
            assert dev.tokens.tolist() == host.tokens
            assert dev.parent.tolist() == host.parent
            assert dev.depth.tolist() == host.depth
            assert dev.conv_idx.tolist() == ref.conv_idx.tolist()
            assert dev.bits.tolist() == ref.bits.tolist()
            assert dev.to_host().tokens == host.tokens


# ---------------------------------------------------------------- settings + pricing


def test_parse_tree_setting():
    assert parse_tree_setting(None) == "auto"
    assert parse_tree_setting(True) == "on" and parse_tree_setting(False) == "off"
    assert parse_tree_setting(0) == "on" and parse_tree_setting(15) == 15
    assert parse_tree_setting(" OFF ") == "off" and parse_tree_setting("11") == 11
    for bad in ("sometimes", -1, "-3", 32, "64"):
        with pytest.raises(ValueError):
            parse_tree_setting(bad)


def test_tree_gain_prices_only_wider_trees_inside_the_measured_regime():
    assert tree_gain(7, 7) == 1.0 and tree_gain(5, 7) == 1.0
    assert tree_gain(15, 7) == pytest.approx(TREE_GAIN_PRIOR[15] / TREE_GAIN_PRIOR[7])
    assert tree_gain(15, 3) == 1.0               # a depth-shrunk cap is never widened
    assert tree_gain(64, 7) == pytest.approx(TREE_GAIN_PRIOR[31] / TREE_GAIN_PRIOR[7])


def test_wide_trees_stay_off_on_an_m4_shaped_curve():
    # measured Qwen3.8-27B-4bit staircase on an M4 Pro (NOTES "SpecPilot ..."), chain by width
    m4 = {2: 71.0, 4: 93.3, 8: 97.2, 9: 160.2, 12: 161.8, 16: 164.0, 24: 280.5, 32: 281.0}
    tree = {7: 97.7, 11: 162.0, 15: 164.5, 23: 281.0, 31: 282.0}     # tree verifies, by nodes
    assert wide_tree_nodes(m4, 12.5, 7, tree) == 0
    assert wide_tree_nodes(m4, 12.5, 7) == 0                 # nothing measured: never widen


def test_wide_trees_go_wide_on_a_flat_curve_and_price_only_measured_tree_verifies():
    flat = {2: 70.0, 8: 72.0, 12: 73.0}
    tree = {7: 72.5, 11: 73.5, 15: 74.0, 23: 75.0, 31: 76.0}
    assert wide_tree_nodes(flat, 12.0, 7, tree) == 31
    assert wide_tree_nodes(flat, 12.0, 7, {7: 72.5, 11: 73.5}) == 11   # 15/23/31 unmeasured
    # a chain's width-16 point never stands in for a 15-node tree (the fallback's leaves)
    assert wide_tree_nodes({**flat, 16: 74.0, 24: 75.0, 32: 76.0}, 12.0, 7, {7: 72.5}) == 0
    # the recurrence fallback pays per leaf: measured in, the pricing sees it and stays narrow
    fallback = {7: 74.0, 11: 80.0, 15: 85.0, 23: 95.0, 31: 108.0}
    assert wide_tree_nodes(flat, 12.0, 7, fallback) == 0


def test_leaves_prior_covers_the_gain_prior_and_grows_with_the_tree():
    assert set(TREE_GAIN_PRIOR) <= set(TREE_LEAVES_PRIOR)
    ks = sorted(TREE_LEAVES_PRIOR)
    assert all(TREE_LEAVES_PRIOR[a] < TREE_LEAVES_PRIOR[b] for a, b in itertools.pairwise(ks))
    assert [prior_leaves(n) for n in (7, 11, 15, 23, 31)] == [2, 3, 5, 7, 10]
    assert prior_leaves(1) == 1 and prior_leaves(64) == 10   # clamped to 1..n, flat past the end


@pytest.mark.parametrize("n,leaves",
                         [(7, 2), (11, 3), (15, 5), (23, 7), (31, 10), (3, 1), (7, 7)])
def test_synthetic_tree_has_the_requested_shape_main_path_first(n, leaves):
    t = synthetic_tree(n, slots=7, leaves=leaves, anchor=99)
    assert t.n_nodes == n and t.tokens[0] == 99 and len(set(t.tokens)) == t.width
    assert max(t.depth) <= 7 and len(t.leaves()) == leaves
    assert all(t.parent[r] < r for r in range(1, t.width))            # topological
    d0 = min(7, n - (leaves - 1))                                       # the main chain first
    assert all(t.parent[r] == r - 1 for r in range(1, d0 + 1)) and t.depth[d0] == d0
    dev = to_device(t, 4)                                               # device arrays build
    assert dev.width == n + 1 and dev.conv_idx.shape == (n + 1, 4)


def test_synthetic_tree_defaults_to_the_prior_and_clamps():
    assert len(synthetic_tree(7, 7).leaves()) == prior_leaves(7) == 2
    assert len(synthetic_tree(31, 7).leaves()) == 10
    assert len(synthetic_tree(3, 7, leaves=9).leaves()) == 3            # one leaf per node, max
    assert synthetic_tree(5, 7, leaves=1).is_chain()
    t = synthetic_tree(20, 3, leaves=2)                 # a shallow lattice forces more leaves
    assert t.n_nodes == 20 and max(t.depth) <= 3 and len(t.leaves()) == 7


# ---------------------------------------------------------------- Target tree verify


@pytest.mark.parametrize("make", [_tiny_hybrid, _tiny_hybrid_kernel, _tiny_dense])
def test_every_tree_row_matches_a_chain_verify_of_its_root_path(make):
    t = Target(make(), tokenizer=None)
    assert t.supports_tree
    t.reset_spec()
    logits, fused = t.verify_tree(BRANCHED, _prefilled(t), [0])
    mx.eval(logits, fused)
    for path in BRANCHED.leaf_paths():
        ref, ref_f = t.verify(mx.array([[BRANCHED.tokens[r] for r in path]]), _prefilled(t), [0])
        t.reset_spec()
        idx = mx.array(path)
        assert mx.allclose(logits[0, idx], ref[0], atol=1e-4, rtol=1e-4).item()
        assert mx.allclose(fused[0, idx], ref_f[0], atol=1e-4, rtol=1e-4).item()


@pytest.mark.parametrize("make", [_tiny_hybrid, _tiny_hybrid_kernel, _tiny_dense])
@pytest.mark.parametrize("path", [[0, 3, 4], [0, 1, 5], [0, 1, 2], [0]])
def test_commit_tree_leaves_the_caches_a_chain_commit_would(make, path):
    t = Target(make(), tokenizer=None)
    t.reset_spec()
    cache = _prefilled(t)
    t.verify_tree(BRANCHED, cache, [0])
    t.commit_tree(cache, path)
    assert t._kv_offset(cache) == len(PROMPT) + len(path)
    got, _ = t.run(mx.array([[13, 14]]), cache, [0])
    ref, _ = t.run(mx.array([PROMPT + [BRANCHED.tokens[r] for r in path] + [13, 14]]),
                   t.make_cache(), [0])
    assert mx.allclose(got[0], ref[0, -2:], atol=1e-4, rtol=1e-4).item()


def test_tree_selftest_passes_and_fails_closed_on_a_missing_patch(monkeypatch):
    t = Target(_tiny_hybrid(), tokenizer=None)
    assert t.tree_selftest() and t._tree_selftest_worst < 1e-3

    @contextlib.contextmanager
    def no_rope(self, depth):
        yield []

    monkeypatch.setattr(Target, "_tree_rope", no_rope)
    assert not t.tree_selftest()                    # structural: rope wrappers never ran
    monkeypatch.undo()

    @contextlib.contextmanager
    def no_linear(self, tree):
        yield ([], [])

    monkeypatch.setattr(Target, "_tree_linear", no_linear)
    t2 = Target(_tiny_hybrid(), tokenizer=None)
    assert not t2.tree_ready()                      # broken plumbing -> chains, not a crash


def test_tree_verify_restores_every_patch():
    import sys

    from mlx_lm.models.cache import KVCache

    t = Target(_tiny_hybrid_kernel(), tokenizer=None)
    attn = next(ly.self_attn for ly in t.model.layers if not ly.is_linear)
    gdn = t._gdn_modules[0]
    mod = sys.modules[type(gdn).__module__]
    before = (attn.rope, KVCache.update_and_fetch, "update_and_fetch" in KVCache.__dict__,
              type(gdn.conv1d).__call__, mod.gated_delta_update)
    t.reset_spec()
    t.verify_tree(BRANCHED, _prefilled(t), [0])
    after = (attn.rope, KVCache.update_and_fetch, "update_and_fetch" in KVCache.__dict__,
             type(gdn.conv1d).__call__, mod.gated_delta_update)
    assert all(a is b for a, b in zip(before, after)) and "rope" in attn


def test_the_kernel_fixture_takes_the_tree_kernel(monkeypatch):
    calls = []
    real = tree_gdn.tree_gated_delta

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    t = Target(_tiny_hybrid_kernel(), tokenizer=None)
    t.reset_spec()
    t._fold_ok = t.fold_selftest()          # the one-time probe runs the kernel too: do it first
    monkeypatch.setattr(tree_gdn, "tree_gated_delta", spy)
    t.verify_tree(BRANCHED, _prefilled(t), [0])
    assert len(calls) == len(t._gdn_modules)        # every linear layer, none path-batched


def test_unsupported_targets_are_not_tree_ready():
    t = Target(_tiny_hybrid(), tokenizer=None)
    t._nanbeige = True                              # looped targets run chains
    assert not t.supports_tree and not t.tree_ready()


# ---------------------------------------------------------------- tree gated-delta kernel


def _stock_simd_kernel():
    from mlx_lm.models import gated_delta as gd

    return getattr(gd, "gated_delta_kernel_unpacked", None) or gd.gated_delta_kernel


@pytest.mark.parametrize("tree", [chain_tree(0, list(range(1, 8))), BRANCHED,
                                  DraftTree(tokens=list(range(16)),
                                            parent=[-1, 0, 1, 2, 3, 0, 5, 1, 7, 2, 9, 0, 11,
                                                    12, 1, 14])])
def test_tree_gated_delta_matches_the_stock_kernel_along_every_path(tree):
    from mlx_lm.models.gated_delta import compute_g

    mx.random.seed(5)
    w, hk, dk, hv, dv = tree.width, 2, 64, 4, 64
    q = mx.random.normal((1, w, hk, dk)).astype(mx.bfloat16)
    k = mx.random.normal((1, w, hk, dk)).astype(mx.bfloat16)
    v = mx.random.normal((1, w, hv, dv)).astype(mx.bfloat16)
    a = mx.random.normal((1, w, hv)).astype(mx.bfloat16)
    b = mx.random.normal((1, w, hv)).astype(mx.bfloat16)
    A_log, dt_bias = mx.random.normal((hv,)), mx.random.normal((hv,))
    state = (0.1 * mx.random.normal((1, hv, dv, dk))).astype(mx.float32)
    g, beta = compute_g(A_log, a, dt_bias), mx.sigmoid(b)
    assert tree_gdn.eligible(q, v, g, state)
    y, base = tree_gdn.tree_gated_delta(q, k, v, g, beta, state,
                                        mx.array(tree.parent, dtype=mx.int32))
    assert base is state                      # nothing pending: no state written
    stock = _stock_simd_kernel()
    for path in tree.leaf_paths():
        pi = mx.array([path])
        ref, _ = stock(q[0][pi], k[0][pi], v[0][pi], g[0][pi], beta[0][pi], state)
        assert mx.array_equal(y[0, mx.array(path)], ref[0]).item()


def test_tree_gated_delta_eligibility():
    q = mx.zeros((1, 8, 2, 8))                     # Dk 8: not a multiple of 32
    v = mx.zeros((1, 8, 4, 8))
    g = mx.zeros((1, 8, 4))
    st = mx.zeros((1, 4, 8, 8))
    assert not tree_gdn.eligible(q, v, g, st)
    q = mx.zeros((1, 8, 2, 64))
    assert tree_gdn.eligible(q, mx.zeros((1, 8, 4, 64)), g, mx.zeros((1, 4, 64, 64)))
    assert not tree_gdn.eligible(q, mx.zeros((1, 8, 4, 64)), mx.zeros((1, 8, 4, 64)),
                                 mx.zeros((1, 4, 64, 64)))          # vector gate
    assert tree_gdn.rows_per_group(8, 128, 128) == 4
    assert tree_gdn.rows_per_group(64, 128, 128) == 0                # does not fit


# ---------------------------------------------------------------- multirow tree mask


def _tree_inputs(L_prefix=301, hq=8, hk=2, d=128, tree=BRANCHED):
    mx.random.seed(9)
    w = tree.width
    q = mx.random.normal((1, hq, w, d)).astype(mx.bfloat16)
    k = mx.random.normal((1, hk, L_prefix + w, d)).astype(mx.bfloat16)
    v = mx.random.normal((1, hk, L_prefix + w, d)).astype(mx.bfloat16)
    dt = to_device(tree, 4)
    mask = mx.concatenate([mx.ones((w, L_prefix), dtype=mx.bool_), dt.anc], axis=1)[None, None]
    return q, k, v, mask, dt.bits


@pytest.mark.parametrize("shape", [(8, 2, 128, BRANCHED),
                                   (24, 4, 256, BRANCHED),          # Qwen3.8, d-split
                                   (24, 4, 256, DraftTree(tokens=list(range(16)),
                                                          parent=[-1, 0, 1, 2, 3, 0, 5, 1, 7,
                                                                  2, 9, 0, 11, 12, 1, 14]))])
def test_multirow_tree_kernel_matches_masked_attention(shape):
    from mlx_dspark.multirow_attn import _ORIG_SDPA, multirow_sdpa

    hq, hk, d, tree = shape
    q, k, v, mask, bits = _tree_inputs(hq=hq, hk=hk, d=d, tree=tree)
    scale = q.shape[-1] ** -0.5
    ref = _ORIG_SDPA(q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
                     scale=scale, mask=mask)
    out = multirow_sdpa(q, k, v, scale=scale, causal=False, tree_anc=bits)
    mlx_err = float(mx.abs(_ORIG_SDPA(q, k, v, scale=scale, mask=mask).astype(mx.float32)
                           - ref).max())
    assert float(mx.abs(out.astype(mx.float32) - ref).max()) <= max(2 * mlx_err, 1e-2)


def test_only_the_registered_tree_mask_takes_the_tree_kernel():
    from mlx_dspark.multirow_attn import (
        _ORIG_SDPA,
        MultirowConfig,
        Window,
        multirow_attention,
        multirow_sdpa,
        tree_mask,
    )

    q, k, v, mask, bits = _tree_inputs()
    scale = q.shape[-1] ** -0.5
    cfg = MultirowConfig(windows=(Window(hq=8, hk=2, d=128, min_kv=16),))
    with multirow_attention(cfg), tree_mask(mask, bits):
        via_tree = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        other = mask + False                       # equal values, a different array object
        passthru = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=other)
    assert mx.array_equal(via_tree, multirow_sdpa(q, k, v, scale=scale, causal=False,
                                                  tree_anc=bits)).item()
    assert mx.array_equal(passthru, _ORIG_SDPA(q, k, v, scale=scale, mask=other)).item()


# ---------------------------------------------------------------- end to end


def _tiny_dflash2_for(target):
    return _tiny_dflash2_for_dims(target, hidden=32)


def _tiny_dflash2_for_dims(target, hidden: int):
    from mlx_dspark.dflash_model import DFlashConfig, DFlashDraftModel

    cfg = DFlashConfig(
        hidden_size=hidden, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, intermediate_size=64, vocab_size=64, rms_norm_eps=1e-6, rope_theta=1e7,
        max_position_embeddings=512, block_size=8, target_layer_ids=(0, 1),
        num_target_layers=2, mask_token_id=3, selector_rank=8, selector_top_k=4,
        conv_kernel_size=2, conv_group_size=16,
    )
    mx.random.seed(21)
    d = DFlashDraftModel(cfg)
    sel = d.candidate_selector
    sel.predecessor_codebook = mx.random.normal((64, 8))
    sel.successor_codebook = mx.random.normal((64, 8))
    mx.eval(d.parameters())
    d.bind(target.model)
    return d


@pytest.mark.parametrize("make", [_tiny_hybrid, _tiny_dense])
def test_dflash_generate_with_trees_reproduces_greedy_decoding(make):
    from mlx_dspark.generate import dflash_generate, greedy_generate

    t = Target(make(), tokenizer=None)
    drafter = _tiny_dflash2_for(t)
    ref = greedy_generate(t, _Tok(), prompt_ids=PROMPT, max_new_tokens=40).token_ids
    chain = dflash_generate(t, _Tok(), drafter, prompt_ids=PROMPT, max_new_tokens=40)
    assert chain.token_ids[:40] == ref[:40] and chain.tree_rounds == 0
    for tree in (0, 11):                            # equal width; a wide tree
        res = dflash_generate(t, _Tok(), drafter, prompt_ids=PROMPT, max_new_tokens=40,
                              tree=tree)
        assert res.token_ids[:40] == ref[:40]
    # sampling keeps chains (tree rounds are greedy-only)
    sampled = dflash_generate(t, _Tok(), drafter, prompt_ids=PROMPT, max_new_tokens=8,
                              tree=0, temperature=0.7, seed=1)
    assert sampled.tree_rounds == 0


def test_tree_rounds_branch_on_an_uncertain_drafter():
    from mlx_dspark.generate import dflash_generate

    t = Target(_tiny_hybrid(), tokenizer=None)
    drafter = _tiny_dflash2_for(t)
    res = dflash_generate(t, _Tok(), drafter, prompt_ids=PROMPT, max_new_tokens=40, tree=0)
    assert res.tree_rounds > 0                      # a random selector is far from confident


def test_tiny_model_helpers_are_fp32_so_the_tight_tolerances_hold():
    t = _tiny_hybrid()
    assert t.parameters()["language_model"]["model"]["embed_tokens"]["weight"].dtype == \
        mx.float32
    assert isinstance(t.layers[3].self_attn.rope, nn.Module)


# ---------------------------------------------------------------- bounds + policy


def test_node_count_is_bounded_by_the_lattice_and_the_ceiling():
    from mlx_dspark.tree import TREE_MAX_NODES, reachable_nodes

    assert reachable_nodes(1, 4) == 4 and reachable_nodes(2, 4) == 20
    assert reachable_nodes(7, 16) == TREE_MAX_NODES
    rng = np.random.default_rng(3)
    cand = rng.permutation(100)[:4].reshape(1, 4).astype(np.int32)       # one slot, K = 4
    scores = rng.normal(size=(1, 4, 4)).astype(np.float32)
    dt = device_tree(mx.array(cand), mx.array(scores), 31, 9)
    assert dt.width == 5                              # clamped to the 4 reachable nodes
    toks = dt.tokens.tolist()[1:]
    assert sorted(toks) == sorted(cand[0].tolist())   # no duplicate root children


def test_wide_trees_only_at_the_full_block(monkeypatch):
    from mlx_dspark import generate as G
    from mlx_dspark.generate import dflash_generate

    seen = []
    real = G.device_tree

    def spy(cand, scores, n_nodes, anchor, conv_k=4):
        seen.append((int(cand.shape[0]), int(n_nodes)))
        return real(cand, scores, n_nodes, anchor, conv_k)

    monkeypatch.setattr(G, "device_tree", spy)
    t = Target(_tiny_hybrid(), tokenizer=None)
    drafter = _tiny_dflash2_for(t)
    dflash_generate(t, _Tok(), drafter, prompt_ids=PROMPT, max_new_tokens=12,
                    max_draft_tokens=3, tree=11)
    assert seen and all(n == s == 3 for s, n in seen)          # shrunk cap: equal width
    seen.clear()
    dflash_generate(t, _Tok(), drafter, prompt_ids=PROMPT, max_new_tokens=12, tree=11)
    assert seen and all(s == 7 and n == 11 for s, n in seen)   # full block: the wide tree


def test_tree_rounds_report_chain_positions_to_telemetry():
    from mlx_dspark.generate import dflash_generate

    t = Target(_tiny_hybrid(), tokenizer=None)
    drafter = _tiny_dflash2_for(t)
    rounds = []
    dflash_generate(t, _Tok(), drafter, prompt_ids=PROMPT, max_new_tokens=24, tree=15,
                    on_round=lambda **kw: rounds.append(kw))
    assert rounds and all(r["drafted"] <= r["cap"] for r in rounds)
    assert any(r["source"] == "tree" for r in rounds)


def test_static_tree_never_calibrates_just_to_price_a_tree(tmp_path, monkeypatch):
    C = _CAL

    monkeypatch.setattr(C, "measure_tree_fallback_ratio",
                        lambda *a, **k: pytest.fail("kernel live: nothing to price"))
    t = Target(_tiny_hybrid_kernel(), tokenizer=None)     # the tree kernel serves every layer
    drafter = _tiny_dflash2_for_dims(t, hidden=64)
    got = static_tree(t, drafter, setting="auto", target_repo="tiny/target",
                      drafter_repo="tiny/drafter", cache_dir=str(tmp_path))
    assert got == 0 and t.tree_kernel_live            # equal width, priced from nothing
    assert not any(tmp_path.iterdir())                # and no calibration was written
    assert static_tree(t, drafter, setting="off", target_repo="x", drafter_repo="y") is None
    assert static_tree(t, drafter, setting=9, target_repo="x", drafter_repo="y") == 9


def test_tree_kernel_live_reflects_the_recurrence_path():
    k = Target(_tiny_hybrid_kernel(), tokenizer=None)
    assert k.tree_ready() and k.tree_kernel_live
    assert k._tree_calls["kernel"] == k._tree_calls["gdu"] == 3
    o = Target(_tiny_hybrid(), tokenizer=None)        # training-mode modules take the ops path
    assert o.tree_ready() and not o.tree_kernel_live and o._tree_calls["kernel"] == 0
    d = Target(_tiny_dense(), tokenizer=None)         # no recurrence: nothing to fall back to
    assert d.tree_ready() and d.tree_kernel_live


@pytest.mark.parametrize("make", [_tiny_hybrid_kernel, _tiny_hybrid, _tiny_dense])
def test_measure_tree_curve_times_real_tree_verifies_and_rolls_back(make):
    t = Target(make(), tokenizer=None)
    assert t.tree_ready()
    curve = measure_tree_curve(t, [0], [3, 7, 11], slots=7, ctx_len=16)
    assert sorted(curve) == [3, 7, 11] and all(v > 0 for v in curve.values())
    assert t._tree_stash is None
    assert measure_tree_fallback_ratio(t, [0], 7, ctx_len=16) > 0


def test_static_tree_prices_the_recurrence_fallback_paired_and_caches_it(tmp_path,
                                                                           monkeypatch):
    C = _CAL

    t = Target(_tiny_hybrid(), tokenizer=None)
    drafter = _tiny_dflash2_for(t)
    assert t.tree_ready() and not t.tree_kernel_live
    seen, ratio = [], {"v": 1.10}

    def fake_ratio(target, tap, cap, ctx_len=C.CTX_LEN):
        seen.append((cap, list(tap)))
        return ratio["v"]

    monkeypatch.setattr(C, "measure_tree_fallback_ratio", fake_ratio)
    kw = {"setting": "auto", "target_repo": "x", "drafter_repo": "y",
          "cache_dir": str(tmp_path)}
    assert static_tree(t, drafter, **kw) is None          # +10% per round: chains
    ratio["v"] = 1.01
    assert static_tree(t, drafter, **kw) == 0             # within the margin: equal width
    assert seen == [(7, [0, 1]), (7, [0, 1])]             # no curve entry to cache on
    assert not any(tmp_path.iterdir())                    # still no calibration written
    assert static_tree(t, drafter, setting="on", target_repo="x", drafter_repo="y") == 0
    assert len(seen) == 2                                 # "on" and N skip the pricing
    # with a curve entry present the ratio is cached on it and read back, per cap
    key = C._cache_key("dflash", "x", "y")
    C.save_cached(key, {"verify": {"2": 70.0, "8": 72.0, "12": 160.0}, "drafter": 10.0,
                        "overhead": 0.0}, str(tmp_path))
    ratio["v"] = 1.20
    assert static_tree(t, drafter, **kw) is None
    assert C.load_cached(key, str(tmp_path))["tree_fallback"] == {"cap": 7, "ratio": 1.20}
    assert static_tree(t, drafter, **kw) is None and len(seen) == 3   # cached, not re-measured
    ratio["v"] = 1.0
    assert static_tree(t, drafter, cap=3, **kw) == 0 and seen[-1][0] == 3


@pytest.mark.parametrize("make", [_tiny_hybrid_kernel])
def test_kernel_path_generation_reproduces_greedy(make):
    from mlx_dspark.generate import dflash_generate, greedy_generate

    t = Target(make(), tokenizer=None)
    drafter = _tiny_dflash2_for_dims(t, hidden=64)
    ref = greedy_generate(t, _Tok(), prompt_ids=PROMPT, max_new_tokens=40).token_ids
    for tree in (0, 11):
        res = dflash_generate(t, _Tok(), drafter, prompt_ids=PROMPT, max_new_tokens=40,
                              tree=tree)
        assert res.token_ids[:40] == ref[:40]
