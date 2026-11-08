"""Deferred commit on gated-DeltaNet hybrids (Target.rollback / commit_tree / flush_pending and
tree_gdn's pending segment): a round's accepted rows fold into the stored recurrence state at
the NEXT verify instead of a rollback re-run. Load-bearing property: tokens AND committed
states identical to the eager re-run — on the kernel path and on the ops fallback, through
every loop that touches the state, and for every reader that looks at the state in between."""

from __future__ import annotations

import mlx.core as mx
import pytest
from test_tree import (
    BRANCHED,
    PROMPT,
    _tiny_dflash2_for_dims,
    _tiny_hybrid,
    _tiny_hybrid_kernel,
    _Tok,
)

from mlx_dspark import target as T
from mlx_dspark import tree_gdn
from mlx_dspark.generate import dflash_generate, greedy_generate
from mlx_dspark.lookup import lookup_generate
from mlx_dspark.target import Target
from mlx_dspark.tree import chain_tree


def _stock():
    from mlx_lm.models import gated_delta as gd

    return getattr(gd, "gated_delta_kernel_unpacked", None) or gd.gated_delta_kernel


def _lin_states(t, cache):
    return [c[1] for c in cache if not t._is_trimmable(c)]


# ---------------------------------------------------------------- the kernel's pending segment


@pytest.mark.parametrize("n_pending", [1, 3, 7])
@pytest.mark.parametrize("tree", [chain_tree(0, list(range(1, 8))), BRANCHED])
def test_pending_segment_matches_the_stock_kernel_bitwise(n_pending, tree):
    from mlx_lm.models.gated_delta import compute_g

    mx.random.seed(9)
    w, hk, dk, hv, dv = tree.width, 2, 64, 4, 64

    def rows(n):
        q = mx.random.normal((1, n, hk, dk)).astype(mx.bfloat16)
        k = mx.random.normal((1, n, hk, dk)).astype(mx.bfloat16)
        v = mx.random.normal((1, n, hv, dv)).astype(mx.bfloat16)
        a = mx.random.normal((1, n, hv)).astype(mx.bfloat16)
        b = mx.random.normal((1, n, hv)).astype(mx.bfloat16)
        return q, k, v, a, b

    A_log, dt_bias = mx.random.normal((hv,)), mx.random.normal((hv,))
    state = (0.1 * mx.random.normal((1, hv, dv, dk))).astype(mx.float32)
    qp, kp, vp, ap, bp = rows(n_pending)
    q, k, v, a, b = rows(w)
    gp, betap = compute_g(A_log, ap, dt_bias), mx.sigmoid(bp)
    g, beta = compute_g(A_log, a, dt_bias), mx.sigmoid(b)
    assert tree_gdn.eligible(q, v, g, state)
    y, base = tree_gdn.tree_gated_delta(q, k, v, g, beta, state,
                                        mx.array(tree.parent, dtype=mx.int32),
                                        pending=(qp, kp, vp, gp, betap))
    stock = _stock()
    _, ref_base = stock(qp, kp, vp, gp, betap, state)        # the fold = the stock chain's end
    assert base is not state and mx.array_equal(base, ref_base).item()
    for path in tree.leaf_paths():                           # every row from that base
        pi = mx.array([path])
        ref, _ = stock(q[0][pi], k[0][pi], v[0][pi], g[0][pi], beta[0][pi], ref_base)
        assert mx.array_equal(y[0][mx.array(path)], ref[0]).item()


# ---------------------------------------------------------------- loops: deferred == eager


@pytest.mark.parametrize("make,hidden", [(_tiny_hybrid_kernel, 64), (_tiny_hybrid, 32)])
@pytest.mark.parametrize("tree", [None, 0, 11])
def test_dflash_loops_commit_the_same_tokens_and_states_as_the_eager_rerun(make, hidden, tree,
                                                                           monkeypatch):
    def run(deferred):
        monkeypatch.setattr(T, "DEFERRED_COMMIT", deferred)
        t = Target(make(), tokenizer=None)
        drafter = _tiny_dflash2_for_dims(t, hidden=hidden)
        cache = t.make_cache()
        res = dflash_generate(t, _Tok(), drafter, prompt_ids=PROMPT, max_new_tokens=40,
                              tree=tree, cache=cache)
        assert t._pending is None                           # flushed at the loop's exit
        states = _lin_states(t, cache)
        mx.eval(*states)
        return res.token_ids, states, t

    toks_d, st_d, t_d = run(True)
    toks_e, st_e, t_e = run(False)
    assert toks_d == toks_e
    ref = greedy_generate(t_e, _Tok(), prompt_ids=PROMPT, max_new_tokens=40).token_ids
    assert toks_d[:40] == ref[:40]
    assert st_d and all(mx.array_equal(x, y).item() for x, y in zip(st_d, st_e))
    # the kernel fixture proves and uses the kernel fold; training-mode modules take the ops
    # path, so the probe fails structurally and the fold stays eager (stock) there
    assert t_d._fold_ok is (make is _tiny_hybrid_kernel)


@pytest.mark.parametrize("make", [_tiny_hybrid_kernel, _tiny_hybrid])
def test_lookup_loop_matches_the_eager_rerun(make, monkeypatch):
    def run(deferred):
        monkeypatch.setattr(T, "DEFERRED_COMMIT", deferred)
        t = Target(make(), tokenizer=None)
        cache = t.make_cache()
        res = lookup_generate(t, _Tok(), prompt_ids=PROMPT * 3, max_new_tokens=24, cache=cache)
        assert t._pending is None
        states = _lin_states(t, cache)
        mx.eval(*states)
        return res.token_ids, states

    toks_d, st_d = run(True)
    toks_e, st_e = run(False)
    assert toks_d == toks_e
    assert st_d and all(mx.array_equal(x, y).item() for x, y in zip(st_d, st_e))


# ---------------------------------------------------------------- readers flush first


def _after_rollback(t, prompt):
    cache = t.make_cache()
    t.reset_spec()
    t.run(mx.array([prompt]), cache, [0])
    t.verify(mx.array([[6, 7, 8, 9]]), cache, [0])
    t.rollback(cache, 2, [7])                               # rows [6, 7] pend
    assert t._pending is not None and t._pending.cache is cache
    return cache


def _committed(t, tokens):
    cache = t.make_cache()
    logits, _ = t.run(mx.array([tokens]), cache, [0])
    return logits, _lin_states(t, cache)


def test_plain_step_run_and_reset_spec_fold_the_pending_rows_first():
    t = Target(_tiny_hybrid_kernel(), tokenizer=None)
    prompt = [1, 2, 3, 4, 5]
    ref_logits, ref_states = _committed(t, prompt + [6, 7, 10])
    # (a) a width-1 verify (a plain step) flushes, then runs: the committed forward's logits
    cache = _after_rollback(t, prompt)
    got, _ = t.verify(mx.array([[10]]), cache, [0])
    assert t._pending is None
    assert mx.allclose(got[0, -1], ref_logits[0, -1], atol=1e-4, rtol=1e-4).item()
    # (b) run() flushes
    cache = _after_rollback(t, prompt)
    got, _ = t.run(mx.array([[10]]), cache, [0])
    assert t._pending is None
    assert mx.allclose(got[0, -1], ref_logits[0, -1], atol=1e-4, rtol=1e-4).item()
    for x, y in zip(_lin_states(t, cache), ref_states):
        assert mx.allclose(x, y, atol=1e-4, rtol=1e-4).item()
    # (c) reset_spec() flushes into the pending rows' own cache
    cache = _after_rollback(t, prompt)
    t.reset_spec()
    assert t._pending is None
    _, ref2 = _committed(t, prompt + [6, 7])
    for x, y in zip(_lin_states(t, cache), ref2):
        assert mx.allclose(x, y, atol=1e-4, rtol=1e-4).item()


def test_commit_tree_defers_the_path_rows_and_the_next_step_matches_the_chain():
    t = Target(_tiny_hybrid_kernel(), tokenizer=None)
    t.reset_spec()
    cache = t.make_cache()
    t.run(mx.array([PROMPT]), cache, [0])
    t.verify_tree(BRANCHED, cache, [0])
    t.commit_tree(cache, [0, 3, 4])
    assert t._pending is not None and len(t._pending.rows) == len(t._gdn_modules)
    assert t._pending.rows[0][0].shape[1] == 3              # the path's three rows
    got, _ = t.run(mx.array([[13]]), cache, [0])            # flushes, then runs
    assert t._pending is None
    ref, _ = t.run(mx.array([PROMPT + [BRANCHED.tokens[r] for r in (0, 3, 4)] + [13]]),
                   t.make_cache(), [0])
    assert mx.allclose(got[0, -1], ref[0, -1], atol=1e-4, rtol=1e-4).item()


def test_eager_switch_folds_at_rollback(monkeypatch):
    monkeypatch.setattr(T, "DEFERRED_COMMIT", False)
    t = Target(_tiny_hybrid_kernel(), tokenizer=None)
    cache = t.make_cache()
    t.reset_spec()
    t.run(mx.array([[1, 2, 3]]), cache, [0])
    t.verify(mx.array([[4, 5, 6]]), cache, [0])
    t.rollback(cache, 1, [5])
    assert t._pending is None                               # folded right away


def test_fold_selftest_proves_the_kernel_and_fails_closed_on_a_broken_one(monkeypatch):
    t = Target(_tiny_hybrid_kernel(), tokenizer=None)
    assert t.fold_selftest() and t._fold_selftest_worst == (0.0, 0.0)   # bitwise on fp32
    real = tree_gdn.tree_gated_delta

    def broken(*a, **k):
        y, base = real(*a, **k)
        return y, (base if k.get("pending") is None else base * 1.5)    # a wrong fold

    monkeypatch.setattr(tree_gdn, "tree_gated_delta", broken)
    assert not t.fold_selftest()
    assert not Target(_tiny_hybrid(), tokenizer=None).fold_selftest()   # ops path: no kernel
