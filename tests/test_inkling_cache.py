"""The in-place KV caches of Inkling 2's attention (inkling_cache) against mlx-vlm's own caches:
what a call sees, what a rollback leaves, and the sink row kept inside the buffer."""
import mlx.core as mx
import pytest

vlm_cache = pytest.importorskip("mlx_vlm.models.cache")
if not hasattr(vlm_cache, "_BaseCache") or not hasattr(vlm_cache, "RotatingKVCache"):
    pytest.skip("this mlx-vlm has another cache layout", allow_module_level=True)

from mlx_dspark import inkling_cache as C  # noqa: E402

WINDOW = 5


def _rows(n, seed):
    key = mx.random.key(seed)
    k, v = mx.random.split(key)
    return (mx.random.normal((1, 2, n, 4), key=k).astype(mx.bfloat16),
            mx.random.normal((1, 2, n, 4), key=v).astype(mx.bfloat16))


def _eq(a, b):
    return a.shape == b.shape and bool(mx.array_equal(a, b))


def _stock_view(kv, k, v):
    """What mlx-vlm's attention reads after an update: the ring in temporal order."""
    k, v = kv.update_and_fetch(k, v)
    return kv._temporal_order(k), kv._temporal_order(v)


def _window(step=2):
    cache = C.classes()[1](WINDOW)
    cache.step = step                 # tiny spare room: the rebuild path runs every few rows
    return cache


@pytest.mark.parametrize("plan", [[1] * 14, [1, 1, 3, 1, 5, 1, 1, 2, 7, 1, 1], [6, 1, 1, 9, 1],
                                  [3, 3, 3, 3, 1, 1, 1, 1], [12, 12, 1, 4]])
def test_window_cache_shows_each_call_what_the_rotating_cache_shows(plan):
    stock = vlm_cache.RotatingKVCache(max_size=WINDOW, keep=0)
    new = _window()
    for i, n in enumerate(plan):
        k, v = _rows(n, i)
        sk, sv = _stock_view(stock, k, v)
        nk, nv = new.update_and_fetch(k, v)
        assert _eq(sk, nk) and _eq(sv, nv), (i, n)
        # the fused attention's views: the same rows, then the sink's zero row — no copy
        wk, wv = new.sink_views()
        assert _eq(wk[..., :-1, :], nk) and _eq(wv[..., :-1, :], nv)
        assert not bool(mx.any(wk[..., -1, :] != 0)) and not bool(mx.any(wv[..., -1, :] != 0))
        assert new.offset == stock.offset and new.size() == stock.size()


def test_window_cache_trim_is_a_verify_rollback():
    # (rows, rows kept): the reference cache only ever sees the kept rows
    plan = [(1, 1), (4, 2), (1, 1), (3, 1), (5, 5), (4, 3), (1, 1), (6, 2), (2, 1), (1, 1), (3, 3)]
    ref = vlm_cache.RotatingKVCache(max_size=WINDOW, keep=0)
    new = _window()
    committed = 0
    for i, (n, keep) in enumerate(plan):
        k, v = _rows(n, 100 + i)
        nk, _ = new.update_and_fetch(k, v)
        # the call saw the last WINDOW - 1 committed rows and all of its own
        assert nk.shape[2] == min(committed, WINDOW - 1) + n
        assert _eq(nk[..., -n:, :], k)
        assert new.trim(n - keep) == n - keep
        rk, _ = _stock_view(ref, k[..., :keep, :], v[..., :keep, :])
        committed += keep
        tail = min(committed, WINDOW - 1)             # what the next call will need
        sk = new.state[0]
        assert sk.shape[2] >= tail
        assert _eq(sk[..., sk.shape[2] - tail:, :], rk[..., rk.shape[2] - tail:, :])
        assert new.offset == ref.offset == committed
    # rejected rows stay in the buffer: the next sink row is rewritten over them
    k, v = _rows(1, 999)
    new.update_and_fetch(k, v)
    wk, wv = new.sink_views()
    assert not bool(mx.any(wk[..., -1, :] != 0)) and not bool(mx.any(wv[..., -1, :] != 0))


def test_window_cache_state_round_trips_and_adopts_assigned_arrays():
    a = _window(step=3)
    for i, n in enumerate([1, 4, 1, 1, 3]):
        a.update_and_fetch(*_rows(n, 200 + i))
    assert a.meta_state == (str(WINDOW), "10") and not a.is_trimmable() and not a.empty()
    b = C.classes()[1](1)
    b.state = tuple(mx.array(x) for x in a.state)      # a prefix-cache snapshot, set back
    b.meta_state = a.meta_state
    assert b.max_size == WINDOW and b.offset == 10 and b.nbytes > 0
    for i, n in enumerate([1, 2, 1, 5, 1]):
        k, v = _rows(n, 300 + i)
        ak, av = a.update_and_fetch(k, v)
        bk, bv = b.update_and_fetch(k, v)
        assert _eq(ak, bk) and _eq(av, bv)
        assert _eq(a.sink_views()[0], b.sink_views()[0])
    with pytest.raises(ValueError):
        b.meta_state = ("0", str(WINDOW), "10", "3")   # a RotatingKVCache's four fields
    # arrays assigned from outside are all valid rows, and nothing past them is trusted
    c = _window()
    c.keys, c.values = _rows(3, 400)
    k, v = _rows(1, 401)
    ck, _ = c.update_and_fetch(k, v)
    assert ck.shape[2] == 4 and _eq(ck[..., -1:, :], k)
    assert not bool(mx.any(c.sink_views()[0][..., -1, :] != 0))
    empty = C.classes()[1](WINDOW)
    assert empty.empty() and empty.state == (None, None) and empty.nbytes == 0


def test_sink_cache_is_the_kv_cache_with_the_sink_row_in_the_buffer():
    stock = vlm_cache.KVCache()
    new = C.classes()[0]()
    new.step = 4
    plan = [(1, 1), (1, 1), (4, 2), (1, 1), (3, 3), (5, 1), (1, 1), (2, 2), (6, 4), (1, 1)]
    for i, (n, keep) in enumerate(plan):
        k, v = _rows(n, 500 + i)
        sk, sv = stock.update_and_fetch(k, v)
        nk, nv = new.update_and_fetch(k, v)
        assert _eq(sk, nk) and _eq(sv, nv), i
        wk, wv = new.sink_views()
        assert _eq(wk[..., :-1, :], nk) and not bool(mx.any(wk[..., -1, :] != 0))
        assert _eq(wv[..., :-1, :], nv) and not bool(mx.any(wv[..., -1, :] != 0))
        stock.trim(n - keep)
        new.trim(n - keep)
        assert new.offset == stock.offset
        assert _eq(new.state[0], stock.state[0])
    # a restored state has no spare row: the next update rebuilds the buffer with one
    other = C.classes()[0]()
    other.state = tuple(mx.array(x) for x in new.state)
    k, v = _rows(2, 600)
    assert _eq(other.update_and_fetch(k, v)[0], new.update_and_fetch(k, v)[0])
    assert not bool(mx.any(other.sink_views()[1][..., -1, :] != 0))


def test_convert_swaps_fresh_inkling_caches_only():
    if not hasattr(vlm_cache, "CacheList"):
        pytest.skip("this mlx-vlm has no CacheList")
    sink_cls, window_cls = C.classes()
    used = vlm_cache.KVCache()
    used.update_and_fetch(*_rows(1, 1))
    entries = [vlm_cache.CacheList(vlm_cache.RotatingKVCache(max_size=8, keep=0), "conv"),
               vlm_cache.CacheList(vlm_cache.KVCache(), "conv"),
               vlm_cache.CacheList(vlm_cache.RotatingKVCache(max_size=8, keep=2), "conv"),
               vlm_cache.CacheList(used, "conv")]
    out = C.convert(entries)
    assert out is entries
    assert type(entries[0].caches[0]) is window_cls and entries[0].caches[0].max_size == 8
    assert type(entries[1].caches[0]) is sink_cls
    assert type(entries[2].caches[0]) is vlm_cache.RotatingKVCache      # keeps a prefix: not ours
    assert entries[3].caches[0] is used                                 # not empty
    assert all(e.caches[1] == "conv" for e in entries)
    assert C.is_sink_cache(entries[0].caches[0]) and C.is_sink_cache(entries[1].caches[0])
    assert not C.is_sink_cache(entries[2].caches[0])


def test_speculative_transaction_rolls_these_caches_back_by_trimming():
    state = pytest.importorskip("mlx_vlm.speculative.cache_state")
    sink_cls, window_cls = C.classes()
    for cache in (window_cls(WINDOW), sink_cls()):
        cache.update_and_fetch(*_rows(3, 700))
        txn = state.start_speculative_cache([cache], 4)
        # an ordinary append-only cache to the transaction: no ring snapshot, no replay
        assert not txn._rotating and id(cache) in txn._positions
        k, v = _rows(4, 701)
        cache.update_and_fetch(k, v)
        txn.commit(2)
        assert cache.offset == 5
        assert _eq(cache.state[0][..., -2:, :], k[..., :2, :])
        txn = state.start_speculative_cache([cache], 3)
        cache.update_and_fetch(*_rows(3, 702))
        txn.abort()
        assert cache.offset == 5 and _eq(cache.state[0][..., -2:, :], k[..., :2, :])
