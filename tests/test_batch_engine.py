"""Model-free tests for the continuous-batching cache (batch_engine.BatchCache).

These validate the per-row-offset cache semantics — left-alignment, per-row scatter writes,
per-row trim as metadata, and the causal+padding mask — in isolation, without loading a model.
Real-model row-isolation losslessness (B=N == B=1 ids) is validated separately in a device run.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_dspark.batch_engine import BatchCache, batchable, build_batch_mask


def _rows(lens, H=2, D=4):
    return [(mx.random.normal((1, H, l, D)), mx.random.normal((1, H, l, D))) for l in lens]


def test_from_rows_left_aligned():
    lens = [3, 1, 4]
    pairs = _rows(lens)
    c = BatchCache.from_rows(pairs)
    assert c.offsets == lens
    assert c.keys.shape[0] == 3 and c.keys.shape[2] >= max(lens)
    for b, l in enumerate(lens):
        assert mx.allclose(c.keys[b : b + 1, :, :l, :], pairs[b][0]).item()
        assert mx.allclose(c.values[b : b + 1, :, :l, :], pairs[b][1]).item()


def test_offset_property_matches_offsets():
    c = BatchCache.from_rows(_rows([3, 1, 4]))
    assert c.offset.tolist() == [3, 1, 4]


def test_update_and_fetch_writes_at_each_row_offset():
    lens = [3, 1, 4]
    c = BatchCache.from_rows(_rows(lens))
    k = mx.random.normal((3, 2, 1, 4))
    v = mx.random.normal((3, 2, 1, 4))
    K, V = c.update_and_fetch(k, v)
    assert c.offsets == [4, 2, 5]
    assert K.shape[2] == 5 and V.shape[2] == 5           # Lcur = max(new offsets)
    for b, l in enumerate(lens):                          # new token landed at old offset
        assert mx.allclose(c.keys[b : b + 1, :, l : l + 1, :], k[b : b + 1]).item()
        assert mx.allclose(c.values[b : b + 1, :, l : l + 1, :], v[b : b + 1]).item()


def test_multi_token_update():
    c = BatchCache.from_rows(_rows([2, 2, 2]))
    k = mx.random.normal((3, 2, 3, 4))                    # verify block width 3
    c.update_and_fetch(k, k)
    assert c.offsets == [5, 5, 5]
    assert mx.allclose(c.keys[:, :, 2:5, :], k).item()


def test_trim_is_per_row_metadata_and_next_write_overwrites():
    c = BatchCache.from_rows(_rows([4, 4, 4]))
    c.update_and_fetch(mx.random.normal((3, 2, 1, 4)), mx.random.normal((3, 2, 1, 4)))
    assert c.offsets == [5, 5, 5]
    c.trim([0, 1, 2])                                     # roll rows back by 0 / 1 / 2
    assert c.offsets == [5, 4, 3]
    k2 = mx.random.normal((3, 2, 1, 4))
    c.update_and_fetch(k2, k2)
    assert c.offsets == [6, 5, 4]                         # each resumes from its trimmed offset
    for b, off in enumerate([5, 4, 3]):
        assert mx.allclose(c.keys[b : b + 1, :, off : off + 1, :], k2[b : b + 1]).item()


def test_trim_clamps_at_zero():
    c = BatchCache.from_rows(_rows([2, 1, 3]))
    c.trim([5, 0, 1])
    assert c.offsets == [0, 1, 2]


def test_update_grows_buffer():
    from mlx_dspark.batch_engine import STEP

    c = BatchCache.from_rows(_rows([STEP - 1, STEP - 1]))
    k = mx.random.normal((2, 2, 3, 4))
    c.update_and_fetch(k, k)                              # crosses the STEP boundary
    assert c.keys.shape[2] >= STEP + 2
    assert c.offsets == [STEP + 2, STEP + 2]
    assert mx.allclose(c.keys[:, :, STEP - 1 : STEP + 2, :], k).item()


def test_build_batch_mask_causal_and_padding():
    m = build_batch_mask([2, 0], T=2)                    # Lcur = 2 + 2 = 4
    assert m.shape == (2, 1, 2, 4) and m.dtype == mx.bool_
    grid = m.astype(mx.int32).tolist()
    assert grid[0][0][0] == [1, 1, 1, 0]                 # row0 off2, q0 -> j<=2
    assert grid[0][0][1] == [1, 1, 1, 1]                 # row0 off2, q1 -> j<=3
    assert grid[1][0][0] == [1, 0, 0, 0]                 # row1 off0, q0 -> j<=0
    assert grid[1][0][1] == [1, 1, 0, 0]                 # row1 off0, q1 -> j<=1


def test_build_batch_mask_single_token_decode():
    m = build_batch_mask([5, 3, 0], T=1)                 # Lcur = 5 + 1 = 6
    assert m.shape == (3, 1, 1, 6)
    grid = m.astype(mx.int32).tolist()
    assert grid[0][0][0] == [1, 1, 1, 1, 1, 1]           # off5: 5 past tokens (cols 0-4) + self (col 5)
    assert grid[1][0][0] == [1, 1, 1, 1, 0, 0]           # off3: cols 0-3
    assert grid[2][0][0] == [1, 0, 0, 0, 0, 0]           # off0: col 0 only


def test_batchable_rejects_vlm():
    class T:
        is_vlm = True

    assert not batchable(T())


def test_batchable_accepts_dense_kvcache():
    from mlx_lm.models.cache import KVCache

    class T:
        is_vlm = False

        def make_cache(self):
            return [KVCache(), KVCache()]

    assert batchable(T())


def test_batchable_rejects_rotating_cache():
    from mlx_lm.models.cache import RotatingKVCache

    class T:
        is_vlm = False

        def make_cache(self):
            return [RotatingKVCache(64)]

    assert not batchable(T())


# --- hybrid (linear-attention) targets: batched baseline, but not batched spec ---

def _hybrid_target():
    from mlx_lm.models.cache import ArraysCache, KVCache

    class T:
        is_vlm = False
        is_hybrid = True

        def make_cache(self):
            # the qwen3_5 shape: mostly recurrent layers, a minority of attention layers
            return [ArraysCache(2), ArraysCache(2), ArraysCache(2), KVCache()]

    return T()


def test_batchable_accepts_hybrid_recurrent_layers():
    # a fresh ArraysCache holds [None, None] — the capability query has to answer before any
    # prefill has allocated the states
    assert batchable(_hybrid_target())


def test_batch_spec_unsupported_for_hybrid_but_supported_for_dense():
    from mlx_lm.models.cache import KVCache

    from mlx_dspark.batch_engine import batch_spec_supported

    assert not batch_spec_supported(_hybrid_target())   # no per-row recurrent rollback yet

    class Dense:
        is_vlm = False

        def make_cache(self):
            return [KVCache(), KVCache()]

    assert batch_spec_supported(Dense())


def test_recurrent_mergeable_rejects_wrapped_rotating_cache():
    # A wrapped RotatingKVCache is also "not trimmable and holds arrays"; merging one by
    # concatenation would decode garbage past the window, so the gate is by class, not shape.
    from mlx_lm.models.cache import RotatingKVCache

    from mlx_dspark.batch_engine import _recurrent_mergeable

    c = RotatingKVCache(8)
    c.update_and_fetch(mx.zeros((1, 1, 32, 4)), mx.zeros((1, 1, 32, 4)))
    assert not c.is_trimmable()          # wrapped: the case that could sneak through
    assert not _recurrent_mergeable(c)


def test_merge_row_caches_concatenates_recurrent_state_across_rows():
    from mlx_lm.models.cache import ArraysCache, KVCache

    from mlx_dspark.batch_engine import _merge_row_caches

    rows = []
    for b in range(3):
        rec = ArraysCache(2)
        rec[0] = mx.full((1, 3, 4), float(b))       # conv window
        rec[1] = mx.full((1, 2, 4, 4), float(b))    # recurrent state
        kv = KVCache()
        # rows deliberately have DIFFERENT lengths — the point of the design
        kv.update_and_fetch(mx.zeros((1, 2, b + 1, 4)), mx.zeros((1, 2, b + 1, 4)))
        rows.append([rec, kv])

    merged = _merge_row_caches(rows)
    assert merged[0][0].shape == (3, 3, 4) and merged[0][1].shape == (3, 2, 4, 4)
    for b in range(3):
        assert float(merged[0][0][b].min()) == float(b)   # row b's state landed in row b
    assert merged[1].offsets == [1, 2, 3]                 # attention layer keeps per-row lengths


# --- slot reuse (M4 dynamic admission) ---

def test_empty_then_set_row_admits():
    c = BatchCache.empty(3, n_heads=2, dk=4, dv=4, dtype=mx.float32)
    assert c.offsets == [0, 0, 0]
    k = mx.random.normal((1, 2, 5, 4))
    v = mx.random.normal((1, 2, 5, 4))
    c.set_row(1, k, v, 5)
    assert c.offsets == [0, 5, 0]
    assert mx.allclose(c.keys[1:2, :, :5, :], k).item()


def test_set_row_grows_buffer():
    from mlx_dspark.batch_engine import STEP

    c = BatchCache.empty(2, n_heads=2, dk=4, dv=4, dtype=mx.float32)
    n = STEP + 7
    k = mx.random.normal((1, 2, n, 4))
    c.set_row(0, k, k, n)
    assert c.keys.shape[2] >= n and c.offsets == [n, 0]
    assert mx.allclose(c.keys[0:1, :, :n, :], k).item()


def test_move_row_compacts_and_vacates():
    pairs = _rows([3, 2, 4])
    c = BatchCache.from_rows(pairs)
    c.move_row(2, 0)                       # retire row 0, compact last row into its slot
    assert c.offsets == [4, 2, 0]
    assert mx.allclose(c.keys[0:1, :, :4, :], pairs[2][0]).item()
    assert mx.allclose(c.values[0:1, :, :4, :], pairs[2][1]).item()


def test_rows_view_narrows_offset_and_update():
    c = BatchCache.from_rows(_rows([3, 1, 4]))
    c.rows = 2                             # active prefix = rows 0..1
    assert c.offset.tolist() == [3, 1]
    k = mx.random.normal((2, 2, 1, 4))     # update covers only the active width
    K, V = c.update_and_fetch(k, k)
    assert c.offsets == [4, 2, 4]          # row 2 untouched
    assert K.shape[0] == 2 and K.shape[2] == 4   # Lcur over the active rows only


if __name__ == "__main__":
    pytest.main([__file__, "-q"])


# --- Stage B: batched drafter helpers ---

def test_ctx_block_mask_hides_context_padding():
    from mlx_dspark.batch_engine import _ctx_block_mask
    m = _ctx_block_mask([2, 0], k=2)              # Lmax=2 -> [B,1,k,Lmax+k]=[2,1,2,4]
    assert m.shape == (2, 1, 2, 4) and m.dtype == mx.bool_
    g = m.astype(mx.int32).tolist()
    assert g[0][0][0] == [1, 1, 1, 1]             # row0 len2: ctx cols 0,1 + block cols 2,3
    assert g[1][0][0] == [0, 0, 1, 1]             # row1 len0: only the block region


def test_batched_ctx_pads_and_reports_lens():
    from mlx_dspark.batch_engine import _batched_ctx
    from mlx_dspark.model import CtxCache

    def mkrow(length, layers=2):
        out = []
        for _ in range(layers):
            c = CtxCache()
            c.append(mx.random.normal((1, 2, length, 4)), mx.random.normal((1, 2, length, 4)))
            out.append(c)
        return out

    rows = [mkrow(3), mkrow(1)]
    bctx, lens = _batched_ctx(rows)
    assert lens == [3, 1] and len(bctx) == 2
    assert bctx[0].k.shape == (2, 2, 3, 4)         # [B, n_kv, Lmax, D]
    assert mx.allclose(bctx[0].k[0:1, :, :3, :], rows[0][0].k).item()
    assert mx.allclose(bctx[0].k[1:2, :, :1, :], rows[1][0].k).item()


def test_batched_sample_block_no_markov_is_argmax():
    from mlx_dspark.batch_engine import _batched_sample_block

    class D:
        markov_head = None

    base = mx.zeros((2, 3, 5))
    for (r, c, v) in [(0, 0, 2), (0, 1, 4), (0, 2, 1), (1, 0, 0), (1, 1, 3), (1, 2, 2)]:
        base[r, c, v] = 1.0
    assert _batched_sample_block(D(), base, [9, 9]).tolist() == [[2, 4, 1], [0, 3, 2]]


def test_batched_sample_block_markov_batches_rows():
    from mlx_dspark.batch_engine import _batched_sample_block

    class FakeMarkov:
        def step_bias(self, prev):                 # [B] -> [B, V], strongly biases token prev+1
            V = 5
            return (mx.arange(V)[None, :] == ((prev + 1) % V)[:, None]).astype(mx.float32) * 100.0

    class D:
        markov_head = FakeMarkov()

    out = _batched_sample_block(D(), mx.zeros((2, 2, 5)), [0, 2])
    assert out.tolist() == [[1, 2], [3, 4]]        # each row follows its own markov chain
