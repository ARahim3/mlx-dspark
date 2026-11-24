"""The DFlash drafter's ctx window (:class:`CtxWindowCache`): rows in temporal order in one buffer
that the block attention reads in place.

Pinned here: the buffer mechanics (fill, slide, compaction, the round's scratch rows), that what
a block sees does not depend on HOW its ctx arrived — one row at a time, as after rounds that
accepted nothing, or all at once (the regression: mlx-lm's ring put the newest row where the
positional mask hid it from every draft slot) — and the per-head ``block_window`` switch: the
whole block on its last row's window, no array mask, a call the multi-row kernel can take."""
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_dspark.calibrate import _drafter_attn_dims
from mlx_dspark.dflash_model import (
    CtxWindowCache,
    DFlashAttention,
    DFlashConfig,
    DFlashDraftModel,
    _build_rope,
)
from mlx_dspark.multirow_attn import MultirowConfig, Window, eligible

H, HEAD, WINDOW, BLOCK = 32, 16, 16, 8


def _cfg(block_causal=False, window=WINDOW, layers=2, block=BLOCK) -> DFlashConfig:
    return DFlashConfig(
        hidden_size=H, num_hidden_layers=layers, num_attention_heads=2, num_key_value_heads=1,
        head_dim=HEAD, intermediate_size=64, vocab_size=32, rms_norm_eps=1e-6, rope_theta=1e4,
        max_position_embeddings=4096, block_size=block, target_layer_ids=(0, 1),
        num_target_layers=2, mask_token_id=3, layer_types=("sliding_attention",) * layers,
        sliding_window=window, block_causal=block_causal)


def _attn(block_causal=False, window=WINDOW, seed=3) -> DFlashAttention:
    mx.random.seed(seed)
    a = DFlashAttention(_cfg(block_causal, window), 0)
    mx.eval(a.parameters())
    return a


def _rows(a, b, scratch=0):
    """ctx rows numbered a..b-1 followed by ``scratch`` rows of -1 (a round's block rows)."""
    r = mx.concatenate([mx.arange(a, b, dtype=mx.float32), -mx.ones((scratch,))])
    r = r.reshape(1, 1, b - a + scratch, 1)
    return r, r * 2


# ---------------------------------------------------------------- the buffer

def test_window_holds_the_last_rows_in_order_through_fill_slide_and_compaction():
    rng = np.random.default_rng(0)
    for max_size in (None, 5, 15):
        c = CtxWindowCache(max_size)
        c.step = 4                                         # compact every few rows
        n, buffers = 0, set()
        for _ in range(60):
            m = int(rng.choice([1, 1, 2, 3, 7]))           # one-row rounds are the common wrap
            scratch = int(rng.choice([0, 4, 8]))
            k, v, ctx_len = c.put(*_rows(n, n + m, scratch), m)
            n += m
            want = list(range(0 if max_size is None else max(0, n - max_size), n))
            held = [int(x) for x in np.array(k).reshape(-1)]
            assert held == want + [-1] * scratch, (max_size, n, held)
            assert ctx_len == len(want) == c.rows and c.offset == n
            assert np.array_equal(np.array(v), np.array(k) * 2)
            buffers.add(int(c.keys.shape[2]))
        if max_size is not None:                           # bounded: the buffer never grows
            assert max(buffers) <= max_size + 7 + 8 + c.step + 8


def test_an_append_wider_than_the_window_keeps_its_last_rows():
    c = CtxWindowCache(4)
    k, _ = c.update_and_fetch(*_rows(0, 10))
    assert [int(x) for x in np.array(k).reshape(-1)] == [6, 7, 8, 9]
    k, _, ctx_len = c.put(*_rows(10, 16, scratch=2), 6)    # in place this time, wider again
    assert [int(x) for x in np.array(k).reshape(-1)] == [12, 13, 14, 15, -1, -1] and ctx_len == 4


def test_a_rounds_scratch_rows_never_enter_the_window():
    c = CtxWindowCache(6)
    c.put(*_rows(0, 3, scratch=8), 3)
    k, _ = c.update_and_fetch(*_rows(3, 4))                # overwrites the first scratch row
    assert [int(x) for x in np.array(k).reshape(-1)] == [0, 1, 2, 3]


# ---------------------------------------------------------------- what a block sees

def _block_out(attn, rope, ctx, x, chunks, block_window=False):
    """The block's output after ``ctx`` arrived in ``chunks`` (row counts): every chunk but the
    last through ``append_ctx``, the last with the block."""
    attn.block_window = block_window
    cache = CtxWindowCache(attn.sliding_window - 1)
    at = 0
    for n in chunks[:-1]:
        attn.append_ctx(ctx[:, at: at + n], rope, cache)
        at += n
    assert at + chunks[-1] == ctx.shape[1]
    return attn(x, ctx[:, at:], rope, cache)


@pytest.mark.parametrize("block_causal", [False, True])
def test_one_row_appends_show_a_block_the_same_ctx_as_one_append(block_causal):
    """Past the window, ctx fed a row at a time (rounds that accepted nothing) must leave every
    block row seeing what one multi-row append shows it — in particular the NEWEST rows."""
    attn = _attn(block_causal)
    rope = _build_rope(HEAD, 1e4, 4096, None)
    n = 40                                                   # 2.5 windows
    ctx, x = mx.random.normal((1, n, H)), mx.random.normal((1, BLOCK, H))
    at_once = _block_out(attn, rope, ctx, x, [n])
    for chunks in ([n - 3, 1, 1, 1], [20] + [1] * 20, [1] * n, [n - 9, 4, 1, 3, 1]):
        got = _block_out(attn, rope, ctx, x, chunks)
        assert np.allclose(np.array(got), np.array(at_once), atol=1e-5), chunks


def test_every_draft_row_attends_the_newest_ctx_row_after_a_one_row_append():
    """The defect itself: change ONLY the newest ctx row and every block row must move."""
    attn = _attn(block_causal=False)
    rope = _build_rope(HEAD, 1e4, 4096, None)
    n = 30
    ctx, x = mx.random.normal((1, n, H)), mx.random.normal((1, BLOCK, H))
    other = mx.concatenate([ctx[:, :-1], ctx[:, -1:] + 3.0], axis=1)
    a = _block_out(attn, rope, ctx, x, [n - 1, 1])
    b = _block_out(attn, rope, other, x, [n - 1, 1])
    moved = np.abs(np.array(a) - np.array(b)).max(axis=-1)[0]
    assert (moved > 1e-4).all(), moved


# ---------------------------------------------------------------- block_window

def _capture_sdpa(monkeypatch):
    calls = []
    real = mx.fast.scaled_dot_product_attention

    def spy(q, k, v, *, scale, mask=None, **kw):
        calls.append((q, k, v, mask))
        return real(q, k, v, scale=scale, mask=mask, **kw)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", spy)
    return calls


@pytest.mark.parametrize("block_causal", [False, True])
def test_block_window_is_the_last_rows_window_and_needs_no_mask(monkeypatch, block_causal):
    attn = _attn(block_causal)
    rope = _build_rope(HEAD, 1e4, 4096, None)
    n = 40
    ctx, x = mx.random.normal((1, n, H)), mx.random.normal((1, BLOCK, H))
    calls = _capture_sdpa(monkeypatch)
    per_row = _block_out(attn, rope, ctx, x, [n - 1, 1])
    assert isinstance(calls[-1][3], mx.array)                # a window per row: an array mask
    shared = _block_out(attn, rope, ctx, x, [n - 1, 1], block_window=True)
    _, k, _, mask = calls[-1]
    assert mask == ("causal" if block_causal else None)
    assert k.shape[2] == WINDOW                              # (window - L) ctx rows + the block
    # the last block row's window is the same either way; earlier rows gave up their oldest
    assert np.allclose(np.array(shared[:, -1]), np.array(per_row[:, -1]), atol=1e-5)
    assert not np.allclose(np.array(shared[:, 1]), np.array(per_row[:, 1]), atol=1e-5)
    # ... and it IS the plain attention over exactly those rows: a fresh cache holding only the
    # last (window - L) ctx rows, at their absolute positions
    attn.block_window = False
    kept = WINDOW - BLOCK
    cache = CtxWindowCache(WINDOW - 1)
    cache.skip(n - kept)
    ref = attn(x, ctx[:, n - kept:], rope, cache)
    assert np.allclose(np.array(shared), np.array(ref), atol=1e-5)


def test_block_window_changes_nothing_while_ctx_and_block_fit_the_window():
    attn = _attn()
    rope = _build_rope(HEAD, 1e4, 4096, None)
    ctx, x = mx.random.normal((1, WINDOW - BLOCK, H)), mx.random.normal((1, BLOCK, H))
    off = _block_out(attn, rope, ctx, x, [4, 1, 3])
    on = _block_out(attn, rope, ctx, x, [4, 1, 3], block_window=True)
    assert np.array_equal(np.array(on), np.array(off))


def test_a_block_window_call_is_one_the_multirow_kernel_takes(monkeypatch):
    window = 64
    attn = _attn(window=window)
    attn.set_dtype(mx.bfloat16)
    rope = _build_rope(HEAD, 1e4, 4096, None)
    ctx = mx.random.normal((1, 100, H)).astype(mx.bfloat16)
    x = mx.random.normal((1, BLOCK, H)).astype(mx.bfloat16)
    cfg = MultirowConfig((Window(2, 1, HEAD, min_q=2, max_q=16, min_kv=16),))
    calls = _capture_sdpa(monkeypatch)
    for on in (False, True):
        attn.block_window = on
        attn(x, ctx, rope, CtxWindowCache(window - 1))
        q, k, v, mask = calls[-1]
        assert eligible(q, k, v, mask, None, cfg) is on


def _bound(target):
    mx.random.seed(1)
    return DFlashDraftModel(_cfg()).bind(target)


def test_block_window_is_off_unless_the_pair_was_measured_with_it(monkeypatch):
    monkeypatch.delenv("MLX_DSPARK_DFLASH_BLOCK_WINDOW", raising=False)
    target = SimpleNamespace(embed_tokens=nn.Embedding(32, H), lm_head=nn.Linear(H, 32, bias=False))
    d = _bound(target)
    assert not any(layer.self_attn.block_window for layer in d.layers)
    assert _drafter_attn_dims(d) is None                     # array masks: no kernel probe
    monkeypatch.setenv("MLX_DSPARK_DFLASH_BLOCK_WINDOW", "1")   # the A/B, any head
    d = _bound(target)
    assert all(layer.self_attn.block_window for layer in d.layers)
    assert _drafter_attn_dims(d) == (2, 1, HEAD)
    d.share_block_window(False)
    assert _drafter_attn_dims(d) is None


def test_block_window_env_can_turn_a_measured_head_off(monkeypatch):
    from mlx_dspark.dflash_model import _block_window_default

    monkeypatch.delenv("MLX_DSPARK_DFLASH_BLOCK_WINDOW", raising=False)
    assert _block_window_default(measured=True) and not _block_window_default(measured=False)
    monkeypatch.setenv("MLX_DSPARK_DFLASH_BLOCK_WINDOW", "0")
    assert not _block_window_default(measured=True)
    monkeypatch.setenv("MLX_DSPARK_DFLASH_BLOCK_WINDOW", "1")
    assert _block_window_default(measured=False)
