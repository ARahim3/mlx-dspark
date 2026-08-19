"""DFlash prefix caching: the ctx-window holder, the attention append_ctx refactor, and
the restore recipe (preset offsets + window append == the fresh path's single append).

The drafter's ctx state is recoverable from a bounded window of PROJECTED ctx rows
(sliding-window heads attend at most ``window - 1`` of them), which is what
:class:`DFlashCtxWindow` holds through the checkpoint-mode prefix cache. These tests pin:
the holder's ``.k``/``.v``/``trim_to`` contract against prefix_cache's snapshot/restore
machinery, that the ``append_ctx`` extraction is the same math the old inline ``__call__``
computed, and that window-restore + suffix-append reproduces the fresh path's attention —
including the rotation case (restored window + suffix overflowing the sliding window)."""
import mlx.core as mx
import numpy as np

from mlx_dspark.dflash_model import (
    DFlashConfig,
    DFlashCtxWindow,
    DFlashDraftModel,
    DFlashGroupedConv,  # noqa: F401  (import guard: same module, same load path)
)
from mlx_dspark.prefix_cache import PrefixCache, _restore, _snapshot


class FakeKV:  # matches tests/test_prefix_cache.py's fake target layer cache
    def __init__(self, offset=0):
        self.offset = offset
        self.state, self.meta_state = (), ()

    def trim(self, n):
        n = min(n, self.offset)
        self.offset -= n
        return n


def _tiny(sliding: int | None = None, layers: int = 2) -> DFlashDraftModel:
    mx.random.seed(7)
    cfg = DFlashConfig(
        hidden_size=32, num_hidden_layers=layers, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, intermediate_size=64, vocab_size=32,
        rms_norm_eps=1e-6, rope_theta=1e4, max_position_embeddings=512, block_size=4,
        target_layer_ids=(0, 1), num_target_layers=2, mask_token_id=3,
        layer_types=("sliding_attention",) * layers if sliding else (),
        sliding_window=sliding,
    )
    import mlx.nn as nn

    model = DFlashDraftModel(cfg)
    model.embed_tokens = nn.Embedding(32, 32)
    model.lm_head = nn.Linear(32, 32, bias=False)
    model.embed_scale = 1.0
    return model


# ---------------------------------------------------------------- window holder

def test_window_set_cap_and_positions():
    w = DFlashCtxWindow(cap=4)
    w.set(mx.arange(7 * 3, dtype=mx.float32).reshape(1, 7, 3), end=10)
    assert (w.start, w.end, w.rows) == (6, 10, 4)          # kept the last 4 of 7 rows
    assert float(w.k[0, 0, 0]) == 3 * 3.0                  # row index 3 of the original 7
    w2 = DFlashCtxWindow(cap=None)                         # unbounded (full-attention head)
    w2.set(mx.zeros((1, 7, 3)), end=7)
    assert (w2.start, w2.end, w2.rows) == (0, 7, 7)


def test_window_trim_partial_noop_and_empty():
    w = DFlashCtxWindow(cap=8)
    w.set(mx.random.normal((1, 6, 3)), end=20)             # covers [14, 20)
    w.trim_to(25)
    assert (w.start, w.end, w.rows) == (14, 20, 6)         # no-op past the end
    w.trim_to(17)
    assert (w.start, w.end, w.rows) == (14, 17, 3)         # partial
    w.trim_to(10)
    assert (w.rows, w.k) == (0, None)                      # too deep -> empty
    assert w.end == 10


def test_window_roundtrips_through_checkpoint_snapshot():
    w = DFlashCtxWindow(cap=8)
    rows = mx.random.normal((1, 5, 3))
    w.set(rows, end=12)
    snap = _snapshot([FakeKV(offset=12)], [w])
    _cache, ctx = _restore(lambda: [FakeKV()], lambda: [DFlashCtxWindow(cap=8)], snap)
    assert (ctx[0].start, ctx[0].end, ctx[0].rows) == (7, 12, 5)
    assert mx.allclose(ctx[0].k, rows).item()
    ctx[0].trim_to(9)                                      # restored copy is independent
    assert w.rows == 5 and ctx[0].rows == 2


def test_window_through_prefix_cache_boundary_and_rung_trim():
    pc = PrefixCache(lambda: [FakeKV()], lambda: [DFlashCtxWindow(cap=8)],
                     min_reuse=2, checkpoint=True)
    prompt = list(range(20))
    live_cache = [FakeKV(offset=16)]
    live_w = DFlashCtxWindow(cap=8)
    live_w.set(mx.random.normal((1, 8, 3)), end=16)
    pc.checkpoint(live_cache, [live_w], 16, prompt)        # boundary at 16
    # boundary hit: longer prompt sharing the first 16 tokens
    _cache, ctx, reuse = pc.acquire(prompt + [99, 98])
    assert reuse == 16
    assert (ctx[0].start, ctx[0].end, ctx[0].rows) == (8, 16, 8)


# ------------------------------------------------- attention refactor equivalence

def _orig_attention_call(attn, x, x_ctx, rope, cache):
    """The pre-refactor __call__ body, verbatim — the reference the extraction must match."""
    from mlx_lm.models.base import create_causal_mask

    B, L, _ = x.shape
    S = x_ctx.shape[1]
    if attn.is_sliding:
        keep_ctx = attn.sliding_window - 1
        if keep_ctx < S:
            skip = S - keep_ctx
            x_ctx = x_ctx[:, skip:]
            S = x_ctx.shape[1]
            cache.offset += skip
    queries = attn.q_proj(x)
    ctx_keys = attn.k_proj(x_ctx)
    ctx_values = attn.v_proj(x_ctx)
    prop_keys = attn.k_proj(x)
    prop_values = attn.v_proj(x)
    queries = attn.q_norm(queries.reshape(B, L, attn.n_heads, -1)).transpose(0, 2, 1, 3)
    ctx_keys = attn.k_norm(ctx_keys.reshape(B, S, attn.n_kv_heads, -1)).transpose(0, 2, 1, 3)
    ctx_values = ctx_values.reshape(B, S, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
    prop_keys = attn.k_norm(prop_keys.reshape(B, L, attn.n_kv_heads, -1)).transpose(0, 2, 1, 3)
    prop_values = prop_values.reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
    queries = rope(queries, offset=cache.offset + S)
    ctx_keys = rope(ctx_keys, offset=cache.offset)
    prop_keys = rope(prop_keys, offset=cache.offset + S)
    keys, values = cache.update_and_fetch(ctx_keys, ctx_values)
    ctx_len = keys.shape[2]
    keys = mx.concatenate([keys, prop_keys], axis=2)
    values = mx.concatenate([values, prop_values], axis=2)
    mask = None
    if attn.is_sliding:
        mask = (
            "causal" if ctx_len + L <= attn.sliding_window
            else create_causal_mask(L, offset=ctx_len, window_size=attn.sliding_window)
        )
    output = mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=attn.scale, mask=mask)
    return attn.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


def test_attention_refactor_matches_original_inline_math():
    for sliding, n_ctx in [(None, 10), (16, 10), (16, 40)]:   # full, sliding, sliding+skip
        model = _tiny(sliding=sliding)
        attn = model.layers[0].self_attn
        x = mx.random.normal((1, 4, 32))
        x_ctx = mx.random.normal((1, n_ctx, 32))
        c_new = model.make_cache()[0]
        c_ref = model.make_cache()[0]
        out_new = attn(x, x_ctx, model.rope, c_new)
        out_ref = _orig_attention_call(attn, x, x_ctx, model.rope, c_ref)
        assert np.allclose(np.array(out_new), np.array(out_ref), atol=1e-6), (sliding, n_ctx)
        assert c_new.offset == c_ref.offset


# ------------------------------------------------- restore recipe equivalence

def _forward(model, fused_parts, offsets_start=None, window_rows=None):
    """Run one draft block; optionally pre-restore a projected-row window first."""
    dcache = model.make_cache()
    if window_rows is not None:
        for c in dcache:
            c.offset = offsets_start
        model.append_ctx(window_rows, dcache)
    block = mx.array([[1] + [3] * 3])
    return model.forward_hidden(block, fused_parts, dcache, logits_start=1), dcache


def test_window_restore_reproduces_fresh_path():
    """Fresh (whole ctx in the first draft call) vs restore (append window rows at preset
    offsets, then the suffix in the first call) — same retained ctx, same rope positions,
    same block output. Cases: under the window, and overflowing it (rotation on append)."""
    for sliding, n, p in [(16, 12, 8),    # everything fits the window
                          (16, 24, 14),   # window + suffix overflow -> rotation on append
                          (None, 12, 8)]:  # full-attention: unbounded window
        model = _tiny(sliding=sliding)
        fused = mx.random.normal((1, n, 2 * 32))
        fresh, _ = _forward(model, fused)
        cap = (sliding - 1) if sliding else None
        w = DFlashCtxWindow(cap=cap)
        w.set(model.project_ctx(fused[:, :p]), end=p)
        restored, dcache = _forward(model, fused[:, p:],
                                    offsets_start=w.start, window_rows=w.k)
        assert np.allclose(np.array(fresh), np.array(restored), atol=1e-5), (sliding, n, p)
        assert dcache[0].offset == n, (sliding, n, p)


def test_append_ctx_advances_offsets_to_boundary():
    model = _tiny(sliding=16)
    w = DFlashCtxWindow(cap=15)
    w.set(model.project_ctx(mx.random.normal((1, 20, 2 * 32))), end=100)
    dcache = model.make_cache()
    for c in dcache:
        c.offset = w.start
    model.append_ctx(w.k, dcache)
    assert all(c.offset == 100 for c in dcache)
