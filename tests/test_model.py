"""Model-free unit tests for the DSpark drafter building blocks."""

from __future__ import annotations

import mlx.core as mx

from mlx_dspark.model import CtxCache, DSparkAttention


class _AttnCfg:
    """Minimal config for a GQA DSparkAttention (n_rep = 8/2 = 4)."""
    num_attention_heads = 8
    n_kv_heads = 2
    attn_head_dim = 16
    attention_k_eq_v = False
    use_v_norm = False
    scaling = 16 ** -0.5
    hidden_size = 32
    attention_bias = False
    rms_norm_eps = 1e-6
    rope_theta = 1e6
    rope_parameters = None
    gated_q_proj = False
    rope_dims = None


class _GatedAttnCfg(_AttnCfg):
    """qwen3_5 flavor: q_proj emits [q ‖ gate] per head (full-dim rope, isolating the gate)."""
    gated_q_proj = True


def test_attend_uses_native_gqa_equivalent_to_tiling():
    """The drafter attention relies on SDPA's native GQA/MQA broadcast (no `_repeat_kv`): tiling
    the K/V up to full heads over the whole context every round was O(n_rep · ctx) of wasted
    bandwidth that collapsed long-context drafting. Guard that the shipped n_kv-head path is
    numerically identical to explicitly tiling — so a refactor can't silently break GQA or bring
    the tiling waste back without this test noticing."""
    mx.random.seed(0)
    attn = DSparkAttention(_AttnCfg())
    mx.eval(attn.parameters())

    cache = CtxCache()
    attn.update_ctx(mx.random.normal((1, 5, _AttnCfg.hidden_size)), 0, cache)   # 5 ctx positions
    hidden = mx.random.normal((1, 3, _AttnCfg.hidden_size))                      # 3-position block
    block_offset = cache.length

    shipped = attn.attend(hidden, block_offset, cache)                           # native GQA

    # reference: identical math but tile K/V to full heads before SDPA (the old `_repeat_kv`)
    B, q_len, _ = hidden.shape
    q = attn.q_proj(hidden).reshape(B, q_len, attn.n_heads, attn.head_dim)
    q = attn.rope(attn.q_norm(q).transpose(0, 2, 1, 3), offset=block_offset)
    k_blk, v_blk = attn._kv(hidden)
    k_blk = attn.rope(k_blk, offset=block_offset)
    k = mx.concatenate([cache.k, k_blk], axis=2)
    v = mx.concatenate([cache.v, v_blk], axis=2)
    n_rep = attn.n_heads // attn.n_kv_heads

    def tile(x):
        b, nkv, s, d = x.shape
        return mx.broadcast_to(mx.expand_dims(x, 2), (b, nkv, n_rep, s, d)).reshape(b, nkv * n_rep, s, d)

    ref = mx.fast.scaled_dot_product_attention(q, tile(k), tile(v), scale=attn.scale)
    ref = attn.o_proj(ref.transpose(0, 2, 1, 3).reshape(B, q_len, -1))

    assert shipped.shape == (1, 3, _AttnCfg.hidden_size)
    assert mx.allclose(shipped, ref, atol=1e-5).item()


def test_gated_q_proj_split_layout_and_gate_application():
    """qwen3_5-flavored drafters (Ornith): q_proj emits [q ‖ gate] interleaved per head and the
    attention output is scaled by sigmoid(gate) before o_proj. With the gate rows zeroed,
    sigmoid(0) = 0.5 — so the gated output must be exactly half the ungated output computed
    from the same q/k/v/o weights. This pins both the per-head split layout (checkpoint
    compatibility) and where the gate is applied."""
    mx.random.seed(1)
    gated = DSparkAttention(_GatedAttnCfg())
    plain = DSparkAttention(_AttnCfg())
    mx.eval(gated.parameters(), plain.parameters())

    H, D = _AttnCfg.num_attention_heads, _AttnCfg.attn_head_dim
    w = gated.q_proj.weight.reshape(H, 2 * D, -1)
    q_half = w[:, :D, :]
    gated.q_proj.weight = mx.concatenate(
        [q_half, mx.zeros_like(q_half)], axis=1).reshape(H * 2 * D, -1)
    plain.q_proj.weight = q_half.reshape(H * D, -1)
    for name in ("k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
        getattr(plain, name).weight = getattr(gated, name).weight

    ctx = mx.random.normal((1, 5, _AttnCfg.hidden_size))
    hidden = mx.random.normal((1, 3, _AttnCfg.hidden_size))
    cache_g, cache_p = CtxCache(), CtxCache()
    gated.update_ctx(ctx, 0, cache_g)
    plain.update_ctx(ctx, 0, cache_p)

    out_g = gated.attend(hidden, 5, cache_g)
    out_p = plain.attend(hidden, 5, cache_p)
    assert mx.allclose(out_g, 0.5 * out_p, atol=1e-6).item()


def test_partial_rotary_ropes_only_rope_dims():
    """rope_dims (qwen3_5 partial rotary, e.g. 64 of head_dim 256) must reach the rope op."""
    class _PartialCfg(_AttnCfg):
        rope_dims = 4  # head_dim 16 × 0.25

    attn = DSparkAttention(_PartialCfg())
    assert attn.rope.dims == 4
    assert DSparkAttention(_AttnCfg()).rope.dims == _AttnCfg.attn_head_dim


def _tiny_drafter(**over):
    from mlx_dspark.config import DSparkConfig
    from mlx_dspark.model import DSparkDrafter

    cfg = DSparkConfig(
        family="qwen3", hidden_size=32, vocab_size=64, num_hidden_layers=2,
        intermediate_size=48, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, attention_k_eq_v=False, block_size=8, mask_token_id=4,
        markov_rank=0, enable_confidence_head=False,
        final_logit_softcapping=None, mlp_activation="silu", norm_style="qwen",
        use_v_norm=False, target_layer_ids=[0, 1],
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    d = DSparkDrafter(cfg)
    mx.eval(d.parameters())
    return d


def test_draft_width_full_for_bidirectional_truncated_for_causal():
    """A bidirectional block (DeepSpec-native heads) must always run the trained full width —
    each position's hidden depends on the whole block. A causal block (DFlash-lineage:
    Nemotron logits_start=1, Muse logits_start=0) computes only the rows the head reads:
    logits_start + cap, clamped to the block."""
    assert _tiny_drafter().draft_width(2) == 8                       # bidirectional: full
    assert _tiny_drafter().draft_width(99) == 8
    d = _tiny_drafter(causal_block=True)                             # anchor-as-pos0 (muse)
    assert [d.draft_width(c) for c in (1, 2, 4, 99)] == [1, 2, 4, 8]
    d = _tiny_drafter(causal_block=True, logits_start=1)             # anchor slot (nemotron)
    assert [d.draft_width(c) for c in (1, 4, 99)] == [2, 5, 8]


def test_causal_backbone_truncation_matches_full_width_rows():
    """The load-bearing invariant behind draft_width: with a causal block mask, position i
    attends only positions <= i, so running the backbone at a truncated width must reproduce
    the full-width forward's first rows (this is what makes the truncation a pure speed
    lever — measured ~16 ms/round on Muse-Glimmer's 15-wide 2.3B backbone). Exercised with
    and without a sliding window over the context."""
    for window in (None, 6):
        d = _tiny_drafter(causal_block=True, sliding_window=window)
        ctx = d.make_ctx_cache()
        fused = mx.random.normal((1, 10, 2 * 32))
        d.update_context(fused, ctx_offset=0, ctx_caches=ctx)
        mx.eval([c.k for c in ctx])
        noise_full = mx.random.normal((1, 8, 32))
        cap = 3
        full = d.backbone(noise_full, 10, ctx)
        trunc = d.backbone(noise_full[:, :cap, :], 10, ctx)
        assert mx.allclose(full[:, :cap, :], trunc, atol=1e-5).item()
        # sanity: bidirectional would NOT truncate cleanly (later rows feed earlier ones)
    d = _tiny_drafter()
    ctx = d.make_ctx_cache()
    d.update_context(mx.random.normal((1, 10, 2 * 32)), ctx_offset=0, ctx_caches=ctx)
    noise = mx.random.normal((1, 8, 32))
    full = d.backbone(noise, 10, ctx)
    trunc = d.backbone(noise[:, :3, :], 10, ctx)
    assert not mx.allclose(full[:, :3, :], trunc, atol=1e-5).item()
