"""Nanbeige4.2 (looped-depth transformer) — vendored module registration, drafter-config
detection, and the unrolled-index tap body. Model-free (tiny random weights)."""

import json

import mlx.core as mx

from mlx_dspark.config import DSparkConfig
from mlx_dspark.load import _route_target
from mlx_dspark.nanbeige_lm import Model, ModelArgs, register
from mlx_dspark.target import Target

# The Nanbeige/Nanbeige4.2-3B-DSpark shape, shrunk: SpecForge-style nested dflash_config
# tagged projector_type "dspark" (taps/mask hoisted from it), DSpark knobs at top level,
# plain qwen3 backbone, no yarn / sliding window / gate markers. The real head also omits
# embed_tokens AND lm_head (reuse-both — detected from the checkpoint, not the config).
NANBEIGE_DSPARK = {
    "architectures": ["Qwen3DSparkModel"],
    "model_type": "qwen3",
    "hidden_size": 16, "vocab_size": 64, "num_hidden_layers": 2,
    "intermediate_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1,
    "head_dim": 8, "rope_theta": 70000000, "rope_scaling": None,
    "rms_norm_eps": 1e-05, "block_size": 7, "num_target_layers": 4,
    "markov_rank": 4, "markov_head_type": "vanilla",
    "layer_types": ["full_attention", "full_attention"],
    "sliding_window": None, "use_sliding_window": False,
    "dflash_config": {
        "projector_type": "dspark", "mask_token_id": 63,
        "target_layer_ids": [0, 3],
        "enable_confidence_head": True, "confidence_head_with_markov": True,
        "markov_head_type": "vanilla", "markov_rank": 4,
        "attention_mode": "gqa",
    },
}


def _args(n_layers: int = 2, n_loops: int = 2) -> ModelArgs:
    return ModelArgs(
        model_type="nanbeige", hidden_size=16, num_hidden_layers=n_layers,
        intermediate_size=32, num_attention_heads=2, num_key_value_heads=1,
        rms_norm_eps=1e-5, vocab_size=64, head_dim=8, rope_theta=10000.0,
        num_loops=n_loops, loop_loss_weights=[], skip_loop_final_norm=False,
    )


def _tiny_nanbeige(**kw):
    mx.random.seed(7)
    model = Model(_args(**kw))
    # deterministic non-trivial weights (fresh modules initialize near-uniform)
    from mlx.utils import tree_map
    model.update(tree_map(lambda a: mx.random.normal(a.shape) * 0.05, model.parameters()))
    return model


def test_nanbeige_dspark_config_parses(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(NANBEIGE_DSPARK))
    cfg = DSparkConfig.from_json(str(p))
    assert cfg.family == "qwen3"
    assert cfg.target_layer_ids == [0, 3] and cfg.mask_token_id == 63   # hoisted
    assert cfg.num_target_layers == 4          # the UNROLLED count (loops x layers)
    assert cfg.logits_start == 0               # SpecForge DSpark: anchor-as-pos0
    assert cfg.block_size == 7
    assert cfg.rope_yarn is None and cfg.rope_dims is None
    assert not cfg.causal_block and cfg.sliding_window is None and not cfg.attention_sink
    assert not cfg.gated_q_proj and not cfg.offset_rms_norm
    assert cfg.rope_theta == 70000000


def test_register_makes_model_type_resolvable():
    register()                                  # idempotent; True or False both fine here
    import importlib
    mod = importlib.import_module("mlx_lm.models.nanbeige")
    assert mod.Model is Model and mod.ModelArgs is ModelArgs
    assert register() is False                  # second call is a no-op
    assert _route_target({"model_type": "nanbeige"}) == "mlx_lm"


def test_make_cache_is_per_loop():
    t = Target(_tiny_nanbeige(), tokenizer=None)
    assert t.family == "nanbeige" and t._nanbeige and not t.is_hybrid
    cache = t.make_cache()
    assert len(cache) == 4                      # num_loops * num_hidden_layers


def test_verify_tap_proves_the_looped_body():
    # The load-time probe compares the replicated looped body (per-loop cache slices +
    # per-loop final norm) against the model's own forward — bit-faithfulness or a raise.
    t = Target(_tiny_nanbeige(), tokenizer=None)
    t.verify_tap()


def test_tap_indexes_the_unrolled_stream():
    t = Target(_tiny_nanbeige(), tokenizer=None)
    ids = mx.array([[1, 2, 3]])
    # tap 2 = loop 1, layer 0 — beyond len(layers); the generic dense body would silently
    # skip it and concatenate one capture short. The looped body must return both.
    logits, fused = t.run(ids, t.make_cache(), [0, 2])
    assert fused.shape == (1, 3, 2 * 16)
    # the two captures come from different loop passes, so they must differ
    a, b = fused[..., :16], fused[..., 16:]
    assert float(mx.abs(a - b).max()) > 0
    ref = t.model(ids, cache=t.make_cache())
    assert logits.shape == ref.shape
    assert float(mx.abs(logits - ref).max()) == 0.0


def test_tap_at_loop_boundary_is_post_norm():
    # The reference (Nanbeige/sglang@nbg42) captures tap k before unrolled step k+1, so a
    # tap at the last layer of a non-final loop sees the value ENTERING the next loop —
    # after the inter-loop RMSNorm — not the raw layer output. Feeding the pre-norm value
    # is the silent acceptance-loss class; pin the boundary semantics.
    t = Target(_tiny_nanbeige(), tokenizer=None)
    ids = mx.array([[1, 2, 3]])
    _, fused = t.run(ids, t.make_cache(), [1])          # boundary tap: end of loop 0
    mm = t.model.model
    h = mm.embed_tokens(ids)
    from mlx_lm.models.base import create_attention_mask
    cache = t.make_cache()
    mask = create_attention_mask(h, cache[0])
    for layer, c in zip(mm.layers, cache[:2]):
        h = layer(h, mask, c)
    boundary = mm.norm(h)                                # what enters loop 1
    assert float(mx.abs(fused - boundary).max()) == 0.0


def test_tap_at_last_unrolled_layer_is_refused():
    t = Target(_tiny_nanbeige(), tokenizer=None)
    ids = mx.array([[1, 2, 3]])
    try:
        t.run(ids, t.make_cache(), [3])                  # last unrolled layer: unreachable
    except ValueError as e:
        assert "not capturable" in str(e)
    else:
        raise AssertionError("expected ValueError for an uncapturable tap id")


def test_dense_rollback_trims_every_loop_cache():
    t = Target(_tiny_nanbeige(), tokenizer=None)
    cache = t.make_cache()
    t.verify(mx.array([[1, 2, 3, 4, 5]]), cache, [0, 2])
    assert all(c.offset == 5 for c in cache)
    t.rollback(cache, n_rejected=2, accepted=[2, 3])
    assert all(c.offset == 3 for c in cache)
