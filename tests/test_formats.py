"""Checkpoint-format robustness: the loader must either load a checkpoint correctly or
refuse it with an error that names the real problem — never silently mis-parse.

Fixtures are condensed from real configs observed on HF (2026-07-07):
- vLLM "speculators" format (RedHatAI/GLM-5.2-speculator.dspark — note model_type "qwen3"!)
- full target model with embedded drafter (deepseek-ai/DeepSeek-V4-Pro-DSpark)
- DFlash+Markov community hybrid (Hikari07jp/DSpark-Gemma-4-31B-draft)
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx_dspark.config import DSparkConfig
from mlx_dspark.load import _route_target, load_dflash, load_drafter
from mlx_dspark.model import DSparkDrafter
from mlx_dspark.target import Target

# ---------------------------------------------------------------- config fixtures

QWEN3_MIN = {
    "model_type": "qwen3", "hidden_size": 16, "vocab_size": 32, "num_hidden_layers": 1,
    "intermediate_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1,
    "head_dim": 8, "block_size": 4, "mask_token_id": 3, "target_layer_ids": [0, 1],
    "num_target_layers": 2, "markov_rank": 4,
}

GEMMA4_MIN = {
    "model_type": "gemma4_text", "hidden_size": 16, "vocab_size": 32, "num_hidden_layers": 1,
    "intermediate_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1,
    "num_global_key_value_heads": 1, "head_dim": 8, "global_head_dim": 8, "block_size": 4,
    "mask_token_id": 3, "target_layer_ids": [0, 1], "num_target_layers": 2, "markov_rank": 4,
}

SPECULATORS = {  # vLLM speculators packaging — drafter backbone is ALSO model_type "qwen3".
    # Truncated transformer_layer_config: not enough to rebuild a backbone.
    "architectures": ["DSparkDraftModel"], "model_type": "qwen3",
    "speculators_model_type": "dspark",
    "speculators_config": {"algorithm": "dspark",
                           "verifier": {"architectures": ["GlmMoeDsaForCausalLM"]}},
    "block_size": 8, "markov_rank": 256, "aux_hidden_state_layer_ids": [1, 2, 3],
    "transformer_layer_config": {"hidden_size": 6144},
}

SPECULATORS_FULL = {  # makora-ai/gemma4-26b-a4b-dspark shape: complete, and reduced-vocab.
    # The verifier is gemma-4 but the drafter backbone really is a plain qwen3 block (its
    # weights carry a separate v_proj and no layer_scalar/sandwich norms), so "qwen3" here
    # is accurate rather than the inherited-noise case the qwen3_5 heads suffer from.
    "architectures": ["DSparkDraftModel"],
    "speculators_model_type": "dspark",
    "speculators_config": {"algorithm": "dspark",
                           "verifier": {"architectures": ["Gemma4ForConditionalGeneration"]}},
    "block_size": 7, "mask_token_id": 4, "draft_vocab_size": 8,
    "markov_rank": 4, "markov_head_type": "vanilla",
    "enable_confidence_head": True, "confidence_head_with_markov": True,
    "aux_hidden_state_layer_ids": [0, 1],
    "transformer_layer_config": {
        "model_type": "qwen3", "hidden_size": 16, "vocab_size": 32, "num_hidden_layers": 1,
        "intermediate_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 8, "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
    },
}

MUSE_SPECULATORS = {  # DaoCloud/Muse-Glimmer-30B-DSpark shape: a speculators DSpark head
    # warm-started from a DFlash assistant. Two things set it apart from the makora/mgoin
    # speculators heads above: (1) its qwen3 backbone attends CAUSALLY over a sliding window
    # (sliding_attention layer_types + use_sliding_window + sliding_window_non_causal False),
    # (2) it reuses the target's embed_tokens AND lm_head (the round-trip test drops both from
    # the checkpoint). draft_vocab_size == vocab_size => full vocab (no reduction).
    "architectures": ["DSparkDraftModel"],
    "speculators_model_type": "dspark",
    "speculators_config": {"algorithm": "dspark",
                           "proposal_methods": [{"speculative_tokens": 15}],
                           "verifier": {"architectures": ["MuseGlimmerForConditionalGeneration"]}},
    "block_size": 15, "mask_token_id": 201818, "draft_vocab_size": 32,
    "markov_rank": 4, "markov_head_type": "vanilla",
    "enable_confidence_head": True, "confidence_head_with_markov": True,
    "sample_from_anchor": True, "sliding_window_non_causal": False,
    "aux_hidden_state_layer_ids": [0, 1],
    "transformer_layer_config": {
        "model_type": "qwen3", "hidden_size": 16, "vocab_size": 32, "num_hidden_layers": 1,
        "intermediate_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 8, "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
        "layer_types": ["sliding_attention"], "use_sliding_window": True, "sliding_window": 2048,
    },
}

EMBEDDED_V4 = {  # full 893 GB target with dspark_* fields — not a standalone drafter
    "model_type": "deepseek_v4", "architectures": ["DeepseekV4ForCausalLM"],
    "hidden_size": 7168, "num_hidden_layers": 61, "vocab_size": 129280,
    "dspark_block_size": 5, "dspark_target_layer_ids": [58, 59, 60],
    "dspark_markov_rank": 512,
}

DFLASH_MARKOV_HYBRID = {  # DFlash block-16 backbone + a DSpark Markov head
    "model_type": "qwen3", "architectures": ["DFlashDraftModel"],
    "hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2,
    "num_key_value_heads": 1, "head_dim": 8, "intermediate_size": 32, "vocab_size": 32,
    "rms_norm_eps": 1e-6, "rope_theta": 1e6, "max_position_embeddings": 1024,
    "block_size": 16, "target_layer_ids": [0, 1], "num_target_layers": 2,
    "markov_rank": 256, "markov_head_type": "vanilla",
}

DFLASH_NESTED_CONFIG = {  # z-lab/Qwen3.6-35B-A3B-DFlash shape: the newer z-lab packaging puts
    # EVERY DFlash-specific field under `dflash_config`, including block_size — the older heads
    # (gemma4, Qwen3-4B) put block_size at the top level. Reading only the top level made this
    # head die with a bare KeyError instead of loading.
    "model_type": "qwen3", "architectures": ["DFlashDraftModel"],
    "hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2,
    "num_key_value_heads": 1, "head_dim": 8, "intermediate_size": 32, "vocab_size": 32,
    "rms_norm_eps": 1e-6, "max_position_embeddings": 1024, "num_target_layers": 2,
    "rope_parameters": {"rope_theta": 1e7, "rope_type": "default"},
    "layer_types": ["full_attention"], "sliding_window": 4096, "use_sliding_window": True,
    "dflash_config": {"block_size": 16, "mask_token_id": 3, "target_layer_ids": [0, 1]},
}

FORK_LABELLED_QWEN36 = {  # a real-world Qwen3.6-27B DSpark head as exported by a vLLM fork:
    # DeepSpec DSpark weights (block 7, markov + confidence, dense-qwen3 backbone) under a
    # "DFlashDraftModel" class label, with target-config noise copied in (gated attention /
    # linear-attention fields the drafter's own weights don't have). Distinguishing quirks:
    # block_size 7 (vs 16 for the DFlash+Markov hybrid above) and heads*head_dim != hidden_size
    # (32*128=4096 vs 5120 in the real one). Not a registered pair — this locks in that the
    # label and the noise never mis-parse into the drafter.
    "model_type": "qwen3", "architectures": ["DFlashDraftModel"],
    "hidden_size": 20, "vocab_size": 32, "num_hidden_layers": 1, "intermediate_size": 32,
    "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
    "block_size": 7, "mask_token_id": 3, "target_layer_ids": [0, 1], "num_target_layers": 2,
    "markov_rank": 4, "enable_confidence_head": True, "confidence_head_with_markov": True,
    "rope_parameters": {"rope_theta": 1e7, "rope_type": "default"},
    # the noise — must be ignored, never mis-parsed into the drafter:
    "attn_output_gate": True, "output_gate_type": "swish", "full_attention_interval": 4,
    "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_conv_kernel_dim": 4,
    "num_anchors": 64, "mtp_num_hidden_layers": 1,
    "dflash_config": {"mask_token_id": 3, "target_layer_ids": [0, 1], "markov_rank": 4},
}


ORNITH_QWEN35 = {  # stanleyphoong/Ornith-1.0-9B-DSpark shape: DeepSpec layout, qwen3_5-flavored
    # backbone — gated q_proj (2× out-features) + partial rotary 0.25 + target-config noise
    # (text_config/vision_config/mrope copied in; mrope is a text-only no-op).
    "model_type": "qwen3_5", "architectures": ["Qwen35DSparkModel"],
    "hidden_size": 16, "vocab_size": 32, "num_hidden_layers": 1, "intermediate_size": 32,
    "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
    "enable_qwen35_gated_q_proj": True, "partial_rotary_factor": 0.25,
    "block_size": 7, "mask_token_id": 3, "target_layer_ids": [0, 1], "num_target_layers": 2,
    "markov_rank": 4, "enable_confidence_head": True, "confidence_head_with_markov": True,
    "confidence_temperatures": [1.5, 0.9, 1.5, 1.6, 0.9, 1.4, 1.6],
    "rope_parameters": {"rope_theta": 1e7, "rope_type": "default",
                        "partial_rotary_factor": 0.25, "mrope_interleaved": True,
                        "mrope_section": [11, 11, 10]},
    "text_config": {"model_type": "qwen3_5_text"}, "vision_config": {"depth": 27},
}


SATGEZE_QWEN36 = {  # satgeze/Qwen3.6-27B-DSpark shape: DeepSpec-STOCK head (architectures
    # Qwen3DSparkModel) for a Qwen3.5/3.6 target, so its config is a deepcopy of the target's
    # and inherits partial_rotary_factor + mrope + gated/linear-attention fields. None of that
    # describes the drafter: the stock reference ropes the FULL head_dim (transformers'
    # Qwen3RotaryEmbedding) and uses plain (non-offset) RMSNorm. Also block_size 15.
    "model_type": "qwen3", "architectures": ["Qwen3DSparkModel"],
    "hidden_size": 20, "vocab_size": 32, "num_hidden_layers": 1, "intermediate_size": 32,
    "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
    "block_size": 15, "mask_token_id": 3, "target_layer_ids": [0, 1], "num_target_layers": 2,
    "markov_rank": 4, "enable_confidence_head": True, "confidence_head_with_markov": True,
    "partial_rotary_factor": 0.25,
    "rope_parameters": {"rope_theta": 1e7, "rope_type": "default",
                        "partial_rotary_factor": 0.25, "mrope_interleaved": True,
                        "mrope_section": [11, 11, 10]},
    "attn_output_gate": True, "output_gate_type": "swish", "full_attention_interval": 4,
    "linear_num_key_heads": 16, "linear_num_value_heads": 48, "num_anchors": 128,
}


SATGEZE_QWEN35_08B = {  # satgeze/Qwen3.5-0.8B-DSpark: same stock lineage, but model_type is
    # the target's "qwen3_5_text" — still NOT the qwen3_5-native fork (no gate weights, plain
    # norms), so partial_rotary_factor stays inert here too.
    "model_type": "qwen3_5_text", "architectures": ["Qwen3DSparkModel"],
    "hidden_size": 16, "vocab_size": 32, "num_hidden_layers": 1, "intermediate_size": 32,
    "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
    "block_size": 7, "mask_token_id": 3, "target_layer_ids": [0, 1], "num_target_layers": 2,
    "markov_rank": 4, "enable_confidence_head": True, "confidence_head_with_markov": True,
    "partial_rotary_factor": 0.25,
    "rope_parameters": {"rope_theta": 1e7, "rope_type": "default",
                        "partial_rotary_factor": 0.25, "mrope_interleaved": True,
                        "mrope_section": [11, 11, 10]},
    "attn_output_gate": True, "num_anchors": 512,
}


KOOPAH_QWEN36_35B_A3B = {  # Koopah/Qwen3.6-35B-A3B-NVFP4-DSPARK: DeepSpec-stock head for the
    # first MoE target here. Notably it is NOT a deepcopy of the target's config — no
    # partial_rotary_factor, no mrope, no gate/linear-attention fields — so it is the control
    # case for the inherited-noise bug class: nothing to ignore, and the result must be the
    # same full-width rope the satgeze head gets by ignoring them. block_size 8, 8 taps.
    "model_type": "qwen3", "architectures": ["Qwen3DSparkModel"],
    "hidden_size": 16, "vocab_size": 32, "num_hidden_layers": 1, "intermediate_size": 32,
    "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
    "block_size": 8, "mask_token_id": 3,
    "target_layer_ids": [0, 1], "num_target_layers": 2,
    "markov_rank": 4, "markov_head_type": "vanilla",
    "enable_confidence_head": True, "confidence_head_with_markov": True,
    "rope_parameters": {"rope_theta": 1e7, "rope_type": "default"},
    "num_anchors": 512, "tie_word_embeddings": False,
}


def _cfg_path(tmp_path, cfg: dict) -> str:
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return str(p)


def test_qwen3_config_parses(tmp_path):
    cfg = DSparkConfig.from_json(_cfg_path(tmp_path, QWEN3_MIN))
    assert cfg.family == "qwen3" and cfg.block_size == 4


def test_gemma4_config_parses(tmp_path):
    cfg = DSparkConfig.from_json(_cfg_path(tmp_path, GEMMA4_MIN))
    assert cfg.family == "gemma4" and cfg.target_layer_ids == [0, 1]


def test_fork_labelled_qwen36_shape_parses_as_qwen3_dspark(tmp_path):
    cfg = DSparkConfig.from_json(_cfg_path(tmp_path, FORK_LABELLED_QWEN36))
    assert cfg.family == "qwen3" and cfg.block_size == 7
    assert cfg.rope_theta == 1e7                      # from rope_parameters, not a default
    assert cfg.head_dim == 8 and cfg.hidden_size == 20  # heads*head_dim != hidden preserved
    assert not cfg.gated_q_proj and cfg.rope_dims is None  # no gate weights in that checkpoint


def test_ornith_qwen35_shape_parses_with_gate_and_partial_rope(tmp_path):
    cfg = DSparkConfig.from_json(_cfg_path(tmp_path, ORNITH_QWEN35))
    assert cfg.family == "qwen3" and cfg.gated_q_proj
    assert cfg.rope_dims == 2                         # head_dim 8 × partial_rotary 0.25
    assert cfg.rope_theta == 1e7
    assert cfg.offset_rms_norm                        # qwen3_5 house style


def test_stock_deepspec_head_ignores_inherited_partial_rotary(tmp_path):
    # The load-bearing distinction: partial_rotary_factor is inherited target-config noise on
    # a DeepSpec-stock head and real semantics only on the qwen3_5-native fork. Honoring it
    # here ropes a quarter of each head and quietly costs acceptance with no error anywhere
    # (measured: 1.29 -> 1.59 code / 1.36 -> 1.78 chat on the 0.8B head when put back to full).
    for shape in (SATGEZE_QWEN36, SATGEZE_QWEN35_08B):
        cfg = DSparkConfig.from_json(_cfg_path(tmp_path, shape))
        assert cfg.family == "qwen3"
        assert cfg.rope_dims is None                  # full head_dim, despite the 0.25 field
        assert not cfg.gated_q_proj and not cfg.offset_rms_norm
        assert cfg.rope_theta == 1e7
    assert DSparkConfig.from_json(_cfg_path(tmp_path, SATGEZE_QWEN36)).block_size == 15


def test_load_drafter_satgeze_block15_roundtrips(tmp_path):
    # block_size 15 (vs 7 everywhere else) is config-driven end to end — nothing may assume 7.
    path = _write_drafter_ckpt(tmp_path, _reference_weights(SATGEZE_QWEN36), cfg=SATGEZE_QWEN36)
    drafter, cfg = load_drafter(path, quantize=False)
    assert drafter.block_size == 15 and cfg.rope_dims is None
    q = drafter.layers[0].self_attn.q_proj.weight
    assert q.shape[0] == 2 * 8                        # heads × head_dim, NOT doubled (no gate)


def test_koopah_moe_head_parses_block8_full_rope(tmp_path):
    # A stock head whose config carries NO target noise at all: no partial rotary to ignore,
    # no gate, plain norms. Must land on exactly the same knobs the satgeze head reaches by
    # ignoring its inherited fields — full head_dim rope, no gate, no offset norms.
    cfg = DSparkConfig.from_json(_cfg_path(tmp_path, KOOPAH_QWEN36_35B_A3B))
    assert cfg.family == "qwen3" and cfg.block_size == 8
    assert cfg.rope_dims is None and not cfg.gated_q_proj and not cfg.offset_rms_norm
    assert len(cfg.target_layer_ids) == 2 and cfg.rope_theta == 1e7


def test_load_drafter_koopah_block8_roundtrips(tmp_path):
    # block 8 is a third distinct block size (7, 15, 8) — nothing may assume any of them.
    path = _write_drafter_ckpt(tmp_path, _reference_weights(KOOPAH_QWEN36_35B_A3B),
                               cfg=KOOPAH_QWEN36_35B_A3B)
    drafter, _cfg = load_drafter(path, quantize=False)
    assert drafter.block_size == 8
    q = drafter.layers[0].self_attn.q_proj.weight
    assert q.shape[0] == 2 * 8                        # heads × head_dim, not doubled (no gate)


def test_route_qwen3_5_moe_target_to_mlxlm_despite_vision_tower():
    # The MoE target ships architectures Qwen3_5MoeForConditionalGeneration + a vision_config,
    # but mlx-lm has a text-only module that digests it — and only the mlx-lm route has the
    # replicated-loop hidden-state tap the drafter needs. Routing it to mlx-vlm would lose it.
    assert _route_target({"model_type": "qwen3_5_moe", "vision_config": {"depth": 27},
                          "text_config": {"model_type": "qwen3_5_moe_text"}}) == "mlx_lm"


def test_plain_qwen3_unaffected_by_qwen35_knobs(tmp_path):
    cfg = DSparkConfig.from_json(_cfg_path(tmp_path, QWEN3_MIN))
    assert not cfg.gated_q_proj and cfg.rope_dims is None and not cfg.offset_rms_norm


def test_speculators_dspark_config_is_translated(tmp_path):
    """The speculators packaging is a rename of the DeepSpec schema, so it loads: the
    backbone comes out of transformer_layer_config and the fusion taps out of
    aux_hidden_state_layer_ids. Must run BEFORE family detection — a speculators head's
    top-level model_type is absent or misleading, and the real one is nested."""
    cfg = DSparkConfig.from_json(_cfg_path(tmp_path, SPECULATORS_FULL))
    assert cfg.family == "qwen3"
    assert (cfg.hidden_size, cfg.vocab_size, cfg.num_attention_heads) == (16, 32, 2)
    assert cfg.head_dim == 8 and cfg.num_key_value_heads == 1
    assert cfg.target_layer_ids == [0, 1]
    assert cfg.block_size == 7 and cfg.mask_token_id == 4
    assert cfg.rope_theta == 10000.0
    # plain qwen3 backbone: neither of the qwen3_5-native knobs, and full-width rope
    assert not cfg.gated_q_proj and not cfg.offset_rms_norm and cfg.rope_dims is None
    # and no softcap is inherited from the gemma-4 verifier
    assert cfg.final_logit_softcapping is None


def test_speculators_anchor_slot_is_read_from_the_declared_proposal_width(tmp_path):
    """logits_start = block_size - speculative_tokens, and 0 is a legitimate answer.

    A head that proposes as many tokens as it has block slots has no anchor slot at all
    (RedHatAI/Qwen3.6-35B-A3B-speculator.dspark: block 8, speculative_tokens 8, val_metrics
    position_0..position_7). Reading slot 1 there shifts every draft one position and costs
    ~2x the speedup with no error and correct output text, so this bound is load-bearing.
    """
    def start_for(**over):
        cfg = {**SPECULATORS_FULL, **over}
        return DSparkConfig.from_json(_cfg_path(tmp_path, cfg)).logits_start

    def proposal(n):
        return {"speculators_config": {**SPECULATORS_FULL["speculators_config"],
                                       "proposal_methods": [{"speculative_tokens": n}]}}

    assert start_for(block_size=8, **proposal(8)) == 0    # no anchor slot
    assert start_for(block_size=8, **proposal(7)) == 1    # mgoin: one anchor slot
    assert start_for(block_size=7, **proposal(6)) == 1    # makora: one anchor slot
    # Nonsense widths fall back to the DFlash-derived default rather than slicing negatively.
    assert start_for(block_size=7, **proposal(9)) == 1


def test_speculators_sample_from_anchor_only_counts_when_present(tmp_path):
    """The explicit field names the same choice, but the pydantic class defaults it to True
    while the heads that OMIT it do reserve an anchor slot — so absent must not read as True."""
    def start_for(**over):
        return DSparkConfig.from_json(_cfg_path(tmp_path, {**SPECULATORS_FULL, **over})).logits_start

    assert start_for(sample_from_anchor=True) == 0
    assert start_for(sample_from_anchor=False) == 1
    assert start_for() == 1                                    # absent -> anchor slot assumed
    # The declared width wins over the flag when both are present (it is the verifiable one).
    assert start_for(block_size=8, sample_from_anchor=False,
                     speculators_config={**SPECULATORS_FULL["speculators_config"],
                                         "proposal_methods": [{"speculative_tokens": 8}]}) == 0


def test_speculators_dflash_lineage_causal_sliding_window(tmp_path):
    """A speculators DSpark head warm-started from DFlash (Muse-Glimmer) declares
    sliding_attention layer_types + a causal window (sliding_window_non_causal False). The
    translation carries these as the causal-block + sliding-window markers the qwen3 branch
    already reads, so the drafter attends exactly as trained. A stock bidirectional head
    (full_attention, e.g. makora) is left untouched — running a causal backbone bidirectionally
    is lossless but silently costs acceptance, the same class of bug as the wrong anchor slot."""
    cfg = DSparkConfig.from_json(_cfg_path(tmp_path, MUSE_SPECULATORS))
    assert cfg.family == "qwen3"
    assert cfg.causal_block is True and cfg.sliding_window == 2048
    assert cfg.logits_start == 0            # sample_from_anchor True / block == speculative_tokens
    assert cfg.draft_vocab_size is None     # draft_vocab == vocab -> full vocab
    assert not cfg.attention_sink and cfg.rope_dims is None
    # regression: the bidirectional makora/mgoin heads (full_attention) stay unwindowed
    base = DSparkConfig.from_json(_cfg_path(tmp_path, SPECULATORS_FULL))
    assert base.causal_block is False and base.sliding_window is None


def test_load_drafter_reuses_target_embed_and_head_when_absent(tmp_path):
    """A DFlash-warm-started head (Muse-Glimmer) ships neither embed_tokens nor lm_head and
    reuses the target's. load_drafter reads that from the checkpoint (the weights are simply
    absent, mirroring nvfp4_convert's has_lm_head detection) and builds a drafter that binds
    both at run time — not a load error. Unbound use raises a directed message, not NoneType."""
    weights = _reference_weights(MUSE_SPECULATORS)
    del weights["embed_tokens.weight"], weights["lm_head.weight"]
    path = _write_drafter_ckpt(tmp_path, weights, MUSE_SPECULATORS)
    drafter, cfg = load_drafter(path, quantize=False)
    assert cfg.has_own_embed is False and cfg.has_own_lm_head is False
    assert drafter.embed_tokens is None and drafter.lm_head is None
    with pytest.raises(RuntimeError, match="bind_embed"):
        drafter.embed(mx.array([[1, 2, 3]]))
    with pytest.raises(RuntimeError, match="bind_lm_head"):
        drafter.compute_logits(mx.zeros((1, cfg.hidden_size)))


def test_speculators_reduced_draft_vocab_sizes_only_the_output_heads(tmp_path):
    """draft_vocab_size narrows lm_head/markov_w2 (draft space) but NOT embed_tokens or
    markov_w1, which are indexed by previously-accepted tokens — i.e. target ids."""
    cfg = DSparkConfig.from_json(_cfg_path(tmp_path, SPECULATORS_FULL))
    assert cfg.draft_vocab_size == 8 and cfg.out_vocab_size == 8 and cfg.vocab_size == 32

    shapes = dict(tree_flatten(DSparkDrafter(cfg).parameters()))
    assert shapes["lm_head.weight"].shape == (8, 16)
    assert shapes["markov_head.markov_w2.weight"].shape == (8, 4)
    assert shapes["embed_tokens.weight"].shape == (32, 16)
    assert shapes["markov_head.markov_w1.weight"].shape == (32, 4)


def test_full_vocab_head_is_unaffected_by_the_reduced_vocab_path(tmp_path):
    """No draft_vocab_size => every head stays full width and ids pass through unmapped."""
    cfg = DSparkConfig.from_json(_cfg_path(tmp_path, QWEN3_MIN))
    assert cfg.draft_vocab_size is None and cfg.out_vocab_size == cfg.vocab_size
    d = DSparkDrafter(cfg)
    ids = mx.array([3, 7])
    assert d.to_target_ids(ids).tolist() == [3, 7]
    q = mx.zeros((cfg.vocab_size,))
    assert d.widen_to_target(q).shape == (cfg.vocab_size,)


def test_speculators_truncated_backbone_refused_with_the_missing_fields(tmp_path):
    with pytest.raises(ValueError, match="transformer_layer_config missing"):
        DSparkConfig.from_json(_cfg_path(tmp_path, SPECULATORS))


def test_speculators_non_dspark_algorithm_refused(tmp_path):
    """eagle3/dflash heads are different models, not different spellings of this one."""
    cfg = {**SPECULATORS_FULL, "speculators_model_type": "eagle3",
           "speculators_config": {"algorithm": "eagle3"}}
    with pytest.raises(ValueError, match="only translates 'dspark'"):
        DSparkConfig.from_json(_cfg_path(tmp_path, cfg))


def test_embedded_full_model_refused_with_reason(tmp_path):
    with pytest.raises(ValueError, match="embedded"):
        DSparkConfig.from_json(_cfg_path(tmp_path, EMBEDDED_V4))


def test_unknown_family_refused_not_silently_gemma(tmp_path):
    with pytest.raises(ValueError, match="unsupported drafter family"):
        DSparkConfig.from_json(_cfg_path(tmp_path, {**QWEN3_MIN, "model_type": "llama"}))


def test_missing_model_type_refused(tmp_path):
    cfg = {k: v for k, v in QWEN3_MIN.items() if k != "model_type"}
    with pytest.raises(ValueError, match="unsupported drafter family"):
        DSparkConfig.from_json(_cfg_path(tmp_path, cfg))


def test_missing_required_field_refused(tmp_path):
    cfg = {k: v for k, v in QWEN3_MIN.items() if k != "target_layer_ids"}
    with pytest.raises(ValueError, match="target_layer_ids"):
        DSparkConfig.from_json(_cfg_path(tmp_path, cfg))


# ---------------------------------------------------------------- target routing

def test_route_multimodal_markers_to_vlm():
    assert _route_target({"model_type": "gemma4_unified", "vision_config": {}}) == "mlx_vlm"
    assert _route_target({"model_type": "whatever", "audio_config": {}}) == "mlx_vlm"


def test_route_mlxlm_families_to_mlxlm():
    for mt in ("qwen3", "qwen3_moe", "llama", "glm_moe_dsa", "deepseek_v3"):
        assert _route_target({"model_type": mt}) == "mlx_lm", mt


def test_route_respects_mlxlm_remap_table():
    assert _route_target({"model_type": "mistral"}) == "mlx_lm"  # remapped to llama


def test_route_unknown_falls_back_to_vlm():
    assert _route_target({"model_type": "totally_unknown_family"}) == "mlx_vlm"
    assert _route_target({}) == "mlx_vlm"


# ---------------------------------------------------------------- strict weight loading

def _write_drafter_ckpt(tmp_path, weights: dict, cfg: dict = QWEN3_MIN) -> str:
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    return str(tmp_path)


def _reference_weights(cfg_dict: dict = QWEN3_MIN) -> dict:
    class _P:  # only the fields DSparkConfig needs, via a real parse for fidelity
        pass
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cfgp = f"{d}/config.json"
        open(cfgp, "w").write(json.dumps(cfg_dict))
        cfg = DSparkConfig.from_json(cfgp)
    ref = DSparkDrafter(cfg)
    weights = dict(tree_flatten(ref.parameters()))
    mx.eval(list(weights.values()))
    return weights


def test_load_drafter_matching_checkpoint_loads(tmp_path):
    path = _write_drafter_ckpt(tmp_path, _reference_weights())
    drafter, cfg = load_drafter(path, quantize=False)
    assert cfg.family == "qwen3" and drafter.block_size == 4


def test_load_drafter_mismatch_raises_by_default(tmp_path):
    weights = _reference_weights()
    weights["bogus.weight"] = weights.pop("lm_head.weight")
    path = _write_drafter_ckpt(tmp_path, weights)
    with pytest.raises(ValueError, match="don't match"):
        load_drafter(path, quantize=False)


def test_load_drafter_strict_false_force_loads(tmp_path):
    weights = _reference_weights()
    weights["bogus.weight"] = weights.pop("lm_head.weight")
    path = _write_drafter_ckpt(tmp_path, weights)
    drafter, _ = load_drafter(path, quantize=False, strict=False)  # warn-and-load
    assert drafter is not None


def test_load_drafter_fork_labelled_qwen36_shape_roundtrips(tmp_path):
    # heads*head_dim != hidden_size (a real Qwen3.6 drafter quirk): construction and the
    # 1:1 key match must both come out of the config, with the fork-label/noise fields inert.
    path = _write_drafter_ckpt(tmp_path, _reference_weights(FORK_LABELLED_QWEN36), cfg=FORK_LABELLED_QWEN36)
    drafter, cfg = load_drafter(path, quantize=False)
    assert cfg.family == "qwen3" and drafter.block_size == 7


def test_load_drafter_ornith_qwen35_shape_roundtrips(tmp_path):
    # gated q_proj means 2× out-features on q_proj — construction and the 1:1 key match
    # must both track the flag (a plain-qwen3 build would fail with a q_proj shape mismatch).
    path = _write_drafter_ckpt(tmp_path, _reference_weights(ORNITH_QWEN35), cfg=ORNITH_QWEN35)
    drafter, cfg = load_drafter(path, quantize=False)
    assert cfg.gated_q_proj and drafter.block_size == 7
    q = drafter.layers[0].self_attn.q_proj.weight
    assert q.shape[0] == 2 * 2 * 8                    # heads × head_dim × 2 (q ‖ gate)


def test_load_drafter_qwen35_offset_norms_materialized(tmp_path):
    # qwen3_5 checkpoints store norm weights as offsets from one — load_drafter must add 1
    # (fresh-init reference norms are all-ones, so loaded values must come back as 2.0).
    # Un-offset qwen3 loads stay untouched (all-ones stay 1.0).
    import mlx.core as mx

    path = _write_drafter_ckpt(tmp_path, _reference_weights(ORNITH_QWEN35), cfg=ORNITH_QWEN35)
    drafter, _ = load_drafter(path, quantize=False)
    assert mx.allclose(drafter.hidden_norm.weight, mx.full_like(drafter.hidden_norm.weight, 2.0)).item()
    assert mx.allclose(drafter.layers[0].self_attn.q_norm.weight,
                       mx.full_like(drafter.layers[0].self_attn.q_norm.weight, 2.0)).item()

    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()
    path = _write_drafter_ckpt(plain_dir, _reference_weights(QWEN3_MIN), cfg=QWEN3_MIN)
    drafter, _ = load_drafter(path, quantize=False)
    assert mx.allclose(drafter.hidden_norm.weight, mx.full_like(drafter.hidden_norm.weight, 1.0)).item()


def test_load_dflash_markov_hybrid_refused_with_reason(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(DFLASH_MARKOV_HYBRID))
    with pytest.raises(ValueError, match="Markov head"):
        load_dflash(str(tmp_path))


def test_load_dflash_reads_block_size_nested_under_dflash_config(tmp_path):
    # z-lab's newer heads nest block_size with the other DFlash fields; the older ones put it
    # at the top level. Both must parse — this one used to raise a bare KeyError.
    (tmp_path / "config.json").write_text(json.dumps(DFLASH_NESTED_CONFIG))
    with pytest.raises(ValueError, match="tensor names"):   # got past the config, no weights
        load_dflash(str(tmp_path))


def test_load_dflash_without_block_size_anywhere_refused_with_reason(tmp_path):
    cfg = {k: v for k, v in DFLASH_NESTED_CONFIG.items() if k != "dflash_config"}
    cfg["target_layer_ids"] = [0, 1]
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="block_size"):
        load_dflash(str(tmp_path))


# ---------------------------------------------------------------- tap sanity probe

class _TinyLayer:
    def __init__(self, h):
        import mlx.nn as nn
        self.lin = nn.Linear(h, h, bias=False)

    def __call__(self, x, mask=None, cache=None):
        return x + self.lin(x)


class _TinyLM:
    """Minimal mlx-lm-shaped dense model: model.{embed_tokens,layers,norm} + lm_head."""

    def __init__(self, model_type="tinydense", embed_shift=0.0, **args):
        import mlx.nn as nn
        h, vocab = 16, 32
        self.model = SimpleNamespace(
            embed_tokens=nn.Embedding(vocab, h),
            layers=[_TinyLayer(h) for _ in range(2)],
            norm=nn.RMSNorm(h),
        )
        self.lm_head = nn.Linear(h, vocab, bias=False)
        self.args = SimpleNamespace(model_type=model_type, **args)
        self._shift = embed_shift

    @property
    def layers(self):
        return self.model.layers

    def __call__(self, ids, cache=None):
        h = self.model.embed_tokens(ids) + self._shift
        for layer, c in zip(self.model.layers, cache or [None, None]):
            h = layer(h, None, c)
        return self.lm_head(self.model.norm(h))


def test_verify_tap_passes_for_plain_dense_model():
    Target(_TinyLM(), tokenizer=None).verify_tap()  # must not raise


def test_verify_tap_catches_forward_divergence():
    # a family whose forward does more than embed→layers→norm (gemma-style embedding
    # treatment; an additive shift here because the tiny fake is positively homogeneous,
    # so a multiplicative scale would cancel through its RMSNorm)
    with pytest.raises(ValueError, match="diverges"):
        Target(_TinyLM(embed_shift=1.0), tokenizer=None).verify_tap()


def test_verify_tap_refuses_windowed_attention():
    model = _TinyLM(model_type="windowed_fam",
                    layer_types=["sliding_attention", "full_attention"])
    with pytest.raises(ValueError, match="windowed"):
        Target(model, tokenizer=None).verify_tap()


def test_verify_tap_refuses_sliding_window_flag():
    model = _TinyLM(model_type="sw_fam", use_sliding_window=True, sliding_window=2048)
    with pytest.raises(ValueError, match="windowed"):
        Target(model, tokenizer=None).verify_tap()


# ------------------------------------------------- reduced draft vocabulary (speculators)

def _d2t_offsets(target_ids: list[int]) -> mx.array:
    """Checkpoints store the map as OFFSETS from the draft id, not absolute target ids."""
    return mx.array([t - d for d, t in enumerate(target_ids)])


def test_set_draft_vocab_map_reads_offsets_not_absolute_ids(tmp_path):
    """`d2t` is an offset table: target = draft + d2t[draft]. Reading it as absolute ids
    would still "work" and silently emit the wrong token for every draft position."""
    cfg = DSparkConfig.from_json(_cfg_path(tmp_path, SPECULATORS_FULL))
    d = DSparkDrafter(cfg)
    targets = [3, 5, 6, 11, 12, 13, 20, 31]           # 8 draft slots -> target vocab of 32
    d.set_draft_vocab_map(_d2t_offsets(targets))
    assert d.to_target_ids(mx.array([0, 1, 7, 4])).tolist() == [3, 5, 31, 12]


def test_sample_block_reduced_vocab_emits_and_feeds_back_target_ids(tmp_path):
    """The greedy block must map the argmax before it is emitted AND before it becomes the
    next position's `prev` — markov_w1 is indexed by target ids, so feeding a raw draft id
    back reads the wrong row with no error anywhere."""
    cfg = DSparkConfig.from_json(_cfg_path(tmp_path, SPECULATORS_FULL))
    d = DSparkDrafter(cfg)
    targets = [3, 5, 6, 11, 12, 13, 20, 31]
    d.set_draft_vocab_map(_d2t_offsets(targets))

    seen = []

    class _Markov:
        def step_bias(self, prev):
            seen.append(int(prev.item()))
            return mx.zeros((1, cfg.out_vocab_size))

    d.markov_head = _Markov()
    base = mx.zeros((3, cfg.out_vocab_size))
    base[0, 1] = 1.0      # draft 1 -> target 5
    base[1, 7] = 1.0      # draft 7 -> target 31
    base[2, 0] = 1.0      # draft 0 -> target 3
    out = d.sample_block(base, first_prev_token=9)
    assert out.tolist() == [5, 31, 3]
    assert seen == [9, 5, 31]          # markov only ever sees target-space ids


def test_widen_to_target_moves_q_onto_the_target_index_space(tmp_path):
    """Speculative sampling compares q with the target's p elementwise and resamples from
    max(0, p - q) over the FULL target vocab, so q has to be widened rather than compared
    in draft space — tokens the draft head cannot represent must keep their residual mass."""
    cfg = DSparkConfig.from_json(_cfg_path(tmp_path, SPECULATORS_FULL))
    d = DSparkDrafter(cfg)
    targets = [3, 5, 6, 11, 12, 13, 20, 31]
    d.set_draft_vocab_map(_d2t_offsets(targets))

    q = mx.zeros((cfg.out_vocab_size,))
    q[1] = 0.25
    q[7] = 0.75
    wide = d.widen_to_target(q)
    assert wide.shape == (cfg.vocab_size,)
    assert abs(float(wide.sum()) - 1.0) < 1e-6          # mass preserved
    assert float(wide[5]) == 0.25 and float(wide[31]) == 0.75
    assert float(wide[0]) == 0.0                        # untouched target ids stay empty


def test_load_drafter_speculators_reduced_vocab_round_trips(tmp_path):
    weights = _reference_weights(SPECULATORS_FULL)
    weights["d2t"] = _d2t_offsets([3, 5, 6, 11, 12, 13, 20, 31])
    weights["t2d"] = mx.zeros((32,), dtype=mx.bool_)    # trainer-only; must be ignored
    path = _write_drafter_ckpt(tmp_path, weights, SPECULATORS_FULL)
    drafter, cfg = load_drafter(path, quantize=False)
    assert cfg.draft_vocab_size == 8 and cfg.family == "qwen3"
    assert drafter.to_target_ids(mx.array([0, 7])).tolist() == [3, 31]


def test_load_drafter_reduced_vocab_without_d2t_refused(tmp_path):
    """No table means draft ids would be emitted as target ids — every token wrong, and
    the drafter would still appear to run."""
    path = _write_drafter_ckpt(tmp_path, _reference_weights(SPECULATORS_FULL), SPECULATORS_FULL)
    with pytest.raises(ValueError, match="no `d2t` table"):
        load_drafter(path, quantize=False)


def test_load_drafter_d2t_without_declared_draft_vocab_refused(tmp_path):
    weights = _reference_weights()
    weights["d2t"] = mx.zeros((4,), dtype=mx.int32)
    path = _write_drafter_ckpt(tmp_path, weights)
    with pytest.raises(ValueError, match="declares no "):
        load_drafter(path, quantize=False)
