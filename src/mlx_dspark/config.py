"""DSpark drafter config — loaded from the HF checkpoint's config.json.

Supports two drafter families with a shared inference path:
  - gemma4  (gemma4_text): k_eq_v attention, v_norm, partial/proportional rope,
            sandwich norms + layer_scalar, gelu-tanh MLP, logit softcap.
  - qwen3   (qwen3):       standard GQA (separate v_proj, no v_norm), default rope,
            Llama-style 2-norm layer, silu MLP, no softcap. Also covers qwen3_5-flavored
            backbones (model_type qwen3_5, e.g. Ornith drafters) via two config-driven
            knobs: gated_q_proj (q_proj emits [q ‖ gate], attn out × sigmoid(gate)) and
            rope_dims (partial rotary).
Only the fields the MLX inference path needs are pulled out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def _translate_speculators(c: dict, path) -> dict:
    """Rewrite a vLLM-``speculators`` DSpark config into the DeepSpec schema.

    The two packagings describe the same drafter with different field names — speculators
    nests the backbone under ``transformer_layer_config`` and calls the fusion taps
    ``aux_hidden_state_layer_ids`` — so this is a rename, not a port. Verified against
    ``makora-ai/gemma4-26b-a4b-dspark`` and ``mgoin/Qwen3-8B-speculator.dspark``: their
    tensor names are already the DeepSpec ones (``fc`` / ``hidden_norm`` / ``layers.N.*`` /
    ``markov_head.markov_w{1,2}`` / ``confidence_head.proj``), and the 5-tap fusion layout
    (``fc`` in-features = 5 x hidden) matches every DeepSpec head we ship.

    The one genuine addition is ``draft_vocab_size`` (+ the ``d2t`` table in the weights);
    see ``DSparkConfig.draft_vocab_size``.

    Only ``dspark`` is translatable — the other algorithms in that repo (eagle/eagle3/
    dflash) are different models, not different spellings of this one.
    """
    spec = c.get("speculators_config") or {}
    algo = c.get("speculators_model_type") or spec.get("algorithm")
    if algo != "dspark":
        raise ValueError(
            f"{path}: this is a vLLM 'speculators' checkpoint for algorithm {algo!r}, and "
            f"mlx-dspark only translates 'dspark' heads. A z-lab DFlash drafter loads via "
            f"--mode dflash; for anything else, drafter-free speculation works with any "
            f"target via --mode lookup / --mode auto."
        )
    tlc = c.get("transformer_layer_config")
    taps = c.get("aux_hidden_state_layer_ids")
    if not tlc or not taps:
        raise ValueError(
            f"{path}: speculators 'dspark' config is missing "
            f"{'transformer_layer_config' if not tlc else 'aux_hidden_state_layer_ids'} — "
            f"without it the drafter backbone/fusion taps are undefined."
        )

    # The backbone fields live in transformer_layer_config; a head that omits them cannot be
    # built at all, so say which ones rather than dying on a KeyError downstream.
    need = ("hidden_size", "vocab_size", "num_hidden_layers", "intermediate_size",
            "num_attention_heads")
    absent = [k for k in need if k not in tlc]
    if absent:
        raise ValueError(
            f"{path}: speculators 'dspark' config has a transformer_layer_config missing "
            f"{absent} — the drafter backbone is underspecified. mlx-dspark reads the "
            f"backbone from that block; a head that omits these cannot be reconstructed."
        )

    heads = tlc["num_attention_heads"]
    vocab = tlc["vocab_size"]
    draft_vocab = c.get("draft_vocab_size")
    out = {
        # The backbone model_type is the one that describes the drafter's own layers. For
        # the gemma-4 heads this says "qwen3" even though the TARGET is gemma4 — that is
        # accurate, not inherited noise: their weights carry a separate v_proj and no
        # layer_scalar / sandwich norms, i.e. a plain qwen3 block.
        "model_type": tlc.get("model_type", "qwen3"),
        "hidden_size": tlc["hidden_size"],
        "vocab_size": vocab,
        # Equal sizes mean no reduction; keep None so the full-vocab path stays untouched.
        "draft_vocab_size": draft_vocab if draft_vocab and draft_vocab != vocab else None,
        "num_hidden_layers": tlc["num_hidden_layers"],
        "intermediate_size": tlc["intermediate_size"],
        "num_attention_heads": heads,
        "num_key_value_heads": tlc.get("num_key_value_heads", heads),
        "head_dim": tlc.get("head_dim", tlc["hidden_size"] // heads),
        "rms_norm_eps": tlc.get("rms_norm_eps", 1e-6),
        "attention_bias": tlc.get("attention_bias", False),
        "rope_parameters": tlc.get("rope_parameters") or {},
        "pad_token_id": tlc.get("pad_token_id") or 0,
        "target_layer_ids": list(taps),
        "num_target_layers": max(taps) + 1,   # unused by the DSpark path; kept for parity
    }
    # DSpark's own knobs already sit at the top level in both schemas.
    for k in ("block_size", "mask_token_id", "markov_rank", "markov_head_type",
              "enable_confidence_head", "confidence_head_with_markov",
              "final_logit_softcapping"):
        if k in c:
            out[k] = c[k]

    # DFlash-lineage CAUSAL SLIDING-WINDOW block attention. A head warm-started from a DFlash
    # assistant (e.g. DaoCloud/Muse-Glimmer-30B-DSpark: "5×Qwen3 GQA, causal SWA 2048") attends
    # causally over a sliding window, declared in the backbone config as `sliding_attention`
    # layer_types + use_sliding_window + sliding_window, with `sliding_window_non_causal` telling
    # causal from bidirectional. Carry these as the DeepSpec-schema markers the qwen3 branch of
    # from_json already reads (dflash_query_causal / sliding_window). A stock DeepSpec head
    # (makora/mgoin/GLM) declares `full_attention` (or nothing) and is left fully bidirectional
    # and unwindowed — running a causal-trained backbone bidirectionally is lossless but costs
    # acceptance, the same class of silent regression as reading the wrong anchor slot.
    lt = tlc.get("layer_types") or []
    if "sliding_attention" in lt and tlc.get("use_sliding_window") and tlc.get("sliding_window"):
        out["sliding_window"] = int(tlc["sliding_window"])
        # sliding_window_non_causal=False => the window is causal (DFlash query-causal). Absent
        # defaults to causal, the DFlash norm for a sliding backbone.
        out["dflash_query_causal"] = not bool(c.get("sliding_window_non_causal", False))
    if c.get("attention_sink_bias"):
        out["attention_sink_bias"] = True

    # How many of the block's slots carry predictions. DFlash-derived heads reserve slot 0 as
    # a pure anchor and predict from slot 1 (the `logits_start=1` convention dflash_model.py
    # documents); DeepSpec's own convention samples the anchor slot too, so predictions start
    # at 0. The checkpoint's declared proposal width settles it, and each head's val_metrics
    # confirms it independently: makora's block_size 7 with speculative_tokens 6 and mgoin's
    # block_size 8 with seven position_N_acc entries both imply exactly one anchor slot, while
    # RedHatAI/Qwen3.6-35B-A3B-speculator.dspark declares block_size 8 with speculative_tokens
    # 8 and reports position_0..position_7 — eight predictions from eight slots, no anchor.
    #
    # So `block - speculative_tokens == 0` is a legitimate answer, not a malformed one. Do not
    # re-tighten this bound: reading from the wrong slot shifts every draft by one position
    # and roughly halves the speedup, with no error anywhere and correct output text (the
    # target verifies the drafts) — only the accept length shows it.
    props = (c.get("speculators_config") or {}).get("proposal_methods") or []
    spec_tokens = props[0].get("speculative_tokens") if props else None
    block = out.get("block_size")
    start = None
    if spec_tokens and block and 0 <= block - spec_tokens < block:
        start = block - spec_tokens
    elif isinstance(c.get("sample_from_anchor"), bool):
        # Newer speculators configs name the same choice directly. Honor it only when it is
        # actually present: the pydantic class defaults it to True, but the heads that omit
        # the field predate it and DO reserve an anchor slot, so reading "absent" as True
        # would break exactly the two checkpoints that validated this path.
        start = 0 if c["sample_from_anchor"] else 1
    out["logits_start"] = 1 if start is None else start
    return out


@dataclass
class DSparkConfig:
    family: str = "gemma4"             # "gemma4" | "qwen3"

    # core dims
    hidden_size: int = 3840
    vocab_size: int = 262144
    # Reduced draft vocabulary (EAGLE-3 style, used by the vLLM-speculators heads): lm_head
    # and markov_w2 emit `draft_vocab_size` logits over a frequent-token subset instead of the
    # full target vocab, and a `d2t` offset table maps a draft id back to a target id
    # (target = draft + d2t[draft]). embed_tokens and markov_w1 stay TARGET-indexed — they
    # consume previously *accepted* tokens, which are target ids. None = no reduction, which
    # is every DeepSpec-native head.
    draft_vocab_size: int | None = None
    num_hidden_layers: int = 5
    intermediate_size: int = 15360
    rms_norm_eps: float = 1e-6

    # attention
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    num_global_key_value_heads: int = 1
    head_dim: int = 256
    global_head_dim: int = 512
    attention_k_eq_v: bool = True
    attention_bias: bool = False

    # rope
    rope_theta: float = 1_000_000.0
    partial_rotary_factor: float = 0.25
    rope_type: str = "proportional"
    # qwen3_5 drafters rope only the first rope_dims of head_dim (partial rotary);
    # None = full head_dim (all other families).
    rope_dims: int | None = None
    # rope convention. mlx's traditional=False is the NeoX / split-half rope (Llama / Qwen3 /
    # gemma-4 — the family default here); traditional=True is the interleaved GPT-J rope.
    # A drafter config that explicitly declares ``rope_is_neox_style: false`` (LiquidAI's LFM2
    # DSpark heads) wants the interleaved rope. Honored only when the field is PRESENT
    # (deepcopy-noise doctrine: absent -> the family default neox), so every existing head is
    # untouched.
    rope_traditional: bool = False
    # YaRN scaling for the drafter's own rope (SpecForge heads, e.g. RadixArk's Qwen3.8-27B
    # head: factor 32 over an 8192-token original window). None = unscaled rope (every other
    # family). Keys follow initialize_rope's yarn schema: factor / beta_fast / beta_slow /
    # original_max_position_embeddings; the attention factor (0.1·ln(factor)+1, applied by
    # mlx-vlm's YarnRoPE as an input mscale on q and k) matches transformers' default exactly.
    rope_yarn: dict | None = None

    # qwen3_5 gated attention: q_proj emits [q ‖ gate] per head (2× out-features),
    # attention output is multiplied by sigmoid(gate) before o_proj.
    gated_q_proj: bool = False

    # qwen3_5 stores every RMSNorm weight as an additive offset from one (Gemma-style
    # (1+w)·x̂ — the reference vLLM patches call this offset_rms_norm). load_drafter adds
    # 1.0 to all RMSNorm weights at load so plain nn.RMSNorm modules compute the right
    # thing. Applying them un-offset multiplies the context fusion by ~0 and silently
    # collapses acceptance to ~1.25 (measured; d0 15% → 90% with the offset).
    offset_rms_norm: bool = False

    # DFlash-lineage drafters (NVIDIA Nemotron-3.5-Lightning DSpark head): the block attends
    # CAUSALLY (dflash_query_causal), over a SLIDING WINDOW of the context, with a per-head
    # learned attention-SINK bias fed to SDPA. All three are absent on the DeepSpec/community
    # heads (bidirectional full-attention block), so they default off and change nothing there.
    causal_block: bool = False
    sliding_window: int | None = None
    attention_sink: bool = False
    # has_lm_head=false drafters ship only embed_tokens and reuse the TARGET's lm_head (like
    # DFlash's bind); generate binds it once at run time. True = the drafter has its own head.
    has_own_lm_head: bool = True
    # has_own_embed=false drafters ship no embed_tokens either and reuse the TARGET's input
    # embedding for the draft block (a DFlash-warm-started head such as Muse-Glimmer reuses
    # BOTH embed and lm_head; Nemotron reuses only the head). load_drafter sets this from the
    # checkpoint (the weight is simply absent), and generate binds Target.draft_embed.
    has_own_embed: bool = True

    # dspark specifics
    block_size: int = 7
    mask_token_id: int = 4
    # Which block slot holds the first *prediction*. DeepSpec heads use anchor-as-pos0: slot 0
    # both embeds the known token and predicts the first draft token. The DFlash-derived
    # speculators heads reserve slot 0 as a pure anchor and predict from slot 1 on — the same
    # `logits_start=1` convention dflash_model.py documents, which DSparkSpeculatorConfig
    # inherits by subclassing DFlashSpeculatorConfig. Getting this wrong costs acceptance with
    # no error: slot 0 re-predicts the token we already have, so draft[i+1] lands on target[i]
    # (measured on makora-ai/gemma4-26b-a4b-dspark: accept 1.50 -> 3.10 at cap 4).
    logits_start: int = 0
    target_layer_ids: list[int] = field(default_factory=lambda: [5, 17, 29, 41, 46])
    num_target_layers: int = 48

    # markov + confidence
    markov_rank: int = 256
    markov_head_type: str = "vanilla"
    enable_confidence_head: bool = True
    confidence_head_with_markov: bool = True

    # GIDD log-SNR conditioning (PrismML Bonsai drafters; absent from DeepSpec's).
    # At inference the per-position pattern is fixed — anchor (block pos 0) at
    # max_log_snr, every masked position at min_log_snr — so the resulting additive
    # embedding is a constant per block position (see model.LogSnrEmbed).
    log_snr_conditioning: bool = False
    min_log_snr: float = -9.0
    max_log_snr: float = 9.0

    # logits
    final_logit_softcapping: float | None = 30.0
    pad_token_id: int = 0

    # ---- family-derived knobs (set in from_json) ----
    mlp_activation: str = "gelu_tanh"   # "gelu_tanh" | "silu"
    norm_style: str = "gemma"           # "gemma" (sandwich+scalar) | "qwen" (llama 2-norm)
    use_v_norm: bool = True             # gemma: RMSNormNoScale v_norm; qwen: none
    attention_scaling: float | None = None  # None -> 1/sqrt(attn_head_dim)

    @property
    def attn_head_dim(self) -> int:
        """Head dim used by the drafter's own attention."""
        return self.global_head_dim if self.family == "gemma4" else self.head_dim

    @property
    def out_vocab_size(self) -> int:
        """Output width of lm_head / markov_w2 — the drafter's *draft* vocabulary."""
        return self.draft_vocab_size or self.vocab_size

    @property
    def n_kv_heads(self) -> int:
        if self.family == "gemma4" and self.attention_k_eq_v:
            return self.num_global_key_value_heads
        return self.num_key_value_heads

    @property
    def scaling(self) -> float:
        if self.attention_scaling is not None:
            return self.attention_scaling
        return self.attn_head_dim ** -0.5 if self.family == "qwen3" else 1.0

    @property
    def rope_parameters(self) -> dict:
        if self.rope_yarn:
            return {"rope_type": "yarn", **self.rope_yarn}
        return {"rope_type": self.rope_type, "partial_rotary_factor": self.partial_rotary_factor}

    @classmethod
    def from_json(cls, path: str | Path) -> DSparkConfig:
        with open(path) as f:
            c = json.load(f)
        mt = c.get("model_type", "")

        # The vLLM-'speculators' packaging describes the same drafter with a different config
        # schema (the weight NAMES already match DeepSpec 1:1), so translate it into the
        # DeepSpec schema and fall through. This must run before family detection: a
        # speculators drafter carries its backbone model_type nested under
        # transformer_layer_config, and its top-level model_type is absent or misleading.
        if "speculators_config" in c or "speculators_model_type" in c:
            c = _translate_speculators(c, path)
            mt = c.get("model_type", "")

        # SpecForge (sgl-project/SpecForge, served by SGLang) packaging — the fourth one, e.g.
        # RadixArk/Qwen3.8-27B-DSpark. Its DSparkDraftModel subclasses SpecForge's
        # DFlashDraftModel, whose config is a plain transformers Qwen3Config plus a nested
        # `dflash_config` dict tagged `projector_type: "dspark"` — the discriminator (absent
        # from every other packaging; the fork-labelled heads carry a dflash_config too, but
        # never that tag). Three things follow from the reference code these repos ship:
        # (1) target_layer_ids / mask_token_id live only inside dflash_config — hoist them;
        # (2) the DSpark flavor samples the ANCHOR slot (anchor-as-pos0, the DeepSpec
        #     convention its markov/confidence code is adapted from): a block_size-7 head
        #     proposes 7 tokens, and the card's own "verify width 8 including the bonus" is
        #     that count. Do NOT trust the shipped dflash.py spec_generate for this — it is
        #     the BASE DFlash loop (reads `[:, -block_size+1:]`, logits_start=1), and running
        #     the head that way shifts every draft one slot and collapses acceptance with no
        #     error anywhere (measured on RadixArk/Qwen3.8-27B-DSpark: accept 1.35 as
        #     logits_start=1 vs 3.42 as 0, same prompt and loop — the makora bug class,
        #     settled the same way: the production loop printing drafted-vs-target rows);
        # (3) the backbone rope is transformers' Qwen3RotaryEmbedding built from the head's
        #     OWN config, so a declared rope_parameters (incl. yarn) is real, not deepcopy
        #     noise — the qwen3 branch below honors it for this packaging only (also
        #     measured the right way up: accept 3.42 yarn vs 3.33 default-rope on a short
        #     prompt, a gap that only grows with position).
        _sf = (c.get("dflash_config") or {})
        specforge = _sf.get("projector_type") == "dspark"
        if specforge:
            for k in ("target_layer_ids", "mask_token_id"):
                if k not in c and _sf.get(k) is not None:
                    c[k] = _sf[k]
            c.setdefault("logits_start", 0)

        # LiquidAI LFM2.5-DSpark packaging (the fifth). A plain qwen3-backbone DSpark head
        # (architectures ["Lfm2DSparkDraftModel"]) for an LFM2 conv+attention target. Like
        # SpecForge it nests target_layer_ids / mask_token_id / num_target_layers inside
        # ``dflash_config`` — but with NO ``projector_type`` tag and no top-level DSpark fields —
        # so hoist them for the required-field check below. The ``target_layer_ids not in c``
        # guard keeps this off the Nemotron head (architectures "Qwen3DSparkModel"), which carries
        # those ids at TOP level. The block is bidirectional full-attention (layer_types all
        # full_attention, no sliding window / dflash_query_causal) and samples the anchor slot
        # (block_size 9 -> 9 predictions; the card's "ceiling is 10" = 9 drafts + the target's
        # bonus token, i.e. logits_start 0 — confirmed by the drafter's own accept length).
        arch = " ".join(c.get("architectures") or [])
        if (not specforge and "DSpark" in arch and "target_layer_ids" not in c
                and _sf.get("target_layer_ids") is not None):
            for k in ("target_layer_ids", "mask_token_id", "num_target_layers"):
                if k not in c and _sf.get(k) is not None:
                    c[k] = _sf[k]
            c.setdefault("logits_start", 0)
        if "block_size" not in c and any(k.startswith("dspark_") for k in c):
            raise ValueError(
                f"{path}: this looks like a full target model with an embedded DSpark drafter "
                f"(dspark_* fields in the target config, e.g. DeepSeek-V4-*-DSpark), not a "
                f"standalone drafter checkpoint. mlx-dspark loads standalone DeepSpec drafters "
                f"(e.g. deepseek-ai/dspark_*_block7)."
            )

        if "qwen3" in mt:
            family = "qwen3"
        elif "gemma4" in mt:
            family = "gemma4"
        else:
            raise ValueError(
                f"{path}: unsupported drafter family (model_type={mt!r}). Supported drafter "
                f"backbones: qwen3, gemma4 (gemma4_text). Drafter-free speculation works with "
                f"any target via --mode lookup / --mode auto; for a new drafter family, open an "
                f"issue: https://github.com/ARahim3/mlx-dspark/issues"
            )

        required = ("hidden_size", "vocab_size", "num_hidden_layers", "intermediate_size",
                    "num_attention_heads", "block_size", "mask_token_id", "target_layer_ids")
        missing = [k for k in required if k not in c]
        if missing:
            raise ValueError(
                f"{path}: config is missing required DeepSpec drafter fields {missing} — this "
                f"does not look like a DeepSpec-format DSpark drafter checkpoint."
            )

        if family == "qwen3":
            rp = c.get("rope_parameters") or {}
            head_dim = c.get("head_dim", c["hidden_size"] // c["num_attention_heads"])
            # qwen3_5-flavored backbones (e.g. Ornith drafters) declare gated q_proj and
            # partial rotary in the same DeepSpec layout; plain qwen3 configs carry neither,
            # so both knobs default to the classic behavior. The config's mrope fields are a
            # text-only no-op (equal position ids collapse mrope to standard rope — mlx-lm's
            # own qwen3_5 text module does the same). Careful: a drafter config is a deepcopy
            # of the TARGET's, so several fields here describe the target and not the drafter
            # (see the rope note below, and `attn_output_gate`/`linear_*`, which the drafter's
            # own weight shapes contradict). Trust weight shapes and provenance, not fields.
            gated_q = bool(c.get("enable_qwen35_gated_q_proj", False))
            # qwen3_5 house style: norm weights stored offset-from-one (see field docs).
            offset_norms = mt == "qwen3_5"
            # `partial_rotary_factor` means partial rotary only on the qwen3_5-NATIVE fork
            # (architectures "Qwen35DSparkModel", e.g. Ornith: Qwen3Next-style gated
            # attention + offset norms — the same code path that ropes a head_dim slice).
            # DeepSpec's stock trainer builds the drafter config as a deepcopy of the
            # TARGET's, so every stock head for a Qwen3.5/3.6 target carries the field as
            # inherited noise while its rope is full head_dim: the reference builds rope
            # with transformers' Qwen3RotaryEmbedding (whose default init keys off head_dim
            # alone) and its apply_rotary_pos_emb multiplies q at full head_dim width — a
            # quarter-width cos would not even broadcast. Honoring the field there ropes a
            # quarter of each head and quietly costs acceptance with no error anywhere
            # (measured on satgeze/Qwen3.5-0.8B-DSpark: accept 1.29 -> 1.59 code,
            # 1.36 -> 1.78 chat, once the rope went back to full width).
            qwen35_native = gated_q or offset_norms
            rope_dims = (int(head_dim * float(rp.get("partial_rotary_factor", 1.0)))
                         if qwen35_native else head_dim)
            # DFlash-lineage markers (Nemotron-3.5-Lightning DSpark). Gate every one of these
            # behind an explicit signal so plain qwen3 heads are untouched: causal block,
            # sliding-window context, per-head sink, and lm_head reuse. sample_from_anchor is
            # the DFlash spelling of logits_start (false -> slot 0 is a pure anchor, predict
            # from slot 1), the same convention the speculators path derives.
            dflc = c.get("dflash_config") or {}
            causal_block = bool(c.get("dflash_query_causal", False) or dflc.get("causal", False))
            attn_sink = bool(c.get("attention_sink_bias", False))
            swa = None
            if causal_block or attn_sink or dflc:
                w = c.get("sliding_window") or (
                    dflc.get("swa_window_size") if dflc.get("use_swa", False) else None)
                swa = int(w) if w else None
            has_own_lm = bool(c.get("has_lm_head", True))
            # YaRN rope — honored ONLY for the SpecForge packaging, where the reference builds
            # transformers' Qwen3RotaryEmbedding from the head's own config and that class
            # applies the declared scaling (RadixArk/Qwen3.8-27B-DSpark: factor 32 over an
            # 8192 original window, and yarn's attention factor 0.1·ln(32)+1 ≈ 1.347 scales
            # cos/sin at EVERY position — so ignoring it would mis-rope even short prompts
            # and silently cost acceptance). Everywhere else a scaling field is deepcopy
            # noise until a reference proves otherwise (see the partial-rotary note above).
            yarn = None
            if specforge and rp.get("rope_type") == "yarn" and rp.get("factor"):
                yarn = {"factor": float(rp["factor"])}
                for k in ("beta_fast", "beta_slow"):
                    if k in rp:
                        yarn[k] = float(rp[k])
                if "original_max_position_embeddings" in rp:
                    yarn["original_max_position_embeddings"] = int(
                        rp["original_max_position_embeddings"])
            # rope convention: honor an explicit rope_is_neox_style (LiquidAI LFM2 DSpark head
            # declares it false = interleaved GPT-J rope = mlx traditional=True). Absent -> the
            # qwen3 family default (neox, traditional=False). See DSparkConfig.rope_traditional.
            rope_trad = (not bool(c["rope_is_neox_style"])) if "rope_is_neox_style" in c else False
            if "logits_start" in c:
                logits_start = int(c["logits_start"])
            elif "sample_from_anchor" in c:
                logits_start = 0 if c["sample_from_anchor"] else 1
            else:
                logits_start = 0
            if c.get("log_snr_conditioning"):
                lo, hi = c.get("min_log_snr"), c.get("max_log_snr")
                if lo is None or hi is None or not (float(hi) > float(lo)):
                    raise ValueError(
                        f"{path}: log_snr_conditioning is enabled but min/max_log_snr are "
                        f"missing or not ordered (min={lo!r}, max={hi!r}) — the featurization "
                        f"divides by (max - min), so a drafter converted without them would "
                        f"draft from silently-wrong embeddings."
                    )
            return cls(
                family="qwen3",
                hidden_size=c["hidden_size"], vocab_size=c["vocab_size"],
                draft_vocab_size=c.get("draft_vocab_size"),
                num_hidden_layers=c["num_hidden_layers"],
                intermediate_size=c["intermediate_size"],
                rms_norm_eps=c.get("rms_norm_eps", 1e-6),
                num_attention_heads=c["num_attention_heads"],
                num_key_value_heads=c.get("num_key_value_heads", 8),
                head_dim=head_dim,
                attention_k_eq_v=False, attention_bias=c.get("attention_bias", False),
                rope_theta=rp.get("rope_theta", c.get("rope_theta", 1_000_000.0)),
                rope_type="default",
                rope_yarn=yarn,
                rope_dims=(rope_dims if rope_dims != head_dim else None),
                rope_traditional=rope_trad,
                gated_q_proj=gated_q,
                offset_rms_norm=offset_norms,
                causal_block=causal_block, sliding_window=swa,
                attention_sink=attn_sink, has_own_lm_head=has_own_lm,
                block_size=c["block_size"], mask_token_id=c["mask_token_id"],
                logits_start=logits_start,
                target_layer_ids=list(c["target_layer_ids"]),
                num_target_layers=c.get("num_target_layers", 36),
                markov_rank=c.get("markov_rank", 256),
                markov_head_type=c.get("markov_head_type", "vanilla"),
                enable_confidence_head=c.get("enable_confidence_head", True),
                confidence_head_with_markov=c.get("confidence_head_with_markov", True),
                final_logit_softcapping=c.get("final_logit_softcapping", None),
                pad_token_id=c.get("pad_token_id") or 0,
                mlp_activation="silu", norm_style="qwen", use_v_norm=False,
                log_snr_conditioning=bool(c.get("log_snr_conditioning", False)),
                min_log_snr=float(c.get("min_log_snr", -9.0)),
                max_log_snr=float(c.get("max_log_snr", 9.0)),
            )

        rope = (c.get("rope_parameters") or {}).get("full_attention", {}) or {}
        return cls(
            family="gemma4",
            hidden_size=c["hidden_size"], vocab_size=c["vocab_size"],
            draft_vocab_size=c.get("draft_vocab_size"),
            num_hidden_layers=c["num_hidden_layers"],
            intermediate_size=c["intermediate_size"],
            rms_norm_eps=c.get("rms_norm_eps", 1e-6),
            num_attention_heads=c["num_attention_heads"],
            num_key_value_heads=c.get("num_key_value_heads", 8),
            num_global_key_value_heads=c.get("num_global_key_value_heads", 1),
            head_dim=c.get("head_dim", 256), global_head_dim=c.get("global_head_dim", 512),
            attention_k_eq_v=c.get("attention_k_eq_v", True),
            attention_bias=c.get("attention_bias", False),
            rope_theta=rope.get("rope_theta", 1_000_000.0),
            partial_rotary_factor=rope.get("partial_rotary_factor", 0.25),
            rope_type=rope.get("rope_type", "proportional"),
            block_size=c["block_size"], mask_token_id=c["mask_token_id"],
            logits_start=int(c.get("logits_start", 0)),
            target_layer_ids=list(c["target_layer_ids"]),
            num_target_layers=c.get("num_target_layers", 48),
            markov_rank=c.get("markov_rank", 256),
            markov_head_type=c.get("markov_head_type", "vanilla"),
            enable_confidence_head=c.get("enable_confidence_head", True),
            confidence_head_with_markov=c.get("confidence_head_with_markov", True),
            final_logit_softcapping=c.get("final_logit_softcapping", 30.0),
            pad_token_id=c.get("pad_token_id", 0),
            mlp_activation="gelu_tanh", norm_style="gemma", use_v_norm=True,
        )
