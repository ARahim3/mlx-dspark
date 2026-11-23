"""Thinking Machines Inkling 2 (``inkling2`` / ``inkling2_loop``): the DSpark head it ships.

Inkling-2-Mini bundles its speculative drafter inside the TARGET repo (``mtp.safetensors`` +
``config.json: mtp_config``), so the "drafter repo" of this pair is the target repo itself. The
head is DSpark-lineage (bidirectional block, Markov head) but differs from every DeepSpec-style
head in three ways, which is why it gets its own module instead of ``DSparkDrafter`` flags:

* **Per-layer context fusion ("DFlare").** Each of the 3 draft layers — and the Markov head —
  mixes the four target taps with its own softmax weights (``fuse_weight[row]``) and its own
  RMSNorm, and projects the context with K/V weights separate from the block's
  (``k_proj_ctx`` / ``v_proj_ctx`` + ``k_norm_ctx``). Context K/V is all that persists across
  rounds; local layers (0 and 2) attend only the last ``sliding_window`` context rows.
* **An 8-row block with a pure anchor slot.** Row 0 is the anchor's TARGET input embedding
  (``embed_norm ∘ embed_tokens``), rows 1..7 a learned ``mask_embed``; drafts are read from rows
  1..7 (``logits_start = 1``). The block is bidirectional, so it always runs full width.
* **The ``rnn_hidden`` Markov head.** A GRU-like recurrence over the previously drafted token:
  ``[state ‖ w1[prev]]`` and the row's ``[x ‖ f_rnn]`` context give gate / candidate / output;
  the output's tanh is folded through the target's lm_head (``fold = lm_head · hidden_out``,
  precomputed once per bound target) and added to the base logits (the target's raw bf16
  lm_head over the draft row), then softcapped. ``f_rnn`` is the Markov-mixed features of the
  NEWEST context row — kept as a fourth ctx-cache entry so prefix-cache snapshots, restores and
  ``trim_to`` carry it like the K/V (a private attribute would draft from stale features after
  a prefix hit).

The math follows vLLM's Inkling 2 DSpark (WoosukKwon/vllm@d664cd1, Apache-2.0) as ported to MLX by
mlx-vlm (``speculative/drafters/inkling2_dspark``, MIT), including its dtype choices (fp32
residual sums, bf16 storage, fp32 recurrence products). One deliberate deviation: the fold is
computed with the bound bf16 head on the GPU (fp32 accumulation) instead of in fp64 on the host —
a drafting-only quantity, so it can change acceptance by rounding noise at most, never output.

The target side (tap, transactional verify/rollback over mlx-vlm's caches) lives in target.py.
"""

from __future__ import annotations

import json
import os

import mlx.core as mx
import mlx.nn as nn

from .config import DSparkConfig
from .model import CtxCache

_FOLD_RANK = 256            # inkling_kernels.FOLD_RANK: the fold kernel's lane layout

# config.json model_type of the targets whose repo carries this head (Inkling-2-Mini is the
# looped variant; Inkling-2-Small is the plain one — same head contract).
TARGET_MODEL_TYPES = ("inkling2", "inkling2_loop")
HEAD_FILE = "mtp.safetensors"
# Bits of the drafter-only copy of the target's head that the base logits use (0 = the target's
# own bf16 head). See Inkling2DSparkDrafter.bind_target.
DRAFT_HEAD_BITS = int(os.environ.get("MLX_DSPARK_INKLING_DRAFT_HEAD_BITS", "6") or 0)
# Bits of the Markov fold (lm_head . hidden_out), stored per vocab row and scanned by one kernel
# per draft step (0 = the reference's bf16 [rank, V] matrix and full-vocab bias). See
# _RnnMarkovHead.fold.
FOLD_BITS = int(os.environ.get("MLX_DSPARK_INKLING_FOLD_BITS", "4") or 0)
# Until an mlx-vlm RELEASE has the looped model, Inkling-2-Mini runs on a git install — of this
# commit, the one the pair was measured on. mlx-vlm's main keeps changing the model's internals
# (its caches and kernels, several times a day in the week of 2026-11-22); the fused pieces are
# self-tested against whatever is installed and fall back to its stock forward, but only this
# commit is the one the published numbers stand on.
MLX_VLM_COMMIT = "d998a6207b7dc613b52618646dbea13a4799f7ca"
MLX_VLM_INSTALL = ("uv pip install 'mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm@"
                   f"{MLX_VLM_COMMIT}'")
MLX_VLM_HINT = ("this Inkling 2 model needs a newer mlx-vlm: 0.8.0 ships the plain `inkling2` model "
                "only, the looped one (Inkling-2-Mini, `inkling2_loop`) merged 2026-11-20, after it. "
                f"Upgrade mlx-vlm, or until a release ships it: {MLX_VLM_INSTALL}")


def is_inkling_config(cfg: dict | None) -> bool:
    return bool(cfg) and str(cfg.get("model_type", "")) in TARGET_MODEL_TYPES


def _head_config(cfg: dict) -> dict | None:
    return cfg.get("mtp_config") or (cfg.get("text_config") or {}).get("mtp_config")


def has_bundled_head(path: str) -> bool:
    """True when ``path`` is an Inkling 2 checkpoint dir that carries its DSpark head."""
    try:
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return False
    return (is_inkling_config(cfg) and bool(_head_config(cfg))
            and os.path.isfile(os.path.join(path, HEAD_FILE)))


def mlx_vlm_supports_inkling2(model_type: str = "inkling2_loop") -> bool:
    """Whether the installed mlx-vlm can build this Inkling 2 ``model_type``.

    The module's presence does not answer for the looped model: mlx-vlm 0.8.0 ships
    ``models/inkling2`` for the plain one only, and ``inkling2_loop`` (Inkling-2-Mini) arrived
    with ``LoopConfig`` after that release."""
    from importlib import import_module

    try:
        config = import_module("mlx_vlm.models.inkling2.config")
    except ImportError:
        return False
    return "loop" not in str(model_type) or hasattr(config, "LoopConfig")


def draft_config(cfg: dict) -> DSparkConfig:
    """The head's :class:`DSparkConfig` from the TARGET's config.json (``mtp_config`` + the text
    config's vocab / softcap). Validates the contract this module implements."""
    text = cfg.get("text_config") or {}
    mtp = _head_config(cfg)
    if not mtp:
        raise ValueError("Inkling 2 checkpoint has no mtp_config (no bundled DSpark head)")
    if mtp.get("markov_head_type") != "rnn_hidden":
        raise ValueError(f"Inkling 2 head: markov_head_type={mtp.get('markov_head_type')!r} "
                         "(only rnn_hidden is implemented)")
    if mtp.get("q_bias") or mtp.get("o_bias"):
        raise ValueError("Inkling 2 head: attention biases are not part of this contract")
    ids = [int(i) for i in mtp["target_layer_ids"]]
    if len(ids) < 2 or ids != sorted(set(ids)) or ids[0] < 0:
        raise ValueError(f"Inkling 2 head: invalid target_layer_ids {ids}")
    n_layers = int(mtp["num_hidden_layers"])
    local = [int(i) for i in mtp.get("local_layer_ids", [])]
    if any(not 0 <= i < n_layers for i in local):
        raise ValueError(f"Inkling 2 head: local_layer_ids {local} out of range")
    c = DSparkConfig(
        family="inkling2",
        hidden_size=int(mtp["hidden_size"]),
        vocab_size=int(text.get("vocab_size", 0)),
        num_hidden_layers=n_layers,
        intermediate_size=int(mtp["intermediate_size"]),
        rms_norm_eps=float(mtp.get("rms_norm_eps", 1e-6)),
        num_attention_heads=int(mtp["num_attention_heads"]),
        num_key_value_heads=int(mtp["num_key_value_heads"]),
        head_dim=int(mtp["head_dim"]),
        attention_k_eq_v=False,
        rope_theta=float(mtp.get("rope_theta", 1_000_000.0)),
        sliding_window=int(mtp.get("sliding_window_size", 0)) or None,
        has_own_lm_head=False,           # bound from the target (Target.lm_head_proj)
        has_own_embed=False,             # bound from the target (Target.draft_embed)
        block_size=int(mtp["block_size"]),
        mask_token_id=0,                 # unused: the mask rows are the learned mask_embed
        logits_start=1,                  # row 0 is a pure anchor
        target_layer_ids=ids,
        num_target_layers=ids[-1],
        markov_rank=int(mtp["markov_rank"]),
        markov_head_type="rnn_hidden",
        enable_confidence_head=False,
        confidence_head_with_markov=False,
        final_logit_softcapping=float(text.get("final_logit_softcapping", 30.0)) or None,
        mlp_activation="silu",
        norm_style="qwen",
        use_v_norm=False,
    )
    if not c.vocab_size or c.hidden_size != int(text.get("hidden_size", c.hidden_size)):
        raise ValueError("Inkling 2 head: mtp_config does not match the text config")
    c.local_layer_ids = local
    c.unpadded_vocab_size = int(text.get("unpadded_vocab_size", c.vocab_size))
    return c


# ---------------------------------------------------------------------------- the head


class _DraftAttention(nn.Module):
    """Block queries over ``[context K/V (own projections), block K/V]``, bidirectional."""

    def __init__(self, config: DSparkConfig, local: bool):
        super().__init__()
        h, hd = config.hidden_size, config.head_dim
        self.n_heads = config.num_attention_heads
        self.n_kv = config.num_key_value_heads
        self.head_dim = hd
        self.window = config.sliding_window if local else None   # trained window (local layers)
        self.ctx_window: int | None = None                       # engine knob (global layer)
        for name, heads in (("q_proj", self.n_heads), ("k_proj", self.n_kv), ("v_proj", self.n_kv),
                            ("k_proj_ctx", self.n_kv), ("v_proj_ctx", self.n_kv)):
            setattr(self, name, nn.Linear(h, heads * hd, bias=False))
        self.o_proj = nn.Linear(self.n_heads * hd, h, bias=False)
        self.q_norm = nn.RMSNorm(hd, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(hd, eps=config.rms_norm_eps)
        self.k_norm_ctx = nn.RMSNorm(hd, eps=config.rms_norm_eps)
        self.rope = nn.RoPE(hd, traditional=False, base=config.rope_theta)

    def _heads(self, x: mx.array, n: int) -> mx.array:
        return x.reshape(*x.shape[:2], n, self.head_dim).transpose(0, 2, 1, 3)

    def update_ctx(self, ctx_in: mx.array, ctx_offset: int, cache: CtxCache) -> None:
        k = self.k_norm_ctx(self._heads(self.k_proj_ctx(ctx_in), self.n_kv))
        v = self._heads(self.v_proj_ctx(ctx_in), self.n_kv)
        cache.append(self.rope(k, offset=ctx_offset), v)        # V is not roped

    def __call__(self, x: mx.array, block_offset: int, cache: CtxCache) -> mx.array:
        q = self.rope(self.q_norm(self._heads(self.q_proj(x), self.n_heads)), offset=block_offset)
        k_blk = self.rope(self.k_norm(self._heads(self.k_proj(x), self.n_kv)), offset=block_offset)
        v_blk = self._heads(self.v_proj(x), self.n_kv)
        k, v = cache.with_block(k_blk, v_blk, window=self.window or self.ctx_window)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim ** -0.5,
                                                   mask=None)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(*x.shape[:2], -1))


class _DraftMLP(nn.Module):
    def __init__(self, config: DSparkConfig):
        super().__init__()
        h, i = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.gate_proj(x).astype(mx.float32)
        up = self.up_proj(x).astype(mx.float32)
        return self.down_proj((nn.silu(gate) * up).astype(x.dtype))


class _DraftLayer(nn.Module):
    """Pre-norm layer with the reference's fused add+norm: sums in fp32, stores bf16."""

    def __init__(self, config: DSparkConfig, index: int):
        super().__init__()
        eps = config.rms_norm_eps
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.dflare_norm = nn.RMSNorm(config.hidden_size, eps=eps)   # this layer's ctx fusion
        self.self_attn = _DraftAttention(config, index in config.local_layer_ids)
        self.mlp = _DraftMLP(config)

    def __call__(self, x, residual, block_offset: int, cache: CtxCache):
        f32 = mx.float32
        summed = x if residual is None else x.astype(f32) + residual.astype(f32)
        residual = summed.astype(x.dtype)
        h = self.self_attn(self.input_layernorm(summed).astype(x.dtype), block_offset, cache)
        summed = h.astype(f32) + residual.astype(f32)
        return (self.mlp(self.post_attention_layernorm(summed).astype(x.dtype)),
                summed.astype(x.dtype))


class _RnnMarkovHead(nn.Module):
    """``rnn_hidden`` Markov head. ``rnn_proj`` maps ``[state ‖ w1[prev] ‖ x ‖ f_rnn]`` to
    gate / candidate / output (3·rank); the ``x ‖ f_rnn`` part is per-row and computed once per
    block (:meth:`context`), the recurrent part per step (:meth:`step`)."""

    def __init__(self, config: DSparkConfig):
        super().__init__()
        r, h = config.markov_rank, config.hidden_size
        self.markov_w1 = nn.Embedding(config.vocab_size, r)
        self.rnn_proj = nn.Linear(2 * r + 2 * h, 3 * r)
        self.hidden_out = nn.Linear(r, h)
        self.dflare_norm = nn.RMSNorm(h, eps=config.rms_norm_eps)
        # derived, never parameters (leading underscore keeps them out of mlx's filter):
        self._w_rec = None      # rnn_proj.weight[:, :2r] in fp32
        self._w_ctx = None      # rnn_proj.weight[:, 2r:] in fp32
        self._fold_t = None     # (lm_head · hidden_out)ᵀ  [rank, V]   (FOLD_BITS == 0)
        self._fold_q = None     # the same per vocab row [V, rank], quantized (w, scales, biases)
        self._fold_bits = 0
        self._fold_b = None     # lm_head · hidden_out.bias  [V]

    def prepare(self) -> None:
        r = self.markov_w1.weight.shape[1]
        w = self.rnn_proj.weight.astype(mx.float32)
        self._w_rec, self._w_ctx = mx.contiguous(w[:, : 2 * r]), mx.contiguous(w[:, 2 * r:])
        mx.eval(self._w_rec, self._w_ctx)

    def fold(self, project) -> None:
        """Fold ``hidden_out`` through the target's raw head: ``project`` maps hidden
        ``[..., H]`` -> raw logits ``[..., V]``; applying it to ``hidden_out``'s columns gives
        ``(lm_head · hidden_out)ᵀ`` directly."""
        w = self.hidden_out.weight                  # [H, r]
        fold_t = project(mx.contiguous(w.T)[None])[0]                       # [r, V]
        self._fold_b = project(self.hidden_out.bias[None, None])[0, 0]      # [V]
        mx.eval(fold_t, self._fold_b)
        self._fold_t = self._fold_q = None
        self._fold_bits = 0
        if (FOLD_BITS in (4, 8) and fold_t.shape[0] == _FOLD_RANK and mx.metal.is_available()
                and mx.default_device() == mx.gpu):
            # Stored per vocab row and quantized: a draft step then reads 29 MB (4-bit; 55 at
            # 8) instead of streaming the 103 MB bf16 matrix, in one launch that keeps only
            # the argmax (inkling_kernels.fold_argmax). A rank-256 correction to drafting
            # logits — the target verifies every token. Teacher-forced on ~1,000 paired
            # anchors: acceptance 2.272 at 8 bits, 2.274 at 4 (cap 3); 2.530 / 2.543 (cap 7).
            from .inkling_kernels import quantize_fold

            self._fold_q = quantize_fold(fold_t, FOLD_BITS)
            self._fold_bits = FOLD_BITS
        else:
            self._fold_t = fold_t

    def context(self, hidden: mx.array, f_rnn: mx.array) -> mx.array:
        inputs = mx.concatenate([hidden, mx.broadcast_to(f_rnn, hidden.shape)], axis=-1)
        return inputs.astype(mx.float32) @ self._w_ctx.T + self.rnn_proj.bias.astype(mx.float32)

    def advance(self, state: mx.array, prev: mx.array, ctx_row: mx.array):
        """One recurrence step: ``(next state, latent)`` — the latent is what the fold turns
        into the vocabulary bias."""
        inputs = mx.concatenate([state, self.markov_w1(prev)], axis=-1)
        raw = (inputs.astype(mx.float32) @ self._w_rec.T + ctx_row).astype(state.dtype)
        gate, cand, out = mx.split(raw, 3, axis=-1)
        gate = mx.sigmoid(gate.astype(mx.float32)).astype(state.dtype)
        cand = mx.tanh(cand.astype(mx.float32)).astype(state.dtype)
        state = gate * state + (1 - gate) * cand
        return state, mx.tanh(out.astype(mx.float32)).astype(state.dtype)

    def bias(self, latent: mx.array) -> mx.array:
        """The full-vocabulary Markov bias ``[B, V]`` of a latent (the sampling path)."""
        if self._fold_q is not None:
            w, scales, biases = self._fold_q
            return self._fold_b + mx.quantized_matmul(latent, w, scales, biases, transpose=True,
                                                      group_size=64, bits=self._fold_bits)
        return mx.addmm(self._fold_b, latent, self._fold_t)

    def step(self, state: mx.array, prev: mx.array, ctx_row: mx.array):
        state, latent = self.advance(state, prev, ctx_row)
        return state, self.bias(latent)

    def greedy(self, latent: mx.array, base_row: mx.array, vocab_out: int):
        """``argmax(base_row + bias(latent))`` over the first ``vocab_out`` ids as one launch,
        or None when the fold is not in the kernel's form."""
        if self._fold_q is None or latent.shape[0] != 1:
            return None
        from .inkling_kernels import fold_argmax

        return fold_argmax(latent.reshape(-1), self._fold_q, self._fold_b, base_row,
                           self._fold_bits, vocab_out)


class Inkling2DSparkDrafter(nn.Module):
    """The Inkling 2 DSpark head behind the ``DSparkDrafter`` interface the spec loop,
    calibration and the server drive (embed / update_context / backbone / head_slice /
    compute_logits / sample_block / sample_block_probs / draft_width / max_draft / ctx caches)."""

    def __init__(self, config: DSparkConfig):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.mask_token_id = config.mask_token_id
        self.logits_start = config.logits_start
        self.causal_block = False
        self.softcap = config.final_logit_softcapping
        self.unpadded = int(getattr(config, "unpadded_vocab_size", config.vocab_size))
        n_taps = len(config.target_layer_ids)
        self.layers = [_DraftLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.markov_head = _RnnMarkovHead(config)
        self.fuse_weight = mx.zeros((config.num_hidden_layers + 1, n_taps))
        self.mask_embed = mx.zeros((config.hidden_size,))
        self.confidence_head = None
        self._ext_embed = None       # Target.draft_embed (embed_norm ∘ embed_tokens)
        self._ext_lm_head = None     # Target.lm_head_proj (raw bf16 head): the Markov fold
        self._base_head = None       # drafter-only quantized copy of that head: the base logits
        self._base_src = None        # the head weight the copy was built from
        self._folded_for = None      # the projection the Markov fold was built from
        self._head_hidden = None     # this round's head rows (compute_logits -> sample_block)
        self._f_rnn = None           # this round's newest-context Markov features (backbone)

    # -- binding (generate / calibrate bind these for has_own_*=False heads) --------------
    def bind_lm_head(self, project) -> None:
        self._ext_lm_head = project

    def bind_embed(self, embed_fn) -> None:
        self._ext_embed = embed_fn

    def bind_target(self, target) -> None:
        """Build the drafter-only quantized copy of the target's output head that the BASE logits
        go through (``DRAFT_HEAD_BITS``; the Markov fold is built from the exact bf16 head).
        Drafting-only: the target verifies every token. Measured on Inkling-2-Mini,
        teacher-forced: 8-bit leaves acceptance unchanged against the bf16 head and cuts the
        3-row base-logit matmul 3.80 -> 2.11 ms; 6-bit (the default; 1.7 ms, 116 MB less) leaves
        it unchanged too — 2.280 vs 2.272 over 1,046 paired anchors at cap 3, 2.521 vs 2.530
        over 929 at cap 7, and no slower at one or two rows; 5-bit is no faster; 4-bit (1.48
        ms) starts to cost acceptance (cap 7: 4.47 -> 4.35 on math). Rebuilt only when the head
        changes."""
        lm = getattr(getattr(target, "model", None), "language_model", None)
        head = getattr(lm, "lm_head", None)
        if not DRAFT_HEAD_BITS or type(head) is not nn.Linear or "bias" in head:
            self._base_head = self._base_src = None
            return
        # A target loaded with --target-bits 8 already holds an 8-bit copy of this head for its
        # own logits: draft from that one (fp32 logits through the same kernel, flat to 4 rows)
        # instead of keeping a second copy (0.35 GB at 6 bits; ~0.4 ms a round dearer).
        from .inkling_kernels import HEAD_Q8_MAX_ROWS, dense_head, head_q8_f32, head_q8_ok

        shared = dense_head(lm) if lm is not None else None
        if (shared is not None and head_q8_ok(shared)
                and self.block_size <= HEAD_Q8_MAX_ROWS and mx.default_device() == mx.gpu):
            if self._base_src is not shared:
                def base(hidden, qh=shared):
                    b, rows, width = hidden.shape
                    return head_q8_f32(hidden.reshape(b * rows, width), qh).reshape(b, rows, -1)

                self._base_head, self._base_src = base, shared
            return
        if self._base_src is head.weight:
            return
        q = nn.QuantizedLinear.from_linear(head, group_size=64, bits=DRAFT_HEAD_BITS)
        mx.eval(q.parameters())
        self._base_head, self._base_src = q, head.weight

    def _ready(self) -> None:
        if self._ext_lm_head is None or self._ext_embed is None:
            raise RuntimeError("Inkling 2 head: bind the target first — drafter.bind_lm_head("
                               "target.lm_head_proj) and drafter.bind_embed(target.draft_embed)")
        if self._folded_for is not self._ext_lm_head:
            if self.markov_head._w_rec is None:
                self.markov_head.prepare()
            self.markov_head.fold(self._ext_lm_head)
            self._folded_for = self._ext_lm_head

    # -- shape of the head ------------------------------------------------------------
    @property
    def max_draft(self) -> int:
        return self.block_size - self.logits_start

    def draft_width(self, cap: int) -> int:
        """Always the full trained block: it is bidirectional (every row sees every row)."""
        return self.block_size

    def head_slice(self, block_hidden: mx.array, cap: int) -> mx.array:
        s = self.logits_start
        return block_hidden[:, s: s + cap, :]

    @property
    def ctx_window(self) -> int | None:
        glob_layers = [ly.self_attn for ly in self.layers if ly.self_attn.window is None]
        return glob_layers[0].ctx_window if glob_layers else None

    @ctx_window.setter
    def ctx_window(self, window: int | None) -> None:
        """The engine's drafter context window applies to the full-context (global) draft
        layer only; the local layers keep their trained window."""
        w = int(window) if window else None
        for ly in self.layers:
            if ly.self_attn.window is None:
                ly.self_attn.ctx_window = w

    # -- context ------------------------------------------------------------------------
    def make_ctx_cache(self) -> list[CtxCache]:
        """One K/V cache per draft layer + one row cache of the Markov-mixed features."""
        return [CtxCache() for _ in self.layers] + [CtxCache()]

    def _mix(self, fused: mx.array, row: int, norm: nn.RMSNorm) -> mx.array:
        b, t, _ = fused.shape
        feats = fused.reshape(b, t, -1, self.config.hidden_size).astype(mx.float32)
        alpha = mx.softmax(self.fuse_weight.astype(mx.float32), axis=-1).astype(fused.dtype)[row]
        mixed = feats[:, :, 0] * alpha[0].astype(mx.float32)
        for i in range(1, feats.shape[2]):
            mixed = mixed + feats[:, :, i] * alpha[i].astype(mx.float32)
        return norm(mixed).astype(fused.dtype)

    def update_context(self, target_hidden_cat: mx.array, ctx_offset: int, ctx_caches) -> None:
        """Append committed context rows (the target's 4 taps, concatenated) at absolute
        position ``ctx_offset``: each layer's fused K/V and the Markov head's mixed features."""
        for i, (layer, cache) in enumerate(zip(self.layers, ctx_caches)):
            layer.self_attn.update_ctx(self._mix(target_hidden_cat, i, layer.dflare_norm),
                                       ctx_offset, cache)
        f = self._mix(target_hidden_cat, len(self.layers), self.markov_head.dflare_norm)
        rows = f[:, None]                                            # [B, 1, T, H]
        ctx_caches[len(self.layers)].append(rows, mx.zeros((*rows.shape[:3], 1), rows.dtype))

    # -- one draft round ------------------------------------------------------------------
    def embed(self, ids: mx.array) -> mx.array:
        """``[anchor, …]`` ids -> the block: the anchor's target input embedding, then the
        learned mask embedding (the other ids are placeholders)."""
        if self._ext_embed is None:
            self._ready()
        e0 = self._ext_embed(ids[:, :1])
        w = ids.shape[1]
        if w == 1:
            return e0
        masks = mx.broadcast_to(self.mask_embed.astype(e0.dtype), (e0.shape[0], w - 1, e0.shape[2]))
        return mx.concatenate([e0, masks], axis=1)

    def backbone(self, noise_embedding: mx.array, block_offset, ctx_caches, mask=None) -> mx.array:
        if not isinstance(block_offset, int):
            raise ValueError("Inkling 2 head: batched drafting is not supported")
        x, residual = noise_embedding, None
        for layer, cache in zip(self.layers, ctx_caches):
            x, residual = layer(x, residual, block_offset, cache)
        rows = ctx_caches[len(self.layers)]
        if rows.length == 0:
            raise RuntimeError("Inkling 2 head: drafting before any context was appended")
        b = noise_embedding.shape[0]
        self._f_rnn = rows.k[:, :, -1:, :].reshape(b, 1, -1)     # newest context row
        return self.norm(x.astype(mx.float32) + residual.astype(mx.float32)).astype(
            noise_embedding.dtype)

    def compute_logits(self, hidden: mx.array) -> mx.array:
        """Base logits: the target's RAW bf16 head over the draft rows (the Markov bias and the
        softcap are applied per step in :meth:`sample_block`)."""
        self._ready()
        self._head_hidden = hidden
        head = self._base_head if self._base_head is not None else self._ext_lm_head
        return head(hidden)

    def _markov_inputs(self, base_logits: mx.array):
        self._ready()
        x = self._head_hidden
        k = int(base_logits.shape[0])
        if x is None or x.shape[1] < k or self._f_rnn is None:
            raise RuntimeError("Inkling 2 head: call backbone() and compute_logits() first")
        ctx = self.markov_head.context(x[:, :k], self._f_rnn.astype(x.dtype))
        state = mx.zeros((x.shape[0], self.config.markov_rank), dtype=x.dtype)
        return k, ctx, state, x.dtype

    def _capped(self, logits: mx.array, dtype) -> mx.array:
        if self.softcap:
            c = self.softcap
            logits = (c * mx.tanh(logits.astype(mx.float32) / c)).astype(dtype)
        return logits[..., : self.unpadded]

    def sample_block(self, base_logits: mx.array, first_prev_token) -> mx.array:
        """Greedy draft block (``[k]`` target ids), lazily: the sequential Markov steps stay in
        the graph, so the round still reaches the device as one batch with one sync."""
        k, ctx, state, dtype = self._markov_inputs(base_logits)
        prev = (first_prev_token.reshape(-1) if isinstance(first_prev_token, mx.array)
                else mx.array([int(first_prev_token)]))
        tokens = []
        head = self.markov_head
        for i in range(k):
            state, latent = head.advance(state, prev, ctx[:, i])
            nxt = head.greedy(latent, base_logits[i], self.unpadded)
            if nxt is None:
                logits = self._capped(base_logits[i][None] + head.bias(latent), dtype)
                nxt = mx.argmax(logits, axis=-1)
            prev = nxt
            tokens.append(prev)
        return mx.concatenate(tokens)

    def sample_block_probs(self, base_logits: mx.array, first_prev_token, temperature: float,
                           top_p: float = 1.0, top_k: int = 0):
        """Temperature draft for speculative sampling: ``(tokens [k], probs [k, V'])`` over the
        same unpadded vocabulary the target's verify logits cover."""
        from .sampling import sample_probs, truncate_probs

        k, ctx, state, dtype = self._markov_inputs(base_logits)
        inv_t = 1.0 / temperature
        prev = mx.array([int(first_prev_token)])
        tokens, probs = [], []
        for i in range(k):
            state, bias = self.markov_head.step(state, prev, ctx[:, i])
            logits = self._capped(base_logits[i][None] + bias, dtype)[0]
            q = truncate_probs(mx.softmax(logits.astype(mx.float32) * inv_t, axis=-1), top_p, top_k)
            prev = sample_probs(q).reshape(1)
            probs.append(q)
            tokens.append(prev)
        return mx.concatenate(tokens), mx.stack(probs, axis=0)

    def confidence_logits(self, block_hidden, prev_token_ids):
        return None

    # identity id maps: the head drafts over the full target vocabulary
    def to_target_ids(self, draft_ids: mx.array) -> mx.array:
        return draft_ids

    def widen_to_target(self, q: mx.array) -> mx.array:
        return q


# ---------------------------------------------------------------------------- loader


def load_inkling2_drafter(path: str, *, quantize: bool = True, bits: int = 4,
                          group_size: int = 64, strict: bool = True):
    """``(drafter, config)`` for the head bundled in an Inkling 2 target dir.

    The three draft layers quantize to ``bits`` like every DSpark head here (drafting-only;
    the target verifies every token). The Markov head stays bf16: its ``rnn_proj`` is split
    into fp32 recurrent / context parts and ``hidden_out`` is folded through the target's
    head, both of which need the plain weights."""
    from mlx.utils import tree_flatten

    with open(os.path.join(path, "config.json")) as f:
        cfg = json.load(f)
    if not is_inkling_config(cfg):
        raise ValueError(f"{path}: not an Inkling 2 checkpoint "
                         f"(model_type {cfg.get('model_type')!r})")
    config = draft_config(cfg)
    head = os.path.join(path, HEAD_FILE)
    if not os.path.isfile(head):
        raise ValueError(f"{path}: no {HEAD_FILE} — this copy of Inkling 2 does not carry its "
                         "DSpark head (e.g. mlx-community/Inkling-2-Mini-4bit dropped it); use "
                         "thinkingmachines/Inkling-2-Mini")
    weights = dict(mx.load(head))
    drafter = Inkling2DSparkDrafter(config)
    model_keys = {k for k, _ in tree_flatten(drafter.parameters())}
    missing = sorted(model_keys - set(weights))
    unexpected = sorted(set(weights) - model_keys)
    if missing or unexpected:
        detail = (f"\n  missing ({len(missing)}): {missing[:8]}" if missing else "") + (
            f"\n  unexpected ({len(unexpected)}): {unexpected[:8]}" if unexpected else "")
        if strict:
            raise ValueError(f"{path}: {HEAD_FILE} does not match the Inkling 2 head layout"
                             f"{detail}")
        print(f"[load_inkling2_drafter] WARNING key mismatch:{detail}")
    drafter.load_weights(list(weights.items()), strict=not (missing or unexpected))
    if quantize:
        nn.quantize(drafter, group_size=group_size, bits=bits,
                    class_predicate=lambda p, m: (isinstance(m, nn.Linear)
                                                  and not p.startswith("markov_head")))
    drafter.markov_head.prepare()
    mx.eval(drafter.parameters())
    return drafter, config
