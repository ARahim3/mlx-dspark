"""DSpark drafter in MLX — Gemma-4 and Qwen3 families.

Faithful port of the DeepSpec inference path. The EAGLE-style cross-attention is shared:
Q comes from the draft block, K/V from concat([fused_target_context, block]), with the
context K/V cached per layer (CtxCache). Family differences (norm layout, rope, MLP act,
v handling, logit softcap) are config-driven. Module attribute names match the HF
checkpoints so weights load 1:1.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

try:  # mlx-vlm <= 0.6.4
    from mlx_vlm.models.gemma4.rope_utils import initialize_rope
except ImportError:  # mlx-vlm >= 0.6.5 consolidated per-family rope utils
    from mlx_vlm.models.rope_utils import initialize_rope

from .config import DSparkConfig


class RMSNormNoScale(nn.Module):
    """RMSNorm with no learnable weight (Gemma-4 v_norm)."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


def _act(name: str):
    return nn.silu if name == "silu" else nn.gelu_approx


class MLP(nn.Module):
    def __init__(self, config: DSparkConfig):
        super().__init__()
        h, i = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)
        self.act = _act(config.mlp_activation)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class CtxCache:
    """Per-layer cache of the target context's projected K/V (roped K, normed/raw V).

    Append-only (the drafter context only ever grows with *committed* tokens — it is
    never trimmed/rolled back, unlike the target KV cache). A preallocated growing buffer
    (mlx-lm KVCache style) was tried to avoid the O(n²) realloc, but measured 0.99× at
    ≤600 tokens — the realloc is negligible at realistic lengths and the scatter overhead
    is not. Plain concatenate is simpler and as fast here."""

    __slots__ = ("k", "v")

    def __init__(self):
        self.k = None
        self.v = None

    def append(self, k: mx.array, v: mx.array) -> None:
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = mx.concatenate([self.k, k], axis=2)
            self.v = mx.concatenate([self.v, v], axis=2)

    def trim_to(self, length: int) -> None:
        """Keep only the first ``length`` context positions (seq axis) — used by prefix
        caching to roll the drafter context back to a shared prefix. The retained K was
        roped at its absolute position, so it stays valid after the trim."""
        if self.k is not None and length < self.k.shape[2]:
            self.k = self.k[:, :, :length, :]
            self.v = self.v[:, :, :length, :]

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[2]


class DSparkAttention(nn.Module):
    """Cross-attention: Q from the draft block, K/V from [target_context, block]."""

    def __init__(self, config: DSparkConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.head_dim = config.attn_head_dim
        self.k_eq_v = config.attention_k_eq_v
        self.n_kv_heads = config.n_kv_heads
        self.scale = config.scaling
        self.use_v_norm = config.use_v_norm
        self.gated = config.gated_q_proj
        # DFlash-lineage block attention (Nemotron DSpark head): causal within the block,
        # sliding-window over the context, per-head learned sink. All no-ops when unset
        # (getattr defaults keep minimal config-like doubles working).
        self.causal_block = getattr(config, "causal_block", False)
        self.sliding_window = getattr(config, "sliding_window", None)
        self.sink = getattr(config, "attention_sink", False)

        h = config.hidden_size
        b = config.attention_bias
        # gated (qwen3_5): q_proj emits [q ‖ gate] interleaved per head (2× out-features),
        # split per head like mlx-lm's Qwen3NextAttention so the checkpoint loads 1:1.
        self.q_proj = nn.Linear(h, self.n_heads * self.head_dim * (2 if self.gated else 1), bias=b)
        self.k_proj = nn.Linear(h, self.n_kv_heads * self.head_dim, bias=b)
        if not self.k_eq_v:
            self.v_proj = nn.Linear(h, self.n_kv_heads * self.head_dim, bias=b)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, h, bias=b)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        if self.use_v_norm:
            self.v_norm = RMSNormNoScale(eps=config.rms_norm_eps)
        if self.sink:
            # per-head sink logit, fed to SDPA as an extra always-present key with no value
            self.attention_sink_bias = mx.zeros((self.n_heads,))

        self.rope = initialize_rope(
            dims=config.rope_dims or self.head_dim, base=config.rope_theta,
            traditional=False, scaling_config=config.rope_parameters,
        )

    def _kv(self, x: mx.array):
        """Project x -> (roped+normed K, V). k_eq_v shares k_proj for V."""
        B, S, _ = x.shape
        kp = self.k_proj(x).reshape(B, S, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_norm(kp)
        if self.k_eq_v:
            v = self.v_norm(kp)
        else:
            v = self.v_proj(x).reshape(B, S, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
            if self.use_v_norm:
                v = self.v_norm(v)
        return k, v

    def update_ctx(self, fused_new: mx.array, ctx_offset: int, cache: CtxCache) -> None:
        k, v = self._kv(fused_new)
        cache.append(self.rope(k, offset=ctx_offset), v)   # V is not roped

    def attend(self, hidden: mx.array, block_offset, cache, mask=None) -> mx.array:
        """``block_offset`` may be an int (single sequence) or a per-row ``[B]`` array (batched
        drafting — rows sit at different context lengths). ``mask`` is None for the single-seq
        path (block attends the whole context + all block positions) or a ``[B, 1, k, Lctx+k]``
        boolean mask that hides each row's context padding when ``cache.k/v`` are a batched buffer."""
        B, q_len, _ = hidden.shape
        q = self.q_proj(hidden).reshape(B, q_len, self.n_heads, -1)
        gate = None
        if self.gated:
            q, gate = mx.split(q, 2, axis=-1)          # per-head [q ‖ gate]
            gate = gate.reshape(B, q_len, -1)          # gate is NOT q-normed (qwen3_5 semantics)
        q = self.rope(self.q_norm(q).transpose(0, 2, 1, 3), offset=block_offset)

        k_blk, v_blk = self._kv(hidden)
        k_blk = self.rope(k_blk, offset=block_offset)
        k = mx.concatenate([cache.k, k_blk], axis=2)
        v = mx.concatenate([cache.v, v_blk], axis=2)

        # DFlash-lineage heads mask the block attention (causal within the block, sliding window
        # over the context) and add a per-head sink logit. The default DSpark head does neither:
        # mask stays None (full bidirectional block over the whole context) and sinks stays None,
        # so this is byte-identical there. The mask is only built for the single-sequence path
        # (int offset); the batched drafter path already supplies its own [B,1,k,L] mask.
        if mask is None and (self.causal_block or self.sliding_window) and isinstance(block_offset, int):
            mask = self._block_mask(block_offset, ctx_len=k.shape[2] - q_len, q_len=q_len)
        sinks = self.attention_sink_bias if self.sink else None

        # SDPA does GQA/MQA head broadcast internally — pass the n_kv-head K/V straight through.
        # The old code tiled K/V up to full heads (`_repeat_kv`, n_rep=4 Qwen / 16 Gemma) across the
        # *whole* context cache every round: O(n_rep · ctx_len) of pure wasted memory traffic that
        # grows with depth. On cheap-verify targets (Qwen-class), where the drafter is the dominant
        # share of each round, that made long-context drafting collapse to net-negative past a few
        # thousand tokens (measured 0.6× at 8k on Qwen3-4B); removing it restores a flat ~1.6× out
        # to 12k+. On expensive-verify targets (Gemma-12B) the drafter is a small fraction, so it was
        # already amortized — neutral there. Bit-for-bit identical output (same math, no redundant
        # tiling), strictly less work on every target.
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask, sinks=sinks)
        out = out.transpose(0, 2, 1, 3).reshape(B, q_len, -1)
        if gate is not None:
            out = out * mx.sigmoid(gate)
        return self.o_proj(out)

    def _block_mask(self, block_offset: int, ctx_len: int, q_len: int) -> mx.array | None:
        """Boolean ``[1, 1, q_len, ctx_len+q_len]`` attention mask (True = attend) for the block.

        Keys are ``[context(0..ctx_len-1), block(ctx_len..ctx_len+q_len-1)]`` at contiguous
        absolute positions (context length == block_offset). Query i sits at absolute
        ``block_offset + i``. Causal keeps ``key_pos <= query_pos`` (block causal, full context);
        the sliding window additionally keeps ``key_pos > query_pos - W``. Returns None when
        neither constraint bites (short context, no causality) so SDPA takes its fast path."""
        total = ctx_len + q_len
        W = self.sliding_window
        if not self.causal_block and (W is None or total <= W):
            return None
        qpos = block_offset + mx.arange(q_len)[:, None]        # [q_len, 1]
        kpos = mx.arange(total)[None, :]                       # [1, total]
        allowed = mx.ones((q_len, total), dtype=mx.bool_)
        if self.causal_block:
            allowed = allowed & (kpos <= qpos)
        if W is not None:
            allowed = allowed & (kpos > qpos - W)
        return allowed[None, None]


class DSparkDecoderLayer(nn.Module):
    def __init__(self, config: DSparkConfig):
        super().__init__()
        eps = config.rms_norm_eps
        self.norm_style = config.norm_style
        self.self_attn = DSparkAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        if self.norm_style == "gemma":
            self.pre_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
            self.post_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
            self.layer_scalar = mx.ones((1,))

    def __call__(self, hidden, block_offset, cache, mask=None):
        if self.norm_style == "gemma":
            residual = hidden
            h = self.input_layernorm(hidden)
            h = self.self_attn.attend(h, block_offset, cache, mask)
            h = self.post_attention_layernorm(h)
            h = residual + h
            residual = h
            h = self.pre_feedforward_layernorm(h)
            h = self.mlp(h)
            h = self.post_feedforward_layernorm(h)
            h = residual + h
            return h * self.layer_scalar
        # qwen / llama 2-norm
        residual = hidden
        h = self.input_layernorm(hidden)
        h = self.self_attn.attend(h, block_offset, cache, mask)
        h = residual + h
        residual = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        return residual + h


class VanillaMarkov(nn.Module):
    """Rank-256 previous-token correction: logits += w2(w1[prev_token])."""

    def __init__(self, config: DSparkConfig):
        super().__init__()
        # Asymmetric on a reduced-draft-vocab head: w1 is indexed by the PREVIOUS token,
        # which is a target id, while w2's bias adds onto lm_head's draft-space logits.
        self.markov_w1 = nn.Embedding(config.vocab_size, config.markov_rank)
        self.markov_w2 = nn.Linear(config.markov_rank, config.out_vocab_size, bias=False)

    def prev_embeddings(self, token_ids: mx.array) -> mx.array:
        return self.markov_w1(token_ids)

    def step_bias(self, token_ids: mx.array) -> mx.array:
        return self.markov_w2(self.markov_w1(token_ids))


class ConfidenceHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)

    def __call__(self, features: mx.array) -> mx.array:
        return self.proj(features).squeeze(-1)


LOG_SNR_FREQS = 128  # sinusoidal feature count of the GIDD LogSnrEmbed (fixed by training)


def log_snr_features(block_size: int, min_log_snr: float, max_log_snr: float) -> mx.array:
    """``[block_size, 128]`` sinusoidal features for the fixed inference-time log-SNR
    pattern (PrismML dspark.cpp): the anchor at block position 0 is "clean" (max_log_snr),
    every masked position is "fully noised" (min_log_snr); each value maps to
    ``t = (snr - min) / (max - min) * 1000`` and is featurized as
    ``[sin(t·f_i), cos(t·f_i)]`` with ``f_i = 10000^(-i/64)`` — the sin half first,
    matching the reference layout exactly."""
    import math

    half = LOG_SNR_FREQS // 2
    pos = mx.arange(block_size)
    log_snr = mx.where(pos == 0, max_log_snr, min_log_snr).astype(mx.float32)
    t = (log_snr - min_log_snr) / (max_log_snr - min_log_snr) * 1000.0
    freq = mx.exp(-math.log(10000.0) * mx.arange(half).astype(mx.float32) / half)
    angle = t[:, None] * freq[None, :]
    return mx.concatenate([mx.sin(angle), mx.cos(angle)], axis=-1)


class LogSnrEmbed(nn.Module):
    """GIDD log-SNR conditioning MLP (Bonsai drafters): 128 sinusoidal features →
    fc1 → silu → fc2 → an additive embedding on the draft block. Since the inference
    pattern is a pure function of the block position, the drafter caches the resulting
    ``[1, block_size, H]`` addend after the first call."""

    def __init__(self, config: DSparkConfig):
        super().__init__()
        self.fc1 = nn.Linear(LOG_SNR_FREQS, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(self, feat: mx.array) -> mx.array:
        return self.fc2(nn.silu(self.fc1(feat)))


class DSparkDrafter(nn.Module):
    def __init__(self, config: DSparkConfig):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.mask_token_id = config.mask_token_id
        self.logits_start = config.logits_start
        self.causal_block = getattr(config, "causal_block", False)
        self.embed_scale = (float(config.hidden_size) ** 0.5) if config.family == "gemma4" else 1.0
        self.softcap = config.final_logit_softcapping

        # A has_own_embed=false drafter (DFlash-warm-started, e.g. Muse-Glimmer) ships no
        # embed_tokens and reuses the TARGET's input embedding for the draft block; bind_embed()
        # supplies it at run time. True = the drafter embeds the block with its own table.
        self.embed_tokens = (nn.Embedding(config.vocab_size, config.hidden_size)
                             if getattr(config, "has_own_embed", True) else None)
        self._ext_embed = None       # target's input-embedding fn, bound by generate for reuse
        self.fc = nn.Linear(
            len(config.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False
        )
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [DSparkDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # A has_lm_head=false drafter (Nemotron DSpark head) reuses the TARGET's lm_head — it
        # ships no lm_head weight, so we don't build one; bind_lm_head() supplies it at run time.
        self.lm_head = (nn.Linear(config.hidden_size, config.out_vocab_size, bias=False)
                        if getattr(config, "has_own_lm_head", True) else None)
        self._ext_lm_head = None    # target's head projection, bound by generate for reuse heads

        self.markov_head = VanillaMarkov(config) if config.markov_rank > 0 else None
        self.confidence_head = None
        if config.enable_confidence_head:
            in_dim = config.hidden_size + (config.markov_rank if config.confidence_head_with_markov else 0)
            self.confidence_head = ConfidenceHead(in_dim)
        self.log_snr_embed = LogSnrEmbed(config) if config.log_snr_conditioning else None
        self._snr_addend = None       # lazily-built [1, block_size, H] constant
        # Draft->target id table for reduced-vocab heads; load_drafter installs it via
        # set_draft_vocab_map(). Leading underscore keeps it out of mlx's parameter filter —
        # it is an index table, and must be neither quantized nor name-checked as a weight.
        self._draft_to_target = None

    def embed(self, ids: mx.array) -> mx.array:
        if self.embed_tokens is not None:
            e = self.embed_tokens(ids) * self.embed_scale
        else:
            if self._ext_embed is None:
                raise RuntimeError(
                    "drafter has has_own_embed=false but no target embedding was bound — call "
                    "drafter.bind_embed(target.draft_embed) before generating.")
            # The bound fn already returns the target's full input embedding (norm/scale folded
            # in), so embed_scale is not reapplied here.
            e = self._ext_embed(ids)
        if self.log_snr_embed is not None:
            if self._snr_addend is None:
                feat = log_snr_features(
                    self.block_size, self.config.min_log_snr, self.config.max_log_snr)
                self._snr_addend = self.log_snr_embed(feat.astype(e.dtype))[None]
            # ids is the draft block ([1, block_size]); slicing keeps shorter probes valid.
            e = e + self._snr_addend[:, : e.shape[1], :]
        return e

    def fuse_target(self, target_hidden_cat: mx.array) -> mx.array:
        return self.hidden_norm(self.fc(target_hidden_cat))

    def make_ctx_cache(self) -> list[CtxCache]:
        return [CtxCache() for _ in self.layers]

    def update_context(self, target_hidden_cat, ctx_offset, ctx_caches) -> None:
        fused = self.fuse_target(target_hidden_cat)
        for layer, cache in zip(self.layers, ctx_caches):
            layer.self_attn.update_ctx(fused, ctx_offset, cache)

    def backbone(self, noise_embedding, block_offset, ctx_caches, mask=None) -> mx.array:
        h = noise_embedding
        if mask is None and isinstance(block_offset, int) and self.layers:
            # Every layer shares one attention config, so the block mask (a function of
            # block_offset / ctx_len / q_len only) is identical across them — build it once
            # here instead of 5x per round. attend() keeps its own fallback for direct
            # callers; None stays None for the default bidirectional DSpark head.
            attn = self.layers[0].self_attn
            if attn.causal_block or attn.sliding_window:
                mask = attn._block_mask(
                    block_offset, ctx_len=ctx_caches[0].length, q_len=h.shape[1])
        for layer, cache in zip(self.layers, ctx_caches):
            h = layer(h, block_offset, cache, mask)
        return self.norm(h)

    @property
    def max_draft(self) -> int:
        """Most tokens this head can propose from one block (anchor slots excluded)."""
        return self.block_size - self.logits_start

    def draft_width(self, cap: int) -> int:
        """Block positions the backbone must compute to draft ``cap`` tokens.

        A bidirectional block (every DeepSpec-native head) needs the full trained width —
        each position's hidden depends on the whole block, so shrinking it would change the
        distribution the drafter was trained on. A CAUSAL block (DFlash-lineage heads:
        Nemotron, Muse-Glimmer) is mathematically invariant to truncation — position i
        attends only positions <= i, so rows past the last one the head reads
        (``logits_start + cap``) contribute nothing to the draft. Muse-Glimmer's block is 15
        wide while the shipped cap is 4: running the 2.3B-param backbone at width 4 instead
        of 15 measured ~16 ms/round back (~10% end-to-end) with the draft rows bit-identical.
        """
        if not self.causal_block:
            return self.block_size
        return min(self.block_size, self.logits_start + max(1, cap))

    def head_slice(self, block_hidden: mx.array, cap: int) -> mx.array:
        """The block positions whose logits are draft predictions: ``[:, s : s+cap, :]``.

        DeepSpec heads use anchor-as-pos0 (``s=0``) — slot 0 both embeds the known token and
        predicts the first draft token. DFlash-derived speculators heads reserve slot 0 as a
        pure anchor and predict from slot 1 (``s=1``). Slicing from 0 on those re-predicts the
        token we already have, so every draft lands one position late (draft[i+1] == target[i])
        and acceptance quietly halves — measured 1.50 -> 3.10 at cap 4 on
        makora-ai/gemma4-26b-a4b-dspark. There is no error and the text stays correct, because
        the target verifies everything; only the speedup disappears.
        """
        s = self.logits_start
        return block_hidden[:, s : s + cap, :]

    def bind_lm_head(self, project) -> None:
        """Give a reuse head (``has_lm_head=false``) the target's own projection. ``project``
        maps drafter hidden ``[..., H]`` -> target-vocab logits; generate binds it once from
        the loaded Target so the drafter emits into the same vocabulary it was trained against."""
        self._ext_lm_head = project

    def bind_embed(self, embed_fn) -> None:
        """Give a reuse-embed drafter (``has_own_embed=false``) the target's input-embedding
        function. ``embed_fn`` maps token ids -> the same ``[..., H]`` representation the
        target's own layers consume (e.g. muse_glimmer's ``embed_norm∘embed_tokens``); generate
        binds it once from the loaded Target so the drafter embeds the block exactly as trained."""
        self._ext_embed = embed_fn

    def compute_logits(self, hidden: mx.array) -> mx.array:
        """Logits over the drafter's own vocabulary — the *draft* space on a reduced-vocab
        head (width ``config.out_vocab_size``). Callers that turn these into token ids must
        go through ``sample_block`` / ``to_target_ids``."""
        head = self.lm_head if self.lm_head is not None else self._ext_lm_head
        if head is None:
            raise RuntimeError(
                "drafter has has_lm_head=false but no target head was bound — call "
                "drafter.bind_lm_head(target.lm_head_proj) before generating.")
        logits = head(hidden)
        if self.softcap is not None:
            logits = mx.tanh(logits / self.softcap) * self.softcap
        return logits

    def set_draft_vocab_map(self, d2t_offsets: mx.array) -> None:
        """Install a reduced-vocab head's draft->target id mapping.

        Checkpoints store ``d2t`` as OFFSETS (``target = draft + d2t[draft]``), not absolute
        ids; we materialize the absolute table once so the hot path is a plain gather.
        The encoding is self-proving: applying the offsets to ``0..draft_vocab-1``
        reproduces exactly the set of ids the checkpoint's own ``t2d`` boolean mask marks
        (verified on makora-ai/gemma4-26b-a4b-dspark, all 32000).
        """
        n = d2t_offsets.size
        self._draft_to_target = (
            mx.arange(n, dtype=mx.int32) + d2t_offsets.astype(mx.int32)
        )

    def to_target_ids(self, draft_ids: mx.array) -> mx.array:
        """Map draft-vocabulary ids to target ids (identity on a full-vocab head)."""
        if self._draft_to_target is None:
            return draft_ids
        return self._draft_to_target[draft_ids]

    def widen_to_target(self, q: mx.array) -> mx.array:
        """Widen a draft-space probability vector to the target vocabulary, zeros elsewhere.

        Speculative *sampling* compares the draft q against the target p index-by-index and
        resamples rejects from ``norm(max(0, p - q))`` over the full target vocab — the
        target routinely puts mass on tokens a reduced draft head cannot represent, and
        those must survive into the residual rather than be dropped.
        """
        if self._draft_to_target is None:
            return q
        full = mx.zeros(q.shape[:-1] + (self.config.vocab_size,), dtype=q.dtype)
        return full.at[..., self._draft_to_target].add(q)

    def sample_block(self, base_logits: mx.array, first_prev_token: int) -> mx.array:
        """Greedy draft block. Returns TARGET-vocabulary ids."""
        k = base_logits.shape[0]
        if self.markov_head is None:
            return self.to_target_ids(mx.argmax(base_logits, axis=-1))
        tokens = []
        prev = mx.array([first_prev_token])
        for i in range(k):
            step = base_logits[i] + self.markov_head.step_bias(prev)[0]
            # Map before the id is emitted OR fed back: markov_w1 indexes target ids.
            nxt = self.to_target_ids(mx.argmax(step, axis=-1, keepdims=True))
            tokens.append(nxt)
            prev = nxt
        return mx.concatenate(tokens)

    def sample_block_probs(self, base_logits: mx.array, first_prev_token: int,
                           temperature: float, top_p: float = 1.0,
                           top_k: int = 0) -> tuple[mx.array, mx.array]:
        """Temperature draft for speculative *sampling*: sample each block position from
        its (temperature-scaled, optionally top-p/top-k truncated) distribution and return
        ``(tokens [k], probs [k, V])``. ``probs[i]`` is the draft distribution q_i that token
        i was sampled from — the verifier needs it for the accept test ``min(1, p_i/q_i)`` and
        residual resampling. Truncating q here (same top-p/top-k as the target) keeps
        acceptance from collapsing when a client asks for nucleus sampling; losslessness comes
        from the *target* side (see ``_spec_sample_accept``). Sequential because the Markov
        bias for position i depends on the token sampled at i-1.

        Both returns are in TARGET index space: on a reduced-vocab head the sampling happens
        in draft space and the id/distribution are mapped out before returning, so V is the
        target vocab either way."""
        from .sampling import sample_probs, truncate_probs

        k = base_logits.shape[0]
        inv_t = 1.0 / temperature
        tokens, probs = [], []
        prev = mx.array([first_prev_token])
        for i in range(k):
            logits = base_logits[i]
            if self.markov_head is not None:
                logits = logits + self.markov_head.step_bias(prev)[0]
            q = truncate_probs(mx.softmax(logits * inv_t, axis=-1), top_p, top_k)
            # Sample against q's own index space, then move both onto the target's.
            nxt = self.to_target_ids(sample_probs(q).reshape(1))
            probs.append(self.widen_to_target(q))
            tokens.append(nxt)
            prev = nxt
        return mx.concatenate(tokens), mx.stack(probs, axis=0)

    def confidence_logits(self, block_hidden, prev_token_ids):
        if self.confidence_head is None:
            return None
        if self.config.confidence_head_with_markov:
            feats = mx.concatenate(
                [block_hidden, self.markov_head.prev_embeddings(prev_token_ids)], axis=-1
            )
        else:
            feats = block_hidden
        return self.confidence_head(feats)


# Backwards-compatible alias
Gemma4DSparkDrafter = DSparkDrafter
