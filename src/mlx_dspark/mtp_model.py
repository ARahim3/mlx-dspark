"""Native MTP drafter — Qwen3.5/Qwen3-Next multi-token-prediction heads as a draft source.

Qwen ships an MTP head inside the base checkpoint (15 tensors under ``mtp.``): one
standard attention layer, an ``fc`` that fuses ``concat(embedding, hidden)``, and the
norms around them. It carries no ``embed_tokens`` and no ``lm_head`` — it reuses the
target's, exactly like the DFlash-warm-started DSpark heads already do here
(``has_own_embed`` / ``has_own_lm_head`` + ``bind_embed`` / ``bind_lm_head``).

The head is trained jointly with the trunk, which is why it accepts more per round than
a separately distilled drafter: measured 2.18 tokens committed per verify against the
DSpark head's 1.38 on the same Japanese chat prompt (M5 Max, Qwen3.8-27B-8bit).

**Why this is a drafter and not a second engine.** The one thing that matters for a
long conversation is that the draft state can be rolled back to an arbitrary earlier
position, because that is what lets the prefix cache reuse a prompt down to the token
instead of to a block boundary. The MTP layer is *full* attention — it is not one of
the 48 ``linear_attention`` layers of the Qwen3.8 trunk — so its K/V rows are
position-local and trim exactly, which is the same property ``prefix_cache`` already
relies on for the DSpark drafter context. So this reuses :class:`CtxCache` verbatim
rather than introducing a second cache shape.

**The contract.** Getting any of this wrong does not raise — it silently drafts worse,
which reads as "MTP is not actually better". Both of the following agree, and the
per-depth acceptance measured after implementing (see ``tests/test_mtp.py``) is the
check that they were read correctly:

    e = pre_fc_norm_embedding(embed(next_token))
    h = pre_fc_norm_hidden(hidden)
    x = fc(concat([e, h]))            # concat_order = "embedding_hidden"
    x = layer(x)                      # one standard attention block
    logits = lm_head(norm(x))         # target's head
    hidden_for_next_depth = norm(x)   # hidden_variant = "post_norm"

with the depth-1 ``hidden`` being the trunk's *post-norm* final hidden
(``base_hidden_variant = "post_norm"``) — the same tensor the trunk feeds its own
lm_head, which :meth:`Target.run` already computes and currently discards.

**Indexing.** MTP position ``t`` fuses ``(hidden_t, embed(x_{t+1}))`` and predicts
``x_{t+2}``. So a context of ``N`` committed tokens supports exactly ``N-1`` MTP rows:
the last hidden has no next token yet. That "always one behind" is the invariant this
module keeps (:meth:`ctx_len_for`), and it is what makes the context a pure function of
the committed prefix — which is what the prefix cache needs in order to snapshot it.
The unpaired last hidden is per-request scratch (:attr:`_tail`), never snapshotted; a
restored request always re-forwards at least one token (the prefix cache deliberately
snapshots below the prompt boundary), so it is always refilled before it is read.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .model import CtxCache


class MTPConfig:
    """Shape of the ``mtp.`` block, read off the checkpoint's text config."""

    def __init__(self, hidden_size: int, num_attention_heads: int, num_key_value_heads: int,
                 head_dim: int, intermediate_size: int, rms_norm_eps: float = 1e-6,
                 rope_theta: float = 100000.0, rope_parameters=None,
                 partial_rotary_factor: float = 0.25, max_position_embeddings: int = 131072,
                 attention_bias: bool = False, max_depth: int = 3):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_parameters = rope_parameters
        # Qwen3-Next ropes only the first `head_dim * partial_rotary_factor` dims (64 of
        # 256 on Qwen3.8). Roping all of them loads fine and drafts nonsense-adjacent
        # tokens that the target rejects — a throughput bug with no error attached.
        self.partial_rotary_factor = partial_rotary_factor
        self.max_position_embeddings = max_position_embeddings
        self.attention_bias = attention_bias
        # Depth is a speed/quality dial, not a correctness one: the target verifies every
        # drafted token, so a bad depth costs throughput and never output. Measured
        # per-depth acceptance on Japanese chat is [0.70, 0.29, 0.18] — depth 3 is the
        # point where the marginal draft token stops paying for its verify row.
        self.max_depth = max_depth

    @classmethod
    def from_text_config(cls, tc: dict, *, max_depth: int = 3) -> "MTPConfig":
        """Read the MTP block's shape off the target's own text config.

        The head is a sibling of the trunk's full-attention layers and shares their
        geometry, so every value here comes from the same place mlx-lm's
        ``Qwen3_5.TextModelArgs`` reads it — including the defaults, because Qwen3.8's
        config leaves ``rope_theta`` unset and nests the rope settings under
        ``rope_parameters``.
        """
        h = tc["hidden_size"]
        n_heads = tc.get("num_attention_heads", 16)
        n_kv = tc.get("num_key_value_heads", n_heads)
        head_dim = tc.get("head_dim") or (h // n_heads)
        rp = tc.get("rope_parameters") or {}
        return cls(
            hidden_size=h,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv,
            head_dim=head_dim,
            intermediate_size=tc.get("intermediate_size", 4 * h),
            rms_norm_eps=tc.get("rms_norm_eps", 1e-6),
            rope_theta=rp.get("rope_theta", tc.get("rope_theta") or 100000.0),
            rope_parameters=tc.get("rope_scaling"),
            partial_rotary_factor=rp.get(
                "partial_rotary_factor", tc.get("partial_rotary_factor", 0.25)),
            max_position_embeddings=tc.get("max_position_embeddings", 131072),
            attention_bias=bool(tc.get("attention_bias", False)),
            max_depth=max_depth,
        )


class MTPAttention(nn.Module):
    """The MTP layer's attention — Qwen3-Next's ``Attention``, driven by a :class:`CtxCache`.

    Kept a faithful mirror of ``mlx_lm.models.qwen3_next.Qwen3NextAttention`` rather than
    a generic GQA block, because two of its quirks are load-bearing and neither of them
    fails loudly if you get it wrong:

      * **gated ``q_proj``** — it emits ``2 · n_heads · head_dim`` and the second half is
        a per-head gate applied as ``o_proj(out · sigmoid(gate))``. Reading it as a plain
        projection makes the shapes work only if you also halve the head count, and the
        head then drafts from the wrong subspace.
      * **partial rope** — only ``head_dim · partial_rotary_factor`` dims are rotated
        (64 of 256 on Qwen3.8).

    Both would load cleanly and simply accept less, which is indistinguishable from
    "native MTP is not worth it" unless the acceptance rate is measured against a
    reference. That is why :func:`load_mtp` checks tensor names strictly and the bench
    reports per-depth acceptance.
    """

    def __init__(self, config: MTPConfig):
        super().__init__()
        try:  # mlx-lm keeps the rope factory next to the model families
            from mlx_lm.models.rope_utils import initialize_rope
        except ImportError:  # older layouts
            from mlx_lm.models.base import initialize_rope

        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        h, b = config.hidden_size, config.attention_bias
        self.q_proj = nn.Linear(h, self.n_heads * self.head_dim * 2, bias=b)
        self.k_proj = nn.Linear(h, self.n_kv_heads * self.head_dim, bias=b)
        self.v_proj = nn.Linear(h, self.n_kv_heads * self.head_dim, bias=b)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, h, bias=b)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rope = initialize_rope(
            int(self.head_dim * config.partial_rotary_factor),
            base=config.rope_theta,
            traditional=False,
            scaling_config=config.rope_parameters,
            max_position_embeddings=config.max_position_embeddings,
        )

    def _q_and_gate(self, x: mx.array, offset: int):
        B, S, _ = x.shape
        q, gate = mx.split(self.q_proj(x).reshape(B, S, self.n_heads, -1), 2, axis=-1)
        q = self.rope(self.q_norm(q).transpose(0, 2, 1, 3), offset=offset)
        return q, gate.reshape(B, S, -1)

    def kv(self, x: mx.array, offset: int):
        """Project ``x`` [1, S, H] -> (roped+normed K, V) at absolute positions ``offset``…

        Split out from :meth:`attend` because committing context rows needs the K/V and
        nothing else: the layer output at a committed position is never read again, so
        building the context costs projections only — no attention, no MLP.
        """
        B, S, _ = x.shape
        k = self.k_proj(x).reshape(B, S, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.rope(self.k_norm(k), offset=offset)
        v = self.v_proj(x).reshape(B, S, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        return k, v

    def attend(self, x: mx.array, offset: int, cache: CtxCache) -> mx.array:
        """Append ``x``'s K/V to ``cache`` and attend over everything held.

        Appending before attending is what lets a row see itself, which the depth-1 query
        needs: its own (hidden, next-token) pair is the newest row of its own context.
        """
        B, S, _ = x.shape
        q, gate = self._q_and_gate(x, offset)
        k, v = self.kv(x, offset)
        cache.append(k, v)
        out = mx.fast.scaled_dot_product_attention(
            q, cache.k, cache.v, scale=self.scale, mask="causal" if S > 1 else None)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out * mx.sigmoid(gate))


class MTPLayer(nn.Module):
    def __init__(self, config: MTPConfig):
        super().__init__()
        self.self_attn = MTPAttention(config)
        h, i = config.hidden_size, config.intermediate_size
        self.mlp = _MLP(h, i)
        self.input_layernorm = nn.RMSNorm(h, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(h, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, offset: int, cache: CtxCache) -> mx.array:
        x = x + self.self_attn.attend(self.input_layernorm(x), offset, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class _MLP(nn.Module):
    def __init__(self, h: int, i: int):
        super().__init__()
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MTPDrafter(nn.Module):
    """The ``mtp.`` block as a draft source for :func:`speculative_generate`.

    Exposes the pieces of the drafter protocol the shared spec loop actually calls for a
    sequential head: :meth:`make_ctx_cache`, :meth:`update_context` and
    :meth:`draft_block`. It does *not* implement the block-parallel protocol
    (``backbone`` / ``sample_block``) — an MTP head cannot produce position ``d+1``
    before position ``d`` is sampled, so the block path does not apply and the loop
    branches on :attr:`sequential`.
    """

    sequential = True
    #: no confidence head — the depth loop stops at ``max_depth``
    confidence_head = None

    def __init__(self, config: MTPConfig):
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.fc = nn.Linear(2 * h, h, bias=False)
        self.pre_fc_norm_embedding = nn.RMSNorm(h, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(h, eps=config.rms_norm_eps)
        self.layers = [MTPLayer(config)]
        self.norm = nn.RMSNorm(h, eps=config.rms_norm_eps)
        self._embed = None      # bound to the target's embed_tokens
        self._lm_head = None    # bound to the target's lm_head projection
        # The hidden whose next token is not known yet — see the module docstring.
        self._tail: mx.array | None = None

    # -- binding to the target (MTP ships neither weight) ---------------------------

    def bind_embed(self, embed_fn) -> None:
        self._embed = embed_fn

    def bind_lm_head(self, project) -> None:
        self._lm_head = project

    @property
    def max_draft(self) -> int:
        return self.config.max_depth

    # -- context -------------------------------------------------------------------

    def make_ctx_cache(self) -> list[CtxCache]:
        self._tail = None
        return [CtxCache() for _ in self.layers]

    @staticmethod
    def ctx_len_for(n_tokens: int) -> int:
        """Rows a context of ``n_tokens`` committed tokens supports. See the docstring."""
        return max(0, n_tokens - 1)

    def _fuse(self, hidden: mx.array, token_ids: mx.array) -> mx.array:
        """``fc(concat(norm_e(embed(next_token)), norm_h(hidden)))`` — the contract."""
        e = self.pre_fc_norm_embedding(self._embed(token_ids))
        h = self.pre_fc_norm_hidden(hidden)
        return self.fc(mx.concatenate([e, h], axis=-1))

    def update_context(self, hidden: mx.array, ctx_offset: int, ctx_caches,
                       token_ids=None) -> None:
        """Commit context rows.

        ``hidden`` [1, m, H] are the trunk's post-norm hiddens at absolute positions
        ``ctx_offset … ctx_offset+m-1``; ``token_ids`` are the *input* tokens at those
        same positions — which is what every caller already has (the prompt ids during
        prefill, ``[pending] + committed`` during the round).

        Row ``i`` needs the token at position ``ctx_offset+i+1``, so this pairs each
        hidden with the *following* entry of ``token_ids`` and carries the last hidden
        over in :attr:`_tail`, together with the previous call's tail paired against
        ``token_ids[0]``. Net effect: after the call the context holds exactly
        ``ctx_offset+m-1`` rows, matching :meth:`ctx_len_for`.
        """
        if token_ids is None:
            raise ValueError("MTP context needs the tokens at these positions")
        toks = token_ids if isinstance(token_ids, mx.array) else mx.array(token_ids)
        toks = toks.reshape(1, -1)
        m = hidden.shape[1]

        carried, rows, ids = self._tail, [], []
        if carried is not None:
            # last call's unpaired hidden (position ctx_offset-1), now that its next
            # token — the first token of this call — has arrived
            rows.append(carried)
            ids.append(toks[:, :1])
        if m > 1:
            rows.append(hidden[:, : m - 1, :])
            ids.append(toks[:, 1:m])
        self._tail = hidden[:, m - 1 :, :]
        if not rows:
            return

        # Committed rows are contiguous: ctx_offset-1 … when a tail was carried in,
        # ctx_offset … when this is the first call on a fresh context.
        start = ctx_offset - 1 if carried is not None else ctx_offset
        fused = self._fuse(mx.concatenate(rows, axis=1), mx.concatenate(ids, axis=1))
        x = self.layers[0].input_layernorm(fused)
        k, v = self.layers[0].self_attn.kv(x, start)
        ctx_caches[0].append(k, v)

    # -- drafting ------------------------------------------------------------------

    def draft_block(self, pending: int, n_cached: int, ctx_caches, cap: int,
                    temperature: float = 0.0, top_p: float = 1.0, top_k: int = 0):
        """Draft up to ``cap`` tokens, sequentially by depth.

        Returns ``(draft_arr, q_probs)`` with ``draft_arr`` an ``[d]`` device array and
        ``q_probs`` the ``[d, V]`` proposal distributions (``None`` when greedy). Nothing
        is synced to the host: each depth embeds the previous depth's sampled token as a
        device array, so the whole rollout — and the verify forward after it — still
        reaches the device as one graph.

        The rows appended here are speculative, so the context is trimmed back to its
        committed length before returning; the accepted ones are re-committed from the
        target's own hiddens by :meth:`update_context`.
        """
        cache = ctx_caches[0]
        committed_len = cache.length
        depth = max(1, min(cap, self.config.max_depth))

        hidden = self._tail                       # trunk hidden at position n_cached-1
        tok = mx.array([[pending]])
        drafts, qs = [], []
        for d in range(depth):
            fused = self._fuse(hidden, tok)
            x = self.layers[0](fused, committed_len + d, cache)
            hidden = self.norm(x)                 # hidden_variant = "post_norm"
            logits = self._lm_head(hidden)[:, -1, :]
            if temperature > 0.0:
                probs = mx.softmax(logits.astype(mx.float32) / temperature, axis=-1)
                qs.append(probs[0])
                tok = mx.random.categorical(mx.log(probs)).reshape(1, 1)
            else:
                tok = mx.argmax(logits, axis=-1).reshape(1, 1)
            drafts.append(tok.reshape(1))
        cache.trim_to(committed_len)              # speculative rows are not context
        draft_arr = mx.concatenate(drafts)
        return draft_arr, (mx.stack(qs) if qs else None)
