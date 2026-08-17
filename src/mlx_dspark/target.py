"""Family-aware target wrapper: KV cache + a hidden-state tap at given layers.

- gemma4 (mlx-vlm): uses the built-in ``capture_layer_ids`` / ``hidden_sink`` hook.
- qwen3  (mlx-lm):  no hook exists, so we replicate the model's forward loop and
  capture the residual stream after the tapped layers.
- qwen3_5 (mlx-lm, **hybrid** linear+full attention — the Bonsai-27B / Qwen3.6 family):
  same replicated-loop tap with per-layer fa/ssm masks, plus exact spec rollback
  (see :meth:`verify`) because the linear-attention layers carry recurrent state
  that cannot be trimmed like a KV cache.

All expose: make_cache(), run(ids, cache, tap)->(logits, fused_hidden),
plain(ids, cache)->logits (no capture), and the spec-round pair verify()/rollback()
(identical to run()+trim for dense targets).
"""

from __future__ import annotations

import sys
from contextlib import contextmanager

import mlx.core as mx

#: ``tap`` sentinel: capture the **post-norm final hidden** instead of a set of
#: intermediate residual-stream layers. This is the tensor the target already feeds its
#: own lm_head, and it is what a native MTP head consumes (``base_hidden_variant =
#: "post_norm"``). Every body below computes it anyway, so passing this only stops it
#: being discarded — the MTP draft path costs the target no extra work.
POST_NORM = "post_norm"


def _tapset(tap):
    """Layer indices to capture. The POST_NORM sentinel captures none of them."""
    return set(tap) if isinstance(tap, (list, tuple, set)) else ()


def _captured(tap, captured, hn):
    """What ``run``/``verify`` hand back as ``fused``, per tap kind."""
    if tap is POST_NORM:
        return hn
    return mx.concatenate(captured, axis=-1) if tap is not None else None


class Target:
    def __init__(self, model, tokenizer, *, kv_bits: int | None = None,
                 kv_group_size: int = 64):
        self.model = model
        self.tokenizer = tokenizer
        # mlx-vlm targets (gemma4) vs mlx-lm; mlx-lm also ships text-only modules for some
        # multimodal checkpoints (qwen3_5), which expose .language_model too — so route by
        # the class's package, not by attribute shape.
        self.is_vlm = type(model).__module__.startswith("mlx_vlm")
        # mlx-lm text stack: model.model for plain families, model.language_model.model for
        # wrapped multimodal-text families (qwen3_5).
        self._inner = (model.language_model
                       if (not self.is_vlm and hasattr(model, "language_model")) else model)
        self.family = ("gemma4" if self.is_vlm else
                       getattr(getattr(model, "args", None), "model_type", "qwen3"))
        # muse_glimmer (Meta, mlx-vlm) is a VLM but — unlike gemma4/qwen3_5 — its language
        # model ships NO capture_layer_ids/hidden_sink hook, so the drafter's hidden-state
        # tap can't use the native capture path. We replicate its text forward instead
        # (_body_muse_glimmer, the same approach the mlx-lm paths take), reusing the model's
        # own layer modules + per-layer sliding/full masks; verify_tap() proves it faithful.
        self._muse = self.is_vlm and getattr(model, "model_type", "") == "muse_glimmer"
        # hybrid = some layers hold recurrent state (no trimmable KV). Two recurrences are
        # supported, each with its own capture/rollback: qwen3_5 gated-DeltaNet (layers flagged
        # ``is_linear``, e.g. Bonsai/Qwen3.6) and Nemotron-H Mamba-2 (blocks ``block_type=="M"``).
        _layers = getattr(model, "layers", [])
        if not self.is_vlm and any(getattr(ly, "is_linear", False) for ly in _layers):
            self._recurrence = "gated_delta"
        elif not self.is_vlm and any(getattr(ly, "block_type", None) == "M" for ly in _layers):
            self._recurrence = "mamba2"
        else:
            self._recurrence = None
        self.is_hybrid = self._recurrence is not None
        if kv_bits and self.is_vlm:
            raise ValueError("--kv-bits is supported for mlx-lm text targets only "
                             "(the mlx-vlm/gemma-4 cache layout is managed by mlx-vlm)")
        if kv_bits and self.is_hybrid:
            raise ValueError("--kv-bits is unsupported for hybrid linear-attention targets "
                             "(their recurrent-state caches are not KV caches; only 16 of "
                             "64 layers even hold KV). Run without --kv-bits.")
        self.kv_bits = int(kv_bits) if kv_bits else None
        self.kv_group_size = int(kv_group_size)
        if not self.is_vlm:
            # mlx-lm convention: tied models simply don't define lm_head (more reliable
            # than args.tie_word_embeddings, which some families omit)
            self._tied = not hasattr(self._inner, "lm_head")
        # hybrid targets: the linear-attention modules, in layer order. verify() hooks
        # their recurrence + conv calls to capture per-round inputs so rollback() can
        # rebuild the state at the accept point exactly (see the section comment below).
        # Mamba-2 (nemotron_h) hooks class-level methods instead of per-instance flags.
        if self._recurrence == "gated_delta":
            self._gdn_modules = [layer.linear_attn for layer in model.layers
                                 if getattr(layer, "is_linear", False)]
            for m in self._gdn_modules:
                m.conv1d._mlx_dspark_capture = True
        # hybrid spec-verify state (see verify()/rollback()); per-generation, reset by
        # reset_spec() at loop start so a served Target never leaks state across requests.
        self.reset_spec()

    # -- cache --
    def make_cache(self):
        if self.is_vlm:
            return self.model.language_model.make_cache()
        if self.kv_bits:
            # Quantized KV from token 0: trimmable (spec rollback + prefix reuse work
            # unchanged), halves-or-quarters the KV bandwidth bill on long contexts.
            # Output is the greedy decoding of the KV-quantized target (a quality knob of
            # the same class as target quantization, not a spec-decoding approximation).
            from mlx_lm.models.cache import QuantizedKVCache
            return [QuantizedKVCache(self.kv_group_size, self.kv_bits)
                    for _ in self.model.layers]
        from mlx_lm.models.cache import make_prompt_cache
        return make_prompt_cache(self.model)

    # -- forward with hidden-state tap --
    def run(self, ids: mx.array, cache, tap: list[int]):
        """ids [1,L] -> (logits [1,L,V], fused_hidden [1,L,n_tap*H])."""
        if self._muse:
            return self._run_muse_glimmer(ids, cache, tap)
        if self.is_vlm:
            if tap is POST_NORM:
                # The mlx-vlm capture hook returns tapped residual-stream layers, not the
                # post-norm hidden, and there is no hook for the latter. Refuse rather than
                # hand an MTP head the wrong tensor — it would draft plausibly and accept
                # badly, which reads as "MTP is not worth it" instead of as a bug.
                raise ValueError(
                    "POST_NORM capture is not available on mlx-vlm targets; native-MTP "
                    "drafting currently supports the mlx-lm dense and hybrid paths.")
            out = self.model.language_model(inputs=ids, cache=cache, capture_layer_ids=tap)
            return out.logits, mx.concatenate(out.hidden_states, axis=-1)
        if self._recurrence == "mamba2":
            return self._run_nemotron(ids, cache, tap)
        if self.is_hybrid:
            return self._run_hybrid(ids, cache, tap)
        return self._run_mlxlm(ids, cache, tap)

    # The mlx-lm paths are split body/head so a prefill chunk can skip the vocab
    # projection it would throw away (see :meth:`prefill`); run() is body+full head,
    # byte-identical to the single-function version it replaced.
    def _body_mlxlm(self, ids, cache, tap):
        """Replicated mlx-lm forward through the final norm -> (hidden, fused|None)."""
        from mlx_lm.models.base import create_attention_mask

        mm = self._inner.model
        tapset = _tapset(tap)
        h = mm.embed_tokens(ids)
        mask = create_attention_mask(h, cache[0])
        captured = []
        for i, (layer, c) in enumerate(zip(mm.layers, cache)):
            h = layer(h, mask, c)
            if i in tapset:
                captured.append(h)
        hn = mm.norm(h)
        return hn, _captured(tap, captured, hn)

    def _body_hybrid(self, ids, cache, tap):
        """Replicates mlx-lm's hybrid text forward (qwen3_5-style: per-layer fa/ssm mask
        selection) with the residual-stream capture. Faithfulness is proven by
        :meth:`verify_tap` at load, exactly like the dense path."""
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        mm = self._inner.model
        tapset = _tapset(tap)
        h = mm.embed_tokens(ids)
        fa_mask = create_attention_mask(h, cache[mm.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[mm.ssm_idx])
        captured = []
        for i, (layer, c) in enumerate(zip(mm.layers, cache)):
            h = layer(h, mask=(ssm_mask if layer.is_linear else fa_mask), cache=c)
            if i in tapset:
                captured.append(h)
        hn = mm.norm(h)
        return hn, _captured(tap, captured, hn)

    def _body_nemotron(self, ids, cache, tap):
        """Replicates mlx-lm's Nemotron-H forward (hybrid Mamba-2 / attention / MoE / MLP)
        with the residual-stream capture. Differs from the qwen3_5 hybrid path in three ways
        the generic loops can't absorb: the backbone hangs off ``.backbone`` (not ``.model``)
        with ``.embeddings`` / ``.norm_f``; the cache list is COMPACTED (only ``M``/``*`` blocks
        hold a cache, indexed by a running counter — MoE/MLP blocks have none); and the per-layer
        mask is chosen by block type. Faithfulness is proven by :meth:`verify_tap` at load."""
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        bb = self._inner.backbone
        tapset = _tapset(tap)
        h = bb.embeddings(ids)
        attn_mask = create_attention_mask(h, cache[bb.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[bb.ssm_idx])
        captured = []
        ci = 0
        for i, layer in enumerate(bb.layers):
            if layer.block_type in ("M", "*"):
                c = cache[ci]
                ci += 1
            else:
                c = None
            mask = attn_mask if layer.block_type == "*" else ssm_mask
            h = layer(h, mask=mask, cache=c)
            if i in tapset:
                captured.append(h)
        hn = bb.norm_f(h)
        return hn, _captured(tap, captured, hn)

    def _run_nemotron(self, ids, cache, tap):
        hn, fused = self._body_nemotron(ids, cache, tap)
        return self._head_mlxlm(hn), fused

    def _head_mlxlm(self, hn):
        # mlx-lm convention: tied models have no lm_head, they project through the embedding
        return (self._inner.model.embed_tokens.as_linear(hn) if self._tied
                else self._inner.lm_head(hn))

    def _run_mlxlm(self, ids, cache, tap):
        hn, fused = self._body_mlxlm(ids, cache, tap)
        return self._head_mlxlm(hn), fused

    def _run_hybrid(self, ids, cache, tap):
        hn, fused = self._body_hybrid(ids, cache, tap)
        return self._head_mlxlm(hn), fused

    def _body_muse_glimmer(self, ids, cache, tap):
        """Replicates mlx-vlm's muse_glimmer text forward (3:1 sliding/full attention with
        NoPE global layers, centered RMSNorms) with the residual-stream capture. muse_glimmer
        is a VLM but ships no capture hook, so — as for the mlx-lm paths — we replicate the
        outer loop and reuse the model's OWN layer modules and its OWN per-layer masks (built
        from a representative cache of each attention type, exactly as TextModel does). Reusing
        the native sliding-window mask is what makes this correct despite the windowing that
        the generic mlx-lm tap refuses. Faithfulness is proven by :meth:`verify_tap` at load."""
        from mlx_vlm.models.base import create_attention_mask

        tm = self.model.language_model.model   # muse_glimmer TextModel
        tapset = _tapset(tap)
        h = tm.embed_norm(tm.embed_tokens(ids))
        full_mask = create_attention_mask(h, cache[tm.full_attention_idx])
        sliding_mask = (create_attention_mask(h, cache[tm.sliding_attention_idx],
                                              window_size=tm.sliding_window)
                        if tm.sliding_attention_idx is not None else None)
        captured = []
        for i, (layer, c) in enumerate(zip(tm.layers, cache)):
            mask = sliding_mask if layer.is_sliding else full_mask
            h = layer(h, mask=mask, cache=c)
            if i in tapset:
                captured.append(h)
        hn = tm.norm(h)
        return hn, _captured(tap, captured, hn)

    def _head_muse_glimmer(self, hn):
        """Project post-norm hidden -> logits with muse_glimmer's head: lm_head, scaled by
        output_multiplier, then tanh logit-softcapping (matches LanguageModel.__call__)."""
        lang = self.model.language_model
        logits = lang.lm_head(hn) * lang.output_multiplier
        sc = lang.final_logit_softcapping
        return mx.tanh(logits / sc) * sc

    def _run_muse_glimmer(self, ids, cache, tap):
        hn, fused = self._body_muse_glimmer(ids, cache, tap)
        return self._head_muse_glimmer(hn), fused

    def lm_head_proj(self, hidden: mx.array) -> mx.array:
        """Project drafter hidden ``[..., H]`` -> target-vocab logits with the TARGET's own
        head. For drafters that ship no lm_head (``has_lm_head=false``, the Nemotron / Muse-Glimmer
        DSpark heads) and reuse the target's — generate binds this via ``drafter.bind_lm_head``.

        This is the drafter's *proposal* head, which is the RAW ``lm_head`` projection — NOT the
        verifier's output transform. muse_glimmer's output head scales logits by
        ``output_multiplier`` (~0.196) and softcaps them; those belong to verification only.
        Applying them to a draft proposal shrinks the base logits so the raw-scale markov bias
        overwhelms them and d0 collapses ~85% -> ~29% with no error anywhere (the standard DSpark
        head adds markov to raw ``lm_head`` logits — every own-head family does the same). So the
        reuse path returns raw logits here while :meth:`run`/:meth:`verify` keep the full head."""
        if self._muse:
            return self.model.language_model.lm_head(hidden)
        if self.is_vlm:
            return self.model.language_model.logits_from_hidden(hidden)
        return self._head_mlxlm(hidden)

    def draft_embed(self, ids: mx.array) -> mx.array:
        """Block embedding for a drafter that reuses the target's embed_tokens
        (``has_own_embed=false``). This is the RAW token-embedding table — the DFlash reuse
        convention (``dflash_model.bind`` uses ``embed_tokens(x) * embed_scale``), which a
        DFlash-warm-started head like Muse-Glimmer was trained on. Do NOT apply muse_glimmer's
        ``embed_norm`` here: feeding the normalized input embedding measured a clear acceptance
        loss (accept 3.73 raw vs 2.86 normed on code), because the drafter's own input_layernorm
        already normalizes and the residual it carries is the raw table output, not the normed
        one. Other families fold their scaling into the table (gemma ×√H, qwen ×1)."""
        inner = self.model.language_model.model if self.is_vlm else self._inner.model
        return inner.embed_tokens(ids) * getattr(inner, "embed_scale", 1.0)

    # -- prefill forward (skips the vocab projection it would discard) ------------------
    #
    # A prefill chunk's logits are read at exactly ONE row — the last one of the last
    # chunk, which becomes the first sampled token. Projecting the other rows to vocab is
    # pure waste: measured ~7-8% of prefill FLOPs on an 8-bit Qwen3-8B, plus a
    # [chunk, vocab] transient (622 MB at chunk 2048) that is the reason PREFILL_CHUNK
    # has to be small at all. See NOTES "Prefill pass".

    def prefill(self, ids: mx.array, cache, tap: list[int] | None = None, *,
                want_logits: bool = False, head_last_row: bool = True):
        """One prefill chunk -> ``(logits | None, fused | None)``.

        ``want_logits=False`` (any non-final chunk) skips the vocab projection entirely
        and is **bit-identical** to :meth:`run` in everything it keeps — the KV cache and
        the tapped hidden states are untouched. ``head_last_row`` additionally projects
        only the final row, which moves that row from the quantized matmul path to the
        matvec path (the one every decode step already takes): the usual fp-tie class,
        measured max|Δlogit| = 0.06 on Qwen3-8B-8bit = 1-2 bf16 ulps. Set
        ``generate.PREFILL_LAST_ROW_HEAD = False`` for the bit-identical-but-slower form.
        ``tap=None`` skips the hidden-state capture."""
        if self._muse:
            hn, fused = self._body_muse_glimmer(ids, cache, tap)
            head = self._head_muse_glimmer
        elif self.is_vlm:
            lm = self.model.language_model
            sink = [] if tap is not None else None
            hn = lm.model(ids, cache=cache, capture_layer_ids=tap, hidden_sink=sink)
            fused = mx.concatenate(sink, axis=-1) if tap is not None else None
            head = lm.logits_from_hidden
        else:
            body = (self._body_nemotron if self._recurrence == "mamba2"
                    else self._body_hybrid if self.is_hybrid else self._body_mlxlm)
            hn, fused = body(ids, cache, tap)
            head = self._head_mlxlm
        if not want_logits:
            return None, fused
        return head(hn[:, -1:, :] if head_last_row else hn), fused

    # -- plain forward (no capture) for the greedy baseline --
    def plain(self, ids: mx.array, cache):
        if self.is_vlm:
            return self.model.language_model(inputs=ids, cache=cache).logits
        return self.model(ids, cache=cache)

    # -- speculative verify + rollback -------------------------------------------------
    #
    # Dense targets: verify == run, rollback == KV trim of the rejected tail (the prior
    # inline behavior, byte-identical).
    #
    # Hybrid targets: linear-attention state advances through *every* token a forward
    # touches and has no trim, so a partially-rejected verify forward would poison it.
    # verify() therefore records, per linear layer, references to the round's recurrence
    # inputs (the exact ``gated_delta_update`` args, post-conv) and the conv input — free:
    # they are intermediates of the round's graph, held one round. On a partial accept,
    # rollback() rebuilds each linear cache AT the accept point:
    #
    #   delta state  → re-run the recurrence over the accepted prefix from the pre-round
    #                  state. The kernel consumes tokens strictly sequentially, so the
    #                  state after ``keep`` tokens is independent of the rejected tail —
    #                  the re-run reproduces the original round's intermediate state
    #                  bit-for-bit (same inputs, same op order).
    #   conv window  → a slice of the recorded conv input (rows [keep, keep+k-1)) — the
    #                  exact values a committed forward of the prefix would have kept.
    #   KV layers    → trim only the rejected tail (their rows for accepted tokens are
    #                  already correct).
    #
    # So rejects cost ~48 tiny recurrence kernels (lazy, folded into the next round's
    # graph) instead of re-forwarding the accepted tokens through the whole model — the
    # earlier "replay backlog" design paid one full-model row per accepted token on every
    # partial accept, which made past-optimum caps and low-acceptance content decay much
    # faster than dense targets. (A two-phase commit+spec variant measured 0.66× baseline
    # — two full weight reads per round; still don't resurrect it.)
    #
    # The capture is two scoped hooks active only while the forward's graph is built:
    # the defining module's ``gated_delta_update`` and the conv class ``__call__`` (only
    # modules flagged ``_mlx_dspark_capture`` record). Both call straight through — no
    # numeric change — and they sit on mlx-lm's own modules, so the capture works for
    # the tap path (_run_hybrid) and the model's native forward (lookup mode) alike.

    def reset_spec(self) -> None:
        """Clear per-generation spec state (hybrid capture). Loops call this at start so
        a long-lived served Target never carries state across generations."""
        self._stash = None       # ([per-layer gated_delta args], [per-layer conv input])
        self._spec_width = 0

    @contextmanager
    def _capture_linear(self):
        """Scoped hooks recording each linear layer's recurrence args + conv input for
        the forward built inside the context. Pure pass-through (records references,
        calls the originals) — verify_tap()'s bit-exactness probe runs with the hooks
        active, so a semantic drift in this plumbing fails loudly at load."""
        gdn = self._gdn_modules[0]
        mod = sys.modules[type(gdn).__module__]
        conv_cls = type(gdn.conv1d)
        orig_gdu = mod.gated_delta_update
        orig_conv = conv_cls.__call__
        rec_delta, rec_conv = [], []

        def rec_gdu(q, k, v, a, b, A_log, dt_bias, state=None, mask=None,
                    use_kernel=True):
            # use_kernel is recorded so the rollback re-run takes the SAME
            # implementation path the forward took (bit-exact prefix re-run)
            rec_delta.append((q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel))
            return orig_gdu(q, k, v, a, b, A_log, dt_bias, state, mask,
                            use_kernel=use_kernel)

        def rec_conv_call(conv_self, x, *args, **kwargs):
            if getattr(conv_self, "_mlx_dspark_capture", False):
                rec_conv.append(x)
            return orig_conv(conv_self, x, *args, **kwargs)

        mod.gated_delta_update = rec_gdu
        conv_cls.__call__ = rec_conv_call
        try:
            yield rec_delta, rec_conv
        finally:
            mod.gated_delta_update = orig_gdu
            conv_cls.__call__ = orig_conv

    @contextmanager
    def _capture_mamba(self):
        """Scoped hooks recording each Nemotron-H Mamba-2 layer's ssm inputs (incl. the
        pre-round ssm state, which arrives as ``ssm_update``'s ``state`` arg) and its conv
        input + pre-round conv window, for the forward built inside the context. Both call
        straight through — verify_tap() runs with them active, so a semantic drift fails loudly
        at load. ``_conv`` fires before ``ssm_update`` within a mixer and mixers run in layer
        order, so ``rec_conv[i]`` / ``rec_ssm[i]`` are the i-th Mamba layer — the same order the
        non-trimmable caches appear in the compacted cache list rollback() walks."""
        import mlx_lm.models.nemotron_h as nmod

        mix_cls = nmod.NemotronHMamba2Mixer
        orig_ssm = nmod.ssm_update
        orig_conv = mix_cls._conv
        rec_ssm, rec_conv = [], []

        def rec_ssm_fn(hidden_states, A_log, B, C, D, dt, dt_bias, state=None,
                       time_step_limit=(0.001, 100.0), mask=None, lengths=None):
            rec_ssm.append((hidden_states, A_log, B, C, D, dt, dt_bias, state,
                            time_step_limit, mask))
            return orig_ssm(hidden_states, A_log, B, C, D, dt, dt_bias, state,
                            time_step_limit, mask, lengths)

        def rec_conv_fn(mix_self, conv_input, cache, mask):
            rec_conv.append((conv_input, None if cache is None else cache[0]))
            return orig_conv(mix_self, conv_input, cache, mask)

        nmod.ssm_update = rec_ssm_fn
        mix_cls._conv = rec_conv_fn
        try:
            yield rec_ssm, rec_conv
        finally:
            nmod.ssm_update = orig_ssm
            mix_cls._conv = orig_conv

    def verify(self, ids: mx.array, cache, tap: list[int] | None):
        """Spec-round verify forward: ``ids`` [1, 1+m] = [anchor] + m draft tokens.
        Returns (logits [1, 1+m, V], fused [1, 1+m, n_tap*H]) at those rows.
        ``tap=None`` skips the hidden-state capture (drafter-free lookup mode) and
        returns ``fused=None``."""

        def fwd(x):
            if tap is None:
                return self.plain(x, cache), None
            return self.run(x, cache, tap)

        if not self.is_hybrid:
            return fwd(ids)
        self._stash = None
        want = int(ids.shape[1])
        if want == 1:
            # no drafts (lookup miss / plain step): fully committed, nothing to capture
            return fwd(ids)
        capture = (self._capture_mamba if self._recurrence == "mamba2"
                   else self._capture_linear)
        with capture() as stash:
            logits, fused = fwd(ids)
        self._stash = stash
        self._spec_width = want
        return logits, fused

    def rollback(self, cache, n_rejected: int, accepted: list[int]) -> None:
        """Undo the rejected tail of the last verify. ``accepted`` = the accepted draft
        tokens (unused — kept for call-site symmetry; the hybrid rebuild slices the
        recorded arrays lazily, so the on-GPU greedy path never adds a device sync)."""
        if not self.is_hybrid:
            if n_rejected > 0:
                for c in cache:
                    if c is not None and hasattr(c, "trim"):
                        c.trim(n_rejected)
            return
        if n_rejected > 0:
            if self._stash is None:
                raise RuntimeError(
                    "rollback() with rejected tokens but no preceding verify() capture "
                    "on a hybrid target")
            keep = self._spec_width - n_rejected     # anchor + accepted drafts, >= 1
            if self._recurrence == "mamba2":
                self._rollback_mamba(cache, keep)
            else:
                self._rollback_gated_delta(cache, keep)
        self._stash = None

    def _rollback_gated_delta(self, cache, keep: int) -> None:
        from mlx_lm.models.gated_delta import gated_delta_update

        rec_delta, rec_conv = self._stash
        li = 0
        for c in cache:
            if self._is_trimmable(c):
                c.trim(self._spec_width - keep)
                continue
            q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel = rec_delta[li]
            conv_input = rec_conv[li]
            m = mask[:, :keep] if mask is not None else None
            _, new_state = gated_delta_update(
                q[:, :keep], k[:, :keep], v[:, :keep], a[:, :keep], b[:, :keep],
                A_log, dt_bias, state, m, use_kernel=use_kernel)
            n_keep = conv_input.shape[1] - self._spec_width   # conv kernel size - 1
            c[0] = mx.contiguous(conv_input[:, keep:keep + n_keep, :])
            c[1] = new_state
            li += 1

    def _rollback_mamba(self, cache, keep: int) -> None:
        """Rebuild each Nemotron-H Mamba-2 cache at the accept point (``keep`` = anchor +
        accepted drafts). Attention layers KV-trim the rejected tail; each Mamba layer:
          ssm state (``cache[1]``) → re-run ``ssm_update`` over the accepted prefix from the
              pre-round state (captured as the ssm ``state`` arg). The recurrence is causal, so
              the state after ``keep`` tokens is a function of the pre-round state and inputs
              [0, keep) only — the re-run reproduces a committed-``keep`` forward bit-for-bit.
          conv window (``cache[0]``) → the last (kernel-1) rows of
              ``[pre-round window ‖ conv_input[:keep]]`` — exactly what a committed forward of
              the accepted prefix would have kept (conv_input is per-position, so its first
              ``keep`` rows are unchanged by the rejected tail)."""
        import mlx_lm.models.nemotron_h as nmod

        rec_ssm, rec_conv = self._stash
        li = 0
        for c in cache:
            if self._is_trimmable(c):
                c.trim(self._spec_width - keep)
                continue
            hs, A_log, B, C, D, dt, dt_bias, state0, tsl, mask = rec_ssm[li]
            conv_input, conv_state0 = rec_conv[li]
            m = mask[..., :keep] if mask is not None else None
            _, new_state = nmod.ssm_update(
                hs[:, :keep], A_log, B[:, :keep], C[:, :keep], D,
                dt[:, :keep], dt_bias, state0, tsl, m)
            head = conv_input[:, :keep, :]
            padded = head if conv_state0 is None else mx.concatenate([conv_state0, head], axis=1)
            n_keep = c[0].shape[1]                    # conv_kernel - 1 (post-round window size)
            c[0] = mx.contiguous(padded[:, -n_keep:, :])
            c[1] = new_state
            li += 1

    @staticmethod
    def _is_trimmable(c) -> bool:
        fn = getattr(c, "is_trimmable", None)
        return bool(fn()) if callable(fn) else hasattr(c, "trim")

    # -- tap sanity probe (drafter modes) --
    def verify_tap(self) -> None:
        """Prove the manual mlx-lm tap is faithful for THIS model, or fail loudly.

        ``_run_mlxlm`` / ``_run_hybrid`` replicate the model's forward (embed → layers →
        norm → head). A family whose forward does more — embedding scaling (gemma),
        per-layer sliding-window masks, extra streams — would draft from a silently-wrong
        hidden stream, which wastes far more user time than an error. Two checks:
        (1) structural: refuse windowed/sliding attention (a short probe can't exercise a
        window, so it must be refused, not probed) — hybrid *linear* attention is fine,
        it has its own replicated path; (2) numeric: the replicated loop must reproduce
        the model's own logits on a tiny input (identical ops/widths → should match
        bit-for-bit; 1e-3 is generous headroom). Costs two 4-token forwards, once per
        load. VLM targets with a native capture hook (gemma4) replicate nothing, so there is
        nothing to verify; muse_glimmer replicates its forward and IS probed like the mlx-lm
        paths (its windowing is reproduced from the model's own masks, so it is not refused)."""
        if self.is_vlm and not self._muse:
            return
        args = getattr(self.model, "args", None)
        mt = getattr(args, "model_type", None) or getattr(self.model, "model_type", "?")
        layer_types = getattr(args, "layer_types", None) or []
        windowed = (any(t not in ("full_attention", "linear_attention") for t in layer_types)
                    or (getattr(args, "use_sliding_window", False)
                        and getattr(args, "sliding_window", None)))
        if windowed:
            raise ValueError(
                f"hidden-state tap unsupported for model_type {mt!r}: it uses "
                f"windowed/alternating attention layers, which the generic tap's single "
                f"causal mask would get wrong past the window. baseline/lookup modes still "
                f"work; for drafter support open an issue: "
                f"https://github.com/ARahim3/mlx-dspark/issues"
            )
        ids = mx.array([[1, 2, 3, 4]])
        try:
            ref = self.plain(ids, self.make_cache())
            if self.is_hybrid:
                # route through verify() so the capture hooks are active during the
                # probe — proves the recording plumbing is a pure pass-through too
                got, _ = self.verify(ids, self.make_cache(), [0])
                self.reset_spec()
            elif self._muse:
                got, _ = self._run_muse_glimmer(ids, self.make_cache(), [0])
            else:
                got, _ = self._run_mlxlm(ids, self.make_cache(), [0])
            diff = float(mx.abs(ref - got).max())
        except Exception as e:
            raise ValueError(
                f"hidden-state tap unsupported for model_type {mt!r}: the generic forward "
                f"loop failed ({e.__class__.__name__}: {e}). baseline/lookup modes still work."
            ) from e
        if diff > 1e-3:
            raise ValueError(
                f"hidden-state tap unsupported for model_type {mt!r}: the replicated forward "
                f"diverges from the model's own (max |Δlogit| = {diff:.4g}) — this family's "
                f"forward does more than embed→layers→norm. baseline/lookup modes still work."
            )
