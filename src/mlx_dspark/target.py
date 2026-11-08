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

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass

import mlx.core as mx


class _RowRoPE:
    """A layer's rope applied at per-ROW positions ``offset + depth[row]`` (tree verify). The
    rows become batch entries so ``mx.fast.rope``'s per-batch offsets apply — bit-identical to
    a single-row call per row. Wraps the layer's own rope object (any mlx-lm rope variant that
    forwards ``offset`` to ``mx.fast.rope``)."""

    def __init__(self, rope, depth: mx.array):
        self.rope, self.depth = rope, depth
        self.calls = 0                       # tree_selftest's structural check

    def __call__(self, x, offset=0):
        self.calls += 1
        # x [1, H, W, D] -> [W, H, 1, D]: one batch entry per row
        y = self.rope(x.transpose(2, 1, 0, 3), offset=self.depth + offset)
        return y.transpose(2, 1, 0, 3)


def _compact_kv(c, start: int, rows: mx.array, keep: int) -> None:
    """Move KV rows ``start + rows`` to ``start .. start+keep-1`` in place (tree commit).
    Gathered before the write (MLX is functional), so overlapping moves are safe. Handles the
    plain cache (keys/values arrays) and the quantized one (tuples of packed/scales/biases)."""
    src = rows + start

    def mv(buf):
        buf[..., start:start + keep, :] = buf[..., src, :]

    for side in (c.keys, c.values):
        for buf in (side if isinstance(side, tuple) else (side,)):
            mv(buf)


# Deferred commit (gated-DeltaNet hybrids, Target.rollback / commit_tree): a round's accepted rows
# are folded into the stored recurrence state by the NEXT verify's kernel, in front of its own
# rows — one state read + one write per round — instead of a rollback re-run (a second read +
# write, ~2.8 ms per round on a 27B; every tree round paid it, a chain on every partial accept).
# MLX_DSPARK_EAGER_COMMIT=1 restores the eager re-run for an A/B: same tokens, same states.
DEFERRED_COMMIT = os.environ.get("MLX_DSPARK_EAGER_COMMIT", "").lower() not in ("1", "true")


@dataclass
class _Pending:
    """Rows of a gated-DeltaNet hybrid's last verify that were accepted but are not in the
    stored recurrence state yet: per linear layer, in layer order (the order the non-trimmable
    caches appear in ``cache``), the exact ``gated_delta_update`` inputs of those rows.
    ``cache[i][1]`` is the state they start from. Folded by the next verify's hook
    (``Target._capture_linear`` / ``_tree_linear``) or by ``Target.flush_pending``."""

    cache: list
    rows: list          # [(q, k, v, a, b, A_log, dt_bias, use_kernel, kw)] per linear layer

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
        # nanbeige (Nanbeige4.2): a looped-depth transformer — num_hidden_layers dense
        # blocks applied num_loops times with shared weights, per-loop KV caches, and the
        # final RMSNorm at the end of every loop pass. The drafter's target_layer_ids index
        # the UNROLLED layer stream (num_loops * num_hidden_layers positions), so the
        # generic dense body (one pass over zip(layers, cache)) cannot serve the tap;
        # _body_nanbeige replicates the looped forward. Everything else is dense-standard:
        # all caches are trimmable KVCache, so verify/rollback/prefix-cache take the
        # ordinary dense paths.
        self._nanbeige = (not self.is_vlm) and self.family == "nanbeige"
        # hybrid = some layers hold recurrent state (no trimmable KV). Two recurrences are
        # supported, each with its own capture/rollback: qwen3_5 gated-DeltaNet (layers flagged
        # ``is_linear``, e.g. Bonsai/Qwen3.6) and Nemotron-H Mamba-2 (blocks ``block_type=="M"``).
        _layers = getattr(model, "layers", [])
        if not self.is_vlm and any(getattr(ly, "is_linear", False) for ly in _layers):
            self._recurrence = "gated_delta"
        elif not self.is_vlm and any(getattr(ly, "block_type", None) == "M" for ly in _layers):
            self._recurrence = "mamba2"
        elif not self.is_vlm and any(
                (not getattr(ly, "is_attention_layer", True)) and hasattr(ly, "conv")
                for ly in _layers):
            # LFM2 (model_type lfm2): short-conv blocks (a kernel-3 causal FIR with an
            # ArraysCache window) interleaved with full-attention blocks. The conv state is
            # non-trimmable like the other recurrences, so it needs capture/rollback — but it
            # is a pure FIR (no SSM accumulation), so rollback is just the conv-window slice.
            self._recurrence = "shortconv"
        else:
            self._recurrence = None
        self.is_hybrid = self._recurrence is not None
        if kv_bits and self.is_vlm:
            raise ValueError("--kv-bits is supported for mlx-lm text targets only "
                             "(the mlx-vlm/gemma-4 cache layout is managed by mlx-vlm)")
        if kv_bits and self._recurrence == "mamba2":
            raise ValueError("--kv-bits is not yet supported for Mamba-2 hybrid targets "
                             "(nemotron_h) — the mixed-cache path is only validated on the "
                             "gated-DeltaNet families. Run without --kv-bits.")
        # gated-DeltaNet hybrids (qwen3_5 / qwen3_5_moe): kv_bits quantizes ONLY the
        # full-attention layers' KV caches (make_cache builds a mixed list); the recurrent
        # ArraysCache layers are fixed-size state, not KV, and stay untouched. Those few KV
        # layers are the ENTIRE per-token cache growth (64 KB/token on Qwen3.8-27B) and the
        # entire verify depth slope — see NOTES "Long-context decode" (2026-08-20).
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
        elif self._recurrence == "shortconv":
            # The class whose __call__ we hook to capture each round's conv inputs. All conv
            # blocks share one ShortConv class, so a class-method hook records every conv call
            # in layer order (the same order the non-trimmable caches appear in the cache list).
            self._shortconv_cls = type(
                next(ly.conv for ly in model.layers if not ly.is_attention_layer))
        # hybrid spec-verify state (see verify()/rollback()); per-generation, reset by
        # reset_spec() at loop start so a served Target never leaks state across requests.
        self._pending: _Pending | None = None    # deferred-commit rows (gated-DeltaNet)
        self._capturing = False                  # inside verify(): run()/plain() don't flush
        self._fold_ok: bool | None = None        # fold_selftest verdict, once per model
        self.reset_spec()

    # -- cache --
    def make_cache(self):
        if self.is_vlm:
            return self.model.language_model.make_cache()
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(self.model)
        if self.kv_bits:
            # Quantized KV from token 0: trimmable (spec rollback + prefix reuse work
            # unchanged), halves-or-quarters the KV bandwidth bill on long contexts.
            # Output is the greedy decoding of the KV-quantized target (a quality knob of
            # the same class as target quantization, not a spec-decoding approximation).
            # Replace exactly the plain-KVCache entries: on a dense target that is every
            # layer (the old uniform build); on a gated-DeltaNet hybrid only the
            # full-attention layers — the recurrent ArraysCache layers are fixed-size
            # state, not KV. mlx-lm's shared attention helper dispatches per-cache
            # (hasattr(cache, "bits")), so a mixed list needs no model-side support.
            from mlx_lm.models.cache import KVCache, QuantizedKVCache
            cache = [QuantizedKVCache(self.kv_group_size, self.kv_bits)
                     if type(c) is KVCache else c for c in cache]
        return cache

    # -- forward with hidden-state tap --
    def run(self, ids: mx.array, cache, tap: list[int]):
        """ids [1,L] -> (logits [1,L,V], fused_hidden [1,L,n_tap*H])."""
        self._flush_if_pending_for(cache)
        if self._muse:
            return self._run_muse_glimmer(ids, cache, tap)
        if self.is_vlm:
            out = self.model.language_model(inputs=ids, cache=cache, capture_layer_ids=tap)
            return out.logits, mx.concatenate(out.hidden_states, axis=-1)
        if self._recurrence == "mamba2":
            return self._run_nemotron(ids, cache, tap)
        if self._recurrence == "shortconv":
            return self._run_lfm2(ids, cache, tap)
        if self.is_hybrid:
            return self._run_hybrid(ids, cache, tap)
        if self._nanbeige:
            return self._run_nanbeige(ids, cache, tap)
        return self._run_mlxlm(ids, cache, tap)

    # The mlx-lm paths are split body/head so a prefill chunk can skip the vocab
    # projection it would throw away (see :meth:`prefill`); run() is body+full head,
    # byte-identical to the single-function version it replaced.
    def _body_mlxlm(self, ids, cache, tap):
        """Replicated mlx-lm forward through the final norm -> (hidden, fused|None)."""
        from mlx_lm.models.base import create_attention_mask

        mm = self._inner.model
        tapset = set(tap) if tap is not None else ()
        h = mm.embed_tokens(ids)
        mask = create_attention_mask(h, cache[0])
        captured = []
        for i, (layer, c) in enumerate(zip(mm.layers, cache)):
            h = layer(h, mask, c)
            if i in tapset:
                captured.append(h)
        return mm.norm(h), (mx.concatenate(captured, axis=-1) if tap is not None else None)

    def _body_hybrid(self, ids, cache, tap):
        """Replicates mlx-lm's hybrid text forward (qwen3_5-style: per-layer fa/ssm mask
        selection) with the residual-stream capture. Faithfulness is proven by
        :meth:`verify_tap` at load, exactly like the dense path."""
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        mm = self._inner.model
        tapset = set(tap) if tap is not None else ()
        h = mm.embed_tokens(ids)
        fa_mask = create_attention_mask(h, cache[mm.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[mm.ssm_idx])
        captured = []
        for i, (layer, c) in enumerate(zip(mm.layers, cache)):
            h = layer(h, mask=(ssm_mask if layer.is_linear else fa_mask), cache=c)
            if i in tapset:
                captured.append(h)
        return mm.norm(h), (mx.concatenate(captured, axis=-1) if tap is not None else None)

    def _body_nemotron(self, ids, cache, tap):
        """Replicates mlx-lm's Nemotron-H forward (hybrid Mamba-2 / attention / MoE / MLP)
        with the residual-stream capture. Differs from the qwen3_5 hybrid path in three ways
        the generic loops can't absorb: the backbone hangs off ``.backbone`` (not ``.model``)
        with ``.embeddings`` / ``.norm_f``; the cache list is COMPACTED (only ``M``/``*`` blocks
        hold a cache, indexed by a running counter — MoE/MLP blocks have none); and the per-layer
        mask is chosen by block type. Faithfulness is proven by :meth:`verify_tap` at load."""
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        bb = self._inner.backbone
        tapset = set(tap) if tap is not None else ()
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
        return bb.norm_f(h), (mx.concatenate(captured, axis=-1) if tap is not None else None)

    def _run_nemotron(self, ids, cache, tap):
        hn, fused = self._body_nemotron(ids, cache, tap)
        return self._head_mlxlm(hn), fused

    def _body_lfm2(self, ids, cache, tap):
        """Replicates mlx-lm's LFM2 text forward (short-conv / full-attention hybrid) with the
        residual-stream capture. Differs from the qwen3_5 hybrid path only in surface detail the
        generic loops can't absorb: the final norm is ``.embedding_norm`` (not ``.norm``); layers
        are typed by ``is_attention_layer`` (not ``is_linear``); the per-layer mask is the causal
        attention mask for attention layers and the SSM mask for conv layers; and the cache list
        is NOT compacted (one entry per layer — KVCache for attention, ArraysCache for conv), so
        it indexes by layer position exactly like the dense path. Faithfulness is proven by
        :meth:`verify_tap` at load."""
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        mm = self._inner.model   # Lfm2Model
        tapset = set(tap) if tap is not None else ()
        h = mm.embed_tokens(ids)
        attn_mask = create_attention_mask(h, cache[mm.fa_idx])
        conv_mask = create_ssm_mask(h, cache[mm.conv_idx])
        captured = []
        for i, (layer, c) in enumerate(zip(mm.layers, cache)):
            mask = attn_mask if layer.is_attention_layer else conv_mask
            h = layer(h, mask=mask, cache=c)
            if i in tapset:
                captured.append(h)
        return mm.embedding_norm(h), (mx.concatenate(captured, axis=-1) if tap is not None else None)

    def _run_lfm2(self, ids, cache, tap):
        hn, fused = self._body_lfm2(ids, cache, tap)
        return self._head_mlxlm(hn), fused

    def _body_nanbeige(self, ids, cache, tap):
        """Replicates the vendored nanbeige forward (looped-depth: the shared layer stack
        applied num_loops times, per-loop KV cache slices, final RMSNorm at the end of every
        loop pass) with the residual-stream capture. Tap indices address the UNROLLED stream
        — loop_idx * n_layers + layer_idx — matching the drafter's target_layer_ids /
        num_target_layers (= num_loops * num_hidden_layers; the Nanbeige4.2-3B head declares
        44 over a 22-layer, 2-loop target).

        Capture convention (from the reference, Nanbeige/sglang@nbg42
        ``srt/models/nanbeige.py``): a tap at unrolled id k is captured *before unrolled
        step k+1* — the residual stream after layer k for a mid-loop tap, but for a tap at
        the LAST layer of a non-final loop (id 21 here) the value entering the next loop,
        i.e. AFTER the inter-loop residual fold + RMSNorm. Capturing that boundary tap
        pre-norm feeds the drafter one silently-wrong fused feature of five (the offset-
        RMSNorm bug class: no error anywhere, only lost acceptance). A tap at the very
        last unrolled layer is never reachable in the reference loop and is refused.
        Faithfulness of the forward itself is proven by :meth:`verify_tap` at load."""
        from mlx_lm.models.base import create_attention_mask

        mm = self._inner.model
        want = {t + 1 for t in tap} if tap is not None else ()
        h = mm.embed_tokens(ids)
        mask = create_attention_mask(h, cache[0])
        captured = []
        n = len(mm.layers)
        total = mm.num_loops * n
        if want and max(want) > total - 1:
            raise ValueError(
                f"nanbeige tap id {max(want) - 1} is not capturable: the reference captures "
                f"before the NEXT unrolled step, and there is none after layer {total - 1}.")
        for loop_idx in range(mm.num_loops):
            loop_cache = cache[loop_idx * n:(loop_idx + 1) * n]
            for j, (layer, c) in enumerate(zip(mm.layers, loop_cache)):
                if loop_idx * n + j in want:
                    captured.append(h)
                h = layer(h, mask, c)
            if not mm.skip_loop_final_norm:
                h = mm.norm(h)
        if mm.skip_loop_final_norm:
            h = mm.norm(h)
        return h, (mx.concatenate(captured, axis=-1) if tap is not None else None)

    def _run_nanbeige(self, ids, cache, tap):
        hn, fused = self._body_nanbeige(ids, cache, tap)
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
        tapset = set(tap) if tap is not None else ()
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
        return tm.norm(h), (mx.concatenate(captured, axis=-1) if tap is not None else None)

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
                    else self._body_lfm2 if self._recurrence == "shortconv"
                    else self._body_hybrid if self.is_hybrid
                    else self._body_nanbeige if self._nanbeige else self._body_mlxlm)
            hn, fused = body(ids, cache, tap)
            head = self._head_mlxlm
        if not want_logits:
            return None, fused
        return head(hn[:, -1:, :] if head_last_row else hn), fused

    # -- plain forward (no capture) for the greedy baseline --
    def plain(self, ids: mx.array, cache):
        self._flush_if_pending_for(cache)
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
        a long-lived served Target never carries state across generations. Pending
        deferred-commit rows (the loops flush at their exit; this is the backstop) are folded
        into their own cache first."""
        self.flush_pending()
        self._stash = None       # ([per-layer gated_delta args], [per-layer conv input])
        self._spec_width = 0

    def _flush_if_pending_for(self, cache) -> None:
        """A forward outside verify() on a cache with pending rows reads the stored state
        directly — fold them first. Inside verify()/verify_tree() the hooks fold them."""
        p = self._pending
        if p is not None and p.cache is cache and not self._capturing:
            self.flush_pending()

    @contextmanager
    def _capture_linear(self, cache=None):
        """Scoped hooks recording each linear layer's recurrence args + conv input for the
        forward built inside the context, and folding the PENDING rows (``_Pending`` for this
        ``cache``: rows earlier rounds accepted but never folded into the stored state) in front
        of the new rows — through the tree kernel where it serves the shape (one state read, one
        state write: the deferred commit), else the stock recurrence twice (fold, then forward:
        the eager cost, the same result). With nothing pending the hooks are a pure pass-through
        for the logits and leave the STORED state untouched (the new rows pend at rollback) —
        verify_tap()'s bit-exactness probe runs with them active."""
        from mlx_lm.models.gated_delta import compute_g

        from .tree_gdn import chain_parent, eligible, tree_gated_delta

        gdn = self._gdn_modules[0]
        mod = sys.modules[type(gdn).__module__]
        conv_cls = type(gdn.conv1d)
        orig_gdu = mod.gated_delta_update
        orig_conv = conv_cls.__call__
        rec_delta, rec_conv = [], []
        pend = (self._pending.rows if self._pending is not None and self._pending.cache is cache
                else None)
        self._fold_kernel_hits = 0

        def rec_gdu(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True,
                    **kw):
            li = len(rec_delta)
            # use_kernel is recorded so the fold takes the SAME implementation path the
            # forward took (bit-exact re-run)
            rec_delta.append((q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel, kw))
            if pend is None:
                out, _ = orig_gdu(q, k, v, a, b, A_log, dt_bias, state, mask,
                                  use_kernel=use_kernel, **kw)
                return out, state                # stored state stays; the rows pend at rollback
            qp, kp, vp, ap, bp = pend[li][:5]
            if self._fold_ok and use_kernel and not kw and mask is None and state is not None:
                g = compute_g(A_log, a, dt_bias)
                if eligible(q, v, g, state):
                    self._fold_kernel_hits += 1
                    return tree_gated_delta(
                        q, k, v, g, mx.sigmoid(b), state, chain_parent(int(q.shape[1])),
                        pending=(qp, kp, vp, compute_g(A_log, ap, dt_bias), mx.sigmoid(bp)))
            _, base = orig_gdu(qp, kp, vp, ap, bp, A_log, dt_bias, state, None,
                               use_kernel=use_kernel, **kw)
            out, _ = orig_gdu(q, k, v, a, b, A_log, dt_bias, base, mask,
                              use_kernel=use_kernel, **kw)
            return out, base

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

    @contextmanager
    def _capture_shortconv(self):
        """Scoped hook recording each LFM2 short-conv block's input ``x`` and its pre-round conv
        window (``cache[0]``, read BEFORE the call overwrites it), for the forward built inside the
        context. Pure pass-through — verify_tap() runs with it active, so a semantic drift fails
        loudly at load. Conv blocks share one ShortConv class and run in layer order, so the
        recorded list is in the same order the non-trimmable caches appear in the cache list."""
        cls = self._shortconv_cls
        orig = cls.__call__
        rec = []

        def rec_call(conv_self, x, mask=None, cache=None):
            rec.append((conv_self, x, None if cache is None else cache[0], mask))
            return orig(conv_self, x, mask=mask, cache=cache)

        cls.__call__ = rec_call
        try:
            yield rec
        finally:
            cls.__call__ = orig

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
            # no drafts (lookup miss / plain step): fully committed, nothing to capture — the
            # stock forward reads the stored state, so pending rows are folded first
            self.flush_pending()
            return fwd(ids)
        if self._recurrence == "gated_delta" and self._fold_ok is None:
            self._fold_ok = self.fold_selftest()
        capture = (self._capture_mamba() if self._recurrence == "mamba2"
                   else self._capture_shortconv() if self._recurrence == "shortconv"
                   else self._capture_linear(cache))
        self._capturing = True
        try:
            with capture as stash:
                logits, fused = fwd(ids)
        finally:
            self._capturing = False
        if self._pending is not None and self._pending.cache is cache:
            self._pending = None             # folded into this forward (lazily, into cache[1])
        self._stash = stash
        self._spec_width = want
        return logits, fused

    def rollback(self, cache, n_rejected: int, accepted: list[int]) -> None:
        """Undo the rejected tail of the last verify. ``accepted`` = the accepted draft
        tokens (unused — kept for call-site symmetry; the hybrid rebuild slices the
        recorded arrays lazily, so the on-GPU greedy path never adds a device sync).

        Gated-DeltaNet hybrids DEFER the state rebuild: the kept rows (anchor + accepted
        drafts — all of them on a full accept, since the forward left the stored state
        untouched) are recorded as pending and folded by the next verify's kernel in front of
        its own rows: one state read + write per round instead of the re-run's second pair.
        Every other reader of the stored state flushes first (:meth:`flush_pending`: plain
        steps, run()/plain(), reset_spec(), the loops' exit). ``MLX_DSPARK_EAGER_COMMIT=1``
        folds right here instead (the A/B)."""
        if not self.is_hybrid:
            if n_rejected > 0:
                for c in cache:
                    if c is not None and hasattr(c, "trim"):
                        c.trim(n_rejected)
            return
        if self._recurrence == "gated_delta":
            if self._stash is None:
                if n_rejected > 0:
                    raise RuntimeError(
                        "rollback() with rejected tokens but no preceding verify() capture "
                        "on a hybrid target")
                return                     # a width-1 step: committed in full by its forward
            self._defer_gated_delta(cache, self._spec_width - n_rejected)
            if not DEFERRED_COMMIT:
                self.flush_pending()
            self._stash = None
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
                self._rollback_shortconv(cache, keep)
        self._stash = None

    def _defer_gated_delta(self, cache, keep: int) -> None:
        """KV layers: trim the rejected rows. Linear layers: the kept rows' recurrence inputs
        become the pending rows (the stored state is still the pre-round one — the forward's
        hook left it, see _capture_linear); the conv window is rebuilt now (cheap, exact)."""
        rec_delta, rec_conv = self._stash
        rows, li = [], 0
        for c in cache:
            if self._is_trimmable(c):
                c.trim(self._spec_width - keep)
                continue
            q, k, v, a, b, A_log, dt_bias, _state, _mask, use_kernel, kw = rec_delta[li]
            rows.append((q[:, :keep], k[:, :keep], v[:, :keep], a[:, :keep], b[:, :keep],
                         A_log, dt_bias, use_kernel, kw))
            conv_input = rec_conv[li]
            n_keep = conv_input.shape[1] - self._spec_width   # conv kernel size - 1
            c[0] = mx.contiguous(conv_input[:, keep:keep + n_keep, :])
            li += 1
        self._pending = _Pending(cache, rows)

    def flush_pending(self) -> None:
        """Fold the pending rows into the stored recurrence state NOW, with the stock
        recurrence on the implementation path their forward took: afterwards the caches hold
        exactly what a chain verify of the committed tokens leaves — what every reader outside
        a verify (prefix-cache snapshots, plain steps, the next generation) expects. No-op when
        nothing is pending."""
        p = self._pending
        if p is None:
            return
        from mlx_lm.models.gated_delta import gated_delta_update

        li = 0
        for c in p.cache:
            if self._is_trimmable(c):
                continue
            q, k, v, a, b, A_log, dt_bias, use_kernel, kw = p.rows[li]
            _, c[1] = gated_delta_update(q, k, v, a, b, A_log, dt_bias, c[1], None,
                                         use_kernel=use_kernel, **kw)
            li += 1
        self._pending = None

    def fold_selftest(self, atol: float = 1.0) -> bool:
        """Prove the deferred commit on THIS model once, before verify() trusts the kernel
        fold: two rounds (verify, defer two rows, verify again — the second folds the pending
        rows in front of its own) through the tree kernel vs the same two rounds folded by the
        stock recurrence. Structural: the kernel must have served every linear layer. Numeric:
        same argmax and |Δlogit| <= ``atol`` (bitwise on mlx-lm's simd_sum and packed kernels —
        the tolerance is the fp-tie headroom two kernels are allowed, not an expectation), and
        the base state the fold wrote within fp32 noise (read right after the fold). Fails closed to the eager fold (stock kernel
        twice: the pre-deferral cost, identical results)."""
        prompt, tap = [1, 2, 3, 4, 5, 6], [0]
        saved = (self._pending, self._stash, self._spec_width, self._fold_ok)
        self._pending = None

        def two_rounds(fold_ok: bool):
            self._fold_ok = fold_ok
            cache = self.make_cache()
            self.run(mx.array([prompt]), cache, tap)
            self.verify(mx.array([[7, 8, 9]]), cache, tap)
            self._defer_gated_delta(cache, 2)                    # rows [7, 8] pend
            logits, _ = self.verify(mx.array([[10, 11, 12]]), cache, tap)
            hits = self._fold_kernel_hits
            # the base the fold wrote, read right here: rows folded later decay a wrong base
            # away (the gate), so a check after them would miss it
            states = [c[1] for c in cache if not self._is_trimmable(c)]
            mx.eval(logits, *states)
            return logits, states, hits

        try:
            la, sa, hits = two_rounds(True)
            lb, sb, _ = two_rounds(False)
            ok = hits == len(self._gdn_modules) > 0
            d = float(mx.abs(la.astype(mx.float32) - lb.astype(mx.float32)).max())
            ok = ok and bool((mx.argmax(la, -1) == mx.argmax(lb, -1)).all()) and d <= atol
            worst = 0.0
            for x, y in zip(sa, sb):
                scale = max(1.0, float(mx.abs(y).max()))
                worst = max(worst, float(mx.abs(x - y).max()) / scale)
            ok = ok and worst <= 1e-4
            self._fold_selftest_worst = (d, worst)
        except Exception:  # noqa: BLE001 — a probe failure means "eager fold", not a crash
            ok = False
        finally:
            self._pending, self._stash, self._spec_width, self._fold_ok = saved
        return ok

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

    def _rollback_shortconv(self, cache, keep: int) -> None:
        """Rebuild each LFM2 short-conv cache at the accept point (``keep`` = anchor + accepted
        drafts). Attention layers KV-trim the rejected tail; each conv layer's only state is the
        window ``cache[0]`` = the last (L_cache-1) rows of ``[pre-round window ‖ Bx]``, where
        ``Bx = (B·x)`` is per-position (each row depends only on that token's residual, which for
        the accepted prefix is causally identical to a committed forward). So the committed-``keep``
        window is the last (L_cache-1) rows of ``[pre-round window ‖ Bx[:keep]]``. Re-running the
        block's ``in_proj`` over ``x[:keep]`` reproduces ``Bx[:keep]`` bit-for-bit (same op,
        position-wise) — the FIR carries no accumulated state, so no recurrence re-run is needed."""
        rec = self._stash
        li = 0
        for c in cache:
            if self._is_trimmable(c):
                c.trim(self._spec_width - keep)
                continue
            conv, x, state0, mask = rec[li]
            BCx = conv.in_proj(x[:, :keep, :])
            B, _C, xs = mx.split(BCx, 3, axis=-1)
            Bx = B * xs
            if mask is not None:                      # None in the batch-1 spec path
                Bx = mx.where(mask[:, :keep, None], Bx, 0)
            n_keep = conv.L_cache - 1
            if state0 is None:                        # first conv call had an empty window
                state0 = mx.zeros((Bx.shape[0], n_keep, Bx.shape[-1]), dtype=Bx.dtype)
            padded = mx.concatenate([state0, Bx], axis=1)
            c[0] = mx.contiguous(padded[:, -n_keep:, :])
            li += 1

    @staticmethod
    def _is_trimmable(c) -> bool:
        fn = getattr(c, "is_trimmable", None)
        return bool(fn()) if callable(fn) else hasattr(c, "trim")

    # -- tree verify (best-first draft trees, see tree.py) ---------------------------------
    #
    # A tree of N drafted nodes verifies in ONE forward of width N+1 (row 0 = the anchor),
    # the same cost as a chain of cap N. Three per-family changes, each a scoped patch over
    # mlx-lm's OWN modules, make every node compute what a chain verify along its own root
    # path computes (each measured bit-identical in isolation: per-row rope offsets, gathered
    # conv windows, the path-batched gated-delta kernel):
    #   attention    rope at position L + depth instead of L + row (a per-row-offset wrapper
    #                around the layer's own rope); an ancestor mask over the tree's rows (the
    #                committed prefix is visible to every row). K/V rows land in tree order;
    #                commit_tree() moves the accepted path into place and trims the rest.
    #   conv1d       each row's causal window is gathered along its own root path
    #                (DraftTree.conv_index) and convolved as a batch of kernel-length windows.
    #   gated delta  the recurrence runs path-batched — one batch row per leaf path, the
    #                pre-round state broadcast; each node reads its output from a path that
    #                contains it. Verify leaves no usable state; commit_tree() re-runs the
    #                recurrence over the accepted path from the pre-round state (the same
    #                capture-and-rerun rollback chains use).
    # tree_selftest() proves the plumbing on THIS model before the loop trusts it — a missed
    # patch (say, a family that applies rope elsewhere) would put rows at wrong positions,
    # i.e. the target would "decide" with the wrong model, so it fails closed to chains.

    @property
    def supports_tree(self) -> bool:
        """Structural eligibility for tree verify: mlx-lm text targets whose attention layers
        take ``self_attn.rope`` + a mask — dense families and the qwen3_5 gated-DeltaNet
        hybrids. VLM, Mamba-2, short-conv, looped (nanbeige) and muse targets run chains."""
        return (not self.is_vlm and not self._muse and not self._nanbeige
                and self._recurrence in (None, "gated_delta"))

    def tree_ready(self) -> bool:
        """``supports_tree`` AND the one-time :meth:`tree_selftest` passed on this model."""
        if not self.supports_tree:
            return False
        ok = getattr(self, "_tree_ok", None)
        if ok is None:
            try:
                ok = self.tree_selftest()
            except Exception:  # noqa: BLE001 — a probe failure means "use chains", not a crash
                ok = False
            self._tree_ok = ok
        return ok

    @property
    def tree_kernel_live(self) -> bool:
        """After :meth:`tree_ready`: True when this model's tree verifies run the recurrence
        through the tree kernel on every linear layer, or the model has no recurrence. False =
        the exact path-batched fallback (no verified kernel on this GPU generation, odd head
        dims), which re-reads and re-writes the fp32 state once per LEAF — a cost a chain never
        pays and ``calibrate.static_tree`` must measure before trusting a tree here."""
        if self._recurrence != "gated_delta":
            return True
        calls = getattr(self, "_tree_calls", None) or {}
        return calls.get("gdu", 0) > 0 and calls.get("kernel", 0) == calls["gdu"]

    def _tree_conv_k(self) -> int:
        """The linear layers' conv kernel size (the window DeviceTree.conv_idx must index)."""
        if self._recurrence == "gated_delta":
            return int(self._gdn_modules[0].conv_kernel_size)
        return 4

    @staticmethod
    def _kv_offset(cache) -> int:
        for c in cache:
            off = getattr(c, "offset", None)
            if off is not None and Target._is_trimmable(c):
                return int(off)
        raise ValueError("tree verify needs at least one KV-cached attention layer")

    @contextmanager
    def _tree_rope(self, depth: mx.array):
        """Swap every attention layer's rope for a per-row-position wrapper (restored on exit)."""
        patched = []
        for layer in self._inner.model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is None or "rope" not in attn:
                continue
            orig = attn.rope
            attn.rope = _RowRoPE(orig, depth)
            patched.append((attn, orig))
        try:
            yield [attn.rope for attn, _ in patched]
        finally:
            for attn, orig in patched:
                attn.rope = orig

    @contextmanager
    def _tree_linear(self, tree, cache=None):
        """Gated-DeltaNet hybrids: the tree recurrence + per-path conv windows for the forward
        built inside the context, recording each linear layer's inputs for commit_tree(); the
        PENDING rows of ``cache`` (deferred commit, see rollback) are folded in front of the tree
        — by the tree kernel's pending segment, or by the stock recurrence on the fallback. A
        no-op context on dense targets."""
        self._tree_kernel_hits = 0        # linear layers the tree kernel served (vs the fallback)
        if self._recurrence != "gated_delta":
            yield None
            return
        from mlx_lm.models.gated_delta import compute_g

        from .tree_gdn import eligible as _tree_kernel_ok
        from .tree_gdn import tree_gated_delta

        gdn = self._gdn_modules[0]
        mod = sys.modules[type(gdn).__module__]
        conv_cls = type(gdn.conv1d)
        orig_gdu = mod.gated_delta_update
        orig_conv = conv_cls.__call__
        conv_idx = tree.conv_idx                                              # [W, K]
        parent = tree.parent
        pend = (self._pending.rows if self._pending is not None and self._pending.cache is cache
                else None)
        rec_delta, rec_conv = [], []
        batch: dict = {}

        def fold_stock(pr, A_log, dt_bias, state, use_kernel, kw):
            qp, kp, vp, ap, bp = pr[:5]
            _, base = orig_gdu(qp, kp, vp, ap, bp, A_log, dt_bias, state, None,
                               use_kernel=use_kernel, **kw)
            return base

        def path_batched(q, k, v, a, b, A_log, dt_bias, state, use_kernel, kw):
            # exact on any kernel: one batch row per leaf path (re-reads the state per leaf);
            # needs the tree on the host — the off-kernel fallback only (CPU, odd head dims)
            if not batch:
                host = tree.to_host()
                paths = host.leaf_paths()
                dmax = max(len(p) for p in paths)
                p_of, d_of = host.row_slots()
                batch.update(n=len(paths), pidx=mx.array(p_of, dtype=mx.int32),
                             didx=mx.array(d_of, dtype=mx.int32),
                             gidx=mx.array([p + [p[-1]] * (dmax - len(p)) for p in paths],
                                           dtype=mx.int32))
            gidx = batch["gidx"]
            st = (None if state is None
                  else mx.broadcast_to(state, (batch["n"],) + tuple(state.shape[1:])))
            out, _ = orig_gdu(q[0][gidx], k[0][gidx], v[0][gidx], a[0][gidx],
                              b[0][gidx], A_log, dt_bias, st, None,
                              use_kernel=use_kernel, **kw)
            return out[batch["pidx"], batch["didx"]][None], state   # the base, not a leaf's end

        def tree_gdu(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True,
                     **kw):
            li = len(rec_delta)
            rec_delta.append((q, k, v, a, b, A_log, dt_bias, state, use_kernel, kw))
            pr = None if pend is None else pend[li]
            if use_kernel and not kw and state is not None:
                # the gate and step size exactly as gated_delta_update derives them
                g = compute_g(A_log, a, dt_bias)
                if _tree_kernel_ok(q, v, g, state):
                    self._tree_kernel_hits += 1
                    pt = None
                    if pr is not None and self._fold_ok:
                        pt = (pr[0], pr[1], pr[2], compute_g(A_log, pr[3], dt_bias),
                              mx.sigmoid(pr[4]))
                    elif pr is not None:
                        state = fold_stock(pr, A_log, dt_bias, state, use_kernel, kw)
                    return tree_gated_delta(q, k, v, g, mx.sigmoid(b), state, parent,
                                            pending=pt)
            if pr is not None:
                state = fold_stock(pr, A_log, dt_bias, state, use_kernel, kw)
            return path_batched(q, k, v, a, b, A_log, dt_bias, state, use_kernel, kw)

        def tree_conv(conv_self, x, *args, **kwargs):
            if not getattr(conv_self, "_mlx_dspark_capture", False):
                return orig_conv(conv_self, x, *args, **kwargs)
            rec_conv.append(x)                             # [1, K-1 + W, C]
            y = orig_conv(conv_self, x[0][conv_idx], *args, **kwargs)        # [W, 1, C]
            return y.reshape(1, y.shape[0], y.shape[-1])

        mod.gated_delta_update = tree_gdu
        conv_cls.__call__ = tree_conv
        try:
            yield rec_delta, rec_conv
        finally:
            mod.gated_delta_update = orig_gdu
            conv_cls.__call__ = orig_conv

    @contextmanager
    def _tree_kv(self, cache):
        """Record the new K/V rows each KV-cached layer appends during the tree verify — the
        exact (post-rope) rows written — so commit_tree() can re-append the accepted ones from
        THESE arrays. Gathering them back out of the cache buffer instead would keep the buffer
        referenced by the very graph that writes it, so MLX could not update it in place and
        copied the whole thing (measured +3.8 ms per commit at 16k context, ~1 GB moved)."""
        rows: dict[int, tuple] = {}
        mine = {id(c) for c in cache if self._is_trimmable(c)}
        patched = []
        for cls in {type(c) for c in cache if self._is_trimmable(c)}:
            orig = cls.update_and_fetch
            own = "update_and_fetch" in cls.__dict__

            def rec(c_self, keys, values, _orig=orig):
                if id(c_self) in mine:
                    rows[id(c_self)] = (keys, values)
                return _orig(c_self, keys, values)

            cls.update_and_fetch = rec
            patched.append((cls, orig, own))
        try:
            yield rows
        finally:
            for cls, orig, own in patched:
                if own:
                    cls.update_and_fetch = orig
                else:
                    del cls.update_and_fetch              # it was inherited: restore that

    def _body_tree(self, ids, cache, tap, mask):
        """The replicated text forward with the tree mask on attention layers (linear layers
        take none — their tree structure lives in the _tree_linear hooks)."""
        mm = self._inner.model
        tapset = set(tap) if tap is not None else ()
        h = mm.embed_tokens(ids)
        captured = []
        for i, (layer, c) in enumerate(zip(mm.layers, cache)):
            h = layer(h, None if getattr(layer, "is_linear", False) else mask, c)
            if i in tapset:
                captured.append(h)
        return mm.norm(h), (mx.concatenate(captured, axis=-1) if tap is not None else None)

    def verify_tree(self, tree, cache, tap: list[int] | None):
        """Tree verify forward over ``tree.tokens`` (row 0 = anchor, then the drafted nodes in
        topological order). Returns ``(logits [1, W, V], fused [1, W, n_tap*H] | None)`` per
        row, like :meth:`verify`; follow with :meth:`commit_tree`."""
        from .multirow_attn import tree_mask
        from .tree import DeviceTree, to_device

        dt = tree if isinstance(tree, DeviceTree) else to_device(tree, self._tree_conv_k())
        L = self._kv_offset(cache)
        w = dt.width
        mask = mx.concatenate([mx.ones((w, L), dtype=mx.bool_), dt.anc], axis=1)[None, None]
        self._tree_stash = None
        if self._recurrence == "gated_delta" and self._fold_ok is None:
            self._fold_ok = self.fold_selftest()
        self._capturing = True
        try:
            with self._tree_rope(dt.depth) as ropes, self._tree_linear(dt, cache) as stash, \
                    self._tree_kv(cache) as kv_rows, tree_mask(mask, dt.bits):
                hn, fused = self._body_tree(dt.tokens[None], cache, tap, mask)
                logits = self._head_mlxlm(hn)
        finally:
            self._capturing = False
        if self._pending is not None and self._pending.cache is cache:
            self._pending = None             # folded into this forward (lazily, into cache[1])
        self._tree_stash = (w, L, stash, kv_rows)
        # structural record for tree_selftest: every patch must actually have been used
        self._tree_calls = {"rope": [r.calls for r in ropes],
                            "gdu": len(stash[0]) if stash else 0,
                            "conv": len(stash[1]) if stash else 0,
                            "kernel": getattr(self, "_tree_kernel_hits", 0)}
        return logits, fused

    def commit_tree(self, cache, path: list[int]) -> None:
        """Leave the caches holding exactly ``[prefix ‖ path rows]`` after :meth:`verify_tree`
        (``path`` = accepted rows incl. the root, topological order — :meth:`DraftTree.walk`):
          KV layers   the path's K/V rows move to positions L..L+len(path)-1 (a no-op when
                      the path is already rows 0..k), then the rest is trimmed;
          linear      the path's recurrence inputs become the PENDING rows (deferred commit —
                      the stored state is the one the forward wrote: the state after the rows
                      that were pending before it, which these rows start from); the conv
                      window = the last (kernel-1) rows of [pre-round window ‖ path].
        Same result as a chain verify of the path followed by its rollback."""
        if getattr(self, "_tree_stash", None) is None:
            raise RuntimeError("commit_tree() without a preceding verify_tree()")
        w, L, stash, kv_rows = self._tree_stash
        keep = len(path)
        contiguous = list(path) == list(range(keep))
        pidx = mx.array(path, dtype=mx.int32)
        rows, li = [], 0
        for c in cache:
            if self._is_trimmable(c):
                if contiguous:                       # the common case (main path first)
                    c.trim(w - keep)
                elif id(c) in kv_rows:
                    k_new, v_new = kv_rows[id(c)]    # back to the prefix, then re-append the
                    c.trim(w)                        # accepted rows from the verify's own
                    c.update_and_fetch(k_new[..., pidx, :], v_new[..., pidx, :])   # arrays
                else:
                    _compact_kv(c, L, pidx, keep)
                    c.trim(w - keep)
                continue
            rec_delta, rec_conv = stash
            q, k, v, a, b, A_log, dt_bias, _state0, use_kernel, kw = rec_delta[li]

            def sel(x):
                return x[:, :keep] if contiguous else x[:, pidx]

            rows.append((sel(q), sel(k), sel(v), sel(a), sel(b), A_log, dt_bias, use_kernel, kw))
            conv_input = rec_conv[li]
            n_keep = conv_input.shape[1] - w                 # conv kernel - 1
            crow = list(range(n_keep)) + [n_keep + r for r in path]
            c[0] = mx.contiguous(conv_input[:, mx.array(crow[-n_keep:], dtype=mx.int32)])
            li += 1
        if rows:
            self._pending = _Pending(cache, rows)
            if not DEFERRED_COMMIT:
                self.flush_pending()
        self._tree_stash = None

    def tree_selftest(self, atol: float = 1.0) -> bool:
        """Prove tree verify on THIS model, once, before the loop trusts it.

        Structural (magnitude-independent): every attention layer's rope wrapper ran exactly
        twice (queries, keys) and, on a hybrid, every linear layer's conv + recurrence went
        through the tree hooks — a family that applies rope elsewhere or a renamed mlx-lm
        function fails here outright. Numeric: every row of a branched tree must give the
        same argmax as a chain verify of its root path, within ``atol`` (the measured
        kernel-path noise on Qwen3.8-27B-4bit is <= 0.31 — tree and chain take different qmm
        widths and mask paths, the same fp-tie class as two different caps); committing a
        NON-contiguous path must leave caches that give the chain-committed next-step logits."""
        from .tree import DraftTree

        prompt = [1, 2, 3, 4, 5, 6]
        #            0   1   2   3   4   5       (rows)
        tree = DraftTree(tokens=[7, 8, 9, 10, 11, 12], parent=[-1, 0, 1, 0, 3, 1])
        tap = [0]

        def prefilled():
            c = self.make_cache()
            self.run(mx.array([prompt]), c, tap)
            return c

        def close(a, b) -> tuple[bool, float]:
            a, b = a.astype(mx.float32), b.astype(mx.float32)
            d = float(mx.abs(a - b).max())
            return bool((mx.argmax(a, -1) == mx.argmax(b, -1)).all()) and d <= atol, d

        self.reset_spec()
        cache = prefilled()
        t_logits, _ = self.verify_tree(tree, cache, tap)
        mx.eval(t_logits)
        calls = self._tree_calls
        n_attn = sum(1 for ly in self._inner.model.layers
                     if getattr(ly, "self_attn", None) is not None)
        ok = len(calls["rope"]) == n_attn > 0 and all(c == 2 for c in calls["rope"])
        if self._recurrence == "gated_delta":
            n_lin = len(self._gdn_modules)
            ok = ok and calls["gdu"] == n_lin and calls["conv"] == n_lin
        worst = 0.0
        for leaf_path in tree.leaf_paths():
            ref_c = prefilled()
            ref, _ = self.verify(mx.array([[tree.tokens[r] for r in leaf_path]]), ref_c, tap)
            self.reset_spec()
            good, d = close(t_logits[0, mx.array(leaf_path)], ref[0])
            ok, worst = ok and good, max(worst, d)
        # commit a non-contiguous path, then one more committed step vs the chain reference
        path = [0, 3, 4]
        self.commit_tree(cache, path)
        got_next, _ = self.run(mx.array([[13]]), cache, tap)
        ref_next, _ = self.run(mx.array([[tree.tokens[r] for r in path] + [13]]), prefilled(),
                               tap)
        good, d = close(got_next[0, -1:], ref_next[0, -1:])
        ok, worst = ok and good, max(worst, d)
        self.reset_spec()
        self._tree_selftest_worst = worst
        return ok

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
        # "conv" = LFM2 short-conv blocks: a recurrence with its own replicated path + rollback
        # (like linear_attention), not windowed attention, so it is allowed, not refused.
        windowed = (any(t not in ("full_attention", "linear_attention", "conv")
                        for t in layer_types)
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
            elif self._nanbeige:
                got, _ = self._run_nanbeige(ids, self.make_cache(), [0])
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
