"""Family-aware target wrapper: KV cache + a hidden-state tap at given layers.

- gemma4 (mlx-vlm): uses the built-in ``capture_layer_ids`` / ``hidden_sink`` hook.
- qwen3  (mlx-lm):  no hook exists, so we replicate the model's forward loop and
  capture the residual stream after the tapped layers.
- qwen3_5 (mlx-lm, **hybrid** linear+full attention — the Bonsai-27B / Qwen3.6 family):
  same replicated-loop tap with per-layer fa/ssm masks, plus two-phase spec verify
  (see :meth:`verify`) because the 48 linear-attention layers carry recurrent state
  that cannot be trimmed like a KV cache.

All expose: make_cache(), run(ids, cache, tap)->(logits, fused_hidden),
plain(ids, cache)->logits (no capture), and the spec-round pair verify()/rollback()
(identical to run()+trim for dense targets).
"""

from __future__ import annotations

import mlx.core as mx


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
        # hybrid = some layers hold recurrent (linear-attention) state instead of KV.
        self.is_hybrid = (not self.is_vlm and any(
            getattr(layer, "is_linear", False) for layer in getattr(model, "layers", [])))
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
        if self.is_vlm:
            out = self.model.language_model(inputs=ids, cache=cache, capture_layer_ids=tap)
            return out.logits, mx.concatenate(out.hidden_states, axis=-1)
        if self.is_hybrid:
            return self._run_hybrid(ids, cache, tap)
        return self._run_mlxlm(ids, cache, tap)

    def _run_mlxlm(self, ids, cache, tap):
        from mlx_lm.models.base import create_attention_mask

        mm = self._inner.model
        tapset = set(tap)
        h = mm.embed_tokens(ids)
        mask = create_attention_mask(h, cache[0])
        captured = []
        for i, (layer, c) in enumerate(zip(mm.layers, cache)):
            h = layer(h, mask, c)
            if i in tapset:
                captured.append(h)
        hn = mm.norm(h)
        if self._tied:
            logits = mm.embed_tokens.as_linear(hn)
        else:
            logits = self._inner.lm_head(hn)
        return logits, mx.concatenate(captured, axis=-1)

    def _run_hybrid(self, ids, cache, tap):
        """Replicates mlx-lm's hybrid text forward (qwen3_5-style: per-layer fa/ssm mask
        selection) with the residual-stream capture. Faithfulness is proven by
        :meth:`verify_tap` at load, exactly like the dense path."""
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        mm = self._inner.model
        tapset = set(tap)
        h = mm.embed_tokens(ids)
        fa_mask = create_attention_mask(h, cache[mm.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[mm.ssm_idx])
        captured = []
        for i, (layer, c) in enumerate(zip(mm.layers, cache)):
            h = layer(h, mask=(ssm_mask if layer.is_linear else fa_mask), cache=c)
            if i in tapset:
                captured.append(h)
        hn = mm.norm(h)
        if self._tied:
            logits = mm.embed_tokens.as_linear(hn)
        else:
            logits = self._inner.lm_head(hn)
        return logits, mx.concatenate(captured, axis=-1)

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
    # touches and cannot be trimmed back, so a verify forward that turns out partially
    # rejected would poison it. Each spec round therefore runs as ONE forward over
    # ``[replay-backlog] + [anchor] + [drafts]`` with the linear caches' state arrays
    # snapshotted by *reference* just before it (MLX arrays are immutable — ops replace,
    # never mutate — so the snapshot copies nothing):
    #
    #   full accept  → keep every cache effect, backlog cleared (the common case at high
    #                  acceptance: the round costs exactly one dense-verify forward).
    #   partial      → restore the linear refs, trim the KV layers by the whole forward
    #                  width, and carry [old backlog + anchor + accepted drafts] as the
    #                  next round's backlog — they re-commit inside the next forward.
    #
    # An earlier two-phase variant (commit forward + speculative forward per round) was
    # measured 0.66× baseline on Bonsai-27B: two forwards per round = two full weight
    # reads of a memory-bound model. The single-forward design pays the replay only on
    # the rounds that actually rejected something. A backlog longer than
    # ``_REPLAY_COMMIT_MAX`` (repeated 0-accepts) is committed by a dedicated forward
    # first, so the round width stays bounded.
    #
    # The returned rows are [anchor, draft...] — the same rows a dense verify returns —
    # so the generate loops use verify()/rollback() identically for every family.

    _REPLAY_COMMIT_MAX = 8

    def reset_spec(self) -> None:
        """Clear per-generation spec state (hybrid replay/snapshot). Loops call this at
        start so a long-lived served Target never carries state across generations."""
        self._replay = None      # mx.array [1, c] committed-but-uncached token backlog
        self._snap = None
        self._verify_ids = None
        self._spec_width = 0

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
        full = ids
        if self._replay is not None:
            if self._replay.shape[1] > self._REPLAY_COMMIT_MAX:
                fwd(self._replay)          # rare: commit an oversized backlog outright
            else:
                full = mx.concatenate([self._replay.astype(ids.dtype), ids], axis=1)
            self._replay = None
        want = int(ids.shape[1])           # the 1+m rows the caller gets back
        if want == 1:
            # no drafts (lookup miss / plain step): everything is committed, no snapshot
            logits, fused = fwd(full)
            self._snap, self._verify_ids = None, None
            return logits[:, -1:], (None if fused is None else fused[:, -1:])
        self._snap = [
            None if self._is_trimmable(c) else list(c.cache) for c in cache]
        self._verify_ids = full
        self._spec_width = int(full.shape[1])
        logits, fused = fwd(full)
        return logits[:, -want:], (None if fused is None else fused[:, -want:])

    def rollback(self, cache, n_rejected: int, accepted: list[int]) -> None:
        """Undo the rejected tail of the last verify. ``accepted`` = the accepted draft
        tokens (unused — kept for call-site symmetry; hybrid replay slices the verify ids
        lazily so the on-GPU greedy path never adds a device sync)."""
        if not self.is_hybrid:
            if n_rejected > 0:
                for c in cache:
                    if c is not None and hasattr(c, "trim"):
                        c.trim(n_rejected)
            return
        if n_rejected > 0:
            if self._snap is None:
                raise RuntimeError(
                    "rollback() with rejected tokens but no preceding verify() snapshot "
                    "on a hybrid target")
            for c, snap in zip(cache, self._snap):
                if snap is not None:
                    c.cache = snap
                elif hasattr(c, "trim"):
                    c.trim(self._spec_width)
            # next round re-commits [backlog + anchor + accepted drafts] (all committed);
            # the rejected tail and the bonus token (the next anchor) are excluded.
            self._replay = self._verify_ids[:, : self._spec_width - n_rejected]
        self._snap = None
        self._verify_ids = None

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
        load. VLM targets use the native capture hook — nothing replicated, nothing to
        verify."""
        if self.is_vlm:
            return
        args = getattr(self.model, "args", None)
        mt = getattr(args, "model_type", "?")
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
        runner = self._run_hybrid if self.is_hybrid else self._run_mlxlm
        try:
            ref = self.plain(ids, self.make_cache())
            got, _ = runner(ids, self.make_cache(), [0])
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
