"""OpenAI-compatible HTTP server for mlx-dspark — serve a DSpark / DFlash / baseline
model over `/v1/chat/completions` so any OpenAI client (LM Studio, the `openai` SDK,
`curl`, LangChain, …) can talk to it locally.

Design choices (deliberate):
  * **Stdlib only.** Built on ``http.server`` (like mlx-lm's own server) so installing
    mlx-dspark stays lean — no FastAPI/uvicorn/pydantic pulled in.
  * **One model, loaded once.** The target + drafter are heavy (~8–15 GB) and load at
    startup; the ``model`` field in a request is echoed back but the loaded pair is always
    used. ``GET /v1/models`` advertises what's loaded.
  * **Serialized generation.** MLX is a single device context and every request builds its
    own KV cache, so generations can't safely interleave — an ``Engine`` lock runs them one
    at a time (correct for a single-user local server; extra requests queue).
  * **Lossless, and it shows.** Whatever the mode, output equals normal decoding of the
    target; the speculative speedup surfaces in a non-standard ``x_mlx_dspark`` block
    (accept length, tok/s) and at ``GET /metrics``.

Endpoints: ``POST /v1/chat/completions`` (stream + non-stream), ``POST /v1/completions``,
``GET /v1/models``, ``GET /health``, ``GET /metrics``.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import queue as _queue
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from . import anthropic_api as A
from .generate import (
    GenResult,
    StopStreaming,
    dflash_generate,
    encode_messages,
    greedy_generate,
    speculative_generate,
)
from .load import apply_wired_limit, load_dflash, load_drafter, load_target, resolve_mode
from .lookup import lookup_generate
from .prefix_cache import PrefixCache, _lcp, target_cache_reusable
from .telemetry import RoundLog, RoundRecorder
from .tools import normalize_tool_messages, parse_tool_calls, schema_types

MODES = ("dspark", "dflash", "lookup", "baseline")

# Seconds between stream keep-alive frames (SSE comments on the OpenAI dialect, `ping`
# events on the Anthropic one). They serve two jobs: keeping idle-timeout clients/proxies
# from aborting through stretches with nothing on the wire (long prefill; the buffered
# tool-calls path emits nothing until generation finishes), and — because a failed
# keep-alive write is the only way to notice a vanished client while nothing streams —
# detecting disconnects so generation stops at the next round instead of grinding to
# max_tokens on the single MLX thread with nobody listening. Issue #14: abandoned
# generations piling up behind that thread is exactly what looked like a wedged server.
# Env-overridable for aggressive proxies (and for exercising the path without a 15 s wait).
STREAM_KEEPALIVE_S = float(os.environ.get("MLX_DSPARK_STREAM_KEEPALIVE_S", "") or 15.0)
# Log any inter-round gap longer than this (stderr, with mode/cap/ctx) — the remote
# diagnostic for "generation stalled mid-stream" reports (issue #19). 0 disables.
SLOW_ROUND_LOG_S = float(os.environ.get("MLX_DSPARK_SLOW_ROUND_LOG_S", "") or 10.0)

# The union of effort vocabularies across template lineages (Qwen3.8 knows low/medium/xhigh,
# harmony-style templates low/medium/high). Values outside a given template's own vocabulary
# still fail there — its raise_exception carries the model-specific list — but this boundary
# check turns typos into a clear 400 before any template runs.
REASONING_EFFORTS = ("low", "medium", "high", "xhigh")


def _reasoning_effort(value) -> str:
    """Normalize a reasoning-effort value, raising ValueError for anything unknown."""
    if not isinstance(value, str) or value.lower() not in REASONING_EFFORTS:
        raise ValueError(
            f"reasoning_effort must be one of {', '.join(REASONING_EFFORTS)}, got {value!r}")
    return value.lower()


def _target_config(target_repo: str) -> dict | None:
    """The target's ``config.json`` as a dict, or ``None`` when it can't be read."""
    try:
        from .load import _resolve

        with open(os.path.join(_resolve(target_repo), "config.json")) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001 — no file / no net / bad json -> caller skips its check
        return None


def _context_window(target_repo: str) -> int | None:
    """The target's trained context length, from its ``config.json``.

    Used to reject an over-long prompt with the message Claude Code recognises as a context
    limit (see :func:`~mlx_dspark.anthropic_api.context_overflow_error`) instead of letting it
    run off the end of the model's positions. Multimodal repos nest the text config, so check
    both. ``None`` when it can't be determined — the check is then skipped.
    """
    cfg = _target_config(target_repo)
    if cfg is None:
        return None
    for c in (cfg, cfg.get("text_config") or {}):
        n = c.get("max_position_embeddings")
        if isinstance(n, int) and n > 0:
            return n
    return None


def _kv_bytes_per_token(cfg: dict, kv_bits: int | None = None) -> int | None:
    """Estimated bytes of *context-growing* target KV cache per token, from ``config.json``.

    ``attn_layers × kv_heads × head_dim × 2 (K+V) × 2 (16-bit)`` — only the full-attention
    layers count, because in a hybrid (qwen3_5 GDN, nemotron_h Mamba) the recurrent layers
    hold fixed-size state and sliding-window layers hold at most their window. Validated
    against the measured number: Qwen3.8-27B → 16 × 4 × 256 × 4 = 64 KB/token exactly (the
    README's long-context column; its 0.086 GB/1k also counts drafter ctx, so this is a
    slight *under*-estimate by design). ``0`` means the KV footprint is bounded (all layers
    sliding/recurrent); ``None`` means the config doesn't say — both skip the RAM check.
    """
    for c in (cfg.get("text_config") or {}, cfg):
        layers = c.get("num_hidden_layers")
        heads = c.get("num_attention_heads")
        if not (isinstance(layers, int) and layers > 0
                and isinstance(heads, int) and heads > 0):
            continue
        kv_heads = c.get("num_key_value_heads") or heads
        head_dim = c.get("head_dim") or (c.get("hidden_size") or 0) // heads
        if not head_dim:
            continue
        attn = layers
        # Hybrid layouts: prefer the explicit per-layer list, fall back to the interval.
        types = c.get("layer_types") or c.get("layers_block_type")
        pattern = c.get("hybrid_override_pattern")          # nemotron_h: "M*M-…", '*' = attn
        if isinstance(types, list) and types:
            attn = sum(1 for t in types
                       if isinstance(t, str) and t in ("full_attention", "attention"))
        elif isinstance(pattern, str) and pattern:
            attn = pattern.count("*")
        elif isinstance(c.get("full_attention_interval"), int) \
                and c["full_attention_interval"] > 0:
            attn = layers // c["full_attention_interval"]
        # Quantized KV shrinks the per-element bytes; each group of 64 adds a 16-bit
        # scale + bias (= 0.5 bits/element), so kv8 ≈ 0.53x and kv4 ≈ 0.28x of 16-bit.
        bytes_per_elem = 2.0 if not kv_bits else (kv_bits + 0.5) / 8
        return int(int(attn) * int(kv_heads) * int(head_dim) * 2 * bytes_per_elem)
    return None


def _context_ram_warning(kv_per_token: int | None, context_window: int | None,
                         resident_bytes: int, budget_bytes: int | None) -> str | None:
    """The startup warning for a context window whose KV cache cannot fit in RAM, or None.

    Pure so it's testable model-free; :meth:`Engine.load` feeds it the measured numbers.
    ``budget_bytes`` is the GPU working-set budget (``max_recommended_working_set_size`` —
    what macOS will wire before paging). Issue #14's secondary finding: the window defaults
    to the model's own maximum (262144 on Qwen3.8-27B ⇒ ~16 GB of KV on top of ~29 GB of
    weights), which silently exhausted a 64 GB Mac.
    """
    if not (kv_per_token and context_window and budget_bytes):
        return None
    # 90% of the working set, not all of it: the KV estimate covers only the target's
    # attention cache, and the rest — drafter ctx, prefix-cache snapshots (a hybrid
    # checkpoint copies whole caches), decode transients — needs real headroom. The
    # issue-#14 machine sat exactly in that band (45 GB of weights+KV on a 48 GB budget)
    # and was exhausted in practice.
    budget = int(budget_bytes * 0.9)
    need = resident_bytes + context_window * kv_per_token
    if need <= budget:
        return None
    fit = (budget - resident_bytes) // kv_per_token // 8192 * 8192
    gb = 1024 ** 3
    msg = (f"note: the context window defaults to the model's own maximum "
           f"({context_window} tokens), but at ~{kv_per_token / 1024:.0f} KB/token the KV "
           f"cache could grow to ~{context_window * kv_per_token / gb:.1f} GB on top of "
           f"~{resident_bytes / gb:.1f} GB already resident — past this machine's "
           f"~{budget_bytes / gb:.0f} GB GPU working set, long contexts will page or stall.")
    if fit >= 8192:
        msg += f" Consider --context-window {fit} (or lower) to bound it."
    else:
        msg += (" This machine cannot hold a useful context for this model alongside its "
                "weights; consider a smaller model or quant.")
    return msg


def _generation_defaults(target_repo: str) -> dict:
    """Sampling defaults from the model's ``generation_config.json`` — what the model
    authors recommend (e.g. Qwen3 ships 0.6 / top_p 0.95 / top_k 20). Applied only when a
    request omits the field, so explicit client settings always win. Without this, OpenAI
    clients that don't send ``temperature`` silently get greedy decoding."""
    try:
        from .load import _resolve

        with open(os.path.join(_resolve(target_repo), "generation_config.json")) as f:
            g = json.load(f)
    except Exception:  # noqa: BLE001 — no file / no net / bad json -> no defaults
        return {}
    out: dict = {}
    if g.get("do_sample", True) and g.get("temperature") is not None:
        out["temperature"] = float(g["temperature"])
        if g.get("top_p") is not None:
            out["top_p"] = float(g["top_p"])
        if g.get("top_k") is not None:
            out["top_k"] = int(g["top_k"])
    return out


# --------------------------------------------------------------------------- engine


class Engine:
    """Holds the loaded target/drafter and turns prompt token ids into a GenResult.

    All generation goes through :meth:`generate`, which is guarded by a lock so only one
    request decodes at a time. Cumulative throughput stats are kept for ``/metrics``.
    """

    def __init__(
        self,
        target,
        tokenizer,
        drafter,
        *,
        mode: str,
        model_id: str,
        target_repo: str,
        drafter_repo: str | None,
        max_draft_tokens: int | None,
        confidence_threshold: float = 0.0,
        template_defaults: dict | None = None,
        prefix_cache: bool = True,
        prefix_cache_dir: str | None = None,
        prefix_cache_max_ram_mb: int = 0,
        cap_controller=None,
        sampling_defaults: dict | None = None,
        default_max_tokens: int = 2048,
        max_tokens_cap: int = 32768,
        prefix_cache_slots: int = 2,
        prefix_cache_rungs: int = 8192,
        lookup_drafts: bool = True,
        lookup_long_draft: int = 32,
        wired_limit: bool = False,
        context_window: int | None = None,
        small_m: bool = False,
        executor: ThreadPoolExecutor | None = None,
        depth_capper=None,
    ):
        self.target = target
        self.tokenizer = tokenizer
        self.drafter = drafter
        self.mode = mode
        self.model_id = model_id
        self.target_repo = target_repo
        self.drafter_repo = drafter_repo
        self.max_draft_tokens = max_draft_tokens
        self.confidence_threshold = confidence_threshold
        self.cap_controller = cap_controller               # --max-draft auto (persists across requests)
        # Depth-aware refinement of a DERIVED default cap (None when the user pinned one):
        # a CapController used purely as the measured cost model, so long-prompt requests
        # shrink the verify width instead of running the chat-depth cap into the measured
        # width-x-depth KV-read term (cap 7 at 32k measured 1.05x vs cap 3's 1.48x on
        # Qwen3.8-27B-4bit — NOTES "Long-context decode").
        self._depth_capper = depth_capper
        self._last_cap = max_draft_tokens                  # effective cap of the last request
        self.sampling_defaults = dict(sampling_defaults or {})
        self.default_max_tokens = default_max_tokens
        self.max_tokens_cap = max_tokens_cap
        self.prefix_cache_slots = max(1, prefix_cache_slots)
        self.prefix_cache_rungs = max(0, prefix_cache_rungs)  # interior-snapshot spacing
        #   (tokens) for checkpoint-mode prefix caching; 0 disables rungs (see prefix_cache)
        self._boundary_probes_cache = None                 # [(gen-suffix tail ids, unstable)]
        self.lookup_drafts = lookup_drafts                 # hybrid n-gram drafts in dspark mode
        self.lookup_long_draft = lookup_long_draft         # match-scaled long-draft ceiling
        self.context_window = context_window               # target's trained positions, if known
        self.small_m = small_m                             # small-M MMA verify kernel active
        #   (i.e. the per-shape probe admitted ≥1 shape AND it wasn't forced off) — reported
        #   in /health so a client can see the knob, and so issue-#14-style A/Bs are possible
        if wired_limit:                                    # opt-in: see apply_wired_limit
            apply_wired_limit()
        # chat-template kwargs applied to every request unless the request overrides them
        # (e.g. {"enable_thinking": False} to silence Qwen3's <think> blocks by default).
        self.template_defaults = dict(template_defaults or {})
        self.prefix = self._build_prefix_cache(
            prefix_cache, prefix_cache_dir, prefix_cache_max_ram_mb)
        # All MLX work runs on ONE dedicated thread. MLX arrays/ops are thread/stream-affine
        # (mlx-vlm's gemma load even switches the loading thread's default stream to a
        # thread-local one), so models must be LOADED on the same thread that generates —
        # Engine.load() does that and hands the executor in; a single worker also keeps every
        # cache create/reuse on one thread and serializes requests.
        self._executor = executor or ThreadPoolExecutor(max_workers=1,
                                                        thread_name_prefix="mlx-gen")
        self.created = int(time.time())
        # Per-round telemetry: fed by the decode loops, read by /events and /metrics. Kept
        # even with no subscribers so /metrics can report position acceptance (d_0, d_1, ...)
        # for whatever has already run.
        self.rounds = RoundLog()
        self.stats = {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "generation_seconds": 0.0,
            "sum_accept_len": 0.0,   # accept-len weighted by tokens, for a token-weighted mean
        }

    def _build_prefix_cache(self, enabled, l2_dir, max_ram_mb):
        """Enable prefix caching for every mode, in whichever mode this target supports
        (see :mod:`~mlx_dspark.prefix_cache`): **trim** for dense (KVCache) and
        sliding-window targets — which latches to checkpoint the first time a window wraps
        — and **checkpoint** from the start for targets whose caches can't be rolled back
        at all (hybrid linear-attention: Ornith, Bonsai, Qwen3.6-27B). Those used to get no
        prefix caching whatsoever, which cost them the agent workload outright: prefill is
        ~90% of an uncached Claude Code turn.

        DFlash (incl. DFlash 2) is **checkpoint-only**: its drafter ctx caches can't roll
        back, but the drafter state is recoverable from a bounded window of projected ctx
        rows (:class:`~mlx_dspark.dflash_model.DFlashCtxWindow`) captured at the snapshot
        boundary — ``sliding_window - 1`` rows when every drafter layer is sliding
        (~21 MB on Qwen3.8-27B), the whole prompt's rows when any layer is full-attention.
        The trim-mode ``store()`` after generation is skipped for dflash (see the call
        site) — its slots come only from the boundary checkpoint."""
        if not enabled:
            return None
        try:
            checkpoint = not target_cache_reusable(self.target.make_cache())
        except Exception:  # noqa: BLE001
            return None
        make_ctx = self.drafter.make_ctx_cache if self.mode == "dspark" else None
        if self.mode == "dflash":
            from .dflash_model import DFlashCtxWindow

            cfg = self.drafter.config
            cap = None
            if cfg.sliding_window and all(
                    t == "sliding_attention" for t in cfg.layer_types):
                cap = int(cfg.sliding_window) - 1
            make_ctx = lambda: [DFlashCtxWindow(cap)]  # noqa: E731
            checkpoint = True
        return PrefixCache(self.target.make_cache, make_ctx,
                           l2_dir=l2_dir, max_ram_bytes=max(0, max_ram_mb) * 1024 * 1024,
                           slots=self.prefix_cache_slots, checkpoint=checkpoint)

    def _boundary_probes(self) -> list[tuple[list[int], int]]:
        """Per-chat-template measurement of the *stable prompt boundary* for checkpoint-mode
        prefix caching: how many trailing tokens of a rendered prompt do NOT survive into the
        next turn's re-rendered transcript. Qwen3.6-class templates prefill a ``<think>``
        opener whose tail the completed turn renders differently, so a snapshot taken at the
        full prompt boundary can never be hit by turn N+1 — it misses by exactly these 2-4
        tokens (the CLAUDE.md "lever 3 does not fire on Qwen3.6" finding, now healed at
        runtime instead of documented).

        Probed empirically from the template itself — render a tiny turn with the generation
        prompt, then re-render it completed plus a next user turn, and diff the tails — once
        per thinking-flag variant, since ``enable_thinking`` changes the suffix. Returns
        ``[(generation-suffix tail ids, unstable count)]``, longest tail first; a request is
        matched to its variant by comparing its prompt tail against the probes' tails."""
        if self._boundary_probes_cache is not None:
            return self._boundary_probes_cache
        probes: dict[tuple, int] = {}
        if getattr(self.tokenizer, "chat_template", None):
            msgs = [{"role": "user", "content": "a"}]
            # both reply shapes: some templates re-render a thinking reply differently
            replies = ["b", "T</think>\n\nA"]
            for kw in ({}, {"enable_thinking": True}, {"enable_thinking": False}):
                try:
                    kw = {**self.template_defaults, **kw}
                    p = encode_messages(self.tokenizer, msgs, **kw)
                    ng = encode_messages(self.tokenizer, msgs,
                                         add_generation_prompt=False, **kw)
                    g = len(p) - _lcp(p, ng)          # generation-prompt suffix length
                    u = 0
                    for reply in replies:
                        nxt = encode_messages(
                            self.tokenizer,
                            msgs + [{"role": "assistant", "content": reply},
                                    {"role": "user", "content": "c"}], **kw)
                        u = max(u, len(p) - _lcp(p, nxt))
                    if 0 < g <= 64 and u <= 64:       # sane template; tail is matchable
                        tail = tuple(p[len(p) - g:])
                        probes[tail] = max(probes.get(tail, 0), u)
                except Exception:  # noqa: BLE001 — a probe failure just means fallback u=1
                    continue
        self._boundary_probes_cache = sorted(
            ([list(t), u] for t, u in probes.items()), key=lambda x: -len(x[0]))
        return self._boundary_probes_cache

    def _unstable_suffix(self, prompt_ids: list[int]) -> int:
        """Trailing tokens of THIS prompt that won't survive re-rendering (>= 1 — snapshotting
        at least one token below the boundary is also what makes a byte-identical repeat a
        hit: the loop re-forwards the tail and gets the logits a boundary snapshot can't)."""
        for tail, u in self._boundary_probes():
            if len(prompt_ids) > len(tail) and prompt_ids[-len(tail):] == tail:
                return max(1, u)
        return 1

    @property
    def supports_reasoning_effort(self) -> bool:
        """True when the loaded chat template reads ``reasoning_effort`` (Qwen3.8-class).

        Reported in ``/health`` so a client can decide whether to show an effort control at
        all — sending the kwarg to a template that ignores it is harmless but misleading UI.
        """
        template = getattr(self.tokenizer, "chat_template", None)
        return bool(template) and "reasoning_effort" in str(template)

    @property
    def is_muse(self) -> bool:
        """True for muse_glimmer targets, whose "Onyx ATEM" harmony output needs the recipient-
        channel parser (analysis `to=self` -> thinking, `to=user` -> answer, `<atem:invoke>` ->
        tool call). The streaming paths use this to engage muse handling without ambiguity; the
        non-streaming split/tool-parse functions auto-detect from muse's unique markers."""
        return bool(getattr(self.target, "_muse", False))

    # --- construction ---
    @classmethod
    def load(
        cls,
        *,
        mode: str = "dspark",
        model: str | None = None,
        drafter: str | None = None,
        family: str | None = None,     # deprecated alias for `model`
        target: str | None = None,     # deprecated alias for `model`
        drafter_bits: int = 4,
        max_draft_tokens: int | str | None = None,   # int, None (mode default) or "auto"
        confidence_threshold: float = 0.0,
        enable_thinking: bool | None = None,
        reasoning_effort: str | None = None,
        prefix_cache: bool = True,
        prefix_cache_dir: str | None = None,
        prefix_cache_max_ram_mb: int = 0,
        default_max_tokens: int = 2048,
        max_tokens_cap: int = 32768,
        default_temperature: float | None = None,
        default_top_p: float | None = None,
        default_top_k: int | None = None,
        prefix_cache_slots: int = 2,
        prefix_cache_rungs: int = 8192,
        lookup_drafts: bool | None = None,       # None = this pair's registry default
        lookup_long_draft: int = 32,
        wired_limit: bool = False,
        batch_widths: list[int] | None = None,   # e.g. [2, max_batch]: calibrate (B,cap) grid
        kv_bits: int | None = None,              # quantize the target KV cache (4/8)
        context_window: int | None = None,       # override the target's own limit
        wide_gemm_min: int | None = None,        # prefill wide-GEMM: None=calibrate, 0=off, N=forced
        small_m: bool | None = None,             # small-M MMA verify kernel: None=probe-gated
        #                                          default, False=force off (serve-side A/B)
    ) -> Engine:
        if mode != "auto" and mode not in MODES:
            raise ValueError(f"mode must be one of {MODES} or 'auto', got {mode!r}")
        # "auto" picks the best available speculation for this target (dspark -> dflash ->
        # drafter-free lookup), so any model repo serves without extra flags.
        mode, target_repo, drafter_repo = resolve_mode(model, mode=mode, drafter=drafter,
                                                       family=family, target=target)
        # Hybrid lookup drafts: unset means "this pair's measured-best configuration" — the
        # registry rows whose stamped numbers were taken with lookup off (every MoE, the
        # 4-bit 27B hybrids, Muse) carry lookup_drafts: False, so serving them needs no flag
        # to reproduce the vouched-for speedup. Re-resolved on every /admin/load hot swap
        # (the server kwargs keep None), so the default follows the model, not the process.
        if lookup_drafts is None:
            from .load import lookup_drafts_default

            lookup_drafts = lookup_drafts_default(target_repo)

        # Pre-fetch hub repos cancellably (download.py) so the loaders below hit a complete
        # cache and never touch the network: the long phase of a first-time load — the one
        # worth cancelling and showing progress for — all happens here, where
        # /admin/load/cancel and /health's download progress can reach it.
        from .download import ensure_local

        ensure_local(target_repo)
        ensure_local(drafter_repo)

        # Load (and calibrate) on the SAME single thread that will generate: MLX ops/arrays
        # are thread/stream-affine, and mlx-vlm's gemma load switches the loading thread's
        # default stream — anything left lazy would then be unevaluatable from another thread.
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-gen")

        def _load_models():
            tgt, tok = load_target(target_repo, require_tap=mode in ("dspark", "dflash"),
                                   kv_bits=kv_bits)
            draft = None
            if mode == "dspark":
                draft, _ = load_drafter(drafter_repo, quantize=drafter_bits > 0,
                                        bits=max(drafter_bits, 2))
            elif mode == "dflash":
                draft, _ = load_dflash(drafter_repo, quantize=drafter_bits > 0,
                                       bits=max(drafter_bits, 2))
                draft.bind(tgt.model)
            # small-M MMA verify kernel (see small_m_qmm.py): on for shapes the cached
            # probe proves faster on this machine, `small_m=False` forces it off (the
            # serve-side A/B issue #14 asked for). MUST precede calibrate()/static_cap —
            # cap curves are measured under the same kernel dispatch generation uses.
            from .calibrate import apply_small_m

            smm_ids = apply_small_m(tgt, draft, target_repo=target_repo,
                                    drafter_repo=drafter_repo,
                                    enabled=False if small_m is False else None)
            # --max-draft auto: measure this machine+pair's cost curves once (disk-cached)
            # and let a CapController pick the cap per round. Only meaningful with a drafter.
            ctrl = None
            if max_draft_tokens == "auto" and mode in ("dspark", "dflash"):
                from .calibrate import calibrate

                ctrl = calibrate(tgt, draft, mode=mode, target_repo=target_repo,
                                 drafter_repo=drafter_repo, batch_widths=batch_widths)
            # prefill wide-GEMM path (dequantize once above a calibrated width) — process
            # -wide, measured once and cached like the cap
            from .calibrate import apply_wide_gemm

            apply_wide_gemm(tgt, draft, target_repo=target_repo, min_rows=wide_gemm_min)
            return tgt, tok, draft, ctrl, bool(smm_ids)

        tgt, tok, draft, cap_controller, small_m_active = \
            executor.submit(_load_models).result()
        user_pinned_cap = isinstance(max_draft_tokens, int)
        if max_draft_tokens == "auto":
            max_draft_tokens = None                     # controller drives, up to the full block
        # default cap: dspark derives it from this machine+model+quant's measured curves
        # (one-time ~5 s, disk-cached — a hardcoded constant is only right for one curve
        # shape, and mlx 0.32 already invalidated the old 2; see calibrate.static_cap);
        # dflash's native point is the full block; lookup drafts are free so a modest 6
        # balances hit gains vs miss-free rounds
        if max_draft_tokens is None and cap_controller is None:
            if mode == "dspark":
                from .calibrate import static_cap

                max_draft_tokens = executor.submit(
                    static_cap, tgt, draft, mode="dspark", target_repo=target_repo,
                    drafter_repo=drafter_repo).result()
            elif mode == "lookup":
                max_draft_tokens = 6
        # DERIVED caps get depth-aware per-request refinement (instant here: static_cap
        # above already measured-or-loaded the curves; this reuses the same cache entry,
        # plus a one-time depth-slope backfill). A user-pinned cap is never overridden.
        depth_capper = None
        if (not user_pinned_cap and cap_controller is None
                and mode in ("dspark", "dflash") and draft is not None):
            from .calibrate import calibrate

            def _mk_capper():
                try:
                    return calibrate(tgt, draft, mode=mode, target_repo=target_repo,
                                     drafter_repo=drafter_repo, verbose=False)
                except Exception:  # noqa: BLE001 — depth pricing is an optimization, never a gate
                    return None

            depth_capper = executor.submit(_mk_capper).result()
        model_id = target_repo.rstrip("/").split("/")[-1]
        template_defaults = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
        if reasoning_effort is not None:
            # Reasoning depth for templates that support it (Qwen3.8-class `reasoning_effort`).
            # Harmless on templates that don't know the kwarg — they simply ignore it.
            template_defaults["reasoning_effort"] = _reasoning_effort(reasoning_effort)
        # sampling defaults: model's generation_config.json, then explicit server flags on top
        # (many mlx-community conversions ship no generation_config — e.g. the Qwen3 repos —
        # so the flags are the way to serve sampled-by-default there)
        sampling_defaults = _generation_defaults(target_repo)
        for key, val in (("temperature", default_temperature), ("top_p", default_top_p),
                         ("top_k", default_top_k)):
            if val is not None:
                sampling_defaults[key] = val
        # RAM sanity check (issue #14's secondary finding): the window defaults to the
        # model's own maximum, and on a big model that KV budget can silently exhaust the
        # machine. A warning, not a changed default — behaviour stays predictable.
        window = context_window or _context_window(target_repo)
        try:
            import mlx.core as mx

            cfg = _target_config(target_repo)
            warn = _context_ram_warning(
                _kv_bytes_per_token(cfg, kv_bits) if cfg else None, window,
                mx.get_active_memory(),
                mx.device_info().get("max_recommended_working_set_size"))
            if warn:
                print(warn, file=sys.stderr, flush=True)
        except Exception:  # noqa: BLE001 — an estimate must never block a load
            pass
        return cls(tgt, tok, draft, mode=mode, model_id=model_id, target_repo=target_repo,
                   drafter_repo=drafter_repo, max_draft_tokens=max_draft_tokens,
                   confidence_threshold=confidence_threshold, template_defaults=template_defaults,
                   prefix_cache=prefix_cache, prefix_cache_dir=prefix_cache_dir,
                   prefix_cache_max_ram_mb=prefix_cache_max_ram_mb,
                   cap_controller=cap_controller,
                   sampling_defaults=sampling_defaults,
                   default_max_tokens=default_max_tokens, max_tokens_cap=max_tokens_cap,
                   prefix_cache_slots=prefix_cache_slots,
                   prefix_cache_rungs=prefix_cache_rungs, lookup_drafts=lookup_drafts,
                   lookup_long_draft=lookup_long_draft,
                   wired_limit=wired_limit,
                   context_window=window,
                   small_m=small_m_active,
                   executor=executor,
                   depth_capper=depth_capper)

    # --- generation ---
    def generate(
        self,
        prompt_ids: list[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        top_k: int = 0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logprobs: int | None = None,
        stop: list[str] | None,
        seed: int | None,
        on_text=None,
    ) -> GenResult:
        # hop onto the single generation thread (keeps all MLX/cache work same-thread)
        return self._executor.submit(
            self._generate_impl, prompt_ids, max_tokens, temperature, top_p, top_k,
            stop, seed, on_text, presence_penalty, frequency_penalty, logprobs).result()

    def _with_slow_round_log(self, inner, n_prompt: int):
        """Wrap a per-round callback with a stall detector: any gap over
        ``SLOW_ROUND_LOG_S`` between rounds is logged with the live shape (mode/cap/ctx).
        Exists for remote diagnosis of reports like issue #19's M5 ~100 s mid-stream
        stall — it answers "was the stall inside the generation loop, and at what depth/
        cap" without a profiler. The first round is exempt (its gap is the prefill)."""
        state = {"t": None, "n": 0}

        def wrapped(**kw):
            now = time.time()
            prev, state["t"] = state["t"], now
            state["n"] += kw.get("committed") or 0
            if SLOW_ROUND_LOG_S and prev is not None and now - prev > SLOW_ROUND_LOG_S:
                print(f"[serve] slow round: {now - prev:.1f}s between rounds "
                      f"(mode={self.mode} cap={kw.get('cap')} "
                      f"ctx~{n_prompt + state['n']})", file=sys.stderr, flush=True)
            if inner is not None:
                inner(**kw)

        return wrapped

    def _generate_impl(self, prompt_ids, max_tokens, temperature, top_p, top_k, stop, seed,
                       on_text, presence_penalty=0.0, frequency_penalty=0.0,
                       logprobs=None) -> GenResult:
        on_round = self._with_slow_round_log(
            RoundRecorder(self.rounds, uuid.uuid4().hex[:8], self.mode), len(prompt_ids))
        # prefix caching: reuse the shared conversation prefix's KV (all modes; dflash is
        # checkpoint-only); `cache is None` means prefix caching is disabled.
        cache = ctx = None
        reuse_len = 0
        on_prefill = None
        prefill_marks = None
        if self.prefix is not None:
            cache, ctx, reuse_len = self.prefix.acquire(prompt_ids)
            if self.prefix.wants_checkpoint():
                # Snapshot at the STABLE prompt boundary (prompt minus the template's
                # re-render-unstable tail — see _boundary_probes; also >= 1 below the
                # boundary so an identical repeat hits), plus interior rungs every
                # `prefix_cache_rungs` tokens and at the anchor acquire() suggested, so
                # requests that diverge mid-prompt (new session on the same system prompt,
                # compacted history) can partially reuse instead of missing outright.
                n = len(prompt_ids)
                stable = n - self._unstable_suffix(prompt_ids)
                marks = set()
                if self.prefix_cache_rungs:
                    marks.update(range(self.prefix_cache_rungs, stable,
                                       self.prefix_cache_rungs))
                anchor = self.prefix.take_anchor()
                if anchor:
                    marks.add(anchor)
                marks = {m for m in marks if reuse_len < m < stable}
                if stable > reuse_len and stable >= self.prefix.min_reuse:
                    marks.add(stable)
                prefill_marks = sorted(marks)

                def on_prefill(c, x, pos, _ids=prompt_ids, _stable=stable):
                    if pos == _stable:
                        self.prefix.checkpoint(c, x, pos, _ids)
                    elif pos < _stable:
                        self.prefix.rung(c, pos)
        # Depth-aware cap for DERIVED defaults: a long prompt shrinks the verify width,
        # because verify cost carries a measured width-x-depth KV-read term the flat
        # chat-depth curves can't see (cap 7 at 32k measured 1.05x vs cap 3's 1.48x;
        # below ~4k ctx the refinement is a no-op by construction). A user-pinned cap
        # (no _depth_capper) is never touched; --max-draft auto prices depth inside the
        # controller instead (set_depth from the loops).
        eff_cap = self.max_draft_tokens
        if self._depth_capper is not None and self.mode in ("dspark", "dflash"):
            from .calibrate import STATIC_PRIOR_P
            default = (self.max_draft_tokens if self.max_draft_tokens is not None
                       else self._depth_capper.max_cap)          # dflash: full block
            eff_cap = self._depth_capper.depth_adjusted_cap(
                len(prompt_ids), default, STATIC_PRIOR_P)
        self._last_cap = eff_cap
        try:
            if self.mode == "dspark":
                res = speculative_generate(
                    self.target, self.tokenizer, self.drafter, prompt_ids=prompt_ids,
                    cache=cache, ctx_caches=ctx, reuse_len=reuse_len,
                    max_new_tokens=max_tokens, max_draft_tokens=eff_cap,
                    cap_controller=self.cap_controller, lookup_drafts=self.lookup_drafts,
                    lookup_long_draft=self.lookup_long_draft,
                    confidence_threshold=self.confidence_threshold,
                    temperature=temperature, top_p=top_p, top_k=top_k,
                    presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
                    logprobs=logprobs, seed=seed, stop=stop, on_text=on_text,
                    on_round=on_round, on_prefill=on_prefill, prefill_marks=prefill_marks,
                )
            elif self.mode == "dflash":
                res = dflash_generate(
                    self.target, self.tokenizer, self.drafter, prompt_ids=prompt_ids,
                    cache=cache, ctx_caches=ctx, reuse_len=reuse_len,
                    max_new_tokens=max_tokens, max_draft_tokens=eff_cap,
                    cap_controller=self.cap_controller,
                    temperature=temperature, top_p=top_p, top_k=top_k,
                    seed=seed, stop=stop, on_text=on_text, on_round=on_round,
                    on_prefill=on_prefill, prefill_marks=prefill_marks,
                )
            elif self.mode == "lookup":
                res = lookup_generate(
                    self.target, self.tokenizer, prompt_ids=prompt_ids,
                    cache=cache, reuse_len=reuse_len,
                    max_new_tokens=max_tokens, max_draft_tokens=self.max_draft_tokens or 6,
                    long_draft_tokens=max(self.max_draft_tokens or 6, self.lookup_long_draft),
                    temperature=temperature, top_p=top_p, top_k=top_k,
                    seed=seed, stop=stop, on_text=on_text, on_round=on_round,
                    on_prefill=on_prefill, prefill_marks=prefill_marks,
                )
            else:
                res = greedy_generate(
                    self.target, self.tokenizer, prompt_ids=prompt_ids,
                    cache=cache, reuse_len=reuse_len,
                    max_new_tokens=max_tokens, temperature=temperature, top_p=top_p,
                    top_k=top_k, presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty, logprobs=logprobs, seed=seed,
                    stop=stop, on_text=on_text, on_round=on_round,
                    on_prefill=on_prefill, prefill_marks=prefill_marks,
                )
        except BaseException:                     # never leave a desynced cache behind
            if self.prefix is not None:
                self.prefix.reset()
            raise
        if self.prefix is not None and cache is not None and self.mode != "dflash":
            # dflash slots come only from the boundary checkpoint during prefill: its
            # drafter ctx window can't serve trim-mode reuse, so a post-generation trim
            # slot would be a broken promise on a dense target (see _build_prefix_cache).
            self.prefix.store(cache, ctx, prompt_ids, res.token_ids)
        self.stats["requests"] += 1
        self.stats["prompt_tokens"] += len(prompt_ids)
        self.stats["completion_tokens"] += res.num_tokens
        self.stats["generation_seconds"] += res.seconds
        self.stats["sum_accept_len"] += res.mean_accept_len * res.num_tokens
        return res

    def spec_info(self, res: GenResult) -> dict:
        """The non-standard block we attach so the spec-decode benefit is visible."""
        info = {
            "mode": self.mode,
            "accept_len": round(res.mean_accept_len, 3),
            "tokens_per_sec": round(res.tokens_per_sec, 1),
            "target_forwards": res.target_forwards,
        }
        if self.cap_controller is not None:
            info["cap"] = self.cap_controller.cap
        elif self._last_cap is not None:
            info["cap"] = self._last_cap             # incl. the curve-derived default,
            #                                          after any depth-aware refinement
        if res.lookup_rounds:
            info["lookup_rounds"] = res.lookup_rounds
        return info

    def metrics(self) -> dict:
        s = self.stats
        ct = s["completion_tokens"]
        return {
            "model": self.model_id,
            "mode": self.mode,
            "requests": s["requests"],
            "prompt_tokens": s["prompt_tokens"],
            "completion_tokens": ct,
            "mean_accept_len": round(s["sum_accept_len"] / ct, 3) if ct else 0.0,
            "mean_tokens_per_sec": round(ct / s["generation_seconds"], 1)
            if s["generation_seconds"] else 0.0,
            "prefix_cache": self.prefix.info() if self.prefix is not None else {"enabled": False},
            "auto_cap": self.cap_controller.info() if self.cap_controller is not None else None,
            # per-round aggregates, incl. position acceptance (d_0, d_1, …) — the drafter
            # quality curve the speedup rests on
            "rounds": self.rounds.stats(),
        }

    def close(self) -> None:
        """Release the models and stop the generation thread. Used by hot model-swapping
        (``/admin/load``) to free GPU memory before the replacement loads.

        Shuts the executor down *waiting* for any in-flight generation to finish — a swap must
        never yank the model out from under a running request — then drops references and asks
        MLX to release the freed Metal buffers so the next model has room. Idempotent.
        """
        import mlx.core as mx

        self._executor.shutdown(wait=True)
        if self.prefix is not None:
            self.prefix.reset()
            self.prefix = None
        self.target = None
        self.drafter = None
        self.cap_controller = None
        with contextlib.suppress(Exception):  # best-effort; a failed cache clear is not fatal
            mx.clear_cache()                 # return the just-freed buffers to the OS

    def race_arms_available(self) -> list[str]:
        """Which decode strategies can be raced with what is currently loaded.

        ``baseline`` and ``lookup`` need only the target; the drafter modes need the drafter
        this engine was loaded with, so dspark and dflash are never both available at once.
        """
        arms = ["baseline", "lookup"]
        if self.mode in ("dspark", "dflash") and self.drafter is not None:
            arms.insert(0, self.mode)
        return arms

    def race(self, prompt_ids: list[int], arms: list[dict], max_tokens: int, on_event) -> None:
        """Run the same prompt through several decode strategies and compare.

        **Sequential by necessity, not choice.** MLX arrays are thread- and stream-affine, so
        every arm shares the one generation thread — there is no way to run two decoders at
        once in-process. The arms are timed independently and the client replays them together;
        the numbers are real either way.

        Fairness rules, both load-bearing:
          - each arm builds a **fresh cache** (the prefix cache is bypassed entirely), or the
            second arm would start warm and look faster for no reason;
          - no arm requests logprobs, because asking for them drops ``greedy_generate`` onto its
            sequential path and would make the baseline artificially slow — which is exactly the
            direction that would flatter this project.
        """
        results: list[dict] = []
        for index, arm in enumerate(arms):
            mode = arm.get("mode", "baseline")
            cap = arm.get("cap")
            conf = arm.get("confidence")
            label = mode if cap is None else f"{mode} cap {cap}"
            if conf is not None:
                label += f" conf {conf:g}"
            on_event("arm_start", {"index": index, "mode": mode, "cap": cap,
                                   "confidence": conf, "label": label})

            started = time.time()

            def emit(piece: str, _index=index, _t0=started):
                on_event("token", {"index": _index, "text": piece,
                                   "t_ms": round((time.time() - _t0) * 1e3, 1)})

            res = self._executor.submit(self._race_arm, prompt_ids, mode, cap, conf,
                                        max_tokens, emit).result()
            summary = {
                "index": index, "label": label, "mode": mode, "cap": cap,
                "tokens": res.num_tokens,
                "seconds": round(res.seconds, 3),
                "tokens_per_sec": round(res.tokens_per_sec, 1),
                "accept_len": round(res.mean_accept_len, 3),
                "target_forwards": res.target_forwards,
                "lookup_rounds": res.lookup_rounds,
            }
            results.append({**summary, "token_ids": res.token_ids, "text": res.text})
            on_event("arm_done", summary)

        verdict = self._race_verdict(results)
        # Measure the logit margin at each real divergence instead of asserting it is a tie.
        # This is the difference between "trust us, it's lossless" and showing the evidence.
        for divergence in verdict.get("divergences", []):
            if divergence.get("length_only"):
                continue
            reference_ids = results[0]["token_ids"]
            margin = self._executor.submit(
                self._logit_margin, prompt_ids, reference_ids,
                divergence["first_diff"], divergence["reference_token"],
                divergence["arm_token"]).result()
            divergence.update(margin)
        verdict["detail"] = self._verdict_detail(verdict)
        on_event("verdict", verdict)

    def _logit_margin(self, prompt_ids, token_ids, position, token_a, token_b) -> dict:
        """How close the two competing tokens were where the arms disagreed — *approximately*.

        Read the caveat before trusting this number. It re-runs the shared prefix on a clean
        cache, which is **not** the cache either arm actually had. A speculative arm builds its
        KV cache through multi-row verify forwards; a baseline arm builds it one row at a time.
        Those take different matmul kernels, so individual KV entries differ in their last bits,
        and over dozens of positions the difference is enough to flip a close call. That is
        precisely the mechanism behind the divergence being measured, so the reconstruction
        cannot reproduce either arm's true logits.

        Probed directly (Qwen3-4B, 2026-07-22): at a real divergence, plain single-step and
        verify at widths 2 and 3 *all* ranked the same token first, by an identical 0.75 — yet
        the arms still diverged there, because their caches differed upstream. So a "large"
        margin here is evidence of accumulated cache drift, NOT of a broken accept rule.

        Reported as context for a human, never as a pass/fail.
        """
        import mlx.core as mx

        try:
            ids = list(prompt_ids) + list(token_ids[:position])
            cache = self.target.make_cache()
            # Condition-matched on purpose: prefill everything *except* the last token, then
            # take a single-token step. Reading the logits straight off a full prefill would
            # measure a different kernel path than the one that actually produced this
            # position during decoding, and the gap being measured is itself a kernel-path
            # artifact — so the confound would be the same size as the signal.
            self.target.plain(mx.array([ids[:-1]]), cache)
            logits = self.target.plain(mx.array([[ids[-1]]]), cache)[0, -1]
            values = logits.astype(mx.float32)
            top = mx.argpartition(-values, 2)[:2]
            pair = [float(values[int(i)].item()) for i in top]
            margin = abs(pair[0] - pair[1])
            scores = {}
            for name, token in (("reference", token_a), ("arm", token_b)):
                if token is not None:
                    scores[name] = round(float(values[int(token)].item()), 5)
            gap = (abs(scores["reference"] - scores["arm"])
                   if len(scores) == 2 else margin)
            return {
                "margin": round(gap, 5),
                "top2_margin": round(margin, 5),
                "logits": scores,
                "approximate": True,     # clean-cache reconstruction; see the docstring
            }
        except Exception as e:  # noqa: BLE001 — evidence is a bonus; never fail the race for it
            return {"margin": None, "is_tie": None, "margin_error": str(e)}

    @staticmethod
    def _verdict_detail(verdict: dict) -> str:
        if not verdict.get("comparable"):
            return verdict.get("reason", "")
        real = [d for d in verdict["divergences"] if not d.get("length_only")]
        if verdict.get("identical") and not verdict["divergences"]:
            return "Every arm produced the same tokens."
        if not real:
            return ("Every arm produced the same tokens where they overlap; they only differ "
                    "in how many tokens they emitted before hitting the limit.")
        first = min(d["first_diff"] for d in real)
        # Deliberately not phrased as a fault. Every arm commits only tokens its own target
        # forward ranked first, so neither output is "wrong": they took slightly different
        # arithmetic paths to the same distribution and parted at a close call.
        return (f"Arms match for the first {first} tokens, then take different but equally "
                f"valid continuations. Speculative arms build their KV cache with multi-row "
                f"forwards and the baseline builds it one row at a time; the last-bit "
                f"differences accumulate and eventually flip a near-tie. Every token any arm "
                f"emitted was its target's own top choice.")

    def _race_arm(self, prompt_ids, mode, cap, confidence, max_tokens, on_text) -> GenResult:
        """One arm, on the generation thread, with a cache of its own. ``confidence`` is a
        per-arm confidence-head threshold (None = the server's loaded setting) — what lets a
        measured bundle like Qwen3.8-27B-4bit's cap 7 + 0.3 race its plain-cap siblings."""
        ctrl = None
        if cap == "auto" and mode in ("dspark", "dflash"):
            # A FRESH controller per arm: the curves come from the disk cache (measured at
            # load time for static_cap, so this is instant), but the acceptance EWMA is
            # runtime state — sharing one across arms/races would let an earlier run bias
            # a later one's caps, which is exactly what a race must not do.
            from .calibrate import calibrate

            ctrl = calibrate(self.target, self.drafter, mode=mode,
                             target_repo=self.target_repo, drafter_repo=self.drafter_repo,
                             verbose=False)
            cap = None
        if mode == "dspark":
            return speculative_generate(
                self.target, self.tokenizer, self.drafter, prompt_ids=prompt_ids,
                max_new_tokens=max_tokens,
                max_draft_tokens=(None if ctrl is not None else (cap or 2)),
                cap_controller=ctrl,
                lookup_drafts=self.lookup_drafts, lookup_long_draft=self.lookup_long_draft,
                confidence_threshold=(self.confidence_threshold if confidence is None
                                      else confidence),
                on_text=on_text)
        if mode == "dflash":
            return dflash_generate(
                self.target, self.tokenizer, self.drafter, prompt_ids=prompt_ids,
                max_new_tokens=max_tokens, max_draft_tokens=cap, cap_controller=ctrl,
                on_text=on_text)
        if mode == "lookup":
            return lookup_generate(
                self.target, self.tokenizer, prompt_ids=prompt_ids,
                max_new_tokens=max_tokens, max_draft_tokens=cap or 6,
                long_draft_tokens=max(cap or 6, self.lookup_long_draft), on_text=on_text)
        return greedy_generate(self.target, self.tokenizer, prompt_ids=prompt_ids,
                               max_new_tokens=max_tokens, on_text=on_text)

    @staticmethod
    def _race_verdict(results: list[dict]) -> dict:
        """Did every arm produce the same tokens?

        This is the claim the whole project rests on — speculation is a speed change, not a
        behaviour change — so the app states it as a checked result rather than a promise.

        A divergence is reported, not hidden, but it is almost always a floating-point tie:
        the target verifies every token, so the only freedom is which of two (near-)equally
        scored tokens argmax returns, and that can differ between a width-1 and a width-N
        forward. Compare against the FIRST arm so the reference is stable.
        """
        if len(results) < 2:
            return {"comparable": False, "reason": "need at least two arms to compare"}

        reference = results[0]
        divergences = []
        for other in results[1:]:
            a, b = reference["token_ids"], other["token_ids"]
            first_diff = -1
            for i in range(min(len(a), len(b))):
                if a[i] != b[i]:
                    first_diff = i
                    break
            if first_diff == -1 and len(a) != len(b):
                # One arm simply stopped earlier (hit max_tokens mid-block); the shared
                # prefix still matches, which is what "lossless" claims.
                first_diff = min(len(a), len(b))
                truncated = True
            else:
                truncated = False
            if first_diff != -1:
                divergences.append({
                    "arm": other["index"], "label": other["label"],
                    "first_diff": first_diff, "length_only": truncated,
                    "reference_token": a[first_diff] if first_diff < len(a) else None,
                    "arm_token": b[first_diff] if first_diff < len(b) else None,
                })

        # A length difference is not a losslessness failure: the arms agree everywhere they
        # overlap, one simply stopped earlier (a spec arm commits a whole block, so it can
        # overshoot the token limit). Counting it as a divergence would cry wolf on almost
        # every race and make the real signal worthless.
        real = [d for d in divergences if not d["length_only"]]
        return {
            "comparable": True,
            "identical": not real,
            "identical_where_overlapping": not real,
            "reference": reference["label"],
            "divergences": divergences,
            "detail": "",                       # filled in by _verdict_detail once measured
        }

    def calibration(self) -> dict:
        """The measured cost curves for this machine+model pair.

        These are already computed and cached on disk by ``--max-draft auto``; until now they
        only ever appeared as one line of terminal output. Nothing here measures anything —
        it reads the cache and reports what is there, so it is safe to call at any time.
        """
        from . import generate as _gen
        from .calibrate import cached_curve_entry, drafter_recommendation

        if self.mode not in ("dspark", "dflash"):
            return {"available": False,
                    "reason": f"calibration applies to dspark/dflash, not {self.mode!r}"}
        # The curves are cached under a "|smm"-tagged key when the small-M kernel was live
        # during calibration (the default since v0.12.0) — reading only the untagged key
        # showed "not calibrated" on every calibrated kernel-on machine.
        key, entry = cached_curve_entry(
            self.mode, self.target_repo, self.drafter_repo,
            kv_bits=getattr(self.target, "kv_bits", None),
            smm_live=bool(_gen.SMALL_M_IDS))
        if entry is None:
            return {"available": False, "key": key,
                    "reason": "not calibrated yet on this machine — loading the pair without "
                              "a fixed cap (or --max-draft auto) measures the curves "
                              "automatically"}

        verify = {int(k): float(v) for k, v in entry["verify"].items()}
        drafter_ms = entry["drafter"]
        out = {
            "available": True,
            "key": key,
            "mode": self.mode,
            "target": self.target_repo,
            "drafter": self.drafter_repo,
            # ms per verify forward at each width — the convex curve whose knee explains
            # why cap 2 wins on M-series, and whose 16–32 plateau is why long lookup
            # drafts pay off.
            "verify_ms": {str(k): round(v, 3) for k, v in sorted(verify.items())},
            "drafter_ms": ({str(k): round(float(v), 3) for k, v in sorted(
                ((int(k2), v2) for k2, v2 in drafter_ms.items()))}
                if isinstance(drafter_ms, dict) else round(float(drafter_ms), 3)),
            "round_overhead_ms": round(float(entry.get("overhead", 0.0)), 3),
            "recommendation": drafter_recommendation(verify),
        }
        if entry.get("verify_grid"):
            out["verify_grid"] = entry["verify_grid"]
        if self.cap_controller is not None:
            out["controller"] = self.cap_controller.info()
        return out


# --------------------------------------------------------------------------- batching engine


_STOP = object()   # sentinel: unwedges the scheduler thread so the process can exit


class _Job:
    """One queued generation request awaiting a (possibly batched) run."""
    __slots__ = ("done", "error", "on_text", "params", "prompt_ids", "result")

    def __init__(self, prompt_ids, params, on_text):
        self.prompt_ids = prompt_ids
        self.params = params
        self.on_text = on_text
        self.result = None
        self.error = None
        self.done = threading.Event()


class BatchEngine:
    """Batching wrapper around an :class:`Engine`. Concurrently-queued greedy **dspark**
    requests run as a **continuous** :class:`~mlx_dspark.batch_engine.SpecSlots` session
    (dynamic admission): each request's result is delivered the moment its row finishes, and
    the freed slot admits the next queued/arriving request mid-flight — a short request never
    waits for a long one. Baseline mode uses the static batched kernel. Dense mlx-lm targets
    only; anything else, a lone request, or a temp>0 dspark request runs the serialized Engine
    path unchanged, so B=1 latency never regresses. Prefix caching and the auto-cap controller
    apply to the serial path only (batched rows use the fixed cap and skip prefix reuse —
    documented).

    All MLX work stays on the Engine's single generation thread: a scheduler loop runs *on* that
    executor and pulls jobs off a queue; HTTP handler threads only enqueue and block for their
    result (streaming ``on_text`` callbacks fire from the scheduler thread, which is safe because
    the handler is parked in :meth:`generate` and not touching its socket)."""

    def __init__(self, engine: Engine, *, max_batch: int = 4, window_s: float = 0.02):
        self.engine = engine
        self.max_batch = max(2, int(max_batch))
        self.window_s = window_s
        self._q: _queue.Queue = _queue.Queue()
        self.batch_stats = {"batched_requests": 0, "batches": 0, "max_batch_seen": 0,
                            "serial_requests": 0}
        engine._executor.submit(self._scheduler)   # occupies the one MLX thread until close()
        # concurrent.futures' shutdown hook joins the executor thread at interpreter exit —
        # a forever-looping scheduler would wedge the process (Ctrl-C'd server, scripts,
        # tests). Regular atexit handlers run BEFORE that join, so this unblocks it.
        atexit.register(self.close)

    def close(self) -> None:
        """Stop the scheduler thread (idempotent). Queued jobs already picked up finish
        normally; the sentinel is consumed at the next idle point."""
        self._q.put(_STOP)

    def __getattr__(self, name):                    # delegate model_id/mode/spec_info/created/…
        return getattr(self.engine, name)

    # --- public API (mirrors Engine.generate) ---
    def generate(self, prompt_ids, *, max_tokens, temperature, top_p=1.0, top_k=0,
                 presence_penalty=0.0, frequency_penalty=0.0, logprobs=None, stop=None,
                 seed=None, on_text=None) -> GenResult:
        job = _Job(prompt_ids,
                   {"max_tokens": max_tokens, "temperature": temperature,
                    "top_p": top_p, "top_k": top_k, "presence_penalty": presence_penalty,
                    "frequency_penalty": frequency_penalty, "logprobs": logprobs,
                    "stop": stop or [], "seed": seed}, on_text)
        self._q.put(job)
        job.done.wait()
        if job.error is not None:
            raise job.error
        return job.result

    def metrics(self) -> dict:
        m = self.engine.metrics()
        m["batching"] = {"max_batch": self.max_batch, **self.batch_stats}
        return m

    # --- scheduler (runs on the MLX thread) ---
    def _scheduler(self):
        while True:
            batch = []
            try:
                job = self._q.get()
                if job is _STOP:
                    return
                batch = [job]
                end = time.time() + self.window_s
                while len(batch) < self.max_batch:
                    rem = end - time.time()
                    if rem <= 0:
                        break
                    try:
                        nxt = self._q.get(timeout=rem)
                    except _queue.Empty:
                        break
                    if nxt is _STOP:
                        self._q.put(nxt)     # finish this batch, exit on the next get()
                        break
                    batch.append(nxt)
                self._run(batch)
            except Exception:  # noqa: BLE001 — a scheduler that dies wedges the server
                for j in batch:
                    if not j.done.is_set():
                        j.error = RuntimeError("batch scheduler error")
                        j.done.set()

    def _run(self, batch: list[_Job]):
        # only requests with identical sampling can share a batch (one temp/top_p/top_k per run);
        # penalized requests take the serial path (the batched kernels don't apply penalties yet)
        groups: dict = {}
        for j in batch:
            p = j.params
            if p["presence_penalty"] or p["frequency_penalty"] or p["logprobs"] is not None:
                self._run_serial(j)           # penalties/logprobs: serial (batched kernels lack them)
                continue
            key = (p["temperature"], p["top_p"], p["top_k"])
            groups.setdefault(key, []).append(j)
        for key, jobs in groups.items():
            temp = key[0]
            if len(jobs) == 1 or (self.engine.mode == "dspark" and temp > 0):
                for j in jobs:                       # size-1, or temp>0 dspark (no batched sampler)
                    self._run_serial(j)
            elif self.engine.mode == "dspark":
                self._run_session(jobs)              # continuous: admit/retire mid-flight (M4)
            else:
                self._run_batched(jobs, key)

    # --- continuous batching (dspark greedy): slot session with dynamic admission ---
    @staticmethod
    def _batchable_greedy(job: _Job) -> bool:
        p = job.params
        return (not p["presence_penalty"] and not p["frequency_penalty"]
                and p["logprobs"] is None and not p["temperature"])

    def _admit(self, slots, job: _Job) -> bool:
        try:
            p = job.params
            slots.admit(job.prompt_ids, max_new_tokens=p["max_tokens"],
                        on_text=job.on_text, stop=p["stop"], meta=job)
            return True
        except BaseException as e:  # noqa: BLE001 — a bad request must not kill the session
            job.error = e
            job.done.set()
            return False

    def _run_session(self, jobs: list[_Job]):
        """Continuous batching (M4): run greedy dspark jobs through a :class:`SpecSlots`
        session. A finished request is delivered the instant its row retires (it does not wait
        for the batch's slowest row), and its freed slot admits the next queued/arriving
        batchable job mid-flight. Retirement compacts rows, so a lone long tail runs at serial
        verify width. A non-batchable arrival (penalties/logprobs/temp>0) is deferred to the end
        of the session and also stops further admissions, so it can't starve."""
        from .batch_engine import SpecSlots

        eng = self.engine
        slots = SpecSlots(eng.target, eng.tokenizer, eng.drafter, capacity=self.max_batch,
                          max_draft_tokens=eng.max_draft_tokens or 2,
                          cap_controller=eng.cap_controller)
        waiting = list(jobs)     # accepted into this session, not yet admitted
        deferred: list[_Job] = []
        admitted = 0
        peak = 0
        t0 = time.time()
        try:
            while slots.n_active or waiting:
                while waiting and slots.has_free_slot:
                    admitted += self._admit(slots, waiting.pop(0))
                if not deferred:                 # pull mid-flight arrivals into free capacity
                    while len(waiting) < self.max_batch:
                        try:
                            nj = self._q.get_nowait()
                        except _queue.Empty:
                            break
                        if nj is _STOP:
                            self._q.put(nj)      # session drains; scheduler exits after it
                            break
                        if self._batchable_greedy(nj):
                            waiting.append(nj)
                        else:
                            deferred.append(nj)  # fairness: stop admitting, drain, then serve
                            break
                    while waiting and slots.has_free_slot:
                        admitted += self._admit(slots, waiting.pop(0))
                peak = max(peak, slots.n_active)
                for job, res in slots.step():
                    job.result = res
                    job.done.set()
                    s = eng.stats
                    s["requests"] += 1
                    s["prompt_tokens"] += len(job.prompt_ids)
                    s["completion_tokens"] += res.num_tokens
                    s["sum_accept_len"] += res.mean_accept_len * res.num_tokens
        except BaseException as e:  # noqa: BLE001
            outstanding = waiting + [slots.meta[b] for b in range(slots.n_active)]
            for j in outstanding:
                if j is not None and not j.done.is_set():
                    j.error = e
                    j.done.set()
        eng.stats["generation_seconds"] += time.time() - t0
        self.batch_stats["batched_requests"] += admitted
        self.batch_stats["batches"] += 1
        self.batch_stats["max_batch_seen"] = max(self.batch_stats["max_batch_seen"], peak)
        for j in deferred:
            self._run_serial(j)

    def _run_serial(self, job: _Job):
        try:
            p = job.params
            job.result = self.engine._generate_impl(
                job.prompt_ids, p["max_tokens"], p["temperature"], p["top_p"], p["top_k"],
                p["stop"], p["seed"], job.on_text, p["presence_penalty"], p["frequency_penalty"],
                p["logprobs"])
            self.batch_stats["serial_requests"] += 1
        except BaseException as e:  # noqa: BLE001
            job.error = e
        finally:
            job.done.set()

    def _run_batched(self, jobs: list[_Job], key):
        # baseline only — dspark groups go through the continuous _run_session path
        from .batch_engine import batch_generate_baseline

        temp, top_p, top_k = key
        prompts = [j.prompt_ids for j in jobs]
        max_toks = [j.params["max_tokens"] for j in jobs]
        on_texts = [j.on_text for j in jobs]
        stops = [j.params["stop"] for j in jobs]
        try:
            res = batch_generate_baseline(
                self.engine.target, self.engine.tokenizer, prompts, max_new_tokens=max_toks,
                temperature=temp, top_p=top_p, top_k=top_k, on_texts=on_texts, stops=stops)
        except BaseException as e:  # noqa: BLE001
            for j in jobs:
                j.error = e
                j.done.set()
            return
        for j, r in zip(jobs, res):
            j.result = r
            j.done.set()
        # metrics: count each row's tokens, but the batch wall time once (aggregate tok/s stays honest)
        s = self.engine.stats
        s["requests"] += len(jobs)
        s["prompt_tokens"] += sum(len(j.prompt_ids) for j in jobs)
        s["completion_tokens"] += sum(r.num_tokens for r in res)
        s["generation_seconds"] += res[0].seconds
        s["sum_accept_len"] += sum(r.mean_accept_len * r.num_tokens for r in res)
        self.batch_stats["batched_requests"] += len(jobs)
        self.batch_stats["batches"] += 1
        self.batch_stats["max_batch_seen"] = max(self.batch_stats["max_batch_seen"], len(jobs))


def maybe_batch_engine(engine: Engine, max_batch: int):
    """Wrap ``engine`` in a :class:`BatchEngine` iff batching can help and is safe here: opt-in
    (``max_batch > 1``), a batchable mlx-lm target, and a mode with a batched kernel
    (dspark/baseline). Otherwise return the engine unchanged (serialized).

    The two modes have different requirements: baseline batching needs only a batched forward
    (:func:`batchable`, which covers the qwen3_5 hybrids), while dspark batching additionally
    needs per-row rollback of every layer (:func:`batch_spec_supported`). A hybrid target in
    dspark mode therefore stays serialized rather than silently taking a path its recurrent
    caches cannot roll back."""
    from .batch_engine import batch_spec_supported, batchable

    if max_batch <= 1 or engine.mode not in ("dspark", "baseline"):
        return engine
    ok = batch_spec_supported if engine.mode == "dspark" else batchable
    if not ok(engine.target):
        return engine
    return BatchEngine(engine, max_batch=max_batch)


class EngineHolder:
    """A swappable reference to the live engine, so a model can be changed **without dropping
    the server** (``POST /admin/load``).

    Everything the request handler does — ``holder.generate(...)``, ``holder.metrics()``,
    ``holder.model_id`` — is delegated to the current engine via ``__getattr__``, so the 57
    places the handler touches the engine need no changes: they just follow the swap.

    Swap policy is **release-then-load**, deliberately. Freeing the old model before loading the
    new one means peak memory is one model, not two — the difference between switching two 12 GB
    models comfortably on a 16 GB Mac and OOM-ing it. The cost is a window with no model loaded;
    the handler answers requests during that window with 503 (see ``ready``). A load that fails
    leaves no model, reported clearly and recovered by loading a valid one.

    It preserves the port, which a full process restart cannot — an external client (Claude
    Code, say) pointed at the server keeps working across a model change.
    """

    def __init__(self, engine, load_kwargs: dict, max_batch: int = 1):
        self._engine = engine
        self._load_kwargs = dict(load_kwargs)     # the flags this server was started with
        self._max_batch = max_batch
        self._swap_lock = threading.Lock()
        self._loading = False
        self._load_error: str | None = None

    def __getattr__(self, name):
        # Only reached for names not found on the holder itself. During a swap `_engine` is
        # None; the request dispatcher gates on `ready` before getting here, so a stray access
        # surfaces as a clear error rather than a confusing AttributeError on None.
        engine = self.__dict__.get("_engine")
        if engine is None:
            raise RuntimeError("no model is loaded (a model swap is in progress or failed)")
        return getattr(engine, name)

    @property
    def ready(self) -> bool:
        return self._engine is not None and not self._loading

    @property
    def current(self):
        return self._engine

    def status(self) -> dict:
        out = {"ready": self.ready, "loading": self._loading,
               "model": self._engine.model_id if self._engine is not None else None,
               "error": self._load_error}
        if self._loading:
            # Live download progress while a first-time load fetches weights — lets a
            # client draw a real progress bar and offer Cancel (POST /admin/load/cancel).
            from .download import progress

            out["download"] = progress()
        return out

    def unload(self) -> dict:
        """Release the loaded model without loading another — frees its memory.

        The server stays up: generation routes 503 (with a "no model" reason) and
        ``/admin/load`` brings a model back on the same port. Unloading twice is a no-op.
        """
        with self._swap_lock:
            old = self._engine
            self._engine = None
            self._load_error = None
            if old is not None:
                old.close()
                inner = getattr(old, "engine", None)
                if inner is not None and inner is not old and hasattr(inner, "close"):
                    inner.close()
            return self.status()

    def swap(self, *, model: str, mode: str | None = None,
             max_draft: int | str | None = None,
             lookup_drafts: bool | None = None,
             confidence_threshold: float | None = None,
             context_window: int | None = None,
             small_m: bool | None = None,
             kv_bits: int | None = None) -> dict:
        """Release the current model and load ``model`` in its place. Returns the new status.

        Serialized by ``_swap_lock`` so two concurrent loads can't race. Raises ``ValueError``
        with the reason if the new model can't be loaded — the caller turns that into a 4xx/5xx.
        """
        with self._swap_lock:
            self._loading = True
            self._load_error = None
            old = self._engine
            self._engine = None                   # `ready` is False from here until success
            try:
                if old is not None:
                    old.close()                   # frees GPU memory before the new load
                    # A BatchEngine's close() only stops its scheduler; the models live on the
                    # Engine it wraps, so close that too or the old weights never leave memory.
                    inner = getattr(old, "engine", None)
                    if inner is not None and inner is not old and hasattr(inner, "close"):
                        inner.close()

                if context_window is not None:
                    # STICKY across swaps: a context window is a machine/RAM policy, not a
                    # per-pair tuning — scripts that set it once expect it to survive a later
                    # /admin/load that omits the field (community report: an omitted override
                    # silently reverted a 32k cap to the model's 262k max). 0 resets to the
                    # model's own maximum. Per-pair knobs (mode/max_draft/lookup_drafts/
                    # confidence) stay per-swap: omitted, they re-resolve to the pair's
                    # measured defaults, which IS their reset semantics.
                    self._load_kwargs["context_window"] = context_window or None
                kwargs = dict(self._load_kwargs)
                kwargs["model"] = model
                if mode is not None:
                    kwargs["mode"] = mode
                if max_draft is not None:
                    kwargs["max_draft_tokens"] = max_draft
                if lookup_drafts is not None:
                    kwargs["lookup_drafts"] = lookup_drafts
                if confidence_threshold is not None:
                    kwargs["confidence_threshold"] = confidence_threshold
                if small_m is not None:
                    kwargs["small_m"] = small_m
                if kv_bits is not None:
                    kwargs["kv_bits"] = kv_bits or None    # 0 -> full precision
                engine = Engine.load(**kwargs)
                engine = maybe_batch_engine(engine, self._max_batch)
                self._engine = engine
            except Exception as e:
                self._load_error = str(e)
                raise
            finally:
                self._loading = False
            # After the finally, so the returned status reflects the settled state (ready=True),
            # not the mid-load snapshot.
            return self.status()


# --------------------------------------------------------------------------- request parsing


def _norm_stop(stop) -> list[str]:
    """OpenAI ``stop`` may be a string, a list, or null -> always a list[str]."""
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    return [str(s) for s in stop]


def _logprobs_content(res: GenResult, tokenizer) -> dict:
    """OpenAI chat ``logprobs.content`` from ``GenResult.logprobs`` (decode ids -> token strings
    + utf-8 bytes; include ``top_logprobs`` when the request asked for them)."""
    def s(tid):
        try:
            return tokenizer.decode([int(tid)])
        except Exception:  # noqa: BLE001
            return ""

    content = []
    for e in res.logprobs or []:
        tok = s(e["token_id"])
        item = {"token": tok, "logprob": e["logprob"], "bytes": list(tok.encode("utf-8"))}
        item["top_logprobs"] = [{"token": s(t), "logprob": lp, "bytes": list(s(t).encode("utf-8"))}
                                for t, lp in e.get("top", [])]
        content.append(item)
    return {"content": content}


def _logprobs_completions(res: GenResult, tokenizer) -> dict:
    """OpenAI /v1/completions ``logprobs`` shape (parallel arrays)."""
    def s(tid):
        try:
            return tokenizer.decode([int(tid)])
        except Exception:  # noqa: BLE001
            return ""

    toks, tlp, tops = [], [], []
    for e in res.logprobs or []:
        toks.append(s(e["token_id"]))
        tlp.append(e["logprob"])
        tops.append({s(t): lp for t, lp in e.get("top", [])})
    return {"tokens": toks, "token_logprobs": tlp, "top_logprobs": tops, "text_offset": []}


def _sampling(req: dict, defaults: dict) -> tuple[float, float, int]:
    """(temperature, top_p, top_k) for a request: explicit request value > the model's
    ``generation_config`` recommendation > library default. Shared by the OpenAI and
    Anthropic routes — it matters most on the latter, where Claude Code sends
    ``temperature`` but never ``top_p``/``top_k``, so the model's own nucleus settings are
    what keep a temperature-1 agent request sane."""
    def pick(key, fallback):
        v = req.get(key)
        return defaults.get(key, fallback) if v is None else v

    return float(pick("temperature", 0.0)), float(pick("top_p", 1.0)), int(pick("top_k", 0))


def _clamp_tokens(v, default: int = 2048, cap: int = 32768) -> int:
    """Requested max_tokens, clamped to [1, cap]; ``default`` when absent/invalid. The cap
    is configurable (``--max-tokens-cap``) — thinking models routinely exceed the old 8192."""
    try:
        n = int(v)
    except (TypeError, ValueError):
        return default
    return max(1, min(n, cap))


# --------------------------------------------------------------------------- HTTP handler


def make_handler(engine: Engine, api_key: str | None):
    """Build a request-handler class bound to this engine (needed since BaseHTTPRequestHandler
    is instantiated per-connection by the server and can't take extra constructor args)."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "mlx-dspark"

        def setup(self):
            super().setup()
            # SSE frames are written from the request thread and, during long prefills, from
            # the keep-alive heartbeat timer — one lock keeps frames from interleaving.
            self._wlock = threading.Lock()

        # -- low-level replies --
        def _route(self) -> str:
            """Path with the query string and trailing slash removed. Claude Code posts to
            ``/v1/messages?beta=true``, so routing on ``self.path`` verbatim would 404."""
            return urlsplit(self.path).path.rstrip("/") or "/"

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header(
                "Access-Control-Allow-Headers", "Authorization, x-api-key, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, HEAD, POST, OPTIONS")

        def _send_json(self, status: int, obj: dict):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _send_error(self, status: int, message: str, etype: str = "invalid_request_error"):
            self._send_json(status, {"error": {"message": message, "type": etype,
                                               "code": status}})

        def _sse_start(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self._cors()
            self.end_headers()

        def _sse(self, obj: dict, event: str | None = None):
            """One SSE frame. Anthropic's stream is a *named*-event stream (``event: <type>``
            before the data line); OpenAI's is unnamed, so ``event`` is optional."""
            head = f"event: {event}\n" if event else ""
            with self._wlock:
                self.wfile.write(f"{head}data: {json.dumps(obj)}\n\n".encode())
                self.wfile.flush()

        def _sse_comment(self, text: str):
            """A comment frame — ignored by every SSE client, but it keeps the socket alive
            through an idle stretch (no rounds running) without inventing a fake event."""
            with self._wlock:
                self.wfile.write(f": {text}\n\n".encode())
                self.wfile.flush()

        # -- auth --
        def _authed(self) -> bool:
            if not api_key:
                return True
            # Which header carries the credential depends on how the client was configured:
            # ANTHROPIC_AUTH_TOKEN -> Authorization: Bearer, ANTHROPIC_API_KEY / an
            # apiKeyHelper -> x-api-key (a helper sends both). Accept either.
            return (self.headers.get("Authorization", "") == f"Bearer {api_key}"
                    or self.headers.get("x-api-key", "") == api_key)

        def log_message(self, fmt, *args):  # quieter default logging
            return

        # -- routing --
        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self._cors()
            self.end_headers()

        def do_HEAD(self):
            # Claude Code opens with a best-effort `HEAD /` connectivity probe.
            self.send_response(200 if self._authed() else 401)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self._cors()
            self.end_headers()

        def _require_ready(self) -> bool:
            """503 while a model swap is in flight, or while no model is loaded at all
            (``--no-model`` start, ``/admin/unload``). Everything but /health, the admin
            status/inventory routes and /doctor needs a loaded model, so they gate on this."""
            if isinstance(engine, EngineHolder) and not engine.ready:
                self._send_error(
                    503,
                    "a model is loading — try again in a moment" if engine.status()["loading"]
                    else "no model is loaded — load one with POST /admin/load",
                    "service_unavailable")
                return False
            return True

        def do_GET(self):
            route = self._route()
            if not self._authed():
                return self._send_error(401, "invalid api key", "authentication_error")
            if route == "/health":
                # Answers even mid-swap so a client can poll the status through a model change.
                # "loading" and "no_model" are distinct states: a client should wait through
                # the first and offer a model picker on the second.
                if isinstance(engine, EngineHolder) and not engine.ready:
                    status = engine.status()
                    return self._send_json(200, {
                        "status": "loading" if status["loading"] else "no_model",
                        "model": status["model"], "loading": status["loading"],
                        # non-null while a first-time load is fetching weights:
                        # {repo, bytes_done, bytes_total} — cancel with /admin/load/cancel
                        "download": status.get("download"),
                        "error": status["error"]})
                # max_draft as a string ("auto" or the pinned/derived cap) so a client can
                # show the configured knob, not just infer it from round telemetry.
                max_draft = ("auto" if getattr(engine, "cap_controller", None) is not None
                             else str(getattr(engine, "max_draft_tokens", None) or "auto"))
                return self._send_json(200, {
                    "status": "ok", "model": engine.model_id, "mode": engine.mode,
                    "target": engine.target_repo, "drafter": engine.drafter_repo,
                    "max_draft": max_draft,
                    # resolved per pair at load (registry rows measured with lookup off carry
                    # it) — reported so a client shows the actual configuration, like max_draft
                    "lookup_drafts": bool(getattr(engine, "lookup_drafts", True)),
                    # 0.0 = off; settable per swap via /admin/load so a client can serve a
                    # pair at a measured cap+confidence bundle (e.g. Qwen3.8-27B-4bit's
                    # best is cap 7 + 0.3)
                    "confidence_threshold": float(
                        getattr(engine, "confidence_threshold", 0.0)),
                    # capability flag: /admin/race arms accept a per-arm "confidence".
                    # A client must gate its conf-bundle arm on this — an engine without
                    # it would silently DROP the field and the lane label would lie.
                    "race_arm_confidence": True,
                    # whether the small-M MMA verify kernel is live for this load (the
                    # per-shape probe admitted shapes and it wasn't forced off) — pairs
                    # with serve --no-small-m / the /admin/load "small_m" override so a
                    # kernel-vs-stock A/B no longer needs a version downgrade (issue #14)
                    "small_m": bool(getattr(engine, "small_m", False)),
                    "context_window": getattr(engine, "context_window", None),
                    # KV-cache quantization for the loaded target: 0 = full precision,
                    # 4/8 = quantized. Always present (0 when off) so a client can gate its
                    # picker on the key's presence — an engine without the /admin/load
                    # "kv_bits" override (< 0.13.1) also lacks this key (issue #17).
                    "kv_bits": int(getattr(getattr(engine, "target", None),
                                           "kv_bits", 0) or 0),
                    "max_output_tokens": engine.max_tokens_cap,
                    # Whether the loaded template reads `reasoning_effort`, and the server's
                    # default when one was configured — so a client only offers the control
                    # for models where it does something.
                    "supports_reasoning_effort": bool(
                        getattr(engine, "supports_reasoning_effort", False)),
                    "reasoning_effort": getattr(engine, "template_defaults", {}).get(
                        "reasoning_effort"),
                })
            if route == "/admin/status":
                if isinstance(engine, EngineHolder):
                    return self._send_json(200, engine.status())
                return self._send_json(200, {"ready": True, "loading": False,
                                             "model": engine.model_id, "error": None})
            # Model-free inventory routes answer without a loaded model — a client's model
            # picker must work from the no-model state (that state's whole point is picking).
            if route == "/doctor":
                from .diagnostics import doctor

                return self._send_json(200, doctor())
            if route == "/admin/models":
                from .diagnostics import disk_usage, installed_models, model_inventory

                installed = installed_models()
                loaded = (engine.target_repo
                          if not isinstance(engine, EngineHolder) or engine.ready else None)
                return self._send_json(200, {"models": model_inventory(),
                                             "installed": installed,
                                             "disk": disk_usage(installed),
                                             "loaded": loaded})
            if not self._require_ready():
                return
            if route in ("/v1/models", "/models"):
                return self._send_json(200, self._models_payload())
            if route == "/metrics":
                from .diagnostics import memory_info

                payload = engine.metrics()
                # Allocator state rides along so a client can show what the loaded model
                # actually holds resident — added handler-side so every engine (incl.
                # BatchEngine) reports it without owning the concern.
                payload["memory"] = memory_info()
                return self._send_json(200, payload)
            if route == "/calibration":
                return self._send_json(200, engine.calibration())
            if route == "/admin/integrations":
                from .integrations import integrations

                # The base URL a client should use is whatever this request came in on — so a
                # user on another machine, or behind a rename, gets a URL that actually reaches
                # the server, not a hardcoded 127.0.0.1.
                host = self.headers.get("Host") or f"{self.server.server_address[0]}:" \
                    f"{self.server.server_address[1]}"
                base = f"http://{host}"
                return self._send_json(200, {
                    "base_url": base,
                    "model": engine.model_id,
                    "integrations": integrations(base, engine.model_id, api_key),
                })
            if route == "/rounds":
                # Recent rounds as one JSON blob — the pull-based sibling of /events, for
                # clients that would rather poll than hold a stream open.
                limit = self._query_int("limit", 128)
                return self._send_json(200, {"rounds": engine.rounds.snapshot(limit),
                                             "stats": engine.rounds.stats()})
            if route == "/events":
                return self._events_stream()
            return self._send_error(404, f"unknown route {self.path}", "not_found")

        def _load(self, req: dict):
            """Swap the loaded model in place, keeping the server and its port.

            The port surviving is the point: an external client (Claude Code, a script) pointed
            at this server keeps working across a model change — a full restart would move the
            kernel-assigned port out from under it.
            """
            if not isinstance(engine, EngineHolder):
                return self._send_error(501, "this server was not started with hot-swap support")
            model = req.get("model")
            if not isinstance(model, str) or not model.strip():
                return self._send_error(400, "load needs a 'model' (repo or path)")
            mode = req.get("mode")
            max_draft = req.get("max_draft")
            lookup_drafts = req.get("lookup_drafts")
            if lookup_drafts is not None and not isinstance(lookup_drafts, bool):
                return self._send_error(400, "'lookup_drafts' must be a boolean (omit it to "
                                             "use the pair's measured default)")
            confidence = req.get("confidence_threshold")
            if confidence is not None:
                if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
                    return self._send_error(400, "'confidence_threshold' must be a number in "
                                                 "[0, 1] (0 disables; omit to keep the "
                                                 "server default)")
                confidence = float(confidence)
            # A LIMIT below the model's own maximum, mainly a RAM lever: the KV cache grows
            # linearly with context, and this caps it — requests past it get the "prompt is
            # too long" wording agent clients auto-compact on. It cannot extend a model past
            # its trained max_position_embeddings.
            context_window = req.get("context_window")
            if context_window is not None and (
                    isinstance(context_window, bool) or not isinstance(context_window, int)
                    or (context_window != 0
                        and not 1024 <= context_window <= 10_000_000)):
                return self._send_error(400, "'context_window' must be an integer token "
                                             "count >= 1024, or 0 to reset to the model's "
                                             "own maximum (omit it to keep the current "
                                             "setting — it is sticky across loads)")
            small_m = req.get("small_m")
            if small_m is not None and not isinstance(small_m, bool):
                return self._send_error(400, "'small_m' must be a boolean (false forces the "
                                             "stock verify kernel; omit it to use the "
                                             "server's setting)")
            # KV-cache quantization for the target (issue #17 — the app had no way to set
            # --kv-bits). 0 = explicitly full precision; omit = keep the server's setting.
            kv_bits = req.get("kv_bits")
            if kv_bits is not None and (isinstance(kv_bits, bool)
                                        or kv_bits not in (0, 4, 8)):
                return self._send_error(400, "'kv_bits' must be 0 (full precision), 4, or 8 "
                                             "(omit it to keep the server's setting)")
            try:
                status = engine.swap(model=model, mode=mode, max_draft=max_draft,
                                     lookup_drafts=lookup_drafts,
                                     confidence_threshold=confidence,
                                     context_window=context_window,
                                     small_m=small_m, kv_bits=kv_bits)
            except ValueError as e:                 # unknown model / unresolvable drafter
                return self._send_error(400, str(e))
            except Exception as e:  # noqa: BLE001 — load failed; report, server stays up
                traceback.print_exc()
                return self._send_error(500, f"could not load {model!r}: "
                                             f"{type(e).__name__}: {e}", "api_error")
            return self._send_json(200, status)

        def _race(self, req: dict):
            """Same prompt, several decode strategies, streamed as SSE.

            The point is not the tok/s — it is the verdict at the end. Every other local-LLM
            app asks you to take "lossless" on faith; this one runs both and compares the
            token ids.
            """
            prompt = req.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                return self._send_error(400, "race needs a non-empty 'prompt'")

            available = engine.race_arms_available()
            raw_arms = req.get("arms") or [{"mode": available[0]}, {"mode": "baseline"}]
            arms = []
            for arm in raw_arms:
                if isinstance(arm, str):
                    arm = {"mode": arm}
                mode = arm.get("mode")
                if mode not in available:
                    return self._send_error(
                        400, f"mode {mode!r} is not available with the loaded model "
                             f"(have: {', '.join(available)})")
                cap = arm.get("cap")
                if cap == "auto":
                    # per-round adaptive cap from this machine's cached cost curves —
                    # only meaningful for the drafter modes (lookup/baseline have no
                    # controller to drive)
                    if mode not in ("dspark", "dflash"):
                        return self._send_error(
                            400, f"cap 'auto' needs a drafter mode, not {mode!r}")
                elif cap is not None:
                    try:
                        cap = int(cap)
                    except (TypeError, ValueError):
                        return self._send_error(
                            400, f"arm cap must be an integer or 'auto', got {cap!r}")
                    if not 1 <= cap <= 64:
                        return self._send_error(400, "arm cap must be in 1..64")
                conf = arm.get("confidence")
                if conf is not None:
                    # per-arm confidence-head threshold — races a measured cap+confidence
                    # bundle (e.g. Qwen3.8-27B-4bit's cap 7 + 0.3) against plain caps.
                    # None = the server's loaded setting, so old clients change nothing.
                    if mode != "dspark":
                        return self._send_error(
                            400, "arm 'confidence' needs a dspark arm (only the DSpark "
                                 "drafter has a confidence head)")
                    if (isinstance(conf, bool) or not isinstance(conf, (int, float))
                            or not 0.0 <= conf <= 1.0):
                        return self._send_error(
                            400, "arm 'confidence' must be a number in [0, 1]")
                    conf = float(conf)
                arms.append({"mode": mode, "cap": cap, "confidence": conf})
            if not 2 <= len(arms) <= 4:
                return self._send_error(400, "race takes between 2 and 4 arms")

            max_tokens = _clamp_tokens(req.get("max_tokens"), 200, engine.max_tokens_cap)
            # Optional per-race thinking override (the Lab's toggle). Omitted = the server's
            # own default, same as every other endpoint; templates that don't know the kwarg
            # ignore it (encode_messages retries without unknown kwargs).
            thinking = req.get("thinking")
            if thinking is not None and not isinstance(thinking, bool):
                return self._send_error(400, "'thinking' must be a boolean (omit it to use "
                                             "the server's default)")
            template_kwargs = dict(engine.template_defaults)
            if thinking is not None:
                template_kwargs["enable_thinking"] = thinking
            effort = req.get("reasoning_effort")
            if effort is not None:
                try:
                    template_kwargs["reasoning_effort"] = _reasoning_effort(effort)
                except ValueError as e:
                    return self._send_error(400, str(e))
            try:
                prompt_ids = encode_messages(engine.tokenizer,
                                             [{"role": "user", "content": prompt}],
                                             **template_kwargs)
            except Exception as e:  # noqa: BLE001
                return self._send_error(400, f"could not apply chat template: {e}")

            self._sse_start()
            start_payload = {"arms": arms, "max_tokens": max_tokens,
                             "model": engine.model_id}
            if thinking is not None:
                start_payload["thinking"] = thinking
            if effort is not None:
                start_payload["reasoning_effort"] = template_kwargs["reasoning_effort"]
            self._sse(start_payload, "start")
            try:
                engine.race(prompt_ids, arms, max_tokens,
                            lambda name, payload: self._sse(payload, name))
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as e:  # noqa: BLE001 — report inside the stream; headers are sent
                self._sse({"message": str(e)}, "error")
            self.wfile.write(b"event: done\ndata: {}\n\n")
            self.wfile.flush()

        def _query_int(self, name: str, default: int) -> int:
            from urllib.parse import parse_qs

            try:
                return int(parse_qs(urlsplit(self.path).query).get(name, [default])[0])
            except (TypeError, ValueError):
                return default

        def _events_stream(self):
            """SSE stream of per-round telemetry.

            Deliberately independent of any one request: a client opens this once and watches
            every round the engine runs, whoever asked for it. That is what lets a UI show a
            live accept ribbon while a *different* client (an agent, say) is the one generating.
            """
            # Capture the round log ONCE. This stream outlives requests — including a hot
            # model swap, during which the holder has no engine and every attribute access
            # raises. Holding the log object keeps subscribe/unsubscribe paired on the same
            # log no matter what the holder does meanwhile.
            try:
                log = engine.rounds
            except RuntimeError:
                return self._send_error(503, "no model is loaded (a model swap is in "
                                             "progress or failed)", "server_error")
            q = log.subscribe()
            try:
                self._sse_start()
                # Replay a little history so a client that connects mid-generation has
                # something to draw immediately instead of an empty chart.
                for event in log.snapshot(32):
                    self._sse(event, "round")
                self._sse(log.stats(), "stats")
                idle = 0.0
                while True:
                    try:
                        event = q.get(timeout=1.0)
                    except _queue.Empty:
                        idle += 1.0
                        # Without traffic a proxy or the client can time the socket out; a
                        # comment frame is the cheapest legal keep-alive.
                        self._sse_comment("keepalive")
                        # A hot swap replaces the engine and its round log; this stream is
                        # then watching a dead object. End it so the client reconnects to
                        # the new engine's stream instead of going silent forever.
                        try:
                            if engine.rounds is not log:
                                break
                        except RuntimeError:
                            break                     # swap in progress — same conclusion
                        if idle >= 15.0:
                            self._sse(log.stats(), "stats")
                            idle = 0.0
                        continue
                    self._sse(event, "round")
            except (BrokenPipeError, ConnectionResetError):
                pass                     # client went away; nothing to report
            finally:
                log.unsubscribe(q)

        def do_POST(self):
            route = self._route()
            anthropic = route.endswith(("/messages", "/messages/count_tokens"))
            if not self._authed():
                if anthropic:
                    return self._send_json(401, A.error_body("invalid api key",
                                                             "authentication_error"))
                return self._send_error(401, "invalid api key", "authentication_error")
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                req = json.loads(raw or b"{}")
            except json.JSONDecodeError as e:
                if anthropic:
                    return self._send_json(400, A.error_body(f"invalid JSON body: {e}"))
                return self._send_error(400, f"invalid JSON body: {e}")

            try:
                # /admin/load runs the swap itself, so it must NOT be gated on readiness —
                # and /admin/unload's whole job is to *leave* the ready state.
                if route == "/admin/load":
                    return self._load(req)
                if route == "/admin/load/cancel":
                    # Cancel an in-flight first-time download (the load then fails cleanly
                    # and the holder's failed-load semantics apply: server up, model-less).
                    # `cleanup` also removes the partial files; default keeps them so a
                    # retried load resumes instead of restarting a 15 GB fetch.
                    from .download import cancel_current

                    cleanup = req.get("cleanup")
                    if cleanup is not None and not isinstance(cleanup, bool):
                        return self._send_error(400, "'cleanup' must be a boolean")
                    return self._send_json(200, cancel_current(cleanup=bool(cleanup)))
                if route == "/admin/unload":
                    if not isinstance(engine, EngineHolder):
                        return self._send_error(
                            501, "this server was not started with hot-swap support")
                    return self._send_json(200, engine.unload())
                if not self._require_ready():
                    return
                if route in ("/v1/chat/completions", "/chat/completions"):
                    return self._chat(req)
                if route in ("/v1/completions", "/completions"):
                    return self._completions(req)
                if route in ("/v1/messages", "/messages"):
                    return self._messages(req)
                if route in ("/v1/messages/count_tokens", "/messages/count_tokens"):
                    return self._count_tokens(req)
                if route == "/admin/race":
                    return self._race(req)
            except (BrokenPipeError, ConnectionResetError):
                # Client hung up mid-stream; nothing more to do — but say so. Swallowing it
                # silently left no server-side record at all, which made a stalled client
                # indistinguishable from a wedged engine (issue #14's diagnostic gap).
                print(f"[serve] client disconnected during {route}",
                      file=sys.stderr, flush=True)
                return
            except Exception as e:  # noqa: BLE001 — keep the server alive on a bad request
                if anthropic:
                    traceback.print_exc()
                    return self._send_json(500, A.error_body(
                        f"generation failed: {type(e).__name__}: {e}", "api_error"))
                # Log the full traceback: a per-request 500 is often an intermittent,
                # state-dependent edge (issue #5) that the client-side message alone can't
                # localize — without this the only record of WHERE it failed is discarded.
                traceback.print_exc()
                return self._send_error(500, f"generation failed: {type(e).__name__}: {e}",
                                        "server_error")
            return self._send_error(404, f"unknown route {self.path}", "not_found")

        # -- payloads --
        def _models_payload(self) -> dict:
            return {
                "object": "list",
                "data": [{
                    "id": engine.model_id,
                    "object": "model",
                    "created": engine.created,
                    "owned_by": "mlx-dspark",
                    # `display_name` is what Anthropic-format clients (Claude Code's gateway
                    # model discovery) label the entry with; harmless for OpenAI clients.
                    "display_name": f"{engine.model_id} (mlx-dspark {engine.mode})",
                    "x_mlx_dspark": {"mode": engine.mode, "target": engine.target_repo,
                                     "drafter": engine.drafter_repo},
                }],
            }

        def _chat(self, req: dict):
            messages = req.get("messages")
            if not isinstance(messages, list) or not messages:
                return self._send_error(400, "'messages' must be a non-empty list")
            # chat-template kwargs: server defaults, then per-request overrides. Supports the
            # common `chat_template_kwargs` extension and a top-level `enable_thinking` shortcut.
            tkw = {**engine.template_defaults, **(req.get("chat_template_kwargs") or {})}
            if "enable_thinking" in req:
                tkw["enable_thinking"] = bool(req["enable_thinking"])
            if req.get("reasoning_effort") is not None:
                # Top-level shortcut (the OpenAI field name); validated here so a typo is a
                # clear 400 rather than a Jinja raise_exception message. Templates that don't
                # know the kwarg ignore it. NOTE: the effort hint lands at the head of the
                # prompt (a system-block instruction), so changing it mid-conversation is a
                # full prefix-cache miss — clients should treat it as per-conversation.
                try:
                    tkw["reasoning_effort"] = _reasoning_effort(req["reasoning_effort"])
                except ValueError as e:
                    return self._send_error(400, str(e))
            if req.get("tools"):                      # let the template render the tool schemas
                tkw["tools"] = req["tools"]
            try:
                prompt_ids = encode_messages(
                    engine.tokenizer, normalize_tool_messages(messages), **tkw)
            except Exception as e:  # noqa: BLE001 — any template failure is a 400, not a crash
                return self._send_error(400, f"could not apply chat template: {e}")
            self._run(req, prompt_ids, chat=True)

        def _completions(self, req: dict):
            prompt = req.get("prompt")
            if isinstance(prompt, list):  # OpenAI allows a batch; we take the first
                prompt = prompt[0] if prompt else ""
            if not isinstance(prompt, str):
                return self._send_error(400, "'prompt' must be a string")
            prompt_ids = list(engine.tokenizer.encode(prompt))
            self._run(req, prompt_ids, chat=False)

        # -- Anthropic Messages API (the dialect Claude Code speaks) --
        def _encode_anthropic(self, req: dict) -> list[int]:
            """Prompt ids for an Anthropic request. Raises ValueError for a bad body."""
            messages = req.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("'messages': must be a non-empty array")
            tkw = dict(engine.template_defaults)
            tools = A.convert_tools(req.get("tools"))
            if tools:
                tkw["tools"] = tools
            th = req.get("thinking")
            if isinstance(th, dict) and th.get("type") == "disabled":
                # Map the request onto the model's own switch rather than only hiding the
                # output: a reasoning model that isn't asked to think is also much faster,
                # which is what a client disabling thinking is actually after.
                tkw["enable_thinking"] = False
            conv = normalize_tool_messages(A.convert_messages(messages, req.get("system")))
            return encode_messages(engine.tokenizer, conv, **tkw)

        def _anthropic_prompt(self, req: dict):
            """(prompt_ids, None) or (None, error_response_already_sent)."""
            try:
                return self._encode_anthropic(req), None
            except ValueError as e:
                return None, self._send_json(400, A.error_body(str(e)))
            except Exception as e:  # noqa: BLE001 — template failures are the client's problem
                return None, self._send_json(
                    400, A.error_body(f"could not apply chat template: {e}"))

        def _messages(self, req: dict):
            prompt_ids, err = self._anthropic_prompt(req)
            if prompt_ids is None:
                return err
            window = getattr(engine, "context_window", None)
            if window and len(prompt_ids) >= window:
                # Phrased so Claude Code's automatic compact-and-retry recognises it.
                return self._send_json(400, A.context_overflow_error(len(prompt_ids), window))
            temperature, top_p, top_k = _sampling(req, engine.sampling_defaults)
            params = {
                "max_tokens": _clamp_tokens(req.get("max_tokens"), engine.default_max_tokens,
                                            engine.max_tokens_cap),
                "temperature": temperature, "top_p": top_p, "top_k": top_k,
                "stop": A.norm_stop_sequences(req), "seed": None,
            }
            model = req.get("model") or engine.model_id
            # A reasoning model emits <think>…</think> whatever the request says; `thinking`
            # only decides whether that becomes a thinking block or is dropped. Absent means
            # emit it — silently discarding model output is the worse failure.
            th = req.get("thinking")
            want_thinking = not (isinstance(th, dict) and th.get("type") == "disabled")
            # tool schemas type the XML tool-call form, whose values are raw text
            schemas = schema_types(req.get("tools"))
            if req.get("stream"):
                return self._messages_stream(prompt_ids, params, model, want_thinking, schemas)
            res = engine.generate(prompt_ids, on_text=None, **params)
            body = A.build_message(res.text, model=model, input_tokens=len(prompt_ids),
                                   output_tokens=res.num_tokens,
                                   finish_reason=res.finish_reason, thinking=want_thinking,
                                   schemas=schemas)
            body["x_mlx_dspark"] = engine.spec_info(res)   # non-standard; clients ignore it
            self._send_json(200, body)

        def _prompt_opens_thinking(self, prompt_ids) -> bool:
            """Whether the chat template left a ``<think>`` open at the end of the prompt, so
            the stream starts *inside* a thinking block (see anthropic_api). Decoding the last
            few ids is enough and costs nothing."""
            try:
                return A.prompt_opens_thinking(engine.tokenizer.decode(prompt_ids[-8:]))
            except Exception:  # noqa: BLE001 — a tokenizer that can't decode just opts out
                return False

        def _messages_stream(self, prompt_ids, params, model, want_thinking=True, schemas=None):
            stream = A.MessageStream(model=model, input_tokens=len(prompt_ids),
                                     thinking=want_thinking, schemas=schemas,
                                     in_thinking=self._prompt_opens_thinking(prompt_ids),
                                     muse=engine.is_muse)
            self._sse_start()
            for name, payload in stream.start():
                self._sse(payload, name)

            # Prefill on a long agent prompt runs for seconds with nothing on the wire. A
            # periodic ping (a real Anthropic event type) keeps the client's socket and any
            # intermediary from timing out the request before the first token lands — and a
            # ping that fails to write is the disconnect signal for stretches where no text
            # flows (a _ToolGate-buffered tool call), so generation stops at the next round
            # instead of holding the MLX thread for a client that's gone (issue #14).
            done = threading.Event()
            gone = threading.Event()

            def _heartbeat():
                while not done.wait(STREAM_KEEPALIVE_S):
                    try:
                        self._sse({"type": "ping"}, "ping")
                    except Exception:  # noqa: BLE001 — socket gone; flag it and stand down
                        gone.set()
                        return

            threading.Thread(target=_heartbeat, daemon=True).start()

            def on_text(piece: str):
                if gone.is_set():
                    raise StopStreaming()
                try:
                    for name, payload in stream.delta(piece):
                        self._sse(payload, name)
                except (BrokenPipeError, ConnectionResetError) as e:
                    raise StopStreaming() from e   # end cleanly, keep the prefix cache

            try:
                res = engine.generate(prompt_ids, on_text=on_text, **params)
            finally:
                done.set()
            if gone.is_set():
                print(f"[serve] client disconnected mid-stream; generation stopped early "
                      f"after {res.num_tokens} tokens", file=sys.stderr, flush=True)
                return
            for name, payload in stream.finish(finish_reason=res.finish_reason,
                                               output_tokens=res.num_tokens):
                self._sse(payload, name)

        def _count_tokens(self, req: dict):
            """``POST /v1/messages/count_tokens``. Optional in the protocol — without it the
            client estimates context usage locally — but exact here, since we tokenize with
            the very model that will answer."""
            prompt_ids, err = self._anthropic_prompt(req)
            if prompt_ids is None:
                return err
            self._send_json(200, {"input_tokens": len(prompt_ids)})

        def _run(self, req: dict, prompt_ids: list[int], *, chat: bool):
            # request value > model's generation_config recommendation > library default —
            # explicit client settings always win; the model defaults only fill absences.
            temperature, top_p, top_k = _sampling(req, engine.sampling_defaults)
            params = {
                "max_tokens": _clamp_tokens(
                    req.get("max_tokens") or req.get("max_completion_tokens"),
                    engine.default_max_tokens, engine.max_tokens_cap),
                "temperature": temperature, "top_p": top_p, "top_k": top_k,
                "presence_penalty": float(req.get("presence_penalty") or 0.0),
                "frequency_penalty": float(req.get("frequency_penalty") or 0.0),
                "stop": _norm_stop(req.get("stop")),
                "seed": req.get("seed"),
            }
            # logprobs: chat sends {logprobs: bool, top_logprobs: int}; completions {logprobs: int}
            if chat:
                params["logprobs"] = (int(req.get("top_logprobs") or 0)
                                      if req.get("logprobs") else None)
            else:
                _lp = req.get("logprobs")
                params["logprobs"] = int(_lp) if _lp is not None else None
            model = req.get("model") or engine.model_id
            stream = bool(req.get("stream", False))
            n = max(1, min(int(req.get("n") or 1), 8))
            want_tools = bool(chat and req.get("tools"))
            cid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
            created = int(time.time())

            if stream:
                if n > 1:
                    return self._send_error(400, "'n' > 1 is not supported with stream=true")
                return self._run_stream(prompt_ids, params, model, cid, created, chat,
                                        req, want_tools)

            if n == 1 or params["temperature"] == 0:
                # greedy is deterministic: n identical choices from one generation
                res_list = [engine.generate(prompt_ids, on_text=None, **params)] * n
                gen_tokens = res_list[0].num_tokens
            else:
                # sampled n-best: submit concurrently so a BatchEngine batches the rows
                # (one shared weight-read per step); a plain Engine serializes them safely
                from concurrent.futures import ThreadPoolExecutor as _Pool

                with _Pool(max_workers=n) as pool:
                    res_list = list(pool.map(
                        lambda _i: engine.generate(prompt_ids, on_text=None, **params),
                        range(n)))
                gen_tokens = sum(r.num_tokens for r in res_list)
            usage = {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": gen_tokens,
                "total_tokens": len(prompt_ids) + gen_tokens,
            }
            choices = []
            for i, res in enumerate(res_list):
                if chat:
                    # split reasoning out of the text (Qwen `<think>`, Gemma thought channel,
                    # muse `to=self` analysis) so it rides in `reasoning_content` instead of
                    # leaking into content — and so tool calls parse from the answer, not the
                    # reasoning. split_thinking auto-detects the format; no reasoning -> ("", text).
                    reasoning, content = A.split_thinking(res.text)
                    finish, tool_calls = res.finish_reason, None
                    if want_tools:
                        parsed, cleaned = parse_tool_calls(content, schema_types(req.get("tools")))
                        if parsed:
                            tool_calls, content, finish = parsed, (cleaned or None), "tool_calls"
                    message = {"role": "assistant", "content": content}
                    if reasoning:
                        message["reasoning_content"] = reasoning
                    if tool_calls:
                        message["tool_calls"] = tool_calls
                    choice = {"index": i, "message": message, "finish_reason": finish}
                    if res.logprobs is not None:
                        choice["logprobs"] = _logprobs_content(res, engine.tokenizer)
                else:
                    choice = {"index": i, "text": res.text, "finish_reason": res.finish_reason}
                    if res.logprobs is not None:
                        choice["logprobs"] = _logprobs_completions(res, engine.tokenizer)
                choices.append(choice)
            obj = {"id": cid, "object": "chat.completion" if chat else "text_completion",
                   "created": created, "model": model, "choices": choices, "usage": usage,
                   "x_mlx_dspark": engine.spec_info(res_list[0])}
            self._send_json(200, obj)

        def _run_stream(self, prompt_ids, params, model, cid, created, chat, req, want_tools):
            self._sse_start()
            obj_type = "chat.completion.chunk" if chat else "text_completion"

            def base(delta_or_text, finish):
                if chat:
                    ch = {"index": 0, "delta": delta_or_text, "finish_reason": finish}
                else:
                    ch = {"index": 0, "text": delta_or_text, "finish_reason": finish}
                return {"id": cid, "object": obj_type, "created": created,
                        "model": model, "choices": [ch]}

            # opening chunk announces the assistant role (chat only)
            if chat:
                self._sse(base({"role": "assistant"}, None))

            # Keep-alive + liveness (see STREAM_KEEPALIVE_S). `gone` flips when a keep-alive
            # write fails — the only disconnect signal available while nothing else is on the
            # wire — and the on_text callbacks below turn it into StopStreaming, so the loop
            # ends at the next round with a normal partial result and the prefix cache intact.
            done = threading.Event()
            gone = threading.Event()

            def _heartbeat():
                while not done.wait(STREAM_KEEPALIVE_S):
                    try:
                        if chat:
                            # A spec-legal EMPTY delta chunk, not an SSE comment: most
                            # client SDKs never surface comments, so a comment doesn't
                            # reset their inter-chunk idle timer and a long quiet stretch
                            # (a 32k prefill, a long tool-gated tail) still times the
                            # stream out client-side (issue #19's DSH at 300 s). An empty
                            # delta parses as a normal chunk everywhere — OpenAI itself
                            # emits them (role-only and final chunks).
                            self._sse(base({}, None))
                        else:
                            self._sse_comment("keepalive")
                    except OSError:
                        gone.set()
                        return

            threading.Thread(target=_heartbeat, daemon=True).start()

            def _alive():
                if gone.is_set():
                    raise StopStreaming()

            try:
                res, finish = self._run_stream_body(prompt_ids, params, base, chat, req,
                                                    want_tools, _alive, gone)
            finally:
                done.set()
            if gone.is_set():
                # The client is gone; generation was cut short at a round boundary. Say so —
                # a silently dropped stream is indistinguishable from a wedge in the logs
                # (issue #14's diagnostic gap) — and skip the writes that would just raise.
                print(f"[serve] client disconnected mid-stream; generation stopped early "
                      f"after {res.num_tokens} tokens", file=sys.stderr, flush=True)
                return

            # final chunk carries finish_reason (+ usage if the client asked for it)
            final = base({} if chat else "", finish)
            opts = req.get("stream_options") or {}
            if opts.get("include_usage"):
                final["usage"] = {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": res.num_tokens,
                    "total_tokens": len(prompt_ids) + res.num_tokens,
                }
            final["x_mlx_dspark"] = engine.spec_info(res)
            self._sse(final)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def _run_stream_body(self, prompt_ids, params, base, chat, req, want_tools,
                             alive, gone):
            """One streaming generation: returns ``(GenResult, finish_reason)``. ``alive``
            raises StopStreaming once the keep-alive thread has seen the socket die
            (``gone`` is the same signal as a checkable flag)."""
            if want_tools:
                # Stream reasoning AND pre-tool-call answer text live; buffer only from the
                # first native tool-call marker on (the _ToolGate composition the Anthropic
                # dialect has had since v0.6.0). This used to buffer the ENTIRE generation
                # into one end-of-stream delta — with thinking models' 4-6k-token reasoning
                # preambles that meant minutes of dead air, and agent clients with
                # inter-chunk idle timeouts (DSH/pi, 300 s) dropped the stream and lost the
                # whole turn (issue #19). Only whole tool_use blocks land atomically at the
                # end — incremental tool-call streaming isn't reliable to reconstruct.
                splitter = A.ThinkingStreamSplitter(
                    in_thinking=self._prompt_opens_thinking(prompt_ids))
                gate = A._ToolGate()

                def _emit_tools_split(chunks):
                    for kind, text in chunks:
                        if not text:
                            continue
                        if kind == "reasoning":
                            self._sse(base({"reasoning_content": text}, None))
                        else:
                            safe = gate.feed(text)
                            if safe:
                                self._sse(base({"content": safe}, None))

                def on_text(piece: str):
                    alive()
                    try:
                        _emit_tools_split(splitter.feed(piece))
                    except (BrokenPipeError, ConnectionResetError) as e:
                        raise StopStreaming() from e
                res = engine.generate(prompt_ids, on_text=on_text, **params)
                if gone.is_set():
                    return res, res.finish_reason   # nobody listening; skip the emission
                _emit_tools_split(splitter.feed("", final=True))
                # the gate's held-back tail: everything from the first marker (a complete
                # tool-call markup), or just the marker-length holdback when none appeared
                tail = gate.buf[gate.sent:]
                parsed, cleaned = parse_tool_calls(tail, schema_types(req.get("tools")))
                if parsed:
                    self._sse(base({"tool_calls": [{"index": i, **tc}
                                                   for i, tc in enumerate(parsed)]}, None))
                    return res, "tool_calls"
                if cleaned:
                    self._sse(base({"content": cleaned}, None))
                return res, res.finish_reason
            if chat and engine.is_muse:
                # muse streams its analysis (`to=self`) and answer (`to=user`) channels
                # interleaved with structural markers; split them incrementally so reasoning
                # rides in `reasoning_content` and only the answer lands in `content`.
                muse = A.MuseChannelParser()

                def _emit_muse(chunks):
                    for kind, text in chunks:
                        if not text:
                            continue
                        field = "reasoning_content" if kind == "reasoning" else "content"
                        self._sse(base({field: text}, None))

                def on_text(piece: str):
                    alive()
                    try:
                        _emit_muse(muse.feed(piece))
                    except (BrokenPipeError, ConnectionResetError) as e:
                        raise StopStreaming() from e
                res = engine.generate(prompt_ids, on_text=on_text, **params)
                _emit_muse(muse.feed("", final=True))   # flush the held-back tail
                return res, res.finish_reason
            if chat:
                # Split reasoning into `reasoning_content` incrementally (the streaming twin
                # of the non-streaming path's split_thinking). Covers both the self-opened
                # `<think>` and the prefilled-opener templates (Qwen3-2507 / qwen3_5 / Qwen3.8
                # prefill the opener in the prompt, so the output holds only the closer —
                # streamed raw, clients render the reasoning as answer text with a stray
                # `</think>` in the middle).
                splitter = A.ThinkingStreamSplitter(
                    in_thinking=self._prompt_opens_thinking(prompt_ids))

                def _emit_split(chunks):
                    for kind, text in chunks:
                        if not text:
                            continue
                        field = "reasoning_content" if kind == "reasoning" else "content"
                        self._sse(base({field: text}, None))

                def on_text(piece: str):
                    alive()
                    try:
                        _emit_split(splitter.feed(piece))
                    except (BrokenPipeError, ConnectionResetError) as e:
                        # client hung up mid-stream: end generation gracefully at the next
                        # round so the engine can still store the prefix cache (raising
                        # anything else would invalidate it)
                        raise StopStreaming() from e
                res = engine.generate(prompt_ids, on_text=on_text, **params)
                _emit_split(splitter.feed("", final=True))   # flush the held-back tail
                return res, res.finish_reason

            def on_text(piece: str):
                alive()
                try:
                    self._sse(base(piece, None))
                except (BrokenPipeError, ConnectionResetError) as e:
                    raise StopStreaming() from e
            res = engine.generate(prompt_ids, on_text=on_text, **params)
            return res, res.finish_reason

    return Handler


# --------------------------------------------------------------------------- entrypoint


def run_server(engine, *, host: str = "127.0.0.1", port: int = 8080,
               api_key: str | None = None) -> None:
    # ``engine`` may be an Engine, a BatchEngine, or an EngineHolder (hot-swap). All three
    # delegate the attributes the banner and handler read, so this is uniform.
    handler = make_handler(engine, api_key)
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    base = f"http://{host}:{port}"
    # A --no-model start hands over a model-less EngineHolder; the banner can't dereference
    # engine attributes then (the holder raises a clear "no model" through __getattr__).
    loaded = not isinstance(engine, EngineHolder) or engine.ready
    print("=" * 64)
    if loaded:
        print(f"  mlx-dspark server  ·  mode={engine.mode}  ·  model={engine.model_id}")
        print(f"  target : {engine.target_repo}")
        if engine.drafter_repo:
            print(f"  drafter: {engine.drafter_repo}")
        if engine.prefix is not None:
            print(f"  prefix cache: on{'  (+SSD spill)' if engine.prefix.l2_dir else ''}")
        else:
            print("  prefix cache: off (not reusable for this mode/target)")
        # Stated up front so a serve session's kernel arm is on record (benchmark prints
        # the same); forced off via --no-small-m, per-swap via /admin/load {"small_m": ...}.
        print(f"  small-M verify kernel: "
              f"{'on (probe-verified shapes)' if getattr(engine, 'small_m', False) else 'off'}")
        if isinstance(engine, BatchEngine):
            print(f"  batching: micro-batch up to {engine.max_batch} concurrent "
                  f"({engine.mode}; serial fallback for temp>0 dspark / lone requests)")
        if engine.cap_controller is not None:
            print(f"  max-draft: auto (calibrated for this machine; starting cap "
                  f"{engine.cap_controller.cap})")
        if engine.sampling_defaults:
            print(f"  sampling defaults (model generation_config; requests override): "
                  f"{engine.sampling_defaults}")
    else:
        print("  mlx-dspark server  ·  no model loaded")
        print(f"  load one:  curl {base}/admin/load -d '{{\"model\":\"<repo>\"}}'")
    print(f"  listening on {base}   (OpenAI base_url: {base}/v1)")
    print(f"  claude code : ANTHROPIC_BASE_URL={base}   (or run `mlx-dspark claude`)")
    if api_key:
        print("  auth   : Bearer <api-key> required")
    print("=" * 64)
    if loaded:
        print(f"  curl {base}/v1/chat/completions -H 'Content-Type: application/json' \\")
        print(f"    -d '{{\"model\":\"{engine.model_id}\",\"messages\":"
              "[{\"role\":\"user\",\"content\":\"Hi\"}],\"stream\":true}'")
        print("=" * 64, flush=True)
    else:
        print(flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down.")
    finally:
        httpd.server_close()
