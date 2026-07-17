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
import json
import os
import queue as _queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
from .prefix_cache import PrefixCache, target_cache_reusable
from .tools import normalize_tool_messages, parse_tool_calls

MODES = ("dspark", "dflash", "lookup", "baseline")


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
        lookup_drafts: bool = True,
        lookup_long_draft: int = 32,
        executor: ThreadPoolExecutor | None = None,
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
        self.sampling_defaults = dict(sampling_defaults or {})
        self.default_max_tokens = default_max_tokens
        self.max_tokens_cap = max_tokens_cap
        self.prefix_cache_slots = max(1, prefix_cache_slots)
        self.lookup_drafts = lookup_drafts                 # hybrid n-gram drafts in dspark mode
        self.lookup_long_draft = lookup_long_draft         # match-scaled long-draft ceiling
        apply_wired_limit()                                # keep the weights resident
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
        self.stats = {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "generation_seconds": 0.0,
            "sum_accept_len": 0.0,   # accept-len weighted by tokens, for a token-weighted mean
        }

    def _build_prefix_cache(self, enabled, l2_dir, max_ram_mb):
        """Enable prefix caching where reuse is exact: dspark/lookup/baseline on dense
        (KVCache) targets, and on sliding-window (RotatingKVCache) targets like Gemma-4
        while under the window — entries are refused at store time once any layer wraps.
        Disabled for DFlash (its drafter cache can't roll back)."""
        if not enabled or self.mode == "dflash":
            return None
        try:
            if not target_cache_reusable(self.target.make_cache()):
                return None
        except Exception:  # noqa: BLE001
            return None
        make_ctx = self.drafter.make_ctx_cache if self.mode == "dspark" else None
        return PrefixCache(self.target.make_cache, make_ctx,
                           l2_dir=l2_dir, max_ram_bytes=max(0, max_ram_mb) * 1024 * 1024,
                           slots=self.prefix_cache_slots)

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
        prefix_cache: bool = True,
        prefix_cache_dir: str | None = None,
        prefix_cache_max_ram_mb: int = 0,
        default_max_tokens: int = 2048,
        max_tokens_cap: int = 32768,
        default_temperature: float | None = None,
        default_top_p: float | None = None,
        default_top_k: int | None = None,
        prefix_cache_slots: int = 2,
        lookup_drafts: bool = True,
        lookup_long_draft: int = 32,
        batch_widths: list[int] | None = None,   # e.g. [2, max_batch]: calibrate (B,cap) grid
        kv_bits: int | None = None,              # quantize the target KV cache (4/8)
    ) -> "Engine":
        if mode != "auto" and mode not in MODES:
            raise ValueError(f"mode must be one of {MODES} or 'auto', got {mode!r}")
        # "auto" picks the best available speculation for this target (dspark -> dflash ->
        # drafter-free lookup), so any model repo serves without extra flags.
        mode, target_repo, drafter_repo = resolve_mode(model, mode=mode, drafter=drafter,
                                                       family=family, target=target)

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
            # --max-draft auto: measure this machine+pair's cost curves once (disk-cached)
            # and let a CapController pick the cap per round. Only meaningful with a drafter.
            ctrl = None
            if max_draft_tokens == "auto" and mode in ("dspark", "dflash"):
                from .calibrate import calibrate

                ctrl = calibrate(tgt, draft, mode=mode, target_repo=target_repo,
                                 drafter_repo=drafter_repo, batch_widths=batch_widths)
            return tgt, tok, draft, ctrl

        tgt, tok, draft, cap_controller = executor.submit(_load_models).result()
        if max_draft_tokens == "auto":
            max_draft_tokens = None                     # controller drives, up to the full block
        # default cap: dspark's measured optimum is 2; dflash's native point is the full
        # block; lookup drafts are free so a modest 6 balances hit gains vs miss-free rounds
        if max_draft_tokens is None and cap_controller is None:
            if mode == "dspark":
                max_draft_tokens = 2
            elif mode == "lookup":
                max_draft_tokens = 6
        model_id = target_repo.rstrip("/").split("/")[-1]
        template_defaults = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
        # sampling defaults: model's generation_config.json, then explicit server flags on top
        # (many mlx-community conversions ship no generation_config — e.g. the Qwen3 repos —
        # so the flags are the way to serve sampled-by-default there)
        sampling_defaults = _generation_defaults(target_repo)
        for key, val in (("temperature", default_temperature), ("top_p", default_top_p),
                         ("top_k", default_top_k)):
            if val is not None:
                sampling_defaults[key] = val
        return cls(tgt, tok, draft, mode=mode, model_id=model_id, target_repo=target_repo,
                   drafter_repo=drafter_repo, max_draft_tokens=max_draft_tokens,
                   confidence_threshold=confidence_threshold, template_defaults=template_defaults,
                   prefix_cache=prefix_cache, prefix_cache_dir=prefix_cache_dir,
                   prefix_cache_max_ram_mb=prefix_cache_max_ram_mb,
                   cap_controller=cap_controller,
                   sampling_defaults=sampling_defaults,
                   default_max_tokens=default_max_tokens, max_tokens_cap=max_tokens_cap,
                   prefix_cache_slots=prefix_cache_slots, lookup_drafts=lookup_drafts,
                   lookup_long_draft=lookup_long_draft,
                   executor=executor)

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

    def _generate_impl(self, prompt_ids, max_tokens, temperature, top_p, top_k, stop, seed,
                       on_text, presence_penalty=0.0, frequency_penalty=0.0,
                       logprobs=None) -> GenResult:
        # prefix caching: reuse the shared conversation prefix's KV (dspark/baseline on a
        # dense target); `cache is None` means this mode/target doesn't reuse.
        cache = ctx = None
        reuse_len = 0
        if self.prefix is not None:
            cache, ctx, reuse_len = self.prefix.acquire(prompt_ids)
        try:
            if self.mode == "dspark":
                res = speculative_generate(
                    self.target, self.tokenizer, self.drafter, prompt_ids=prompt_ids,
                    cache=cache, ctx_caches=ctx, reuse_len=reuse_len,
                    max_new_tokens=max_tokens, max_draft_tokens=self.max_draft_tokens,
                    cap_controller=self.cap_controller, lookup_drafts=self.lookup_drafts,
                    lookup_long_draft=self.lookup_long_draft,
                    confidence_threshold=self.confidence_threshold,
                    temperature=temperature, top_p=top_p, top_k=top_k,
                    presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
                    logprobs=logprobs, seed=seed, stop=stop, on_text=on_text,
                )
            elif self.mode == "dflash":
                res = dflash_generate(
                    self.target, self.tokenizer, self.drafter, prompt_ids=prompt_ids,
                    max_new_tokens=max_tokens, max_draft_tokens=self.max_draft_tokens,
                    cap_controller=self.cap_controller,
                    temperature=temperature, top_p=top_p, top_k=top_k,
                    seed=seed, stop=stop, on_text=on_text,
                )
            elif self.mode == "lookup":
                res = lookup_generate(
                    self.target, self.tokenizer, prompt_ids=prompt_ids,
                    cache=cache, reuse_len=reuse_len,
                    max_new_tokens=max_tokens, max_draft_tokens=self.max_draft_tokens or 6,
                    long_draft_tokens=max(self.max_draft_tokens or 6, self.lookup_long_draft),
                    temperature=temperature, top_p=top_p, top_k=top_k,
                    seed=seed, stop=stop, on_text=on_text,
                )
            else:
                res = greedy_generate(
                    self.target, self.tokenizer, prompt_ids=prompt_ids,
                    cache=cache, reuse_len=reuse_len,
                    max_new_tokens=max_tokens, temperature=temperature, top_p=top_p,
                    top_k=top_k, presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty, logprobs=logprobs, seed=seed,
                    stop=stop, on_text=on_text,
                )
        except BaseException:                     # never leave a desynced cache behind
            if self.prefix is not None:
                self.prefix.reset()
            raise
        if self.prefix is not None and cache is not None:
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
        }


# --------------------------------------------------------------------------- batching engine


_STOP = object()   # sentinel: unwedges the scheduler thread so the process can exit


class _Job:
    """One queued generation request awaiting a (possibly batched) run."""
    __slots__ = ("prompt_ids", "params", "on_text", "result", "error", "done")

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
        job = _Job(prompt_ids, dict(max_tokens=max_tokens, temperature=temperature,
                                    top_p=top_p, top_k=top_k, presence_penalty=presence_penalty,
                                    frequency_penalty=frequency_penalty, logprobs=logprobs,
                                    stop=stop or [], seed=seed), on_text)
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
    (``max_batch > 1``), a batchable dense mlx-lm target, and a mode with a batched kernel
    (dspark/baseline). Otherwise return the engine unchanged (serialized)."""
    from .batch_engine import batchable

    if max_batch <= 1 or engine.mode not in ("dspark", "baseline") or not batchable(engine.target):
        return engine
    return BatchEngine(engine, max_batch=max_batch)


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

        # -- low-level replies --
        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

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

        def _sse(self, obj: dict):
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode("utf-8"))
            self.wfile.flush()

        # -- auth --
        def _authed(self) -> bool:
            if not api_key:
                return True
            auth = self.headers.get("Authorization", "")
            return auth == f"Bearer {api_key}"

        def log_message(self, fmt, *args):  # quieter default logging
            return

        # -- routing --
        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self._cors()
            self.end_headers()

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                return self._send_json(200, {"status": "ok", "model": engine.model_id,
                                             "mode": engine.mode})
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                return self._send_json(200, self._models_payload())
            if self.path.rstrip("/") == "/metrics":
                return self._send_json(200, engine.metrics())
            return self._send_error(404, f"unknown route {self.path}", "not_found")

        def do_POST(self):
            if not self._authed():
                return self._send_error(401, "invalid api key", "authentication_error")
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                req = json.loads(raw or b"{}")
            except json.JSONDecodeError as e:
                return self._send_error(400, f"invalid JSON body: {e}")

            route = self.path.rstrip("/")
            try:
                if route in ("/v1/chat/completions", "/chat/completions"):
                    return self._chat(req)
                if route in ("/v1/completions", "/completions"):
                    return self._completions(req)
            except (BrokenPipeError, ConnectionResetError):
                return  # client hung up mid-stream; nothing more to do
            except Exception as e:  # keep the server alive on a bad request
                return self._send_error(500, f"generation failed: {e}", "server_error")
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
            if req.get("tools"):                      # let the template render the tool schemas
                tkw["tools"] = req["tools"]
            try:
                prompt_ids = encode_messages(
                    engine.tokenizer, normalize_tool_messages(messages), **tkw)
            except Exception as e:
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

        def _run(self, req: dict, prompt_ids: list[int], *, chat: bool):
            # request value > model's generation_config recommendation > library default —
            # explicit client settings always win; the model defaults only fill absences.
            sd = engine.sampling_defaults

            def _or_default(key: str, fallback):
                v = req.get(key)
                return sd.get(key, fallback) if v is None else v

            params = dict(
                max_tokens=_clamp_tokens(req.get("max_tokens") or req.get("max_completion_tokens"),
                                         engine.default_max_tokens, engine.max_tokens_cap),
                temperature=float(_or_default("temperature", 0.0)),
                top_p=float(_or_default("top_p", 1.0)),
                top_k=int(_or_default("top_k", 0)),
                presence_penalty=float(req.get("presence_penalty") or 0.0),
                frequency_penalty=float(req.get("frequency_penalty") or 0.0),
                stop=_norm_stop(req.get("stop")),
                seed=req.get("seed"),
            )
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
                    content, finish, tool_calls = res.text, res.finish_reason, None
                    if want_tools:
                        parsed, cleaned = parse_tool_calls(res.text)
                        if parsed:
                            tool_calls, content, finish = parsed, (cleaned or None), "tool_calls"
                    message = {"role": "assistant", "content": content}
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

            if want_tools:
                # buffer, then emit tool_calls (or cleaned content) in one delta — incremental
                # tool-call streaming isn't reliable to reconstruct, so we resolve at the end
                res = engine.generate(prompt_ids, on_text=None, **params)
                parsed, cleaned = parse_tool_calls(res.text)
                if parsed:
                    self._sse(base({"tool_calls": [{"index": i, **tc}
                                                   for i, tc in enumerate(parsed)]}, None))
                    finish = "tool_calls"
                else:
                    if cleaned:
                        self._sse(base({"content": cleaned}, None))
                    finish = res.finish_reason
            else:
                def on_text(piece: str):
                    try:
                        self._sse(base({"content": piece} if chat else piece, None))
                    except (BrokenPipeError, ConnectionResetError) as e:
                        # client hung up mid-stream: end generation gracefully at the next
                        # round so the engine can still store the prefix cache (raising
                        # anything else would invalidate it)
                        raise StopStreaming() from e
                res = engine.generate(prompt_ids, on_text=on_text, **params)
                finish = res.finish_reason

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

    return Handler


# --------------------------------------------------------------------------- entrypoint


def run_server(engine: Engine, *, host: str = "127.0.0.1", port: int = 8080,
               api_key: str | None = None) -> None:
    handler = make_handler(engine, api_key)
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    base = f"http://{host}:{port}"
    print("=" * 64)
    print(f"  mlx-dspark server  ·  mode={engine.mode}  ·  model={engine.model_id}")
    print(f"  target : {engine.target_repo}")
    if engine.drafter_repo:
        print(f"  drafter: {engine.drafter_repo}")
    if engine.prefix is not None:
        print(f"  prefix cache: on{'  (+SSD spill)' if engine.prefix.l2_dir else ''}")
    else:
        print("  prefix cache: off (not reusable for this mode/target)")
    if isinstance(engine, BatchEngine):
        print(f"  batching: micro-batch up to {engine.max_batch} concurrent "
              f"({engine.mode}; serial fallback for temp>0 dspark / lone requests)")
    if engine.cap_controller is not None:
        print(f"  max-draft: auto (calibrated for this machine; starting cap "
              f"{engine.cap_controller.cap})")
    if engine.sampling_defaults:
        print(f"  sampling defaults (model generation_config; requests override): "
              f"{engine.sampling_defaults}")
    print(f"  listening on {base}   (OpenAI base_url: {base}/v1)")
    if api_key:
        print("  auth   : Bearer <api-key> required")
    print("=" * 64)
    print(f"  curl {base}/v1/chat/completions -H 'Content-Type: application/json' \\")
    print(f"    -d '{{\"model\":\"{engine.model_id}\",\"messages\":"
          "[{\"role\":\"user\",\"content\":\"Hi\"}],\"stream\":true}'")
    print("=" * 64, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down.")
    finally:
        httpd.server_close()
