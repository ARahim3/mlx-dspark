"""Hardware-aware auto-calibration for the speculative cap (``--max-draft auto``).

Why this exists: on Apple Silicon the verify cost is NOT linear in the number of tokens
verified per round — it is convex with a machine/model-dependent knee (measured on
gemma-4-12B-8bit / M4 Pro: ~5 ms/tok for widths 1–3, then +19 ms for the 4th, where MLX's
quantized matmul leaves its cheap few-rows path for GEMM tiling; see NOTES "Perf pass 2").
So the optimal cap is a property of the (machine, target, drafter) triple and moves across
chip generations. Instead of a hard-coded default, measure the cost curves once per pair
(a few seconds, cached on disk) and let a small controller pick the cap that maximizes
expected tokens/second under a live acceptance estimate.

Pieces:
  * ``measure_verify_curve`` / ``measure_dspark_drafter_curve`` / ``measure_dflash_drafter_cost``
    — warm synthetic-forward timings. Content-independent (pure compute), so a constant token
    id is fine; the caches are trimmed back between iterations so every timing sees the same state.
  * ``CapController`` — per-*position* conditional acceptance EWMA. Each round contributes
    Bernoulli observations (n successes, plus one failure unless the block was fully accepted,
    which is censored), so rounds run at ANY cap update the same estimate; expected committed
    tokens at a candidate cap C is the geometric extrapolation 1 + Σ_{i<=C} p^i. The cap picked
    is argmax of committed / (drafter(C) + verify(C+1)), with hysteresis against flapping.
  * ``calibrate`` — measure-or-load with a JSON disk cache keyed by device, mlx version, mode,
    and model basenames.

Correctness: the cap only chooses how many drafted tokens are *verified* per round — the
backbone always drafts full-width and the target verifies every emitted token — so output is
lossless for ANY per-round cap sequence. Auto-cap can never change what is generated.
"""

from __future__ import annotations

import itertools
import json
import os
import time

import mlx.core as mx

CACHE_DIR = os.path.expanduser("~/.cache/mlx_dspark")
CACHE_FILE = "calibration.json"
SCHEMA = 3   # 2: verify curve includes width 1; adds the measured per-round overhead
# 3: causal-block (DFlash-lineage) drafter curves are measured at draft_width(cap) — the
# truncated backbone the loop now runs — so cached full-width curves must re-measure
# (they priced Muse's drafter ~2.5x too high at small caps and flat in cap).
CTX_LEN = 512          # curve shape is nearly ctx-independent (SDPA is flat in width)
TOKEN_ID = 7           # arbitrary; timings are content-independent


def _bench(fn, iters: int = 8, warmup: int = 3) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


# --------------------------------------------------------------------------- measurement


def measure_verify_curve(target, tap: list[int], widths: list[int],
                         ctx_len: int = CTX_LEN) -> dict[int, float]:
    """ms per verify forward (with the hidden-state tap) at each width, warm cache."""
    cache = target.make_cache()
    mx.eval(target.run(mx.array([[TOKEN_ID] * ctx_len]), cache, tap)[0])
    curve: dict[int, float] = {}
    for m in sorted(widths):
        x = mx.array([[TOKEN_ID] * m])

        # loop vars bound as defaults: fn is consumed inside this iteration, but binding
        # makes that explicit rather than relying on it (ruff B023)
        def fn(x=x, m=m):
            logits, _ = target.run(x, cache, tap)
            for c in cache:                      # roll back so every iteration is identical
                if c is not None and hasattr(c, "trim"):
                    c.trim(m)
            return logits

        curve[m] = _bench(fn)
    return curve


def measure_dspark_drafter_curve(drafter, caps: list[int],
                                 ctx_len: int = CTX_LEN) -> dict[int, float]:
    """ms per DSpark draft round at each cap (backbone runs at ``drafter.draft_width(cap)``
    — full block for bidirectional heads, truncated for causal DFlash-lineage heads, exactly
    as the production loop does; lm_head/markov run over ``cap`` positions — the slice fix).
    So cost grows mildly with cap on a bidirectional head and tracks the truncated width on
    a causal one — the controller's model must price what the loop will actually run."""
    cfg = drafter.config
    ctx = drafter.make_ctx_cache()
    fused = mx.random.normal(
        (1, ctx_len, len(cfg.target_layer_ids) * cfg.hidden_size)).astype(mx.bfloat16)
    drafter.update_context(fused, ctx_offset=0, ctx_caches=ctx)
    mx.eval([c.k for c in ctx])
    curve: dict[int, float] = {}
    for cap in sorted(caps):
        block_ids = [TOKEN_ID] * drafter.draft_width(cap)

        def fn(cap=cap, block_ids=block_ids):
            noise = drafter.embed(mx.array([block_ids]))
            hidden = drafter.backbone(noise, ctx_len, ctx)
            logits = drafter.compute_logits(drafter.head_slice(hidden, cap))[0]
            return drafter.sample_block(logits, first_prev_token=TOKEN_ID)

        curve[cap] = _bench(fn)
    return curve


def measure_batch_verify_grid(target, tap: list[int], batch_widths: list[int],
                              widths: list[int], ctx_len: int = CTX_LEN
                              ) -> dict[int, dict[int, float]]:
    """``verify_ms[B][width]`` measured through the real batched forward (BatchCache rows at
    ``ctx_len``). The qmm knee is a function of B×width total matmul rows, so the optimal cap
    *shrinks* as the batch widens — this grid is what lets the controller pick a per-batch-width
    cap instead of assuming the B=1 curve. One prefill builds one row; the same KV is copied
    into all B slots (timings are content-independent)."""
    from .batch_engine import BatchCache, batched_forward, build_batch_mask

    grid: dict[int, dict[int, float]] = {}
    for B in sorted({int(b) for b in batch_widths if int(b) > 1}):
        cache = target.make_cache()
        mx.eval(target.run(mx.array([[TOKEN_ID] * ctx_len]), cache, tap)[0])
        caches = []
        for c in cache:
            bc = BatchCache.empty(B, c.keys.shape[1], c.keys.shape[3],
                                  c.values.shape[3], c.keys.dtype)
            row_k = c.keys[..., : c.offset, :]
            row_v = c.values[..., : c.offset, :]
            for b in range(B):
                bc.set_row(b, row_k, row_v, c.offset)
            caches.append(bc)
        row: dict[int, float] = {}
        for m in sorted(widths):
            ids = mx.array([[TOKEN_ID] * m] * B)

            def fn(ids=ids, caches=caches, m=m, B=B):
                mask = build_batch_mask(caches[0].offsets, m)
                logits, _ = batched_forward(target, ids, caches, tap, mask=mask)
                for c in caches:                 # roll back so every iteration is identical
                    c.trim([m] * B)
                return logits

            row[m] = _bench(fn)
        grid[B] = row
    return grid


def measure_dspark_round_overhead(target, drafter, verify_ms: dict[int, float],
                                  drafter_ms: dict[int, float], cap: int = 2,
                                  ctx_len: int = CTX_LEN) -> float:
    """ms of per-round cost the drafter/verify curves do NOT capture — graph build for
    ~70 layers, the round's device sync, the sequential markov sampling, the ctx-append
    projections, Python bookkeeping. Measured as (full synthetic round at ``cap``, full-
    accept bookkeeping — the common case) minus the two curves' parts. This is what makes
    the controller's absolute tok/s predictions land close enough to reality that the
    cap-0 "don't speculate" option competes on fair terms."""
    cfg = drafter.config
    tap = list(cfg.target_layer_ids)
    k = int(cfg.block_size)
    cap = max(1, min(int(cap), k))
    w = drafter.draft_width(cap)   # what the production loop will run at this cap
    cache = target.make_cache()
    if hasattr(target, "reset_spec"):
        target.reset_spec()
    mx.eval(target.run(mx.array([[TOKEN_ID] * ctx_len]), cache, tap)[0])
    ctx = drafter.make_ctx_cache()
    fused = mx.random.normal(
        (1, ctx_len, len(tap) * cfg.hidden_size)).astype(mx.bfloat16)
    drafter.update_context(fused, ctx_offset=0, ctx_caches=ctx)
    mx.eval([c.k for c in ctx])
    pos = ctx_len

    def round_fn():
        nonlocal pos
        noise = drafter.embed(mx.array([[TOKEN_ID] * w]))
        hidden = drafter.backbone(noise, pos, ctx)
        logits = drafter.compute_logits(drafter.head_slice(hidden, cap))[0]
        draft_arr = drafter.sample_block(logits, first_prev_token=TOKEN_ID)
        verify_ids = mx.concatenate(
            [mx.array([TOKEN_ID], dtype=draft_arr.dtype), draft_arr]).reshape(1, -1)
        v_logits, v_fused = target.verify(verify_ids, cache, tap)
        tt_arr = mx.argmax(v_logits[0], axis=-1)
        match = (draft_arr == tt_arr[: draft_arr.shape[0]]).astype(mx.int32)
        n_arr = mx.cumprod(match).sum()
        mx.eval(n_arr, tt_arr, draft_arr)
        int(n_arr.item()), draft_arr.tolist(), tt_arr.tolist()   # pay the real syncs
        target.rollback(cache, 0, [])                            # full-accept bookkeeping
        drafter.update_context(v_fused[:, : cap + 1, :], ctx_offset=pos, ctx_caches=ctx)
        pos += cap + 1
        mx.async_eval([c.k for c in ctx])
        return n_arr

    round_ms = _bench(round_fn)
    parts = _interp(drafter_ms, cap) + _interp(verify_ms, cap + 1)
    return max(0.0, round_ms - parts)


def measure_dflash_drafter_cost(drafter, ctx_len: int = CTX_LEN) -> float:
    """ms per DFlash draft round (cap-independent: the drafter always denoises and reads
    logits for the full block)."""
    cfg = drafter.config
    n_tap = len(cfg.target_layer_ids)
    dcache = drafter.make_cache()
    ctx = mx.random.normal((1, ctx_len, n_tap * cfg.hidden_size)).astype(mx.bfloat16)
    block = mx.array([[TOKEN_ID] + [int(cfg.mask_token_id)] * (int(cfg.block_size) - 1)])
    mx.eval(drafter(block, ctx, dcache, logits_start=1))     # appends ctx to the draft cache
    step_ctx = mx.random.normal((1, 2, n_tap * cfg.hidden_size)).astype(mx.bfloat16)
    # each timed call appends 2 ctx positions (like a real round); the slight cache growth
    # over the timing loop is negligible at ctx_len=512
    return _bench(lambda: drafter(block, step_ctx, dcache, logits_start=1))


def knee_width(verify_ms: dict[int, float]) -> int:
    """The verify-cost curve's convex knee: the smallest width whose marginal cost jumps clearly
    above the cheap region below it (MLX's quantized matmul leaving its few-rows path for GEMM
    tiling). On this M4 Pro the knee sits at width ~4 (see NOTES "Perf pass 2"). This is the
    machine-dependent quantity that decides dspark-vs-DFlash: a *small* knee means wide verify is
    expensive → dspark's short blocks win (what ``--mode auto`` already picks); a knee that has
    moved out to the DFlash block width (M5-class hardware) means DFlash full-block verify stays
    cheap and re-enters play. Returns the top measured width if no clear knee is found.

    A jump has to clear **two** bars, because either alone misfires on a real curve:

    - *relative*: above 1.8x the mean marginal cost of the cheap region so far. The mean, not
      the running minimum — a minimum collapses to ~0 the moment one step is flat, after which
      every later step trivially "jumps". (Measured Qwen3-4B 8-bit, mlx 0.32, ms by width
      1..8 = 21 22 22 23 24 30 34 36: the flat 22->22 step drove the old baseline to 0 and the
      knee was reported at 4, where the curve is still flat. The real jump is at 6.)
    - *absolute*: at least 15% of the single-row cost. On a nearly-flat cheap region the
      deltas are +-1 ms of measurement noise, and a ratio test on noise is meaningless.

    Reporting only — cap selection reads the curves directly (see :meth:`CapController.rate`),
    so this never changes what gets drafted, only what is shown and recommended.
    """
    ks = sorted(verify_ms)
    if len(ks) < 3:
        return ks[-1] if ks else 0
    deltas = [(ks[i], verify_ms[ks[i]] - verify_ms[ks[i - 1]]) for i in range(1, len(ks))]
    floor = 0.15 * verify_ms[ks[0]]          # noise gate, scaled to the model's own cost

    # The very first step can itself be the knee — that is exactly the bf16 case, where an
    # unquantized matmul reads the full weight stream twice from width 2 and then stays flat
    # (Ornith-9B: 67.8 -> 134.4 -> ~139 ms). It has no cheap region *below* it to compare
    # against, so judge it against the rest of the curve instead. Without this the detector
    # structurally cannot see a cliff at width 2, and reports the top width instead.
    if len(deltas) >= 2:
        rest = [d for _, d in deltas[1:]]
        tail_mean = sum(rest) / len(rest)
        if deltas[0][1] > max(1.8 * tail_mean, floor):
            return deltas[0][0]

    seen = [deltas[0][1]]
    for w, d in deltas[1:]:
        base = sum(seen) / len(seen)
        if d > max(1.8 * base, floor):       # marginal cost left the cheap region
            return w
        seen.append(d)
    return ks[-1]


def drafter_recommendation(verify_ms: dict[int, float], dflash_block: int = 16) -> dict:
    """Hardware-aware dspark-vs-DFlash signal from the measured verify curve. If the qmm knee is
    below the DFlash block width, DFlash's wide-block verify is in the expensive regime → dspark
    wins (the measured M-series answer); if the knee has moved out past it, DFlash full-block
    re-enters play on structured content. Purely a recommendation surfaced to the user."""
    knee = knee_width(verify_ms)
    dflash_viable = knee >= dflash_block
    return {"knee_width": knee, "dflash_full_block_viable": dflash_viable,
            "recommend": "dflash-on-structured" if dflash_viable else "dspark"}


def _interp(curve: dict[int, float], x: int) -> float:
    """Piecewise-linear interpolation over the measured widths (extrapolates the last
    segment's slope above the top measured width)."""
    ks = sorted(curve)
    if x in curve:
        return curve[x]
    if x <= ks[0]:
        return curve[ks[0]]
    if x >= ks[-1]:
        if len(ks) == 1:
            return curve[ks[-1]]
        slope = (curve[ks[-1]] - curve[ks[-2]]) / (ks[-1] - ks[-2])
        return curve[ks[-1]] + slope * (x - ks[-1])
    for lo, hi in itertools.pairwise(ks):
        if lo < x < hi:
            f = (x - lo) / (hi - lo)
            return curve[lo] + f * (curve[hi] - curve[lo])
    return curve[ks[-1]]  # unreachable


# --------------------------------------------------------------------------- controller


class CapController:
    """Live cap picker: maximizes expected committed tokens per unit round time.

    ``update(accepted_n, cap_used)`` feeds the per-position acceptance EWMA; ``cap`` is the
    current choice. Thread-unsafe by design — generation is already serialized (one engine
    thread), and a per-generation instance is fine too.
    """

    def __init__(self, verify_ms: dict[int, float], drafter_ms: dict[int, float] | float,
                 max_cap: int, *, init_cap: int = 2, prior_p: float = 0.65,
                 alpha: float = 0.02, hysteresis: float = 1.03, repick_every: int = 4,
                 verify_grid: dict[int, dict[int, float]] | None = None,
                 allow_zero: bool = False, hybrid_replay: bool = False,
                 probe_every: int = 8, overhead_ms: float = 0.0):
        self.verify_ms = {int(k): float(v) for k, v in verify_ms.items()}
        self.drafter_ms = drafter_ms
        self.max_cap = max(1, int(max_cap))
        self.cap = max(1, min(init_cap, self.max_cap))
        self.p = prior_p
        self.alpha = alpha
        self.hysteresis = hysteresis
        self.repick_every = max(1, repick_every)
        self.rounds = 0
        self.verify_grid = ({int(B): {int(k): float(v) for k, v in row.items()}
                             for B, row in verify_grid.items()} if verify_grid else None)
        # cap 0 = "don't speculate this round" (a plain committed step). Only offered when
        # the driving loop supports it (the dspark loop does); it wins whenever live
        # acceptance is too low for the verify slope — e.g. open chat on a 2-bit target,
        # where every extra verify row costs ~40% of a full step and speculation is a net
        # loss no positive cap can fix. While parked at 0, every ``probe_every``-th round
        # runs at cap 1 so the acceptance estimate keeps tracking content shifts.
        # KEEP THE CADENCE FIXED: probes are the controller's ONLY acceptance/round-time
        # signal while parked, so they are load-bearing for un-parking, not just overhead.
        # An exponential probe backoff (8→64 on failed probes) was measured 2026-07-16:
        # neutral on steady chat (a probe commits ~p·cap extra tokens — near break-even
        # at p≈0.75) but it starved p/_corr recovery on content shifts and wedged the
        # canonical chat→code→math sweep parked (23.4 vs 27.4 tok/s). Reverted; see NOTES.
        self.allow_zero = bool(allow_zero)
        self.probe_every = max(2, int(probe_every))
        self._best = self.cap
        # Bernoulli observations behind the acceptance estimate. Early on the EWMA runs
        # as a plain running mean (effective alpha 1/n), so p converges in a handful of
        # rounds instead of dragging the prior for hundreds; and parking (cap 0) is only
        # offered once enough real observations back the estimate — otherwise a
        # pessimistic prior parks a perfectly good drafter at round 4 (measured: code
        # acceptance 0.93 got parked by the 0.65 prior before any data came in).
        self._obs = 0
        self.min_obs_to_park = 24
        # Observed round timings, fed by the loop via update(..., round_ms=, committed=).
        # The synthetic cost model can't see everything (replay cascades on hybrid
        # targets, per-round sync/Python, content-dependent effects), so measured
        # ms-per-committed-token at a cap OVERRIDES the model once that cap has enough
        # rounds; caps without data use model × a live global correction (observed vs
        # modeled at whatever cap is running). Cap 0's model (one plain step) is exact,
        # so the correction applies to speculative caps only. Stale entries expire so a
        # content shift can't pin an old measurement forever.
        self._t_obs: dict[int, float] = {}     # cap -> EWMA ms per committed token
        self._t_cnt: dict[int, int] = {}       # cap -> rounds observed
        self._t_age: dict[int, int] = {}       # cap -> rounds since last observation
        self._corr = 1.0
        self.t_alpha = 0.1
        self.min_rounds_observed = 4
        self.obs_max_age = 32
        # LEGACY pricing knob, no longer wired by calibrate(): it modeled the old hybrid
        # design where a partially-accepted round re-committed [anchor + accepted] inside
        # the next forward (one full-model row each). Target's capture-and-rerun rollback
        # made hybrid rejects ~free (a few ms of recurrence kernels), so pricing this
        # would over-penalize high caps. Kept for cost-model experiments/tests.
        self.hybrid_replay = bool(hybrid_replay)
        # measured fixed per-round cost the curves miss (sync/graph-build/markov/ctx —
        # see measure_dspark_round_overhead). Charged to speculative caps only: cap-0
        # rounds skip the drafter graph entirely, and biasing the comparison slightly
        # toward parking is the conservative direction (its downside is bounded).
        self.overhead_ms = float(overhead_ms)

    # -- cost model --
    def _draft_cost(self, cap: int) -> float:
        if isinstance(self.drafter_ms, dict):
            return _interp({int(k): float(v) for k, v in self.drafter_ms.items()}, cap)
        return float(self.drafter_ms)

    def expected_committed(self, cap: int) -> float:
        """1 bonus/replacement token + geometric accepted prefix.

        A per-position variant of this (chaining separately-tracked p_i instead of one
        scalar) was implemented and measured 2026-07-22, and **reverted** — it modelled
        reality better yet performed worse. See NOTES "Knee detection"; don't re-derive it.
        """
        return 1.0 + sum(self.p ** i for i in range(1, cap + 1))

    def _replay_ms(self, cap: int) -> float:
        """Expected extra next-round cost a hybrid target pays for this round's partial
        acceptance: [anchor + accepted] rows re-commit at the verify curve's marginal
        per-row slope. Zero for dense targets (their rollback is a free KV trim)."""
        if not self.hybrid_replay or cap == 0:
            return 0.0
        # E[(1 + n) · 1{partial}] under the geometric acceptance model
        exp_rows = sum((1 + i) * (self.p ** i) * (1.0 - self.p) for i in range(cap))
        slope = _interp(self.verify_ms, cap + 2) - _interp(self.verify_ms, cap + 1)
        return exp_rows * max(slope, 0.0)

    def rate(self, cap: int) -> float:
        """Expected committed tokens per ms of round time at this cap. Cap 0 is a plain
        step: one committed token for one width-1 forward, no drafter."""
        if cap == 0:
            return 1.0 / max(_interp(self.verify_ms, 1), 1e-6)
        t = (self._draft_cost(cap) + _interp(self.verify_ms, cap + 1)
             + self._replay_ms(cap) + self.overhead_ms)
        return self.expected_committed(cap) / max(t, 1e-6)

    def _rate_eff(self, cap: int) -> float:
        """Best available estimate of committed tokens/ms at this cap: measured when this
        cap has enough observed rounds, else the model (correction-scaled for cap >= 1).

        Shrinking the measured rate toward the model by observation count (instead of this
        hard switch) was implemented and measured 2026-07-22, and **reverted** together
        with the per-position acceptance model — see NOTES "Knee detection"."""
        if self._t_cnt.get(cap, 0) >= self.min_rounds_observed:
            return 1.0 / max(self._t_obs[cap], 1e-6)
        return self.rate(cap) * (self._corr if cap >= 1 else 1.0)

    # -- live updates --
    def update(self, accepted_n: int, cap_used: int, *, round_ms: float | None = None,
               committed: int | None = None) -> None:
        """Feed one round's outcome: ``accepted_n`` drafted tokens survived out of
        ``cap_used``. Full acceptance is censored (no failure observed); cap-0 rounds
        (``cap_used == 0``) carry no acceptance signal but still advance the round
        counter that schedules probes and repicks. ``round_ms``/``committed`` (optional)
        feed the observed-timing layer that grounds the cost model in reality."""
        for _ in range(accepted_n):
            self._bump(1.0)
        if accepted_n < cap_used:
            self._bump(0.0)
        if round_ms is not None and committed and round_ms > 0:
            tpt = float(round_ms) / committed
            prev = self._t_obs.get(cap_used)
            self._t_obs[cap_used] = (tpt if prev is None
                                     else prev + self.t_alpha * (tpt - prev))
            self._t_cnt[cap_used] = self._t_cnt.get(cap_used, 0) + 1
            self._t_age[cap_used] = 0
            if cap_used >= 1:
                inst = (1.0 / tpt) / max(self.rate(cap_used), 1e-6)
                self._corr += 0.05 * (inst - self._corr)
        for c in list(self._t_age):
            if c != cap_used:
                self._t_age[c] += 1
                if self._t_age[c] > self.obs_max_age:     # stale: let the model take over
                    self._t_obs.pop(c, None)
                    self._t_cnt.pop(c, None)
                    self._t_age.pop(c, None)
        self.rounds += 1
        if self.rounds % self.repick_every == 0:
            self._repick()
        if self._best == 0:
            # parked: schedule a cap-1 probe round so acceptance keeps tracking
            self.cap = 1 if self.rounds % self.probe_every == 0 else 0

    def _bump(self, x: float) -> None:
        """One Bernoulli observation into the acceptance estimate: running mean while
        young (alpha 1/n), settling into the standard EWMA."""
        self._obs += 1
        a = max(self.alpha, 1.0 / self._obs)
        self.p += a * (x - self.p)

    def _repick(self) -> None:
        lo = 0 if (self.allow_zero and self._obs >= self.min_obs_to_park) else 1
        best = max(range(lo, self.max_cap + 1), key=self._rate_eff)
        if best != self._best and self._rate_eff(best) > self._rate_eff(self._best) * self.hysteresis:
            self._best = best
            self.cap = best

    # -- batched operating point --
    def cap_for(self, batch_width: int) -> int:
        """Best cap when B rows verify together: the (B, cap) grid's argmax under the current
        acceptance estimate. B multiplies into the same qmm-knee arithmetic as the verify width,
        so the optimal cap usually *shrinks* as the batch widens. Uses the nearest measured
        B ≥ batch_width; falls back to the live single-stream cap when no grid was measured.
        (Drafter cost uses the B=1 curve — a slight underestimate at high B, conservative since
        verify dominates round time.)"""
        if batch_width <= 1 or not self.verify_grid:
            return self.cap
        bs = sorted(self.verify_grid)
        b_key = next((b for b in bs if b >= batch_width), bs[-1])
        curve = self.verify_grid[b_key]

        def rate_b(c: int) -> float:
            t = self._draft_cost(c) + _interp(curve, c + 1) + self.overhead_ms
            return self.expected_committed(c) / max(t, 1e-6)

        return max(range(1, self.max_cap + 1), key=rate_b)

    def static_best(self, p: float | None = None) -> int:
        """The single best FIXED cap for this machine+model+quant, straight off the measured
        curves. This is the default-cap resolver — see :func:`static_cap`."""
        if p is not None:
            saved, self.p = self.p, float(p)
            try:
                return max(range(1, self.max_cap + 1), key=self.rate)
            finally:
                self.p = saved
        return max(range(1, self.max_cap + 1), key=self.rate)

    def info(self) -> dict:
        lo = 0 if self.allow_zero else 1
        out = {"cap": self.cap, "p": round(self.p, 3), "rounds": self.rounds,
               "knee_width": knee_width(self.verify_ms),  # qmm knee → dspark-vs-dflash signal
               "predicted_rates": {c: round(self.rate(c) * 1e3, 1)   # tok/s at each cap
                                   for c in range(lo, self.max_cap + 1)}}
        if self.verify_grid:
            out["batch_caps"] = {b: self.cap_for(b) for b in sorted(self.verify_grid)}
        return out


# --------------------------------------------------------------------------- disk cache


def _cache_path(cache_dir: str | None = None) -> str:
    return os.path.join(cache_dir or CACHE_DIR, CACHE_FILE)


def _cache_key(mode: str, target_repo: str, drafter_repo: str | None,
               ctx_len: int = CTX_LEN, kv_bits: int | None = None) -> str:
    dev = mx.device_info().get("device_name", "unknown")
    tgt = os.path.basename(str(target_repo).rstrip("/"))
    drf = os.path.basename(str(drafter_repo).rstrip("/")) if drafter_repo else "-"
    kv = f"|kv{kv_bits}" if kv_bits else ""     # quantized KV changes the verify curve
    return f"v{SCHEMA}|{dev}|mlx{mx.__version__}|{mode}|{tgt}|{drf}|ctx{ctx_len}{kv}"


def load_cached(key: str, cache_dir: str | None = None) -> dict | None:
    try:
        with open(_cache_path(cache_dir)) as f:
            return json.load(f).get(key)
    except (OSError, json.JSONDecodeError):
        return None


def save_cached(key: str, entry: dict, cache_dir: str | None = None) -> None:
    path = _cache_path(cache_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    data[key] = entry
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


def wide_gemm_min_rows(target, drafter=None, *, target_repo: str,
                       cache_dir: str | None = None,
                       verbose: bool = True) -> tuple[int | None, list[str]]:
    """Row count from which prefill should route ``QuantizedLinear`` through
    dequantize-once + GEMM instead of ``quantized_matmul`` (see :mod:`.wide_gemm`), or
    ``None`` when that never pays on this machine.

    Same doctrine as the cap: the crossover is a property of
    (chip x mlx version x weight shape), so it is measured once (~1-2 s) and cached to disk
    under a key that includes both the device and the mlx version — a hardcoded threshold
    would be stale the day a chip or a kernel changes."""
    from .wide_gemm import measure_crossover

    key = _cache_key("widegemm", target_repo, None)
    entry = load_cached(key, cache_dir)
    if entry is None:
        if verbose:
            print("calibrating prefill wide-GEMM crossover (one-time, cached)…", flush=True)
        # the drafter's context projections run at prefill width too, so its shapes are
        # part of the bit-identity check
        got = measure_crossover(target.model, drafter)
        entry = ({"min_rows": got[0], "shapes": got[1]} if got
                 else {"min_rows": None, "shapes": []})
        save_cached(key, entry, cache_dir)
        if verbose:
            mr, sh = entry["min_rows"], entry["shapes"]
            state = (f"on from M={mr} for {len(sh)} weight shapes (verified bit-identical)"
                     if mr else "off (no gain, or not exact on this machine)")
            print(f"  prefill wide-GEMM: {state}", flush=True)
    return entry.get("min_rows"), entry.get("shapes")


def apply_wide_gemm(target, drafter=None, *, target_repo: str,
                    min_rows: int | None = None, verbose: bool = True) -> int | None:
    """Turn the prefill wide-GEMM path on for this process and return the threshold used.

    ``min_rows=None`` calibrates (and caches); ``0`` disables; ``N`` forces N. Sets the
    process-wide ``generate.WIDE_GEMM_MIN_ROWS`` rather than a library default, because a
    plain ``speculative_generate`` call must never silently trigger a device benchmark."""
    from . import generate as _gen
    from .wide_gemm import verified_shapes

    if min_rows is None:
        _gen.WIDE_GEMM_MIN_ROWS, _gen.WIDE_GEMM_SHAPES = wide_gemm_min_rows(
            target, drafter, target_repo=target_repo, verbose=verbose)
    else:
        # an explicit threshold still only applies to shapes verified at it
        _gen.WIDE_GEMM_MIN_ROWS = int(min_rows) or None
        _gen.WIDE_GEMM_SHAPES = (
            verified_shapes([target.model, drafter], _gen.WIDE_GEMM_MIN_ROWS)
            if _gen.WIDE_GEMM_MIN_ROWS else [])
    return _gen.WIDE_GEMM_MIN_ROWS


# --------------------------------------------------------------------------- entry point


def calibrate(target, drafter, *, mode: str, target_repo: str, drafter_repo: str | None,
              cache_dir: str | None = None, verbose: bool = True,
              batch_widths: list[int] | None = None) -> CapController:
    """Measure (or load from the disk cache) this machine+pair's cost curves and return a
    ready :class:`CapController`. ``mode`` is ``"dspark"`` or ``"dflash"``. ``batch_widths``
    (e.g. ``[2, 4]`` when serving with ``--max-batch 4``) additionally measures the batched
    verify grid so :meth:`CapController.cap_for` can pick a per-batch-width cap."""
    cfg = drafter.config
    # A has_lm_head=false drafter (Nemotron DSpark head) reuses the target's lm_head; the
    # drafter-curve measurement below calls compute_logits directly, so bind it here too
    # (speculative_generate binds independently at generation time).
    if not getattr(cfg, "has_own_lm_head", True) and getattr(drafter, "_ext_lm_head", None) is None:
        drafter.bind_lm_head(target.lm_head_proj)
    if not getattr(cfg, "has_own_embed", True) and getattr(drafter, "_ext_embed", None) is None:
        drafter.bind_embed(target.draft_embed)
    if mode == "dspark":
        max_cap = int(cfg.block_size)
        widths = list(range(1, max_cap + 2))     # verify width = cap + 1; width 1 = the
        caps = list(range(1, max_cap + 1))       # plain step, needed to price cap 0
    elif mode == "dflash":
        max_cap = int(cfg.block_size) - 1
        widths = sorted({2, 3, 4, 5, 7, 9, 12, max_cap + 1})    # sample + interpolate
        caps = []
    else:
        raise ValueError(f"auto cap supports dspark/dflash, not {mode!r}")

    key = _cache_key(mode, target_repo, drafter_repo,
                     kv_bits=getattr(target, "kv_bits", None))
    entry = load_cached(key, cache_dir)
    if entry is None:
        if verbose:
            print(f"calibrating {mode} cap for this machine (one-time, cached)…", flush=True)
        tap = list(cfg.target_layer_ids)
        verify = measure_verify_curve(target, tap, widths)
        if mode == "dspark":
            drafter_ms: dict | float = measure_dspark_drafter_curve(drafter, caps)
            overhead = measure_dspark_round_overhead(target, drafter, verify, drafter_ms)
        else:
            drafter_ms = measure_dflash_drafter_cost(drafter)
            overhead = 0.0
        entry = {"verify": {str(k): v for k, v in verify.items()},
                 "drafter": ({str(k): v for k, v in drafter_ms.items()}
                             if isinstance(drafter_ms, dict) else drafter_ms),
                 "overhead": overhead}
        save_cached(key, entry, cache_dir)
        if verbose:
            vs = " ".join(f"{k}:{v:.0f}" for k, v in sorted(verify.items()))
            print(f"  verify ms by width: {vs}", flush=True)
            rec = drafter_recommendation(verify)
            print(f"  qmm knee at width {rec['knee_width']} -> "
                  f"{'DFlash full-block viable' if rec['dflash_full_block_viable'] else 'dspark wins'} "
                  f"(dspark-vs-dflash signal)", flush=True)

    # batched verify grid, measured on demand (also backfills an older cached entry)
    want_bs = sorted({int(b) for b in (batch_widths or []) if int(b) > 1})
    have_bs = sorted(int(b) for b in (entry.get("verify_grid") or {}))
    if mode == "dspark" and want_bs and any(b not in have_bs for b in want_bs):
        if verbose:
            print(f"calibrating batched verify grid (B={want_bs}, one-time, cached)…",
                  flush=True)
        grid = measure_batch_verify_grid(target, list(cfg.target_layer_ids), want_bs, widths)
        vg = entry.setdefault("verify_grid", {})
        for B, row in grid.items():
            vg[str(B)] = {str(k): v for k, v in row.items()}
        save_cached(key, entry, cache_dir)

    verify = {int(k): float(v) for k, v in entry["verify"].items()}
    drafter_ms = (entry["drafter"] if not isinstance(entry["drafter"], dict)
                  else {int(k): float(v) for k, v in entry["drafter"].items()})
    ctrl = CapController(verify, drafter_ms, max_cap, init_cap=2,
                         verify_grid=entry.get("verify_grid"),
                         # the dspark loop supports cap-0 (skip drafting) rounds.
                         # hybrid_replay stays False: since the capture-and-rerun
                         # rollback (Target.verify/rollback), a hybrid reject costs ~a
                         # few ms of recurrence kernels, not a full-model row per
                         # accepted token — pricing the old replay would over-penalize
                         # high caps (and observed round timings ground the rest).
                         allow_zero=(mode == "dspark"),
                         overhead_ms=float(entry.get("overhead", 0.0)))
    if verbose:
        r = ctrl.info()["predicted_rates"]
        best = max(r, key=r.get)
        print(f"  predicted tok/s by cap (prior accept): {r} -> starting cap 2, "
              f"knee-best {best}", flush=True)
    return ctrl


# Acceptance prior used to pick the static default cap. Deliberately NOT the controller's
# `prior_p` (0.65): that one guards the cap-0 parking gate, where being pessimistic is the
# safe direction, whereas here pessimism picks a cap that is too shallow. Fitted against the
# measured optima of all 8 curves in the calibration cache (2026-07-22): 0.65 scores 5/8 and
# notably picks cap 1 for Bonsai (measured best 2) and 6 for Ornith-bf16 (7); 0.70 scores 7/8,
# with the only miss Ornith-4bit (picks 2, measured 3 — adjacent caps, inside noise there).
# Re-fit this if the measured-optima table in NOTES "Knee detection" is ever regenerated.
STATIC_PRIOR_P = 0.70


def static_cap(target, drafter, *, mode: str, target_repo: str, drafter_repo: str | None,
               cache_dir: str | None = None, verbose: bool = False,
               fallback: int = 2) -> int:
    """The default cap for this machine+model+quant: the cost model's argmax over the
    MEASURED curves. One-time on-device measurement (~5 s), disk-cached thereafter.

    **Why this is not a constant.** The optimal cap is a property of
    (model x QUANTIZATION x chip x mlx version), not of the model family. Measured
    2026-07-22 on one M4 Pro under mlx 0.32, the best cap spans **2 to 7** across the
    cached curves, and the *same registry row* needs different caps per quantization —
    Ornith-1.0-9B wants cap 2 at 4-bit, 4 at 8-bit, 6 at bf16 — while drafter resolution
    is deliberately quant-agnostic, so no per-entry constant can express it. Nor is a
    knee-derived rule enough: bf16 has a 2x cliff at width 2 and is then FLAT, so the
    right move there is to go *wide* once the cliff is paid. What generalizes is the
    argmax itself, which reproduced all 7 documented optima. The old hardcoded 2 was
    measured when mlx 0.31.2 put the knee at width 4; 0.32 moved it to 6 for every 8-bit
    family, leaving 10-32% on the table. It will drift again — that is the point.

    Returns ``fallback`` if the curves cannot be measured (unsupported mode, no drafter),
    so a failure here degrades to the historical default rather than breaking generation.
    """
    if mode not in ("dspark", "dflash") or drafter is None:
        return fallback
    try:
        ctrl = calibrate(target, drafter, mode=mode, target_repo=target_repo,
                         drafter_repo=drafter_repo, cache_dir=cache_dir, verbose=verbose)
    except Exception:  # noqa: BLE001 — calibration is an optimization, never a gate
        return fallback
    return ctrl.static_best(STATIC_PRIOR_P)
