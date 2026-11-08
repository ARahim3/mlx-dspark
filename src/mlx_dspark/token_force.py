"""Token forcing: a lossless gate that stays informative past the first fp tie.

The lossless contract (CLAUDE.md rule 2) is that every committed token is the target's
greedy choice, fp-tie flips at near-zero logit margin excepted. The old check, "firstdiff"
(decode the same prompt with spec decoding and with plain greedy, compare the streams),
goes blind at the first tie: after one legitimate flip the two streams condition on
different prefixes, and nothing after that point can be compared.

Token forcing (SpecPilot, arXiv 2611.02501 §3.3) asks a better question of each stream on
its OWN terms: run one teacher-forced target forward over ``prompt + output`` and, at every
output position, compare the committed token with the target's argmax there. Because the
forward conditions on exactly the prefix the generator committed, a flip at position i says
nothing about position i+1. Every position is scored, so a bug (a wrong rollback, KV
corruption, an off-by-one in an accept path) shows up as a mismatch wherever it happens,
not only if it happens before the first tie.

What a mismatch means depends on its margin, ``logit[argmax] - logit[committed]``:

- ``margin <= delta`` (default 0.25 nats): a **near-tie**. Kernel rounding differs between
  the generator's forward (width-1 decode, width-K verify, the small-M kernel ...) and the
  teacher-forced one (a chunked GEMM), so near-ties are expected from any correct system.
- ``margin > 2*delta``: a **large-margin event**. That is not rounding; a correct system
  should essentially never produce one.
- in between: counted in the rate, reported, neither class.

The gate compares a candidate system with a reference ("stock") run on the same prompts, so
that whatever rate of near-ties the hardware/kernels produce is calibrated out:

- **rate rule** -- reject if the candidate's mismatch rate exceeds the reference's by more
  than ``tolerance`` (0.5 percentage points) with 95% confidence: :func:`rate_rule`.
- **event rule** -- reject if the candidate has significantly more large-margin events than
  the reference: :func:`event_rule` (with zero reference events and equal exposure, 3
  candidate events fail -- the rule of three).

The statistics are pure functions over counts (model-free, unit-testable). The one function
that touches a model is :func:`score_forced`; it uses the stock forward
(:meth:`Target.prefill`) outside every scoped kernel patch, so it is a fixed reference for
any candidate. ``scripts/token_force.py`` is the driver.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from statistics import NormalDist

import mlx.core as mx

DEFAULT_DELTA = 0.25      # nats: a mismatch at or under this margin is a near-tie
RATE_TOLERANCE = 0.005    # the rate rule's allowance: 0.5 percentage points
ALPHA = 0.05              # one-sided: both rules reject at 95% confidence
SCORE_CHUNK = 256         # rows per teacher-forced scoring forward. Each one materializes a
#                           [chunk, vocab] logits block (127 MB at 248k vocab in bf16) that is
#                           reduced to argmax/margin per row immediately and dropped.


# --------------------------------------------------------------------------- scoring


@dataclass
class ForcedScores:
    """Teacher-forced verdict for ONE generation: per output position i, the committed
    token, the target's argmax at that position, and the fp32 margin
    ``logit[argmax] - logit[committed]`` (exactly 0.0 where they agree)."""

    committed: list[int]
    argmax: list[int]
    margin: list[float]

    def __post_init__(self):
        if not (len(self.committed) == len(self.argmax) == len(self.margin)):
            raise ValueError("committed / argmax / margin must have the same length")

    def __len__(self) -> int:
        return len(self.committed)

    def mismatch_positions(self) -> list[int]:
        return [i for i, (c, a) in enumerate(zip(self.committed, self.argmax)) if c != a]


def _reduce_rows(logits: mx.array, committed: mx.array) -> tuple[mx.array, mx.array]:
    """``[C, V]`` logits + ``[C]`` committed ids -> (argmax ``[C]``, fp32 margin ``[C]``).
    The two gathered logits are upcast before subtracting (exact: bf16/fp16 -> fp32 is
    lossless), so the [C, V] block itself never needs an fp32 copy."""
    am = mx.argmax(logits, axis=-1)
    top = mx.take_along_axis(logits, am[:, None], axis=-1)[:, 0]
    got = mx.take_along_axis(logits, committed[:, None], axis=-1)[:, 0]
    return am, top.astype(mx.float32) - got.astype(mx.float32)


def score_forced(target, prompt_ids: Sequence[int], output_ids: Sequence[int], *,
                 chunk: int = SCORE_CHUNK, prefill_chunk: int | None = None) -> ForcedScores:
    """Teacher-force ``output_ids`` through ``target`` (a :class:`~mlx_dspark.target.Target`)
    after ``prompt_ids`` and score every output position.

    Output token i is predicted by the row at absolute position ``P - 1 + i`` (P = prompt
    length), so the forward runs over ``prompt + output[:-1]``:

    - rows ``[0, P-1)`` are only context: they go through the body-only prefill
      (``want_logits=False``), chunked at ``prefill_chunk`` (default
      ``generate.PREFILL_CHUNK``), which skips the vocab projection entirely;
    - rows ``[P-1, P-1+L)`` are scored ``chunk`` at a time with the full head
      (``head_last_row=False``); each ``[chunk, V]`` logits block is reduced to argmax and
      margin per row and freed before the next chunk runs, so peak memory is one block, not
      ``[L, V]`` (248k vocab makes that ~1 GB per 1k fp32 rows).

    A fresh cache is used; ``target`` is left untouched. Runs outside every scoped kernel
    patch (small-M, SDPA split, multi-row, wide-GEMM, CPU co-prefill), i.e. on mlx's stock
    kernels, whatever the process has applied for generation."""
    from .generate import PREFILL_CHUNK, _cache_arrays

    P, L = len(prompt_ids), len(output_ids)
    if P == 0:
        raise ValueError("token forcing needs at least one prompt token (the row that "
                         "predicts the first output token)")
    if L == 0:
        return ForcedScores([], [], [])
    if chunk < 1:
        raise ValueError("chunk must be >= 1")
    pc = prefill_chunk or PREFILL_CHUNK
    seq = [int(t) for t in prompt_ids] + [int(t) for t in output_ids[:-1]]
    out = [int(t) for t in output_ids]
    cache = target.make_cache()

    for i in range(0, P - 1, pc):                         # context-only rows: no head
        target.prefill(mx.array([seq[i:min(i + pc, P - 1)]]), cache, None,
                       want_logits=False)
        mx.eval(_cache_arrays(cache))
        mx.clear_cache()

    argmax: list[int] = []
    margin: list[float] = []
    for j in range(0, L, chunk):
        rows = seq[P - 1 + j:P - 1 + min(j + chunk, L)]
        logits, _ = target.prefill(mx.array([rows]), cache, None,
                                   want_logits=True, head_last_row=False)
        vocab = logits.shape[-1]
        want = out[j:j + len(rows)]
        if max(want) >= vocab or min(want) < 0:
            raise ValueError(f"committed token id out of range for vocab {vocab}")
        am, mg = _reduce_rows(logits[0], mx.array(want, dtype=mx.int32))
        mx.eval(am, mg, _cache_arrays(cache))
        argmax.extend(int(a) for a in am.tolist())
        margin.extend(float(m) for m in mg.tolist())
        del logits, am, mg
        mx.clear_cache()
    return ForcedScores(out, argmax, margin)


# --------------------------------------------------------------------------- statistics


@dataclass(frozen=True)
class Mismatch:
    """One position where the committed token is not the teacher-forced argmax.
    ``sample`` indexes the generation in a pooled set; ``position`` is the output index
    (0 = first generated token)."""

    sample: int
    position: int
    committed: int
    argmax: int
    margin: float


@dataclass
class ForceStats:
    """Pooled token-forcing statistics for one system over a prompt set."""

    positions: int
    mismatches: list[Mismatch] = field(default_factory=list)
    delta: float = DEFAULT_DELTA

    @property
    def n_mismatches(self) -> int:
        return len(self.mismatches)

    @property
    def mismatch_rate(self) -> float:
        return self.n_mismatches / self.positions if self.positions else 0.0

    @property
    def near_ties(self) -> int:
        """Mismatches with margin <= delta (expected kernel-rounding flips)."""
        return sum(m.margin <= self.delta for m in self.mismatches)

    @property
    def large_events(self) -> int:
        """Mismatches with margin > 2*delta (not explainable by rounding)."""
        return sum(m.margin > 2 * self.delta for m in self.mismatches)

    @property
    def gray(self) -> int:
        """Mismatches with delta < margin <= 2*delta: in the rate, neither class."""
        return self.n_mismatches - self.near_ties - self.large_events

    @property
    def max_margin(self) -> float:
        return max((m.margin for m in self.mismatches), default=0.0)

    def to_dict(self) -> dict:
        return {"positions": self.positions, "mismatches": self.n_mismatches,
                "mismatch_rate": self.mismatch_rate, "delta": self.delta,
                "near_ties": self.near_ties, "gray": self.gray,
                "large_events": self.large_events, "max_margin": self.max_margin,
                "list": [asdict(m) for m in self.mismatches]}


def force_stats(scores: ForcedScores | Sequence[ForcedScores], *,
                delta: float = DEFAULT_DELTA) -> ForceStats:
    """Pool one or many :class:`ForcedScores` (one per prompt) into :class:`ForceStats`.
    A single ``ForcedScores`` is the single-system sanity check: rate, max margin, and the
    mismatch list."""
    if delta <= 0:
        raise ValueError("delta must be > 0")
    if isinstance(scores, ForcedScores):
        scores = [scores]
    mism: list[Mismatch] = []
    n = 0
    for s_idx, s in enumerate(scores):
        n += len(s)
        for i in s.mismatch_positions():
            mism.append(Mismatch(s_idx, i, s.committed[i], s.argmax[i], float(s.margin[i])))
    return ForceStats(n, mism, float(delta))


# --------------------------------------------------------------------------- the gate


@dataclass(frozen=True)
class RuleResult:
    """One gate rule. Both rules share the shape "fail iff ``observed > threshold``"."""

    name: str
    passed: bool
    observed: float
    threshold: float
    detail: str


def rate_rule(k_cand: int, n_cand: int, k_ref: int, n_ref: int, *,
              tolerance: float = RATE_TOLERANCE, alpha: float = ALPHA) -> RuleResult:
    """Reject iff the candidate's mismatch rate exceeds the reference's by more than
    ``tolerance`` with ``1 - alpha`` confidence (SpecPilot §3.3's rate rule).

    One-sided two-proportion z-test of H0: ``p_c - p_r <= tolerance`` against
    H1: ``p_c - p_r > tolerance`` (a non-inferiority test with margin ``tolerance``), using
    the UNPOOLED Wald standard error ``sqrt(p_c(1-p_c)/n_c + p_r(1-p_r)/n_r)`` -- the
    pooled SE is only valid under H0 ``p_c == p_r``, which is not this H0. Equivalently:
    fail iff the one-sided lower confidence bound ``(p_c - p_r) - z_{1-alpha}*SE`` exceeds
    ``tolerance``; that bound is reported as ``observed``. When SE is 0 (each rate is 0 or
    1) there is no sampling noise left and the rule reduces to ``p_c - p_r > tolerance``.
    Wald is anti-conservative at small counts, i.e. it errs toward rejecting -- the safe
    side for a correctness gate."""
    if n_cand <= 0 or n_ref <= 0:
        raise ValueError("rate rule needs scored positions in both systems")
    if not (0 <= k_cand <= n_cand and 0 <= k_ref <= n_ref):
        raise ValueError("mismatch counts must lie in [0, positions]")
    pc, pr = k_cand / n_cand, k_ref / n_ref
    diff = pc - pr
    se = math.sqrt(pc * (1 - pc) / n_cand + pr * (1 - pr) / n_ref)
    z_crit = NormalDist().inv_cdf(1 - alpha)
    lower = diff - z_crit * se
    passed = not lower > tolerance
    z = f"z={(diff - tolerance) / se:.2f}" if se > 0 else "no variance"
    detail = (f"rate {pc:.3%} vs ref {pr:.3%} (diff {diff * 100:+.2f}pp, one-sided "
              f"{1 - alpha:.0%} lower bound {lower * 100:+.2f}pp, {z}) vs tolerance "
              f"{tolerance * 100:.2f}pp")
    return RuleResult("rate", passed, lower, tolerance, detail)


def _poisson_cdf(x: int, mu: float) -> float:
    """P(Poisson(mu) <= x), summed in log space (no scipy; counts here are small)."""
    if mu <= 0:
        return 1.0
    logs = [-mu + k * math.log(mu) - math.lgamma(k + 1) for k in range(x + 1)]
    m = max(logs)
    return min(1.0, math.exp(m) * sum(math.exp(v - m) for v in logs))


def poisson_upper(x: int, alpha: float = ALPHA) -> float:
    """Exact (Garwood) one-sided ``1 - alpha`` upper confidence limit for a Poisson mean
    after observing ``x`` events: the ``mu`` with ``P(Poisson(mu) <= x) = alpha``.
    ``x = 0`` gives ``-ln(alpha)`` = 2.996 at 95% (the "rule of three"); 1 -> 4.744,
    2 -> 6.296, 10 -> 16.96."""
    if x < 0:
        raise ValueError("event count must be >= 0")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if x == 0:
        return -math.log(alpha)
    lo, hi = 0.0, float(x) + 10.0 * math.sqrt(x + 1.0) + 10.0
    while _poisson_cdf(x, hi) > alpha:
        hi *= 2.0
    for _ in range(200):                                  # cdf is decreasing in mu
        mid = 0.5 * (lo + hi)
        if _poisson_cdf(x, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def event_rule(x_cand: int, n_cand: int, x_ref: int, n_ref: int, *,
               alpha: float = ALPHA) -> RuleResult:
    """Reject iff the candidate has more large-margin events than the reference's rate can
    plausibly explain (SpecPilot §3.3's event rule), as a one-sided Poisson test:

        allowance = poisson_upper(x_ref, alpha) * min(1, n_cand / n_ref)
        fail iff x_cand > allowance

    i.e. the candidate's count exceeds the exact ``1 - alpha`` upper confidence limit on
    the count the reference's event rate would produce over the candidate's exposure. With
    zero reference events and ``n_cand >= n_ref`` the allowance is ``-ln(0.05)`` = 2.996, so
    **3 candidate events fail** (the rule of three); 1 reference event -> 5 fail; 2 -> 7.

    Why not the exact conditional binomial two-sample test (``x_c ~ Bin(x_c + x_r,
    n_c/(n_c+n_r))``)? At zero reference events and equal exposure it needs 5 candidate
    events to reach p < 0.05 (0.5^5 = 0.031) -- too lenient for an event class a lossless
    system should never produce at all. This rule is stricter exactly where it matters
    (small reference counts) and converges to a normal-ish comparison as counts grow.

    Why ``min(1, ...)``: exposure scales the allowance DOWN when the candidate scored fewer
    positions, but never up -- outputs diverge after a tie and one arm may run longer, and
    a longer candidate must not earn extra room for events that should not exist."""
    if n_cand <= 0 or n_ref <= 0:
        raise ValueError("event rule needs scored positions in both systems")
    if x_cand < 0 or x_ref < 0:
        raise ValueError("event counts must be >= 0")
    allowance = poisson_upper(x_ref, alpha) * min(1.0, n_cand / n_ref)
    passed = not x_cand > allowance
    detail = (f"{x_cand} large-margin events vs ref {x_ref} (allowance {allowance:.2f} = "
              f"{1 - alpha:.0%} Poisson upper limit of the ref count at the candidate's "
              f"exposure)")
    return RuleResult("events", passed, float(x_cand), allowance, detail)


@dataclass(frozen=True)
class GateVerdict:
    passed: bool
    rate: RuleResult
    events: RuleResult

    def to_dict(self) -> dict:
        return {"passed": self.passed, "rate": asdict(self.rate),
                "events": asdict(self.events)}


def gate(candidate: ForceStats, reference: ForceStats, *,
         tolerance: float = RATE_TOLERANCE, alpha: float = ALPHA) -> GateVerdict:
    """Candidate-vs-reference verdict: passes iff BOTH :func:`rate_rule` and
    :func:`event_rule` pass. Both stats must use the same ``delta`` (it defines what a
    large-margin event is)."""
    if candidate.delta != reference.delta:
        raise ValueError(f"delta differs: candidate {candidate.delta} vs reference "
                         f"{reference.delta}")
    r = rate_rule(candidate.n_mismatches, candidate.positions,
                  reference.n_mismatches, reference.positions,
                  tolerance=tolerance, alpha=alpha)
    e = event_rule(candidate.large_events, candidate.positions,
                   reference.large_events, reference.positions, alpha=alpha)
    return GateVerdict(r.passed and e.passed, r, e)
