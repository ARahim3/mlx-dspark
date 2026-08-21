"""Split a wide-verify SDPA into narrow sub-calls to dodge mlx's multi-row cliff.

mlx's `mx.fast.scaled_dot_product_attention` picks a good few-row path for q_len up to
~5, then falls off a **cliff** at q_len 6 (measured on M4 Pro / mlx 0.32.1, Qwen3.8 shape
Hq=24 Hk=4 D=256, L=32k: q5 2.48 ms -> q6 7.09 ms, ~2.9x, flat through q15, tiled path
recovers only at q16). Speculative verify at cap 6-8 lands exactly in that cliff, which is
the whole of our documented cap-7 -> 0.53x-at-32k long-context collapse. A verify width of 4 or less never feels it (a depth-3 native-MTP cap sits at width 4, below the cliff).

Each query row's attention is **independent** (row i attends to all KV with row i's query
only), so splitting the q_len rows into <=`max_chunk`-row sub-calls and concatenating is
equivalent per row — the only difference is bf16 reduction order, i.e. fp-tie class, which
the lossless verify loop already tolerates (the target verifies every token). Measured win
(M4 Pro, L=32k): width 6 ~2.2x, width 7 ~2.0x, width 8 ~1.9x on the attention call;
1.5-1.8x at 16k, 1.5-1.6x at 8k.

This is NOT a custom Metal kernel. It is a scoped monkeypatch of
`mx.fast.scaled_dot_product_attention` (mirroring `small_m_qmm.small_m_matmul`), active only
during generation, that routes calls whose q_len is in the measured cliff window and whose
KV is long enough to matter through the split; every other call (prefill's wide q, q_len-1
decode, the short-context drafter) passes straight through unchanged. The cliff is a
kernel-path-selection artifact of (mlx version x GPU), so the window is **measured**
(`measure_split_window`) and empties to a no-op where no cliff exists — never a regression.

The GQA-group-sharing "v2" idea was measured and rejected: mlx already reads unique KV once
per KV head and broadcasts to the group (Hk=4 is 3.8x faster than Hk=24 at q1), so there is
no redundant DRAM traffic to reclaim. See NOTES "SDPA verify-split".
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass

import mlx.core as mx

# The real kernel, captured at import so the split sub-calls never re-enter the patch.
_ORIG_SDPA = mx.fast.scaled_dot_product_attention


@dataclass(frozen=True)
class SplitConfig:
    """Measured cliff window. Split fires iff ``min_q <= q_len <= max_q`` and
    ``kv_len >= min_kv``. ``max_chunk`` bounds each sub-call's q_len (must be < min_q so a
    chunk never re-triggers the cliff)."""
    min_q: int = 6
    max_q: int = 15
    min_kv: int = 4096
    max_chunk: int = 5


def _partition(q_len: int, max_chunk: int) -> list[int]:
    """Balanced split of ``q_len`` into ceil(q_len/max_chunk) chunks (larger chunks first).
    Balanced is near-optimal for the measured cost curve and dead simple; a per-machine DP
    over the measured cost(q) could shave a few % but isn't worth the complexity yet."""
    n = max(1, math.ceil(q_len / max_chunk))
    base, rem = divmod(q_len, n)
    return [base + 1] * rem + [base] * (n - rem)


def split_sdpa(q, k, v, *, scale, mask=None, sinks=None, cfg: SplitConfig, _orig=None, **kw):
    """Equivalent to ``scaled_dot_product_attention`` but computes the q rows in
    <=cfg.max_chunk-row sub-calls when the (q_len, kv_len) gate says the cliff bites.
    fp-tie class (reduction-order only). Falls back to a single call otherwise."""
    orig = _orig or _ORIG_SDPA
    q_len = q.shape[-2]
    kv_len = k.shape[-2]
    if not (cfg.min_q <= q_len <= cfg.max_q and kv_len >= cfg.min_kv):
        return orig(q, k, v, scale=scale, mask=mask, sinks=sinks, **kw)

    is_str_mask = isinstance(mask, str)          # "causal"
    is_arr_mask = isinstance(mask, mx.array)
    if is_arr_mask:
        mq = mask.shape[-2]
        if mq not in (1, q_len):                 # unexpected shape -> don't risk it
            return orig(q, k, v, scale=scale, mask=mask, sinks=sinks, **kw)

    outs = []
    off = 0
    for p in _partition(q_len, cfg.max_chunk):
        qc = q[..., off:off + p, :]
        if mask is None:
            oc = orig(qc, k, v, scale=scale, mask=None, sinks=sinks, **kw)
        elif is_str_mask:                        # causal: the p chunk rows are the LAST p
            end = kv_len - q_len + off + p       # keys they can see -> slice + reuse "causal"
            oc = orig(qc, k[..., :end, :], v[..., :end, :],
                      scale=scale, mask=mask, sinks=sinks, **kw)
        else:                                    # array mask: slice q rows, keep full KV
            mc = mask if mask.shape[-2] == 1 else mask[..., off:off + p, :]
            oc = orig(qc, k, v, scale=scale, mask=mc, sinks=sinks, **kw)
        outs.append(oc)
        off += p
    return mx.concatenate(outs, axis=-2)


# ---- scoped patch (mirrors small_m_qmm.small_m_matmul) --------------------------------

_ACTIVE: SplitConfig | None = None


def _patched_call(q, k, v, *, scale, mask=None, sinks=None, **kw):
    cfg = _ACTIVE
    if cfg is None:
        return _ORIG_SDPA(q, k, v, scale=scale, mask=mask, sinks=sinks, **kw)
    return split_sdpa(q, k, v, scale=scale, mask=mask, sinks=sinks, cfg=cfg, _orig=_ORIG_SDPA, **kw)


@contextlib.contextmanager
def sdpa_split(cfg: SplitConfig | None):
    """Route ``mx.fast.scaled_dot_product_attention`` through the split while active.
    No-op when ``cfg`` is None or already installed (nests cleanly)."""
    global _ACTIVE
    if cfg is None or mx.fast.scaled_dot_product_attention is _patched_call:
        yield
        return
    prev = mx.fast.scaled_dot_product_attention
    prev_cfg = _ACTIVE
    _ACTIVE = cfg
    mx.fast.scaled_dot_product_attention = _patched_call
    try:
        yield
    finally:
        mx.fast.scaled_dot_product_attention = prev
        _ACTIVE = prev_cfg


def patchable() -> bool:
    """True iff mx.fast.scaled_dot_product_attention can be monkeypatched on this build."""
    try:
        orig = mx.fast.scaled_dot_product_attention
        mx.fast.scaled_dot_product_attention = orig
        return True
    except (AttributeError, TypeError):
        return False
