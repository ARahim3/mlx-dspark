"""Gated-DeltaNet recurrence over a draft TREE in one kernel (tree verify, see tree.py), with an
optional PENDING chain folded in front of it (deferred commit, see Target.rollback).

The chain verify runs mlx-lm's ``gated_delta_update``: per (value head, value row) one
simdgroup walks the T tokens in order, carrying a [Dk] slice of the fp32 state in registers.
A tree's node needs its PARENT's state instead of the previous row's. Running the stock kernel
path-batched (one batch row per leaf path) is exact but re-reads and re-writes the whole fp32
state once per leaf — on Qwen3.8-27B that is 2 x 144 MB per extra leaf per round (~2 ms each,
measured), more than the tree saves on confident content.

This kernel walks the nodes in topological order with ``parent[t]`` pointers: each node starts
from its parent's state (kept per node in threadgroup memory — every lane reads back only its
own slice, so no barrier) or from the base state at the root. The per-step arithmetic is
copied verbatim from mlx-lm's simd_sum kernel (0.31.x), so on that kernel every node is
bit-identical to a chain over its root path; mlx-lm >= 0.32.2 runs chains on a re-laid-out
"packed" kernel whose reduction is built to equal simd_sum on current Apple GPUs (measured
bitwise equal on Qwen3.8-27B).

**Pending segment (deferred commit).** A round's accepted rows are not folded into the stored
state when they are accepted (that re-run cost a state read + write per round — ~2.8 ms on a
27B); they are handed to the NEXT verify as ``pending``: this kernel first walks them as a chain
in registers from the stored state, writes the state they end in ONCE (``state_out`` — the base
every new row starts from, and what the cache keeps), then walks the new rows. One state read
and one state write per round, whatever the round did. With nothing pending the base is the
stored state and no state is written (the accepted path's state is rebuilt by whoever folds the
pending rows next: this kernel, or ``Target.flush_pending`` with the stock kernel).
"""

from __future__ import annotations

import mlx.core as mx

_SOURCE = r"""
    auto hv_idx = thread_position_in_grid.z;              // batch 1: z = value head
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;
    auto dk_idx = thread_position_in_threadgroup.x;       // lane
    auto dv_idx = thread_position_in_grid.y;
    auto r_idx = thread_position_in_threadgroup.y;        // value row within the threadgroup

    threadgroup float st[NR * W * Dk];
    threadgroup float* my = st + r_idx * W * Dk + n_per_t * dk_idx;   // node t: my[t * Dk + i]
    auto i_state = state_in + (hv_idx * Dv + dv_idx) * Dk;
    const int P = n_pending[0];

    // segment 1: the pending chain, walked in registers from the stored state; its end is the
    // base every new row starts from, written once. Same per-step state arithmetic as the stock
    // kernel; the readout `out` is skipped (no row needs these outputs).
    float base[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      base[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
    }
    for (int t = 0; t < P; ++t) {
      auto k_ = kp + (t * Hk + hk_idx) * Dk;
      auto v_ = vp + (t * Hv + hv_idx) * Dv;
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        base[i] = base[i] * gp[t * Hv + hv_idx];
        kv_mem += base[i] * k_[s_idx];
      }
      kv_mem = simd_sum(kv_mem);
      auto delta = (v_[dv_idx] - kv_mem) * betap[t * Hv + hv_idx];
      for (int i = 0; i < n_per_t; ++i) {
        base[i] = base[i] + k_[n_per_t * dk_idx + i] * delta;
      }
    }
    if (P > 0) {
      auto o_state = state_out + (hv_idx * Dv + dv_idx) * Dk;
      for (int i = 0; i < n_per_t; ++i) {
        o_state[n_per_t * dk_idx + i] = static_cast<StT>(base[i]);
      }
    }

    // segment 2: the new rows (a chain or a tree), each from its parent's state or the base
    float state[n_per_t];
    for (int t = 0; t < W; ++t) {
      const int p = parent[t];
      if (p < 0) {
        for (int i = 0; i < n_per_t; ++i) {
          state[i] = base[i];
        }
      } else {
        for (int i = 0; i < n_per_t; ++i) {
          state[i] = my[p * Dk + i];
        }
      }
      auto q_ = q + (t * Hk + hk_idx) * Dk;
      auto k_ = k + (t * Hk + hk_idx) * Dk;
      auto v_ = v + (t * Hv + hv_idx) * Dv;

      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * g[t * Hv + hv_idx];
        kv_mem += state[i] * k_[s_idx];
      }
      kv_mem = simd_sum(kv_mem);
      auto delta = (v_[dv_idx] - kv_mem) * beta[t * Hv + hv_idx];

      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + k_[s_idx] * delta;
        out += state[i] * q_[s_idx];
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0) {
        y[(t * Hv + hv_idx) * Dv + dv_idx] = static_cast<InT>(out);
      }
      for (int i = 0; i < n_per_t; ++i) {
        my[t * Dk + i] = state[i];
      }
    }
"""

_KERNEL = None
_TG_BYTES = 16384          # per-threadgroup node-state budget (Metal's limit is 32 KB)


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="dspark_tree_gated_delta",
            input_names=["qp", "kp", "vp", "gp", "betap", "q", "k", "v", "g", "beta",
                         "state_in", "parent", "n_pending"],
            output_names=["y", "state_out"],
            source=_SOURCE,
        )
    return _KERNEL


def rows_per_group(width: int, dk: int, dv: int) -> int:
    """Value rows per threadgroup: as many as the node-state budget allows (<= 4, mlx-lm's
    own grouping), dividing Dv. 0 = the tree does not fit (caller falls back)."""
    for nr in (4, 2, 1):
        if dv % nr == 0 and nr * width * dk * 4 <= _TG_BYTES:
            return nr
    return 0


# Like the project's other custom kernels, verified end to end on M4 (g16) only: newer GPU
# generations take the exact stock-kernel fallback (Target._tree_linear's path batching) until
# an M5 run earns it; MLX_DSPARK_FORCE_TREE_KERNEL=1 overrides for that A/B.
_MAX_GEN = 16


def _gen_ok() -> bool:
    import os
    import re

    if os.environ.get("MLX_DSPARK_FORCE_TREE_KERNEL", "").lower() in ("1", "true"):
        return True
    m = re.search(r"applegpu_g(\d+)", str(mx.device_info().get("architecture", "")))
    return m is None or int(m.group(1)) <= _MAX_GEN


def eligible(q, v, g, state) -> bool:
    """Shapes the kernel serves: batch 1, scalar gating, fp32 state, Dk a multiple of 32, the
    tree's node states within the threadgroup budget, on a Metal GPU of a verified generation."""
    if not (mx.metal.is_available() and mx.default_device() == mx.gpu and _gen_ok()):
        return False
    if q.ndim != 4 or q.shape[0] != 1 or g.ndim != 3 or state is None:
        return False
    w, dk = q.shape[1], q.shape[-1]
    if dk < 32 or dk % 32 or w > 64 or state.dtype != mx.float32:
        return False
    return rows_per_group(w, dk, v.shape[-1]) > 0


def chain_parent(width: int) -> mx.array:
    """``parent`` for a plain chain of ``width`` rows: -1, 0, 1, ... (row t follows row t-1)."""
    return mx.concatenate([mx.array([-1], dtype=mx.int32),
                           mx.arange(width - 1, dtype=mx.int32)]) if width > 1 \
        else mx.array([-1], dtype=mx.int32)


def tree_gated_delta(q, k, v, g, beta, state, parent: mx.array, pending=None):
    """Outputs ``(y [1, W, Hv, Dv], base)`` of the gated-delta recurrence run along each row's
    root path (``parent`` int32 [W], -1 at the root, topological order; a chain is
    :func:`chain_parent`). ``g``/``beta`` are the gate and step size exactly as
    ``gated_delta_update`` derives them; ``state`` [1, Hv, Dv, Dk] is the stored state.

    ``pending`` = ``(qp, kp, vp, gp, betap)``, P rows of an accepted chain the stored state does
    not include yet (``Target.rollback``): they are walked first, from ``state``, and ``base`` is
    the state they end in — the one state this call writes, what the cache keeps. With nothing
    pending ``base`` IS ``state`` (the same array: nothing written)."""
    _, w, hk, dk = k.shape
    hv, dv = v.shape[2:]
    nr = rows_per_group(w, dk, dv)
    if pending is None:
        n_p = 0
        qp = kp = mx.zeros((1, 1, hk, dk), dtype=q.dtype)
        vp = mx.zeros((1, 1, hv, dv), dtype=v.dtype)
        gp = mx.zeros((1, 1, hv), dtype=g.dtype)
        betap = mx.zeros((1, 1, hv), dtype=beta.dtype)
    else:
        qp, kp, vp, gp, betap = pending
        n_p = int(qp.shape[1])
    y, s_out = _kernel()(
        inputs=[qp, kp, vp, gp, betap, q, k, v, g, beta, state, parent,
                mx.array([n_p], dtype=mx.int32)],
        template=[("InT", q.dtype), ("StT", state.dtype), ("Dk", dk), ("Dv", dv),
                  ("Hk", hk), ("Hv", hv), ("W", w), ("NR", nr)],
        grid=(32, dv, hv),
        threadgroup=(32, nr, 1),
        output_shapes=[(1, w, hv, dv), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )
    return y, (state if n_p == 0 else s_out)
