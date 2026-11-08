"""Draft trees for speculative verify: best-first trees over a DFlash 2 selector lattice.

A DFlash 2 drafter proposes a *lattice*, not just a chain: per mask slot ``s`` (draft depth
``s + 1``) it keeps the target head's top-K candidates and the selector scores every
adjacent-slot transition, ``scores[s, parent, cand]`` (slot 0's parent rows are all the
verified anchor). The stock loop verifies the greedy walk through it — one chain. A
**best-first tree** (DDTree / SpecPilot §4.2) keeps instead the ``N`` nodes with the highest
path log-probability, each edge scored by the log-softmax over the K candidates of its
parent's row. Edge log-probs are <= 0, so the top-N set is prefix-closed and best-first
expansion emits every parent before its children (row order = topological order).

Why an EQUAL-width tree is the lever on Apple GPUs. A tree of ``N`` drafted nodes costs one
target forward of width ``N + 1`` — exactly a chain of cap ``N``. SpecPilot's H100 trees win by
adding rows (32-node trees; verify rows are ~free there: +3.9% from 1 to 64 rows), which is
the wrong trade on M1-M4, where the 4-bit 27B verify is a staircase (97 ms at 8 rows, 160 at
9-16, 281 at 17-32). Spending the SAME rows better is free of that cost: exact replay of
recorded Qwen3.8-27B + DFlash 2 lattices gives 4.80 vs 4.43 committed tokens per round at
width 8 (+8.4%; chat +12.5%, confident code/math ~0 — the tree degenerates to the chain
there). See NOTES "SpecPilot (arXiv 2611.02501) mapped onto an M4 Pro".

Losslessness is untouched: the target decides every token. :meth:`DraftTree.walk` accepts
the deepest root path whose every edge matches the target's argmax at the parent, then
takes the target's own token after the last accepted node as the bonus — the chain rule,
generalized. Target-side verify / commit live in ``Target.verify_tree`` / ``commit_tree``.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

# Relative committed tokens per round of a best-first tree of N drafted nodes vs the full
# DFlash 2 chain (cap 7, block 8), from exact replay of recorded lattices: Qwen3.8-27B-4bit +
# incoai/z-lab DFlash2, 6 prompts (chat/code/math x2, thinking on), 456 rounds — NOTES
# "SpecPilot ... mapped onto an M4 Pro". Used only to PRICE trees wider than the chain cap on
# hardware whose measured verify curve makes wide verifies cheap (the M5 question); the
# equal-width tree needs no prior (it never costs extra rows).
TREE_GAIN_PRIOR: dict[int, float] = {7: 1.084, 11: 1.151, 15: 1.203, 23: 1.269, 31: 1.305}

# Hard ceiling on drafted nodes: the prior's last measured point, the multirow kernel's uint32
# ancestor bitmask (width <= 32), and a bound on the unrolled GPU builder (each node is a step;
# 63 nodes took ~5 s to compile).
TREE_MAX_NODES = 31

# Mean LEAF count of best-first trees by node count, replayed on the same six recorded
# Qwen3.8-27B DFlash 2 lattices as TREE_GAIN_PRIOR (2026-11-07, drafts/specpilot/replay). Where
# the recurrence runs the path-batched fallback instead of the tree kernel (no verified kernel
# on this GPU generation, odd head dims) every leaf costs one more state read + write, so a
# synthetic tree with this many leaves prices a node count honestly (calibrate.measure_tree_curve).
# A 7-node tree is a plain chain in 39% (chat) to 87% (math) of rounds; the pooled mean is what
# the fallback pays per round on average.
TREE_LEAVES_PRIOR: dict[int, float] = {3: 1.17, 5: 1.42, 7: 1.74, 11: 3.27, 15: 4.67,
                                       23: 7.47, 31: 10.48}


def prior_leaves(n_nodes: int) -> int:
    """Expected leaf count of an ``n_nodes`` best-first tree: ``TREE_LEAVES_PRIOR`` interpolated
    (flat past its ends), rounded, clamped to ``1..n_nodes``."""
    ks = sorted(TREE_LEAVES_PRIOR)
    v = float(np.interp(n_nodes, ks, [TREE_LEAVES_PRIOR[k] for k in ks]))
    return int(max(1, min(int(n_nodes), round(v))))


def reachable_nodes(slots: int, k: int) -> int:
    """Distinct nodes a ``slots`` x ``k`` lattice can hold (sum of k^d, d = 1..slots), capped at
    ``TREE_MAX_NODES`` — asking for more would make the GPU builder pad with duplicates."""
    total, layer = 0, 1
    for _ in range(max(0, slots)):
        layer *= k
        total += layer
        if total >= TREE_MAX_NODES:
            return TREE_MAX_NODES
    return total


def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x.astype(np.float64)
    m = x.max(axis=axis, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=axis, keepdims=True))


@dataclass
class DraftTree:
    """A verify tree. Row 0 is the anchor (the root: the last committed token, not yet run
    through the target); rows ``1..N`` are drafted nodes in best-first (topological) order."""

    tokens: list[int]
    parent: list[int]            # parent row per row; parent[0] == -1
    depth: list[int] = field(default_factory=list)
    children: list[list[int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n = len(self.tokens)
        if len(self.parent) != n or self.parent[0] != -1:
            raise ValueError("parent must have one entry per row, with parent[0] == -1")
        if not self.depth:
            self.depth = [0] * n
            for r in range(1, n):
                p = self.parent[r]
                if not 0 <= p < r:
                    raise ValueError("rows must be topologically ordered (parent < child)")
                self.depth[r] = self.depth[p] + 1
        self.children = [[] for _ in range(n)]
        for r in range(1, n):
            self.children[self.parent[r]].append(r)

    # -- shape --
    @property
    def width(self) -> int:
        """Verify width = anchor + drafted nodes."""
        return len(self.tokens)

    @property
    def n_nodes(self) -> int:
        return len(self.tokens) - 1

    def is_chain(self) -> bool:
        """True when the tree is a single path in row order — verify it as a plain chain."""
        return all(self.parent[r] == r - 1 for r in range(1, len(self.tokens)))

    def path_to(self, r: int) -> list[int]:
        """Rows from the root to ``r`` inclusive."""
        out = []
        while r >= 0:
            out.append(r)
            r = self.parent[r]
        return out[::-1]

    def leaves(self) -> list[int]:
        return [r for r in range(len(self.tokens)) if not self.children[r]]

    def leaf_paths(self) -> list[list[int]]:
        """One root path per leaf. Every row lies on at least one — the path-batched
        recurrence runs these as a batch (``Target.verify_tree``)."""
        return [self.path_to(leaf) for leaf in self.leaves()]

    def ancestor_matrix(self) -> np.ndarray:
        """``anc[r, c]`` = row ``c`` is ``r`` or one of its ancestors (the attention mask over
        the tree's own rows; the committed prefix is visible to every row)."""
        n = len(self.tokens)
        anc = np.zeros((n, n), dtype=bool)
        for r in range(n):
            for c in self.path_to(r):
                anc[r, c] = True
        return anc

    def row_slots(self) -> tuple[list[int], list[int]]:
        """For each row: (index into :meth:`leaf_paths`, position on that path). Any path
        containing the row gives the same values (the recurrence along a shared prefix is
        identical), so the first is used."""
        p_of, d_of = [0] * len(self.tokens), [0] * len(self.tokens)
        seen = [False] * len(self.tokens)
        for pi, path in enumerate(self.leaf_paths()):
            for d, r in enumerate(path):
                if not seen[r]:
                    seen[r], p_of[r], d_of[r] = True, pi, d
        return p_of, d_of

    def conv_index(self, kernel: int) -> np.ndarray:
        """``[W, kernel]`` row indices into ``[conv window (kernel-1 rows) ‖ tree rows]`` for a
        causal depthwise conv evaluated along each row's own root path: row ``r``'s window is
        the last ``kernel`` entries of ``[window rows, path_to(r) rows]``. For a chain this is
        ``r .. r+kernel-1`` — the plain sliding window."""
        k1 = kernel - 1
        idx = np.zeros((len(self.tokens), kernel), dtype=np.int32)
        for r in range(len(self.tokens)):
            seq = list(range(k1)) + [k1 + c for c in self.path_to(r)]
            idx[r] = seq[-kernel:]
        return idx

    # -- accept --
    def walk(self, target_next) -> tuple[list[int], int]:
        """Greedy acceptance. ``target_next[r]`` = the target's argmax token AFTER row ``r``.
        Returns ``(accepted rows incl. the root, bonus token)``: follow the child whose token
        equals the target's choice at the current row until none does; the bonus is the
        target's own token after the last accepted row (lossless, like the chain rule). At most
        one child can match — siblings carry distinct tokens."""
        cur, path = 0, [0]
        while True:
            want = int(target_next[cur])
            nxt = next((c for c in self.children[cur] if self.tokens[c] == want), None)
            if nxt is None:
                return path, want
            path.append(nxt)
            cur = nxt


def best_first_tree(cand: np.ndarray, scores: np.ndarray, n_nodes: int,
                    anchor: int) -> DraftTree:
    """Best-first top-``n_nodes`` tree over a selector lattice.

    ``cand`` [S, K] candidate token ids per slot; ``scores`` [S, K, K] =
    ``scores[slot, parent_candidate, candidate]`` (slot 0: every parent row is the anchor, row 0
    is used). Edge score = log-softmax over the K candidates of the parent's row (SpecPilot's
    builder, temperature 1); path score = sum along the path; ties go to the earlier-pushed
    entry (lower candidate index). Depth is bounded by the lattice's ``S`` slots."""
    cand = np.asarray(cand)
    S, K = cand.shape
    lp0 = _log_softmax(np.asarray(scores[0, 0]))
    lps = _log_softmax(np.asarray(scores), axis=-1)
    tokens, parent, depth, cidx = [int(anchor)], [-1], [0], [-1]
    seq = 0
    heap: list[tuple[float, int, int, int]] = []          # (-score, seq, parent_row, cand)
    for c in range(K):
        heap.append((-float(lp0[c]), seq, 0, c))
        seq += 1
    heapq.heapify(heap)
    score = [0.0]
    while heap and len(tokens) - 1 < n_nodes:
        neg, _, prow, c = heapq.heappop(heap)
        d = depth[prow] + 1
        row = len(tokens)
        tokens.append(int(cand[d - 1, c]))
        parent.append(prow)
        depth.append(d)
        cidx.append(c)
        score.append(-neg)
        if d < S:
            edges = lps[d, c]
            for c2 in range(K):
                heapq.heappush(heap, (neg - float(edges[c2]), seq, row, c2))
                seq += 1
    return _main_path_first(tokens, parent)


def _main_path_first(tokens: list[int], parent: list[int]) -> DraftTree:
    """Re-lay the rows in depth-first preorder, children in best-first (score) order. The
    tree is unchanged; its rows are not: the best child of the root, ITS best child, ... — the
    greedy chain the stock loop would verify — now occupy rows 1..d contiguously, so the
    common accept (a prefix of that chain) commits as a plain KV trim, and only an accept
    through a side branch pays the compaction (Target.commit_tree)."""
    kids: list[list[int]] = [[] for _ in tokens]
    for r in range(1, len(tokens)):
        kids[parent[r]].append(r)              # rows were pushed in score order
    order, stack = [], [0]
    while stack:
        r = stack.pop()
        order.append(r)
        stack.extend(reversed(kids[r]))
    new_of = {old: new for new, old in enumerate(order)}
    return DraftTree(tokens=[tokens[r] for r in order],
                     parent=[-1] + [new_of[parent[r]] for r in order[1:]])


@dataclass
class DeviceTree:
    """A verify tree as DEVICE arrays (lazy until evaluated) — what ``Target.verify_tree``
    consumes. Built on the GPU by :func:`device_tree` inside the round's graph, so the drafter,
    the tree build and the verify forward reach the device as one graph with one sync (a
    host-built tree costs a mid-round sync that forfeits the drafter/verify-build overlap,
    ~4% of a round). ``host`` is set when it came from a :class:`DraftTree`."""

    tokens: mx.array        # [W] int32, row 0 = anchor
    parent: mx.array        # [W] int32, -1 at the root
    depth: mx.array         # [W] int32
    anc: mx.array           # [W, W] bool: anc[r, c] = c is r or an ancestor of r
    bits: mx.array          # [W] uint32: anc rows as bitmasks (multirow_attn tree mask)
    conv_idx: mx.array      # [W, conv_k] int32: per-row causal conv window (see conv_index)
    width: int
    host: DraftTree | None = None

    def to_host(self) -> DraftTree:
        if self.host is None:
            self.host = DraftTree(tokens=self.tokens.tolist(), parent=self.parent.tolist())
        return self.host


def to_device(tree: DraftTree, conv_k: int) -> DeviceTree:
    """A host tree as device arrays (tests, the self-test, the host-built path)."""
    anc = tree.ancestor_matrix()
    w = tree.width
    bits = [sum(1 << c for c in range(w) if anc[r, c]) for r in range(w)]
    return DeviceTree(tokens=mx.array(tree.tokens, dtype=mx.int32),
                      parent=mx.array(tree.parent, dtype=mx.int32),
                      depth=mx.array(tree.depth, dtype=mx.int32),
                      anc=mx.array(anc), bits=mx.array(bits, dtype=mx.uint32),
                      conv_idx=mx.array(tree.conv_index(conv_k)), width=w, host=tree)


def _device_tree_impl(cand, scores, anchor, *, n_nodes: int, conv_k: int):
    """Best-first top-``n_nodes`` tree + main-path-first relayout + the verify's index arrays,
    all as fixed-shape MLX ops (unrolled over the nodes; compiled per shape by
    :func:`device_tree`). Same tree as :func:`best_first_tree` up to exact score ties."""
    S, K = cand.shape
    W = n_nodes + 1
    NEG = -1e30
    scores = scores.astype(mx.float32)
    lp0 = scores[0, 0] - mx.logsumexp(scores[0, 0])
    lps = scores - mx.logsumexp(scores, axis=-1, keepdims=True)               # [S, K, K]
    rows, cols = mx.arange(W), mx.arange(K)
    frontier = mx.concatenate([lp0[None], mx.full((W - 1, K), NEG)], axis=0)  # [W, K]
    depth = mx.zeros((W,), dtype=mx.int32)
    cidx = mx.zeros((W,), dtype=mx.int32)
    parent = mx.full((W,), -1, dtype=mx.int32)
    for j in range(1, W):                         # pop the best frontier entry, push its kids
        flat = frontier.reshape(-1)
        best = mx.argmax(flat)
        n = (best // K).astype(mx.int32)
        c = (best % K).astype(mx.int32)
        s = flat[best]
        d = depth[n] + 1
        is_j = rows == j
        depth = mx.where(is_j, d, depth)
        cidx = mx.where(is_j, c, cidx)
        parent = mx.where(is_j, n, parent)
        frontier = mx.where((rows[:, None] == n) & (cols[None, :] == c), NEG, frontier)
        kids = mx.where(d < S, s + lps[mx.minimum(d, S - 1), c], NEG)
        frontier = mx.where(is_j[:, None], kids[None, :], frontier)
    node_tok = mx.take(cand.reshape(-1), mx.maximum(depth[1:] - 1, 0) * K + cidx[1:])
    tokens = mx.concatenate([anchor.reshape(1).astype(mx.int32), node_tok.astype(mx.int32)])
    # ancestor-or-self matrix from parent powers
    P = (parent[:, None] == rows[None, :]).astype(mx.float32)
    cur = mx.eye(W, dtype=mx.float32)
    acc = cur
    for _ in range(S):
        cur = cur @ P
        acc = acc + cur
    anc = acc > 0
    # main path first: depth-first preorder rank, children in pop (score) order —
    # rank(r) = depth(r) + sizes of the earlier siblings of r and of each of its ancestors
    size = anc.astype(mx.float32).sum(axis=0)
    earlier = ((parent[:, None] == parent[None, :]) & (rows[None, :] < rows[:, None])
               & (parent[:, None] >= 0)).astype(mx.float32)
    pre = (depth.astype(mx.float32) + anc.astype(mx.float32) @ (earlier @ size)).astype(mx.int32)
    order = mx.argsort(pre)
    tokens = tokens[order]
    depth = depth[order]
    par_old = parent[order]
    parent = mx.where(par_old < 0, -1, pre[mx.maximum(par_old, 0)])
    anc = anc[order][:, order]
    bits = (anc.astype(mx.uint32) * (mx.ones((W,), mx.uint32) << rows.astype(mx.uint32))).sum(-1)
    # per-row causal conv window along the root path (DraftTree.conv_index, on device)
    k1 = conv_k - 1
    e = depth[:, None] + mx.arange(conv_k)[None, :]                                # [W, Kc]
    at = (anc[:, :, None] & (depth[None, :, None] == mx.arange(S + 1)[None, None, :]))
    anc_at = mx.argmax(at.astype(mx.int32), axis=1)                               # [W, S+1]
    pick = mx.take_along_axis(anc_at, mx.clip(e - k1, 0, S), axis=1)
    conv_idx = mx.where(e < k1, e, k1 + pick).astype(mx.int32)
    return tokens, parent.astype(mx.int32), depth, anc, bits.astype(mx.uint32), conv_idx


_COMPILED: dict = {}


def device_tree(cand: mx.array, scores: mx.array, n_nodes: int, anchor: int,
                conv_k: int = 4) -> DeviceTree:
    """GPU-side :func:`best_first_tree` (main path first) as a lazy :class:`DeviceTree`.
    ``n_nodes`` is clamped to what the lattice can hold (:func:`reachable_nodes`)."""
    n_nodes = max(1, min(int(n_nodes), reachable_nodes(cand.shape[0], cand.shape[1])))
    key = (int(n_nodes), int(conv_k))
    fn = _COMPILED.get(key)
    if fn is None:
        def fn(c, s, a, _n=key[0], _k=key[1]):
            return _device_tree_impl(c, s, a, n_nodes=_n, conv_k=_k)
        fn = _COMPILED[key] = mx.compile(fn)
    tokens, parent, depth, anc, bits, conv_idx = fn(cand, scores,
                                                   mx.array([int(anchor)], dtype=mx.int32))
    return DeviceTree(tokens=tokens, parent=parent, depth=depth, anc=anc, bits=bits,
                      conv_idx=conv_idx, width=int(n_nodes) + 1)


def chain_tree(anchor: int, drafted) -> DraftTree:
    """The stock chain as a tree (anchor + drafted tokens in sequence)."""
    toks = [int(anchor)] + [int(t) for t in drafted]
    return DraftTree(tokens=toks, parent=[-1] + list(range(len(toks) - 1)))


def synthetic_tree(n_nodes: int, slots: int, leaves: int | None = None,
                   anchor: int = 0) -> DraftTree:
    """A timing stand-in for a best-first tree: ``n_nodes`` drafted nodes, depth <= ``slots``
    (the lattice depth), ``leaves`` leaves (default :func:`prior_leaves`; exact whenever the
    shape allows, never fewer) and the main path laid out first like the real builder. A tree
    verify's cost depends on its width (attention, GEMMs) and its leaf count (the path-batched
    recurrence fallback) and on nothing else about the shape, so this is what
    ``calibrate.measure_tree_curve`` times. Tokens are distinct placeholders ``1..n``."""
    n = max(1, int(n_nodes))
    slots = max(1, int(slots))
    want = prior_leaves(n) if leaves is None else int(leaves)
    want = max(1, min(n, want))
    tokens, parent = [int(anchor)], [-1]

    def add(p: int) -> int:
        tokens.append(len(tokens))
        parent.append(p)
        return len(tokens) - 1

    d0 = max(1, min(slots, n - (want - 1)))     # main chain; one node kept per extra leaf
    row = 0
    for _ in range(d0):
        row = add(row)
    tips = [(row, d0)]                           # (row, depth) of every branch tip
    left = n - d0
    attach = 0
    while left > 0 and len(tips) < want:         # one branch start per extra leaf, attached
        k = attach % min(d0, slots)              # to the shallowest chain rows first (row k
        tips.append((add(k), k + 1))             # has depth k; k <= slots - 1 keeps depth)
        attach += 1
        left -= 1
    while left > 0:                              # the rest extends the shallowest tip
        open_tips = [i for i, (_, d) in enumerate(tips) if d < slots]
        if not open_tips:                        # every tip at full depth: one more leaf
            tips.append((add(0), 1))
        else:
            i = min(open_tips, key=lambda i: tips[i][1])
            r, d = tips[i]
            tips[i] = (add(r), d + 1)
        left -= 1
    return DraftTree(tokens=tokens, parent=parent)


def tree_gain(n_nodes: int, cap: int) -> float:
    """Prior committed-token gain of a best-first tree WIDER than the chain cap
    (``n_nodes > cap``) over the equal-width tree at that cap: ``prior(n) / prior(cap)`` from
    ``TREE_GAIN_PRIOR`` (piecewise-linear, flat past its last point). 1.0 when not wider, and
    1.0 when ``cap`` lies below the measured regime (the prior was replayed on full-block
    7-slot lattices — a depth-shrunk cap is never widened on its say-so)."""
    if n_nodes <= cap:
        return 1.0
    ks = sorted(TREE_GAIN_PRIOR)
    if cap < ks[0]:
        return 1.0
    vals = [TREE_GAIN_PRIOR[k] for k in ks]
    return float(max(np.interp(n_nodes, ks, vals) / np.interp(cap, ks, vals), 1.0))
