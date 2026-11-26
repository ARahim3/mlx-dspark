"""Multi-row decode-attention kernel (multirow_attn.py): numerics, gating, the scoped patch.

The kernel is fp-tie class (fp32 softmax/accumulation, input-dtype MMA operands), so the
load-bearing checks are (1) it matches an fp32 reference as closely as mlx's own SDPA does,
over every structural case it claims — causal and bidirectional, GQA 1..16, head dims 64 /
128 / 256 (the d-split path), KV lengths off the 16-key tile grid (the shifted tail tile),
strided cache views, batch > 1, bf16 and fp16, row groups past one threadgroup; (2) the gate
never routes anything outside a measured window (array masks, sinks, unmeasured shapes,
prefill-wide q, short KV) — those must be mlx's exact bits; (3) the patch composes with the
SDPA split and always restores.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_dspark import multirow_attn as mra
from mlx_dspark.multirow_attn import (
    MultirowConfig,
    Window,
    eligible,
    multirow_attention,
    multirow_sdpa,
    plan,
)
from mlx_dspark.sdpa_split import SplitConfig, sdpa_split


def _mk(B, hq, hk, q_len, L, d, dtype=mx.bfloat16, slack=0, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((B, hq, q_len, d)).astype(dtype)
    k = mx.random.normal((B, hk, L + slack, d)).astype(dtype)
    v = mx.random.normal((B, hk, L + slack, d)).astype(dtype)
    if slack:                                    # a KVCache-style view into a larger buffer
        k, v = k[:, :, :L, :], v[:, :, :L, :]
    mx.eval(q, k, v)
    return q, k, v


def _err(out, q, k, v, causal):
    scale = q.shape[-1] ** -0.5
    ref = mra._ORIG_SDPA(q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
                         scale=scale, mask="causal" if causal else None)
    return float(mx.abs(out.astype(mx.float32) - ref).max())


CASES = [
    # B, hq, hk, q_len, L, d
    (1, 24, 4, 8, 1037, 256),     # Qwen3.8 geometry, d-split, off-grid tail
    (1, 24, 4, 1, 300, 256),
    (1, 32, 8, 15, 777, 128),     # DSpark drafter block
    (1, 32, 8, 2, 64, 128),
    (1, 32, 2, 16, 500, 128),     # GQA 16 -> 256 rows: row groups
    (2, 8, 8, 5, 129, 64),        # MHA, batch 2
    (1, 16, 2, 7, 16, 128),       # KV exactly one tile
    (1, 4, 1, 3, 23, 64),         # MQA, tail shorter than a tile past one tile
]


@pytest.mark.parametrize("B,hq,hk,q_len,L,d", CASES)
@pytest.mark.parametrize("causal", [True, False])
def test_matches_reference_as_well_as_mlx(B, hq, hk, q_len, L, d, causal):
    q, k, v = _mk(B, hq, hk, q_len, L, d, slack=37)
    scale = d ** -0.5
    mask = "causal" if causal and q_len > 1 else None
    try:
        out = multirow_sdpa(q, k, v, scale=scale, causal=causal and q_len > 1)
        mine = _err(out, q, k, v, mask is not None)
    except ValueError as e:
        # A geometry this GPU won't launch (GitHub's virtualized M1 caps the compiled kernel
        # at 384 threads per threadgroup; the GQA-16 x 16-row case needs 1024). Production
        # never launches it there: measure_window ends that shape's window below it
        # (test_measure_window_stops_below_an_unlaunchable_q).
        if "threads per threadgroup" not in str(e):
            raise
        pytest.skip(f"launch geometry unavailable on this GPU: {e}")
    theirs = _err(mra._ORIG_SDPA(q, k, v, scale=scale, mask=mask), q, k, v, mask is not None)
    assert out.shape == q.shape and out.dtype == q.dtype
    assert mine <= 2.0 * theirs + 2e-3, (mine, theirs)


def test_fp16_inputs():
    q, k, v = _mk(1, 24, 4, 6, 900, 256, dtype=mx.float16)
    out = multirow_sdpa(q, k, v, scale=256 ** -0.5, causal=True)
    assert out.dtype == mx.float16
    assert _err(out, q, k, v, True) < 5e-3


def test_causal_rows_see_only_their_prefix():
    # the LAST query row sees every key; the first sees all but the last q_len-1 — make the
    # masked keys dominate so a mask bug would swamp the output
    q, k, v = _mk(1, 4, 4, 4, 64, 64)
    k = mx.concatenate([k[:, :, :-3], 50.0 * mx.ones_like(k[:, :, -3:])], axis=2)
    v = mx.concatenate([v[:, :, :-3], 1e3 * mx.ones_like(v[:, :, -3:])], axis=2)
    out = multirow_sdpa(q, k, v, scale=0.125, causal=True)
    # rows that may see the 1e3-valued V legitimately round at bf16 ulp ~4 there; a leak
    # into a row that must not see them would be off by ~1e3
    assert _err(out, q, k, v, True) < 50.0


def test_plan_row_groups_and_blocks():
    p = plan(32, 2, 16, 128, 32768)                # 256 rows = 32 tiles -> one full tg
    assert p["NRG"] == 1 and p["NSG"] == 32
    p = plan(64, 2, 16, 128, 32768)                # 512 rows -> 2 row groups
    assert p["NRG"] == 2 and p["RP"] >= 512
    p = plan(24, 4, 8, 256, 32768)                 # d-split at 256
    assert p["DS"] == 2 and p["NSG"] == 12
    assert p["CHUNK"] % 16 == 0 and p["NB"] * p["CHUNK"] >= 32768


CFG = MultirowConfig((Window(8, 2, 64, min_q=2, max_q=16, min_kv=64),))


def test_eligible_gate():
    q, k, v = _mk(1, 8, 2, 4, 128, 64)
    assert eligible(q, k, v, "causal", None, CFG)
    assert eligible(q, k, v, None, None, CFG)
    assert not eligible(q, k, v, mx.ones((1, 1, 4, 128), dtype=mx.bool_), None, CFG)  # array
    assert not eligible(q, k, v, "causal", mx.zeros((8,)), CFG)                        # sinks
    q1, _, _ = _mk(1, 8, 2, 1, 128, 64)
    assert not eligible(q1, k, v, "causal", None, CFG)                 # q_len 1 below window
    qw, _, _ = _mk(1, 8, 2, 17, 128, 64)
    assert not eligible(qw, k, v, "causal", None, CFG)                 # prefill-wide q
    qs, ks, vs = _mk(1, 8, 2, 4, 32, 64)
    assert not eligible(qs, ks, vs, "causal", None, CFG)               # short KV
    qo, ko, vo = _mk(1, 16, 2, 4, 128, 64)
    assert not eligible(qo, ko, vo, "causal", None, CFG)               # unmeasured shape
    assert not eligible(q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
                        "causal", None, CFG)                           # fp32


def test_scope_routes_eligible_and_passes_through_the_rest_bit_exact():
    orig = mx.fast.scaled_dot_product_attention
    q, k, v = _mk(1, 8, 2, 4, 128, 64)
    qo, ko, vo = _mk(1, 16, 2, 4, 128, 64)                             # unmeasured shape
    stock = orig(qo, ko, vo, scale=0.125, mask="causal")
    with multirow_attention(CFG):
        assert mra.active()
        routed = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125, mask="causal")
        passed = mx.fast.scaled_dot_product_attention(qo, ko, vo, scale=0.125, mask="causal")
    assert mx.fast.scaled_dot_product_attention is orig and not mra.active()
    assert _err(routed, q, k, v, True) < 5e-3
    assert float(mx.abs(passed.astype(mx.float32) - stock.astype(mx.float32)).max()) == 0.0


def test_scope_composes_over_the_split_and_restores():
    orig = mx.fast.scaled_dot_product_attention
    split = SplitConfig(min_q=6, max_q=15, min_kv=64, max_chunk=5)
    qa, ka, va = _mk(1, 8, 2, 8, 128, 64)
    arr = mx.ones((1, 1, 8, 128), dtype=mx.bool_)                      # array mask -> split
    with sdpa_split(split), multirow_attention(CFG):
        assert mra.active()
        out = mx.fast.scaled_dot_product_attention(qa, ka, va, scale=0.125, mask=arr)
    assert mx.fast.scaled_dot_product_attention is orig
    ref = orig(qa, ka, va, scale=0.125, mask=arr)
    assert float(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max()) < 0.02


def test_none_config_is_a_noop():
    orig = mx.fast.scaled_dot_product_attention
    with multirow_attention(None):
        assert mx.fast.scaled_dot_product_attention is orig


def test_config_window_lookup():
    assert CFG.window(8, 2, 64) is not None
    assert CFG.window(8, 2, 128) is None


_LIMIT = "Thread group size (1024) is greater than the maximum allowed threads per threadgroup (384)."


def test_measure_window_stops_below_an_unlaunchable_q(monkeypatch):
    # A GPU whose pipeline limit refuses the wide-q geometries (GitHub's virtualized M1)
    # must get a window that ends below the first refused q — never admitting it, and not
    # switching the whole shape off (v0.20.0 did: one refused q16 cost Qwen3.8 its q 2-8).
    real = mra.multirow_sdpa

    def limited(q, k, v, *, scale, causal, tree_anc=None):
        if q.shape[2] >= 8:
            raise ValueError(_LIMIT)
        return real(q, k, v, scale=scale, causal=causal, tree_anc=tree_anc)

    monkeypatch.setattr(mra, "multirow_sdpa", limited)
    monkeypatch.setattr(mra, "_race", lambda *a, **kw: 2.0)     # the kernel wins every race
    monkeypatch.setattr(mra, "PROBE_KV", (1024,))
    # 8x2x128: every real width launches under the CI ceiling (384 threads), so only the
    # fake refusal at q8 ends the tables (32x2x128 there would end them at q7 on its own)
    window, report = mra.measure_window(8, 2, 128)
    served = {str(q): 1024 for q in range(2, 8)}
    assert window == {"causal": served, "tree": served}
    assert report["points"]["q8@1024"] == "unavailable"
    assert report["points"]["tree_q8@1024"] == "unavailable"
    assert "q9@1024" not in report["points"]                  # nothing wider was launched
    assert "threads per threadgroup" in report["unavailable"]


def test_measure_window_refused_at_the_first_q_is_off(monkeypatch):
    def refused(q, k, v, *, scale, causal, tree_anc=None):
        raise ValueError(_LIMIT)

    monkeypatch.setattr(mra, "multirow_sdpa", refused)
    monkeypatch.setattr(mra, "PROBE_KV", (1024,))
    window, report = mra.measure_window(32, 2, 128)
    assert window is None
    assert report["reason"].startswith("kernel unavailable")


def test_only_stale_unavailable_verdicts_are_reprobed(tmp_path, monkeypatch):
    # A v0.20.0 "kernel unavailable" verdict (no "probe" stamp) is re-probed once; windows
    # and every other refusal stamped with this build's launch size stay cached, and the
    # re-probed verdict (probe 2) is never re-probed again.
    import importlib

    C = importlib.import_module("mlx_dspark.calibrate")
    dev = mx.device_info().get("device_name", "unknown")
    key = lambda s: f"v{C.SCHEMA}|{dev}|mlx{mx.__version__}|mra2|{s}"  # noqa: E731
    cache = str(tmp_path)
    sg = {"max_sg": mra.MAX_SG}
    C.save_cached(key("32x2x128"), {"window": None, "report": {
        "reason": "kernel unavailable: ValueError: " + _LIMIT, **sg}}, cache)
    C.save_cached(key("24x4x256"), {"window": {"min_q": 2, "max_q": 16, "min_kv": 1024},
                                    "report": dict(sg)}, cache)
    C.save_cached(key("16x2x128"), {"window": None, "report": {"reason": "no win", **sg}},
                  cache)
    probed = []

    def fake(hq, hk, d, **kw):
        probed.append((hq, hk, d))
        return {"min_q": 2, "max_q": 6, "min_kv": 1024}, {"probe": 2, **sg}

    monkeypatch.setattr(mra, "measure_window", fake)
    shapes = [(32, 2, 128), (24, 4, 256), (16, 2, 128)]
    wins = C.multirow_windows(shapes, cache_dir=cache, verbose=False)
    assert probed == [(32, 2, 128)]
    assert {(w.hq, w.max_q) for w in wins} == {(32, 6), (24, 16)}
    C.multirow_windows(shapes, cache_dir=cache, verbose=False)
    assert probed == [(32, 2, 128)]                        # the re-probed verdict stands


def test_windows_probed_at_another_launch_size_are_reprobed(tmp_path, monkeypatch):
    # PR #39's branch probed at 14 simdgroups and cached windows up to q=16; replayed at this
    # build's 32 (1024 threads) the launch was refused mid-request on an M2 Pro. A verdict is
    # only trusted at the launch size it was probed at — unstamped (0.20.0-0.20.2) or foreign
    # stamps are re-probed once, then stand.
    import importlib

    C = importlib.import_module("mlx_dspark.calibrate")
    dev = mx.device_info().get("device_name", "unknown")
    key = lambda s: f"v{C.SCHEMA}|{dev}|mlx{mx.__version__}|mra2|{s}"  # noqa: E731
    cache = str(tmp_path)
    wide = {"min_q": 2, "max_q": 16, "min_kv": 1024}
    C.save_cached(key("16x2x256"), {"window": wide, "report": {"probe": 2, "max_sg": 14}},
                  cache)
    C.save_cached(key("32x8x128"), {"window": wide, "report": {"probe": 2}}, cache)
    probed = []

    def fake(hq, hk, d, **kw):
        probed.append((hq, hk, d))
        return {"min_q": 2, "max_q": 12, "min_kv": 1024}, {"probe": 2, "max_sg": mra.MAX_SG}

    monkeypatch.setattr(mra, "measure_window", fake)
    shapes = [(16, 2, 256), (32, 8, 128)]
    wins = C.multirow_windows(shapes, cache_dir=cache, verbose=False)
    assert probed == shapes
    assert {w.max_q for w in wins} == {12}
    C.multirow_windows(shapes, cache_dir=cache, verbose=False)
    assert probed == shapes                                 # stamped now: never again


def test_measure_window_stamps_its_launch_size(monkeypatch):
    monkeypatch.setattr(mra, "PROBE_KV", (1024,))
    monkeypatch.setattr(mra, "PROBE_Q", (2,))
    _, report = mra.measure_window(32, 8, 128)
    assert report["max_sg"] == mra.MAX_SG


# ---- per-width tables (mlx 0.32.5: the kernel's wins are per width and per call kind) ----


def test_window_tables_gate_each_kind_by_width():
    # mlx 0.32.5 on an M4 Pro at Qwen3.8's shape: causal q 6-8 tie mlx's new mid-row kernel
    # below 16k while a tree verify of those widths is 3-4x ahead of mlx's masked path
    w = Window.from_probe(8, 2, 64, {"causal": {"2": 64, "9": 64, "6": 128},
                                     "tree": {"2": 128, "6": 64}})
    assert hash(w) and w.min_kv_for(2) == 64 and w.min_kv_for(6) == 128
    assert w.min_kv_for(7) is None and w.min_kv_for(6, tree=True) == 64
    assert w.min_kv_for(9, tree=True) is None
    cfg = MultirowConfig((w,))
    q, k, v = _mk(1, 8, 2, 6, 64, 64)
    assert not eligible(q, k, v, "causal", None, cfg)           # causal q6 only from kv 128
    assert eligible(q, k, v, None, None, cfg, tree=True)         # the tree table admits it
    q9, k9, v9 = _mk(1, 8, 2, 9, 64, 64)
    assert eligible(q9, k9, v9, "causal", None, cfg)
    assert not eligible(q9, k9, v9, None, None, cfg, tree=True)
    # a kind the probe never ran (a drafter's shape: no tree table) serves no call of it
    drafter = Window.from_probe(8, 2, 64, {"causal": {"6": 64}})
    assert drafter.min_kv_for(6) == 64 and drafter.min_kv_for(6, tree=True) is None
    # an older contiguous verdict stays contiguous, for both kinds
    old = Window.from_probe(8, 2, 64, {"min_q": 2, "max_q": 8, "min_kv": 64})
    assert old.min_kv_for(8) == 64 == old.min_kv_for(8, tree=True) and old.min_kv_for(9) is None


def test_scope_routes_tree_and_causal_calls_by_their_own_tables(monkeypatch):
    cfg = MultirowConfig((Window.from_probe(8, 2, 64, {"causal": {}, "tree": {"6": 64}}),))
    q, k, v = _mk(1, 8, 2, 6, 128, 64)
    rows = mx.arange(128 - 6, 128).reshape(6, 1)
    arr = (mx.arange(128).reshape(1, 128) <= rows).reshape(1, 1, 6, 128)
    anc = mx.array([(1 << (i + 1)) - 1 for i in range(6)], dtype=mx.uint32)
    stock = mra._ORIG_SDPA(q, k, v, scale=0.125, mask="causal")
    real, calls = mra.multirow_sdpa, []

    def counted(*a, tree_anc=None, **kw):
        calls.append("tree" if tree_anc is not None else "causal")
        return real(*a, tree_anc=tree_anc, **kw)

    monkeypatch.setattr(mra, "multirow_sdpa", counted)
    with multirow_attention(cfg), mra.tree_mask(arr, anc):
        causal = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125, mask="causal")
        tree = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125, mask=arr)
    assert calls == ["tree"]
    assert float(mx.abs(causal.astype(mx.float32) - stock.astype(mx.float32)).max()) == 0.0
    assert _err(tree, q, k, v, True) < 0.02           # a chain-shaped tree is causal attention


def test_measure_window_builds_per_width_tables_deepest_first(monkeypatch):
    monkeypatch.setattr(mra, "PROBE_Q", (2, 4, 6, 8, 9))
    win_at = {("causal", 2): {1024, 4096, 16384}, ("causal", 6): {16384},
              ("causal", 8): {1024, 4096}, ("causal", 9): {1024, 4096, 16384},
              ("tree", 2): {4096, 16384}, ("tree", 6): {1024, 4096, 16384},
              ("tree", 8): {1024, 4096, 16384}, ("tree", 9): {1024, 4096, 16384}}
    probed = []

    def fake(kind, hq, ql, d, dtype, kvs, kv, scale, report):
        probed.append((kind, ql, kv))
        return "won" if kv in win_at.get((kind, ql), ()) else "lost"

    monkeypatch.setattr(mra, "_probe_point", fake)
    window, report = mra.measure_window(8, 2, 64)
    assert window == {"causal": {"2": 1024, "6": 16384, "9": 1024},
                      "tree": {"2": 4096, "6": 1024, "8": 1024, "9": 1024}}
    assert report["probe"] == 3
    # deepest first, every width there; a width is dropped at its first loss (causal q8 loses
    # at 16k and is never raced shallower even though it would "win" there; causal q6 stops
    # after its 4k loss)
    assert probed[:5] == [("causal", q, 16384) for q in (2, 4, 6, 8, 9)]
    assert ("causal", 8, 4096) not in probed and ("causal", 6, 1024) not in probed
    probed.clear()                                     # a drafter's shape: causal only
    window, _ = mra.measure_window(8, 2, 64, tree=False)
    assert set(window) == {"causal"} and {kind for kind, _, _ in probed} == {"causal"}


def test_a_borderline_point_must_win_its_second_race(monkeypatch):
    _, k, v = _mk(1, 8, 2, 4, 64, 64)
    ratios = iter([1.10, 1.02,          # borderline, then a loss: lost
                   1.10, 1.08,          # borderline twice: won on the lower ratio
                   1.30])               # a clear win: one race
    monkeypatch.setattr(mra, "_race", lambda *a, **kw: next(ratios))
    report = {"points": {}}
    args = (8, 4, 64, mx.bfloat16, [(k, v)], 64, 0.125, report)
    assert mra._probe_point("causal", *args) == "lost"
    assert mra._probe_point("causal", *args) == "won"
    assert mra._probe_point("tree", *args) == "won"
    assert report["points"] == {"q4@64": 1.08, "tree_q4@64": 1.3}


def test_a_shape_probed_as_a_drafters_is_reprobed_as_a_target(tmp_path, monkeypatch):
    import importlib

    C = importlib.import_module("mlx_dspark.calibrate")
    seen = []

    def fake(hq, hk, d, *, tree=True, **kw):
        seen.append(tree)
        win = {"causal": {"4": 1024}}
        if tree:
            win["tree"] = {"6": 1024}
        return win, {"probe": 3, "max_sg": mra.MAX_SG}

    monkeypatch.setattr(mra, "measure_window", fake)
    shape, cache = [(8, 2, 64)], str(tmp_path)
    C.multirow_windows(shape, tree_shapes=set(), cache_dir=cache, verbose=False)
    C.multirow_windows(shape, tree_shapes=set(), cache_dir=cache, verbose=False)   # cached
    (w,) = C.multirow_windows(shape, tree_shapes={(8, 2, 64)}, cache_dir=cache, verbose=False)
    assert seen == [False, True] and w.min_kv_for(6, tree=True) == 1024


def test_the_kernel_is_gated_on_through_m5_and_off_past_it(tmp_path, monkeypatch):
    import importlib
    from types import SimpleNamespace

    from mlx_dspark import generate as G

    C = importlib.import_module("mlx_dspark.calibrate")
    target = SimpleNamespace(model=SimpleNamespace(args=SimpleNamespace(
        num_attention_heads=8, num_key_value_heads=2, head_dim=64)))
    monkeypatch.setattr(mra, "measure_window", lambda hq, hk, d, **kw: (
        {"causal": {"4": 1024}, "tree": {"4": 1024}}, {"probe": 3, "max_sg": mra.MAX_SG}))
    monkeypatch.delenv("MLX_DSPARK_FORCE_MULTIROW", raising=False)
    try:
        for gen, on in ((16, True), (17, True), (18, False)):
            monkeypatch.setattr(C, "_gpu_gen", lambda g=gen: g)
            cfg = C.apply_multirow_attn(target, cache_dir=str(tmp_path), verbose=False)
            assert (cfg is not None) is on
        monkeypatch.setenv("MLX_DSPARK_FORCE_MULTIROW", "1")
        assert C.apply_multirow_attn(target, cache_dir=str(tmp_path), verbose=False) is not None
    finally:
        G.MULTIROW_CFG = None


def test_window_text():
    import importlib

    C = importlib.import_module("mlx_dspark.calibrate")
    assert C._window_text({"min_q": 2, "max_q": 8, "min_kv": 1024}) == "q 2-8 from kv 1024"
    assert C._window_text({"causal": {"2": 1024, "3": 1024, "9": 1024, "6": 16384},
                           "tree": {}}) == ("causal q 2-3,9 from kv 1024, q 6 from kv 16384"
                                            " · tree off")
