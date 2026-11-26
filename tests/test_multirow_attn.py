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

    times = iter([2.0, 1.0] * 64)          # (mlx, kernel) per timed probe: the kernel wins
    monkeypatch.setattr(mra, "multirow_sdpa", limited)
    monkeypatch.setattr(mra, "_chain_ms", lambda fn, q, kvs: next(times))
    monkeypatch.setattr(mra, "PROBE_KV", (1024,))
    window, report = mra.measure_window(32, 2, 128, per_q=False)
    assert window == {"min_q": 2, "max_q": 6, "min_kv": 1024}
    assert report["points"]["q8@1024"] == "unavailable"
    assert "threads per threadgroup" in report["unavailable"]

    monkeypatch.setattr(mra, "_race_pair", lambda fx, fm, q, kvs, kv: (2.0, True))
    window_pq, report_pq = mra.measure_window(32, 2, 128, per_q=True)
    assert window_pq == {"min_q": 2, "max_q": 7, "min_kv": 1024}
    assert report_pq["probe"] == 3


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
    monkeypatch.setattr(C, "_gpu_gen", lambda: 16)
    dev = mx.device_info().get("device_name", "unknown")
    key = lambda s: f"v{C.SCHEMA}|{dev}|mlx{mx.__version__}|mra1|{s}"  # noqa: E731
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
    monkeypatch.setattr(C, "_gpu_gen", lambda: 16)
    dev = mx.device_info().get("device_name", "unknown")
    key = lambda s: f"v{C.SCHEMA}|{dev}|mlx{mx.__version__}|mra1|{s}"  # noqa: E731
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


def test_noncontiguous_window_and_per_q_min_kv_gating():
    # M5 NAX-gap pattern: q=2..5, 9..15 win (with q=15 needing deeper KV), q=6..8, 16 lose causal,
    # while tree_mask recovers q=6..8.
    w = Window(
        8, 2, 64,
        q_min_kv={"2": 64, "4": 64, "5": 64, "9": 64, "15": 128},
        tree_q_min_kv={"2": 64, "4": 64, "5": 64, "6": 64, "7": 64, "8": 64, "9": 64, "15": 128},
    )
    assert hash(w)  # frozen & hashable after dict normalization
    cfg = MultirowConfig((w,))
    for ql, L, exp in [(2, 64, True), (3, 128, False), (5, 64, True), (6, 128, False),
                       (8, 128, False), (9, 64, True), (15, 64, False), (15, 128, True),
                       (16, 128, False)]:
        q, k, v = _mk(1, 8, 2, ql, L, 64)
        assert eligible(q, k, v, "causal", None, cfg) is exp
    # Tree mask path admits q=6..8 while causal passes them through bit-exact to stock.
    q6, k6, v6 = _mk(1, 8, 2, 6, 128, 64)
    assert not eligible(q6, k6, v6, "causal", None, cfg)
    assert eligible(q6, k6, v6, None, None, cfg, tree=True)
    stock = mra._ORIG_SDPA(q6, k6, v6, scale=0.125, mask="causal")
    rows = mx.arange(128 - 6, 128).reshape(6, 1)
    cols = mx.arange(128).reshape(1, 128)
    arr_mask = (cols <= rows).reshape(1, 1, 6, 128)
    anc = mx.array([(1 << (i + 1)) - 1 for i in range(6)], dtype=mx.uint32)
    with multirow_attention(cfg), mra.tree_mask(arr_mask, anc):
        causal_out = mx.fast.scaled_dot_product_attention(q6, k6, v6, scale=0.125, mask="causal")
        tree_out = mx.fast.scaled_dot_product_attention(q6, k6, v6, scale=0.125, mask=arr_mask)
    assert float(mx.abs(causal_out.astype(mx.float32) - stock.astype(mx.float32)).max()) == 0.0
    assert _err(tree_out, q6, k6, v6, True) < 0.02


def test_measure_window_noncontiguous_with_tree_recovery_and_json_roundtrip(tmp_path, monkeypatch):
    import importlib

    C = importlib.import_module("mlx_dspark.calibrate")
    monkeypatch.setattr(C, "_gpu_gen", lambda: 17)
    monkeypatch.setattr(mra, "PER_Q_PROBE_Q", (2, 4, 6, 8, 9, 16))
    monkeypatch.setattr(mra, "PROBE_KV", (1024, 4096))

    # Simulate M5 NAX gaps at q=6,8,16 for causal, while tree_mask wins at q=6,8 (still loses at 16).
    causal_win = {(2, 1024), (2, 4096), (4, 4096), (9, 1024), (9, 4096)}
    tree_win = {(6, 4096), (8, 4096)}
    seen_causal = set()

    def fake_race(fx, fm, q, kvs, kv):
        ql = int(q.shape[2])
        if (ql, kv) not in seen_causal:
            seen_causal.add((ql, kv))
            won = (ql, kv) in causal_win
        else:
            won = (ql, kv) in tree_win
        return (2.0, True) if won else (0.67, False)

    monkeypatch.setattr(mra, "_race_pair", fake_race)
    win, report = mra.measure_window(8, 2, 64, per_q=True)
    assert report["probe"] == 3
    assert win == {
        "q_min_kv": {"2": 1024, "4": 4096, "9": 1024},
        "tree_q_min_kv": {"2": 1024, "4": 4096, "6": 4096, "8": 4096, "9": 1024},
    }
    cache = str(tmp_path)
    dev = mx.device_info().get("device_name", "unknown")
    key = f"v{C.SCHEMA}|{dev}|mlx{mx.__version__}|mra2|8x2x64"
    C.save_cached(key, {"window": win, "report": {"probe": 3, "max_sg": mra.MAX_SG}}, cache)
    loaded = C.multirow_windows([(8, 2, 64)], cache_dir=cache, verbose=False)
    assert len(loaded) == 1
    assert loaded[0].min_kv_for(2) == 1024
    assert loaded[0].min_kv_for(4) == 4096
    assert loaded[0].min_kv_for(6) is None
    assert loaded[0].min_kv_for(6, tree=True) == 4096
    assert loaded[0].min_kv_for(16, tree=True) is None


def test_m5_uses_mra2_cache_key_and_bypasses_stale_mra1(tmp_path, monkeypatch):
    import importlib

    C = importlib.import_module("mlx_dspark.calibrate")
    dev = mx.device_info().get("device_name", "unknown")
    cache = str(tmp_path)
    mra1_key = f"v{C.SCHEMA}|{dev}|mlx{mx.__version__}|mra1|8x2x64"
    stale_win = {"min_q": 2, "max_q": 4, "min_kv": 16384}
    C.save_cached(mra1_key, {"window": stale_win, "report": {"probe": 2, "max_sg": mra.MAX_SG}}, cache)

    seen_per_q = []

    def fake(hq, hk, d, **kw):
        seen_per_q.append(kw.get("per_q"))
        return {"q_min_kv": {"2": 1024, "9": 4096}}, {"probe": 3, "max_sg": mra.MAX_SG}

    monkeypatch.setattr(mra, "measure_window", fake)
    # On g16 (M4), mra1 stands untouched and never re-probes.
    monkeypatch.setattr(C, "_gpu_gen", lambda: 16)
    wins16 = C.multirow_windows([(8, 2, 64)], cache_dir=cache, verbose=False)
    assert seen_per_q == []
    assert wins16[0].max_q == 4

    # On g17 (M5), stale mra1 (e.g. from FORCE_MULTIROW) is ignored and mra2 is calibrated.
    monkeypatch.setattr(C, "_gpu_gen", lambda: 17)
    wins17 = C.multirow_windows([(8, 2, 64)], cache_dir=cache, verbose=False)
    assert seen_per_q == [True]
    assert wins17[0].q_min_kv == ((2, 1024), (9, 4096))
    assert C.load_cached(f"v{C.SCHEMA}|{dev}|mlx{mx.__version__}|mra2|8x2x64", cache) is not None
    # mra1 remains untouched so older builds reading mra1 never see q_min_kv.
    assert C.load_cached(mra1_key, cache)["window"] == stale_win


def test_per_q_borderline_confirmation_rejects_noisy_flip(monkeypatch):
    monkeypatch.setattr(mra, "PER_Q_PROBE_Q", (2, 4, 6))
    monkeypatch.setattr(mra, "PROBE_KV", (1024,))
    # Exercises the real _race_pair -> _abba_race confirmation path:
    # q=2 is borderline on pass 1 (1.18 >= 1.15) then loses confirmation (1.05).
    # q=4 is borderline on pass 1 (1.18 >= 1.15) and passes confirmation (1.21).
    # q=6 is a clear loser (1.08 < 1.15) and does NOT trigger a confirmation pass.
    seq = iter([
        (1.18, 1.15), (1.05, 1.02),   # q=2: pass 1 + confirm (fails)
        (1.18, 1.15), (1.21, 1.18),   # q=4: pass 1 + confirm (passes)
        (1.08, 1.05),                 # q=6: clear loser (no confirm)
        (0.67, 0.65),                 # q=2 tree_miss (loses)
        (0.67, 0.65),                 # q=6 tree_miss (loses)
    ])
    monkeypatch.setattr(mra, "_abba_race", lambda *a, **kw: next(seq))
    win, rep = mra.measure_window(8, 2, 64, per_q=True)
    assert win == {"q_min_kv": {"4": 1024}}
    assert rep["points"]["q2@1024"] == 1.05
    assert rep["points"]["q6@1024"] == 1.08


def test_abba_race_runs_interleaved_and_confirms_inconsistent_blocks(monkeypatch):
    q, k, v = _mk(1, 8, 2, 2, 64, 64)
    fx = lambda qq, kk, vv: mra._ORIG_SDPA(qq, kk, vv, scale=0.125, mask="causal")  # noqa: E731
    fm = lambda qq, kk, vv: mra.multirow_sdpa(qq, kk, vv, scale=0.125, causal=True)  # noqa: E731
    med, mn = mra._abba_race(fx, fm, q, [(k, v)], iters=1, blocks=3)
    assert med >= mn > 0.0

    # Even if median >= thr (1.25), an inconsistent block (min < thr) triggers re-check.
    calls = iter([(1.25, 1.10), (1.12, 1.05)])
    monkeypatch.setattr(mra, "_abba_race", lambda *a, **kw: next(calls))
    r, won = mra._race_pair(fx, fm, q, [(k, v)], 1024, thr=1.15)
    assert won is False and r == 1.12


def test_tree_miss_exception_recorded_in_report(monkeypatch):
    monkeypatch.setattr(mra, "PER_Q_PROBE_Q", (2, 6, 8))
    monkeypatch.setattr(mra, "PROBE_KV", (1024,))
    real = mra.multirow_sdpa

    def fail_tree(q, k, v, *, scale, causal, tree_anc=None):
        if tree_anc is not None:
            raise RuntimeError("boom")
        return real(q, k, v, scale=scale, causal=causal)

    # q=2 wins causal (ratio 2.0); q=6,8 lose causal (0.5) -> q=6 errors in tree and stops before q=8.
    seq = iter([(2.0, True), (0.5, False), (0.5, False)])
    monkeypatch.setattr(mra, "multirow_sdpa", fail_tree)
    monkeypatch.setattr(mra, "_race_pair", lambda fx, fm, q, kvs, kv: next(seq))
    win, rep = mra.measure_window(8, 2, 64, per_q=True)
    assert win == {"min_q": 2, "max_q": 2, "min_kv": 1024}
    assert rep["points"]["tree_q6@1024"] == "unavailable"
    assert "tree_q8@1024" not in rep["points"]
    assert "RuntimeError: boom" in rep["tree_unavailable"]


def test_measure_window_defaults_to_legacy_strict_win(monkeypatch):
    monkeypatch.setattr(mra, "PROBE_Q", (2, 4))
    monkeypatch.setattr(mra, "PROBE_KV", (1024,))
    # Exact WIN boundary (1.05 / 1.0 == WIN) must NOT win under legacy strict `t_x > WIN * t_m`.
    seq = iter([1.2, 1.0, mra.WIN, 1.0])
    monkeypatch.setattr(mra, "_chain_ms", lambda fn, q, kvs: next(seq))
    win, rep = mra.measure_window(8, 2, 64)
    assert rep["probe"] == 2
    assert win == {"min_q": 2, "max_q": 2, "min_kv": 1024}


def test_apply_multirow_attn_gen_gate_allows_m5_and_blocks_g18(tmp_path, monkeypatch):
    import importlib
    from types import SimpleNamespace

    C = importlib.import_module("mlx_dspark.calibrate")
    from mlx_dspark import generate as _gen

    fake_target = SimpleNamespace(model=SimpleNamespace(
        args=SimpleNamespace(num_attention_heads=8, num_key_value_heads=2, head_dim=64)))
    monkeypatch.setattr(mra, "measure_window",
                        lambda hq, hk, d, **kw: ({"min_q": 2, "max_q": 8, "min_kv": 1024},
                                                 {"probe": 3, "max_sg": mra.MAX_SG}))
    monkeypatch.setattr(C, "_gpu_gen", lambda: 16)
    assert C.apply_multirow_attn(fake_target, cache_dir=str(tmp_path), verbose=False) is not None
    monkeypatch.setattr(C, "_gpu_gen", lambda: 17)
    cfg = C.apply_multirow_attn(fake_target, cache_dir=str(tmp_path), verbose=False)
    assert cfg is not None and _gen.MULTIROW_CFG is cfg
    monkeypatch.setattr(C, "_gpu_gen", lambda: 18)
    monkeypatch.delenv("MLX_DSPARK_FORCE_MULTIROW", raising=False)
    assert C.apply_multirow_attn(fake_target, cache_dir=str(tmp_path), verbose=False) is None
    assert _gen.MULTIROW_CFG is None
    monkeypatch.setenv("MLX_DSPARK_FORCE_MULTIROW", "1")
    assert C.apply_multirow_attn(fake_target, cache_dir=str(tmp_path), verbose=False) is not None
    _gen.MULTIROW_CFG = None


def test_format_multirow_window():
    import importlib

    C = importlib.import_module("mlx_dspark.calibrate")
    assert C._format_multirow_window({"min_q": 2, "max_q": 8, "min_kv": 1024}) == "q 2-8 from kv 1024"
    assert C._format_multirow_window({
        "q_min_kv": {"4": 1024, "5": 1024, "9": 1024, "14": 4096, "3": 16384},
        "tree_q_min_kv": {"4": 1024, "5": 1024, "6": 1024, "7": 1024, "8": 1024, "9": 1024},
    }) == "q 4-5,9 from kv 1024; q 14 from kv 4096; q 3 from kv 16384 (tree: q 4-9 from kv 1024)"
    assert C._format_multirow_window({
        "q_min_kv": {},
        "tree_q_min_kv": {"6": 4096},
    }) == "causal off (tree: q 6 from kv 4096)"
