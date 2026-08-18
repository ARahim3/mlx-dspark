"""Model-free tests for the auto-cap machinery: interpolation, the acceptance EWMA,
cap choice against synthetic (measured-shaped) cost curves, and the disk cache."""

from __future__ import annotations

from mlx_dspark.calibrate import (
    CapController,
    _cache_key,
    _interp,
    load_cached,
    save_cached,
)


def test_interp_exact_between_and_extrapolate():
    curve = {2: 10.0, 4: 20.0, 8: 60.0}
    assert _interp(curve, 2) == 10.0
    assert _interp(curve, 3) == 15.0            # linear between 2 and 4
    assert _interp(curve, 1) == 10.0            # clamped below
    assert _interp(curve, 10) == 80.0           # extrapolates the last slope (10/step)
    assert _interp({3: 7.0}, 9) == 7.0          # single point -> constant


def _gemma_like_controller(**kw):
    # measured M4 Pro shape: ~flat to width 3, knee at 4 (NOTES "Perf pass 2")
    verify = {2: 62.5, 3: 67.6, 4: 86.4, 5: 105.6, 6: 124.8, 7: 144.0, 8: 163.2}
    drafter = {c: 7.0 + 0.5 * c for c in range(1, 8)}
    return CapController(verify, drafter, max_cap=7, **kw)


def test_controller_prefers_pre_knee_cap_at_typical_acceptance():
    ctrl = _gemma_like_controller()
    ctrl.p = 0.65                                # ~measured chat acceptance
    best = max(range(1, 8), key=ctrl.rate)
    assert best in (1, 2, 3)                     # never past the knee at this acceptance
    # near-perfect acceptance should justify crossing the knee
    ctrl.p = 0.99
    assert max(range(1, 8), key=ctrl.rate) >= 4


def test_controller_ewma_and_censoring():
    ctrl = _gemma_like_controller(alpha=0.5)
    p0 = ctrl.p
    ctrl.update(accepted_n=2, cap_used=2)        # full acceptance: successes only (censored)
    assert ctrl.p > p0
    p1 = ctrl.p
    ctrl.update(accepted_n=0, cap_used=2)        # immediate reject: one failure
    assert ctrl.p < p1


def test_controller_repick_moves_cap_with_hysteresis():
    ctrl = _gemma_like_controller(alpha=0.4, repick_every=1)
    ctrl.cap = 4                                 # start past the knee on purpose
    for _ in range(20):
        ctrl.update(accepted_n=0, cap_used=ctrl.cap)   # rejections -> low p
    assert ctrl.cap <= 2                         # migrates below the knee
    for _ in range(60):
        ctrl.update(accepted_n=ctrl.cap, cap_used=ctrl.cap)   # perfect acceptance
    assert ctrl.cap >= 4                         # climbs back over the knee


def test_controller_update_at_any_cap_feeds_one_estimate():
    a = _gemma_like_controller(alpha=0.3)
    b = _gemma_like_controller(alpha=0.3)
    a.update(2, 2)
    a.update(1, 2)
    b.update(2, 4)                               # same successes at a different cap...
    b.update(1, 4)                               # ...but observed failures differ (censoring)
    assert a.p != b.p                            # cap_used matters only via the failure obs
    c = _gemma_like_controller(alpha=0.3)
    c.update(2, 2)
    c.update(1, 1)                               # full accept at cap1 == success, censored
    assert c.p > 0.65


def test_static_best_tracks_the_measured_curve_shape_not_a_constant():
    """The default cap is per-(model x quant x chip x mlx version). These are the REAL
    measured M4 Pro / mlx 0.32 curves for one model at three quantizations — one registry
    row serves all three, so no per-entry constant can be right."""
    from mlx_dspark.calibrate import CapController
    drafter = {c: 10.4 + 0.9 * c for c in range(1, 8)}
    ornith = {
        # 4-bit: rises from width 3 -> stay shallow
        "4bit": ({1: 21.5, 2: 22.0, 3: 25.4, 4: 31.8, 5: 37.4, 6: 45.6, 7: 56.9, 8: 57.8}, 2),
        # 8-bit: flat through width 5, knee at 6 -> cap 4
        "8bit": ({1: 36.7, 2: 37.0, 3: 37.8, 4: 38.5, 5: 40.6, 6: 51.8, 7: 61.9, 8: 62.7}, 4),
        # bf16: 2x cliff at width 2 then FLAT -> once the cliff is paid, width is free.
        # (>=6 rather than an exact value: the argmax there is flat-topped, so the drafter
        # curve and per-round overhead decide between 6 and 7. "Go wide" is the claim.)
        "bf16": ({1: 67.8, 2: 134.4, 3: 135.0, 4: 136.8, 5: 136.8, 6: 137.3, 7: 138.1,
                  8: 139.1}, 6),
    }
    picks = {}
    for name, (curve, _want) in ornith.items():
        ctrl = CapController(curve, drafter, max_cap=7)
        ctrl.p = 0.65
        picks[name] = ctrl.static_best()
    assert picks["4bit"] == ornith["4bit"][1], picks
    assert picks["8bit"] == ornith["8bit"][1], picks
    assert picks["bf16"] >= ornith["bf16"][1], picks
    # the ordering is the load-bearing part: same model, same registry row, three answers
    assert picks["4bit"] < picks["8bit"] < picks["bf16"], picks


def test_static_prior_reproduces_the_measured_optima():
    """Locks `STATIC_PRIOR_P` against the real M4 Pro / mlx 0.32 curves and the caps that
    were actually measured fastest. The prior is load-bearing: at the controller's 0.65
    parking prior this picks cap 1 for Bonsai (measured best 2) — i.e. WORSE than the old
    hardcoded 2 — and 6 for Ornith-bf16 (7). Re-fit if the curves are regenerated."""
    from mlx_dspark.calibrate import STATIC_PRIOR_P, CapController
    # (verify curve, drafter curve, per-round overhead ms, cap measured fastest end-to-end).
    # All four inputs are the REAL cached measurements — the drafter curve and overhead
    # matter as much as the verify curve here (a synthetic drafter cost flips Bonsai to 1).
    cases = {
        "bonsai-27b-2bit": (
            {1: 40.8, 2: 53.2, 3: 72.0, 4: 89.6, 5: 107.9},
            {1: 7.7, 2: 7.9, 3: 8.3, 4: 9.7}, 7.9, 2),
        "qwen3.6-27b-4bit": (
            {1: 68.4, 2: 69.6, 3: 75.2, 4: 93.8, 5: 112.3, 6: 141.5, 7: 175.3, 8: 176.3},
            {1: 14.4, 2: 14.8, 3: 15.3, 4: 16.5, 5: 17.6, 6: 19.3, 7: 21.3}, 4.7, 2),
        "ornith-9b-8bit": (
            {1: 36.7, 2: 37.0, 3: 37.8, 4: 38.5, 5: 40.6, 6: 51.8, 7: 61.9, 8: 62.7},
            {1: 10.4, 2: 10.8, 3: 11.4, 4: 12.5, 5: 13.5, 6: 14.9, 7: 16.5}, 0.3, 4),
        "qwen3-4b-8bit": (
            {1: 21.4, 2: 22.0, 3: 22.2, 4: 22.8, 5: 24.1, 6: 29.6, 7: 34.5, 8: 36.2},
            {1: 5.4, 2: 5.5, 3: 5.8, 4: 6.3, 5: 6.9, 6: 8.2, 7: 8.0}, 0.0, 4),
        "gemma-12b-8bit": (
            {1: 57.9, 2: 58.4, 3: 59.6, 4: 60.8, 5: 63.7, 6: 81.9, 7: 97.0, 8: 97.7},
            {1: 11.6, 2: 11.9, 3: 12.6, 4: 13.8, 5: 14.9, 6: 16.1, 7: 18.0}, 4.7, 4),
    }
    for name, (verify, drafter, overhead, want) in cases.items():
        ctrl = CapController(verify, drafter, max_cap=min(7, max(drafter)),
                             overhead_ms=overhead)
        assert ctrl.static_best(STATIC_PRIOR_P) == want, name


def test_static_best_goes_wide_on_a_flat_curve_and_shallow_on_a_steep_one():
    """Sanity on the two extremes, independent of any measured data."""
    from mlx_dspark.calibrate import CapController
    flat = CapController(dict.fromkeys(range(1, 9), 50.0), 5.0, max_cap=7)
    flat.p = 0.8
    assert flat.static_best() == 7                # free width -> draft as deep as allowed
    steep = CapController({w: 20.0 * w for w in range(1, 9)}, 5.0, max_cap=7)
    steep.p = 0.8
    assert steep.static_best() == 1               # every row costs a full step -> barely draft


def test_static_cap_falls_back_rather_than_raising():
    """Calibration is an optimization, never a gate: an unsupported mode or a missing
    drafter must degrade to the historical default, not break generation."""
    from mlx_dspark.calibrate import static_cap
    assert static_cap(None, None, mode="lookup", target_repo="x", drafter_repo=None) == 2
    assert static_cap(None, None, mode="dspark", target_repo="x", drafter_repo=None) == 2
    assert static_cap(object(), object(), mode="dspark", target_repo="x",
                      drafter_repo="y", fallback=3) == 3      # calibrate() blows up -> 3


def test_disk_cache_roundtrip(tmp_path):
    key = _cache_key("dspark", "org/Model-8bit", "org/drafter")
    assert "dspark" in key and "Model-8bit" in key
    entry = {"verify": {"2": 10.0}, "drafter": {"1": 3.0}}
    assert load_cached(key, str(tmp_path)) is None
    save_cached(key, entry, str(tmp_path))
    assert load_cached(key, str(tmp_path)) == entry
    save_cached("other", {"verify": {}}, str(tmp_path))   # merge keeps the first entry
    assert load_cached(key, str(tmp_path)) == entry


def test_knee_width_convex_curve():
    from mlx_dspark.calibrate import knee_width
    # cheap 1..3 (+5/step), then a jump at width 4 (+19) — the qmm knee
    assert knee_width({1: 5, 2: 10, 3: 15, 4: 34, 5: 53}) == 4


def test_knee_width_linear_no_knee():
    from mlx_dspark.calibrate import knee_width
    assert knee_width({1: 5, 2: 10, 3: 15, 4: 20, 5: 25}) == 5   # no jump -> top width


def test_knee_width_flat_step_does_not_fake_a_knee():
    """Regression: a flat step used to collapse the running baseline to ~0, after which any
    +1 ms read as a jump. Measured Qwen3-4B 8-bit (mlx 0.32) reported a knee at 4 while the
    curve is still flat there; the real jump is 24 -> 30 at width 6."""
    from mlx_dspark.calibrate import knee_width
    measured = {1: 21.0, 2: 22.0, 3: 22.0, 4: 23.0, 5: 24.0, 6: 30.0, 7: 34.0, 8: 36.0}
    assert knee_width(measured) == 6


def test_knee_width_detects_a_cliff_at_the_first_step():
    """Regression: the first delta was the baseline and so could never be reported as the
    knee — but that is exactly the bf16 shape, where an unquantized matmul reads the weight
    stream twice from width 2 and then stays flat (measured Ornith-9B ctx512)."""
    from mlx_dspark.calibrate import knee_width
    assert knee_width({1: 67.8, 2: 134.4, 4: 136.0, 8: 139.0}) == 2


def test_knee_width_noise_does_not_trigger_on_a_flat_cheap_region():
    """+-1 ms of measurement noise on a nearly-flat region must not read as leaving it."""
    from mlx_dspark.calibrate import knee_width
    measured = {1: 36.7, 2: 37.5, 3: 38.4, 4: 39.5, 5: 40.6, 6: 52.0, 7: 58.0}
    assert knee_width(measured) == 6


def test_drafter_recommendation_small_knee_is_dspark():
    from mlx_dspark.calibrate import drafter_recommendation
    rec = drafter_recommendation({1: 5, 2: 10, 3: 15, 4: 34, 5: 53}, dflash_block=16)
    assert rec["knee_width"] == 4 and not rec["dflash_full_block_viable"]
    assert rec["recommend"] == "dspark"


def test_drafter_recommendation_wide_knee_reopens_dflash():
    from mlx_dspark.calibrate import drafter_recommendation
    # a hypothetical M5-class curve: cheap all the way to width ~18
    curve = {w: 5.0 + 0.2 * w for w in range(1, 18)}
    curve[18] = curve[17] + 20.0
    rec = drafter_recommendation(curve, dflash_block=16)
    assert rec["dflash_full_block_viable"] and rec["recommend"] == "dflash-on-structured"


def test_cap_for_shrinks_under_batched_grid():
    from mlx_dspark.calibrate import CapController
    # single-stream curve: modest slope; B=4 grid: wide verify much pricier -> cap shrinks
    verify = {2: 20.0, 3: 25.0, 4: 40.0, 5: 60.0}
    grid = {"4": {"2": 60.0, "3": 90.0, "4": 130.0, "5": 170.0}}  # str keys, as disk-cached
    c = CapController(verify, 5.0, max_cap=4, verify_grid=grid)
    assert c.cap_for(1) == c.cap                     # no batch -> live single-stream cap
    assert c.cap_for(4) == 1                         # argmax under the pricier B=4 curve
    assert c.cap_for(2) == 1                         # nearest measured B >= 2 is 4
    assert c.cap_for(9) == 1                         # beyond the grid -> top measured B
    assert c.info()["batch_caps"] == {4: 1}


def test_cap_for_without_grid_falls_back():
    from mlx_dspark.calibrate import CapController
    c = CapController({2: 20.0, 3: 25.0}, 5.0, max_cap=4)
    assert c.cap_for(4) == c.cap
    assert "batch_caps" not in c.info()


class TestCachedCurveEntry:
    """The /calibration reader must find curves saved under the "|smm"-tagged key —
    reading only the untagged key showed 'not calibrated' on every kernel-on machine
    (v0.12.0 regression, found via the app's permanently-empty Curves tab)."""

    def _base(self, tmp_path):
        from mlx_dspark.calibrate import _cache_key
        return _cache_key("dspark", "T", "D"), str(tmp_path)

    def test_finds_the_smm_tagged_entry_when_kernel_is_live(self, tmp_path):
        from mlx_dspark.calibrate import cached_curve_entry
        base, cache = self._base(tmp_path)
        save_cached(base + "|smm", {"verify": {"1": 1.0}}, cache)
        key, entry = cached_curve_entry("dspark", "T", "D", smm_live=True, cache_dir=cache)
        assert entry is not None and key == base + "|smm"

    def test_falls_back_across_kernel_state(self, tmp_path):
        # Only the tagged entry exists but the kernel is off now (--no-small-m after a
        # kernel-on calibration): stale-but-real curves beat an empty screen, and the
        # returned key names which variant was found.
        from mlx_dspark.calibrate import cached_curve_entry
        base, cache = self._base(tmp_path)
        save_cached(base + "|smm", {"verify": {"1": 1.0}}, cache)
        key, entry = cached_curve_entry("dspark", "T", "D", smm_live=False, cache_dir=cache)
        assert entry is not None and key == base + "|smm"

    def test_prefers_the_variant_matching_the_live_state(self, tmp_path):
        from mlx_dspark.calibrate import cached_curve_entry
        base, cache = self._base(tmp_path)
        save_cached(base, {"verify": {"1": 2.0}}, cache)
        save_cached(base + "|smm", {"verify": {"1": 1.0}}, cache)
        assert cached_curve_entry("dspark", "T", "D",
                                  smm_live=True, cache_dir=cache)[0] == base + "|smm"
        assert cached_curve_entry("dspark", "T", "D",
                                  smm_live=False, cache_dir=cache)[0] == base

    def test_never_measured_returns_base_key_and_none(self, tmp_path):
        from mlx_dspark.calibrate import cached_curve_entry
        base, cache = self._base(tmp_path)
        key, entry = cached_curve_entry("dspark", "T", "D", smm_live=True, cache_dir=cache)
        assert entry is None and key == base
