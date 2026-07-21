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
