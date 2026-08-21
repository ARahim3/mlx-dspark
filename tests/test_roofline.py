"""Model-free tests for the roofline/interpretation layer: chip table, weight footprint,
ceiling math, the verdict ladder, the warnings list and the sysctl readers."""

from __future__ import annotations

import sys

import pytest

from mlx_dspark import roofline as R


class TestChipTable:
    def test_family_parsing(self):
        assert R.chip_family("Apple M4 Pro") == ("M4", "M4 Pro")
        assert R.chip_family("Apple M1") == ("M1", "M1")
        assert R.chip_family("Apple M3 Ultra") == ("M3", "M3 Ultra")
        assert R.chip_family(None) == (None, None)
        assert R.chip_family("Intel Iris") == (None, None)

    def test_table_lookup_and_the_m3_pro_regression(self):
        assert R.theoretical_bandwidth("Apple M4 Pro") == (273.0, "table")
        # newer != faster: the M3 Pro regressed vs M2 Pro
        assert R.theoretical_bandwidth("Apple M3 Pro")[0] < R.theoretical_bandwidth("Apple M2 Pro")[0]

    def test_max_binning_by_gpu_cores(self):
        assert R.theoretical_bandwidth("Apple M4 Max", gpu_cores=40) == (546.0, "table-binned")
        assert R.theoretical_bandwidth("Apple M4 Max", gpu_cores=32) == (410.0, "table-binned")
        # unknown core count -> top binning, still labelled binned
        assert R.theoretical_bandwidth("Apple M5 Max") == (614.0, "table-binned")
        assert R.theoretical_bandwidth("Apple M9 Quantum") == (None, "unknown")

    def test_bandwidth_scale_is_like_for_like(self):
        assert R.bandwidth_scale(273.0, 226.0) == 1.0          # the reference machine itself
        assert R.bandwidth_scale(614.0, None) == pytest.approx(2.249, abs=1e-3)
        assert R.bandwidth_scale(None, 452.0) == 2.0          # unknown chip: measured vs measured
        assert R.bandwidth_scale(None, None) is None

    def test_chip_info_shape(self):
        info = R.chip_info("Apple M4 Pro", gpu_cores=20)
        assert info["family"] == "M4 Pro" and info["gpu_cores"] == 20
        assert info["bandwidth_gb_s"] == 273.0 and info["bandwidth_source"] == "table"


class TestWeightFootprint:
    def test_dense_counts_everything_but_the_embedding_gather(self):
        params = [("model.embed_tokens.weight", 1000), ("model.layers.0.mlp.weight", 5000),
                  ("lm_head.weight", 1000)]
        fp = R.weight_footprint(params, {})
        assert fp["total_bytes"] == 7000
        assert fp["active_bytes"] == 6000          # embed gather excluded, lm_head read in full
        assert fp["is_moe"] is False and fp["active_is_estimate"] is False

    def test_tied_embeddings_count_once(self):
        params = [("model.embed_tokens.weight", 1000), ("model.layers.0.mlp.weight", 5000)]
        fp = R.weight_footprint(params, {})
        assert fp["active_bytes"] == 6000          # no lm_head -> the table IS the output head

    def test_moe_active_is_top_k_over_n_of_expert_bytes(self):
        params = [("model.layers.0.mlp.switch_mlp.up_proj.weight", 1000),
                  ("model.layers.0.mlp.shared_expert.up_proj.weight", 100),
                  ("model.layers.0.self_attn.q_proj.weight", 50),
                  ("lm_head.weight", 10)]
        fp = R.weight_footprint(params, {"num_experts": 100, "num_experts_per_tok": 10})
        assert fp["is_moe"] and fp["n_experts"] == 100 and fp["experts_per_tok"] == 10
        assert fp["expert_bytes"] == 1000
        assert fp["active_bytes"] == 100 + 100 + 50 + 10   # 10% of experts + everything else
        assert fp["active_is_estimate"] is False

    def test_moe_without_top_k_in_config_counts_experts_fully_and_says_so(self):
        params = [("m.switch_mlp.w", 1000), ("lm_head.weight", 10)]
        fp = R.weight_footprint(params, {"num_experts": 8})
        assert fp["active_bytes"] == 1010 and fp["active_is_estimate"] is True

    def test_nested_text_config(self):
        params = [("m.switch_mlp.w", 1000), ("lm_head.weight", 10)]
        fp = R.weight_footprint(params, {"text_config": {"num_local_experts": 4,
                                                         "num_experts_per_tok": 1}})
        assert fp["active_bytes"] == 260


class TestRooflineMath:
    def test_ceiling_and_context_growth(self):
        # 10 GB of weights at 250 GB/s -> 25 tok/s at zero context
        r0 = R.roofline(bandwidth_gb_s=250.0, active_bytes=10 * 10**9, kv_bytes_per_token=0)
        assert r0["ceiling_tps"] == pytest.approx(25.0)
        # 64 KB/token at 32k context adds ~2.1 GB per token -> ceiling drops
        r32 = R.roofline(bandwidth_gb_s=250.0, active_bytes=10 * 10**9,
                         kv_bytes_per_token=65536, context=32768)
        assert r32["bytes_per_token"] == 10 * 10**9 + 65536 * 32768
        assert r32["ceiling_tps"] < r0["ceiling_tps"]

    def test_unknown_bandwidth_gives_no_ceiling(self):
        assert R.ceiling_tps(None, 10**9) is None
        assert R.roofline(bandwidth_gb_s=None, active_bytes=1, kv_bytes_per_token=0)["ceiling_tps"] is None

    def test_baseline_mbu_from_a_measured_step(self):
        # 4.3 GB step in 20 ms = 215 GB/s; vs 219 measured -> ~98%
        out = R.baseline_mbu(20.0, int(4.3e9), 219.0)
        assert out["achieved_gb_s"] == pytest.approx(215.0, abs=0.1)
        assert 0.97 < out["mbu"] < 0.99
        assert R.baseline_mbu(None, 1, 1.0) is None
        assert R.baseline_mbu(20.0, 1, None)["mbu"] is None


class TestVerdict:
    def test_healthy_baseline_and_spec_beating_the_roofline(self):
        v = R.verdict(mbu=0.9, ratio_to_ceiling=1.8, mode="dflash", accept_len=4.0,
                      decode_tps=30.0)
        assert v["level"] == "healthy"
        assert any("1.80x the single-stream roofline" in f for f in v["findings"])

    def test_structural_when_far_below_roofline(self):
        v = R.verdict(mbu=0.3, decode_tps=5.0)
        assert v["level"] == "attention"
        assert any("memory pressure" in s for s in v["levers"])

    def test_tiny_model_regime_is_not_alarmed(self):
        v = R.verdict(mbu=0.4, decode_tps=320.0)
        assert v["level"] == "ok" and "small-model" in v["headline"]

    def test_memory_pressure_overrides_everything(self):
        v = R.verdict(mbu=0.9, pressure="critical")
        assert v["level"] == "problem"
        assert "CRITICAL" in v["findings"][0]
        assert v["levers"][0].startswith("Free memory")

    def test_swap_growth_is_the_fits_but_swaps_cliff(self):
        v = R.verdict(mbu=0.8, swap_delta_bytes=512 * 1024 * 1024)
        assert v["level"] == "problem" and "Swap grew" in v["findings"][0]
        # tiny swap jitter is not reported
        assert not any("Swap grew" in f for f in R.verdict(mbu=0.8, swap_delta_bytes=1024)["findings"])

    def test_spec_losing_on_content_suggests_a_smaller_cap(self):
        v = R.verdict(mbu=0.85, ratio_to_ceiling=0.7, mode="dspark", accept_len=1.4)
        assert any("not paying" in f for f in v["findings"])
        assert any("smaller draft cap" in s for s in v["levers"])

    def test_run_shape_findings(self):
        v = R.verdict(mbu=0.85, decay_ratio=0.6, cold=True, context_tokens=95_000,
                      context_window=100_000)
        texts = " ".join(v["findings"])
        assert "slowed to 60%" in texts and "First generation" in texts and "95% full" in texts

    def test_no_numbers_at_all(self):
        v = R.verdict(mbu=None)
        assert v["level"] == "info" and v["findings"] == [] and v["levers"] == []

    def test_baseline_mode_does_not_get_a_spec_reading(self):
        v = R.verdict(mbu=0.9, ratio_to_ceiling=0.95, mode="baseline")
        assert not any("roofline" in f for f in v["findings"])


class TestWarnings:
    def test_quiet_when_all_is_well(self):
        assert R.system_warnings({"pressure": "normal"}) == []
        assert R.system_warnings(None) == []

    def test_pressure_and_load_notes(self):
        rows = R.system_warnings({"pressure": "warn", "swap_used_bytes": 2 * 1024**3},
                                 ["note: the context window defaults to …"])
        assert [r["code"] for r in rows] == ["memory_pressure", "load_note"]
        assert rows[0]["level"] == "attention" and "2.0 GB swapped" in rows[0]["message"]
        assert rows[0]["action"]
        crit = R.system_warnings({"pressure": "critical"})
        assert crit[0]["level"] == "problem"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sysctls")
class TestSysctlReaders:
    def test_system_memory_reads_real_values(self):
        mem = R.system_memory()
        assert mem["total_bytes"] and mem["total_bytes"] > 2**30
        assert mem["pressure"] in ("normal", "warn", "critical")
        assert mem["swap_total_bytes"] >= mem["swap_used_bytes"] >= 0
        assert mem["free_percent"] is None or 0 <= mem["free_percent"] <= 100

    def test_swap_usage_shape(self):
        s = R.swap_usage()
        assert set(s) == {"used_bytes", "total_bytes"}


def test_sysctl_readers_degrade_to_unknown_without_libc(monkeypatch):
    monkeypatch.setattr(R, "_sysctlbyname", lambda name, buf: False)
    assert R.memory_pressure() == {"level": None, "label": "unknown"}
    assert R.swap_usage() == {"used_bytes": 0, "total_bytes": 0}
    mem = R.system_memory()
    assert mem["total_bytes"] is None and mem["pressure"] == "unknown"
