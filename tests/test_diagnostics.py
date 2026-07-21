"""Machine + model inventory. Model-free: nothing here loads weights."""

import os

from mlx_dspark import diagnostics


class TestEnvironment:
    def test_reports_the_fields_the_app_gates_on(self):
        env = diagnostics.environment()
        for key in ("apple_silicon", "metal_ok", "ram_gb", "packages", "version"):
            assert key in env

    def test_package_versions_include_the_mlx_stack(self):
        packages = diagnostics.environment()["packages"]
        assert set(packages) >= {"mlx", "mlx_lm", "mlx_vlm", "transformers"}

    def test_wired_limit_hint_is_a_runnable_command_or_absent(self):
        hint = diagnostics.environment()["wired_limit_hint"]
        assert hint is None or hint.startswith("sudo sysctl iogpu.wired_limit_mb=")


class TestRamParsing:
    def test_parses_registry_ram_strings(self):
        assert diagnostics._parse_ram_gb("~15 GB") == 15.0
        assert diagnostics._parse_ram_gb("~8 GB") == 8.0

    def test_returns_none_for_unparseable(self):
        assert diagnostics._parse_ram_gb("") is None
        assert diagnostics._parse_ram_gb("lots") is None


class TestInventory:
    def test_every_registry_entry_is_reported(self):
        from mlx_dspark.load import REGISTRY

        assert len(diagnostics.model_inventory(ram_gb=48)) == len(REGISTRY)

    def test_fit_uses_a_headroom_margin(self):
        """A model must not be called a fit just because it technically fits in the last GB —
        the OS and everything else the user has open need room too."""
        rows = {r["id"]: r for r in diagnostics.model_inventory(ram_gb=16)}
        gemma = rows["gemma-4-12b"]                 # ~15 GB
        assert gemma["ram_gb"] == 15.0
        assert gemma["fits"] is False              # 15 > 16 * 0.8

    def test_small_model_fits_a_small_machine(self):
        rows = {r["id"]: r for r in diagnostics.model_inventory(ram_gb=16)}
        assert rows["qwen3-4b"]["fits"] is True     # ~8 GB

    def test_fits_is_none_when_ram_is_unknown(self):
        for row in diagnostics.model_inventory(ram_gb=None):
            assert row["fits"] is None

    def test_rows_expose_the_target_drafter_pair(self):
        rows = {r["id"]: r for r in diagnostics.model_inventory(ram_gb=48)}
        qwen = rows["qwen3-4b"]
        assert qwen["target"] == "mlx-community/Qwen3-4B-8bit"
        assert qwen["dspark_drafter"] == "deepseek-ai/dspark_qwen3_4b_block7"

    def test_ready_requires_both_halves_present(self):
        for row in diagnostics.model_inventory(ram_gb=48):
            assert row["ready"] == (row["target_installed"] and row["drafter_installed"])


class TestIsLocal:
    def test_absent_repo_is_not_local(self):
        assert diagnostics._is_local("definitely-not/a-real-model-xyzzy") is False

    def test_none_is_not_local(self):
        assert diagnostics._is_local(None) is False

    def test_existing_directory_is_local(self, tmp_path):
        assert diagnostics._is_local(str(tmp_path)) is True

    def test_gguf_scheme_is_unwrapped(self, tmp_path, monkeypatch):
        """`gguf:{repo}/{file}.gguf` drafters must be checked against the repo, not the URL."""
        (tmp_path / "some-repo").mkdir()
        monkeypatch.setattr(os.path, "expanduser",
                            lambda p: str(tmp_path) if p == "~/.cache/mlx_dspark/models" else p)
        assert diagnostics._is_local("gguf:owner/some-repo/weights.gguf") is True

    def test_does_not_download(self, monkeypatch):
        """The inventory runs on every app launch — it must never touch the network."""
        import mlx_dspark.load as load

        def boom(*a, **k):
            raise AssertionError("snapshot_download called from the inventory")

        monkeypatch.setattr(load, "snapshot_download", boom, raising=False)
        diagnostics.model_inventory(ram_gb=48)


class TestDoctor:
    def test_reports_ok_and_problems(self):
        report = diagnostics.doctor()
        assert isinstance(report["ok"], bool)
        assert isinstance(report["problems"], list)
        assert report["ok"] == (not report["problems"])

    def test_includes_environment_and_models(self):
        report = diagnostics.doctor()
        assert "environment" in report
        assert "models" in report
        assert len(report["models"]) > 0
