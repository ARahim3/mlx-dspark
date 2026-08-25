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

    def test_rows_carry_the_measured_speedup_hook(self):
        """The model pickers show the measured ratio next to each pair — the number that
        answers 'why this one'. Every registered pair has been benchmarked (registry policy),
        so every row must carry it."""
        for row in diagnostics.model_inventory(ram_gb=48):
            assert row["speedup"]

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

    def test_lmstudio_mlx_dir_is_local(self, tmp_path, monkeypatch):
        """Issue #28: 'installed?' must agree with the loader, which reads LM Studio's folder."""
        import mlx_dspark.load as load

        model = tmp_path / "org" / "Some-Model-MLX-4bit"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}")
        (model / "model.safetensors").write_bytes(b"w")
        monkeypatch.setattr(load, "LMSTUDIO_ROOTS", (str(tmp_path),))
        assert diagnostics._is_local("org/Some-Model-MLX-4bit") is True
        assert diagnostics._local_dir("org/Some-Model-MLX-4bit") == str(model)
        assert diagnostics._is_local("org/Absent-Model") is False

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


class TestMemoryInfo:
    def test_reports_allocator_state_when_mlx_present(self):
        info = diagnostics.memory_info()
        if info["available"]:
            for key in ("active_bytes", "peak_bytes", "cache_bytes"):
                assert isinstance(info[key], int) and info[key] >= 0
        else:                                       # non-mlx environment: shape still valid
            assert set(info) == {"available"}


def _write(path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


class TestInstalledModels:
    """The on-disk scan behind the app's 'what's already on this Mac' list."""

    def _fake_caches(self, tmp_path):
        hub = tmp_path / "hub"
        plain = tmp_path / "plain"
        # HF layout: blobs hold the bytes, snapshots symlink into them.
        target = hub / "models--mlx-community--Qwen3-4B-8bit"
        _write(target / "blobs" / "abc", 1000)
        snap = target / "snapshots" / "deadbeef"
        snap.mkdir(parents=True)
        os.symlink(target / "blobs" / "abc", snap / "model.safetensors")
        _write(hub / "models--deepseek-ai--dspark_qwen3_4b_block7" / "blobs" / "d", 200)
        _write(plain / "Ornith-1.0-9B-8bit" / "model.safetensors", 500)
        (hub / "datasets--something").mkdir()       # non-model entry: ignored
        return str(hub), str(plain)

    def test_lists_hub_and_plain_dir_models(self, tmp_path):
        hub, plain = self._fake_caches(tmp_path)
        rows = {r["repo"]: r for r in diagnostics.installed_models(hub, plain)}
        assert "mlx-community/Qwen3-4B-8bit" in rows
        assert "Ornith-1.0-9B-8bit" in rows
        assert "datasets/something" not in rows

    def test_symlinks_are_not_double_counted(self, tmp_path):
        """HF snapshots are symlink farms into blobs/ — following them would double every
        weight file and the disk numbers would be off by ~2x."""
        hub, plain = self._fake_caches(tmp_path)
        rows = {r["repo"]: r for r in diagnostics.installed_models(hub, plain)}
        row = rows["mlx-community/Qwen3-4B-8bit"]
        assert row["size_bytes"] < 1200              # 1000 blob + tiny symlink, not 2000

    def test_drafters_are_marked_not_offered_as_models(self, tmp_path):
        hub, plain = self._fake_caches(tmp_path)
        rows = {r["repo"]: r for r in diagnostics.installed_models(hub, plain)}
        assert rows["deepseek-ai/dspark_qwen3_4b_block7"]["kind"] == "drafter"
        assert rows["mlx-community/Qwen3-4B-8bit"]["kind"] == "model"

    def test_registry_pairing_is_annotated_quant_agnostically(self, tmp_path):
        hub, plain = self._fake_caches(tmp_path)
        rows = {r["repo"]: r for r in diagnostics.installed_models(hub, plain)}
        assert rows["mlx-community/Qwen3-4B-8bit"]["registry_id"] == "qwen3-4b"
        assert rows["Ornith-1.0-9B-8bit"]["registry_id"] == "ornith-1.0-9b"

    def test_rows_carry_paths_for_reveal_and_delete(self, tmp_path):
        hub, plain = self._fake_caches(tmp_path)
        for row in diagnostics.installed_models(hub, plain):
            assert os.path.isdir(row["path"])

    def test_models_sort_before_drafters(self, tmp_path):
        hub, plain = self._fake_caches(tmp_path)
        kinds = [r["kind"] for r in diagnostics.installed_models(hub, plain)]
        assert kinds == sorted(kinds, key=lambda k: k != "model")

    def test_empty_caches_are_fine(self, tmp_path):
        assert diagnostics.installed_models(str(tmp_path / "nope"),
                                            str(tmp_path / "also-nope"),
                                            lmstudio_roots=()) == []

    def test_disk_usage_totals_the_rows(self, tmp_path):
        hub, plain = self._fake_caches(tmp_path)
        rows = diagnostics.installed_models(hub, plain, lmstudio_roots=())
        usage = diagnostics.disk_usage(rows, drafters_dir=str(tmp_path / "none"))
        assert usage["total_bytes"] == sum(r["size_bytes"] for r in rows)
        assert usage["total"]


class TestLMStudioScan:
    """Models LM Studio already downloaded are offered (issue #12) — readable, clearly
    labeled as another app's files, and never billed against our disk total."""

    def _roots(self, tmp_path):
        root = tmp_path / "lmstudio"
        mlx = root / "lmstudio-community" / "Qwen3-8B-MLX-8bit"
        _write(mlx / "model.safetensors", 700)
        _write(mlx / "config.json", 10)
        gguf = root / "someone" / "Llama-3-GGUF"     # GGUF-only: our loaders can't read it
        _write(gguf / "model.gguf", 300)
        return (str(root),)

    def test_mlx_dirs_listed_with_lmstudio_source(self, tmp_path):
        rows = {r["repo"]: r for r in diagnostics.installed_models(
            str(tmp_path / "hub"), str(tmp_path / "plain"),
            lmstudio_roots=self._roots(tmp_path))}
        row = rows["lmstudio-community/Qwen3-8B-MLX-8bit"]
        assert row["source"] == "lmstudio"
        assert row["kind"] == "model"

    def test_gguf_only_dirs_are_skipped(self, tmp_path):
        rows = {r["repo"] for r in diagnostics.installed_models(
            str(tmp_path / "hub"), str(tmp_path / "plain"),
            lmstudio_roots=self._roots(tmp_path))}
        assert "someone/Llama-3-GGUF" not in rows

    def test_our_copy_wins_a_duplicate_repo_id(self, tmp_path):
        hub = tmp_path / "hub"
        _write(hub / "models--lmstudio-community--Qwen3-8B-MLX-8bit" / "blobs" / "a", 100)
        rows = {r["repo"]: r for r in diagnostics.installed_models(
            str(hub), str(tmp_path / "plain"), lmstudio_roots=self._roots(tmp_path))}
        assert rows["lmstudio-community/Qwen3-8B-MLX-8bit"]["source"] == "cache"

    def test_disk_usage_excludes_lmstudio_rows(self, tmp_path):
        rows = diagnostics.installed_models(
            str(tmp_path / "hub"), str(tmp_path / "plain"),
            lmstudio_roots=self._roots(tmp_path))
        usage = diagnostics.disk_usage(rows, drafters_dir=str(tmp_path / "none"))
        assert usage["total_bytes"] == 0             # every row here is LM Studio's disk


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


class TestBandwidthAndCeilings:
    def test_bandwidth_info_shape(self):
        bw = diagnostics.bandwidth_info()
        assert bw["reference_gb_s"] == 273.0
        assert bw["source"] in ("measured", "theoretical", "unknown")
        if bw["theoretical_gb_s"]:
            assert abs(bw["scale"] - bw["theoretical_gb_s"] / 273.0) < 1e-3

    def test_environment_reports_chip_and_memory(self):
        env = diagnostics.environment()
        assert "bandwidth_gb_s" in env["chip"] and "family" in env["chip"]
        assert "pressure" in env["memory"]

    def test_local_weight_bytes_follows_hub_snapshot_links(self, tmp_path, monkeypatch):
        hub = tmp_path / "hub" / "models--org--Tiny"
        blobs = hub / "blobs"
        snap = hub / "snapshots" / "abc"
        blobs.mkdir(parents=True)
        snap.mkdir(parents=True)
        (blobs / "b1").write_bytes(b"x" * 1000)
        (blobs / "b2").write_bytes(b"y" * 500)
        os.symlink(blobs / "b1", snap / "model-00001.safetensors")
        os.symlink(blobs / "b2", snap / "model-00002.safetensors")
        (snap / "config.json").write_text("{}")
        monkeypatch.setattr(diagnostics, "_local_dir", lambda repo: str(hub))
        assert diagnostics.local_weight_bytes("org/Tiny") == 1500   # links followed, blobs skipped

    def test_local_weight_bytes_none_when_absent(self, monkeypatch):
        monkeypatch.setattr(diagnostics, "_local_dir", lambda repo: None)
        assert diagnostics.local_weight_bytes("org/Nope") is None

    def test_inventory_rows_carry_ceiling_keys(self):
        for row in diagnostics.model_inventory(ram_gb=64):
            assert "weight_bytes" in row and "ceiling_tps" in row
            if row["ceiling_tps"] is not None:
                assert row["weight_bytes"] and row["ceiling_tps"] > 0
