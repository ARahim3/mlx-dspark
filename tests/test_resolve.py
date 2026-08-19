"""Tests for the target->drafter resolver (model-centric CLI/library interface)."""

from __future__ import annotations

import pytest

from mlx_dspark.load import resolve


def test_full_repo_auto_resolves_drafter():
    assert resolve("mlx-community/Qwen3-8B-8bit", mode="dspark") == (
        "mlx-community/Qwen3-8B-8bit", "deepseek-ai/dspark_qwen3_8b_block7")
    assert resolve("mlx-community/Qwen3-8B-8bit", mode="dflash")[1] == "z-lab/Qwen3-8B-DFlash-b16"


def test_quantization_agnostic():
    # the drafter matches the model, not the quant
    for repo in ("mlx-community/Qwen3-8B-4bit", "some-org/Qwen3-8B-bf16", "x/Qwen3-8B-8bit"):
        assert resolve(repo, mode="dspark")[1] == "deepseek-ai/dspark_qwen3_8b_block7"


def test_gemma_naming_variants():
    assert resolve("mlx-community/gemma-4-12B-it-4bit", mode="dspark")[1] == \
        "deepseek-ai/dspark_gemma4_12b_block7"


def test_no_cross_match_between_sizes():
    assert resolve("mlx-community/Qwen3-4B-8bit", mode="dspark")[1] == \
        "deepseek-ai/dspark_qwen3_4b_block7"
    assert resolve("mlx-community/Qwen3-8B-8bit", mode="dspark")[1] == \
        "deepseek-ai/dspark_qwen3_8b_block7"


def test_qwen36_27b_resolves_satgeze_drafter_quant_agnostic():
    # Resolution is quant-agnostic: the 4-bit target gets the same head, it is just not the
    # measured pairing (the registry's `target` field is the quant we benchmarked).
    for repo in ("mlx-community/Qwen3.6-27B-4bit", "mlx-community/Qwen3.6-27B-8bit",
                 "mlx-community/Qwen3.6-27B-OptiQ-4bit", "Qwen/Qwen3.6-27B",
                 "some-org/Qwen3.6-27B-bf16"):
        assert resolve(repo, mode="dspark")[1] == "satgeze/Qwen3.6-27B-DSpark", repo


def test_qwen36_27b_no_cross_match_with_bonsai_or_dense_qwen3():
    # Ternary-Bonsai is the same architecture but its drafter is variant-specific
    assert resolve("prism-ml/Ternary-Bonsai-27B-mlx-2bit", mode="dspark")[1] == \
        "Rahim/Ternary-Bonsai-27B-dspark"
    # and the dense qwen3-4b/8b ids must not swallow the 3.6 naming (or vice versa)
    assert resolve("mlx-community/Qwen3-4B-8bit", mode="dspark")[1] == \
        "deepseek-ai/dspark_qwen3_4b_block7"


def test_qwen36_27b_has_no_dflash_drafter():
    with pytest.raises(ValueError, match="no built-in"):
        resolve("mlx-community/Qwen3.6-27B-4bit", mode="dflash")


def test_ornith_9b_resolves_community_drafter_quant_agnostic():
    for repo in ("mlx-community/Ornith-1.0-9B-4bit", "mlx-community/Ornith-1.0-9B-8bit",
                 "deepreinforce-ai/Ornith-1.0-9B"):
        assert resolve(repo, mode="dspark")[1] == "stanleyphoong/Ornith-1.0-9B-DSpark", repo


def test_qwen36_35b_a3b_resolves_koopah_drafter_quant_agnostic():
    for repo in ("mlx-community/Qwen3.6-35B-A3B-4bit", "mlx-community/Qwen3.6-35B-A3B-8bit",
                 "Qwen/Qwen3.6-35B-A3B", "unsloth/Qwen3.6-35B-A3B-MLX-8bit"):
        assert resolve(repo, mode="dspark")[1] == "Koopah/Qwen3.6-35B-A3B-NVFP4-DSPARK", repo


def test_qwen36_35b_a3b_does_not_collide_with_27b_or_dense_qwen3():
    # the two Qwen3.6 ids must not swallow each other, and neither may match dense qwen3-4b
    assert resolve("mlx-community/Qwen3.6-27B-8bit", mode="dspark")[1] == \
        "satgeze/Qwen3.6-27B-DSpark"
    assert resolve("mlx-community/Qwen3.6-35B-A3B-4bit", mode="dspark")[1] == \
        "Koopah/Qwen3.6-35B-A3B-NVFP4-DSPARK"
    assert resolve("mlx-community/Qwen3-4B-8bit", mode="dspark")[1] == \
        "deepseek-ai/dspark_qwen3_4b_block7"


def test_legacy_family_alias():
    assert resolve("qwen3", mode="dspark")[0] == "mlx-community/Qwen3-4B-8bit"
    assert resolve(None, mode="dspark", family="gemma4")[0] == "mlx-community/gemma-4-12B-it-8bit"


def test_legacy_target_alias():
    assert resolve(None, mode="dflash", target="mlx-community/Qwen3-8B-8bit")[1] == \
        "z-lab/Qwen3-8B-DFlash-b16"


def test_explicit_drafter_override_any_target():
    assert resolve("my/Custom", mode="dspark", drafter="my/drafter") == ("my/Custom", "my/drafter")


def test_baseline_has_no_drafter():
    assert resolve("mlx-community/Qwen3-8B-8bit", mode="baseline") == (
        "mlx-community/Qwen3-8B-8bit", None)


def test_unknown_target_without_drafter_errors():
    with pytest.raises(ValueError) as e:
        resolve("my/Unknown-Model", mode="dspark")
    assert "no built-in" in str(e.value) and "--drafter" in str(e.value)


def test_local_path_basename_matched():
    # a local path is matched by its basename
    assert resolve("/models/Qwen3-8B-8bit", mode="dspark")[1] == "deepseek-ai/dspark_qwen3_8b_block7"


# --------------------------------------------------------------------------- resolve_mode


def test_resolve_mode_auto_known_target_picks_dspark():
    from mlx_dspark.load import resolve_mode

    mode, _tgt, drf = resolve_mode("mlx-community/Qwen3-8B-8bit", mode="auto")
    assert mode == "dspark" and drf == "deepseek-ai/dspark_qwen3_8b_block7"


def test_resolve_mode_auto_unknown_target_falls_back_to_lookup():
    from mlx_dspark.load import resolve_mode

    mode, tgt, drf = resolve_mode("some-org/Weird-Model-3B-4bit", mode="auto")
    assert mode == "lookup" and drf is None and tgt == "some-org/Weird-Model-3B-4bit"


def test_resolve_mode_auto_with_explicit_drafter_is_dspark():
    from mlx_dspark.load import resolve_mode

    mode, _tgt, drf = resolve_mode("some-org/Weird-Model-3B", mode="auto", drafter="org/d")
    assert mode == "dspark" and drf == "org/d"


def test_resolve_mode_passthrough_non_auto():
    from mlx_dspark.load import resolve_mode

    assert resolve_mode("mlx-community/Qwen3-8B-8bit", mode="lookup") == (
        "lookup", "mlx-community/Qwen3-8B-8bit", None)
    mode, _, drf = resolve_mode("mlx-community/Qwen3-8B-8bit", mode="dflash")
    assert mode == "dflash" and drf == "z-lab/Qwen3-8B-DFlash-b16"


def test_lookup_drafts_default_follows_the_registry_row():
    """Pairs whose stamped-best configuration ran with hybrid lookup drafts OFF carry
    lookup_drafts: False in the registry — the shipped default must reproduce the vouched-for
    numbers with no flag. Quant-agnostic like the drafter resolution itself."""
    from mlx_dspark.load import lookup_drafts_default

    # measured net-loss rows: every MoE and the 4-bit 27B hybrids (+ Muse's stamped config)
    assert lookup_drafts_default("mlx-community/Qwen3.8-27B-4bit") is False
    assert lookup_drafts_default("mlx-community/Qwen3.8-27B-8bit") is False
    assert lookup_drafts_default("mlx-community/Qwen3.6-35B-A3B-4bit") is False
    assert lookup_drafts_default(
        "mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit") is False
    assert lookup_drafts_default("mlx-community/Muse-Glimmer-30B-8bit") is False
    # measured net-win rows keep the global default on
    assert lookup_drafts_default("mlx-community/Qwen3-8B-8bit") is True
    assert lookup_drafts_default("mlx-community/gemma-4-12B-it-8bit") is True
    assert lookup_drafts_default("mlx-community/Qwen3.6-27B-8bit") is True
    # unknown target / no target: global default
    assert lookup_drafts_default("some-org/Weird-Model-3B") is True
    assert lookup_drafts_default(None) is True


# --------------------------------------------------------------------- LM Studio reuse


class TestLMStudioResolve:
    """`_resolve` reuses models LM Studio already downloaded (issue #12) — exact
    <publisher>/<model> match, MLX layouts only."""

    def _lmstudio(self, tmp_path, *, mlx: bool):
        d = tmp_path / "lmstudio" / "lmstudio-community" / "Qwen3-8B-MLX-8bit"
        d.mkdir(parents=True)
        if mlx:
            (d / "config.json").write_text("{}")
            (d / "model.safetensors").write_bytes(b"x")
        else:
            (d / "model.gguf").write_bytes(b"x")     # GGUF-only: targets can't load it
        return str(tmp_path / "lmstudio"), str(d)

    def test_mlx_dir_is_reused_without_downloading(self, tmp_path, monkeypatch):
        from mlx_dspark import load

        root, d = self._lmstudio(tmp_path, mlx=True)
        monkeypatch.setattr(load, "LMSTUDIO_ROOTS", (root,))
        monkeypatch.setattr(load, "snapshot_download",
                            lambda *a, **k: pytest.fail("should not download"),
                            raising=False)
        assert load._resolve("lmstudio-community/Qwen3-8B-MLX-8bit") == d

    def test_gguf_only_dir_falls_through_to_the_hub(self, tmp_path, monkeypatch):
        from mlx_dspark import load

        root, _ = self._lmstudio(tmp_path, mlx=False)
        monkeypatch.setattr(load, "LMSTUDIO_ROOTS", (root,))
        monkeypatch.setattr(load, "snapshot_download", lambda repo: "HUB", raising=False)
        assert load._resolve("lmstudio-community/Qwen3-8B-MLX-8bit") == "HUB"


def test_resolve_mode_auto_honors_row_best_mode_dflash():
    # Qwen3.8-27B rows are stamped "mode": "dflash" — DFlash 2 beat both DSpark heads at
    # the identical verify width (2026-08-19), so auto (and the app) serve it by default.
    from mlx_dspark.load import resolve_mode

    for tgt in ("mlx-community/Qwen3.8-27B-4bit", "mlx-community/Qwen3.8-27B-8bit"):
        mode, _t, drf = resolve_mode(tgt, mode="auto")
        assert (mode, drf) == ("dflash", "incoai/Qwen3.8-27B-DFlash2"), tgt


def test_resolve_mode_explicit_dspark_still_gets_the_dspark_heads():
    from mlx_dspark.load import resolve_mode

    mode, _t, drf = resolve_mode("mlx-community/Qwen3.8-27B-4bit", mode="dspark")
    assert (mode, drf) == ("dspark", "DimInfer/Qwen3.8-27B-Dspark-v1")
    mode, _t, drf = resolve_mode("mlx-community/Qwen3.8-27B-8bit", mode="dspark")
    assert (mode, drf) == ("dspark", "RadixArk/Qwen3.8-27B-DSpark")


def test_resolve_mode_rows_without_best_mode_keep_dspark_first():
    # rows without a "mode" stamp (everything but Qwen3.8-27B) are unchanged: dspark first
    from mlx_dspark.load import REGISTRY, resolve_mode

    assert sum(1 for e in REGISTRY if e.get("mode")) == 2   # exactly the two Qwen3.8 rows
    mode, _t, _d = resolve_mode("mlx-community/Qwen3-4B-8bit", mode="auto")
    assert mode == "dspark"


def test_resolve_mode_explicit_dflash_resolves_dflash2_head():
    from mlx_dspark.load import resolve_mode

    mode, _t, drf = resolve_mode("mlx-community/Qwen3.8-27B-8bit", mode="dflash")
    assert (mode, drf) == ("dflash", "incoai/Qwen3.8-27B-DFlash2")
