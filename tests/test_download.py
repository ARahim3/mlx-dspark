"""Model-free tests for the cancellable download path (download.py).

No network: everything here exercises the routing/no-op/cancel bookkeeping. The one
subprocess test uses a stub command, not a real snapshot_download.
"""

from __future__ import annotations

import os

from mlx_dspark.download import (
    DownloadCancelled,
    _looks_like_repo,
    cancel_current,
    ensure_local,
    progress,
)

assert DownloadCancelled is not None  # exported for the server; keep it importable


def test_looks_like_repo_routing(tmp_path):
    assert _looks_like_repo("no-such-org/no-such-model-xyz")
    assert not _looks_like_repo(None)
    assert not _looks_like_repo("")
    assert not _looks_like_repo("gguf:org/name/file.gguf")     # scheme -> loader's business
    assert not _looks_like_repo(str(tmp_path))                 # local dir
    assert not _looks_like_repo("just-a-name")                 # no org
    assert not _looks_like_repo("a/b/c")                       # not an org/name id


def test_looks_like_repo_prefers_plain_dir_cache(tmp_path, monkeypatch):
    """A repo whose basename exists under ~/.cache/mlx_dspark/models is served from disk
    by _resolve, so the pre-fetch must not re-download it (the standing 6-GB lesson)."""
    models = tmp_path / "models"
    (models / "Some-Model-4bit").mkdir(parents=True)
    real_expanduser = os.path.expanduser
    monkeypatch.setattr(
        "mlx_dspark.download.os.path.expanduser",
        lambda p: str(models) if p.endswith("mlx_dspark/models") else real_expanduser(p))
    assert not _looks_like_repo("org/Some-Model-4bit")
    assert _looks_like_repo("org/Other-Model")


def test_looks_like_repo_skips_lmstudio_mlx_dir(tmp_path, monkeypatch):
    """Issue #28: a model present only in LM Studio's folder must not be pre-fetched into the
    HF cache (the loader reads LM Studio's copy). GGUF-only dirs there still route to the hub."""
    import mlx_dspark.load as load

    root = tmp_path / "lmstudio"
    mlx_dir = root / "org" / "Some-Model-MLX-4bit"
    mlx_dir.mkdir(parents=True)
    (mlx_dir / "config.json").write_text("{}")
    (mlx_dir / "model.safetensors").write_bytes(b"weights")
    gguf_dir = root / "org" / "Some-Model-GGUF"
    gguf_dir.mkdir(parents=True)
    (gguf_dir / "model.gguf").write_bytes(b"gguf")
    monkeypatch.setattr(load, "LMSTUDIO_ROOTS", (str(root),))
    assert not _looks_like_repo("org/Some-Model-MLX-4bit")
    assert _looks_like_repo("org/Some-Model-GGUF")
    assert _looks_like_repo("org/Other-Model")
    # and the three answers agree: where to load from == "on disk" == "no pre-fetch"
    assert load._resolve("org/Some-Model-MLX-4bit") == str(mlx_dir)
    assert load.local_dir("org/Some-Model-MLX-4bit") == str(mlx_dir)
    assert load.local_dir("org/Some-Model-GGUF") is None


def test_ensure_local_is_a_noop_for_non_repos(tmp_path):
    ensure_local(None)
    ensure_local(str(tmp_path))
    ensure_local("gguf:org/name/file.gguf")   # returns without touching the network


def test_cancel_with_no_download_is_idempotent():
    out = cancel_current()
    assert out["cancelled"] is False
    assert "no download" in out["reason"]


def test_progress_none_when_idle():
    assert progress() is None
