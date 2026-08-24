"""Issue #26: a checkpoint that asks the loader to import its own Python is refused unless
the process opted in (``--trust-remote-code`` / ``MLX_DSPARK_TRUST_REMOTE_CODE=1``)."""

from __future__ import annotations

import json

import pytest

from mlx_dspark import load as L


def _write(tmp_path, name, obj):
    (tmp_path / name).write_text(json.dumps(obj))


def test_clean_checkpoint_has_no_markers(tmp_path):
    _write(tmp_path, "config.json", {"model_type": "qwen3", "quantization": {"bits": 4}})
    _write(tmp_path, "tokenizer_config.json", {"chat_template": "{{ messages }}"})
    assert L.remote_code_markers(str(tmp_path)) == []
    L.refuse_remote_code(str(tmp_path), "clean")                  # no raise


@pytest.mark.parametrize("name, obj, marker", [
    ("config.json", {"model_type": "qwen3", "model_file": "modeling.py"}, "config.json:model_file"),
    ("config.json", {"model_type": "x", "auto_map": {"AutoModel": "m.M"}}, "config.json:auto_map"),
    ("config.json", {"text_config": {"auto_map": {"AutoModel": "m.M"}}},
     "config.json:text_config.auto_map"),
    ("tokenizer_config.json", {"auto_map": {"AutoTokenizer": ["t.T", None]}},
     "tokenizer_config.json:auto_map"),
    ("processor_config.json", {"auto_map": {"AutoProcessor": "p.P"}},
     "processor_config.json:auto_map"),
])
def test_markers_are_found_and_refused(tmp_path, monkeypatch, name, obj, marker):
    _write(tmp_path, name, obj)
    assert L.remote_code_markers(str(tmp_path)) == [marker]
    monkeypatch.setattr(L, "TRUST_REMOTE_CODE", False)
    with pytest.raises(ValueError, match="import its own Python"):
        L.refuse_remote_code(str(tmp_path), "evil/repo")
    monkeypatch.setattr(L, "TRUST_REMOTE_CODE", True)
    L.refuse_remote_code(str(tmp_path), "evil/repo")              # opted in


def test_unreadable_config_is_not_a_marker(tmp_path):
    (tmp_path / "config.json").write_text("{not json")
    assert L.remote_code_markers(str(tmp_path)) == []
