"""CLI-boundary tests: the reasoning budget is opt-in (default off) on both commands.

The engine-layer tests all set ``default_think_budget`` themselves, so they would keep
passing if an argparse default silently regressed to a non-zero value — these pin the
actual defaults. Both commands import their collaborators inside the function body, so the
fakes are patched onto the SOURCE modules; the call-time ``from .x import y`` then binds
them.
"""

import importlib
from types import SimpleNamespace

import mlx_dspark.generate as generate
import mlx_dspark.load as load
import mlx_dspark.server as server
from mlx_dspark import cli

# `mlx_dspark.calibrate` the attribute is the calibrate() FUNCTION (re-exported by the
# package __init__), which shadows the submodule — go through importlib for the module.
calibrate = importlib.import_module("mlx_dspark.calibrate")


def _serve_load_kwargs(monkeypatch, argv):
    captured = {}

    def fake_run_server(holder, **kw):
        captured["holder"] = holder

    monkeypatch.setattr(server, "run_server", fake_run_server)
    cli.cmd_serve(["--no-model", *argv])
    return captured["holder"]._load_kwargs


def test_serve_reasoning_budget_is_opt_in(monkeypatch):
    assert _serve_load_kwargs(monkeypatch, [])["default_think_budget"] is None
    assert _serve_load_kwargs(
        monkeypatch, ["--reasoning-budget", "8192"])["default_think_budget"] == 8192
    assert _serve_load_kwargs(
        monkeypatch, ["--reasoning-budget", "0"])["default_think_budget"] is None


def _generate_think_budget(monkeypatch, argv):
    captured = {}

    def fake_greedy(target, tok, prompt, **kw):
        captured.update(kw)
        return SimpleNamespace(text="ok", num_tokens=3, seconds=0.1, tokens_per_sec=30.0)

    monkeypatch.setattr(load, "resolve_mode",
                        lambda *a, **k: ("baseline", "fake/target", None))
    monkeypatch.setattr(load, "load_target", lambda *a, **k: (object(), object()))
    monkeypatch.setattr(calibrate, "apply_small_m", lambda *a, **k: None)
    monkeypatch.setattr(calibrate, "apply_wide_gemm", lambda *a, **k: None)
    monkeypatch.setattr(generate, "greedy_generate", fake_greedy)
    cli.cmd_generate(["--mode", "baseline", "--model", "fake/target", "--no-stream", *argv])
    return captured["think_budget"]


def test_generate_reasoning_budget_is_opt_in(monkeypatch):
    assert _generate_think_budget(monkeypatch, []) is None
    assert _generate_think_budget(monkeypatch, ["--reasoning-budget", "8192"]) == 8192
