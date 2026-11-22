"""The race's ``stock`` arm: plain decoding on the library's own forward and MLX's own kernels —
what the stock runtime gives a model — beside mlx-dspark's arms. Model-free."""
import mlx.core as mx
import pytest
from test_tree import PROMPT, _tiny_dense, _Tok

from mlx_dspark import generate as g
from mlx_dspark import server as S

metal = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal kernels")


def _engine(target, tok, **kw):
    return S.Engine(target, tok, None, mode="baseline", model_id="tiny", target_repo="tiny",
                    drafter_repo=None, max_draft_tokens=None, prefix_cache=False, **kw)


def _race(engine, arms, n=8):
    events = []
    engine.race(PROMPT, arms, n, lambda kind, data: events.append((kind, data)))
    return events


def test_an_mlx_lm_target_offers_the_stock_arm_and_runs_it_without_our_prefill_kernels(monkeypatch):
    from mlx_dspark.target import Target

    engine = _engine(Target(_tiny_dense(), tokenizer=_Tok()), _Tok())
    assert engine.stock_runtime().startswith("mlx-lm ")
    assert engine.race_arms_available() == ["baseline", "lookup", "stock"]
    seen = []
    real = S.greedy_generate

    def spy(*a, **k):
        seen.append((g.WIDE_GEMM_MIN_ROWS, g.WIDE_GEMM_SHAPES, g.CPU_SPLIT))
        return real(*a, **k)

    monkeypatch.setattr(S, "greedy_generate", spy)
    monkeypatch.setattr(g, "WIDE_GEMM_MIN_ROWS", 4096)
    monkeypatch.setattr(g, "CPU_SPLIT", {"min_rows": 4096, "fracs": {}})
    events = _race(engine, [{"mode": "stock", "cap": 5}, {"mode": "baseline"}])
    assert seen[0] == (None, None, None) and seen[1][0] == 4096      # off inside, back after
    assert g.WIDE_GEMM_MIN_ROWS == 4096 and g.CPU_SPLIT["min_rows"] == 4096
    starts = [d for kind, d in events if kind == "arm_start"]
    assert starts[0]["label"] == f"stock {engine.stock_runtime()}"
    verdict = events[-1][1]
    assert events[-1][0] == "verdict" and verdict["identical"]


def test_no_stock_arm_where_the_stock_library_would_not_run_this(monkeypatch):
    from types import SimpleNamespace

    engine = _engine(SimpleNamespace(is_vlm=True, _inkling=False, target_bits=0,
                                     make_cache=list), _Tok())
    assert engine.stock_runtime() is None and "stock" not in engine.race_arms_available()
    engine.target = SimpleNamespace(is_vlm=True, _inkling=True, target_bits=8)
    assert engine.stock_runtime() is None                       # re-quantized: not the stock weights
    engine.target = SimpleNamespace(is_vlm=True, _inkling=True, target_bits=0)
    assert engine.stock_runtime().startswith("mlx-vlm")
    engine.target = None
    assert engine.stock_runtime() is None


def test_a_git_install_is_named_by_its_commit_not_the_last_release(monkeypatch):
    """mlx-vlm installed from git still says 0.8.0 — a release that cannot load Inkling-2-Mini."""
    from importlib import metadata
    from types import SimpleNamespace

    files = {"direct_url.json": '{"url": "https://github.com/Blaizzy/mlx-vlm", "vcs_info": '
                                '{"vcs": "git", "commit_id": "d998a6207b7dc613b52618646dbea13a4799f7ca"}}'}

    def dist(name):
        return SimpleNamespace(version="0.8.0", read_text=lambda f: files.get(f))

    monkeypatch.setattr(metadata, "distribution", dist)
    assert S._runtime_label("mlx-vlm") == "mlx-vlm git d998a62"
    files.clear()                                             # a release from an index
    assert S._runtime_label("mlx-vlm") == "mlx-vlm 0.8.0"
    files["direct_url.json"] = '{"url": "file:///src/mlx-vlm", "dir_info": {"editable": true}}'
    assert S._runtime_label("mlx-vlm") == "mlx-vlm 0.8.0"

    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "distribution", missing)
    assert S._runtime_label("mlx-vlm") == "mlx-vlm"


@metal
def test_inkling_stock_switches_everything_off_and_puts_it_back():
    from test_inkling_batch import PROMPTS, _target

    from mlx_dspark import inkling_fused as F
    from mlx_dspark import inkling_kernels as K

    t, lm = _target()
    attn = lm.model.layers[0].self_attn

    def state():
        return (F.enabled_kinds(lm), F.kv_caches(lm), F.packed_attn(lm),
                attn.__dict__.get(F._PACKED), lm.__dict__.get(K._FAST_HEAD, "unset"),
                type(t.make_cache()[0][0]).__name__)

    before = state()
    assert before[0] and before[1]

    def run():
        cache = t.make_cache()
        t.prefill(mx.array([PROMPTS[1]]), cache, None)
        return [t.plain(mx.array([[tok]]), cache)[0, -1] for tok in (10, 20, 31)]

    with t.stock_forward():
        assert F.enabled_kinds(lm) == () and not F.kv_caches(lm) and not F.packed_attn(lm)
        assert type(t.make_cache()[0][0]).__name__ in ("RotatingKVCache", "KVCache")
        inside = run()
        mx.eval(inside)
    assert state() == before
    # the same numbers as the model with our pieces switched off by hand
    F.set_enabled(lm, False)
    lm.__dict__[F._KV] = False
    K.set_fast_head(lm, False)
    by_hand = run()
    for a, b in zip(inside, by_hand):
        assert mx.array_equal(a, b)


@metal
def test_inkling_engine_races_stock_against_its_own_plain_decode():
    from concurrent.futures import ThreadPoolExecutor

    from test_inkling_batch import _target

    ex = ThreadPoolExecutor(max_workers=1)
    t = ex.submit(lambda: _target()[0]).result()
    engine = _engine(t, t.tokenizer, executor=ex)
    assert engine.stock_runtime().startswith("mlx-vlm")
    events = _race(engine, [{"mode": "stock"}, {"mode": "baseline"}], n=10)
    done = [d for kind, d in events if kind == "arm_done"]
    assert [d["label"] for d in done] == [f"stock {engine.stock_runtime()}", "baseline"]
    assert done[0]["tokens"] == done[1]["tokens"] == 10
    assert events[-1][0] == "verdict"
