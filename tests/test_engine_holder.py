"""EngineHolder delegation + swap bookkeeping — model-free (a fake engine stands in for the
real one, since the swap's actual model load needs weights and is exercised on-device)."""

import pytest

from mlx_dspark.server import EngineHolder


class FakeEngine:
    def __init__(self, model_id="m1"):
        self.model_id = model_id
        self.mode = "dspark"
        self.closed = False

    def close(self):
        self.closed = True

    def metrics(self):
        return {"model": self.model_id}


def holder(engine=None):
    # load_kwargs is only read by swap(); delegation/status tests never load, so {} is fine.
    return EngineHolder(engine or FakeEngine(), load_kwargs={})


class TestDelegation:
    def test_attributes_delegate_to_current_engine(self):
        h = holder(FakeEngine("qwen"))
        assert h.model_id == "qwen"          # via __getattr__
        assert h.mode == "dspark"

    def test_methods_delegate(self):
        h = holder(FakeEngine("g"))
        assert h.metrics() == {"model": "g"}

    def test_holder_own_attributes_win_over_delegation(self):
        h = holder()
        assert h.ready is True               # a real property, not delegated
        assert callable(h.swap)


class TestStatus:
    def test_ready_when_engine_present(self):
        h = holder(FakeEngine("x"))
        assert h.ready is True
        s = h.status()
        assert s == {"ready": True, "loading": False, "model": "x", "error": None}

    def test_not_ready_without_engine(self):
        h = holder()
        h._engine = None
        assert h.ready is False
        assert h.status()["model"] is None

    def test_access_without_engine_raises_clearly(self):
        h = holder()
        h._engine = None
        # The dispatcher gates on `ready` first; a stray delegated access should still be a
        # clear message, not an AttributeError on None.
        with pytest.raises(RuntimeError, match="no model is loaded"):
            _ = h.model_id


class TestSwap:
    def test_successful_swap_closes_old_and_installs_new(self, monkeypatch):
        old = FakeEngine("old")
        h = holder(old)
        new = FakeEngine("new")

        import mlx_dspark.server as server
        monkeypatch.setattr(server.Engine, "load", staticmethod(lambda **kw: new))
        monkeypatch.setattr(server, "maybe_batch_engine", lambda e, b: e)

        status = h.swap(model="new-repo")
        assert old.closed is True            # released before the new one
        assert h.current is new
        assert status == {"ready": True, "loading": False, "model": "new", "error": None}

    def test_failed_swap_leaves_no_engine_but_records_the_error(self, monkeypatch):
        old = FakeEngine("old")
        h = holder(old)

        import mlx_dspark.server as server

        def boom(**kw):
            raise ValueError("unknown model")

        monkeypatch.setattr(server.Engine, "load", staticmethod(boom))

        with pytest.raises(ValueError, match="unknown model"):
            h.swap(model="bogus")
        assert old.closed is True            # old was still released
        assert h.ready is False
        assert h.status()["error"] == "unknown model"
        assert h._loading is False           # flag cleared even on failure

    def test_swap_passes_model_and_overrides_through(self, monkeypatch):
        h = EngineHolder(FakeEngine("old"), load_kwargs={"mode": "dspark", "prefix_cache": True})
        captured = {}

        import mlx_dspark.server as server

        def capture(**kw):
            captured.update(kw)
            return FakeEngine("new")

        monkeypatch.setattr(server.Engine, "load", staticmethod(capture))
        monkeypatch.setattr(server, "maybe_batch_engine", lambda e, b: e)

        h.swap(model="repo", mode="lookup", max_draft=4)
        assert captured["model"] == "repo"
        assert captured["mode"] == "lookup"          # override applied
        assert captured["max_draft_tokens"] == 4
        assert captured["prefix_cache"] is True      # base kwarg preserved

    def test_batch_engine_inner_is_also_closed(self, monkeypatch):
        inner = FakeEngine("old-inner")

        class FakeBatch:
            def __init__(self, e):
                self.engine = e
                self.closed = False

            def close(self):
                self.closed = True

        batch = FakeBatch(inner)
        h = EngineHolder(batch, load_kwargs={})

        import mlx_dspark.server as server
        monkeypatch.setattr(server.Engine, "load", staticmethod(lambda **kw: FakeEngine("new")))
        monkeypatch.setattr(server, "maybe_batch_engine", lambda e, b: e)

        h.swap(model="repo")
        assert batch.closed is True          # scheduler stopped
        assert inner.closed is True          # AND the wrapped models freed
