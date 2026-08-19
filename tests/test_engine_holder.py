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

    def test_swap_lookup_drafts_override_and_per_pair_default(self, monkeypatch):
        """An explicit lookup_drafts rides the swap; absent, the server's stored kwargs
        pass through unchanged — a serve started without the flag stores None, so each
        swapped-in pair re-resolves its own registry default inside Engine.load."""
        h = EngineHolder(FakeEngine("old"), load_kwargs={"lookup_drafts": None})
        captured = {}

        import mlx_dspark.server as server

        def capture(**kw):
            captured.update(kw)
            return FakeEngine("new")

        monkeypatch.setattr(server.Engine, "load", staticmethod(capture))
        monkeypatch.setattr(server, "maybe_batch_engine", lambda e, b: e)

        h.swap(model="repo", lookup_drafts=False)
        assert captured["lookup_drafts"] is False    # request override wins
        captured.clear()
        h.swap(model="repo2")
        assert captured["lookup_drafts"] is None     # unset -> Engine.load resolves per pair

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


class TestUnload:
    def test_unload_releases_and_reports_no_model(self):
        eng = FakeEngine("m")
        h = holder(eng)
        s = h.unload()
        assert eng.closed is True
        assert h.ready is False
        assert s == {"ready": False, "loading": False, "model": None, "error": None}

    def test_unload_twice_is_a_noop(self):
        h = holder()
        h.unload()
        assert h.unload()["ready"] is False   # no raise on the empty holder

    def test_unload_closes_a_batch_engine_inner(self):
        inner = FakeEngine("inner")

        class FakeBatch:
            def __init__(self, e):
                self.engine = e
                self.closed = False

            def close(self):
                self.closed = True

        batch = FakeBatch(inner)
        h = EngineHolder(batch, load_kwargs={})
        h.unload()
        assert batch.closed is True and inner.closed is True

    def test_load_after_unload_works(self, monkeypatch):
        h = holder()
        h.unload()

        import mlx_dspark.server as server
        monkeypatch.setattr(server.Engine, "load", staticmethod(lambda **kw: FakeEngine("new")))
        monkeypatch.setattr(server, "maybe_batch_engine", lambda e, b: e)

        status = h.swap(model="repo")
        assert status["ready"] is True and status["model"] == "new"


class TestSwapConfidence:
    def test_swap_confidence_override_and_default_passthrough(self, monkeypatch):
        """An explicit confidence_threshold rides the swap (the cap+confidence bundle a
        client applies for pairs whose measured best needs it — Qwen3.8-27B-4bit's is
        cap 7 + 0.3); absent, the server's stored kwargs pass through untouched."""
        h = EngineHolder(FakeEngine("old"), load_kwargs={"confidence_threshold": 0.0})
        captured = {}

        import mlx_dspark.server as server

        def capture(**kw):
            captured.update(kw)
            return FakeEngine("new")

        monkeypatch.setattr(server.Engine, "load", staticmethod(capture))
        monkeypatch.setattr(server, "maybe_batch_engine", lambda e, b: e)

        h.swap(model="repo", confidence_threshold=0.3)
        assert captured["confidence_threshold"] == 0.3
        captured.clear()
        h.swap(model="repo2")
        assert captured["confidence_threshold"] == 0.0   # unset -> server default kept

    def test_swap_small_m_override_and_default_passthrough(self, monkeypatch):
        """An explicit small_m rides the swap (the serve-side kernel A/B issue #14 asked
        for); absent, the server's stored kwargs pass through untouched — a serve started
        without the flag stores None, so Engine.load applies its probe-gated default."""
        h = EngineHolder(FakeEngine("old"), load_kwargs={"small_m": None})
        captured = {}

        import mlx_dspark.server as server

        def capture(**kw):
            captured.update(kw)
            return FakeEngine("new")

        monkeypatch.setattr(server.Engine, "load", staticmethod(capture))
        monkeypatch.setattr(server, "maybe_batch_engine", lambda e, b: e)

        h.swap(model="repo", small_m=False)
        assert captured["small_m"] is False              # request override wins
        captured.clear()
        h.swap(model="repo2")
        assert captured["small_m"] is None               # unset -> probe-gated default

    def test_status_reports_download_progress_while_loading(self, monkeypatch):
        """While a swap is fetching weights, status() carries the download progress so
        /health can show a real bar and the client can offer Cancel."""
        import mlx_dspark.download as download

        h = EngineHolder(FakeEngine(), load_kwargs={})
        h._loading = True
        monkeypatch.setattr(download, "progress", lambda: {
            "repo": "org/model", "bytes_done": 5, "bytes_total": 10})
        s = h.status()
        assert s["download"]["repo"] == "org/model"
        h._loading = False
        assert "download" not in h.status()

    def test_swap_context_window_override_is_sticky(self, monkeypatch):
        """context_window rides the swap as a load override (the KV-RAM lever) and is
        STICKY: a later swap that omits it keeps the last explicit value instead of
        reverting to the model's 262k max (community report — scripts set it once).
        Explicit 0 resets to the model's own maximum."""
        h = EngineHolder(FakeEngine("old"), load_kwargs={"context_window": None})
        captured = {}

        import mlx_dspark.server as server

        def capture(**kw):
            captured.update(kw)
            return FakeEngine("new")

        monkeypatch.setattr(server.Engine, "load", staticmethod(capture))
        monkeypatch.setattr(server, "maybe_batch_engine", lambda e, b: e)

        h.swap(model="repo", context_window=32768)
        assert captured["context_window"] == 32768
        captured.clear()
        h.swap(model="repo2")                            # omitted -> keeps 32768
        assert captured["context_window"] == 32768
        captured.clear()
        h.swap(model="repo3", context_window=0)          # 0 -> back to the model's max
        assert captured["context_window"] is None
        captured.clear()
        h.swap(model="repo4")                            # and the reset sticks too
        assert captured["context_window"] is None


class TestKvBitsOverride:
    def test_swap_kv_bits_override_and_zero_means_full_precision(self, monkeypatch):
        """issue #17: `kv_bits` rides /admin/load. Explicit 4/8 quantize the swapped-in
        target's KV cache; explicit 0 means full precision (Engine.load's None); omitted
        keeps whatever the server was started with."""

        import mlx_dspark.server as server

        h = server.EngineHolder(FakeEngine("old"), load_kwargs={"kv_bits": 8})
        captured = {}

        def capture(**kw):
            captured.update(kw)
            return FakeEngine("new")

        monkeypatch.setattr(server.Engine, "load", staticmethod(capture))
        monkeypatch.setattr(server, "maybe_batch_engine", lambda e, b: e)

        h.swap(model="repo", kv_bits=4)
        assert captured["kv_bits"] == 4              # override applied
        captured.clear()
        h.swap(model="repo", kv_bits=0)
        assert captured["kv_bits"] is None           # explicit full precision
        captured.clear()
        h.swap(model="repo")
        assert captured["kv_bits"] == 8              # omitted -> server's startup setting
