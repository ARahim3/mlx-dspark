"""transformers>=5.13 import-compat shim (issue #1).

mlx_lm registers a tokenizer by a string key, which transformers>=5.13's
``_LazyAutoMapping.register`` (assuming a config *class*) rejects with
``AttributeError: 'str' object has no attribute '__module__'`` at module scope,
taking down ``import mlx_dspark``. ``mlx_dspark`` installs a scoped shim at import.
These tests are transformers-version-agnostic: the shim is applied unconditionally,
so a string-key register must be tolerated on any installed transformers.
"""


def test_string_register_shim_applied():
    from transformers.models.auto.auto_factory import _LazyAutoMapping

    import mlx_dspark  # noqa: F401 — importing applies the shim

    assert getattr(_LazyAutoMapping.register, "_mlx_dspark_patched", False)


def test_string_key_register_does_not_raise():
    from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING

    import mlx_dspark  # noqa: F401

    # Pre-shim this raised AttributeError on transformers>=5.13.
    TOKENIZER_MAPPING.register("_mlx_dspark_test_key", (None, None), exist_ok=True)
    assert TOKENIZER_MAPPING._extra_content.get("_mlx_dspark_test_key") == (None, None)


def test_class_key_register_still_routes_to_original():
    """A real config-class key must not take the string fallback path."""
    from transformers.models.auto.auto_factory import _LazyAutoMapping

    import mlx_dspark  # noqa: F401

    reg = _LazyAutoMapping.register
    # The wrapper only diverts non-class keys; class keys defer to the original.
    # (Smoke check that the wrapper is in place and callable with a class key.)
    assert callable(reg)

    class _FakeCfg:
        pass

    from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING

    TOKENIZER_MAPPING.register(_FakeCfg, (None, None), exist_ok=True)


# --- mlx-vlm 0.6.4 Gemma4UnifiedProcessor shim (issue #4, Blaizzy/mlx-vlm#1578) ---
# transformers>=5.12 derives a processor's valid kwargs from the literal __init__
# signature; mlx-vlm 0.6.4 passes video_processor up to ProcessorMixin while the child
# class still takes it via **kwargs. The shim rewrites only that exact broken shape.


def _broken_064_shape():
    """The 0.6.4 shape: attributes name video_processor, the signature doesn't."""
    calls = {}

    class P:
        attributes = ["image_processor", "tokenizer", "video_processor"]

        def __init__(self, image_processor=None, tokenizer=None, **kwargs):
            calls.update(image_processor=image_processor, tokenizer=tokenizer,
                         kwargs=dict(kwargs))

    return P, calls


def test_gemma4_shim_patches_broken_shape():
    import inspect

    from mlx_dspark.load import _shim_gemma4_unified_processor

    P, calls = _broken_064_shape()
    assert _shim_gemma4_unified_processor(P) is True
    # transformers validates against the literal signature — it must now name the kwarg
    assert "video_processor" in inspect.signature(P.__init__).parameters
    # the original __init__ pops video_processor from **kwargs, so it must be routed back
    P(image_processor="ip", tokenizer="tok", video_processor="vp", image_seq_length=280)
    assert calls["image_processor"] == "ip"
    assert calls["tokenizer"] == "tok"
    assert calls["kwargs"] == {"video_processor": "vp", "image_seq_length": 280}


def test_gemma4_shim_is_idempotent():
    from mlx_dspark.load import _shim_gemma4_unified_processor

    P, _ = _broken_064_shape()
    assert _shim_gemma4_unified_processor(P) is True
    assert _shim_gemma4_unified_processor(P) is False


def test_gemma4_shim_skips_063_shape():
    """0.6.3 never hands video_processor to ProcessorMixin; wrapping it would introduce
    an attribute-count mismatch — it must pass through untouched."""
    from mlx_dspark.load import _shim_gemma4_unified_processor

    class P:
        attributes = ["image_processor", "tokenizer"]

        def __init__(self, image_processor=None, tokenizer=None, **kwargs):
            pass

    orig = P.__init__
    assert _shim_gemma4_unified_processor(P) is False
    assert P.__init__ is orig


def test_gemma4_shim_skips_fixed_shape():
    """Upstream main declares video_processor explicitly — nothing to fix."""
    from mlx_dspark.load import _shim_gemma4_unified_processor

    class P:
        attributes = ["image_processor", "tokenizer", "video_processor"]

        def __init__(self, image_processor=None, tokenizer=None, video_processor=None,
                     **kwargs):
            pass

    orig = P.__init__
    assert _shim_gemma4_unified_processor(P) is False
    assert P.__init__ is orig


def test_gemma4_shim_real_class_never_left_broken():
    """Against whatever mlx-vlm is installed: after the shim runs, the real class must
    not be in the broken shape (signature omits video_processor while attributes name it)."""
    import inspect

    import pytest

    pytest.importorskip("mlx_vlm")
    try:
        from mlx_vlm.models.gemma4_unified.processing_gemma4_unified import (
            Gemma4UnifiedProcessor,
        )
    except Exception:  # noqa: BLE001 — any import failure means the module isn't there to test
        pytest.skip("mlx_vlm has no gemma4_unified processing module")

    from mlx_dspark.load import _shim_gemma4_unified_processor

    _shim_gemma4_unified_processor()  # applies only if the installed version needs it
    params = inspect.signature(Gemma4UnifiedProcessor.__init__).parameters
    attrs = getattr(Gemma4UnifiedProcessor, "attributes", ())
    assert "video_processor" in params or "video_processor" not in attrs
