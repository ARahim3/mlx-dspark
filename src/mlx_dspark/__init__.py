"""mlx-dspark: DSpark speculative decoding for Apple Silicon (MLX).

DSpark (from DeepSeek's DeepSpec codebase) is a semi-autoregressive,
EAGLE-family speculative-decoding drafter:

  - a *parallel backbone* proposes base logits for all K draft positions at once,
  - a tiny *sequential head* (low-rank, previous-token-conditioned) corrects
    suffix decay,
  - a *confidence head* scores how likely each drafted token survives
    verification (used here for adaptive draft length instead of the
    server-side load-aware scheduler, which is irrelevant single-user).

This package targets single-user local inference on Apple Silicon.
"""

# Single source of truth for the package version — pyproject.toml reads it from here
# (hatchling dynamic version), so the two can never drift again (0.4.0 shipped with a
# stale module attribute because they were bumped separately).
import contextlib

__version__ = "0.12.0"


# --- transformers>=5.13 compat shim (must run before mlx_lm/mlx_vlm import below) ---
# mlx_lm registers a tokenizer by a *string* key (its tokenizer_utils.py does
# ``AutoTokenizer.register("NewlineTokenizer", ...)``). transformers>=5.13 made
# ``_LazyAutoMapping.register`` assume the key is a config *class* (it reads
# ``key.__module__``), so importing mlx_lm — and therefore mlx_dspark — crashes with
# ``AttributeError: 'str' object has no attribute '__module__'``, which a fresh install
# hits because it resolves transformers to 5.13+. The defect is upstream in
# ml-explore/mlx-lm; drop this shim once it's fixed there. Reported by @zboyles (#1);
# branches on ``isinstance(key, type)`` so real config-class keys reach the original
# ``register`` untouched and only string keys take the pre-5.13 fallback.
def _patch_transformers_string_register() -> None:
    try:
        from transformers.models.auto.auto_factory import _LazyAutoMapping
    except Exception:  # noqa: BLE001 — any transformers layout we don't recognise: no shim
        return
    _orig = _LazyAutoMapping.register
    if getattr(_orig, "_mlx_dspark_patched", False):
        return

    def register(self, key, value, exist_ok=False):
        if not isinstance(key, type):
            # pre-5.13 behavior: non-class keys are stored directly, not introspected
            with contextlib.suppress(AttributeError):
                self._extra_content[key] = value
            return None
        return _orig(self, key, value, exist_ok=exist_ok)

    register._mlx_dspark_patched = True
    _LazyAutoMapping.register = register


_patch_transformers_string_register()

from .calibrate import CapController, calibrate
from .config import DSparkConfig
from .dflash_model import DFlashConfig, DFlashDraftModel
from .generate import (
    GenResult,
    StopStreaming,
    dflash_generate,
    encode_messages,
    encode_prompt,
    greedy_generate,
    speculative_generate,
)
from .load import (
    DEFAULT_DRAFTER,
    DEFAULT_TARGET,
    DFLASH_PRESETS,
    PRESETS,
    REGISTRY,
    apply_wired_limit,
    load_dflash,
    load_dflash_pair,
    load_drafter,
    load_pair,
    load_target,
    resolve,
    resolve_mode,
)
from .lookup import NGramIndex, lookup_generate
from .server import Engine, run_server
from .target import Target

__all__ = [
    "DEFAULT_DRAFTER",
    "DEFAULT_TARGET",
    "DFLASH_PRESETS",
    "PRESETS",
    "REGISTRY",
    "CapController",
    "DFlashConfig",
    "DFlashDraftModel",
    "DSparkConfig",
    "Engine",
    "GenResult",
    "NGramIndex",
    "StopStreaming",
    "Target",
    "apply_wired_limit",
    "calibrate",
    "dflash_generate",
    "encode_messages",
    "encode_prompt",
    "greedy_generate",
    "load_dflash",
    "load_dflash_pair",
    "load_drafter",
    "load_pair",
    "load_target",
    "lookup_generate",
    "resolve",
    "resolve_mode",
    "run_server",
    "speculative_generate",
]
