"""Model-free tests for the PrismML Bonsai integration: the dspark-GGUF converter, the
GIDD log-SNR conditioning, hybrid (qwen3_5-style) target verify/rollback, the cap-0
("don't speculate") controller extension, and target routing / registry resolution."""

from __future__ import annotations

import json
import math
import struct

import mlx.core as mx
import pytest

from mlx_dspark.calibrate import CapController
from mlx_dspark.config import DSparkConfig
from mlx_dspark.gguf_convert import convert_dspark_gguf, read_gguf_header
from mlx_dspark.load import _route_target, resolve, resolve_mode
from mlx_dspark.model import DSparkDrafter, log_snr_features
from mlx_dspark.target import Target

# ---------------------------------------------------------------- gguf writer (test-only)

_T_U32, _T_I32, _T_F32, _T_BOOL, _T_STR, _T_ARR = 4, 5, 6, 7, 8, 9
_GG_F32, _GG_BF16, _GG_Q4K = 0, 30, 12


def _pack_str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _pack_kv(key: str, vtype: int, val) -> bytes:
    out = _pack_str(key) + struct.pack("<I", vtype)
    if vtype == _T_STR:
        out += _pack_str(val)
    elif vtype == _T_ARR:
        etype, items = val
        out += struct.pack("<IQ", etype, len(items))
        for it in items:
            out += struct.pack("<i" if etype == _T_I32 else "<I", it)
    elif vtype == _T_U32:
        out += struct.pack("<I", val)
    elif vtype == _T_F32:
        out += struct.pack("<f", val)
    elif vtype == _T_BOOL:
        out += struct.pack("<B", int(val))
    else:
        raise AssertionError(vtype)
    return out


def _bf16_bytes(arr: mx.array) -> bytes:
    import numpy as np

    return np.array(arr.astype(mx.bfloat16).view(mx.uint16)).tobytes()


def _write_gguf(path, meta_kvs: list[tuple], tensors: list[tuple]) -> None:
    """tensors: (name, shape_hf, ggml_type). Data is deterministic per-tensor."""
    align = 32
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta_kvs))
    for kv in meta_kvs:
        header += _pack_kv(*kv)
    infos, blobs, off = b"", [], 0
    for i, (name, shape_hf, ttype) in enumerate(tensors):
        ne = list(reversed(shape_hf))                     # gguf order: fastest first
        n = math.prod(shape_hf)
        data = mx.arange(n).reshape(shape_hf) * 0.01 + i  # deterministic, distinct
        raw = (_bf16_bytes(data) if ttype == _GG_BF16
               else __import__("numpy").array(data.astype(mx.float32)).tobytes())
        infos += _pack_str(name) + struct.pack("<I", len(ne))
        for d in ne:
            infos += struct.pack("<Q", d)
        infos += struct.pack("<IQ", ttype, off)
        blobs.append((off, raw))
        off += (len(raw) + align - 1) // align * align
    head = header + infos
    data_start = (len(head) + align - 1) // align * align
    with open(path, "wb") as f:
        f.write(head)
        f.write(b"\x00" * (data_start - len(head)))
        for rel, raw in blobs:
            f.seek(data_start + rel)
            f.write(raw)


def _dspark_meta(arch: str = "dspark") -> list[tuple]:
    return [
        ("general.architecture", _T_STR, arch),
        ("general.name", _T_STR, "tiny-dspark"),
        ("dspark.block_count", _T_U32, 1),
        ("dspark.embedding_length", _T_U32, 8),
        ("dspark.feed_forward_length", _T_U32, 16),
        ("dspark.attention.head_count", _T_U32, 2),
        ("dspark.attention.head_count_kv", _T_U32, 1),
        ("dspark.attention.key_length", _T_U32, 4),
        ("dspark.attention.layer_norm_rms_epsilon", _T_F32, 1e-6),
        ("dspark.rope.freq_base", _T_F32, 1e7),
        ("dspark.dspark.block_size", _T_U32, 4),
        ("dspark.dspark.mask_token_id", _T_U32, 15),
        ("dspark.dspark.target_layers", _T_ARR, (_T_I32, [1, 3, 5, 7, 9])),
        ("dspark.dspark.markov_rank", _T_U32, 4),
        ("dspark.dspark.confidence_head", _T_BOOL, True),
        ("dspark.dspark.confidence_head_with_markov", _T_BOOL, True),
        ("dspark.dspark.log_snr_conditioning", _T_BOOL, True),
        ("dspark.dspark.min_log_snr", _T_F32, -9.0),
        ("dspark.dspark.max_log_snr", _T_F32, 9.0),
    ]


def _dspark_tensors(embd_type: int = _GG_BF16) -> list[tuple]:
    h, v, ff, rank = 8, 16, 16, 4
    return [
        ("token_embd.weight", (v, h), embd_type),
        ("output.weight", (v, h), _GG_BF16),
        ("output_norm.weight", (h,), _GG_F32),
        ("dspark.fc.weight", (h, 5 * h), _GG_BF16),
        ("dspark.hidden_norm.weight", (h,), _GG_F32),
        ("dspark.markov_head_a.weight", (v, rank), _GG_BF16),
        ("dspark.markov_head_b.weight", (v, rank), _GG_BF16),
        ("dspark.confidence_head.weight", (1, h + rank), _GG_BF16),
        ("dspark.confidence_head.bias", (1,), _GG_F32),
        ("dspark.log_snr_fc1.weight", (h, 128), _GG_BF16),
        ("dspark.log_snr_fc1.bias", (h,), _GG_F32),
        ("dspark.log_snr_fc2.weight", (h, h), _GG_BF16),
        ("dspark.log_snr_fc2.bias", (h,), _GG_F32),
        ("blk.0.attn_norm.weight", (h,), _GG_F32),
        ("blk.0.ffn_norm.weight", (h,), _GG_F32),
        ("blk.0.attn_q.weight", (8, h), _GG_BF16),
        ("blk.0.attn_k.weight", (4, h), _GG_BF16),
        ("blk.0.attn_v.weight", (4, h), _GG_BF16),
        ("blk.0.attn_output.weight", (h, 8), _GG_BF16),
        ("blk.0.attn_q_norm.weight", (4,), _GG_F32),
        ("blk.0.attn_k_norm.weight", (4,), _GG_F32),
        ("blk.0.ffn_gate.weight", (ff, h), _GG_BF16),
        ("blk.0.ffn_up.weight", (ff, h), _GG_BF16),
        ("blk.0.ffn_down.weight", (h, ff), _GG_BF16),
    ]


def test_gguf_convert_roundtrip(tmp_path):
    gguf = tmp_path / "tiny-dspark-bf16.gguf"
    _write_gguf(gguf, _dspark_meta(), _dspark_tensors())

    meta, _tensors, _ = read_gguf_header(gguf)
    assert meta["general.architecture"] == "dspark"
    assert meta["dspark.dspark.target_layers"] == [1, 3, 5, 7, 9]

    out = convert_dspark_gguf(gguf, tmp_path / "out")
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["model_type"] == "qwen3"
    assert cfg["block_size"] == 4
    assert cfg["mask_token_id"] == 15
    assert cfg["target_layer_ids"] == [1, 3, 5, 7, 9]
    assert cfg["vocab_size"] == 16                        # from token_embd's own shape
    assert cfg["log_snr_conditioning"] is True and cfg["max_log_snr"] == 9.0

    weights = mx.load(str(out / "model.safetensors"))
    assert weights["fc.weight"].shape == (8, 40)          # [out, 5*hidden]
    assert weights["lm_head.weight"].shape == (16, 8)
    assert weights["markov_head.markov_w1.weight"].shape == (16, 4)
    assert weights["confidence_head.proj.weight"].shape == (1, 12)
    assert weights["log_snr_embed.fc1.weight"].shape == (8, 128)
    assert weights["layers.0.self_attn.q_proj.weight"].shape == (8, 8)
    assert weights["layers.0.mlp.down_proj.weight"].shape == (8, 16)
    assert all(w.dtype == mx.bfloat16 for w in weights.values())

    # the converted checkpoint must load 1:1 into the drafter (strict key match)
    from mlx_dspark.load import load_drafter
    _drafter, dcfg = load_drafter(str(out), quantize=False)
    assert dcfg.log_snr_conditioning and dcfg.num_hidden_layers == 1

    # idempotent: converting again returns without rewriting
    assert convert_dspark_gguf(gguf, tmp_path / "out") == out


def test_gguf_convert_refuses_non_dspark(tmp_path):
    gguf = tmp_path / "not-dspark.gguf"
    meta = [(k, t, ("qwen3" if k == "general.architecture" else v))
            for k, t, v in _dspark_meta()]
    _write_gguf(gguf, meta, _dspark_tensors())
    with pytest.raises(ValueError, match="not a dspark drafter"):
        convert_dspark_gguf(gguf, tmp_path / "out")


def test_gguf_convert_refuses_quantized_types(tmp_path):
    gguf = tmp_path / "tiny-dspark-q4.gguf"
    _write_gguf(gguf, _dspark_meta(), _dspark_tensors(embd_type=_GG_Q4K))
    with pytest.raises(ValueError, match="bf16"):
        convert_dspark_gguf(gguf, tmp_path / "out")


# ---------------------------------------------------------------- log-SNR conditioning


def test_log_snr_features_match_reference():
    """Bit-parity with PrismML dspark.cpp's host-side featurization loop."""
    bs, lo, hi = 4, -9.0, 9.0
    got = log_snr_features(bs, lo, hi)
    assert got.shape == (bs, 128)
    half = 64
    for pos in range(bs):
        snr = hi if pos == 0 else lo
        t = (snr - lo) / (hi - lo) * 1000.0
        for i in (0, 1, 31, 63):
            freq = math.exp(-math.log(10000.0) * i / half)
            # 2e-4: fp32 sin/cos at angles up to t=1000 rad (the reference uses sinf/cosf
            # too); still far below anything a formula/ordering mistake would produce
            assert abs(float(got[pos, i]) - math.sin(t * freq)) < 2e-4
            assert abs(float(got[pos, half + i]) - math.cos(t * freq)) < 2e-4


def _tiny_drafter_config(**kw) -> DSparkConfig:
    base = {"family": "qwen3", "hidden_size": 16, "vocab_size": 32, "num_hidden_layers": 1,
                "intermediate_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 8, "attention_k_eq_v": False, "rope_theta": 1e4, "rope_type": "default",
                "block_size": 4, "mask_token_id": 31, "target_layer_ids": [0, 1],
                "markov_rank": 4, "enable_confidence_head": False,
                "final_logit_softcapping": None, "mlp_activation": "silu",
                "norm_style": "qwen", "use_v_norm": False}
    base.update(kw)
    return DSparkConfig(**base)


def test_drafter_log_snr_addend_applied_and_cached():
    drafter = DSparkDrafter(_tiny_drafter_config(log_snr_conditioning=True))
    mx.eval(drafter.parameters())
    ids = mx.array([[31, 31, 31, 31]])
    raw = drafter.embed_tokens(ids)             # embed_scale is 1.0 for qwen-family
    e_cond = drafter.embed(ids)
    assert e_cond.shape == (1, 4, 16)
    assert not bool(mx.allclose(e_cond, raw).item())
    addend = drafter._snr_addend
    assert bool(mx.allclose(e_cond, raw + addend, atol=1e-5).item())   # purely additive
    assert addend.shape == (1, 4, 16)
    # anchor (pos 0) conditioning differs from the masked positions; masks are identical
    assert not bool(mx.allclose(addend[0, 0], addend[0, 1]).item())
    assert bool(mx.allclose(addend[0, 1], addend[0, 3]).item())
    # cached after the first call, and shorter inputs slice it
    assert drafter._snr_addend is addend
    assert drafter.embed(ids[:, :2]).shape == (1, 2, 16)


def test_config_rejects_log_snr_without_bounds(tmp_path):
    cfg = {"model_type": "qwen3", "hidden_size": 16, "vocab_size": 32,
           "num_hidden_layers": 1, "intermediate_size": 32, "num_attention_heads": 2,
           "block_size": 4, "mask_token_id": 31, "target_layer_ids": [0, 1],
           "log_snr_conditioning": True}
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="min/max_log_snr"):
        DSparkConfig.from_json(p)


# ---------------------------------------------------------------- hybrid target verify


def _tiny_hybrid(head_dim: int = 8):
    """A real (tiny, random) mlx-lm qwen3_5 hybrid model: 3 linear + 1 full-attn layers."""
    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    args = ModelArgs.from_dict({
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 4,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": head_dim,
            "vocab_size": 64, "rms_norm_eps": 1e-6,
            "linear_num_value_heads": 4, "linear_num_key_heads": 2,
            "linear_key_head_dim": 8, "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
            "tie_word_embeddings": False,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                "partial_rotary_factor": 0.25},
        },
    })
    mx.random.seed(3)
    model = Model(args)
    mx.eval(model.parameters())
    return model


def _committed_reference(target, tokens):
    """Logits at every position of `tokens`, processed as one committed forward."""
    cache = target.make_cache()
    logits, _ = target.run(mx.array([tokens]), cache, [0])
    return logits[0]


def test_hybrid_detection_and_tap_probe():
    t = Target(_tiny_hybrid(), tokenizer=None)
    assert t.is_hybrid and not t.is_vlm
    t.verify_tap()                                   # replicated hybrid loop is faithful


def test_hybrid_partial_accept_rollback_matches_sequential():
    t = Target(_tiny_hybrid(), tokenizer=None)
    prompt = [1, 2, 3, 4, 5]
    committed = prompt + [6, 7, 10, 11, 12]          # what ends up accepted overall
    ref = _committed_reference(t, committed)

    cache = t.make_cache()
    t.reset_spec()
    t.run(mx.array([prompt]), cache, [0])            # prefill
    # round 1: anchor 6, drafts [7, 8, 9] -> 1 accepted (7), 2 rejected
    v1, f1 = t.verify(mx.array([[6, 7, 8, 9]]), cache, [0])
    assert v1.shape[1] == 4 and f1.shape[1] == 4
    t.rollback(cache, 2, [7])                        # rebuilds state at [.., 6, 7]
    # round 2 verifies just [anchor 10, drafts 11, 12] — no replay backlog
    v2, _ = t.verify(mx.array([[10, 11, 12]]), cache, [0])
    assert bool(mx.allclose(v2[0], ref[-3:], atol=1e-4, rtol=1e-4).item())


def test_hybrid_full_accept_keeps_state():
    t = Target(_tiny_hybrid(), tokenizer=None)
    prompt = [1, 2, 3, 4, 5]
    ref = _committed_reference(t, prompt + [6, 7, 8, 10])

    cache = t.make_cache()
    t.reset_spec()
    t.run(mx.array([prompt]), cache, [0])
    _v1, _ = t.verify(mx.array([[6, 7, 8]]), cache, [0])
    t.rollback(cache, 0, [7, 8])                     # full accept: keep everything
    assert t._stash is None
    v2, _ = t.verify(mx.array([[10]]), cache, [0])   # width-1 = pure commit step
    assert bool(mx.allclose(v2[0, -1], ref[-1], atol=1e-4, rtol=1e-4).item())


def test_hybrid_repeated_zero_accept_rounds():
    t = Target(_tiny_hybrid(), tokenizer=None)
    prompt = [1, 2, 3]
    cache = t.make_cache()
    t.reset_spec()
    t.run(mx.array([prompt]), cache, [0])
    # three consecutive 0-accept rounds: each keeps only its anchor
    for anchor in (4, 5, 6):
        t.verify(mx.array([[anchor, 9, 9]]), cache, [0])
        t.rollback(cache, 2, [])
    v2, _ = t.verify(mx.array([[7, 11, 12]]), cache, [0])
    ref = _committed_reference(t, prompt + [4, 5, 6, 7, 11, 12])
    assert bool(mx.allclose(v2[0], ref[-3:], atol=1e-4, rtol=1e-4).item())


def test_hybrid_rollback_trims_kv_by_rejected_only():
    t = Target(_tiny_hybrid(), tokenizer=None)
    prompt = [1, 2, 3, 4, 5]
    cache = t.make_cache()
    t.reset_spec()
    t.run(mx.array([prompt]), cache, [0])
    t.verify(mx.array([[6, 7, 8, 9]]), cache, [0])
    t.rollback(cache, 2, [7])
    kv = [c for c in cache if hasattr(c, "offset")]
    assert kv and all(c.offset == len(prompt) + 2 for c in kv)   # 5 + [6, 7] kept


def test_hybrid_rollback_state_matches_committed_forward():
    # the rebuilt linear state after a partial accept must match a plain committed
    # forward of the same prefix (the whole point of the capture-and-rerun design)
    prompt = [1, 2, 3, 4, 5]
    t = Target(_tiny_hybrid(), tokenizer=None)
    cache = t.make_cache()
    t.reset_spec()
    t.run(mx.array([prompt]), cache, [0])
    t.verify(mx.array([[6, 7, 8, 9]]), cache, [0])
    t.rollback(cache, 2, [7])

    t2 = Target(_tiny_hybrid(), tokenizer=None)      # same seed -> same weights
    cache2 = t2.make_cache()
    t2.reset_spec()
    t2.run(mx.array([prompt + [6, 7]]), cache2, [0])

    lin = [c for c in cache if not hasattr(c, "offset")]
    lin2 = [c for c in cache2 if not hasattr(c, "offset")]
    assert lin and len(lin) == len(lin2)
    for c, c2 in zip(lin, lin2):
        assert bool(mx.allclose(c[0], c2[0], atol=1e-5, rtol=1e-5).item())   # conv window
        assert bool(mx.allclose(c[1], c2[1], atol=1e-4, rtol=1e-4).item())   # delta state


def test_hybrid_reset_spec_clears_capture():
    t = Target(_tiny_hybrid(), tokenizer=None)
    cache = t.make_cache()
    t.reset_spec()
    t.run(mx.array([[1, 2, 3]]), cache, [0])
    t.verify(mx.array([[4, 9]]), cache, [0])
    assert t._stash is not None                      # capture live until the rollback
    t.rollback(cache, 1, [])
    assert t._stash is None
    t.verify(mx.array([[5, 9]]), cache, [0])
    t.reset_spec()
    assert t._stash is None


def test_dense_rollback_still_trims():
    class _Trim:
        def __init__(self):
            self.trimmed = 0
        def trim(self, n):
            self.trimmed += n

    class _Dense:
        layers = []

    t = Target.__new__(Target)
    t.is_hybrid = False
    cache = [_Trim(), _Trim()]
    t.rollback(cache, 3, [1])
    assert all(c.trimmed == 3 for c in cache)
    t.rollback(cache, 0, [])                          # no-op
    assert all(c.trimmed == 3 for c in cache)


# ---------------------------------------------------------------- controller cap-0


def _bonsai_like_controller(**kw):
    # measured Bonsai-27B / M4 Pro shape: near-linear 2-bit verify curve
    verify = {1: 41.0, 2: 53.0, 3: 71.5, 4: 89.0, 5: 108.0}
    drafter = {c: 9.0 + 0.4 * c for c in range(1, 5)}
    kw.setdefault("allow_zero", True)
    kw.setdefault("hybrid_replay", True)
    kw.setdefault("overhead_ms", 6.0)
    return CapController(verify, drafter, max_cap=4, **kw)


def test_cap0_priced_from_width1():
    ctrl = _bonsai_like_controller()
    assert abs(ctrl.rate(0) - 1.0 / 41.0) < 1e-9


def test_parking_gated_until_enough_observations():
    ctrl = _bonsai_like_controller()
    ctrl.p = 0.3                                     # terrible acceptance, cap 0 is best
    for _ in range(4):                               # a repick fires, but _obs is tiny
        ctrl.update(0, 2)
    assert ctrl.cap >= 1                             # cold start may not park


def test_parks_on_low_acceptance_and_probes():
    ctrl = _bonsai_like_controller()
    for _ in range(40):                              # 0-of-2 rounds: p collapses
        ctrl.update(0, 2)
    assert ctrl._best == 0
    caps = []
    for _ in range(16):
        ctrl.update(0, 0)
        caps.append(ctrl.cap)
    assert caps.count(1) == 2                        # a probe every probe_every rounds
    assert caps.count(0) == 14


def test_parked_controller_recovers_on_content_shift():
    # REGRESSION GUARD for the reverted probe backoff (2026-07-16, see NOTES): while
    # parked, probes are the only acceptance signal, so the fixed every-8 cadence is
    # load-bearing — a parked controller fed good probe outcomes must un-park within
    # a bounded number of rounds (the backoff variant stayed parked ~forever).
    ctrl = _bonsai_like_controller()
    for _ in range(40):                              # low-acceptance content: parks
        ctrl.update(0, 2, round_ms=110.0, committed=1)
    assert ctrl._best == 0
    rounds_to_unpark = None
    for i in range(400):                             # content shifts: probes all accept
        if ctrl.cap == 1:                            # probe: 1 draft + bonus in ~69 ms
            ctrl.update(1, 1, round_ms=69.0, committed=2)
        else:                                        # parked plain step
            ctrl.update(0, 0, round_ms=43.0, committed=1)
        if ctrl._best >= 1:
            rounds_to_unpark = i + 1
            break
    # recovery rides the observed-timing layer: ~4 good probes at the every-8 cadence
    assert rounds_to_unpark is not None and rounds_to_unpark <= 100


def test_observed_timings_override_model():
    ctrl = _bonsai_like_controller()
    ctrl.p = 0.95                                    # model loves speculation
    # ... but measured rounds at cap 2 are catastrophically slow vs a plain step
    for _ in range(40):
        ctrl.update(2, 2, round_ms=200.0, committed=3)
    assert ctrl._best == 0                           # observed reality wins


def test_update_backward_compatible_positional():
    ctrl = _bonsai_like_controller(allow_zero=False)
    ctrl.update(2, 2)                                # old call shape still fine
    assert ctrl.rounds == 1


# ---------------------------------------------------------------- routing + registry


def test_route_qwen3_5_multimodal_to_mlx_lm():
    cfg = {"model_type": "qwen3_5", "vision_config": {"depth": 27}}
    assert _route_target(cfg) == "mlx_lm"


def test_route_gemma4_unified_stays_mlx_vlm():
    cfg = {"model_type": "gemma4_unified", "vision_config": {}}
    assert _route_target(cfg) == "mlx_vlm"


def test_registry_resolves_bonsai_targets():
    _tgt, drf = resolve("prism-ml/Ternary-Bonsai-27B-mlx-2bit", mode="dspark")
    assert drf == "Rahim/Ternary-Bonsai-27B-dspark"
    mode, _tgt, drf = resolve_mode("prism-ml/Ternary-Bonsai-27B-mlx-2bit", mode="auto")
    assert mode == "dspark" and drf == "Rahim/Ternary-Bonsai-27B-dspark"
    # the 1-bit variant is deliberately NOT registered (1-bit pack needs PrismML's MLX fork)
    with pytest.raises(ValueError, match="no built-in"):
        resolve("prism-ml/Bonsai-27B-mlx-1bit", mode="dspark")


def test_load_target_refuses_unsupported_quant_bits(tmp_path):
    from mlx_dspark.load import load_target

    (tmp_path / "config.json").write_text(json.dumps(
        {"model_type": "qwen3_5", "quantization": {"group_size": 128, "bits": 1}}))
    with pytest.raises(ValueError, match=r"1 bits.*stock MLX"):
        load_target(str(tmp_path))


def test_gguf_drafter_scheme_parses():
    """The "gguf:" drafter scheme stays supported for converting future GGUF-only drops."""
    from unittest.mock import patch

    from mlx_dspark.load import _resolve

    with patch("mlx_dspark.gguf_convert.ensure_converted", return_value="/tmp/x") as ec:
        assert _resolve("gguf:some-org/some-repo/some-drafter-bf16.gguf") == "/tmp/x"
        ec.assert_called_once_with("some-org/some-repo", "some-drafter-bf16.gguf")


def test_hybrid_kv_bits_builds_mixed_cache_and_rollback_matches_committed():
    """kv_bits on a gated-DeltaNet hybrid quantizes ONLY the full-attention layers' KV —
    the entire per-token cache growth, and the whole verify depth slope at long context
    (NOTES "Long-context decode", 2026-08-20) — while recurrent layers keep their
    fixed-size ArraysCache. The dense kv-bits guarantee carries over: spec verify +
    partial-accept rollback on the kv8 cache reproduces the kv8 committed forward.
    (head_dim 32 + kv_group_size 32: mlx quantization needs groups of >= 32.)"""
    from mlx_lm.models.cache import KVCache, QuantizedKVCache

    t = Target(_tiny_hybrid(head_dim=32), tokenizer=None, kv_bits=8, kv_group_size=32)
    kinds = [type(c) for c in t.make_cache()]
    assert kinds.count(QuantizedKVCache) == 1        # 1 of 4 layers is full attention
    assert KVCache not in kinds

    committed = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
    ref = _committed_reference(t, committed)         # kv8 end to end (same make_cache)

    cache = t.make_cache()
    t.reset_spec()
    t.run(mx.array([committed[:5]]), cache, [0])
    v1, _ = t.verify(mx.array([[6, 7, 8, 9]]), cache, [0])
    assert v1.shape[1] == 4
    t.rollback(cache, 2, [7])                        # rebuilds state at [.., 6, 7]
    v2, _ = t.verify(mx.array([[10, 11, 12]]), cache, [0])
    assert bool(mx.allclose(v2[0], ref[-3:], atol=1e-4, rtol=1e-4).item())


def test_hybrid_kv_bits_mamba2_refused_with_reason():
    """The Mamba-2 (nemotron_h) mixed-cache path is unvalidated — refuse it by name."""
    import pytest

    class _Layer:
        block_type = "M"

    class _Fake:
        layers = [_Layer()]

    with pytest.raises(ValueError, match="Mamba-2"):
        Target(_Fake(), tokenizer=None, kv_bits=8)
