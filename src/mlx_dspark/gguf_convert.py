"""Convert a PrismML **dspark drafter GGUF** into a DeepSpec-layout safetensors checkpoint.

PrismML ships DSpark drafters for the Bonsai-27B targets as GGUF only (e.g.
``prism-ml/Ternary-Bonsai-27B-gguf/Ternary-Bonsai-27B-dspark-bf16.gguf``); no safetensors
export is published. Their converter (``conversion/dspark.py`` in PrismML-Eng/llama.cpp)
maps the original HF ``Qwen3DSparkModel`` export onto GGUF names — this module inverts that
map, so the output is exactly the DeepSpec standalone layout ``load_drafter`` already reads
(``fc`` / ``hidden_norm`` / ``layers.{i}.*`` / ``markov_head.markov_w1|w2`` /
``confidence_head.proj`` / ``lm_head`` / ``embed_tokens`` / ``norm``), plus the Bonsai-only
``log_snr_embed.fc1|fc2`` GIDD-conditioning MLP.

Use the **bf16** drafter GGUF: it carries only BF16/F32 tensors (this parser is
deliberately minimal and does not dequantize llama.cpp quant formats — the Q4_1 file also
packs the token embedding in a PrismML-fork ternary type that upstream tools can't read).
``load_drafter`` quantizes to 4-bit at load anyway, which does not affect acceptance.

The GGUF container format (v3) is stable and tiny to parse; doing it here avoids depending
on ``mx.load``'s gguf coverage (no BF16) or the ``gguf`` pip package.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np

_GGUF_MAGIC = b"GGUF"
_ALIGN_DEFAULT = 32

# ggml tensor type ids this parser accepts (bf16 drafter GGUFs contain only these)
_GGML_F32, _GGML_F16, _GGML_BF16 = 0, 1, 30
_NP_OF = {_GGML_F32: (np.float32, 4), _GGML_F16: (np.float16, 2), _GGML_BF16: (np.uint16, 2)}

# GGUF metadata value types
_SCALAR_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
               7: "<B", 10: "<Q", 11: "<q", 12: "<d"}

# structural / head tensors: gguf name -> DeepSpec (HF) name. Per-layer ``blk.{i}.*``
# tensors are handled by _LAYER_MAP below.
_HEAD_MAP = {
    "token_embd.weight":              "embed_tokens.weight",
    "output.weight":                  "lm_head.weight",
    "output_norm.weight":             "norm.weight",
    "dspark.fc.weight":               "fc.weight",
    "dspark.hidden_norm.weight":      "hidden_norm.weight",
    "dspark.markov_head_a.weight":    "markov_head.markov_w1.weight",
    "dspark.markov_head_b.weight":    "markov_head.markov_w2.weight",
    "dspark.confidence_head.weight":  "confidence_head.proj.weight",
    "dspark.confidence_head.bias":    "confidence_head.proj.bias",
    "dspark.log_snr_fc1.weight":      "log_snr_embed.fc1.weight",
    "dspark.log_snr_fc1.bias":        "log_snr_embed.fc1.bias",
    "dspark.log_snr_fc2.weight":      "log_snr_embed.fc2.weight",
    "dspark.log_snr_fc2.bias":        "log_snr_embed.fc2.bias",
}
_LAYER_MAP = {
    "attn_norm":   "input_layernorm",
    "ffn_norm":    "post_attention_layernorm",
    "attn_q":      "self_attn.q_proj",
    "attn_k":      "self_attn.k_proj",
    "attn_v":      "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
    "ffn_gate":    "mlp.gate_proj",
    "ffn_up":      "mlp.up_proj",
    "ffn_down":    "mlp.down_proj",
}


class _Reader:
    def __init__(self, buf: bytes):
        self.buf = buf
        self.off = 0

    def read(self, fmt: str):
        v = struct.unpack_from(fmt, self.buf, self.off)
        self.off += struct.calcsize(fmt)
        return v[0] if len(v) == 1 else v

    def read_str(self) -> str:
        n = self.read("<Q")
        s = self.buf[self.off:self.off + n].decode("utf-8")
        self.off += n
        return s

    def read_value(self, vtype: int):
        if vtype == 8:
            return self.read_str()
        if vtype == 9:
            etype = self.read("<I")
            n = self.read("<Q")
            return [self.read_value(etype) for _ in range(n)]
        v = self.read(_SCALAR_FMT[vtype])
        return bool(v) if vtype == 7 else v


def read_gguf_header(path: str | Path, probe_bytes: int = 64 * 1024 * 1024):
    """Parse a GGUF file's metadata + tensor table (not the tensor data).

    Returns ``(metadata: dict, tensors: list[(name, shape_ne, ggml_type, rel_offset)],
    data_start: int)``. ``shape_ne`` is GGUF order (ne0 = fastest-varying)."""
    with open(path, "rb") as f:
        buf = f.read(probe_bytes)
    if buf[:4] != _GGUF_MAGIC:
        raise ValueError(f"{path}: not a GGUF file (magic {buf[:4]!r})")
    r = _Reader(buf)
    r.off = 4
    version = r.read("<I")
    if version < 2:
        raise ValueError(f"{path}: GGUF v{version} is too old (need v2+)")
    n_tensors = r.read("<Q")
    n_kv = r.read("<Q")
    meta = {}
    for _ in range(n_kv):
        key = r.read_str()
        vtype = r.read("<I")
        meta[key] = r.read_value(vtype)
    tensors = []
    for _ in range(n_tensors):
        name = r.read_str()
        n_dims = r.read("<I")
        ne = [r.read("<Q") for _ in range(n_dims)]
        ttype = r.read("<I")
        offset = r.read("<Q")
        tensors.append((name, ne, ttype, offset))
    align = int(meta.get("general.alignment", _ALIGN_DEFAULT))
    data_start = (r.off + align - 1) // align * align
    return meta, tensors, data_start


def _map_name(gguf_name: str) -> str:
    if gguf_name in _HEAD_MAP:
        return _HEAD_MAP[gguf_name]
    m = re.fullmatch(r"blk\.(\d+)\.(\w+)\.(weight|bias)", gguf_name)
    if m and m.group(2) in _LAYER_MAP:
        return f"layers.{m.group(1)}.{_LAYER_MAP[m.group(2)]}.{m.group(3)}"
    raise ValueError(
        f"unrecognized dspark GGUF tensor {gguf_name!r} — this drafter's layout is newer "
        f"than this converter. Open an issue: https://github.com/ARahim3/mlx-dspark/issues"
    )


def _synthesize_config(meta: dict, vocab_size: int) -> dict:
    """DeepSpec-style drafter config.json from the GGUF's dspark.* metadata."""
    def need(key):
        if key not in meta:
            raise ValueError(f"dspark GGUF is missing required metadata {key!r}")
        return meta[key]

    hidden = int(need("dspark.embedding_length"))
    head_dim = int(meta.get("dspark.attention.key_length",
                            hidden // int(need("dspark.attention.head_count"))))
    cfg = {
        "architectures": ["Qwen3DSparkModel"],
        "model_type": "qwen3",           # the drafter trunk is a plain Qwen3-style stack
        "hidden_size": hidden,
        "intermediate_size": int(need("dspark.feed_forward_length")),
        "num_hidden_layers": int(need("dspark.block_count")),
        "num_attention_heads": int(need("dspark.attention.head_count")),
        "num_key_value_heads": int(need("dspark.attention.head_count_kv")),
        "head_dim": head_dim,
        "rms_norm_eps": float(meta.get("dspark.attention.layer_norm_rms_epsilon", 1e-6)),
        "vocab_size": vocab_size,
        "rope_parameters": {"rope_type": "default",
                            "rope_theta": float(meta.get("dspark.rope.freq_base", 1e6))},
        "block_size": int(need("dspark.dspark.block_size")),
        "mask_token_id": int(need("dspark.dspark.mask_token_id")),
        "target_layer_ids": [int(x) for x in need("dspark.dspark.target_layers")],
        "markov_rank": int(meta.get("dspark.dspark.markov_rank", 0)),
        "enable_confidence_head": bool(meta.get("dspark.dspark.confidence_head", False)),
        "confidence_head_with_markov": bool(
            meta.get("dspark.dspark.confidence_head_with_markov", False)),
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "converted_from": str(meta.get("general.name", "dspark gguf")),
    }
    if meta.get("dspark.dspark.log_snr_conditioning"):
        cfg["log_snr_conditioning"] = True
        cfg["min_log_snr"] = float(meta["dspark.dspark.min_log_snr"])
        cfg["max_log_snr"] = float(meta["dspark.dspark.max_log_snr"])
    return cfg


def convert_dspark_gguf(gguf_path: str | Path, out_dir: str | Path) -> Path:
    """Convert a PrismML dspark drafter GGUF (bf16) to a DeepSpec-layout safetensors
    checkpoint at ``out_dir`` (created). Returns ``out_dir``. Idempotent per output dir:
    an existing complete conversion is returned as-is."""
    gguf_path, out_dir = Path(gguf_path), Path(out_dir)
    cfg_path, st_path = out_dir / "config.json", out_dir / "model.safetensors"
    if cfg_path.exists() and st_path.exists():
        return out_dir

    meta, tensors, data_start = read_gguf_header(gguf_path)
    if meta.get("general.architecture") != "dspark":
        raise ValueError(
            f"{gguf_path}: general.architecture is "
            f"{meta.get('general.architecture')!r}, not 'dspark' — this is not a dspark "
            f"drafter GGUF (the drafter ships as a separate *dspark*.gguf file alongside "
            f"the model weights)."
        )
    unsupported = sorted({t for _, _, t, _ in tensors} - set(_NP_OF))
    if unsupported:
        raise ValueError(
            f"{gguf_path}: contains ggml tensor types {unsupported} this converter does not "
            f"dequantize — use the drafter's **bf16** GGUF (…dspark-bf16.gguf); "
            f"mlx-dspark quantizes it to 4-bit at load."
        )

    # vocab width lives only in token_embd's own shape (the drafter ships no tokenizer)
    vocab_size = next(ne[1] for name, ne, _, _ in tensors if name == "token_embd.weight")
    cfg = _synthesize_config(meta, int(vocab_size))

    weights: dict[str, mx.array] = {}
    with open(gguf_path, "rb") as f:
        for name, ne, ttype, rel_off in tensors:
            hf_name = _map_name(name)
            np_dtype, itemsize = _NP_OF[ttype]
            count = 1
            for d in ne:
                count *= int(d)
            f.seek(data_start + rel_off)
            raw = np.frombuffer(f.read(count * itemsize), dtype=np_dtype)
            # GGUF ne is [fastest, ..., slowest]; row-major HF shape is the reverse.
            arr = mx.array(raw.reshape([int(d) for d in reversed(ne)]))
            if ttype == _GGML_BF16:
                arr = arr.view(mx.bfloat16)
            weights[hf_name] = arr.astype(mx.bfloat16)   # uniform bf16, like DeepSpec exports

    out_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(st_path), weights)
    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh, indent=2)
    return out_dir


def ensure_converted(repo: str, filename: str) -> str:
    """Materialize the DeepSpec-layout conversion of ``repo/filename`` (a dspark GGUF on
    the HF hub), downloading + converting on first use. The conversion is cached under
    ``~/.cache/mlx_dspark/drafters/<stem>`` (~7.3 GB bf16; the source GGUF stays in the HF
    cache — ``huggingface-cli delete-cache`` reclaims it). Returns the local drafter path."""
    from huggingface_hub import hf_hub_download

    stem = Path(filename).stem.lower()
    out_dir = Path.home() / ".cache" / "mlx_dspark" / "drafters" / stem
    if (out_dir / "config.json").exists() and (out_dir / "model.safetensors").exists():
        return str(out_dir)
    print(f"[mlx-dspark] fetching dspark drafter {repo}/{filename} (one-time, ~7.3 GB)…")
    gguf = hf_hub_download(repo, filename)
    print(f"[mlx-dspark] converting to {out_dir} …")
    convert_dspark_gguf(gguf, out_dir)
    return str(out_dir)
