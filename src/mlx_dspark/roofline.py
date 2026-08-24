"""Roofline + machine telemetry: the interpretation layer behind "is this Mac saturated?"

Every "GPU utilization" number on a Mac lies about LLM decode: token-by-token generation
reads every weight per token and does ~1 FLOP per byte, so the GPU reads "busy" while its
ALUs idle waiting on memory. The number that tells the truth is a comparison against physics:

    decode ceiling (tok/s) = memory bandwidth / (weight bytes touched per token + KV bytes x context)

This module is the pure half of that comparison — the chip table, the macOS memory-pressure
and swap readings, the weight-footprint accounting (MoE-active, gather-aware) and the verdict
that turns the numbers into a sentence with a next action. It imports no mlx so it runs on any
thread and is testable model-free; the one measurement it needs (this machine's *measured*
bandwidth) lives in :mod:`calibrate` and is cached like every other curve.

Two things make this more honest here than in a generic monitor:

- the engine knows the model exactly — loaded bytes per tensor (quantized), which layers grow a
  KV cache, how many experts route per token — so the denominator is measured, not guessed;
- speculative decoding legitimately *beats* the single-stream ceiling (several tokens per
  weight read), so the ratio is reported as "x the roofline", never as a utilization. The
  machine-health question ("is the baseline healthy?") is answered from the calibration's
  measured width-1 step instead, which costs nothing extra.

Lineage: the framing and thresholds come from the author's unpublished ``inferviz`` roofline
dashboard (2026-07), whose ">100% MBU = speculative decoding working" verdict branch was
written after pointing it at mlx-dspark.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import re
import subprocess
from functools import lru_cache

# --------------------------------------------------------------------------- chip table

# Theoretical unified-memory bandwidth, GB/s, by chip family. Two traps: newer != faster
# (M3 Pro regressed vs M2 Pro) and one name can span two binnings (M3/M4/M5 Max — told apart
# by GPU core count, see _BINNED). Sourced from the mlx skill's measured/published table.
BANDWIDTH_GB_S: dict[str, float] = {
    "M1": 68, "M2": 100, "M3": 102, "M4": 120, "M5": 153,
    "M1 Pro": 200, "M2 Pro": 200, "M3 Pro": 150, "M4 Pro": 273, "M5 Pro": 307,
    "M1 Max": 400, "M2 Max": 400,
    "M1 Ultra": 800, "M2 Ultra": 800, "M3 Ultra": 819,
}
_BINNED: dict[str, list[tuple[int, float]]] = {
    "M3 Max": [(30, 307.0), (40, 410.0)],
    "M4 Max": [(32, 410.0), (40, 546.0)],
    "M5 Max": [(32, 460.0), (40, 614.0)],
}

# The machine every registry speedup badge was measured on: an M4 Pro, 273 GB/s on the spec
# sheet, ~226 GB/s by the microbench below. A client scales a stamped tok/s by
# :func:`bandwidth_scale` for a rough figure on *this* Mac — compared like-for-like
# (theoretical vs theoretical when the chip is in the table, else measured vs measured), so
# the reference machine itself reads 1.0.
REFERENCE_BANDWIDTH_GB_S = 273.0
REFERENCE_MEASURED_GB_S = 226.0


def bandwidth_scale(theoretical_gb_s: float | None, measured_gb_s: float | None) -> float | None:
    """This Mac's decode bandwidth relative to the reference M4 Pro (1.0 = same class)."""
    if theoretical_gb_s:
        return round(theoretical_gb_s / REFERENCE_BANDWIDTH_GB_S, 3)
    if measured_gb_s:
        return round(measured_gb_s / REFERENCE_MEASURED_GB_S, 3)
    return None


def chip_family(device_name: str | None) -> tuple[str | None, str | None]:
    """``("M4", "M4 Pro")`` from ``"Apple M4 Pro"``; ``(None, None)`` when unparseable."""
    m = re.search(r"\bM(\d+)(?:\s+(Pro|Max|Ultra))?\b", device_name or "")
    if not m:
        return None, None
    gen = f"M{m.group(1)}"
    return gen, gen + (f" {m.group(2)}" if m.group(2) else "")


def theoretical_bandwidth(device_name: str | None,
                          gpu_cores: int | None = None) -> tuple[float | None, str]:
    """``(GB/s, source)`` — source is ``"table"``, ``"table-binned"`` (Max chips told apart by
    GPU cores; top binning assumed when unknown) or ``"unknown"``."""
    _, family = chip_family(device_name)
    if family is None:
        return None, "unknown"
    if family in BANDWIDTH_GB_S:
        return float(BANDWIDTH_GB_S[family]), "table"
    bins = _BINNED.get(family)
    if bins is None:
        return None, "unknown"
    if gpu_cores is None:
        return bins[-1][1], "table-binned"
    return min(bins, key=lambda b: abs(b[0] - gpu_cores))[1], "table-binned"


@lru_cache(maxsize=1)
def gpu_core_count() -> int | None:
    """GPU core count from the IO registry (no sudo, ~30 ms, cached). None off-macOS."""
    try:
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
                             capture_output=True, text=True, timeout=5)
        m = re.search(r'"gpu-core-count"\s*=\s*(\d+)', out.stdout)
        return int(m.group(1)) if m else None
    except Exception:  # noqa: BLE001 — no ioreg (Linux CI) / timeout
        return None


def chip_info(device_name: str | None, gpu_cores: int | None = None) -> dict:
    """The chip row a client renders: name, family, cores, theoretical bandwidth."""
    if gpu_cores is None:
        gpu_cores = gpu_core_count()
    gen, family = chip_family(device_name)
    bw, source = theoretical_bandwidth(device_name, gpu_cores)
    return {"name": device_name, "generation": gen, "family": family,
            "gpu_cores": gpu_cores, "bandwidth_gb_s": bw, "bandwidth_source": source}


# --------------------------------------------------------------------------- sysctl

_libc = None


def _sysctlbyname(name: bytes, buf) -> bool:
    global _libc
    try:
        if _libc is None:
            _libc = ctypes.CDLL(ctypes.util.find_library("c"))
        size = ctypes.c_size_t(ctypes.sizeof(buf))
        return _libc.sysctlbyname(name, ctypes.byref(buf), ctypes.byref(size), None, 0) == 0
    except Exception:  # noqa: BLE001 — not darwin / no libc symbol
        return False


def _sysctl_int(name: str) -> int | None:
    v = ctypes.c_int64(0)
    return int(v.value) if _sysctlbyname(name.encode(), v) else None


class _XswUsage(ctypes.Structure):
    # <sys/sysctl.h> struct xsw_usage — what `sysctl vm.swapusage` decodes.
    _fields_ = [("total", ctypes.c_uint64), ("avail", ctypes.c_uint64),
                ("used", ctypes.c_uint64), ("pagesize", ctypes.c_uint32),
                ("encrypted", ctypes.c_uint32)]


# kern.memorystatus_vm_pressure_level: DISPATCH_MEMORYPRESSURE_NORMAL=1 / WARN=2 / CRITICAL=4
PRESSURE_LABELS = {1: "normal", 2: "warn", 4: "critical"}


def memory_pressure() -> dict:
    """macOS's own memory-pressure verdict — the signal Activity Monitor's gauge colours by.
    ``{"level": 1|2|4|None, "label": "normal"|"warn"|"critical"|"unknown"}``."""
    lvl = _sysctl_int("kern.memorystatus_vm_pressure_level")
    return {"level": lvl, "label": PRESSURE_LABELS.get(lvl, "unknown")}


def swap_usage() -> dict:
    """``{"used_bytes", "total_bytes"}`` from ``vm.swapusage`` (zeros off-macOS)."""
    x = _XswUsage()
    if _sysctlbyname(b"vm.swapusage", x):
        return {"used_bytes": int(x.used), "total_bytes": int(x.total)}
    return {"used_bytes": 0, "total_bytes": 0}


def system_memory() -> dict:
    """One cheap snapshot (a handful of sysctls, microseconds) of what the OS sees.

    ``free_percent`` is ``kern.memorystatus_level`` — macOS's own "memory level": the share
    of RAM it could reclaim *without swapping* (free + inactive/file-cache pages), the number
    its pressure verdict keys off. Not the inverse of Activity Monitor's "Used" (app + wired +
    compressed), so it reads higher than ``100 − used%``. ``wired_limit_mb`` is the sysctl a
    user may have raised (0/None = default)."""
    swap = swap_usage()
    wired = _sysctl_int("iogpu.wired_limit_mb")
    return {
        "total_bytes": _sysctl_int("hw.memsize"),
        "pressure": memory_pressure()["label"],
        "pressure_level": memory_pressure()["level"],
        "free_percent": _sysctl_int("kern.memorystatus_level"),
        "swap_used_bytes": swap["used_bytes"],
        "swap_total_bytes": swap["total_bytes"],
        "wired_limit_mb": wired or None,
    }


# --------------------------------------------------------------------------- weight footprint

_ROUTED_EXPERT_MARKERS = (".switch_mlp.",)   # mlx-lm / mlx-vlm SwitchGLU/SwitchMLP convention


def _experts_from_config(cfg: dict | None) -> tuple[int | None, int | None]:
    cfg = cfg or {}
    for c in (cfg.get("text_config") or {}, cfg):
        n = (c.get("num_experts") or c.get("num_local_experts") or c.get("n_routed_experts"))
        k = c.get("num_experts_per_tok") or c.get("experts_per_token")
        if isinstance(n, int) and n > 1:
            return n, (k if isinstance(k, int) and k > 0 else None)
    return None, None


def weight_footprint(params: list[tuple[str, int]], cfg: dict | None = None) -> dict:
    """Bytes a single decode step actually *reads*, from ``[(param_name, nbytes), ...]``.

    - ``total_bytes``: everything resident (what RAM pays for).
    - ``active_bytes``: the roofline denominator. Routed-expert tensors (under ``switch_mlp``,
      the mlx-lm convention) count ``top_k / n_experts`` of their bytes — a MoE pulls only its
      routed experts per token; shared experts and everything else count in full.
    - ``embed_tokens`` is a row *gather*, not a full read (the AUDIT lesson: counting it as
      active overstated MoE roofline by ~10%) — excluded unless the checkpoint has no separate
      ``lm_head`` (tied embeddings: then the same matrix IS read in full by the output head).
    """
    names = [n for n, _ in params]
    has_lm_head = any(n.endswith("lm_head.weight") for n in names)
    n_experts, top_k = _experts_from_config(cfg)
    total = active = expert_bytes = embed_bytes = 0
    for name, nbytes in params:
        nbytes = int(nbytes or 0)
        total += nbytes
        if any(m in name for m in _ROUTED_EXPERT_MARKERS):
            expert_bytes += nbytes
            continue
        if ".embed_tokens." in name or name.endswith("embed_tokens.weight"):
            embed_bytes += nbytes
            continue
        active += nbytes
    is_moe = expert_bytes > 0 and bool(n_experts)
    if expert_bytes:
        frac = (top_k / n_experts) if (is_moe and top_k) else 1.0
        active += int(expert_bytes * frac)
    if not has_lm_head:
        active += embed_bytes          # tied: the output projection reads the whole table
    return {
        "total_bytes": total,
        "active_bytes": active,
        "expert_bytes": expert_bytes,
        "embed_bytes": embed_bytes,
        "is_moe": is_moe,
        "n_experts": n_experts if is_moe else None,
        "experts_per_tok": top_k if is_moe else None,
        # honest label for the estimate: exact for dense; MoE assumes top_k distinct experts
        # per token (true at width 1) and no shared-expert double counting
        "active_is_estimate": bool(expert_bytes and not (is_moe and top_k)),
    }


# --------------------------------------------------------------------------- roofline math


def bytes_per_token(active_bytes: int, kv_bytes_per_token: int | None, context: int) -> int:
    return int(active_bytes) + int(kv_bytes_per_token or 0) * max(int(context or 0), 0)


def ceiling_tps(bandwidth_gb_s: float | None, bytes_per_tok: int) -> float | None:
    """Single-stream decode ceiling: one full weight (+KV) read per token at this bandwidth."""
    if not bandwidth_gb_s or bytes_per_tok <= 0:
        return None
    return bandwidth_gb_s * 1e9 / bytes_per_tok


def roofline(*, bandwidth_gb_s: float | None, active_bytes: int,
             kv_bytes_per_token: int | None, context: int = 0) -> dict:
    """Ceiling + bytes/token at one context depth. ``None`` ceiling when bandwidth is unknown."""
    bpt = bytes_per_token(active_bytes, kv_bytes_per_token, context)
    return {"context": int(context or 0), "bytes_per_token": bpt,
            "ceiling_tps": ceiling_tps(bandwidth_gb_s, bpt)}


def baseline_mbu(step_ms: float | None, bytes_per_tok: int,
                 bandwidth_gb_s: float | None) -> dict | None:
    """Model-bandwidth utilization of a measured single-row step — the machine-health number.

    ``step_ms`` is the calibration's width-1 verify time (a plain decode step at ctx 512), so
    this costs no new measurement. ``achieved_gb_s = bytes / step``; ``mbu`` against the
    measured bandwidth. >=0.75 healthy, 0.5–0.75 software overhead, <0.5 structural."""
    if not step_ms or step_ms <= 0 or bytes_per_tok <= 0:
        return None
    achieved = bytes_per_tok / (step_ms / 1e3) / 1e9
    out = {"step_ms": round(float(step_ms), 3), "achieved_gb_s": round(achieved, 1),
           "mbu": None}
    if bandwidth_gb_s:
        out["mbu"] = round(achieved / bandwidth_gb_s, 4)
    return out


# --------------------------------------------------------------------------- verdict

HEALTHY_MBU = 0.75
OK_MBU = 0.50
SWAP_GROWTH_BYTES = 64 * 1024 * 1024      # "fits-but-swaps": swap grew during one generation
DECAY_ATTENTION = 0.85                    # late-run rate below 85% of early = decaying
TINY_MODEL_TPS = 200.0                    # above this, per-token fixed costs dominate, not bytes


def verdict(*, mbu: float | None, ratio_to_ceiling: float | None = None, mode: str | None = None,
            accept_len: float | None = None, decode_tps: float | None = None,
            context_tokens: int | None = None, context_window: int | None = None,
            pressure: str | None = None, swap_delta_bytes: int | None = None,
            decay_ratio: float | None = None, cold: bool = False,
            is_moe_estimate: bool = False) -> dict:
    """Turn the numbers into a judgement: ``{level, headline, findings[], levers[]}``.

    Pure and data-driven — no measured speedups are baked into the strings (they rot; the
    original inferviz verdict carried this project's own stale numbers). The ladder: memory
    cliff first (it invalidates everything else), then the roofline comparison, then the
    speculative-decoding reading, then run-shape findings (decay, cold, context fill).
    ``level`` is one of ``info | healthy | ok | attention | problem``.
    """
    findings: list[str] = []
    levers: list[str] = []
    level = "info"
    headline = "No decode measured yet."
    gb = 1024 ** 3

    if mbu is not None:
        pct = mbu * 100
        if decode_tps is not None and decode_tps > TINY_MODEL_TPS and mbu < HEALTHY_MBU:
            level = "ok"
            headline = (f"Baseline step at {pct:.0f}% of this Mac's bandwidth — small-model "
                        "regime: per-step fixed costs, not weight bytes, set the pace here.")
        elif mbu >= HEALTHY_MBU:
            level = "healthy"
            headline = (f"Plain decode runs at {pct:.0f}% of this Mac's measured bandwidth — "
                        "the machine is saturated; only fewer bytes per token or more tokens "
                        "per weight read can go faster.")
        elif mbu >= OK_MBU:
            level = "ok"
            headline = (f"Plain decode at {pct:.0f}% of measured bandwidth — decent, "
                        f"~{100 - pct:.0f}% is going to overhead.")
            levers.append("Re-check after the machine is warm; a cold process (kernel "
                          "compile + clock ramp) reads low.")
        else:
            level = "attention"
            headline = (f"Plain decode at only {pct:.0f}% of measured bandwidth — something "
                        "structural (memory pressure, thermals, Low Power Mode).")
            levers.append("Check memory pressure first: a model that nearly fills RAM makes "
                          "macOS compress/swap and tok/s falls off a cliff.")
        if is_moe_estimate:
            findings.append("MoE active-bytes are an estimate (routed experts x top_k / n); "
                            "treat the utilization as approximate.")

    if ratio_to_ceiling is not None and mode and mode != "baseline":
        x = ratio_to_ceiling
        if x >= 1.05:
            findings.append(f"Speculative decoding ({mode}) is delivering {x:.2f}x the "
                            "single-stream roofline — several tokens per weight read.")
        elif x >= 0.85:
            findings.append(f"Speculative decoding ({mode}) is at {x:.2f}x the roofline: "
                            "near parity with plain decoding on this content.")
        else:
            findings.append(f"Speculative decoding ({mode}) is at {x:.2f}x the roofline — "
                            "below what a plain step would do; the drafts are not paying "
                            "on this content.")
            if accept_len is not None and accept_len < 2.0:
                levers.append(f"Accept length {accept_len:.2f} is low: a smaller draft cap "
                              "(or baseline mode) costs less per rejected draft.")
        if accept_len is not None and accept_len >= 2.0 and x >= 1.05:
            levers.append("Content decides acceptance: code and math accept far more than "
                          "open chat, so the same model runs much faster there.")

    if pressure in ("warn", "critical"):
        level = "problem"
        findings.insert(0, f"macOS reports memory pressure: {pressure.upper()}. Expect "
                           "reduced speed until it clears.")
        levers.insert(0, "Free memory: close other apps, lower the context window, or pick "
                         "a smaller quant / model.")
    if swap_delta_bytes is not None and swap_delta_bytes > SWAP_GROWTH_BYTES:
        level = "problem"
        findings.insert(0, f"Swap grew {swap_delta_bytes / gb:.2f} GB during this generation "
                           "— the fits-but-swaps cliff. These numbers are not the model's "
                           "fault.")
        if "Free memory" not in " ".join(levers):
            levers.insert(0, "Free memory or lower the context window; the working set no "
                             "longer fits alongside everything else that is open.")

    if cold:
        findings.append("First generation since load with warmup off: includes Metal kernel "
                        "compilation and clock ramp — judge the next one.")
    if decay_ratio is not None and decay_ratio < DECAY_ATTENTION:
        findings.append(f"Decode slowed to {decay_ratio * 100:.0f}% of its early rate within "
                        "this generation — growing context cost or thermal throttling; "
                        "pronounced on a short output points at thermals.")
    if context_tokens and context_window:
        fill = context_tokens / context_window
        if fill >= 0.9:
            findings.append(f"Context is {fill * 100:.0f}% full ({context_tokens} of "
                            f"{context_window} tokens) — the next turns will hit the "
                            "'prompt is too long' limit.")
            levers.append("Start a new conversation, or raise the context window if RAM "
                          "allows (KV cache grows linearly with it).")

    return {"level": level, "headline": headline, "findings": findings, "levers": levers}


# --------------------------------------------------------------------------- warnings


def system_warnings(system: dict | None, load_notes: list[str] | None = None) -> list[dict]:
    """The ``/health.warnings`` list: ``{code, level, message, action}`` rows a client shows as
    a banner. Memory pressure is read live; ``load_notes`` are the engine's own load-time
    notes (e.g. the context-window RAM estimate) that used to reach only stderr."""
    out: list[dict] = []
    pressure = (system or {}).get("pressure")
    if pressure in ("warn", "critical"):
        gb = 1024 ** 3
        swap = (system or {}).get("swap_used_bytes") or 0
        out.append({
            "code": "memory_pressure", "level": "problem" if pressure == "critical" else "attention",
            "message": (f"macOS memory pressure is {pressure.upper()}"
                        + (f" ({swap / gb:.1f} GB swapped)" if swap else "")
                        + " — generation will be slower until it clears."),
            "action": "Close other apps, lower the context window, or use a smaller model/quant.",
        })
    for note in load_notes or []:
        out.append({"code": "load_note", "level": "attention", "message": note, "action": None})
    return out
