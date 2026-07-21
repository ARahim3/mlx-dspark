"""Machine + model inventory as structured data.

``mlx-dspark doctor`` already answered "will this work here?", but only as coloured text for a
human. The Mac app needs the same answers as JSON — to gate onboarding on Apple Silicon, to
filter the model picker by what actually fits this machine's RAM, and to show which models are
already on disk versus a multi-gigabyte download away.

Everything here is model-free (no weights loaded, no MLX arrays allocated) so it stays fast
enough to call on every app launch and testable without a GPU.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys

from .load import REGISTRY


def _sysctl(name: str) -> str | None:
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 — missing sysctl / non-macOS
        return None


def total_ram_gb() -> float | None:
    raw = _sysctl("hw.memsize")
    try:
        return int(raw) / (1024 ** 3) if raw else None
    except ValueError:
        return None


def _package_versions() -> dict:
    versions: dict[str, str | None] = {}
    for pkg, module in (("mlx", "mlx.core"), ("mlx_lm", "mlx_lm"), ("mlx_vlm", "mlx_vlm"),
                        ("transformers", "transformers")):
        try:
            mod = __import__(module, fromlist=["__version__"])
            versions[pkg] = getattr(mod, "__version__", None)
        except Exception:  # noqa: BLE001 — not installed / import error
            versions[pkg] = None
    return versions


def _metal_ok() -> tuple[bool, str | None]:
    try:
        import mlx.core as mx

        mx.zeros((2, 2))                        # exercise the Metal path, not just the import
        return True, mx.device_info().get("device_name")
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def environment() -> dict:
    """Everything about *this machine* that decides whether mlx-dspark can run well."""
    from . import __version__

    is_mac = sys.platform == "darwin"
    is_arm = platform.machine() == "arm64"
    metal_ok, device = _metal_ok()
    ram_gb = total_ram_gb()

    wired_mb = None
    raw = _sysctl("iogpu.wired_limit_mb")
    if raw:
        try:
            wired_mb = int(raw)
        except ValueError:
            pass

    return {
        "version": __version__,
        "platform": platform.system(),
        "machine": platform.machine(),
        "os_version": platform.mac_ver()[0] or None,
        "python": platform.python_version(),
        "apple_silicon": is_mac and is_arm,
        "metal_ok": metal_ok,
        "device": device if metal_ok else None,
        "metal_error": None if metal_ok else device,
        "ram_gb": round(ram_gb, 1) if ram_gb else None,
        "iogpu_wired_limit_mb": wired_mb,
        # The classic silent slowdown: with a big model and no raised limit, macOS pages
        # weights out mid-generation. Surface the exact command rather than a vague warning.
        "wired_limit_hint": (
            f"sudo sysctl iogpu.wired_limit_mb={int(ram_gb * 0.75 * 1024)}"
            if (ram_gb and ram_gb >= 16 and not wired_mb) else None
        ),
        "packages": _package_versions(),
    }


def _parse_ram_gb(text: str) -> float | None:
    """'~15 GB' -> 15.0. The registry stores human strings; the app needs a number."""
    m = re.search(r"([\d.]+)", text or "")
    return float(m.group(1)) if m else None


def _is_local(repo: str | None) -> bool:
    """Whether a repo's weights are already on disk — no download, no network.

    Checks the plain-dir cache that ``_resolve`` prefers and the HF hub cache layout directly;
    calling ``_resolve`` itself would *start a download*, which is exactly what this must not do.
    """
    if not repo:
        return False
    if repo.startswith("gguf:"):
        repo = repo[len("gguf:"):].rsplit("/", 1)[0]
    if os.path.isdir(repo):
        return True
    plain = os.path.join(os.path.expanduser("~/.cache/mlx_dspark/models"),
                         os.path.basename(repo.rstrip("/")))
    if os.path.isdir(plain):
        return True
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    return os.path.isdir(os.path.join(hub, "models--" + repo.replace("/", "--")))


_DETECT = object()      # distinguishes "measure this machine" from "RAM is genuinely unknown"


def model_inventory(ram_gb: float | None = _DETECT) -> list[dict]:  # type: ignore[assignment]
    """The registry, annotated with what this machine can actually run and already has.

    ``fits`` is deliberately conservative: the model has to sit in RAM alongside the OS and
    whatever else is open, so the check is against ~80% of physical memory. Telling someone a
    27B "fits" in the last gigabyte and watching them swap is worse than saying no.
    """
    if ram_gb is _DETECT:
        ram_gb = total_ram_gb()
    rows = []
    for entry in REGISTRY:
        need = _parse_ram_gb(entry.get("ram", ""))
        drafter = entry.get("dspark") or entry.get("dflash")
        rows.append({
            "id": entry["id"],
            "target": entry["target"],
            "dspark_drafter": entry.get("dspark"),
            "dflash_drafter": entry.get("dflash"),
            "ram": entry.get("ram"),
            "ram_gb": need,
            "fits": (None if (need is None or ram_gb is None) else need <= ram_gb * 0.8),
            "target_installed": _is_local(entry["target"]),
            "drafter_installed": _is_local(drafter),
            # Ready to run right now, with no download at all.
            "ready": _is_local(entry["target"]) and _is_local(drafter),
        })
    return rows


def doctor() -> dict:
    """Combined environment + inventory, with a single pass/fail the app can gate on."""
    env = environment()
    problems = []
    if not env["apple_silicon"]:
        problems.append("mlx-dspark needs an Apple Silicon Mac.")
    if not env["metal_ok"]:
        problems.append(f"MLX could not use the GPU: {env['metal_error']}")
    for pkg in ("mlx", "mlx_lm", "mlx_vlm"):
        if not env["packages"].get(pkg):
            problems.append(f"{pkg} is not importable.")
    if env["ram_gb"] and env["ram_gb"] < 15:
        problems.append(f"Only {env['ram_gb']:.0f} GB RAM — the smaller targets still work, "
                        "but 12B+ models will not.")

    return {"ok": not problems, "problems": problems,
            "environment": env, "models": model_inventory(env["ram_gb"])}
