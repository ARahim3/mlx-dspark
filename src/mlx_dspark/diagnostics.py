"""Machine + model inventory as structured data.

``mlx-dspark doctor`` already answered "will this work here?", but only as coloured text for a
human. The Mac app needs the same answers as JSON — to gate onboarding on Apple Silicon, to
filter the model picker by what actually fits this machine's RAM, and to show which models are
already on disk versus a multi-gigabyte download away.

Everything here is model-free (no weights loaded, no MLX arrays allocated) so it stays fast
enough to call on every app launch and testable without a GPU.
"""

from __future__ import annotations

import contextlib
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
        with contextlib.suppress(ValueError):
            wired_mb = int(raw)

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


def memory_info() -> dict:
    """MLX allocator state — what the loaded model actually holds resident right now.

    Reads the allocator, so it costs nothing and never touches the GPU. ``available: False``
    (rather than an exception) when mlx isn't importable, so ``/metrics`` stays serveable
    from any environment.
    """
    try:
        import mlx.core as mx

        return {
            "available": True,
            "active_bytes": int(mx.get_active_memory()),
            "peak_bytes": int(mx.get_peak_memory()),
            "cache_bytes": int(mx.get_cache_memory()),
        }
    except Exception:  # noqa: BLE001 — mlx missing / non-macOS
        return {"available": False}


_HUB_DIR = "~/.cache/huggingface/hub"
_PLAIN_DIR = "~/.cache/mlx_dspark/models"
_DRAFTERS_DIR = "~/.cache/mlx_dspark/drafters"


def _dir_size_bytes(path: str) -> int:
    """Real bytes under ``path``. ``lstat`` on purpose: HF hub snapshots are symlink farms
    into ``blobs/``, so following links would double-count every weight file."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                continue
    return total


def _human_size(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.1f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.0f} MB"
    return f"{n} B"


def _drafter_repos() -> set[str]:
    """Every drafter repo the registry names (gguf: scheme reduced to its repo)."""
    repos = set()
    for entry in REGISTRY:
        for key in ("dspark", "dflash"):
            repo = entry.get(key)
            if not repo:
                continue
            if repo.startswith("gguf:"):
                repo = repo[len("gguf:"):].rsplit("/", 1)[0]
            repos.add(repo)
    return repos


def installed_models(hub_dir: str | None = None, plain_dir: str | None = None,
                     lmstudio_roots: tuple[str, ...] | None = None) -> list[dict]:
    """Every model actually on this disk — the inventory the registry can't see.

    The registry answers "which pairs do we vouch for"; this answers "what has this user
    already downloaded", which is what a model picker should offer first. Sources are the
    two places the loaders read from: the HF hub cache and the plain-dir cache that
    ``_resolve`` prefers. Rows carry ``path`` and ``size_bytes`` so a client can offer
    reveal/delete and honest disk accounting without re-walking anything.
    """
    from .load import LMSTUDIO_ROOTS, _registry_entry, is_mlx_model_dir

    hub = os.path.expanduser(hub_dir or _HUB_DIR)
    plain = os.path.expanduser(plain_dir or _PLAIN_DIR)
    lmstudio = lmstudio_roots if lmstudio_roots is not None else LMSTUDIO_ROOTS
    drafters = _drafter_repos()
    drafter_basenames = {os.path.basename(r).lower() for r in drafters}

    rows: dict[str, dict] = {}

    def add(repo: str, path: str, source: str = "cache") -> None:
        if repo in rows:                       # plain dir wins (scanned first, _resolve's order)
            return
        entry = _registry_entry(repo)
        is_drafter = (repo in drafters
                      or os.path.basename(repo).lower() in drafter_basenames
                      or "dspark" in repo.lower() or "dflash" in repo.lower())
        size = _dir_size_bytes(path)
        rows[repo] = {
            "repo": repo,
            "path": path,
            "size_bytes": size,
            "size": _human_size(size),
            # A drafter checkpoint is not something you can load as a target; the app
            # shows these greyed/grouped rather than pretending they're chat models.
            "kind": "drafter" if is_drafter else "model",
            # Which registry pairing this repo would resolve into (quant-agnostic) —
            # i.e. whether `--mode auto` gets it a real drafter or falls back to lookup.
            "registry_id": entry["id"] if (entry and not is_drafter) else None,
            # "cache" (ours: plain dir or HF hub) vs "lmstudio" (another app's download,
            # readable but not ours to delete or bill against our disk total).
            "source": source,
        }

    if os.path.isdir(plain):
        for name in sorted(os.listdir(plain)):
            path = os.path.join(plain, name)
            if os.path.isdir(path):
                add(name, path)
    if os.path.isdir(hub):
        for name in sorted(os.listdir(hub)):
            if not name.startswith("models--"):
                continue
            path = os.path.join(hub, name)
            if not os.path.isdir(path):
                continue
            repo = name[len("models--"):].replace("--", "/")
            add(repo, path)

    # LM Studio's caches last (our copy wins a duplicate repo id): plain
    # <publisher>/<model> dirs, MLX-loadable ones only — a GGUF-only download would just
    # produce a row that can't load. `_resolve` reads the same roots, so every row listed
    # here is one `--model <publisher>/<model>` (or a picker click) away from serving.
    for root in lmstudio:
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            continue
        for org in sorted(os.listdir(root)):
            org_path = os.path.join(root, org)
            if org.startswith(".") or not os.path.isdir(org_path):
                continue
            for name in sorted(os.listdir(org_path)):
                path = os.path.join(org_path, name)
                if os.path.isdir(path) and is_mlx_model_dir(path):
                    add(f"{org}/{name}", path, source="lmstudio")

    return sorted(rows.values(), key=lambda r: (r["kind"] != "model", -r["size_bytes"]))


def disk_usage(rows: list[dict] | None = None, *, hub_dir: str | None = None,
               plain_dir: str | None = None, drafters_dir: str | None = None) -> dict:
    """Total bytes the model caches hold, including auto-converted drafters.

    Pass ``rows`` (a prior :func:`installed_models` result) to avoid re-walking the caches.
    """
    if rows is None:
        rows = installed_models(hub_dir, plain_dir)
    # LM Studio's copies are listed as loadable but they are not OUR disk usage — counting
    # them would tell a user "12 GB of models on disk" they can't reclaim from this app.
    total = sum(r["size_bytes"] for r in rows if r.get("source") != "lmstudio")
    conv = os.path.expanduser(drafters_dir or _DRAFTERS_DIR)
    if os.path.isdir(conv):
        total += _dir_size_bytes(conv)
    return {"total_bytes": total, "total": _human_size(total)}


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
    # Both hand-download naming conventions: bare basename and org-prefixed "<org>_<name>"
    # (see load._resolve — these two must agree, one answers "installed?" and the other "where").
    models = os.path.expanduser("~/.cache/mlx_dspark/models")
    stripped = repo.rstrip("/")
    for name in (os.path.basename(stripped), stripped.replace("/", "_")):
        if os.path.isdir(os.path.join(models, name)):
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
        best_mode = entry.get("mode") or ("dspark" if entry.get("dspark") else "dflash")
        drafter = entry.get(best_mode) or entry.get("dspark") or entry.get("dflash")
        rows.append({
            "id": entry["id"],
            "target": entry["target"],
            # the mode `--mode auto` (and the app) resolves for this row — its measured best
            "mode": best_mode,
            "dspark_drafter": entry.get("dspark"),
            "dflash_drafter": entry.get("dflash"),
            "ram": entry.get("ram"),
            "speedup": entry.get("speedup"),
            # The pair's measured-best hybrid-lookup setting (False where the stamped numbers
            # were taken with lookup off); the engine applies it as the shipped default.
            "lookup_drafts": bool(entry.get("lookup_drafts", True)),
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
