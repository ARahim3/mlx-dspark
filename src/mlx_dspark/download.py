"""Cancellable model downloads for the server's load path.

The problem this solves: a first-time ``/admin/load`` of a 15-30 GB model runs
``snapshot_download`` deep inside the loaders, blocking and uninterruptible — a client
(the Mac app) had no way to stop a mis-click short of killing the whole server, and no
progress beyond the log tail. The fix is to *pre-fetch* every hub repo the load will
need **before** the loaders run, in a child process that can be killed:

- :func:`ensure_local` — no-op for local paths, ``~/.cache/mlx_dspark/models`` dirs,
  non-repo strings (``gguf:`` schemes) and repos already complete in the HF cache;
  otherwise it runs ``snapshot_download`` in a subprocess and waits. The subsequent
  loader finds everything in cache and never touches the network, so the whole download
  phase — the part worth cancelling — lives here. The RAM-load phase that follows
  (~seconds to ~a minute) stays uninterruptible, which is fine.
- :func:`cancel_current` — kills the child (SIGTERM, then SIGKILL); ``ensure_local``
  raises :class:`DownloadCancelled`, which the server maps to a clear 4xx and the
  holder's existing failed-load semantics (server stays up, model-less). Partial files
  stay in the HF cache **by design** — a resumed download of a 15 GB model continues
  where it stopped — unless ``cleanup=True`` deletes the repo's cache folder.
- :func:`progress` — bytes on disk so far (scanning the repo's cache folder, incomplete
  blobs included) plus a best-effort total from the hub API, surfaced through
  ``/health`` while loading so a client can draw a real progress bar.

One download at a time (loads are serialized by the holder's swap lock anyway). The
child is registered with ``atexit`` so a dying server never leaves an orphan fetching
gigabytes — quitting the app quits the download.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import shutil
import subprocess
import sys
import threading
import time

_LOCK = threading.Lock()
_CURRENT: dict | None = None   # {"repo", "proc", "total", "cancelled", "cleanup"}


class DownloadCancelled(RuntimeError):
    """A client cancelled the in-flight model download."""


def _hub_cache_dir(repo: str) -> str:
    from huggingface_hub.constants import HF_HUB_CACHE

    return os.path.join(HF_HUB_CACHE, "models--" + repo.replace("/", "--"))


def _dir_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            with contextlib.suppress(OSError):
                p = os.path.join(root, f)
                if not os.path.islink(p):
                    total += os.path.getsize(p)
    return total


def _looks_like_repo(repo_or_path: str | None) -> bool:
    """Only plain ``org/name`` hub ids go through the pre-fetch; everything else (local
    paths, ``gguf:`` schemes, bare names) is the loaders' business and needs no network
    here."""
    if not repo_or_path or ":" in repo_or_path:
        return False
    if os.path.isdir(os.path.expanduser(repo_or_path)):
        return False
    if repo_or_path.count("/") != 1:
        return False
    # the plain-dir cache that _resolve prefers over the hub (see load.py) — match BOTH the
    # bare basename and the org-prefixed "<org>_<name>" form, exactly as _resolve/_is_local do.
    # Without the second form a hand-downloaded copy under the org-prefixed name (e.g.
    # DimInfer_Qwen3.8-27B-Dspark-v1) gets needlessly re-fetched here even though it's on disk.
    # These three checks must stay in lockstep.
    models = os.path.expanduser("~/.cache/mlx_dspark/models")
    stripped = repo_or_path.rstrip("/")
    for name in (os.path.basename(stripped), stripped.replace("/", "_")):
        if os.path.isdir(os.path.join(models, name)):
            return False
    return True


def _fetch_total(repo: str, entry: dict) -> None:
    """Best-effort size of the full snapshot, for the progress denominator."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, files_metadata=True)
        entry["total"] = sum(s.size or 0 for s in (info.siblings or []))
    except Exception:  # noqa: BLE001 — progress denominator is a bonus, never a gate
        pass


def ensure_local(repo_or_path: str | None) -> None:
    """Make sure ``repo_or_path`` is fully present in the HF cache, cancellably.

    Returns immediately when there is nothing to download. Raises
    :class:`DownloadCancelled` if :func:`cancel_current` killed the fetch, or
    ``RuntimeError`` if the child failed on its own (network, auth, bad repo) — the
    caller reports either exactly like any other failed load.
    """
    global _CURRENT
    if not _looks_like_repo(repo_or_path):
        return
    repo = str(repo_or_path)
    with contextlib.suppress(Exception):
        from huggingface_hub import snapshot_download

        snapshot_download(repo, local_files_only=True)   # complete already -> no child
        return

    env = dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS="1")
    # The child watches its parent and exits when orphaned: the server dies by SIGTERM
    # (the app's supervisor), which runs no atexit hooks — without this, killing the
    # server would leave the download running headless, which is the original bug report.
    child_src = (
        "import os, sys, threading, time\n"
        "def _watch():\n"
        "    while True:\n"
        "        if os.getppid() == 1: os._exit(1)\n"
        "        time.sleep(2)\n"
        "threading.Thread(target=_watch, daemon=True).start()\n"
        "from huggingface_hub import snapshot_download\n"
        "snapshot_download(sys.argv[1])\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", child_src, repo],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    entry = {"repo": repo, "proc": proc, "total": None, "cancelled": False,
             "cleanup": False, "started": time.time()}
    with _LOCK:
        _CURRENT = entry
    threading.Thread(target=_fetch_total, args=(repo, entry), daemon=True).start()
    try:
        _stderr = proc.communicate()[1]
    finally:
        with _LOCK:
            _CURRENT = None
    if entry["cancelled"]:
        if entry["cleanup"]:
            with contextlib.suppress(OSError):
                shutil.rmtree(_hub_cache_dir(repo))
        raise DownloadCancelled(
            f"download of {repo} cancelled"
            + (" (partial files removed)" if entry["cleanup"] else
               " (partial files kept — loading it again resumes)"))
    if proc.returncode != 0:
        tail = (_stderr or "").strip().splitlines()[-1:] or ["no error output"]
        raise RuntimeError(f"download of {repo} failed: {tail[0]}")


def progress() -> dict | None:
    """The in-flight download, or ``None``. ``bytes_done`` scans the repo's cache folder
    (incomplete blobs included); ``bytes_total`` is best-effort and may be null early."""
    with _LOCK:
        entry = _CURRENT
    if entry is None:
        return None
    return {"repo": entry["repo"],
            "bytes_done": _dir_bytes(_hub_cache_dir(entry["repo"])),
            "bytes_total": entry["total"]}


def cancel_current(cleanup: bool = False) -> dict:
    """Kill the in-flight download (idempotent). ``cleanup`` also deletes the repo's
    partial cache folder — otherwise partials stay and the next attempt resumes."""
    with _LOCK:
        entry = _CURRENT
        if entry is None:
            return {"cancelled": False, "reason": "no download in progress"}
        entry["cancelled"] = True
        entry["cleanup"] = bool(cleanup)
        proc = entry["proc"]
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    return {"cancelled": True, "repo": entry["repo"], "cleanup": bool(cleanup)}


@atexit.register
def _kill_orphan() -> None:
    # A dying server must take its download child with it — an orphan quietly fetching
    # gigabytes after the app quit is exactly the bug report this module exists to fix.
    with _LOCK:
        entry = _CURRENT
    if entry is not None and entry["proc"].poll() is None:
        with contextlib.suppress(OSError):
            entry["proc"].kill()
