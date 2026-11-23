"""Exit together with the process that started the server.

The Mac app runs ``mlx-dspark serve`` as its child and stops it when it quits. A crash or a
force-quit skips that stop, and the engine lived on as a child of launchd, holding the app's
port and, once a model was loaded, its memory, until something killed it; every relaunch then
started one more (issue #44). The app now names itself in ``MLX_DSPARK_PARENT_PID``, and
``serve`` ends when that process ends, however it ends.

kqueue's ``NOTE_EXIT`` reports the exit the moment it happens, with no polling, and tracks
the process itself rather than its pid, so a reused pid can't fool it. A process that is
already gone when the watch starts counts as exited. The engine then sends itself the SIGTERM
the app's own quit would have sent, so both ways of ending take the same path.
"""

from __future__ import annotations

import contextlib
import os
import select
import signal
import sys
import threading
import time

PARENT_PID_ENV = "MLX_DSPARK_PARENT_PID"


def _terminate() -> None:
    os.kill(os.getpid(), signal.SIGTERM)


def exit_with_parent(pid: int, on_exit=None) -> threading.Thread | None:
    """Watch process ``pid`` and call ``on_exit`` (default: SIGTERM to this process) once it
    has exited. Returns the watching thread, or None when ``pid`` was already gone (then
    ``on_exit`` has run before this returns)."""
    on_exit = on_exit or _terminate

    def gone() -> None:
        with contextlib.suppress(OSError, ValueError):     # stderr was that process's pipe
            print(f"[serve] the process that started this server (pid {pid}) has exited; "
                  "shutting down", file=sys.stderr, flush=True)
        on_exit()

    try:
        kq = select.kqueue()
        kq.control([select.kevent(pid, select.KQ_FILTER_PROC,
                                  select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                                  select.KQ_NOTE_EXIT)], 0)
    except ProcessLookupError:
        gone()
        return None
    except (AttributeError, OSError):
        kq = None                            # no kqueue here: poll instead

    def watch() -> None:
        if kq is not None:
            kq.control(None, 1)
        else:
            while True:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                except PermissionError:      # alive, owned by someone else
                    pass
                time.sleep(1.0)
        gone()

    thread = threading.Thread(target=watch, name="parent-watch", daemon=True)
    thread.start()
    return thread


def watch_parent_from_env(environ=None) -> threading.Thread | None:
    """Start :func:`exit_with_parent` for the pid in ``MLX_DSPARK_PARENT_PID``, if one is set.
    The variable is removed, so a process this server starts doesn't inherit the watch."""
    environ = os.environ if environ is None else environ
    raw = environ.pop(PARENT_PID_ENV, None)
    if raw is None:
        return None
    try:
        pid = int(raw)
    except ValueError:
        pid = 0
    if pid <= 1 or pid == os.getpid():
        print(f"[serve] ignoring {PARENT_PID_ENV}={raw!r}: not another process's pid",
              file=sys.stderr, flush=True)
        return None
    return exit_with_parent(pid)
