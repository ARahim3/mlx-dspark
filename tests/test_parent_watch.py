"""parent_watch: `serve` ends when the process that started it ends (issue #44: a crashed Mac
app left its engine running as an orphan, holding the port and the model). Model-free."""
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import pytest

from mlx_dspark.parent_watch import PARENT_PID_ENV, exit_with_parent, watch_parent_from_env


def _short_lived(seconds: float) -> subprocess.Popen:
    return subprocess.Popen(["sleep", str(seconds)])


def test_the_watch_fires_when_the_parent_exits_and_not_before():
    parent = _short_lived(0.4)
    fired = threading.Event()
    thread = exit_with_parent(parent.pid, on_exit=fired.set)
    assert thread is not None and not fired.wait(0.1)      # still running: nothing yet
    parent.wait()
    assert fired.wait(5.0)
    thread.join(1.0)
    assert not thread.is_alive()


def test_a_parent_that_is_already_gone_counts_as_exited():
    parent = _short_lived(0)
    parent.wait()
    fired = []
    assert exit_with_parent(parent.pid, on_exit=lambda: fired.append(1)) is None
    assert fired == [1]


@pytest.mark.parametrize("raw", ["abc", "", "0", "1", "-5"])
def test_a_value_that_is_not_another_process_is_ignored(raw, capsys):
    env = {PARENT_PID_ENV: raw}
    assert watch_parent_from_env(env) is None
    assert PARENT_PID_ENV not in env
    assert "ignoring" in capsys.readouterr().err


def test_no_variable_no_watch_and_the_variable_is_not_inherited(monkeypatch):
    from mlx_dspark import parent_watch

    fired = threading.Event()
    monkeypatch.setattr(parent_watch, "_terminate", fired.set)    # not this test process
    assert watch_parent_from_env({}) is None
    assert watch_parent_from_env({PARENT_PID_ENV: str(os.getpid())}) is None   # itself
    parent = _short_lived(30)
    try:
        env = {PARENT_PID_ENV: str(parent.pid), "OTHER": "x"}
        thread = watch_parent_from_env(env)
        assert thread is not None
        assert env == {"OTHER": "x"}                # popped: a child of serve won't watch
    finally:
        parent.kill()
        parent.wait()
    assert fired.wait(5.0)                          # the default action, once the parent died
    thread.join(1.0)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_serve_exits_when_the_process_that_started_it_dies():
    """The wiring the app relies on, end to end: a real `serve --no-model` given the pid of a
    stand-in parent stops by itself once that parent is killed, the way a crashed app dies."""
    parent = _short_lived(120)
    port = _free_port()
    env = dict(os.environ, **{PARENT_PID_ENV: str(parent.pid)})
    serve = subprocess.Popen([sys.executable, "-m", "mlx_dspark", "serve", "--no-model",
                              "--no-warmup", "--port", str(port)],
                             env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
                    if json.load(r).get("status"):
                        break
            except OSError:
                time.sleep(0.2)
            assert serve.poll() is None, serve.stderr.read().decode()[-2000:]
        else:
            pytest.fail("serve never answered /health")
        assert serve.poll() is None                 # the parent is alive: serve keeps running
        parent.kill()                               # SIGKILL, like a crash: no cleanup runs
        parent.wait()
        assert serve.wait(10) == -signal.SIGTERM
        assert b"has exited; shutting down" in serve.stderr.read()
    finally:
        for proc in (serve, parent):
            if proc.poll() is None:
                proc.kill()
                proc.wait()
