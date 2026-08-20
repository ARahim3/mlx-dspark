"""Protocol tests for the OpenAI-compatible server.

These use a *mock* engine (no model weights), so they run in CI in milliseconds and
verify the HTTP surface: routing, JSON shapes, SSE framing, stop handling wiring, auth,
and error paths. End-to-end correctness with a real drafter is exercised separately.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from mlx_dspark import server as S
from mlx_dspark.generate import GenResult


class _FakeTok:
    def encode(self, text):
        return [ord(c) for c in text][:64]

    def decode(self, ids):
        return "".join(chr(int(i) % 0x110000) for i in ids)


class _FakeEngine:
    mode = "dspark"
    model_id = "FakeModel"
    created = 123
    target_repo = "org/Target"
    drafter_repo = "org/Drafter"
    template_defaults = {}
    sampling_defaults = {}
    default_max_tokens = 2048
    max_tokens_cap = 32768
    default_think_budget = None   # mirrors Engine (server default reasoning budget)
    think_budget_message = None
    cap_controller = None
    is_muse = False           # mirrors Engine.is_muse (muse_glimmer channel parsing off)

    def __init__(self):
        self.tokenizer = _FakeTok()
        self.calls = []
        self.response_text = "Hello world from mlx dspark"
        # instance-level: /admin/config mutates it, and a class-level dict would leak
        # one test's mutation into the next
        self.template_defaults = {}

    def generate(self, prompt_ids, *, max_tokens, temperature, top_p=1.0, top_k=0,
                 presence_penalty=0.0, frequency_penalty=0.0, logprobs=None,
                 stop, seed, think_budget=None, budget_message=None, budget_explicit=False,
                 on_text=None):
        self.calls.append({"prompt_ids": prompt_ids, "max_tokens": max_tokens,
                               "temperature": temperature, "top_p": top_p, "top_k": top_k,
                               "presence_penalty": presence_penalty,
                               "frequency_penalty": frequency_penalty, "logprobs": logprobs,
                               "stop": stop, "seed": seed, "think_budget": think_budget,
                               "budget_message": budget_message,
                               "budget_explicit": budget_explicit})
        text = self.response_text
        if on_text:
            for w in text.split(" "):
                on_text(w + " ")
        lp = None
        if logprobs is not None:
            lp = [{"token_id": t, "logprob": -0.5,
                   "top": [(t, -0.5)] if logprobs else []} for t in [1, 2, 3, 4, 5]]
        return GenResult(text=text, token_ids=[1, 2, 3, 4, 5], num_tokens=5, num_rounds=2,
                         accept_lengths=[2, 3], target_forwards=2, seconds=0.1,
                         finish_reason="stop", logprobs=lp)

    def spec_info(self, res):
        return {"mode": self.mode, "accept_len": res.mean_accept_len,
                "tokens_per_sec": res.tokens_per_sec, "target_forwards": res.target_forwards}

    def metrics(self):
        return {"model": self.model_id, "mode": self.mode, "requests": len(self.calls)}


@pytest.fixture
def server():
    eng = _FakeEngine()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(eng, api_key=None))
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield eng, f"http://127.0.0.1:{port}"
    httpd.shutdown()


def _get(base, path):
    return json.loads(urllib.request.urlopen(base + path).read())


def _post(base, path, obj, stream=False, headers=None):
    h = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(base + path, data=json.dumps(obj).encode(), headers=h, method="POST")
    r = urllib.request.urlopen(req)
    return r.read().decode() if stream else json.loads(r.read())


def test_health(server):
    _, base = server
    h = _get(base, "/health")
    assert h["status"] == "ok" and h["model"] == "FakeModel" and h["mode"] == "dspark"
    # The configured draft cap, so a client can show the knob's real state ("auto" or "N").
    assert h["max_draft"] == "auto"


def test_models(server):
    _, base = server
    m = _get(base, "/v1/models")
    assert m["object"] == "list"
    assert m["data"][0]["id"] == "FakeModel"
    assert m["data"][0]["x_mlx_dspark"]["mode"] == "dspark"


def test_chat_non_stream(server):
    _eng, base = server
    c = _post(base, "/v1/chat/completions",
              {"model": "x", "messages": [{"role": "user", "content": "hi"}]})
    assert c["object"] == "chat.completion"
    assert c["choices"][0]["message"]["content"] == "Hello world from mlx dspark"
    assert c["choices"][0]["finish_reason"] == "stop"
    assert c["usage"]["completion_tokens"] == 5 and c["usage"]["prompt_tokens"] > 0
    assert c["usage"]["total_tokens"] == c["usage"]["prompt_tokens"] + 5
    assert "x_mlx_dspark" in c


def test_chat_stream_sse(server):
    _, base = server
    sse = _post(base, "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "hi"}], "stream": True,
                 "stream_options": {"include_usage": True}}, stream=True)
    lines = [l for l in sse.split("\n\n") if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(l[6:]) for l in lines if l != "data: [DONE]"]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert all(ch["object"] == "chat.completion.chunk" for ch in chunks)
    content = "".join(ch["choices"][0]["delta"].get("content", "") for ch in chunks)
    assert content == "Hello world from mlx dspark "
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"]["completion_tokens"] == 5


def test_stop_forwarded(server):
    eng, base = server
    _post(base, "/v1/chat/completions",
          {"messages": [{"role": "user", "content": "hi"}], "stop": "END", "temperature": 0.7})
    assert eng.calls[-1]["stop"] == ["END"]
    assert eng.calls[-1]["temperature"] == 0.7


def test_completions_legacy(server):
    _, base = server
    lc = _post(base, "/v1/completions", {"prompt": "once upon"})
    assert lc["object"] == "text_completion"
    assert lc["choices"][0]["text"]
    assert lc["choices"][0]["finish_reason"] == "stop"


_TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {}}}]


def test_tool_calls_non_stream(server):
    eng, base = server
    eng.response_text = 'ok<tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call>'
    c = _post(base, "/v1/chat/completions",
              {"messages": [{"role": "user", "content": "call f"}], "tools": _TOOLS})
    msg = c["choices"][0]["message"]
    assert c["choices"][0]["finish_reason"] == "tool_calls"
    assert msg["tool_calls"][0]["function"]["name"] == "f"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"x": 1}


def test_tool_calls_stream(server):
    eng, base = server
    eng.response_text = '<tool_call>{"name": "f", "arguments": {}}</tool_call>'
    sse = _post(base, "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "call f"}], "tools": _TOOLS,
                 "stream": True}, stream=True)
    chunks = [json.loads(l[6:]) for l in sse.split("\n\n")
              if l.startswith("data: ") and l != "data: [DONE]"]
    tc = [c for c in chunks if c["choices"][0]["delta"].get("tool_calls")]
    assert tc and tc[0]["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_no_tools_means_plain_text(server):
    eng, base = server
    eng.response_text = '<tool_call>{"name": "f", "arguments": {}}</tool_call>'
    # without `tools` in the request we do NOT parse tool calls — return raw text
    c = _post(base, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
    assert c["choices"][0]["message"].get("tool_calls") is None
    assert "<tool_call>" in c["choices"][0]["message"]["content"]


def test_sampling_defaults_fill_absent_fields_only(server):
    eng, base = server
    eng.sampling_defaults = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
    # request omits sampling params -> the model's generation_config recommendations apply
    _post(base, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
    call = eng.calls[-1]
    assert call["temperature"] == 0.6 and call["top_p"] == 0.95 and call["top_k"] == 20
    # explicit values (including an explicit 0.0) always win over the defaults
    _post(base, "/v1/chat/completions",
          {"messages": [{"role": "user", "content": "hi"}], "temperature": 0.0, "top_p": 1.0})
    call = eng.calls[-1]
    assert call["temperature"] == 0.0 and call["top_p"] == 1.0 and call["top_k"] == 20


def test_max_tokens_default_and_cap(server):
    eng, base = server
    eng.default_max_tokens = 777
    eng.max_tokens_cap = 1000
    _post(base, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
    assert eng.calls[-1]["max_tokens"] == 777          # absent -> engine default
    _post(base, "/v1/chat/completions",
          {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5000})
    assert eng.calls[-1]["max_tokens"] == 1000         # above the configurable ceiling -> clamped


def test_metrics(server):
    _, base = server
    _post(base, "/v1/completions", {"prompt": "x"})
    mt = _get(base, "/metrics")
    assert mt["model"] == "FakeModel" and mt["requests"] >= 1


def test_metrics_reports_allocator_memory(server):
    """The app's memory gauge reads this — it must exist for any engine, fakes included."""
    _, base = server
    memory = _get(base, "/metrics")["memory"]
    assert "available" in memory
    if memory["available"]:
        assert memory["active_bytes"] >= 0 and memory["peak_bytes"] >= 0


def test_events_stream_ends_cleanly_across_a_model_swap():
    """A hot swap replaces the engine (and its round log) under a live /events stream.

    The stream must END — so the client reconnects to the new engine's log — never
    traceback through the holder's no-engine guard (that stack trace lands in the app's
    loading screen and reads as a crash)."""
    from mlx_dspark.server import EngineHolder
    from mlx_dspark.telemetry import RoundLog

    class Eng(_FakeEngine):
        def __init__(self):
            super().__init__()
            self.rounds = RoundLog()

        def close(self):
            pass

    holder = EngineHolder(Eng(), load_kwargs={})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(holder, api_key=None))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        stream = urllib.request.urlopen(base + "/events", timeout=5)
        # Swap the engine out from under the stream (a real swap loads weights; identity
        # of `rounds` changing is all the stream watches).
        holder._engine = Eng()
        start = time.time()
        stream.read()                              # blocks until the server closes the stream
        assert time.time() - start < 10
        # The server is still healthy and serving the new engine.
        assert _get(base, "/health")["status"] == "ok"
    finally:
        httpd.shutdown()


def test_admin_models_lists_registry_installed_and_disk(server):
    _, base = server
    payload = _get(base, "/admin/models")
    assert payload["loaded"] == "org/Target"
    assert isinstance(payload["models"], list) and payload["models"]
    assert isinstance(payload["installed"], list)      # may be empty on a clean machine
    for row in payload["installed"]:
        assert {"repo", "path", "size_bytes", "size", "kind"} <= set(row)
    assert payload["disk"]["total_bytes"] >= 0


def test_unknown_route_404(server):
    _, base = server
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base, "/nope")
    assert e.value.code == 404


def test_bad_chat_body_400(server):
    _, base = server
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base, "/v1/chat/completions", {"messages": []})
    assert e.value.code == 400


def test_auth_required():
    eng = _FakeEngine()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(eng, api_key="secret"))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(base, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
        assert e.value.code == 401
        # with the right key it works
        c = _post(base, "/v1/chat/completions",
                  {"messages": [{"role": "user", "content": "hi"}]},
                  headers={"Authorization": "Bearer secret"})
        assert c["object"] == "chat.completion"
    finally:
        httpd.shutdown()


def test_logprobs_chat_response_shape(server):
    eng, base = server
    c = _post(base, "/v1/chat/completions",
              {"messages": [{"role": "user", "content": "hi"}],
               "logprobs": True, "top_logprobs": 3})
    lp = c["choices"][0]["logprobs"]
    assert "content" in lp and len(lp["content"]) == 5
    first = lp["content"][0]
    assert set(first) >= {"token", "logprob", "bytes", "top_logprobs"}
    assert len(first["top_logprobs"]) == 1              # fake returns one top per token
    assert eng.calls[-1]["logprobs"] == 3              # top_logprobs threaded through


def test_logprobs_absent_by_default(server):
    eng, base = server
    c = _post(base, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
    assert "logprobs" not in c["choices"][0]
    assert eng.calls[-1]["logprobs"] is None


def test_completions_logprobs_shape(server):
    eng, base = server
    c = _post(base, "/v1/completions", {"prompt": "hi", "logprobs": 2})
    lp = c["choices"][0]["logprobs"]
    assert "tokens" in lp and "token_logprobs" in lp and "top_logprobs" in lp
    assert eng.calls[-1]["logprobs"] == 2


def test_penalties_passthrough(server):
    eng, base = server
    _post(base, "/v1/chat/completions",
          {"messages": [{"role": "user", "content": "hi"}],
           "presence_penalty": 1.5, "frequency_penalty": 0.7})
    assert eng.calls[-1]["presence_penalty"] == 1.5
    assert eng.calls[-1]["frequency_penalty"] == 0.7


def test_n_greedy_replicates_one_generation(server):
    eng, base = server
    r = _post(base, "/v1/chat/completions",
              {"messages": [{"role": "user", "content": "hi"}], "n": 3})
    assert [c["index"] for c in r["choices"]] == [0, 1, 2]
    assert len({c["message"]["content"] for c in r["choices"]}) == 1
    assert len(eng.calls) == 1                      # greedy: one generation serves all n
    assert r["usage"]["completion_tokens"] == 5     # counts actual generated tokens


def test_n_sampled_generates_n(server):
    eng, base = server
    r = _post(base, "/v1/chat/completions",
              {"messages": [{"role": "user", "content": "hi"}], "n": 3, "temperature": 0.8})
    assert len(r["choices"]) == 3
    assert len(eng.calls) == 3                      # independent samples
    assert r["usage"]["completion_tokens"] == 15


def test_n_with_stream_is_rejected(server):
    _, base = server
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base, "/v1/chat/completions",
              {"messages": [{"role": "user", "content": "hi"}], "n": 2, "stream": True})
    assert e.value.code == 400


def test_generation_error_reports_type_and_logs_traceback(server, capfd):
    """A mid-generation failure must return a 500 that NAMES the exception type and must
    leave the traceback in the server log. Issue #5 reported an intermittent
    'generation failed: list index out of range' with no way to localize it: the handler
    caught the exception, formatted str(e) only, and dropped the traceback on the floor,
    so neither the user nor a maintainer could tell which list, in which module, blew up.
    """
    eng, base = server

    def boom(*a, **k):
        raise IndexError("list index out of range")

    eng.generate = boom
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
    assert e.value.code == 500
    err = json.loads(e.value.read())["error"]
    assert err["type"] == "server_error"
    assert "IndexError" in err["message"]           # bare str(e) hid which exception it was
    assert "list index out of range" in err["message"]
    assert "Traceback" in capfd.readouterr().err    # the part that makes it diagnosable


def test_race_thinking_param_validated_and_echoed(server):
    """/admin/race takes an optional boolean `thinking` (the Lab's toggle): a non-boolean is
    a 400 with the reason, a boolean rides into the chat-template kwargs and is echoed in the
    SSE start event so a client can display the race's actual configuration."""
    eng, base = server
    eng.race_arms_available = lambda: ["dspark", "baseline"]
    eng.race = lambda prompt_ids, arms, max_tokens, on_event: None

    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base, "/admin/race",
              {"prompt": "hi", "arms": ["dspark", "baseline"], "thinking": "yes"})
    assert e.value.code == 400
    assert "thinking" in json.loads(e.value.read())["error"]["message"]

    out = _post(base, "/admin/race",
                {"prompt": "hi", "arms": ["dspark", "baseline"], "thinking": False},
                stream=True)
    start = next(line for line in out.splitlines() if line.startswith("data:"))
    assert json.loads(start[5:])["thinking"] is False

    # omitted -> server default; the start event then carries no thinking key at all
    out = _post(base, "/admin/race", {"prompt": "hi", "arms": ["dspark", "baseline"]},
                stream=True)
    start = next(line for line in out.splitlines() if line.startswith("data:"))
    assert "thinking" not in json.loads(start[5:])


# --- checkpoint-mode boundary probes (stable prompt boundary per chat template) ----------


class _ThinkTok:
    """Qwen3.6-shaped template double: the generation prompt appends a `<think>` opener
    (tokens 91, 92) that a completed turn re-renders WITHOUT — the stable boundary sits
    2 tokens below the prompt boundary."""

    chat_template = "fake"

    def apply_chat_template(self, messages, add_generation_prompt=True, **kw):
        role = {"system": 1, "user": 2, "assistant": 3}
        out = []
        for m in messages:
            out += [10, role.get(m.get("role"), 4)]
            out += [ord(c) % 40 + 100 for c in str(m.get("content", ""))]
            out += [11]
        if add_generation_prompt:
            out += [10, 3, 91, 92]
        return out

    def encode(self, text):
        return [ord(c) % 40 + 100 for c in text]

    def decode(self, ids):
        return "".join(chr(int(i)) for i in ids)


class _NullTarget:
    def make_cache(self):
        return []


def _probe_engine(tok):
    return S.Engine(_NullTarget(), tok, None, mode="baseline", model_id="m",
                    target_repo="t", drafter_repo=None, max_draft_tokens=None,
                    prefix_cache=False)


def test_unstable_suffix_probed_from_the_template():
    eng = _probe_engine(_ThinkTok())
    prompt = eng.tokenizer.apply_chat_template([{"role": "user", "content": "hello"}])
    assert prompt[-2:] == [91, 92]
    assert eng._unstable_suffix(prompt) == 2        # the <think> opener doesn't survive
    # a prompt that doesn't end in this template's generation suffix: conservative 1
    assert eng._unstable_suffix(prompt[:-2] + [55, 56]) == 1


def test_unstable_suffix_defaults_to_one_without_a_template():
    eng = _probe_engine(_FakeTok())                 # no chat_template attribute
    assert eng._boundary_probes() == []
    assert eng._unstable_suffix([1, 2, 3, 4]) == 1


# --- reasoning effort (Qwen3.8-class chat-template kwarg) --------------------------------


class _EffortTok(_FakeTok):
    """Kwarg-capturing template double for the reasoning-effort passthrough tests."""

    chat_template = "{% if reasoning_effort %}hint{% endif %}"

    def __init__(self):
        self.template_kwargs = []

    def apply_chat_template(self, messages, add_generation_prompt=True, **kw):
        self.template_kwargs.append(kw)
        return [1, 2, 3]


def test_reasoning_effort_normalization():
    assert S._reasoning_effort("XHigh") == "xhigh"
    for bad in ("extreme", 3, None, ""):
        with pytest.raises(ValueError):
            S._reasoning_effort(bad)


def test_reasoning_effort_reaches_the_template(server):
    eng, base = server
    tok = _EffortTok()
    eng.tokenizer = tok
    _post(base, "/v1/chat/completions",
          {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "LOW"})
    assert tok.template_kwargs[-1]["reasoning_effort"] == "low"     # normalized
    # The top-level field wins over chat_template_kwargs, like enable_thinking's shortcut.
    _post(base, "/v1/chat/completions",
          {"messages": [{"role": "user", "content": "hi"}],
           "chat_template_kwargs": {"reasoning_effort": "medium"},
           "reasoning_effort": "xhigh"})
    assert tok.template_kwargs[-1]["reasoning_effort"] == "xhigh"
    # Omitted -> not injected at all; the model's own default applies.
    _post(base, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
    assert "reasoning_effort" not in tok.template_kwargs[-1]


def test_reasoning_effort_invalid_is_400(server):
    """A typo is a clear boundary 400, not a Jinja raise_exception buried in a template error."""
    eng, base = server
    eng.tokenizer = _EffortTok()
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base, "/v1/chat/completions",
              {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "extreme"})
    assert e.value.code == 400
    assert "reasoning_effort" in json.loads(e.value.read())["error"]["message"]


def test_health_reports_reasoning_effort_support(server):
    _eng, base = server
    h = _get(base, "/health")
    assert h["supports_reasoning_effort"] is False
    assert h["reasoning_effort"] is None


def test_supports_reasoning_effort_tracks_the_template():
    # _ThinkTok's template string doesn't mention the kwarg; a template that does flips it.
    assert _probe_engine(_ThinkTok()).supports_reasoning_effort is False
    tok = _ThinkTok()
    tok.chat_template = "{% if reasoning_effort == 'low' %}brief{% endif %}"
    assert _probe_engine(tok).supports_reasoning_effort is True


def test_nonstream_chat_truncated_prefilled_thinking_stays_reasoning(server):
    # A template that PREFILLS <think> (the prompt tail ends with the opener) plus a
    # generation truncated by max_tokens leaves neither opener nor closer in the text.
    # The non-streaming split must still classify it as reasoning, not leak it as content.
    eng, base = server
    eng.response_text = "half a thought, cut by max_tokens"
    body = {"messages": [{"role": "user", "content": "hi\n<think>"}]}
    msg = _post(base, "/v1/chat/completions", body)["choices"][0]["message"]
    assert msg["reasoning_content"] == "half a thought, cut by max_tokens"
    assert not msg["content"]
    # when the closer DID arrive, the split is what it always was
    eng.response_text = "done thinking</think>\nThe answer."
    msg = _post(base, "/v1/chat/completions", body)["choices"][0]["message"]
    assert msg["reasoning_content"] == "done thinking"
    assert msg["content"] == "The answer."
    # and a prompt that does NOT end in an opener keeps the old all-answer reading
    eng.response_text = "half a thought, cut by max_tokens"
    msg = _post(base, "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "hi"}]})["choices"][0]["message"]
    assert msg["content"] == "half a thought, cut by max_tokens"
    assert "reasoning_content" not in msg


def test_think_budget_normalization():
    assert S._think_budget(512) == 512
    assert S._think_budget(0) is None                    # explicit 0 = disabled
    for bad in (True, False, "lots", 1.5, -1, None):
        with pytest.raises(ValueError):
            S._think_budget(bad)


def test_reasoning_budget_field_reaches_generate(server):
    eng, base = server
    _post(base, "/v1/chat/completions",
          {"messages": [{"role": "user", "content": "hi"}], "reasoning_budget": 512,
           "reasoning_budget_message": "Wrap it up."})
    assert eng.calls[-1]["think_budget"] == 512
    assert eng.calls[-1]["budget_message"] == "Wrap it up."
    assert eng.calls[-1]["budget_explicit"] is True
    # "" is preserved (close the block with no message), not swapped for the default
    eng.think_budget_message = "Server default."
    _post(base, "/v1/chat/completions",
          {"messages": [{"role": "user", "content": "hi"}], "reasoning_budget": 512,
           "reasoning_budget_message": ""})
    assert eng.calls[-1]["budget_message"] == ""
    # streaming goes through the same params dict
    _post(base, "/v1/chat/completions",
          {"messages": [{"role": "user", "content": "hi"}], "reasoning_budget": 256,
           "stream": True}, stream=True)
    assert eng.calls[-1]["think_budget"] == 256


def test_reasoning_budget_engine_default_and_zero_disables(server):
    eng, base = server
    body = {"messages": [{"role": "user", "content": "hi"}]}
    _post(base, "/v1/chat/completions", body)
    assert eng.calls[-1]["think_budget"] is None         # no default configured
    eng.default_think_budget = 8192
    eng.think_budget_message = "Answer."
    _post(base, "/v1/chat/completions", body)
    assert eng.calls[-1]["think_budget"] == 8192         # server default applies
    assert eng.calls[-1]["budget_message"] == "Answer."
    assert eng.calls[-1]["budget_explicit"] is False     # ambient: batching keeps working
    _post(base, "/v1/chat/completions", {**body, "reasoning_budget": 0})
    assert eng.calls[-1]["think_budget"] is None         # explicit 0 overrides the default


def test_reasoning_budget_invalid_is_400(server):
    _eng, base = server
    for bad in (True, "lots", 1.5, -1):
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(base, "/v1/chat/completions",
                  {"messages": [{"role": "user", "content": "hi"}], "reasoning_budget": bad})
        assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base, "/v1/chat/completions",
              {"messages": [{"role": "user", "content": "hi"}], "reasoning_budget": 5,
               "reasoning_budget_message": 7})
    assert e.value.code == 400


def test_reasoning_budget_not_passed_on_muse(server):
    eng, base = server
    eng.is_muse = True
    eng.default_think_budget = 8192
    _post(base, "/v1/chat/completions",
          {"messages": [{"role": "user", "content": "hi"}], "reasoning_budget": 512})
    assert eng.calls[-1]["think_budget"] is None


def test_health_reports_reasoning_budget(server):
    eng, base = server
    assert _get(base, "/health")["reasoning_budget"] is None
    eng.default_think_budget = 4096
    assert _get(base, "/health")["reasoning_budget"] == 4096


def test_health_supports_reasoning_budget_gates_on_muse(server):
    # A per-request budget control must gate on this, not admin_config: the muse request
    # path deliberately skips think_budget, so the control would do nothing there.
    eng, base = server
    assert _get(base, "/health")["supports_reasoning_budget"] is True
    eng.is_muse = True
    assert _get(base, "/health")["supports_reasoning_budget"] is False


def test_health_reports_reasoning_budget_with_no_model_loaded():
    holder = S.EngineHolder(None, {"default_think_budget": 4096}, max_batch=1)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(holder, api_key=None))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        h = _get(f"http://127.0.0.1:{httpd.server_address[1]}", "/health")
        assert h["status"] == "no_model" and h["reasoning_budget"] == 4096
        # no model = no answer about what the model supports; the key stays absent
        assert "supports_reasoning_budget" not in h
    finally:
        httpd.shutdown()


def test_engine_load_rejects_bad_budget_knobs():
    with pytest.raises(ValueError):
        S.Engine.load(model="whatever", default_think_budget=-5)
    with pytest.raises(ValueError):
        S.Engine.load(model="whatever", think_budget_message=7)


def test_admin_config_mutates_live_engine_no_reload(server):
    eng, base = server
    r = _post(base, "/admin/config", {"enable_thinking": False, "reasoning_budget": 512,
                                      "reasoning_budget_message": "Hurry."})
    assert r == {"enable_thinking": False, "reasoning_budget": 512,
                 "reasoning_budget_message": "Hurry."}
    assert eng.default_think_budget == 512
    assert eng.think_budget_message == "Hurry."
    assert eng.template_defaults["enable_thinking"] is False
    # effective immediately: the very next request runs with the new default
    _post(base, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
    assert eng.calls[-1]["think_budget"] == 512
    assert eng.calls[-1]["budget_message"] == "Hurry."


def test_admin_config_explicit_true_null_zero_and_empty(server):
    eng, base = server
    # explicit ON is stored as True — it must WIN over a template whose default is off,
    # not merely clear the override
    _post(base, "/admin/config", {"enable_thinking": True})
    assert eng.template_defaults["enable_thinking"] is True
    _post(base, "/admin/config", {"enable_thinking": None})
    assert "enable_thinking" not in eng.template_defaults      # null clears the override
    eng.default_think_budget = 8192
    _post(base, "/admin/config", {"reasoning_budget": 0})
    assert eng.default_think_budget is None                     # 0 disables
    _post(base, "/admin/config", {"reasoning_budget_message": ""})
    assert eng.think_budget_message == ""                       # explicit empty preserved
    _post(base, "/admin/config", {"reasoning_budget_message": None})
    assert eng.think_budget_message is None                     # null = engine default


def test_admin_config_validates_whole_patch_before_mutating(server):
    eng, base = server
    # a valid budget beside an invalid message must change NOTHING
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base, "/admin/config", {"reasoning_budget": 512,
                                      "reasoning_budget_message": 7})
    assert e.value.code == 400
    assert eng.default_think_budget is None
    for body in ({"reasoning_budget": -1}, {"reasoning_budget": True},
                 {"reasoning_budget": "x"}, {"enable_thinking": 3}):
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(base, "/admin/config", body)
        assert e.value.code == 400


def test_admin_config_rejects_non_object_bodies(server):
    eng, base = server
    # do_POST decodes any JSON value; each of these must be a clean 400 (not a 500 from
    # the `in` membership checks) and must mutate nothing — including [] , which the `in`
    # operator would otherwise wave through as an accidental read-back.
    for body in (None, [], ["reasoning_budget"], "reasoning_budget", 5):
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(base, "/admin/config", body)
        assert e.value.code == 400
    assert eng.default_think_budget is None


def test_admin_config_empty_body_reads_back(server):
    eng, base = server
    eng.default_think_budget = 4096
    assert _post(base, "/admin/config", {}) == {
        "enable_thinking": None, "reasoning_budget": 4096,
        "reasoning_budget_message": None}


def _serve_handler(target):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(target, api_key=None))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def test_admin_config_reaches_inner_engine_through_holder_and_batch_wrapper():
    inner = _FakeEngine()
    batch = object.__new__(S.BatchEngine)   # real class (isinstance + __getattr__), no threads
    batch.engine = inner
    holder = S.EngineHolder(batch, {"default_think_budget": 8192})
    httpd, base = _serve_handler(holder)
    try:
        _post(base, "/admin/config", {"reasoning_budget": 256, "enable_thinking": True})
        # the INNERMOST engine mutated — and no shadowing attrs on the delegating wrappers,
        # which would permanently break their __getattr__ forwarding
        assert inner.default_think_budget == 256
        assert inner.template_defaults["enable_thinking"] is True
        assert "default_think_budget" not in batch.__dict__
        assert "default_think_budget" not in holder.__dict__
        # and the stored load kwargs carry the values into any later swap
        assert holder._load_kwargs["default_think_budget"] == 256
        assert holder._load_kwargs["enable_thinking"] is True
    finally:
        httpd.shutdown()


def test_admin_config_no_model_updates_kwargs():
    holder = S.EngineHolder(None, {"default_think_budget": 8192})
    httpd, base = _serve_handler(holder)
    try:
        r = _post(base, "/admin/config", {"reasoning_budget": 512,
                                          "reasoning_budget_message": "Go.",
                                          "enable_thinking": False})
        assert r == {"enable_thinking": False, "reasoning_budget": 512,
                     "reasoning_budget_message": "Go."}
        assert holder._load_kwargs["default_think_budget"] == 512
        assert holder._load_kwargs["think_budget_message"] == "Go."
        assert holder._load_kwargs["enable_thinking"] is False
        # /health (no-model branch) reflects the stored values + the capability flag
        h = _get(base, "/health")
        assert h["status"] == "no_model" and h["admin_config"] is True
        assert h["reasoning_budget"] == 512
        assert h["reasoning_budget_message"] == "Go."
        assert h["enable_thinking"] is False
    finally:
        httpd.shutdown()


def test_health_reports_thinking_and_message_keys(server):
    eng, base = server
    h = _get(base, "/health")
    assert h["admin_config"] is True
    assert h["enable_thinking"] is None and h["reasoning_budget_message"] is None
    eng.template_defaults["enable_thinking"] = True
    eng.think_budget_message = "Answer."
    h = _get(base, "/health")
    assert h["enable_thinking"] is True and h["reasoning_budget_message"] == "Answer."


def test_direct_batch_engine_callers_get_explicit_budget_enforcement():
    # a library caller passing think_budget without knowing about the scheduler hint must
    # queue an EXPLICIT job (enforced via the serial path), never an ambient one that a
    # concurrent batch could silently skip
    be = object.__new__(S.BatchEngine)

    class _CaptureQ:
        def __init__(self):
            self.jobs = []

        def put(self, job):
            self.jobs.append(job)
            job.result = "done"
            job.done.set()

    be._q = _CaptureQ()
    be.generate([1, 2], max_tokens=5, temperature=0.0, think_budget=512)
    assert be._q.jobs[0].params["budget_explicit"] is True
    assert not S.BatchEngine._batchable_greedy(be._q.jobs[0])


def test_batch_routing_explicit_budget_serial_ambient_budget_batches():
    from types import SimpleNamespace

    ok = {"presence_penalty": 0, "frequency_penalty": 0, "logprobs": None,
          "temperature": 0.0, "think_budget": None, "budget_explicit": False}
    assert S.BatchEngine._batchable_greedy(SimpleNamespace(params=ok))
    # an EXPLICITLY requested budget forfeits batching for exact enforcement...
    assert not S.BatchEngine._batchable_greedy(
        SimpleNamespace(params={**ok, "think_budget": 512, "budget_explicit": True}))
    # ...but the ambient server default must NOT disable --max-batch across the board
    assert S.BatchEngine._batchable_greedy(
        SimpleNamespace(params={**ok, "think_budget": 8192}))
    be = object.__new__(S.BatchEngine)
    be.engine = SimpleNamespace(mode="dspark")
    routed = []
    be._run_serial = lambda j: routed.append("serial")
    be._run_session = lambda jobs: routed.append("session")
    be._run_batched = lambda jobs, key: routed.append("batched")
    jobs = [SimpleNamespace(params={**ok, "top_p": 1.0, "top_k": 0,
                                    "think_budget": 512, "budget_explicit": True}),
            SimpleNamespace(params={**ok, "top_p": 1.0, "top_k": 0, "think_budget": 8192}),
            SimpleNamespace(params={**ok, "top_p": 1.0, "top_k": 0, "think_budget": 8192})]
    be._run(jobs)
    assert routed == ["serial", "session"]


def test_race_reasoning_effort_param_validated_and_echoed(server):
    eng, base = server
    eng.race_arms_available = lambda: ["dspark", "baseline"]
    eng.race = lambda prompt_ids, arms, max_tokens, on_event: None

    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base, "/admin/race",
              {"prompt": "hi", "arms": ["dspark", "baseline"], "reasoning_effort": "max"})
    assert e.value.code == 400

    out = _post(base, "/admin/race",
                {"prompt": "hi", "arms": ["dspark", "baseline"], "reasoning_effort": "Low"},
                stream=True)
    start = next(line for line in out.splitlines() if line.startswith("data:"))
    assert json.loads(start[5:])["reasoning_effort"] == "low"


# --- no-model server state (`serve --no-model`, /admin/unload) ---------------------------


class _CloseableEngine(_FakeEngine):
    def __init__(self):
        super().__init__()
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def holder_server():
    holder = S.EngineHolder(_CloseableEngine(), load_kwargs={})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(holder, api_key=None))
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield holder, f"http://127.0.0.1:{port}"
    httpd.shutdown()


def test_no_model_health_and_503_wording(holder_server):
    """A model-less server reports `no_model` (distinct from `loading` — a client waits
    through one and shows a picker on the other), and generation 503s with the reason."""
    holder, base = holder_server
    holder._engine = None
    h = _get(base, "/health")
    assert h["status"] == "no_model" and h["loading"] is False

    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
    assert e.value.code == 503
    assert "no model is loaded" in json.loads(e.value.read())["error"]["message"]


def test_admin_unload_frees_the_engine(holder_server):
    holder, base = holder_server
    eng = holder.current
    s = _post(base, "/admin/unload", {})
    assert s["ready"] is False and s["model"] is None
    assert eng.closed is True
    assert _get(base, "/health")["status"] == "no_model"
    # A second unload is a no-op, not an error.
    assert _post(base, "/admin/unload", {})["ready"] is False


def test_inventory_routes_answer_without_a_model(holder_server):
    """/doctor and /admin/models are model-free by design — a picker must work from the
    no-model state. /admin/models reports loaded=None there."""
    holder, base = holder_server
    holder._engine = None
    assert "ok" in _get(base, "/doctor")
    inv = _get(base, "/admin/models")
    assert inv["loaded"] is None
    assert "models" in inv and "installed" in inv


# --- streaming reasoning split (OpenAI chat SSE) -----------------------------------------


def _stream_fields(sse):
    chunks = [json.loads(l[6:]) for l in sse.split("\n\n")
              if l.startswith("data: ") and l != "data: [DONE]"]
    reasoning = "".join(c["choices"][0]["delta"].get("reasoning_content", "") for c in chunks)
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    return reasoning.strip(), content.strip()


def test_stream_splits_self_opened_thinking(server):
    """Inline `<think>…</think>` rides in `reasoning_content` when streaming, matching the
    non-streaming path — clients otherwise render reasoning as answer text."""
    eng, base = server
    eng.response_text = "<think>plan deeply</think>Sure thing"
    sse = _post(base, "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "hi"}], "stream": True}, stream=True)
    assert _stream_fields(sse) == ("plan deeply", "Sure thing")


def test_stream_splits_prefilled_thinking(server):
    """Prefilled-opener templates (the prompt tail ends in `<think>`) generate only the
    closer; the split keys off the decoded prompt tail, not the output."""
    eng, base = server
    eng.response_text = "I reason here</think>The answer"
    sse = _post(base, "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "hi<think>"}], "stream": True},
                stream=True)
    assert _stream_fields(sse) == ("I reason here", "The answer")


def test_race_cap_auto_and_validation(server):
    """Race arms accept cap 'auto' for drafter modes (per-round adaptive cap from the
    cached curves), reject it for modes with no controller to drive, and reject garbage
    caps — each with the reason, not a silent int() crash."""
    eng, base = server
    eng.race_arms_available = lambda: ["dspark", "baseline", "lookup"]
    captured = {}
    eng.race = lambda prompt_ids, arms, max_tokens, on_event: captured.update(arms=arms)

    out = _post(base, "/admin/race",
                {"prompt": "hi", "arms": [{"mode": "dspark", "cap": "auto"},
                                          {"mode": "baseline"}]},
                stream=True)
    assert captured["arms"][0] == {"mode": "dspark", "cap": "auto", "confidence": None}
    start = next(line for line in out.splitlines() if line.startswith("data:"))
    assert json.loads(start[5:])["arms"][0]["cap"] == "auto"   # echoed for the client

    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base, "/admin/race",
              {"prompt": "hi", "arms": [{"mode": "baseline", "cap": "auto"},
                                        {"mode": "dspark"}]})
    assert e.value.code == 400
    assert "auto" in json.loads(e.value.read())["error"]["message"]

    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base, "/admin/race",
              {"prompt": "hi", "arms": [{"mode": "dspark", "cap": "seven"},
                                        {"mode": "baseline"}]})
    assert e.value.code == 400

    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base, "/admin/race",
              {"prompt": "hi", "arms": [{"mode": "dspark", "cap": 0},
                                        {"mode": "baseline"}]})
    assert e.value.code == 400


def test_race_custom_int_cap_passes_through(server):
    """Any cap in 1..64 rides through to the arm — the Lab's custom-cap field depends on
    the server not silently clamping to its chip presets."""
    eng, base = server
    eng.race_arms_available = lambda: ["dspark", "baseline"]
    captured = {}
    eng.race = lambda prompt_ids, arms, max_tokens, on_event: captured.update(arms=arms)
    _post(base, "/admin/race",
          {"prompt": "hi", "arms": [{"mode": "dspark", "cap": 13}, {"mode": "baseline"}]},
          stream=True)
    assert captured["arms"][0] == {"mode": "dspark", "cap": 13, "confidence": None}


def test_race_arm_confidence_validated_and_passed(server):
    """A per-arm confidence threshold rides through for dspark arms (the cap+conf bundle
    race), is a clear 400 on non-drafter arms and out-of-range values, and its presence is
    advertised via /health's race_arm_confidence capability flag — a client must gate on
    that, or an older engine would silently drop the field and the lane label would lie."""
    eng, base = server
    eng.race_arms_available = lambda: ["dspark", "baseline", "lookup"]
    captured = {}
    eng.race = lambda prompt_ids, arms, max_tokens, on_event: captured.update(arms=arms)

    out = _post(base, "/admin/race",
                {"prompt": "hi", "arms": [{"mode": "dspark", "cap": 7, "confidence": 0.3},
                                          {"mode": "baseline"}]},
                stream=True)
    assert captured["arms"][0] == {"mode": "dspark", "cap": 7, "confidence": 0.3}
    start = next(line for line in out.splitlines() if line.startswith("data:"))
    assert json.loads(start[5:])["arms"][0]["confidence"] == 0.3

    for bad_arm in ({"mode": "baseline", "confidence": 0.3},
                    {"mode": "lookup", "confidence": 0.3},
                    {"mode": "dspark", "confidence": 1.5},
                    {"mode": "dspark", "confidence": True}):
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(base, "/admin/race",
                  {"prompt": "hi", "arms": [bad_arm, {"mode": "dspark"}]})
        assert e.value.code == 400, bad_arm

    assert _get(base, "/health")["race_arm_confidence"] is True


# --------------------------------------------------------------- issue #14: stream liveness


def test_health_reports_small_m(server):
    """/health carries the small-M kernel's live state so a serve-side A/B is visible
    (issue #14: with no flag and no report, the only A/B was a version downgrade)."""
    eng, base = server
    assert _get(base, "/health")["small_m"] is False    # fake engine: attribute absent
    eng.small_m = True
    assert _get(base, "/health")["small_m"] is True


class _SlowStreamEngine(_FakeEngine):
    """Generation that emits a piece every ``delay`` seconds, honouring StopStreaming the
    way the real loops do (stop at the next boundary, return a normal partial result)."""

    def __init__(self, pieces: int = 8, delay: float = 0.05):
        super().__init__()
        self.pieces = pieces
        self.delay = delay
        self.stopped_early = False
        self.finished = threading.Event()

    def generate(self, prompt_ids, *, on_text=None, **kw):
        from mlx_dspark.generate import StopStreaming

        emitted = 0
        try:
            for _ in range(self.pieces):
                time.sleep(self.delay)
                if on_text is not None:
                    try:
                        on_text("x ")
                    except StopStreaming:
                        self.stopped_early = True
                        break
                emitted += 1
        finally:
            self.finished.set()
        return GenResult(text="x " * emitted, token_ids=list(range(emitted)),
                         num_tokens=emitted, num_rounds=emitted, accept_lengths=[1],
                         target_forwards=emitted, seconds=self.delay * emitted,
                         finish_reason="stop")


_TOOLS_STREAM_REQ = {
    "messages": [{"role": "user", "content": "hi"}],
    "stream": True,
    "tools": [{"type": "function",
               "function": {"name": "t", "description": "d",
                            "parameters": {"type": "object", "properties": {}}}}],
}


def _slow_server(monkeypatch, keepalive: float, **engine_kw):
    monkeypatch.setattr(S, "STREAM_KEEPALIVE_S", keepalive)
    eng = _SlowStreamEngine(**engine_kw)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(eng, api_key=None))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return eng, httpd


def test_tools_stream_streams_text_live_and_keepalives_are_real_chunks(monkeypatch):
    """Issue #19: the tool-calls stream used to buffer the WHOLE generation (role chunk,
    then one delta at the end) — a thinking model's 4-6k-token reasoning preamble meant
    minutes of dead air, and agent clients' inter-chunk idle timers (DSH/pi, 300 s)
    dropped the stream. Now pre-tool-call text streams live through the splitter+gate,
    and the keep-alive on the chat dialect is a spec-legal EMPTY DELTA chunk (SSE
    comments never reset most SDKs' idle timers)."""
    # 40 pieces x "x " = 80 chars: well past the gate's marker holdback (20 chars), so a
    # healthy stretch must stream live; keepalive fires several times over the ~0.4 s run
    eng, httpd = _slow_server(monkeypatch, keepalive=0.03, pieces=40, delay=0.01)
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        raw = _post(base, "/v1/chat/completions", _TOOLS_STREAM_REQ, stream=True)
        chunks = [json.loads(l[6:]) for l in raw.split("\n\n")
                  if l.startswith("data: ") and l != "data: [DONE]"]
        # live streaming: several separate content deltas, not one end-of-stream blob
        content = [c for c in chunks if c["choices"][0]["delta"].get("content")]
        assert len(content) >= 5
        # keep-alives are data chunks with an empty delta — every SDK parses them
        assert any(c["choices"][0]["delta"] == {} and c["choices"][0].get("finish_reason")
                   is None for c in chunks)
        assert ": keepalive" not in raw
        assert raw.rstrip().endswith("data: [DONE]")    # and the stream finishes clean
        assert not eng.stopped_early
    finally:
        httpd.shutdown()


def test_tools_stream_stops_generation_when_client_disconnects(monkeypatch):
    """A vanished client must stop a buffered-tools generation at the next round — not let
    it grind to max_tokens holding the single MLX thread while retries pile up behind it
    (the issue-#14 'wedge': /health green, every later request queued for minutes)."""
    eng, httpd = _slow_server(monkeypatch, keepalive=0.03, pieces=400, delay=0.01)
    try:
        host, port = httpd.server_address
        body = json.dumps(_TOOLS_STREAM_REQ)
        with socket.create_connection((host, port)) as s:
            s.sendall((f"POST /v1/chat/completions HTTP/1.1\r\nHost: {host}\r\n"
                       f"Content-Type: application/json\r\n"
                       f"Content-Length: {len(body)}\r\n\r\n{body}").encode())
            s.recv(4096)                # headers + the role chunk have arrived
        # socket closed: the keep-alive write fails, gone flips, on_text raises StopStreaming
        assert eng.finished.wait(5.0)
        assert eng.stopped_early            # cut short, nowhere near all 400 pieces
    finally:
        httpd.shutdown()


def test_plain_chat_stream_stops_generation_when_client_disconnects(monkeypatch):
    """Same liveness contract on the ordinary (non-tools) chat stream: mid-thinking or
    mid-answer, a dead socket ends generation at the next round via the same flag (the
    write-failure path already covered it whenever a delta went out; the flag covers
    stretches where nothing does)."""
    eng, httpd = _slow_server(monkeypatch, keepalive=0.03, pieces=400, delay=0.01)
    try:
        host, port = httpd.server_address
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "stream": True})
        with socket.create_connection((host, port)) as s:
            s.sendall((f"POST /v1/chat/completions HTTP/1.1\r\nHost: {host}\r\n"
                       f"Content-Type: application/json\r\n"
                       f"Content-Length: {len(body)}\r\n\r\n{body}").encode())
            s.recv(4096)
        assert eng.finished.wait(5.0)
        assert eng.stopped_early
    finally:
        httpd.shutdown()


# ------------------------------------------------- issue #14: RAM-aware context warning


def test_kv_bytes_per_token_qwen38_hybrid():
    """Qwen3.8-27B's real layout: 64 layers, every 4th full attention, 4 kv-heads,
    head_dim 256 -> exactly the measured 64 KB/token of context-growing KV."""
    cfg = {"text_config": {
        "num_hidden_layers": 64, "num_attention_heads": 24, "num_key_value_heads": 4,
        "head_dim": 256, "hidden_size": 5120, "full_attention_interval": 4,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"]
                       + (["linear_attention"] * 3 + ["full_attention"]) * 15,
    }}
    assert S._kv_bytes_per_token(cfg) == 16 * 4 * 256 * 4 == 65536


def test_kv_bytes_per_token_dense_and_patterns():
    dense = {"num_hidden_layers": 36, "num_attention_heads": 32,
             "num_key_value_heads": 8, "head_dim": 128, "hidden_size": 2560}
    assert S._kv_bytes_per_token(dense) == 36 * 8 * 128 * 4      # every layer counts

    nemotron = dict(dense, hybrid_override_pattern="M*M-M*M-")   # '*' marks attention
    assert S._kv_bytes_per_token(nemotron) == 2 * 8 * 128 * 4

    sliding = dict(dense, layer_types=["sliding_attention"] * 36)
    assert S._kv_bytes_per_token(sliding) == 0                   # bounded cache: no warning

    assert S._kv_bytes_per_token({}) is None                     # config doesn't say


def test_kv_bytes_per_token_scales_with_kv_bits():
    """Quantized KV shrinks the estimate: bits + 0.5 bits/element of group scale+bias
    (group 64, 16-bit scale + bias), so kv8 = 8.5/16 and kv4 = 4.5/16 of full precision."""
    dense = {"num_hidden_layers": 36, "num_attention_heads": 32,
             "num_key_value_heads": 8, "head_dim": 128, "hidden_size": 2560}
    full = S._kv_bytes_per_token(dense)
    assert S._kv_bytes_per_token(dense, 8) == int(full * 8.5 / 16)
    assert S._kv_bytes_per_token(dense, 4) == int(full * 4.5 / 16)


def test_context_ram_warning_triggers_and_suggests_a_cap():
    gb = 1024 ** 3
    # Issue #14's shape: ~29 GB resident, 262144-token window at 64 KB/token (~16 GB of
    # KV) against a 64 GB Mac's ~48 GB working set -> warn, and suggest a window that fits.
    msg = S._context_ram_warning(65536, 262144, 29 * gb, 48 * gb)
    assert msg is not None and "--context-window" in msg
    suggested = int(msg.split("--context-window ")[1].split()[0])
    assert suggested % 8192 == 0
    assert 29 * gb + suggested * 65536 <= 0.9 * 48 * gb          # the suggestion itself fits

    # Fits comfortably -> silent.
    assert S._context_ram_warning(65536, 32768, 29 * gb, 48 * gb) is None
    # Unknown KV cost / bounded cache / unknown budget -> silent, never a false alarm.
    assert S._context_ram_warning(None, 262144, 29 * gb, 48 * gb) is None
    assert S._context_ram_warning(0, 262144, 29 * gb, 48 * gb) is None
    assert S._context_ram_warning(65536, 262144, 29 * gb, None) is None
    # Weights alone already blow the budget -> warn without a useless tiny suggestion.
    msg = S._context_ram_warning(65536, 262144, 47 * gb, 48 * gb)
    assert msg is not None and "--context-window" not in msg


# --------------------------------------------------------------- issue #17: kv_bits

def test_health_reports_kv_bits(server):
    """/health always carries kv_bits (0 = full precision) so a client can gate its picker
    on the key's presence — engines without the /admin/load override also lack the key."""
    eng, base = server
    assert _get(base, "/health")["kv_bits"] == 0     # fake engine: no target attribute

    class _T:
        kv_bits = 8

    eng.target = _T()
    assert _get(base, "/health")["kv_bits"] == 8


def test_admin_load_rejects_bad_kv_bits(holder_server):
    _holder, base = holder_server
    for bad in (2, 16, "8", True):
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(base, "/admin/load", {"model": "repo", "kv_bits": bad})
        assert e.value.code == 400, bad
        assert "kv_bits" in json.loads(e.value.read())["error"]["message"]


def test_tools_stream_reasoning_and_pretool_text_stream_incrementally(server):
    """Issue #19's shape end to end: thinking + answer + tool call, with `tools` in the
    request. The reasoning must arrive as MULTIPLE reasoning_content deltas while
    generation runs (a thinking model's preamble is most of the wait), pre-marker answer
    text streams as content, and the tool call still lands atomically at the end with
    finish_reason tool_calls."""
    eng, base = server
    eng.response_text = ("<think>first I plan then I act on the plan carefully"
                         "</think>Sure — calling it now for you. "
                         '<tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call>')
    sse = _post(base, "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "call f"}], "tools": _TOOLS,
                 "stream": True}, stream=True)
    chunks = [json.loads(l[6:]) for l in sse.split("\n\n")
              if l.startswith("data: ") and l != "data: [DONE]"]
    reasoning = [c for c in chunks if c["choices"][0]["delta"].get("reasoning_content")]
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    tc = [c for c in chunks if c["choices"][0]["delta"].get("tool_calls")]
    assert len(reasoning) >= 3                      # incremental, not one end-of-stream blob
    assert "".join(c["choices"][0]["delta"]["reasoning_content"]
                   for c in reasoning).strip() == "first I plan then I act on the plan carefully"
    assert content.strip() == "Sure — calling it now for you."
    assert tc and tc[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "f"
    assert json.loads(tc[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]) \
        == {"x": 1}
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
    # ordering: every reasoning/content delta precedes the tool_calls delta
    assert max(i for i, c in enumerate(chunks)
               if c["choices"][0]["delta"].get("reasoning_content")
               or c["choices"][0]["delta"].get("content")) \
        < next(i for i, c in enumerate(chunks) if c["choices"][0]["delta"].get("tool_calls"))
