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
    cap_controller = None
    is_muse = False           # mirrors Engine.is_muse (muse_glimmer channel parsing off)
    # Borrow the real Engine's reasoning-effort logic (pure over self.tokenizer) so the
    # server's map_reasoning_effort wiring is exercised against the real behavior.
    supports_reasoning_effort = S.Engine.supports_reasoning_effort
    reasoning_effort_vocab = S.Engine.reasoning_effort_vocab
    map_reasoning_effort = S.Engine.map_reasoning_effort

    def __init__(self):
        self.tokenizer = _FakeTok()
        self.calls = []
        self.response_text = "Hello world from mlx dspark"
        self.reused_tokens = 0        # prompt tokens a prefix-cache hit would serve

    def generate(self, prompt_ids, *, max_tokens, temperature, top_p=1.0, top_k=0,
                 presence_penalty=0.0, frequency_penalty=0.0, logprobs=None,
                 stop, seed, on_text=None):
        self.calls.append({"prompt_ids": prompt_ids, "max_tokens": max_tokens,
                               "temperature": temperature, "top_p": top_p, "top_k": top_k,
                               "presence_penalty": presence_penalty,
                               "frequency_penalty": frequency_penalty, "logprobs": logprobs,
                               "stop": stop, "seed": seed})
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
                         finish_reason="stop", logprobs=lp,
                         reused_tokens=self.reused_tokens)

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


def test_usage_reports_cached_tokens(server):
    """PR #9 (@joeOGsan): prefix-cache reuse is visible to the client in OpenAI's own shape,
    `usage.prompt_tokens_details.cached_tokens` — the difference between measuring the cache
    and guessing at it from TTFT."""
    eng, base = server
    eng.reused_tokens = 0
    c = _post(base, "/v1/chat/completions",
              {"model": "x", "messages": [{"role": "user", "content": "hi"}]})
    assert c["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    eng.reused_tokens = 7
    c = _post(base, "/v1/chat/completions",
              {"model": "x", "messages": [{"role": "user", "content": "hi"}]})
    assert c["usage"]["prompt_tokens_details"]["cached_tokens"] == 7


def test_stream_usage_reports_cached_tokens(server):
    """Same field on the streamed final chunk, where clients actually read it."""
    eng, base = server
    eng.reused_tokens = 5
    sse = _post(base, "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "hi"}], "stream": True,
                 "stream_options": {"include_usage": True}}, stream=True)
    chunks = [json.loads(l[6:]) for l in sse.split("\n\n")
              if l.startswith("data: ") and l != "data: [DONE]"]
    assert chunks[-1]["usage"]["prompt_tokens_details"]["cached_tokens"] == 5


def test_completions_usage_reports_cached_tokens(server):
    """/v1/completions shares the usage builder, so it reports it too."""
    eng, base = server
    eng.reused_tokens = 3
    c = _post(base, "/v1/completions", {"model": "x", "prompt": "hi"})
    assert c["usage"]["prompt_tokens_details"]["cached_tokens"] == 3


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


def test_events_stream_names_prefill_and_ends_cleanly_across_a_model_swap():
    """Named live events share the stream, which still ends cleanly on a hot swap.

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
        payload = {"req": "abc", "mode": "dflash", "processed": 2048,
                   "total": 22000, "active": True}
        holder._engine.rounds.publish("prefill", payload)
        while stream.readline() != b"event: prefill\n":
            pass
        assert json.loads(stream.readline().removeprefix(b"data: ")) == payload
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


def test_map_effort_to_vocab():
    """Unsupported-but-valid effort maps to the nearest supported value, ties round DOWN
    (issue #19: 'high' -> 'medium' on a Qwen3.8-style {low, medium, xhigh} template)."""
    qwen = frozenset({"low", "medium", "xhigh"})
    assert S._map_effort_to_vocab("high", qwen) == "medium"     # tie -> less thinking
    assert S._map_effort_to_vocab("medium", qwen) == "medium"   # supported -> unchanged
    assert S._map_effort_to_vocab("xhigh", qwen) == "xhigh"
    assert S._map_effort_to_vocab("low", qwen) == "low"
    assert S._map_effort_to_vocab("high", None) == "high"       # template ignores effort
    assert S._map_effort_to_vocab("high", frozenset()) == "high"
    sparse = frozenset({"low", "xhigh"})                        # nearest by distance, no tie
    assert S._map_effort_to_vocab("high", sparse) == "xhigh"    # high(2): xhigh d1 < low d2
    assert S._map_effort_to_vocab("medium", sparse) == "low"    # medium(1): low d1 < xhigh d2


class _Qwen38Tok(_FakeTok):
    """A template that accepts {low, medium, xhigh} and REJECTS 'high' like Qwen3.8's does —
    via a non-ValueError raise (encode_messages only retries TypeError/ValueError, so a real
    template's raise_exception propagates), so the vocab probe genuinely excludes 'high'."""

    chat_template = "{% if reasoning_effort %}hint{% endif %}"
    SUPPORTED = {"low", "medium", "xhigh"}

    def __init__(self):
        self.template_kwargs = []

    def apply_chat_template(self, messages, add_generation_prompt=True, **kw):
        eff = kw.get("reasoning_effort")
        if eff is not None and eff not in self.SUPPORTED:
            raise RuntimeError(f"Unexpected reasoning effort {eff}")
        self.template_kwargs.append(kw)
        return [1, 2, 3]


def test_reasoning_effort_maps_unsupported_instead_of_400(server):
    """issue #19: a client's 'high' on a {low,medium,xhigh} model reaches the template as
    'medium' (not a 400), while a supported value passes through unchanged."""
    eng, base = server
    eng.tokenizer = _Qwen38Tok()
    if hasattr(eng, "_effort_vocab"):
        del eng._effort_vocab                       # probe freshly against this tokenizer
    _post(base, "/v1/chat/completions",
          {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "high"})
    assert eng.tokenizer.template_kwargs[-1]["reasoning_effort"] == "medium"
    _post(base, "/v1/chat/completions",
          {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "xhigh"})
    assert eng.tokenizer.template_kwargs[-1]["reasoning_effort"] == "xhigh"


def test_anthropic_output_config_effort_overrides_and_clamps(server):
    """issue #25: Claude Code ships /effort (and --effort) in `output_config.effort`, NOT in
    `thinking`. It must reach the template as a per-request override of the server default,
    clamped to what THIS template accepts, degrade to the default on a bad value, and be
    skipped when thinking is disabled."""
    eng, base = server
    eng.tokenizer = _Qwen38Tok()
    eng.template_defaults = {"reasoning_effort": "low"}   # server default, to be overridden
    if hasattr(eng, "_effort_vocab"):
        del eng._effort_vocab                             # probe freshly against this tokenizer

    def eff():
        return eng.tokenizer.template_kwargs[-1].get("reasoning_effort")

    def msg(**extra):
        return {"model": "x", "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}], **extra}

    # 'high' (not in this template's vocab) -> clamped to 'medium', overriding the 'low' default
    _post(base, "/v1/messages", msg(output_config={"effort": "high"}))
    assert eff() == "medium"
    # 'xhigh' passes through
    _post(base, "/v1/messages", msg(output_config={"effort": "xhigh"}))
    assert eff() == "xhigh"
    # a bogus value keeps the server default rather than 400ing
    _post(base, "/v1/messages", msg(output_config={"effort": "bogus"}))
    assert eff() == "low"
    # output_config with only `format` (Claude Code's internal JSON tasks) touches nothing
    _post(base, "/v1/messages", msg(output_config={"format": {"type": "json_schema"}}))
    assert eff() == "low"
    # thinking disabled -> effort is NOT re-injected (stays the default, not the xhigh override)
    _post(base, "/v1/messages",
          msg(output_config={"effort": "xhigh"}, thinking={"type": "disabled"}))
    kw = eng.tokenizer.template_kwargs[-1]
    assert kw.get("enable_thinking") is False and kw.get("reasoning_effort") == "low"


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


def test_health_reports_cpu_split(server):
    eng, base = server
    assert _get(base, "/health")["cpu_split"] is None
    eng.cpu_split = {"min_rows": 512, "fracs": {512: 0.2}}
    assert _get(base, "/health")["cpu_split"]["min_rows"] == 512


# --- on-load warmup: kernels are primed before "ready" so the first request is fast --------


def test_health_reports_warmup(server):
    """/health carries whether the load ran a warmup pass, so a client can see --no-warmup
    took effect (default on)."""
    eng, base = server
    assert _get(base, "/health")["warmup"] is False     # fake engine: attribute absent
    eng.warmup_enabled = True
    assert _get(base, "/health")["warmup"] is True


def test_holder_status_reports_load_phase():
    """status() distinguishes the weights stage from the warmup stage while loading, and
    carries no phase once ready — so the app can show "Warming up…" only when it's true."""
    holder = S.EngineHolder(_CloseableEngine(), load_kwargs={})
    assert "phase" not in holder.status()               # ready: no phase key
    holder._loading = True
    holder._load_phase = "warming_up"
    assert holder.status()["phase"] == "warming_up"
    holder._load_phase = None                            # loading weights, warmup not reached
    assert holder.status()["phase"] == "loading"         # defaults to "loading", never null


def test_health_loading_reports_phase(holder_server):
    """/health surfaces the load phase mid-load so a polling client (the app) can say
    'Warming up…' on the last stage instead of a bar that looks stuck."""
    holder, base = holder_server
    holder._loading = True
    holder._load_phase = "warming_up"
    h = _get(base, "/health")
    assert h["status"] == "loading" and h["phase"] == "warming_up"


def test_admin_load_rejects_bad_warmup(holder_server):
    _holder, base = holder_server
    for bad in ("yes", 1, 0):
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(base, "/admin/load", {"model": "repo", "warmup": bad})
        assert e.value.code == 400, bad
        assert "warmup" in json.loads(e.value.read())["error"]["message"]


def test_admin_load_validates_and_passes_cpu_split(holder_server):
    holder, base = holder_server
    seen = []
    holder.swap = lambda **kw: seen.append(kw) or holder.status()

    _post(base, "/admin/load", {"model": "repo", "cpu_split": "auto"})
    _post(base, "/admin/load", {"model": "repo", "cpu_split": 0.25})
    assert [call["cpu_split"] for call in seen] == ["auto", 0.25]

    for bad in ("fast", True, -0.1, 1):
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(base, "/admin/load", {"model": "repo", "cpu_split": bad})
        assert e.value.code == 400, bad
        assert "cpu_split" in json.loads(e.value.read())["error"]["message"]


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


# --------------------------------------------------------------------------- roofline surfaces


def test_health_carries_a_warnings_list(server):
    """Memory-pressure / load notes reach clients as {code, level, message, action} rows."""
    _, base = server
    h = _get(base, "/health")
    assert isinstance(h["warnings"], list)
    for row in h["warnings"]:
        assert {"code", "level", "message", "action"} <= set(row)


def test_metrics_reports_system_memory_and_verdict(server):
    _, base = server
    mt = _get(base, "/metrics")
    assert "pressure" in mt["system"] and "swap_used_bytes" in mt["system"]
    assert "verdict" in mt                       # None for a fake engine, a dict for a real one


def test_machine_answers_for_any_engine(server):
    """/machine falls back to chip + bandwidth + memory when the engine has no report
    (fakes, engines built directly) — the shape a client needs to scale its estimates."""
    _, base = server
    m = _get(base, "/machine")
    assert set(m) >= {"chip", "bandwidth", "memory", "model", "roofline", "baseline", "verdict"}
    assert m["bandwidth"]["reference_gb_s"] == 273.0
    assert "allocator" in m["memory"] and "pressure" in m["memory"]
    assert m["model"] is None


def test_machine_answers_during_model_swap():
    """/machine stays usable while an EngineHolder has temporarily no engine."""
    from mlx_dspark.server import EngineHolder

    holder = EngineHolder(None, load_kwargs={})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(holder, api_key=None))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        machine = _get(f"http://127.0.0.1:{port}", "/machine")
        assert machine["model"] is None
        assert "chip" in machine and "memory" in machine
    finally:
        httpd.shutdown()


def test_admin_models_reports_bandwidth_scale(server):
    _, base = server
    bw = _get(base, "/admin/models")["bandwidth"]
    assert bw["reference_gb_s"] == 273.0 and bw["source"] in ("measured", "theoretical", "unknown")
    if bw["theoretical_gb_s"]:
        assert bw["scale"] == pytest.approx(bw["theoretical_gb_s"] / 273.0, rel=1e-3)


class TestEngineSpecInfoTiles:
    """The per-request tiles spec_info adds — pure over a stamped GenResult."""

    def _engine(self, machine=None):
        eng = S.Engine.__new__(S.Engine)
        eng.mode = "dspark"
        eng.cap_controller = None
        eng._last_cap = 4
        eng.machine = machine or {}
        eng.memory_guard = None
        eng.context_window = 4096
        eng.target_repo, eng.drafter_repo = "org/T", "org/D"
        eng.calibration = lambda: {"available": False}
        return eng

    def _result(self, **kw):
        base = {"text": "x", "token_ids": [1] * 40, "num_tokens": 40, "num_rounds": 10,
                "accept_lengths": [4] * 10, "target_forwards": 10, "seconds": 2.5,
                "prefill_seconds": 0.5, "prompt_tokens": 1200, "reused_tokens": 1000,
                "ttft_seconds": 0.55}
        base.update(kw)
        return GenResult(**base)

    def test_timing_tiles(self):
        info = self._engine().spec_info(self._result())
        assert info["prompt_tokens"] == 1200 and info["cached_tokens"] == 1000
        assert info["completion_tokens"] == 40 and info["context_tokens"] == 1240
        assert info["prefill_seconds"] == 0.5 and info["ttft_seconds"] == 0.55
        assert info["decode_seconds"] == 2.0
        assert info["prefill_tokens_per_sec"] == 400.0     # 200 fresh tokens / 0.5 s
        assert "decay_ratio" not in info and "cold" not in info and "swap_delta_bytes" not in info
        assert "ceiling_tokens_per_sec" not in info         # no machine facts -> no roofline

    def test_optional_tiles_appear_when_measured(self):
        info = self._engine().spec_info(self._result(decay_ratio=0.7, cold=True,
                                                     swap_delta_bytes=300_000_000))
        assert info["decay_ratio"] == 0.7 and info["cold"] is True
        assert info["swap_delta_bytes"] == 300_000_000

    def test_roofline_ratio_from_machine_facts(self):
        # 10 GB active weights, 64 KB/token KV, 250 GB/s -> ceiling at ctx 1240
        machine = {"bandwidth": {"gb_s": 250.0}, "kv_bytes_per_token": 65536,
                   "target": {"active_bytes": 10 * 10**9, "active_is_estimate": False}}
        eng = self._engine(machine)
        res = self._result()
        info = eng.spec_info(res)
        bpt = 10 * 10**9 + 65536 * 1240
        assert info["ceiling_tokens_per_sec"] == pytest.approx(250e9 / bpt, rel=1e-3)
        assert info["roofline_ratio"] == pytest.approx(res.decode_tokens_per_sec / (250e9 / bpt),
                                                       rel=1e-2)
        # and the same facts drive the verdict + machine report
        v = eng._verdict_for(res)
        assert v["level"] in ("info", "healthy", "ok", "attention", "problem")
        eng.last_verdict, eng._last_context = v, 1240
        report = eng.machine_report()
        assert report["model"]["target_weights"]["active_bytes"] == 10 * 10**9
        assert report["roofline"]["at_zero"]["ceiling_tps"] == pytest.approx(25.0)
        assert report["roofline"]["at_context_window"]["context"] == 4096
        assert report["baseline"] is None                 # no calibration -> no step time


# --- memory-pressure guard: state on /health, /machine, /metrics; /admin/load override -------


def test_health_reports_memory_guard_state(server):
    _, base = server
    assert _get(base, "/health")["memory_guard"] == {"enabled": False}   # fake: no guard
    assert _get(base, "/metrics")["memory_guard"] == {"enabled": False}


def test_admin_load_rejects_bad_memory_guard(holder_server):
    _, base = holder_server
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base, "/admin/load", {"model": "repo", "memory_guard": "yes"})
    assert e.value.code == 400
    assert "memory_guard" in json.loads(e.value.read())["error"]["message"]


def test_health_warnings_include_a_recent_guard_shed(server):
    """A shed shows up as a /health warning row so a client can explain the re-prefill."""
    from mlx_dspark.memory_guard import MemoryGuard

    eng, base = server
    guard = MemoryGuard(prefix=None, submit=None, is_busy=lambda: False,
                        clear_cache=lambda: None, allocator_bytes=lambda: 0, log=lambda m: None)
    guard.shed("warn")
    eng.memory_guard = guard
    try:
        h = _get(base, "/health")
        assert h["memory_guard"]["sheds"] == 1
        assert any(w["code"] == "memory_guard" for w in h["warnings"])
    finally:
        del eng.memory_guard


class TestEngineMachineReportGuard:
    def test_guard_block_present(self):
        eng = S.Engine.__new__(S.Engine)
        eng.mode, eng.machine, eng.memory_guard = "dspark", {}, None
        eng.last_verdict = None
        eng.target_repo = eng.drafter_repo = None
        assert eng.machine_report()["guard"] == {"enabled": False}


# --- thinking default for API clients (issue #19 part 2) ------------------------------------


def test_health_reports_thinking_default(server):
    eng, base = server
    assert _get(base, "/health")["thinking_default"] == "on"
    eng.template_defaults = {"enable_thinking": False}
    try:
        assert _get(base, "/health")["thinking_default"] == "off"
    finally:
        eng.template_defaults = {}


def test_admin_load_validates_thinking_overrides(holder_server):
    _, base = holder_server
    for body in ({"model": "repo", "enable_thinking": "no"},
                 {"model": "repo", "reasoning_effort": "maximum"}):
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(base, "/admin/load", body)
        assert e.value.code == 400
        msg = json.loads(e.value.read())["error"]["message"]
        assert "enable_thinking" in msg or "reasoning_effort" in msg


def test_swap_makes_thinking_default_sticky(monkeypatch):
    """enable_thinking=false on one swap carries into the next (like context_window);
    true restores the model's own default."""
    seen = []

    class _E:
        model_id = "m"

        def close(self):
            pass

    monkeypatch.setattr(S.Engine, "load", staticmethod(lambda **kw: seen.append(kw) or _E()))
    monkeypatch.setattr(S, "maybe_batch_engine", lambda e, n: e)
    holder = S.EngineHolder(None, {"mode": "auto"})
    holder.swap(model="a", enable_thinking=False, reasoning_effort="low")
    holder.swap(model="b")
    holder.swap(model="c", enable_thinking=True)
    assert seen[0]["enable_thinking"] is False and seen[0]["reasoning_effort"] == "low"
    assert seen[1]["enable_thinking"] is False and seen[1]["reasoning_effort"] == "low"
    assert seen[2]["enable_thinking"] is None


def test_swap_keeps_cpu_split_only_for_the_server_session(monkeypatch):
    seen = []

    class _E:
        model_id = "m"

        def close(self):
            pass

    monkeypatch.setattr(S.Engine, "load", staticmethod(lambda **kw: seen.append(kw) or _E()))
    monkeypatch.setattr(S, "maybe_batch_engine", lambda e, n: e)
    holder = S.EngineHolder(None, {"mode": "auto", "cpu_split": None})
    holder.swap(model="a", cpu_split="auto")
    holder.swap(model="b")
    holder.swap(model="c", cpu_split=0)
    assert [call["cpu_split"] for call in seen] == ["auto", "auto", 0]


# ------------------------------------------------------------------ issue #27: GET auth


def test_get_routes_require_the_key_except_health():
    eng = _FakeEngine()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(eng, api_key="secret"))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        assert _get(base, "/health")["status"] == "ok"           # readiness probe stays open
        for path in ("/admin/integrations", "/v1/models", "/metrics", "/calibration"):
            with pytest.raises(urllib.error.HTTPError) as e:
                _get(base, path)
            assert e.value.code == 401, path
        req = urllib.request.Request(base + "/admin/integrations",
                                     headers={"Authorization": "Bearer secret"})
        body = json.loads(urllib.request.urlopen(req).read())
        assert body["integrations"]                              # the key-bearing configs
        req = urllib.request.Request(base + "/v1/models", headers={"x-api-key": "secret"})
        assert json.loads(urllib.request.urlopen(req).read())["object"] == "list"
    finally:
        httpd.shutdown()
