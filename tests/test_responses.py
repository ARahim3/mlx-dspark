"""OpenAI Responses API (`/v1/responses`) — the surface Codex talks to once its provider is
configured with `wire_api = "responses"`.

Two layers, both model-free, same split as `test_anthropic.py`:

* the pure translation in ``responses_api`` (request -> chat messages, generated text/tool
  calls -> output items / SSE events), tested directly;
* the HTTP endpoints, tested against the same mock-engine harness `test_anthropic.py` and
  `test_server.py` use, so routing, SSE framing, and the error envelope run in milliseconds
  without weights.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from mlx_dspark import responses_api as R
from mlx_dspark import server as S
from mlx_dspark.generate import GenResult

# --------------------------------------------------------------------------- translation


def test_convert_input_plain_string():
    assert R.convert_input("hello") == [{"role": "user", "content": "hello"}]


def test_convert_input_with_instructions():
    msgs = R.convert_input("hello", instructions="be terse")
    assert msgs == [{"role": "system", "content": "be terse"},
                    {"role": "user", "content": "hello"}]


def test_convert_input_message_items():
    msgs = R.convert_input([
        {"type": "message", "role": "user", "content": "hi"},
        {"type": "message", "role": "assistant", "content": "hello"},
    ])
    assert msgs == [{"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"}]


def test_convert_input_content_parts_are_flattened():
    msgs = R.convert_input([{"type": "message", "role": "user",
                            "content": [{"type": "input_text", "text": "a"},
                                       {"type": "input_text", "text": "b"}]}])
    assert msgs == [{"role": "user", "content": "a\nb"}]


def test_convert_input_bare_message_without_type_defaults_to_message():
    # some clients send {"role", "content"} without the "type": "message" wrapper
    msgs = R.convert_input([{"role": "user", "content": "hi"}])
    assert msgs == [{"role": "user", "content": "hi"}]


def test_convert_input_function_call_round_trip():
    msgs = R.convert_input([
        {"type": "message", "role": "user", "content": "weather in Boston?"},
        {"type": "function_call", "call_id": "call_1", "name": "get_weather",
         "arguments": '{"city": "Boston"}'},
        {"type": "function_call_output", "call_id": "call_1",
         "output": '{"temp_f": 52}'},
    ])
    assert msgs == [
        {"role": "user", "content": "weather in Boston?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city": "Boston"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temp_f": 52}'},
    ]


def test_convert_input_function_call_output_serializes_non_string_output():
    msgs = R.convert_input([
        {"type": "function_call_output", "call_id": "call_1", "output": {"temp_f": 52}},
    ])
    assert json.loads(msgs[0]["content"]) == {"temp_f": 52}


def test_convert_input_unknown_item_types_are_skipped_not_rejected():
    # a prior response's reasoning item, replayed verbatim by a stateless client — this
    # server has nothing to do with it, and it must not 400 (same policy anthropic_api
    # follows for fields it doesn't implement)
    msgs = R.convert_input([
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking..."}]},
        {"type": "message", "role": "user", "content": "hi"},
    ])
    assert msgs == [{"role": "user", "content": "hi"}]


def test_convert_input_non_list_non_string_returns_empty():
    assert R.convert_input(None) == []
    assert R.convert_input(42) == []


def test_convert_tools_flattens_to_chat_completions_shape():
    tools = R.convert_tools([{"type": "function", "name": "get_weather",
                             "description": "d",
                             "parameters": {"type": "object",
                                           "properties": {"city": {"type": "string"}}}}])
    assert tools == [{"type": "function", "function": {
        "name": "get_weather", "description": "d",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]


def test_convert_tools_accepts_already_nested_shape():
    nested = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    assert R.convert_tools(nested) == nested


def test_convert_tools_defaults_missing_parameters():
    tools = R.convert_tools([{"type": "function", "name": "f"}])
    assert tools[0]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_convert_tools_empty_and_none():
    assert R.convert_tools(None) is None
    assert R.convert_tools([]) is None


def test_build_response_text_only():
    body = R.build_response(resp_id="resp_1", model="m", content="hi", input_tokens=5,
                            output_tokens=1, finish_reason="stop", created=100)
    assert body["status"] == "completed"
    assert body["output"] == [{
        "id": body["output"][0]["id"], "type": "message", "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "hi", "annotations": []}]}]
    assert body["usage"] == {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6}


def test_build_response_incomplete_on_length():
    body = R.build_response(resp_id="resp_1", model="m", content="cut off", input_tokens=5,
                            output_tokens=10, finish_reason="length", created=100)
    assert body["status"] == "incomplete"
    assert body["incomplete_details"] == {"reason": "max_output_tokens"}


def test_build_response_order_is_reasoning_then_tool_calls_then_text():
    body = R.build_response(
        resp_id="resp_1", model="m", content="answer", reasoning="thinking",
        tool_calls=[{"id": "call_1", "function": {"name": "f", "arguments": "{}"}}],
        input_tokens=5, output_tokens=1, finish_reason="stop", created=100)
    types = [item["type"] for item in body["output"]]
    assert types == ["reasoning", "function_call", "message"]


def test_build_response_no_reasoning_no_tool_calls_omits_them():
    body = R.build_response(resp_id="resp_1", model="m", content="hi", input_tokens=1,
                            output_tokens=1, finish_reason="stop", created=100)
    assert len(body["output"]) == 1
    assert body["output"][0]["type"] == "message"


# --------------------------------------------------------------------------- ResponseStream


def _names(events):
    return [name for name, _ in events]


def test_stream_plain_text_event_sequence():
    s = R.ResponseStream(model="m", input_tokens=5)
    start = _names(s.start())
    assert start == ["response.created"]
    d = _names(s.delta("hello"))
    assert d == ["response.output_item.added", "response.content_part.added",
                "response.output_text.delta"]
    f = _names(s.finish(finish_reason="stop", output_tokens=1))
    assert f == ["response.output_text.done", "response.content_part.done",
                "response.output_item.done", "response.completed"]


def test_stream_second_delta_does_not_reopen_the_item():
    s = R.ResponseStream(model="m", input_tokens=5)
    s.start()
    first = _names(s.delta("a"))
    second = _names(s.delta("b"))
    assert first == ["response.output_item.added", "response.content_part.added",
                     "response.output_text.delta"]
    assert second == ["response.output_text.delta"]


def test_stream_empty_delta_is_a_no_op():
    s = R.ResponseStream(model="m", input_tokens=5)
    assert s.delta("") == []
    assert s.delta(None) == []


def test_stream_pure_tool_call_never_opens_a_message_item():
    # a turn where the model only calls a tool (no preceding text) must not announce and
    # then close an empty message item that then isn't in the final `output` array
    s = R.ResponseStream(model="m", input_tokens=5)
    s.start()
    events = s.finish(finish_reason="tool_calls", output_tokens=3,
                      tool_calls=[{"id": "call_1",
                                  "function": {"name": "f", "arguments": "{}"}}])
    names = _names(events)
    assert "response.output_item.added" in names
    assert "response.output_text.done" not in names   # no message item was ever opened
    completed = dict(events)["response.completed"]
    types = [item["type"] for item in completed["response"]["output"]]
    assert types == ["function_call"]                 # nothing phantom in the final output


def test_stream_text_then_tool_call_indices_are_sequential():
    s = R.ResponseStream(model="m", input_tokens=5)
    s.start()
    s.delta("checking...")
    events = s.finish(finish_reason="tool_calls", output_tokens=3,
                      tool_calls=[{"id": "call_1",
                                  "function": {"name": "f", "arguments": "{}"}}])
    added = [payload for name, payload in events if name == "response.output_item.added"]
    assert [a["output_index"] for a in added] == [1]     # message already claimed index 0
    completed = dict(events)["response.completed"]
    types = [item["type"] for item in completed["response"]["output"]]
    assert types == ["message", "function_call"]


def test_stream_multiple_tool_calls_get_increasing_indices():
    s = R.ResponseStream(model="m", input_tokens=5)
    s.start()
    events = s.finish(finish_reason="tool_calls", output_tokens=3, tool_calls=[
        {"id": "call_1", "function": {"name": "f", "arguments": "{}"}},
        {"id": "call_2", "function": {"name": "g", "arguments": "{}"}},
    ])
    added = [payload for name, payload in events if name == "response.output_item.added"]
    assert [a["output_index"] for a in added] == [0, 1]


def test_stream_sequence_numbers_are_monotonic():
    s = R.ResponseStream(model="m", input_tokens=5)
    events = s.start() + s.delta("hi") + s.finish(finish_reason="stop", output_tokens=1)
    seqs = [payload["sequence_number"] for _, payload in events]
    assert seqs == sorted(seqs) == list(range(1, len(seqs) + 1))


def test_stream_completed_response_carries_usage():
    s = R.ResponseStream(model="m", input_tokens=7)
    s.start()
    s.delta("hi")
    events = s.finish(finish_reason="stop", output_tokens=2)
    completed = dict(events)["response.completed"]
    assert completed["response"]["usage"] == {"input_tokens": 7, "output_tokens": 2,
                                              "total_tokens": 9}


def test_stream_incomplete_emits_response_incomplete_not_completed():
    s = R.ResponseStream(model="m", input_tokens=5)
    s.start()
    s.delta("cut off")
    events = s.finish(finish_reason="length", output_tokens=100)
    names = _names(events)
    assert "response.incomplete" in names
    assert "response.completed" not in names


# --------------------------------------------------------------------------- HTTP endpoint


class _FakeTok:
    """Deterministic, template-free: encode() is 1 id per character."""

    def encode(self, text):
        return [ord(c) for c in text]

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
    context_window = None
    is_muse = False

    def __init__(self, response_text="Hello world"):
        self.tokenizer = _FakeTok()
        self.calls = []
        self.response_text = response_text

    def generate(self, prompt_ids, *, max_tokens, temperature, top_p=1.0, top_k=0,
                presence_penalty=0.0, frequency_penalty=0.0, logprobs=None,
                stop=None, seed=None, on_text=None):
        self.calls.append({"prompt_ids": prompt_ids, "max_tokens": max_tokens,
                               "temperature": temperature, "stop": stop, "seed": seed})
        text = self.response_text
        if on_text:
            for i in range(0, len(text), 5):
                on_text(text[i:i + 5])
        return GenResult(text=text, token_ids=[1, 2, 3], num_tokens=3, num_rounds=2,
                         accept_lengths=[2, 1], target_forwards=2, seconds=0.1,
                         finish_reason="stop")

    def spec_info(self, res):
        return {"mode": self.mode, "accept_len": res.mean_accept_len}

    def metrics(self):
        return {"model": self.model_id, "requests": len(self.calls)}


def _serve(engine, api_key=None):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(engine, api_key=api_key))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


@pytest.fixture
def api():
    eng = _FakeEngine()
    httpd, base = _serve(eng)
    yield eng, base
    httpd.shutdown()
    httpd.server_close()


def _post(base, path, body, raw=False, timeout=10):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        text = r.read().decode()
    return text if raw else json.loads(text)


def _read_sse(body: str):
    out = []
    for block in body.strip().split("\n\n"):
        name = data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if data is not None:
            out.append((name, data))
    return out


def test_responses_non_streaming(api):
    _, base = api
    r = _post(base, "/v1/responses", {"model": "gpt-4o", "input": "hi",
                                      "max_output_tokens": 100})
    assert r["object"] == "response" and r["status"] == "completed"
    assert r["model"] == "gpt-4o"      # echoed back, as a gateway would
    assert r["output"][0]["content"][0]["text"] == "Hello world"
    assert r["usage"]["output_tokens"] == 3


def test_responses_input_reaches_the_engine_as_a_user_message(api):
    eng, base = api
    _post(base, "/v1/responses", {"input": "hi there", "max_output_tokens": 10})
    assert len(eng.calls) == 1
    assert eng.calls[0]["prompt_ids"] == [ord(c) for c in "hi there"]


def test_responses_missing_input_is_a_400(api):
    _, base = api
    req = urllib.request.Request(base + "/v1/responses",
                                 data=json.dumps({"model": "m"}).encode(),
                                 headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)
    assert e.value.code == 400
    body = json.loads(e.value.read().decode())
    assert "input" in body["error"]["message"]


def test_responses_max_output_tokens_is_clamped_to_the_server_cap(api):
    eng, base = api
    _post(base, "/v1/responses", {"input": "hi", "max_output_tokens": 999999})
    assert eng.calls[0]["max_tokens"] == eng.max_tokens_cap


def test_responses_streaming_event_sequence(api):
    _, base = api
    body = _post(base, "/v1/responses",
                {"input": "hi", "max_output_tokens": 100, "stream": True}, raw=True)
    events = _read_sse(body)
    names = [n for n, _ in events]
    assert names[0] == "response.created"
    assert names[-1] == "response.completed"
    assert "response.output_text.delta" in names
    full_text = "".join(p["delta"] for n, p in events if n == "response.output_text.delta")
    assert full_text == "Hello world"


def test_responses_streaming_usage_in_final_event(api):
    _, base = api
    body = _post(base, "/v1/responses",
                {"input": "hi", "max_output_tokens": 100, "stream": True}, raw=True)
    events = dict(_read_sse(body))
    assert events["response.completed"]["response"]["usage"]["output_tokens"] == 3


def test_responses_tool_call_round_trip_reaches_the_model(api):
    eng, base = api
    eng.response_text = "irrelevant for this assertion"
    tools = [{"type": "function", "name": "get_weather",
             "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}]
    _post(base, "/v1/responses", {
        "input": [
            {"type": "message", "role": "user", "content": "weather in Boston?"},
            {"type": "function_call", "call_id": "call_1", "name": "get_weather",
             "arguments": '{"city": "Boston"}'},
            {"type": "function_call_output", "call_id": "call_1", "output": "sunny, 70F"},
        ],
        "tools": tools, "max_output_tokens": 100,
    })
    # the tool round trip must survive normalize_tool_messages and reach the template as
    # real text the fake (template-free) tokenizer can encode — not raise or vanish
    assert len(eng.calls) == 1
    assert eng.calls[0]["prompt_ids"]


def test_responses_model_emitting_a_tool_call_is_reported_as_a_function_call(api):
    eng, base = api
    eng.response_text = '<tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>'
    tools = [{"type": "function", "name": "get_weather",
             "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}]
    r = _post(base, "/v1/responses", {"input": "weather?", "tools": tools,
                                      "max_output_tokens": 100})
    assert r["output"][0]["type"] == "function_call"
    assert r["output"][0]["name"] == "get_weather"
    assert json.loads(r["output"][0]["arguments"]) == {"city": "NYC"}


def test_responses_streaming_tool_call_never_leaks_native_syntax_as_text(api):
    eng, base = api
    eng.response_text = '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'
    tools = [{"type": "function", "name": "get_weather", "parameters": {}}]
    body = _post(base, "/v1/responses", {"input": "weather?", "tools": tools,
                                         "max_output_tokens": 100, "stream": True}, raw=True)
    events = _read_sse(body)
    for name, payload in events:
        if name == "response.output_text.delta":
            assert "<tool_call>" not in payload["delta"]
    completed = dict(events)["response.completed"]
    types = [item["type"] for item in completed["response"]["output"]]
    assert "function_call" in types


def test_responses_openai_route_still_works_alongside(api):
    # adding /v1/responses must not disturb the existing Chat Completions route
    _, base = api
    r = _post(base, "/v1/chat/completions",
             {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10})
    assert r["object"] == "chat.completion"


def test_responses_unknown_request_fields_never_fail(api):
    # Codex's own field set grows across releases; an unrecognised field must not 400
    _, base = api
    r = _post(base, "/v1/responses", {"input": "hi", "max_output_tokens": 10,
                                      "parallel_tool_calls": True, "store": False,
                                      "previous_response_id": None, "metadata": {"a": 1}})
    assert r["object"] == "response"
