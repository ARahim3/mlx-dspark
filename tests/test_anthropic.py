"""Anthropic Messages API (`/v1/messages`) — the surface Claude Code talks to.

Two layers, both model-free:

* the pure translation in ``anthropic_api`` (request blocks -> chat messages, generated text
  -> content blocks / SSE events), tested directly;
* the HTTP endpoints, tested against the same mock engine ``test_server.py`` uses, so routing,
  SSE framing, auth, and the error envelope run in milliseconds without weights.

Several tests here pin behaviour that Claude Code *depends on* rather than merely accepts —
the `?beta=true` query it posts to, the `prompt is too long` wording its auto-compact matches,
and the rule that an unrecognised request field must never 400. Those are marked inline.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from mlx_dspark import anthropic_api as A
from mlx_dspark import server as S
from mlx_dspark.generate import GenResult

# --------------------------------------------------------------------------- translation


def test_system_accepts_string_and_blocks():
    assert A.system_text("be brief") == "be brief"
    # Claude Code sends the block form (its cache breakpoints ride on the blocks)
    assert A.system_text([{"type": "text", "text": "one",
                           "cache_control": {"type": "ephemeral"}},
                          {"type": "text", "text": "two"}]) == "one\ntwo"
    assert A.system_text(None) == ""


def test_convert_messages_plain_text():
    msgs = A.convert_messages([{"role": "user", "content": "hi"}], system="sys")
    assert msgs == [{"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"}]


def test_convert_messages_text_blocks():
    msgs = A.convert_messages([{"role": "user",
                                "content": [{"type": "text", "text": "a"},
                                            {"type": "text", "text": "b"}]}])
    assert msgs == [{"role": "user", "content": "a\nb"}]


def test_tool_use_becomes_openai_tool_calls():
    msgs = A.convert_messages([{
        "role": "assistant",
        "content": [{"type": "text", "text": "reading"},
                    {"type": "tool_use", "id": "toolu_1", "name": "Read",
                     "input": {"file_path": "/tmp/a"}}],
    }])
    assert len(msgs) == 1
    m = msgs[0]
    assert m["content"] == "reading"
    (tc,) = m["tool_calls"]
    assert tc["id"] == "toolu_1"
    assert tc["function"]["name"] == "Read"
    # OpenAI shape carries arguments as a JSON *string*
    assert json.loads(tc["function"]["arguments"]) == {"file_path": "/tmp/a"}


def test_tool_result_becomes_tool_message_before_text():
    msgs = A.convert_messages([{
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "file body"},
                    {"type": "text", "text": "now what?"}],
    }])
    # tool results answer the previous assistant turn, so they come first
    assert msgs[0] == {"role": "tool", "tool_call_id": "toolu_1", "content": "file body"}
    assert msgs[1] == {"role": "user", "content": "now what?"}


def test_tool_result_error_and_block_content():
    (m,) = A.convert_messages([{
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t", "is_error": True,
                     "content": [{"type": "text", "text": "no such file"}]}],
    }])
    assert m["content"] == "Error: no such file"


def test_tool_result_only_turn_emits_no_empty_user_message():
    msgs = A.convert_messages([{
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}],
    }])
    assert [m["role"] for m in msgs] == ["tool"]


def test_mid_conversation_system_message_folds_into_the_user_turn():
    # Claude Code sends operator instructions as role:"system" entries *inside* messages.
    # Most chat templates only accept system in first position and raise otherwise, which
    # failed the whole request (caught on Ornith-1.0-9B; Qwen3 happened to tolerate it).
    msgs = A.convert_messages([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "do it"},
        {"role": "system", "content": "Terse mode enabled."},
    ], system="base prompt")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[0]["content"] == "base prompt"        # untouched
    assert "<system-reminder>" in msgs[-1]["content"]
    assert "Terse mode enabled." in msgs[-1]["content"]
    assert msgs[-1]["content"].startswith("do it")


def test_mid_conversation_system_message_before_a_user_turn():
    msgs = A.convert_messages([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "system", "content": "Now be terse."},
        {"role": "user", "content": [{"type": "text", "text": "go"}]},
    ])
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert "<system-reminder>" in msgs[-1]["content"]
    assert msgs[-1]["content"].endswith("go")


def test_leading_system_message_merges_into_the_system_prompt():
    msgs = A.convert_messages([
        {"role": "system", "content": "extra rules"},
        {"role": "user", "content": "hi"},
    ], system="base prompt")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == "base prompt\n\nextra rules"


def test_system_role_never_appears_after_the_first_message():
    # the invariant the templates actually enforce
    msgs = A.convert_messages(
        [{"role": "user", "content": "a"}, {"role": "system", "content": "s1"},
         {"role": "assistant", "content": "b"}, {"role": "system", "content": "s2"},
         {"role": "user", "content": "c"}],
        system="base")
    assert "system" not in [m["role"] for m in msgs[1:]]
    assert "s1" in msgs[1]["content"] and "s2" in msgs[-1]["content"]


def test_image_and_thinking_blocks_are_dropped():
    # text targets can't consume images, and we never emit thinking blocks, so echoing one
    # back would put a foreign trace in the prompt
    (m,) = A.convert_messages([{
        "role": "user",
        "content": [{"type": "image", "source": {"type": "base64", "data": "xx"}},
                    {"type": "thinking", "thinking": "hmm", "signature": "s"},
                    {"type": "text", "text": "describe"}],
    }])
    assert m["content"] == "describe"


def test_convert_tools_to_openai_shape_and_drops_beta_fields():
    out = A.convert_tools([
        {"name": "Bash", "description": "run", "input_schema": {"type": "object"},
         "strict": True, "defer_loading": True, "cache_control": {"type": "ephemeral"}},
        {"name": "NoSchema"},
        {"type": "web_search_20250305"},          # server tool: no name -> skipped
    ])
    assert [t["function"]["name"] for t in out] == ["Bash", "NoSchema"]
    assert out[0]["type"] == "function"
    assert out[0]["function"]["parameters"] == {"type": "object"}
    assert "strict" not in out[0] and "strict" not in out[0]["function"]
    assert out[1]["function"]["parameters"] == {"type": "object", "properties": {}}
    assert A.convert_tools(None) is None


def test_stop_reason_mapping():
    assert A.stop_reason("stop", False) == "end_turn"
    assert A.stop_reason("length", False) == "max_tokens"
    # a turn that produced tool calls always reports tool_use — that's what Claude Code
    # branches on to run the tools
    assert A.stop_reason("stop", True) == "tool_use"
    assert A.stop_reason("length", True) == "tool_use"


def test_tool_use_blocks_parse_arguments_and_survive_bad_json():
    (ok, bad) = A.tool_use_blocks([
        {"function": {"name": "f", "arguments": '{"a": 1}'}},
        {"function": {"name": "g", "arguments": "{not json"}},
    ])
    assert ok["type"] == "tool_use" and ok["name"] == "f" and ok["input"] == {"a": 1}
    assert ok["id"].startswith("toolu_")
    assert bad["input"] == {}          # degrade, never fail the turn


def test_build_message_text_only():
    msg = A.build_message("hello", model="m", input_tokens=3, output_tokens=2,
                          finish_reason="stop")
    assert msg["type"] == "message" and msg["role"] == "assistant"
    assert msg["content"] == [{"type": "text", "text": "hello"}]
    assert msg["stop_reason"] == "end_turn"
    assert msg["usage"] == {"input_tokens": 3, "output_tokens": 2}
    assert msg["id"].startswith("msg_")


def test_build_message_lifts_tool_calls_out_of_text():
    raw = 'ok<tool_call>{"name": "Read", "arguments": {"p": "/a"}}</tool_call>'
    msg = A.build_message(raw, model="m", input_tokens=1, output_tokens=1,
                          finish_reason="stop")
    assert msg["content"][0] == {"type": "text", "text": "ok"}
    assert msg["content"][1]["type"] == "tool_use"
    assert msg["content"][1]["name"] == "Read"
    assert msg["content"][1]["input"] == {"p": "/a"}
    assert msg["stop_reason"] == "tool_use"


def test_build_message_always_has_a_block():
    msg = A.build_message("", model="m", input_tokens=1, output_tokens=0, finish_reason="stop")
    assert msg["content"] == [{"type": "text", "text": ""}]


# --------------------------------------------------- XML tool-call form (Ornith-1.0 et al.)

ORNITH_CALL = (
    "<tool_call>\n<function=Edit>\n"
    "<parameter=file_path>\n/tmp/calc.py\n</parameter>\n"
    "<parameter=old_string>\ndef add(a, b):\n    return a - b\n</parameter>\n"
    "<parameter=replace_all>\ntrue\n</parameter>\n"
    "<parameter=limit>\n5\n</parameter>\n"
    "</function>\n</tool_call>"
)

EDIT_SCHEMA = [{"name": "Edit", "input_schema": {"type": "object", "properties": {
    "file_path": {"type": "string"}, "old_string": {"type": "string"},
    "replace_all": {"type": "boolean"}, "limit": {"type": "integer"}}}}]


def test_xml_tool_call_is_parsed_with_schema_types():
    from mlx_dspark.tools import parse_tool_calls, schema_types

    calls, cleaned = parse_tool_calls("I'll fix it.\n" + ORNITH_CALL,
                                      schema_types(EDIT_SCHEMA))
    assert cleaned == "I'll fix it."
    (c,) = calls
    args = json.loads(c["function"]["arguments"])
    assert c["function"]["name"] == "Edit"
    assert args["file_path"] == "/tmp/calc.py"
    # multi-line string preserved verbatim — coercing it would corrupt the edit
    assert args["old_string"] == "def add(a, b):\n    return a - b"
    assert args["replace_all"] is True          # declared boolean, not the string "true"
    assert args["limit"] == 5                   # declared integer


def test_xml_tool_call_without_schemas_keeps_multiline_values_as_strings():
    from mlx_dspark.tools import parse_tool_calls

    (c,), _ = parse_tool_calls(ORNITH_CALL)
    args = json.loads(c["function"]["arguments"])
    assert args["old_string"] == "def add(a, b):\n    return a - b"
    assert args["replace_all"] is True          # short single-line scalar: heuristic applies
    assert args["limit"] == 5


def test_a_numeric_looking_multiline_string_is_not_coerced():
    from mlx_dspark.tools import parse_tool_calls

    call = "<tool_call>\n<function=Write>\n<parameter=content>\n42\n\n43\n</parameter>\n" \
           "</function>\n</tool_call>"
    (c,), _ = parse_tool_calls(call)
    assert json.loads(c["function"]["arguments"])["content"] == "42\n\n43"


def test_schema_types_reads_both_tool_shapes():
    from mlx_dspark.tools import schema_types

    anthropic = schema_types(EDIT_SCHEMA)
    openai = schema_types([{"type": "function", "function": {
        "name": "Edit", "parameters": EDIT_SCHEMA[0]["input_schema"]}}])
    assert anthropic == openai
    assert anthropic["Edit"]["replace_all"] == "boolean"


def test_truncated_xml_call_does_not_leak_markup_as_prose():
    from mlx_dspark.tools import parse_tool_calls

    # cut off at max_tokens mid-call: closing </tool_call> never arrives
    partial = "Working on it.\n<tool_call>\n<function=Read>\n<parameter=file_path>\n/a\n" \
              "</parameter>\n</function>"
    calls, cleaned = parse_tool_calls(partial)
    assert cleaned == "Working on it."
    assert json.loads(calls[0]["function"]["arguments"])["file_path"] == "/a"
    # and one truncated before </function> yields no raw markup either
    _, cleaned2 = parse_tool_calls("Working.\n<tool_call>\n<function=Read>\n<parameter=fi")
    assert cleaned2 == "Working."


def test_xml_and_hermes_do_not_interfere():
    from mlx_dspark.tools import parse_tool_calls

    both = ORNITH_CALL + '\n<tool_call>{"name": "Bash", "arguments": {"cmd": "ls"}}</tool_call>'
    calls, cleaned = parse_tool_calls(both)
    assert [c["function"]["name"] for c in calls] == ["Edit", "Bash"]
    assert cleaned == ""


def test_stream_emits_an_xml_tool_call_as_a_tool_use_block():
    st = A.MessageStream(model="m", input_tokens=1,
                         schemas={"Edit": {"replace_all": "boolean"}})
    ev = list(st.start())
    ev += st.delta("On it.")
    ev += st.delta(ORNITH_CALL)
    ev += st.finish(finish_reason="stop", output_tokens=1)
    assert _assert_well_formed(ev) == {0: "text", 1: "tool_use"}
    assert _streamed_text(ev) == "On it."
    (js,) = [d["delta"]["partial_json"] for n, d in ev
             if n == "content_block_delta" and d["delta"]["type"] == "input_json_delta"]
    assert json.loads(js)["replace_all"] is True


# --------------------------------------------------------------------------- tool gate


def test_gate_holds_back_a_possible_marker_prefix():
    g = A._ToolGate()
    emitted = ""
    for piece in ("hello there ", "world", "!"):
        emitted += g.feed(piece)
        # never runs ahead of the input, and withholds only enough tail to catch a marker
        # that straddles two rounds
        assert g.buf.startswith(emitted)
        assert len(g.buf) - len(emitted) < A._MAX_MARKER
    assert not g.tripped
    assert g.buf == "hello there world!"


def test_gate_trips_on_marker_split_across_chunks():
    g = A._ToolGate()
    g.feed("call now<tool_")
    assert not g.tripped
    out = g.feed('call>{"name": "f"}')
    assert g.tripped
    assert g.sent_text == "call now"              # never leaks the marker itself
    assert "<tool" not in out


def test_gate_stays_tripped_and_emits_nothing_further():
    g = A._ToolGate()
    g.feed('x<tool_call>{"name": "f", "arguments": {}}')
    assert g.feed("</tool_call>") == ""
    assert g.sent_text == "x"


def test_gate_recognises_the_gemma_marker():
    g = A._ToolGate()
    g.feed("sure <|tool_call>call:f{}")
    assert g.tripped
    assert g.sent_text == "sure "


# --------------------------------------------------------------------------- stream builder


def _drive(text_pieces, finish_reason="stop", output_tokens=5):
    st = A.MessageStream(model="m", input_tokens=7)
    events = list(st.start())
    for p in text_pieces:
        events.extend(st.delta(p))
    events.extend(st.finish(finish_reason=finish_reason, output_tokens=output_tokens))
    return events


def _names(events):
    return [n for n, _ in events]


def _streamed_text(events):
    return "".join(d["delta"]["text"] for n, d in events
                   if n == "content_block_delta" and d["delta"]["type"] == "text_delta")


def _assert_well_formed(events):
    """The structural contract a client's accumulator relies on: the message brackets
    everything, every delta lands inside an open block of a declared type, every block that
    opens also closes, and indices are handed out 0..n-1 in opening order."""
    assert _names(events)[0] == "message_start"
    assert _names(events)[-1] == "message_stop"
    assert _names(events)[-2] == "message_delta"
    open_blocks, kinds, seen = {}, {}, []
    for name, d in events:
        if name == "content_block_start":
            assert d["index"] not in open_blocks, "index reused while open"
            assert d["index"] not in seen, "index reused after close"
            open_blocks[d["index"]] = True
            kinds[d["index"]] = d["content_block"]["type"]
            seen.append(d["index"])
        elif name == "content_block_delta":
            assert open_blocks.get(d["index"]), "delta outside an open block"
            # the delta variant must match the block type it lands in
            allowed = {"text": {"text_delta"},
                       "thinking": {"thinking_delta", "signature_delta"},
                       "tool_use": {"input_json_delta"}}[kinds[d["index"]]]
            assert d["delta"]["type"] in allowed
        elif name == "content_block_stop":
            assert open_blocks.pop(d["index"], False), "stop without start"
    assert not open_blocks, "block left open"
    assert seen == list(range(len(seen))), f"non-sequential indices: {seen}"
    return kinds


def test_stream_event_order_for_plain_text():
    ev = _drive(["Hello ", "world", "!"])
    assert _assert_well_formed(ev) == {0: "text"}
    start = ev[0][1]["message"]
    assert start["role"] == "assistant" and start["content"] == []
    assert start["usage"]["input_tokens"] == 7


def test_stream_delivers_every_character_exactly_once():
    ev = _drive(["Hello ", "world", "!"])
    assert _streamed_text(ev) == "Hello world!"


def test_stream_reports_usage_and_stop_reason():
    ev = _drive(["hi"], finish_reason="length", output_tokens=11)
    (delta,) = [d for n, d in ev if n == "message_delta"]
    assert delta["delta"]["stop_reason"] == "max_tokens"
    assert delta["usage"]["output_tokens"] == 11


def test_stream_emits_tool_use_blocks_after_the_text_block():
    ev = _drive(["I will read it.", '<tool_call>{"name": "Read", ',
                 '"arguments": {"file_path": "/tmp/a"}}</tool_call>'])
    assert _streamed_text(ev) == "I will read it."
    assert _assert_well_formed(ev) == {0: "text", 1: "tool_use"}
    starts = [d for n, d in ev if n == "content_block_start"]
    assert starts[1]["content_block"]["name"] == "Read"
    (js,) = [d["delta"]["partial_json"] for n, d in ev
             if n == "content_block_delta" and d["delta"]["type"] == "input_json_delta"]
    assert json.loads(js) == {"file_path": "/tmp/a"}
    (delta,) = [d for n, d in ev if n == "message_delta"]
    assert delta["delta"]["stop_reason"] == "tool_use"


def test_stream_never_leaks_the_native_tool_syntax_as_text():
    ev = _drive(['<tool_call>{"name": "f", "arguments": {}}</tool_call>'])
    assert _streamed_text(ev) == ""
    assert any(n == "content_block_start" and d["content_block"]["type"] == "tool_use"
               for n, d in ev)


def test_stream_indices_are_unique_and_ordered():
    ev = _drive(['<tool_call>{"name": "a", "arguments": {}}</tool_call>'
                 '<tool_call>{"name": "b", "arguments": {}}</tool_call>'])
    assert _assert_well_formed(ev) == {0: "text", 1: "tool_use", 2: "tool_use"}


def test_stream_always_produces_at_least_one_block():
    ev = _drive([])
    assert _assert_well_formed(ev) == {0: "text"}


# --------------------------------------------------------------------------- thinking


def test_split_thinking_self_opened_and_prefilled_and_absent():
    assert A.split_thinking("<think>reasoning</think>\n\nanswer") == ("reasoning", "answer")
    # some chat templates prefill the opener, so the output only closes it
    assert A.split_thinking("reasoning</think>\n\nanswer") == ("reasoning", "answer")
    assert A.split_thinking("just an answer") == ("", "just an answer")
    # truncated mid-thought: it's all reasoning, and no stray markup escapes
    assert A.split_thinking("<think>cut off") == ("cut off", "")


def test_prompt_opens_thinking_returns_the_matching_closer():
    assert A.prompt_opens_thinking("<|im_start|>assistant\n<think>\n") == "</think>"
    assert A.prompt_opens_thinking("<|im_start|>assistant\n") is None
    # Gemma-4 prefills the whole *empty* pair when thinking is off, so the prompt ends on the
    # closer — that must not be read as an open block
    assert A.prompt_opens_thinking("<|turn>model\n<|channel>thought\n<channel|>") is None


def test_gemma4_channel_reasoning_is_split_out():
    # Gemma-4's own response grammar (tokenizer_config response_schema.x-regex):
    #   (<|channel>thought\n THINKING <channel|>)? TOOLCALLS? CONTENT
    # It only emits these markers itself after a tool response, which is why the leak showed
    # up in a Claude Code session and not in plain chat.
    raw = "<|channel>thought\nweighing it\n<channel|>The bug is fixed."
    assert A.split_thinking(raw) == ("weighing it\n", "The bug is fixed.")
    msg = A.build_message(raw, model="m", input_tokens=1, output_tokens=1,
                          finish_reason="stop")
    assert [b["type"] for b in msg["content"]] == ["thinking", "text"]
    assert msg["content"][1]["text"] == "The bug is fixed."
    assert "<|channel>" not in json.dumps(msg) and "<channel|>" not in json.dumps(msg)


def test_gemma4_empty_thought_channel_yields_only_text():
    raw = "<|channel>thought\n<channel|>Hello there!"
    msg = A.build_message(raw, model="m", input_tokens=1, output_tokens=1,
                          finish_reason="stop")
    assert msg["content"] == [{"type": "text", "text": "Hello there!"}]


def test_stream_gemma4_channel_reasoning():
    ev = _drive(["<|channel>thou", "ght\nlet me check", "\n<channel|>", "Fixed it."])
    assert _assert_well_formed(ev) == {0: "thinking", 1: "text"}
    assert _streamed_text(ev) == "Fixed it."
    think = "".join(d["delta"]["thinking"] for n, d in ev
                    if n == "content_block_delta" and d["delta"]["type"] == "thinking_delta")
    assert think.strip() == "let me check"


def test_build_message_lifts_reasoning_into_a_thinking_block():
    msg = A.build_message("<think>let me see</think>\n\nPONG", model="m", input_tokens=1,
                          output_tokens=1, finish_reason="stop")
    assert msg["content"][0] == {"type": "thinking", "thinking": "let me see",
                                 "signature": A._SIGNATURE}
    assert msg["content"][1] == {"type": "text", "text": "PONG"}
    # the raw markup must never surface as assistant prose
    assert "<think>" not in json.dumps(msg)


def test_build_message_drops_reasoning_when_thinking_is_disabled():
    msg = A.build_message("<think>hidden</think>\n\nPONG", model="m", input_tokens=1,
                          output_tokens=1, finish_reason="stop", thinking=False)
    assert msg["content"] == [{"type": "text", "text": "PONG"}]


def test_stream_reasoning_as_a_thinking_block():
    ev = _drive(["<thi", "nk>step ", "one</thi", "nk>\n\nThe ", "answer"])
    assert _assert_well_formed(ev) == {0: "thinking", 1: "text"}
    think = "".join(d["delta"]["thinking"] for n, d in ev
                    if n == "content_block_delta" and d["delta"]["type"] == "thinking_delta")
    assert think == "step one"
    assert _streamed_text(ev) == "The answer"
    # the real API signs the block just before closing it
    sig = [d for n, d in ev
           if n == "content_block_delta" and d["delta"]["type"] == "signature_delta"]
    assert len(sig) == 1 and sig[0]["index"] == 0


def test_stream_never_leaks_think_markup_as_text():
    ev = _drive(["<think>secret</think>answer"])
    assert "<think>" not in _streamed_text(ev)
    assert "</think>" not in _streamed_text(ev)


def test_stream_suppresses_thinking_when_disabled_and_text_stays_index_zero():
    st = A.MessageStream(model="m", input_tokens=1, thinking=False)
    ev = list(st.start())
    for p in ["<think>hidden</think>", "visible"]:
        ev += st.delta(p)
    ev += st.finish(finish_reason="stop", output_tokens=1)
    assert _assert_well_formed(ev) == {0: "text"}   # text is block 0, not 1
    assert _streamed_text(ev) == "visible"
    assert not [1 for n, d in ev
                if n == "content_block_delta" and d["delta"]["type"] == "thinking_delta"]


def test_stream_handles_a_prefilled_thinking_opener():
    # template already opened <think>, so the output only ever closes it
    st = A.MessageStream(model="m", input_tokens=1, in_thinking="</think>")
    ev = list(st.start())
    for p in ["weighing it", "</think>", "done"]:
        ev += st.delta(p)
    ev += st.finish(finish_reason="stop", output_tokens=1)
    assert _assert_well_formed(ev) == {0: "thinking", 1: "text"}
    assert _streamed_text(ev) == "done"


def test_stream_unterminated_thinking_block_still_closes():
    st = A.MessageStream(model="m", input_tokens=1)
    ev = list(st.start()) + st.delta("<think>ran out of tokens")
    ev += st.finish(finish_reason="length", output_tokens=1)
    assert _assert_well_formed(ev) == {0: "thinking"}
    (delta,) = [d for n, d in ev if n == "message_delta"]
    assert delta["delta"]["stop_reason"] == "max_tokens"


def test_stream_reasoning_then_tool_call():
    ev = _drive(["<think>need to read</think>",
                 '<tool_call>{"name": "Read", "arguments": {"p": "/a"}}</tool_call>'])
    assert _assert_well_formed(ev) == {0: "thinking", 1: "text", 2: "tool_use"}
    (delta,) = [d for n, d in ev if n == "message_delta"]
    assert delta["delta"]["stop_reason"] == "tool_use"


# --------------------------------------------------------------------------- HTTP surface


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
    is_muse = False           # mirrors Engine.is_muse (muse_glimmer channel parsing off)

    def __init__(self, response_text="Hello world"):
        self.tokenizer = _FakeTok()
        self.calls = []
        self.response_text = response_text

    def generate(self, prompt_ids, *, max_tokens, temperature, top_p=1.0, top_k=0,
                 presence_penalty=0.0, frequency_penalty=0.0, logprobs=None,
                 stop=None, seed=None, on_text=None):
        self.calls.append({"prompt_ids": prompt_ids, "max_tokens": max_tokens,
                               "temperature": temperature, "top_p": top_p, "top_k": top_k,
                               "stop": stop, "seed": seed})
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


def _post(base, path, body, headers=None, raw=False):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **(headers or {})})
    with urllib.request.urlopen(req, timeout=10) as r:
        text = r.read().decode()
    return text if raw else json.loads(text)


def _read_sse(body: str):
    """[(event_name, data)] from an SSE body."""
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


USER = [{"role": "user", "content": "hi"}]


def test_messages_non_streaming(api):
    _, base = api
    r = _post(base, "/v1/messages", {"model": "claude-sonnet-4-6", "max_tokens": 100,
                                     "messages": USER})
    assert r["type"] == "message" and r["role"] == "assistant"
    assert r["content"] == [{"type": "text", "text": "Hello world"}]
    assert r["stop_reason"] == "end_turn"
    assert r["model"] == "claude-sonnet-4-6"      # echoed back, as a gateway would
    assert r["usage"]["output_tokens"] == 3


def test_messages_accepts_the_beta_query_claude_code_sends(api):
    # Claude Code posts to /v1/messages?beta=true — routing on the raw path would 404
    _, base = api
    r = _post(base, "/v1/messages?beta=true", {"max_tokens": 10, "messages": USER})
    assert r["type"] == "message"


def test_messages_streaming_event_sequence(api):
    _, base = api
    body = _post(base, "/v1/messages", {"max_tokens": 50, "messages": USER, "stream": True},
                 raw=True)
    ev = _read_sse(body)
    names = [n for n, _ in ev]
    assert names[0] == "message_start"
    assert names[-1] == "message_stop"
    assert "content_block_start" in names and "message_delta" in names
    # every frame is a *named* event (Anthropic's stream is not OpenAI's unnamed one)
    assert all(n for n, _ in ev)
    # and it must not be terminated with OpenAI's sentinel
    assert "[DONE]" not in body
    text = "".join(d["delta"]["text"] for n, d in ev
                   if n == "content_block_delta" and d["delta"]["type"] == "text_delta")
    assert text == "Hello world"


def test_messages_streaming_tool_call():
    eng = _FakeEngine('sure<tool_call>{"name": "Bash", "arguments": {"cmd": "ls"}}</tool_call>')
    httpd, base = _serve(eng)
    try:
        body = _post(base, "/v1/messages", {"max_tokens": 50, "messages": USER,
                                            "stream": True}, raw=True)
    finally:
        httpd.shutdown()
        httpd.server_close()
    ev = _read_sse(body)
    tool = [d for n, d in ev
            if n == "content_block_start" and d["content_block"]["type"] == "tool_use"]
    assert len(tool) == 1 and tool[0]["content_block"]["name"] == "Bash"
    (delta,) = [d for n, d in ev if n == "message_delta"]
    assert delta["delta"]["stop_reason"] == "tool_use"


def test_history_with_tool_use_and_result_reaches_the_model(api):
    eng, base = api
    _post(base, "/v1/messages", {"max_tokens": 10, "messages": [
        {"role": "user", "content": "read it"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"p": "/a"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "CONTENTS"}]},
    ]})
    # the fake tokenizer is 1 id per char, so the prompt text is recoverable
    prompt = "".join(chr(i) for i in eng.calls[-1]["prompt_ids"])
    assert "read it" in prompt and "CONTENTS" in prompt


def test_tools_are_passed_to_the_chat_template(api):
    eng, base = api
    calls = []
    eng.tokenizer.chat_template = "x"
    eng.tokenizer.apply_chat_template = lambda msgs, **kw: (
        calls.append(kw) or [1, 2, 3])
    _post(base, "/v1/messages", {"max_tokens": 10, "messages": USER, "tools": [
        {"name": "Bash", "description": "run", "input_schema": {"type": "object"}}]})
    assert calls[-1]["tools"][0]["function"]["name"] == "Bash"


def test_count_tokens(api):
    _, base = api
    r = _post(base, "/v1/messages/count_tokens",
              {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "abcd"}]})
    assert r == {"input_tokens": 4}      # 1 id per char in the fake tokenizer


def test_count_tokens_counts_the_system_prompt_too(api):
    _, base = api
    a = _post(base, "/v1/messages/count_tokens", {"messages": USER})["input_tokens"]
    b = _post(base, "/v1/messages/count_tokens",
              {"messages": USER, "system": "longer system prompt"})["input_tokens"]
    assert b > a


@pytest.mark.parametrize("field,value", [
    ("thinking", {"type": "adaptive"}),          # sent for any model name it doesn't know
    ("context_management", {"edits": [{"type": "clear_tool_uses_20250919"}]}),
    ("output_config", {"effort": "high"}),
    ("metadata", {"user_id": "u"}),
    ("tool_choice", {"type": "auto"}),
    ("service_tier", "auto"),
    ("some_capability_from_a_future_release", {"nested": [1, 2]}),
])
def test_unknown_request_fields_never_fail(api, field, value):
    # Claude Code's field set grows every release and it sends the newest fields to any
    # endpoint whose model name it doesn't recognise — which is every local server.
    _, base = api
    r = _post(base, "/v1/messages", {"max_tokens": 10, "messages": USER, field: value})
    assert r["type"] == "message"


def test_stop_sequences_reach_the_engine(api):
    eng, base = api
    _post(base, "/v1/messages", {"max_tokens": 10, "messages": USER,
                                 "stop_sequences": ["\n\nHuman:"]})
    assert eng.calls[-1]["stop"] == ["\n\nHuman:"]


def test_max_tokens_is_clamped_to_the_server_cap(api):
    eng, base = api
    _post(base, "/v1/messages", {"max_tokens": 10 ** 9, "messages": USER})
    assert eng.calls[-1]["max_tokens"] == eng.max_tokens_cap


def test_model_generation_config_fills_absent_top_p(api):
    # Claude Code sends temperature but never top_p/top_k; the model's own nucleus settings
    # are what keep a temperature-1 agent request sane.
    eng, base = api
    eng.sampling_defaults = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
    _post(base, "/v1/messages", {"max_tokens": 10, "messages": USER, "temperature": 1.0})
    c = eng.calls[-1]
    assert c["temperature"] == 1.0 and c["top_p"] == 0.95 and c["top_k"] == 20


def _expect_error(base, path, body, headers=None):
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base, path, body, headers)
    return e.value.code, json.loads(e.value.read().decode())


def test_error_uses_the_anthropic_envelope(api):
    _, base = api
    code, body = _expect_error(base, "/v1/messages", {"max_tokens": 10, "messages": []})
    assert code == 400
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert "messages" in body["error"]["message"]


def test_context_overflow_uses_the_wording_claude_code_recovers_from():
    # Claude Code's automatic compact-and-retry matches on "prompt is too long"; rewording
    # this turns a recoverable session into a dead one.
    eng = _FakeEngine()
    eng.context_window = 8
    httpd, base = _serve(eng)
    try:
        code, body = _expect_error(base, "/v1/messages",
                                   {"max_tokens": 10,
                                    "messages": [{"role": "user", "content": "x" * 50}]})
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert code == 400
    assert body["error"]["message"].startswith("prompt is too long")
    assert eng.calls == []                # rejected before touching the model


def test_auth_accepts_both_credential_headers():
    # ANTHROPIC_AUTH_TOKEN -> Authorization: Bearer; ANTHROPIC_API_KEY / apiKeyHelper -> x-api-key
    eng = _FakeEngine()
    httpd, base = _serve(eng, api_key="secret")
    try:
        body = {"max_tokens": 10, "messages": USER}
        assert _post(base, "/v1/messages", body,
                     {"Authorization": "Bearer secret"})["type"] == "message"
        assert _post(base, "/v1/messages", body,
                     {"x-api-key": "secret"})["type"] == "message"
        code, err = _expect_error(base, "/v1/messages", body, {"x-api-key": "wrong"})
        assert code == 401 and err["type"] == "error"
        assert err["error"]["type"] == "authentication_error"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_head_probe_and_health_advertise_the_model(api):
    _, base = api
    req = urllib.request.Request(base + "/", method="HEAD")
    with urllib.request.urlopen(req, timeout=5) as r:
        assert r.status == 200                      # Claude Code's startup connectivity probe
    with urllib.request.urlopen(base + "/health", timeout=5) as r:
        h = json.loads(r.read().decode())
    assert h["model"] == "FakeModel"
    assert "context_window" in h and "max_output_tokens" in h


def test_models_endpoint_carries_a_display_name(api):
    _, base = api
    with urllib.request.urlopen(base + "/v1/models", timeout=5) as r:
        data = json.loads(r.read().decode())["data"]
    assert data[0]["id"] == "FakeModel"
    assert "mlx-dspark" in data[0]["display_name"]


def test_openai_route_still_works_alongside(api):
    # the Anthropic surface is additive: existing OpenAI clients are untouched
    _, base = api
    r = _post(base, "/v1/chat/completions", {"messages": USER, "max_tokens": 10})
    assert r["object"] == "chat.completion"


# --------------------------------------------------------------------------- launcher


HEALTH = {"status": "ok", "model": "Qwen3-8B-8bit", "mode": "dspark",
          "context_window": 40960, "max_output_tokens": 32768}


def test_claude_env_points_the_session_at_the_server(monkeypatch):
    from mlx_dspark.cli import claude_env

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-real-account-key")
    env = claude_env("http://127.0.0.1:8080", HEALTH, None)
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8080"
    # a credential variable is required: with only a base URL, Claude Code keeps
    # authenticating with the saved claude.ai login and bills the subscription
    assert env["ANTHROPIC_AUTH_TOKEN"] == "mlx-dspark"
    # and the real account key must not shadow it in the child
    assert "ANTHROPIC_API_KEY" not in env


def test_claude_env_maps_every_alias_including_the_background_model():
    from mlx_dspark.cli import claude_env

    env = claude_env("http://h:1", HEALTH, "secret")
    assert env["ANTHROPIC_AUTH_TOKEN"] == "secret"
    assert env["ANTHROPIC_MODEL"] == "Qwen3-8B-8bit"
    for alias in ("OPUS", "SONNET", "HAIKU", "FABLE"):
        assert env[f"ANTHROPIC_DEFAULT_{alias}_MODEL"] == "Qwen3-8B-8bit"
        assert "mlx-dspark" in env[f"ANTHROPIC_DEFAULT_{alias}_MODEL_NAME"]
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "32768"


def test_claude_env_drops_conflicting_provider_selection(monkeypatch):
    from mlx_dspark.cli import claude_env

    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    env = claude_env("http://h:1", HEALTH, None)
    assert "CLAUDE_CODE_USE_BEDROCK" not in env


def test_claude_env_skips_an_auto_compact_window_claude_code_would_clamp():
    from mlx_dspark.cli import claude_env

    # Claude Code clamps this to >=100k, so setting it under that is a no-op; small windows
    # are handled by the server's "prompt is too long" reply instead.
    small = claude_env("http://h:1", {**HEALTH, "context_window": 32768}, None)
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW" not in small
    big = claude_env("http://h:1", {**HEALTH, "context_window": 200_000}, None)
    assert int(big["CLAUDE_CODE_AUTO_COMPACT_WINDOW"]) == 160_000


# ------------------------------------------------------- muse_glimmer "Onyx ATEM" channels

# muse emits recipient-tagged sub-messages `[<|start|>]HEADER<|message|>BODY<|eom|>`, where
# `to=self` is the analysis channel and `to=user` (or a tool name) the final one. These strings
# are the real shapes captured from the 8-bit target (math reasoning -> answer, and a tool call).
_MUSE_MATH = (" to=self<|message|>17*23 = 391. Provide answer.<|eom|>"
              "<|start|>assistant to=user<|message|>17 × 23 = **391**.")
_MUSE_TOOL = (" to=self<|message|>I should look it up.<|eom|>"
              "<|start|>assistant to=get_weather<|message|><atem:function_calls>\n"
              '<atem:invoke name="get_weather">\n'
              '<atem:parameter name="city">Paris</atem:parameter>\n'
              "</atem:invoke>\n</atem:function_calls>")


def test_split_thinking_muse_reasoning_then_answer():
    assert A.split_thinking(_MUSE_MATH) == ("17*23 = 391. Provide answer.", "17 × 23 = **391**.")


def test_split_thinking_muse_answer_only():
    # a response with no analysis channel goes straight to `to=user`
    assert A.split_thinking(" to=user<|message|>Hello there.") == ("", "Hello there.")


def test_split_thinking_muse_unterminated_reasoning_is_all_reasoning():
    # muse reasons by default and can ramble past max_tokens without ever reaching the answer;
    # that must read as all-thinking, with no stray markers leaking
    assert A.split_thinking(" to=self<|message|>still thinking") == ("still thinking", "")


def test_parse_atem_tool_call():
    from mlx_dspark.tools import parse_tool_calls

    _, answer = A.split_thinking(_MUSE_TOOL)
    calls, cleaned = parse_tool_calls(answer)
    assert cleaned == ""
    assert len(calls) == 1 and calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}


def test_parse_atem_types_values_as_json_with_string_fallback():
    from mlx_dspark.tools import parse_tool_calls

    body = ('<atem:invoke name="f">'
            '<atem:parameter name="n">3</atem:parameter>'
            '<atem:parameter name="flag">true</atem:parameter>'
            '<atem:parameter name="who">Paris</atem:parameter>'
            "</atem:invoke>")
    calls, _ = parse_tool_calls(body)
    assert json.loads(calls[0]["function"]["arguments"]) == {"n": 3, "flag": True, "who": "Paris"}


def test_parse_atem_truncated_call_does_not_leak_markup():
    from mlx_dspark.tools import parse_tool_calls

    # cut off at max_tokens mid-call: the unclosed invoke is an aborted call, not prose
    calls, cleaned = parse_tool_calls('<atem:function_calls>\n<atem:invoke name="f">\n'
                                      '<atem:parameter name="x">1')
    assert cleaned == "" and calls and calls[0]["function"]["name"] == "f"


def test_build_message_muse_reasoning_and_tool_call():
    msg = A.build_message(_MUSE_TOOL, model="m", input_tokens=1, output_tokens=1,
                          finish_reason="stop")
    assert [b["type"] for b in msg["content"]] == ["thinking", "tool_use"]
    assert msg["content"][0]["thinking"] == "I should look it up."
    assert msg["content"][1]["name"] == "get_weather"
    assert msg["stop_reason"] == "tool_use"


def _drive_muse(pieces, *, thinking=True, finish_reason="stop"):
    st = A.MessageStream(model="m", input_tokens=7, thinking=thinking, muse=True)
    events = list(st.start())
    for p in pieces:
        events.extend(st.delta(p))
    events.extend(st.finish(finish_reason=finish_reason, output_tokens=5))
    return events


def _thinking_text(events):
    return "".join(d["delta"]["thinking"] for n, d in events
                   if n == "content_block_delta" and d["delta"]["type"] == "thinking_delta")


def test_muse_stream_reasoning_becomes_thinking_then_answer_text():
    ev = _drive_muse([_MUSE_MATH])
    assert _assert_well_formed(ev) == {0: "thinking", 1: "text"}
    assert _thinking_text(ev) == "17*23 = 391. Provide answer."
    assert _streamed_text(ev) == "17 × 23 = **391**."


def test_muse_stream_markers_split_across_chunks():
    # the structural special tokens can straddle a round boundary; the parser must still
    # recover the channels and never leak a marker as prose
    pieces = [" to=se", "lf<|mess", "age|>weighing<|e", "om|><|start|>assistant to=u",
              "ser<|message|>Answer here."]
    ev = _drive_muse(pieces)
    assert _assert_well_formed(ev) == {0: "thinking", 1: "text"}
    assert _thinking_text(ev) == "weighing"
    assert _streamed_text(ev) == "Answer here."


def test_muse_stream_emits_atem_tool_call_and_never_leaks_it():
    ev = _drive_muse([_MUSE_TOOL])
    kinds = _assert_well_formed(ev)
    assert kinds[max(kinds)] == "tool_use"
    assert "<atem:" not in _streamed_text(ev)          # tool XML never streamed as prose
    starts = [d for n, d in ev if n == "content_block_start"]
    tool = next(d for d in starts if d["content_block"]["type"] == "tool_use")
    assert tool["content_block"]["name"] == "get_weather"
    (delta,) = [d for n, d in ev if n == "message_delta"]
    assert delta["delta"]["stop_reason"] == "tool_use"


def test_muse_stream_thinking_disabled_drops_the_analysis_channel():
    ev = _drive_muse([_MUSE_MATH], thinking=False)
    assert _assert_well_formed(ev) == {0: "text"}      # no thinking block published
    assert _thinking_text(ev) == ""
    assert _streamed_text(ev) == "17 × 23 = **391**."


def test_muse_channel_parser_is_char_by_char_incremental():
    # feeding one character at a time must yield the same channels as one whole feed
    whole = A.MuseChannelParser().feed(_MUSE_MATH, final=True)
    p = A.MuseChannelParser()
    chunks = []
    for c in _MUSE_MATH:
        chunks += p.feed(c)
    chunks += p.feed("", final=True)
    join = lambda cs, k: "".join(t for kk, t in cs if kk == k)  # noqa: E731
    for kind in ("reasoning", "answer"):
        assert join(chunks, kind) == join(whole, kind)


def _muse_engine(text):
    eng = _FakeEngine(text)
    eng.is_muse = True                 # engage the muse_glimmer channel parser server-side
    return eng


def test_messages_streaming_muse_splits_channels_into_blocks():
    eng = _muse_engine(_MUSE_MATH)
    httpd, base = _serve(eng)
    try:
        body = _post(base, "/v1/messages", {"max_tokens": 50, "messages": USER,
                                            "stream": True}, raw=True)
    finally:
        httpd.shutdown()
        httpd.server_close()
    ev = _read_sse(body)
    thinking = "".join(d["delta"]["thinking"] for n, d in ev
                       if n == "content_block_delta" and d["delta"]["type"] == "thinking_delta")
    text = "".join(d["delta"]["text"] for n, d in ev
                   if n == "content_block_delta" and d["delta"]["type"] == "text_delta")
    assert thinking == "17*23 = 391. Provide answer."
    assert text == "17 × 23 = **391**."
    assert "<|message|>" not in body and "to=self" not in body   # no raw markers leak


def _openai_stream_deltas(body):
    return [json.loads(line[6:]) for line in body.splitlines()
            if line.startswith("data: ") and line[6:].strip() != "[DONE]"]


def test_openai_streaming_muse_splits_reasoning_from_content():
    eng = _muse_engine(_MUSE_MATH)
    httpd, base = _serve(eng)
    try:
        body = _post(base, "/v1/chat/completions", {"model": "m", "messages": USER,
                                                    "stream": True}, raw=True)
    finally:
        httpd.shutdown()
        httpd.server_close()
    deltas = [c["choices"][0]["delta"] for c in _openai_stream_deltas(body)]
    reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
    content = "".join(d.get("content", "") for d in deltas)
    assert reasoning == "17*23 = 391. Provide answer."
    assert content == "17 × 23 = **391**."
    assert "<|message|>" not in body


def test_openai_nonstream_muse_puts_reasoning_in_reasoning_content():
    eng = _muse_engine(_MUSE_MATH)
    httpd, base = _serve(eng)
    try:
        r = _post(base, "/v1/chat/completions", {"model": "m", "messages": USER})
    finally:
        httpd.shutdown()
        httpd.server_close()
    msg = r["choices"][0]["message"]
    assert msg["reasoning_content"] == "17*23 = 391. Provide answer."
    assert msg["content"] == "17 × 23 = **391**."
