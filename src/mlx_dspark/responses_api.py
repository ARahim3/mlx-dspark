"""OpenAI Responses API (``/v1/responses``) — the dialect **Codex** speaks.

Codex posts to ``/v1/responses`` instead of ``/v1/chat/completions`` once its provider is
configured with ``wire_api = "responses"``, so an OpenAI Chat-Completions-only server is
invisible to it (the same gap ``anthropic_api.py`` closes for Claude Code). This module is
the translation layer: Responses request -> the OpenAI-shaped message list the rest of
mlx-dspark already speaks (:func:`~mlx_dspark.generate.encode_messages` +
:func:`~mlx_dspark.tools.normalize_tool_messages`), and generated text/tool-calls ->
Responses output items / SSE events.

Pure and model-free (no MLX, no weights), same as ``anthropic_api.py`` — ``server.py`` owns
the HTTP plumbing.

Scope, deliberately: this is a *stateless single-turn* implementation, matching what Ollama
ships (its own Responses support is documented as "non-stateful" — no ``previous_response_id``
/ server-side conversation store). A client resubmits its own history every request (Codex
does exactly this), so there's nothing to persist. Reasoning is captured in the non-streaming
body (split out the same way the Chat Completions path already splits ``reasoning_content``)
but is not surfaced as incremental streaming events — a streaming response only emits the
answer text and any tool calls, which is what a client whose model is running with
``enable_thinking=False`` (this server's Qwen3.8 default) actually produces anyway.
"""

from __future__ import annotations

import json
import time
import uuid


def _resp_id() -> str:
    return "resp_" + uuid.uuid4().hex


def _item_id(prefix: str) -> str:
    return f"{prefix}_" + uuid.uuid4().hex


def error_body(message: str, etype: str = "invalid_request_error") -> dict:
    return {"error": {"message": message, "type": etype, "code": None, "param": None}}


def _flatten_input_text(content) -> str:
    """A Responses ``content`` value -> plain text. Content is either a bare string or a list
    of typed parts (``input_text``/``output_text``/``input_image``/``input_file``); join the
    text parts and drop the rest — mirrors ``tools._flatten_content`` for the Chat dialect."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for p in content:
        if isinstance(p, str):
            parts.append(p)
        elif isinstance(p, dict) and isinstance(p.get("text"), str):
            parts.append(p["text"])
    return "\n".join(parts)


def convert_input(input_, instructions=None) -> list[dict]:
    """Responses ``input`` (+ top-level ``instructions``) -> an OpenAI chat ``messages`` list.

    ``input`` is either a bare string (shorthand for one user turn) or a list of items. Two
    item shapes matter here beyond a plain message: ``function_call`` and
    ``function_call_output`` — the tool-use round trip a client replays each request since
    this server holds no state between calls (see module docstring). Anything else (bare
    ``reasoning`` items a previous response emitted, etc.) is server-side context a stateless
    backend has no use for on replay, so it's skipped rather than rejected — an unrecognized
    item type must not 400, the same policy ``anthropic_api`` follows for unknown fields.
    """
    messages: list[dict] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    if isinstance(input_, str):
        messages.append({"role": "user", "content": input_})
        return messages
    if not isinstance(input_, list):
        return messages
    for item in input_:
        if not isinstance(item, dict):
            continue
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role", "user")
            messages.append({"role": role, "content": _flatten_input_text(item.get("content"))})
        elif kind == "function_call":
            args = item.get("arguments", "{}")
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            messages.append({
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": item.get("call_id") or item.get("id") or _item_id("call"),
                    "type": "function",
                    "function": {"name": item.get("name", ""), "arguments": args},
                }],
            })
        elif kind == "function_call_output":
            out = item.get("output", "")
            if not isinstance(out, str):
                out = json.dumps(out, ensure_ascii=False)
            messages.append({"role": "tool", "tool_call_id": item.get("call_id", ""),
                             "content": out})
        # else: reasoning items and anything future — skip, see docstring
    return messages


def convert_tools(tools) -> list[dict] | None:
    """Responses tool defs (flat: ``{"type":"function","name","description","parameters"}``)
    -> Chat Completions shape (nested under ``function``), which is what the rest of the
    server (``schema_types``, the chat template's tool rendering) already expects."""
    if not tools:
        return None
    out = []
    for t in tools:
        if not isinstance(t, dict) or t.get("type") != "function":
            continue
        if "function" in t:      # already nested; accept as-is
            out.append(t)
            continue
        out.append({"type": "function", "function": {
            "name": t.get("name"), "description": t.get("description", ""),
            "parameters": t.get("parameters") or {"type": "object", "properties": {}},
        }})
    return out or None


def _output_text_item(item_id: str, text: str) -> dict:
    return {"id": item_id, "type": "message", "status": "completed", "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


def _reasoning_item(text: str) -> dict:
    return {"id": _item_id("rs"), "type": "reasoning",
            "summary": [{"type": "summary_text", "text": text}]}


def _function_call_item(call_id: str, name: str, arguments: str, *, status="completed") -> dict:
    return {"id": _item_id("fc"), "type": "function_call", "status": status,
            "call_id": call_id, "name": name, "arguments": arguments}


def build_response(*, resp_id: str, model: str, content: str, reasoning: str = "",
                   tool_calls: list[dict] | None = None, input_tokens: int, output_tokens: int,
                   finish_reason: str, created: int | None = None) -> dict:
    """Non-streaming ``/v1/responses`` body. ``content``/``reasoning``/``tool_calls`` are
    already split the way the Chat Completions path splits them (``server._run``) — this only
    repackages the same three pieces of information into Responses output items, in the order
    a real turn produces them: reasoning, then either the answer or the calls it made instead."""
    output = []
    if reasoning:
        output.append(_reasoning_item(reasoning))
    for tc in (tool_calls or []):
        fn = tc.get("function", {})
        output.append(_function_call_item(tc.get("id") or _item_id("call"),
                                          fn.get("name", ""), fn.get("arguments", "{}")))
    if content:
        output.append(_output_text_item(_item_id("msg"), content))
    status = "incomplete" if finish_reason == "length" else "completed"
    body = {
        "id": resp_id, "object": "response", "created_at": created or int(time.time()),
        "status": status, "model": model, "output": output,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens,
                  "total_tokens": input_tokens + output_tokens},
    }
    if status == "incomplete":
        body["incomplete_details"] = {"reason": "max_output_tokens"}
    return body


class ResponseStream:
    """Streaming ``/v1/responses`` events at round granularity, the Responses-API twin of
    ``anthropic_api.MessageStream``. ``start()``/``delta()``/``finish()`` each return a list
    of ``(event_name, payload)`` pairs ready for ``server._sse``."""

    def __init__(self, *, model: str, input_tokens: int, resp_id: str | None = None,
                created: int | None = None):
        self.model = model
        self.input_tokens = input_tokens
        self.id = resp_id or _resp_id()
        self.created = created or int(time.time())
        self.seq = 0
        self.msg_item_id = _item_id("msg")
        self.text = ""
        # Allocated lazily on the first text delta, not up front: a pure tool-call turn (the
        # model calls a tool with no preceding text, same as `_run_stream_body`'s want_tools
        # path) must never announce-then-close an empty message item that then isn't in the
        # final `output` array — a client tracking items by id from the stream would see a
        # completed item that vanishes from the response it supposedly belongs to.
        self.msg_index = None

    def _n(self) -> int:
        self.seq += 1
        return self.seq

    def _base_response(self, status: str) -> dict:
        return {"id": self.id, "object": "response", "created_at": self.created,
                "status": status, "model": self.model, "output": []}

    def start(self) -> list[tuple[str, dict]]:
        resp = self._base_response("in_progress")
        return [("response.created", {"type": "response.created", "response": resp,
                                      "sequence_number": self._n()})]

    def delta(self, piece: str) -> list[tuple[str, dict]]:
        if not piece:
            return []
        events = []
        if self.msg_index is None:
            self.msg_index = 0
            item = {"id": self.msg_item_id, "type": "message", "status": "in_progress",
                   "role": "assistant", "content": []}
            part = {"type": "output_text", "text": "", "annotations": []}
            events.append(("response.output_item.added",
                          {"type": "response.output_item.added", "output_index": self.msg_index,
                           "item": item, "sequence_number": self._n()}))
            events.append(("response.content_part.added",
                          {"type": "response.content_part.added", "item_id": self.msg_item_id,
                           "output_index": self.msg_index, "content_index": 0, "part": part,
                           "sequence_number": self._n()}))
        self.text += piece
        events.append(("response.output_text.delta",
                       {"type": "response.output_text.delta", "item_id": self.msg_item_id,
                        "output_index": self.msg_index, "content_index": 0, "delta": piece,
                        "sequence_number": self._n()}))
        return events

    def finish(self, *, finish_reason: str, output_tokens: int,
              tool_calls: list[dict] | None = None) -> list[tuple[str, dict]]:
        events = []
        output = []
        if self.msg_index is not None:
            part = {"type": "output_text", "text": self.text, "annotations": []}
            msg_item = {"id": self.msg_item_id, "type": "message", "status": "completed",
                       "role": "assistant", "content": [part]}
            events += [
                ("response.output_text.done",
                 {"type": "response.output_text.done", "item_id": self.msg_item_id,
                  "output_index": self.msg_index, "content_index": 0, "text": self.text,
                  "sequence_number": self._n()}),
                ("response.content_part.done",
                 {"type": "response.content_part.done", "item_id": self.msg_item_id,
                  "output_index": self.msg_index, "content_index": 0, "part": part,
                  "sequence_number": self._n()}),
                ("response.output_item.done",
                 {"type": "response.output_item.done", "output_index": self.msg_index,
                  "item": msg_item, "sequence_number": self._n()}),
            ]
            output.append(msg_item)
        # Tool calls land atomically once generation finishes — same limitation the Chat
        # Completions streaming path documents (incremental tool-call streaming isn't
        # reliable to reconstruct from XML/JSON markup that may split across chunks). Each
        # gets a single added -> one-shot arguments delta -> done triple rather than N
        # incremental argument deltas; a client that only accumulates delta text sees the
        # same final arguments either way.
        next_index = 1 if self.msg_index is not None else 0
        for tc in (tool_calls or []):
            i = next_index
            next_index += 1
            fn = tc.get("function", {})
            call_id = tc.get("id") or _item_id("call")
            name, args = fn.get("name", ""), fn.get("arguments", "{}")
            item = _function_call_item(call_id, name, args, status="in_progress")
            item_id = item["id"]
            events += [
                ("response.output_item.added",
                 {"type": "response.output_item.added", "output_index": i, "item": item,
                  "sequence_number": self._n()}),
                ("response.function_call_arguments.delta",
                 {"type": "response.function_call_arguments.delta", "item_id": item_id,
                  "output_index": i, "delta": args, "sequence_number": self._n()}),
                ("response.function_call_arguments.done",
                 {"type": "response.function_call_arguments.done", "item_id": item_id,
                  "output_index": i, "arguments": args, "sequence_number": self._n()}),
            ]
            done_item = {**item, "status": "completed"}
            events.append(("response.output_item.done",
                          {"type": "response.output_item.done", "output_index": i,
                           "item": done_item, "sequence_number": self._n()}))
            output.append(done_item)
        status = "incomplete" if finish_reason == "length" else "completed"
        resp = self._base_response(status)
        resp["output"] = output
        resp["usage"] = {"input_tokens": self.input_tokens, "output_tokens": output_tokens,
                         "total_tokens": self.input_tokens + output_tokens}
        if status == "incomplete":
            resp["incomplete_details"] = {"reason": "max_output_tokens"}
        final_type = "response.completed" if status == "completed" else "response.incomplete"
        events.append((final_type, {"type": final_type, "response": resp,
                                    "sequence_number": self._n()}))
        return events
