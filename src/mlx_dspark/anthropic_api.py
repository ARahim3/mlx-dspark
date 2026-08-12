"""Anthropic Messages API (``/v1/messages``) — the dialect **Claude Code** speaks.

Claude Code talks to whatever ``ANTHROPIC_BASE_URL`` points at using Anthropic's Messages
API, *not* the OpenAI one, so an OpenAI-compatible server is invisible to it. This module is
the translation layer: Anthropic request -> the OpenAI-shaped message list the rest of
mlx-dspark already speaks (:func:`~mlx_dspark.generate.encode_messages` +
:func:`~mlx_dspark.tools.normalize_tool_messages`), and generated text -> Anthropic content
blocks / SSE events.

Everything here is **pure and model-free** (no MLX, no weights) so the whole surface is
unit-testable; ``server.py`` owns the HTTP plumbing.

What the contract actually requires (Anthropic's own gateway-protocol reference,
`code.claude.com/docs/en/llm-gateway-protocol`) and how we satisfy it:

* **``POST /v1/messages``, streaming.** Claude Code posts to ``/v1/messages?beta=true`` — match
  on the *path*, not the full URL — and consumes SSE as it arrives; a server that buffers the
  whole response before relaying stalls the client. :class:`MessageStream` emits events at
  round granularity.
* **``POST /v1/messages/count_tokens`` is optional** — without it Claude Code estimates context
  usage locally. We serve it (exact for the loaded tokenizer, and it costs a tokenizer call).
* **Unknown request fields must not 400.** Claude Code sends capabilities that grow every
  release (``thinking``, ``context_management``, ``output_config``, beta tool-schema fields
  like ``strict``/``defer_loading``, ``cache_control`` markers, ``metadata``…) and treats model
  names it doesn't recognize — which includes every name a local server serves — as current
  models that get the newest fields. So we read the fields we implement and **ignore the rest**
  rather than validating; a strict parser here breaks on a Claude Code release we've never seen.
* **Error bodies are matched on wording.** Claude Code's auto-compact-and-retry fires on
  Anthropic's ``prompt is too long`` phrasing, so :func:`context_overflow_error` reproduces it
  verbatim — a local model's short window then behaves like a context limit the client knows
  how to recover from, instead of a hard failure.

Reasoning models are the second thing that needs unpacking: Qwen3 and friends wrap their
chain of thought in ``<think>…</think>`` *inside the generated text*, while Anthropic carries
it as a separate ``thinking`` content block. Left alone the raw markup renders as assistant
prose in Claude Code, so we split it out (:func:`split_thinking`) and emit a real thinking
block — or drop it when the client asked for ``thinking: {"type": "disabled"}``.

Tool calls are the one lossy join. The models we serve emit calls in their own syntax
(Hermes ``<tool_call>{json}</tool_call>`` for Qwen3, Gemma-4's bespoke form) which is only
parseable once complete, while Anthropic streams ``tool_use`` blocks as partial JSON. We
stream text right up to the first tool-call marker (:class:`_ToolGate` holds back just enough
tail to catch a marker split across rounds), then buffer and emit whole ``tool_use`` blocks at
the end. Text streams live; tool calls arrive atomically — which is what a client can act on
anyway.
"""

from __future__ import annotations

import json
import re
import uuid

from .tools import parse_tool_calls

# Native tool-call openers we must not stream to the client as prose. Only the *opening*
# markers matter: once one appears, everything after it is buffered for `parse_tool_calls`.
# muse_glimmer wraps its calls in `<atem:function_calls>` then `<atem:invoke …>`; gate on both
# so a call is caught whichever the round boundary lands on.
_TOOL_MARKERS = ("<tool_call>", "<|tool_call>", "<atem:function_calls>", "<atem:invoke")
_MAX_MARKER = max(len(m) for m in _TOOL_MARKERS)

# Reasoning models wrap their chain of thought in these. Anthropic carries reasoning as
# `thinking` content blocks, so the markup must never reach the client as assistant prose.
# Gemma-4's pair is not a guess: its `tokenizer_config.json` ships the model's own response
# grammar at `response_schema.x-regex`, which reads
#   (<\|channel>thought\n(?P<thinking>.*?)<channel\|>)? (?P<tool_calls>…)? (?P<content>…)?
_THINK_PAIRS = (
    ("<think>", "</think>"),                    # Qwen3 family
    ("<|channel>thought\n", "<channel|>"),      # Gemma-4
)
_THINK_OPEN, _THINK_CLOSE = _THINK_PAIRS[0]

# muse_glimmer ("Onyx ATEM" harmony format) doesn't use an open/close *pair* — it emits a run of
# recipient-tagged sub-messages, `[<|start|>]HEADER<|message|>BODY<|eom|>`, where the header
# carries `to=<recipient>` (`to=self` = analysis/reasoning, anything else = the final answer or a
# tool call) and `<|eot|>` (eos, never reaches us) ends the turn. These three special tokens are
# the only structural delimiters and they survive detokenization, so a small state machine over
# them recovers the channels. The tokenizer ships the authoritative grammar in its own
# `response_template` (fields.reasoning_content.open_pattern = `to=self<|message|>` … close
# `<|eom|>`; fields.content.open_pattern = `to=user<|message|>`). See NOTES "Muse-Glimmer-30B".
_MUSE_MSG, _MUSE_EOM, _MUSE_START = "<|message|>", "<|eom|>", "<|start|>"
_MUSE_STRUCT = (_MUSE_MSG, _MUSE_EOM, _MUSE_START)
_MUSE_HOLD = max(len(s) for s in _MUSE_STRUCT) - 1     # straddle look-back when streaming
_MUSE_RECIPIENT = re.compile(r"to=([^\s<]+)\s*$")      # the recipient at the end of a header

ANTHROPIC_VERSION = "2023-06-01"

# Anthropic signs thinking blocks so it can verify one that's replayed to it. Nothing here
# verifies anything (we drop inbound thinking blocks), but the field is part of the shape, so
# we send a constant marker rather than omitting it and tripping a strict client.
_SIGNATURE = "mlx-dspark"


def prompt_opens_thinking(tail_text: str) -> str | None:
    """The closing marker to expect if the chat template already *opened* a reasoning block at
    the end of the prompt, else ``None``.

    Qwen3 emits its own opener, but the Qwen3-2507 / qwen3_5 thinking templates *prefill* one,
    so the model's first generated token is already inside the block and no opener will ever
    appear in the output. Streaming can't detect that from the output alone — a few tokens of
    lookahead can't see a marker that isn't coming — so the caller decodes the prompt tail and
    tells us. Without this the closing marker leaks into the text block on those models.

    Note Gemma-4 prefills the *whole empty pair* (`…thought\\n<channel|>`) when thinking is off,
    which ends on the closer and correctly reports ``None`` here.
    """
    tail = tail_text.rstrip()
    for open_, close in _THINK_PAIRS:
        if tail.endswith(open_.rstrip()):
            return close
    return None


class MuseChannelParser:
    """Incrementally splits muse_glimmer's raw Onyx-ATEM output into a stream of
    ``('reasoning' | 'answer', text)`` chunks. Tool XML is left inside ``'answer'`` for
    :func:`~mlx_dspark.tools.parse_tool_calls`, so this is the *channel* split, not the tool one.

    A three-state machine driven only by the structural special tokens (``<|start|>``,
    ``<|message|>``, ``<|eom|>``): ``header`` buffers a sub-message header (the ``to=<recipient>``
    line) without emitting; on ``<|message|>`` the recipient decides the channel and we enter
    ``content``; ``<|eom|>`` ends the body (``idle``) until the next ``<|start|>`` opens the next
    header. In ``content`` we hold back the last ``_MUSE_HOLD`` chars so a special token split
    across two rounds is still recognised (the tool gate and stop-string scanner use the same
    trick). ``feed`` is incremental; ``feed(text, final=True)`` in one shot is the whole-string
    parse that :func:`split_thinking` uses."""

    def __init__(self):
        self.buf = ""
        self.state = "header"     # 'header' | 'content' | 'idle'
        self.channel: str | None = None   # 'reasoning' | 'answer'

    def _next_struct(self) -> tuple[int, str]:
        best, tok = -1, ""
        for t in _MUSE_STRUCT:
            j = self.buf.find(t)
            if j != -1 and (best == -1 or j < best):
                best, tok = j, t
        return best, tok

    def feed(self, piece: str, final: bool = False) -> list[tuple[str, str]]:
        self.buf += piece
        out: list[tuple[str, str]] = []
        while True:
            idx, tok = self._next_struct()

            if self.state == "content":
                if idx == -1:                       # no marker yet: emit, holding back a tail
                    hold = 0 if final else _MUSE_HOLD
                    upto = max(0, len(self.buf) - hold)
                    if upto:
                        out.append((self.channel, self.buf[:upto]))
                        self.buf = self.buf[upto:]
                    return out
                if idx:
                    out.append((self.channel, self.buf[:idx]))
                self.buf = self.buf[idx + len(tok):]
                if tok == _MUSE_EOM:
                    self.state, self.channel = "idle", None
                elif tok == _MUSE_START:
                    self.state, self.channel = "header", None
                # a bare <|message|> mid-body (no header) keeps the current channel
                continue

            if self.state == "header":              # buffer the recipient line, emit nothing
                if idx == -1:
                    if final:
                        self.buf = ""               # drop a trailing partial header
                    return out
                if tok == _MUSE_MSG:
                    header, self.buf = self.buf[:idx], self.buf[idx + len(tok):]
                    m = _MUSE_RECIPIENT.search(header)
                    rcpt = m.group(1) if m else "user"
                    self.channel = "reasoning" if rcpt == "self" else "answer"
                    self.state = "content"
                    continue
                self.buf = self.buf[idx + len(tok):]  # <|start|> / <|eom|> inside a header: skip
                continue

            # idle: between <|eom|> and the next sub-message; drop stray text, act on markers
            if idx == -1:
                if final:
                    self.buf = ""
                return out
            self.buf = self.buf[idx + len(tok):]
            if tok == _MUSE_START:
                self.state = "header"
            elif tok == _MUSE_MSG:                  # body with no header: assume answer
                self.channel, self.state = "answer", "content"
            # <|eom|> in idle: ignore


def _split_muse(text: str) -> tuple[str, str]:
    chunks = MuseChannelParser().feed(text, final=True)
    reasoning = "".join(t for k, t in chunks if k == "reasoning")
    answer = "".join(t for k, t in chunks if k == "answer")
    return reasoning.strip(), answer.strip()


def split_thinking(text: str) -> tuple[str, str]:
    """``(reasoning, answer)``. Handles the self-opened form, the prefilled one (output that
    closes a block it never opened), and muse_glimmer's recipient-tagged channels. Returns
    ``("", text)`` when there's no reasoning, and treats an unterminated block as all-reasoning."""
    if _MUSE_MSG in text:            # muse Onyx-ATEM channels (the marker is muse-only)
        return _split_muse(text)
    s = text.lstrip()
    for open_, close in _THINK_PAIRS:
        if s.startswith(open_):
            body = s[len(open_):]
            end = body.find(close)
            return (body, "") if end == -1 else (body[:end], body[end + len(close):].lstrip())
    for open_, close in _THINK_PAIRS:      # prefilled opener: only the closer is generated
        end = s.find(close)
        if end != -1 and open_ not in s:
            return s[:end], s[end + len(close):].lstrip()
    return "", text


# --------------------------------------------------------------------------- request -> ours


def _blocks_to_text(content) -> str:
    """Anthropic ``content`` (a string, or a list of typed blocks) -> plain text.

    Only ``text`` blocks carry text; ``image`` blocks are dropped (the targets we serve are
    text models), and ``thinking`` / ``redacted_thinking`` blocks are dropped because we never
    produce them, so echoing them back would put a foreign trace in the prompt.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts = []
    for b in content:
        if isinstance(b, str):
            parts.append(b)
        elif isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str):
            parts.append(b["text"])
    return "\n".join(p for p in parts if p)


def system_text(system) -> str:
    """The ``system`` field, which Anthropic allows as a string *or* a list of text blocks.

    Claude Code sends the list form (its cache breakpoints ride on the blocks), with a short
    attribution block first. That block is stable for the lifetime of a conversation, so it
    doesn't defeat our prefix cache; users who'd rather not send it at all can set
    ``CLAUDE_CODE_ATTRIBUTION_HEADER=0``.
    """
    return _blocks_to_text(system)


def _tool_result_text(block: dict) -> str:
    """A ``tool_result`` block's payload as text. ``content`` may be a string, a list of
    blocks, or absent; errors are prefixed so the model sees the failure the way its own
    chat template would render one."""
    text = _blocks_to_text(block.get("content"))
    if block.get("is_error"):
        return f"Error: {text}" if text else "Error"
    return text


def _attach_reminder(out: list[dict], text: str) -> bool:
    """Fold an operator instruction into the adjacent user turn. True if it landed."""
    if out and out[-1].get("role") == "user":
        out[-1]["content"] = f"{out[-1]['content']}\n\n{text}".strip()
        return True
    return False


def convert_messages(messages: list[dict], system=None) -> list[dict]:
    """Anthropic ``messages`` (+ ``system``) -> the OpenAI-shaped list the chat templates take.

    Content blocks map as follows:

    ============================  ===========================================================
    Anthropic                     OpenAI-shaped
    ============================  ===========================================================
    ``text``                      message ``content``
    ``tool_use`` (assistant)      ``tool_calls[]`` entry (``function.arguments`` JSON string)
    ``tool_result`` (user)        its own ``{"role": "tool", "tool_call_id": …}`` message
    ``image`` / ``thinking``      dropped (text targets; we never emit thinking blocks)
    ============================  ===========================================================

    A user turn holding both tool results and text becomes the tool messages followed by the
    text message, which is the ordering every chat template expects.

    **Mid-conversation ``system`` messages** get special handling. Recent Claude Code sends
    operator instructions as ``{"role": "system"}`` entries *inside* ``messages`` (an Opus-4.8
    API feature it also applies to model names it doesn't recognise — i.e. ours). Most chat
    templates only accept a system message in first position and raise otherwise, which fails
    the whole request. We use the documented fallback for models without the feature: render it
    as a ``<system-reminder>`` block folded into the adjacent user turn. Leading system messages
    still merge into the real system prompt.
    """
    out: list[dict] = []
    sys_parts = [system_text(system)]
    pending = ""          # a reminder waiting for the next user turn

    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        content = m.get("content")

        if role == "system":
            text = _blocks_to_text(content)
            if not text:
                continue
            if not out:                       # still ahead of any real turn: it's the prompt
                sys_parts.append(text)
                continue
            block = f"<system-reminder>\n{text}\n</system-reminder>"
            if not _attach_reminder(out, block):
                pending = f"{pending}\n\n{block}".strip()
            continue

        if not isinstance(content, list):
            text = _blocks_to_text(content)
            if role == "user" and pending:
                text, pending = f"{pending}\n\n{text}".strip(), ""
            out.append({"role": role, "content": text})
            continue

        # tool results first: they answer the *previous* assistant turn
        for b in content:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                out.append({"role": "tool",
                            "tool_call_id": b.get("tool_use_id", ""),
                            "content": _tool_result_text(b)})
        text = _blocks_to_text(content)
        if role == "user" and pending:
            text, pending = f"{pending}\n\n{text}".strip(), ""
        calls = [b for b in content
                 if isinstance(b, dict) and b.get("type") == "tool_use"]
        if calls:
            out.append({
                "role": role,
                "content": text,
                "tool_calls": [{"id": b.get("id") or _tool_id(),
                                "type": "function",
                                "function": {"name": b.get("name", ""),
                                             "arguments": json.dumps(b.get("input") or {},
                                                                     ensure_ascii=False)}}
                               for b in calls],
            })
        elif text or not any(isinstance(b, dict) and b.get("type") == "tool_result"
                             for b in content):
            # keep empty turns only when they weren't purely tool results (templates choke
            # on a bare empty assistant/user turn injected between real ones)
            out.append({"role": role, "content": text})

    if pending and not _attach_reminder(out, pending):
        out.append({"role": "user", "content": pending})
    sys_txt = "\n\n".join(p for p in sys_parts if p)
    return ([{"role": "system", "content": sys_txt}] if sys_txt else []) + out


def convert_tools(tools) -> list[dict] | None:
    """Anthropic tool schemas -> the OpenAI ``function`` shape the chat templates render.

    Anthropic: ``{"name", "description", "input_schema"}`` (plus beta extras like ``strict``
    and ``defer_loading``, and server-tool entries carrying a ``type`` but no schema).
    We take the three fields we can render and drop the rest — an unknown extra field must
    never fail the request.
    """
    if not tools:
        return None
    out = []
    for t in tools:
        if not isinstance(t, dict) or not t.get("name"):
            continue
        fn = {"name": t["name"], "description": t.get("description") or ""}
        schema = t.get("input_schema")
        fn["parameters"] = schema if isinstance(schema, dict) else {"type": "object",
                                                                   "properties": {}}
        out.append({"type": "function", "function": fn})
    return out or None


def norm_stop_sequences(req: dict) -> list[str]:
    """``stop_sequences`` (Anthropic's name for OpenAI's ``stop``)."""
    s = req.get("stop_sequences")
    if not s:
        return []
    if isinstance(s, str):
        return [s]
    return [str(x) for x in s if x]


# --------------------------------------------------------------------------- ours -> Anthropic


def _tool_id() -> str:
    return "toolu_" + uuid.uuid4().hex[:24]


def _msg_id() -> str:
    return "msg_" + uuid.uuid4().hex[:24]


def tool_use_blocks(openai_calls: list[dict]) -> list[dict]:
    """OpenAI ``tool_calls`` (what :func:`~mlx_dspark.tools.parse_tool_calls` returns) ->
    Anthropic ``tool_use`` content blocks. ``function.arguments`` is a JSON *string* in the
    OpenAI shape and an ``input`` *object* in Anthropic's; unparseable arguments degrade to
    an empty object rather than failing the turn."""
    blocks = []
    for tc in openai_calls:
        fn = tc.get("function", {})
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {}
        if not isinstance(args, dict):
            args = {}
        blocks.append({"type": "tool_use", "id": _tool_id(),
                       "name": fn.get("name", ""), "input": args})
    return blocks


def stop_reason(finish_reason: str, has_tool_use: bool) -> str:
    """GenResult.finish_reason (``stop`` | ``length``) -> Anthropic ``stop_reason``.

    A turn that ended with tool calls reports ``tool_use`` regardless of how generation
    stopped — that's the signal Claude Code branches on to run the tools.
    """
    if has_tool_use:
        return "tool_use"
    return "max_tokens" if finish_reason == "length" else "end_turn"


def build_message(text: str, *, model: str, input_tokens: int, output_tokens: int,
                  finish_reason: str, msg_id: str | None = None,
                  thinking: bool = True, schemas: dict | None = None) -> dict:
    """A complete non-streaming ``Message``, with reasoning and tool calls lifted out of
    ``text`` into their own content blocks. ``thinking=False`` discards the reasoning instead
    of emitting a block for it; ``schemas`` types the XML tool-call form (see ``tools.py``)."""
    reasoning, text = split_thinking(text)
    calls, cleaned = parse_tool_calls(text, schemas)
    content: list[dict] = []
    if reasoning and thinking:
        content.append({"type": "thinking", "thinking": reasoning, "signature": _SIGNATURE})
    if cleaned:
        content.append({"type": "text", "text": cleaned})
    blocks = tool_use_blocks(calls)
    content.extend(blocks)
    if not content:                       # Anthropic always returns at least one block
        content.append({"type": "text", "text": ""})
    return {
        "id": msg_id or _msg_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason(finish_reason, bool(blocks)),
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


# --------------------------------------------------------------------------- streaming


class _ToolGate:
    """Splits streamed text into "safe to send now" and "part of a tool call".

    A native tool-call marker can straddle two generation rounds, so while no marker has been
    seen we hold back the last ``_MAX_MARKER - 1`` characters (same trick the generator's
    stop-string scanner uses). Once a complete marker appears the gate *trips*: everything from
    the marker on is buffered for :func:`parse_tool_calls` and nothing further is streamed.
    """

    def __init__(self):
        self.buf = ""        # everything the model has produced
        self.sent = 0        # how much of buf has gone out as text_delta
        self.scan = 0        # incremental marker-scan resume point
        self.tripped = False

    @property
    def sent_text(self) -> str:
        return self.buf[:self.sent]

    def feed(self, piece: str) -> str:
        """Absorb a new chunk; return the text to emit right now (possibly empty)."""
        self.buf += piece
        if self.tripped:
            return ""
        hit = -1
        for m in _TOOL_MARKERS:
            i = self.buf.find(m, self.scan)
            if i != -1:
                hit = i if hit == -1 else min(hit, i)
        if hit != -1:
            self.tripped = True
            out = self.buf[self.sent:hit]
            self.sent = hit
            return out
        self.scan = max(0, len(self.buf) - (_MAX_MARKER - 1))
        safe = max(self.sent, len(self.buf) - (_MAX_MARKER - 1))
        out = self.buf[self.sent:safe]
        self.sent = safe
        return out


class MessageStream:
    """Builds the Anthropic SSE event sequence for one streamed response.

    Each method returns a list of ``(event_name, payload)`` pairs; the HTTP layer does the
    ``event:``/``data:`` framing. Event order matches the real API:
    ``message_start`` -> ``content_block_start`` -> ``content_block_delta``\\* ->
    ``content_block_stop`` (per block) -> ``message_delta`` -> ``message_stop``.

    The one structural constraint: a block's *type* is announced in ``content_block_start``,
    before any of its content exists. A reasoning model opens with ``<think>`` and we can't
    know that until a few tokens have arrived, so ``start()`` deliberately does **not** open a
    block — it sends ``message_start`` (bytes on the wire immediately, which is what matters
    for a long prefill) and the first block opens once its type is known.
    """

    def __init__(self, *, model: str, input_tokens: int, msg_id: str | None = None,
                 thinking: bool = True, in_thinking: str | None = None,
                 schemas: dict | None = None, muse: bool = False):
        self.model = model
        self.input_tokens = input_tokens
        self.id = msg_id or _msg_id()
        self.emit_thinking = thinking
        self.schemas = schemas
        self.gate = _ToolGate()
        self.pending = ""             # input not yet routed to a block
        # muse_glimmer routes its recipient-tagged channels through a dedicated parser (the
        # open/close-pair machinery below doesn't model its `<|eom|>` handoffs); everything else
        # uses the pair state machine. The flag comes from the server (it knows the model), so a
        # non-muse stream never touches muse code and stays byte-identical.
        self._muse = MuseChannelParser() if muse else None
        # "thinking" when the chat template prefilled the opener (see prompt_opens_thinking);
        # `close` is the marker that ends the reasoning block, which differs per model family
        self.phase = "thinking" if in_thinking else "pending"
        self.close = in_thinking or _THINK_CLOSE
        self.next_index = 0           # next block index to hand out
        self.open: tuple[str, int] | None = None   # (kind, index) of the block being written

    # --- block bookkeeping ---
    def _open_block(self, kind: str) -> list[tuple[str, dict]]:
        if kind == "thinking" and not self.emit_thinking:
            self.open = ("thinking", -1)          # consumed but not published
            return []
        body = ({"type": "thinking", "thinking": "", "signature": ""} if kind == "thinking"
                else {"type": "text", "text": ""})
        i = self.next_index
        self.next_index += 1
        self.open = (kind, i)
        return [("content_block_start", {"type": "content_block_start", "index": i,
                                         "content_block": body})]

    def _close_block(self) -> list[tuple[str, dict]]:
        if self.open is None:
            return []
        kind, i = self.open
        self.open = None
        if i < 0:
            return []
        ev = []
        if kind == "thinking":
            # the real API sends a signature_delta immediately before the stop
            ev.append(("content_block_delta",
                       {"type": "content_block_delta", "index": i,
                        "delta": {"type": "signature_delta", "signature": _SIGNATURE}}))
        ev.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
        return ev

    def _content_delta(self, text: str) -> list[tuple[str, dict]]:
        kind, i = self.open
        if i < 0 or not text:
            return []
        d = ({"type": "thinking_delta", "thinking": text} if kind == "thinking"
             else {"type": "text_delta", "text": text})
        return [("content_block_delta",
                 {"type": "content_block_delta", "index": i, "delta": d})]

    # --- driving ---
    def start(self) -> list[tuple[str, dict]]:
        """``message_start`` + a ``ping``. Sent before generation begins so the client sees
        bytes immediately — prefill on a long agent prompt can take seconds."""
        return [
            ("message_start", {"type": "message_start", "message": {
                "id": self.id, "type": "message", "role": "assistant", "model": self.model,
                "content": [], "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": self.input_tokens, "output_tokens": 0}}}),
            ("ping", {"type": "ping"}),
        ]

    def delta(self, piece: str) -> list[tuple[str, dict]]:
        self.pending += piece
        return self._consume(final=False)

    def _consume_muse(self, final: bool) -> list[tuple[str, dict]]:
        """Drive the muse channel parser and route its chunks to blocks: reasoning -> the
        thinking block, answer -> the text block (via the tool gate, so atem XML is buffered for
        the tool pass instead of streamed as prose). A channel switch closes the open block and
        opens the other, so a client never sees interleaved kinds."""
        ev: list[tuple[str, dict]] = []
        piece, self.pending = self.pending, ""
        for kind, text in self._muse.feed(piece, final=final):
            want = "thinking" if kind == "reasoning" else "text"
            if self.open is None or self.open[0] != want:
                if self.open is not None:
                    ev += self._close_block()
                ev += self._open_block(want)
            ev += self._content_delta(text if want == "thinking" else self.gate.feed(text))
        return ev

    def _consume(self, final: bool) -> list[tuple[str, dict]]:
        if self._muse is not None:
            return self._consume_muse(final)
        ev: list[tuple[str, dict]] = []
        while True:
            if self.phase == "pending":
                s = self.pending.lstrip()
                opener = next((p for p in _THINK_PAIRS if s.startswith(p[0])), None)
                if opener:
                    cut = len(self.pending) - len(s) + len(opener[0])
                    self.pending = self.pending[cut:]
                    self.phase, self.close = "thinking", opener[1]
                    ev += self._open_block("thinking")
                    continue
                if not final and any(o.startswith(s) for o, _ in _THINK_PAIRS):
                    return ev                     # could still grow into an opener
                self.phase = "text"
                ev += self._open_block("text")
                continue

            if self.phase == "thinking":
                if self.open is None:      # started mid-block (prefilled opener)
                    ev += self._open_block("thinking")
                i = self.pending.find(self.close)
                if i == -1:
                    hold = 0 if final else len(self.close) - 1
                    upto = max(0, len(self.pending) - hold)
                    out, self.pending = self.pending[:upto], self.pending[upto:]
                    return ev + self._content_delta(out)
                out = self.pending[:i]
                self.pending = self.pending[i + len(self.close):].lstrip()
                ev += self._content_delta(out)
                ev += self._close_block()
                self.phase = "text"
                ev += self._open_block("text")
                continue

            piece, self.pending = self.pending, ""
            return ev + self._content_delta(self.gate.feed(piece))

    def finish(self, *, finish_reason: str, output_tokens: int) -> list[tuple[str, dict]]:
        """Drain, close the open block, emit any ``tool_use`` blocks, then ``message_delta`` /
        ``message_stop``."""
        events = self._consume(final=True)
        calls, cleaned = parse_tool_calls(self.gate.buf, self.schemas)

        # Whatever the gate held back (marker lookahead, or text the model wrote after a tool
        # call) still belongs to the text block. Only send a clean continuation of what the
        # client already has — never risk duplicating text it rendered.
        sent = self.gate.sent_text
        if not sent:
            tail = cleaned
        elif cleaned.startswith(sent):
            tail = cleaned[len(sent):]
        else:
            tail = ""
        if tail and self.open is not None and self.open[0] == "text":
            events += self._content_delta(tail)
        events += self._close_block()
        if self.next_index == 0 and not calls:
            # never produced anything: Anthropic always returns at least one block
            events += self._open_block("text")
            events += self._close_block()

        blocks = tool_use_blocks(calls)
        base = self.next_index
        for i, b in enumerate(blocks, start=base):
            events.append(("content_block_start",
                           {"type": "content_block_start", "index": i,
                            "content_block": {"type": "tool_use", "id": b["id"],
                                              "name": b["name"], "input": {}}}))
            # `input` streams as partial-JSON deltas that the client concatenates; we know the
            # whole object only once the call is parsed, so it goes out as a single delta.
            events.append(("content_block_delta",
                           {"type": "content_block_delta", "index": i,
                            "delta": {"type": "input_json_delta",
                                      "partial_json": json.dumps(b["input"],
                                                                 ensure_ascii=False)}}))
            events.append(("content_block_stop", {"type": "content_block_stop", "index": i}))

        events.append(("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason(finish_reason, bool(blocks)),
                      "stop_sequence": None},
            "usage": {"input_tokens": self.input_tokens, "output_tokens": output_tokens}}))
        events.append(("message_stop", {"type": "message_stop"}))
        return events


# --------------------------------------------------------------------------- errors


def error_body(message: str, etype: str = "invalid_request_error") -> dict:
    """Anthropic's error envelope. Claude Code matches on the *wording* of upstream errors to
    decide whether it can recover (and forwards them unmodified through gateways for exactly
    that reason), so error text here is behaviour, not just diagnostics."""
    return {"type": "error", "error": {"type": etype, "message": message}}


def context_overflow_error(n_tokens: int, window: int) -> dict:
    """The 400 to return when a request won't fit the loaded model's context window.

    The leading phrase is Anthropic's verbatim: Claude Code's automatic compact-and-retry
    matches on ``prompt is too long``, so a local model with a 32k window then triggers the
    same graceful ``/compact`` path a real context limit would, instead of dead-ending the
    session. Rewording this string silently disables that recovery.
    """
    return error_body(
        f"prompt is too long: {n_tokens} tokens > {window} maximum",
        "invalid_request_error")
