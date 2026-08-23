"""Tool-calling glue: translate between OpenAI ``tool_calls`` and each model's native syntax.

Five output formats are parsed (detected by markers, so it doesn't matter which model is
loaded — it also covers any model that borrows one of these conventions):

  * **Hermes / JSON** (Qwen3, DeepSpec drafters' targets, many others)::

        <tool_call>{"name": "f", "arguments": {...}}</tool_call>

  * **Gemma-4** (bespoke)::

        <|tool_call>call:f{key:<|"|>str val<|"|>,n:3,flag:true}<tool_call|>

    where string values are wrapped in Gemma's ``<|"|>`` quote markers and other scalars are
    bare. Flat arguments are parsed fully; deeply nested structures fall back to string values
    (rare for tool calls, and documented).

  * **XML / "function"** (Ornith-1.0, and the several other models that copy this convention)::

        <tool_call>
        <function=example_function_name>
        <parameter=example_parameter_1>
        value_1
        </parameter>
        </function>
        </tool_call>

    Values are *raw text* and may span multiple lines, so their JSON type isn't recoverable
    from the syntax. Pass the request's tool schemas (:func:`schema_types`) and each value is
    coerced to its declared type; without them we fall back to conservative heuristics that
    only touch short single-line scalars, since a multi-line value is essentially always a
    string (file contents, code) and mis-coercing one would corrupt it.

  * **ATEM** (Meta's muse_glimmer "Onyx ATEM" harmony format)::

        <atem:function_calls>
        <atem:invoke name="example_function_name">
        <atem:parameter name="example_parameter_1">value_1</atem:parameter>
        </atem:invoke>
        </atem:function_calls>

    The value grammar is the model's own: the tokenizer ships a ``response_template`` whose
    ``value_parser`` is *JSON with an allow-non-json fallback* — so each value is ``json.loads``'d
    (numbers, booleans, ``null``, objects/arrays parse to their type) and left as a raw string
    when that fails (a bare unquoted string, code, file contents). No schema needed; the format
    types its own scalars. See NOTES "Muse-Glimmer-30B".

  * **LFM2 / pythonic** (LiquidAI LFM2.5)::

        <|tool_call_start|>[get_weather(location="Paris", days=3)]<|tool_call_end|>

    A Python-style list of function calls between LFM2's ``<|tool_call_start|>`` /
    ``<|tool_call_end|>`` tokens (both emitted as *non-special* tokens, so they survive
    detokenization). Parsed with :mod:`ast` (not a regex — a quoted comma or a nested list/dict
    would defeat one): each call's keyword arguments carry their own Python types, and a rare
    positional argument is mapped to the request's tool-schema parameter order.

:func:`parse_tool_calls` returns OpenAI ``tool_calls`` (``function.arguments`` serialized to a
JSON string, per the OpenAI schema) plus the assistant text with the call blocks removed.
:func:`normalize_tool_messages` goes the other way for inbound history: it turns an OpenAI
assistant message's ``function.arguments`` JSON *string* back into a dict so the model's chat
template (which iterates a mapping) renders prior tool calls correctly.
"""

from __future__ import annotations

import ast
import contextlib
import json
import re
import uuid

_HERMES = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_GEMMA = re.compile(r"<\|tool_call>\s*call:\s*([^\{]+?)\s*\{(.*?)\}\s*<tool_call\|>", re.DOTALL)
_GEMMA_STR = '<|"|>'
_GEMMA_FIELD = re.compile(r'([A-Za-z_]\w*)\s*:\s*(<\|"\|>.*?<\|"\|>|[^,]*)', re.DOTALL)
# XML form: <tool_call><function=NAME><parameter=KEY>value</parameter>…</function></tool_call>.
# The closing </tool_call> is optional — models truncated at max_tokens routinely omit it, and
# a call we can parse is better than one dropped for a missing suffix.
_XML = re.compile(r"<tool_call>\s*<function=\s*([^>\s]+?)\s*>(.*?)</function>\s*(?:</tool_call>)?",
                  re.DOTALL)
_XML_PARAM = re.compile(r"<parameter=\s*([^>\s]+?)\s*>\n?(.*?)\n?</parameter>", re.DOTALL)
# ATEM (muse_glimmer): <atem:invoke name="NAME">…<atem:parameter name="K">V</atem:parameter>…
# </atem:invoke>. The regexes are the tokenizer's own (response_template.fields.tool_calls);
# the closing </atem:invoke> is optional so a call truncated at max_tokens still parses.
_ATEM = re.compile(r'<atem:invoke\b[^>]*?\bname="([^"]+)"\s*>(.*?)(?:</atem:invoke>|\Z)', re.DOTALL)
_ATEM_PARAM = re.compile(r'<atem:parameter\b[^>]*?\bname="([^"]+)"[^>]*?>(.*?)</atem:parameter>',
                         re.DOTALL)
# LFM2 (LiquidAI): <|tool_call_start|>[func_name(key="v", n=3)]<|tool_call_end|>. The body is a
# Python-style call list, so it's parsed with `ast` (below), not a regex. `<|tool_call_start|>`
# is emitted as a NON-special token (it survives detokenization), which is what makes it
# parseable at all. The closing token is optional so a call truncated at max_tokens still parses.
_LFM = re.compile(r"<\|tool_call_start\|>\s*(.*?)\s*(?:<\|tool_call_end\|>|\Z)", re.DOTALL)


def _coerce(raw: str):
    raw = raw.strip()
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _parse_gemma_args(body: str) -> dict:
    args: dict = {}
    n = len(_GEMMA_STR)
    for m in _GEMMA_FIELD.finditer(body):
        key, raw = m.group(1), m.group(2).strip()
        if raw.startswith(_GEMMA_STR) and raw.endswith(_GEMMA_STR) and len(raw) >= 2 * n:
            args[key] = raw[n:-n]
        else:
            args[key] = _coerce(raw)
    return args


def schema_types(tools) -> dict[str, dict[str, str]]:
    """``{tool_name: {param: json_type}}`` from a tool list in either the OpenAI shape
    (``{"type": "function", "function": {"name", "parameters"}}``) or the Anthropic one
    (``{"name", "input_schema"}``). Used to coerce raw XML parameter text to its real type."""
    out: dict[str, dict[str, str]] = {}
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        name = fn.get("name")
        if not name:
            continue
        schema = fn.get("parameters") or fn.get("input_schema") or {}
        props = schema.get("properties") if isinstance(schema, dict) else None
        if not isinstance(props, dict):
            continue
        out[name] = {k: v.get("type") for k, v in props.items() if isinstance(v, dict)}
    return out


def _coerce_typed(raw: str, jtype: str | None):
    """A raw XML parameter value -> its declared JSON type. Unparseable values stay strings
    rather than raising: a wrong-typed argument is recoverable by the client, a 500 isn't."""
    if jtype == "string":
        return raw
    if jtype == "boolean":
        return raw.strip().lower() == "true"
    if jtype in ("integer", "number"):
        try:
            return int(raw.strip()) if jtype == "integer" else float(raw.strip())
        except ValueError:
            return raw
    if jtype in ("object", "array"):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw
    # No schema for this parameter: only touch short single-line scalars. A multi-line value
    # is essentially always a string (file contents, code) and coercing it would corrupt it.
    if "\n" not in raw and len(raw) <= 32:
        return _coerce(raw)
    return raw


def _parse_xml_args(body: str, types: dict[str, str] | None) -> dict:
    return {m.group(1): _coerce_typed(m.group(2), (types or {}).get(m.group(1)))
            for m in _XML_PARAM.finditer(body)}


def _atem_value(raw: str):
    """A muse ATEM parameter value -> its type, per the model's own ``value_parser`` (JSON with
    an allow-non-json fallback): parse as JSON (numbers/booleans/null/objects/arrays get their
    real type), else keep the raw string verbatim. Spaces are intentionally *not* stripped — the
    template documents that string values keep their whitespace (file contents, code)."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def _parse_atem_args(body: str) -> dict:
    return {m.group(1): _atem_value(m.group(2)) for m in _ATEM_PARAM.finditer(body)}


def _lfm_call_name(func) -> str:
    """Name of the called function in an LFM2 pythonic call — a bare ``Name`` (``f``) or a
    dotted ``Attribute`` (``mod.f``, which we join back to ``mod.f``)."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        base = _lfm_call_name(func.value)
        return f"{base}.{func.attr}" if base else func.attr
    return ""


def _lfm_value(node):
    """A pythonic-call argument node -> its Python value. ``ast.literal_eval`` covers every JSON
    scalar/collection (str, int, float, bool, None, list, dict, tuple); anything else (a bare
    identifier, an expression) falls back to its source text so the argument is never dropped."""
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        try:
            return ast.unparse(node)
        except Exception:  # noqa: BLE001 — a best-effort string beats losing the argument
            return None


def _parse_pythonic_calls(body: str, schemas: dict[str, dict[str, str]] | None = None
                          ) -> list[tuple[str, dict]]:
    """Parse an LFM2 tool-call body — ``[func_name(key="v", n=3), other(x=1)]`` — into
    ``(name, args)`` pairs. Uses `ast` rather than a regex so quoted commas, nested lists/dicts
    and mixed scalar types parse correctly. Keyword args are used by name; the rare positional
    arg is mapped to the tool schema's parameter order when a schema is available (else dropped,
    since OpenAI ``tool_calls`` arguments are named). A syntactically invalid body (e.g. a call
    truncated mid-argument) yields no calls rather than raising."""
    body = body.strip()
    if not body:
        return []
    try:
        tree = ast.parse(body, mode="eval").body
    except SyntaxError:
        return []
    nodes = tree.elts if isinstance(tree, (ast.List, ast.Tuple)) else [tree]
    out: list[tuple[str, dict]] = []
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        name = _lfm_call_name(node.func)
        if not name:
            continue
        args: dict = {}
        params = list((schemas or {}).get(name, {}).keys())
        for i, a in enumerate(node.args):          # positional -> schema param order
            if i < len(params):
                args[params[i]] = _lfm_value(a)
        for kw in node.keywords:                   # keyword args win over positional
            if kw.arg is not None:
                args[kw.arg] = _lfm_value(kw.value)
        out.append((name, args))
    return out


def _as_openai(name: str, args) -> dict:
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)
    return {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
            "function": {"name": name, "arguments": args}}


def parse_tool_calls(text: str, schemas: dict[str, dict[str, str]] | None = None
                     ) -> tuple[list[dict], str]:
    """(tool_calls, cleaned_text). ``tool_calls`` is OpenAI-shaped; empty if none found.

    ``schemas`` (from :func:`schema_types`) is optional and only affects the XML form, whose
    values carry no type information of their own.
    """
    calls: list[tuple[str, object]] = []
    cleaned = text
    # XML first: its body can contain '{', and Hermes requires a '{' immediately after the
    # opening tag, so the two can't be confused — but parse the stricter shape first anyway.
    if "<function=" in text:
        for m in _XML.finditer(text):
            name = m.group(1)
            calls.append((name, _parse_xml_args(m.group(2), (schemas or {}).get(name))))
        cleaned = _XML.sub("", cleaned)
    if "<tool_call>" in cleaned:
        for m in _HERMES.finditer(cleaned):
            try:
                obj = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("name"):
                calls.append((obj["name"], obj.get("arguments", {})))
        cleaned = _HERMES.sub("", cleaned)
    if "<|tool_call>" in text:
        for m in _GEMMA.finditer(text):
            calls.append((m.group(1).strip(), _parse_gemma_args(m.group(2))))
        cleaned = _GEMMA.sub("", cleaned)
    if "<atem:invoke" in text:
        for m in _ATEM.finditer(text):
            calls.append((m.group(1), _parse_atem_args(m.group(2))))
        cleaned = _ATEM.sub("", cleaned)
        # the invokes are wrapped in a <atem:function_calls> block; drop the now-empty wrapper
        cleaned = re.sub(r"</?atem:function_calls>\s*", "", cleaned)
    if "<|tool_call_start|>" in cleaned:
        for m in _LFM.finditer(cleaned):
            calls.extend(_parse_pythonic_calls(m.group(1), schemas))
        cleaned = _LFM.sub("", cleaned)
    # A generation cut off at max_tokens mid-call leaves an unclosed opener; everything from
    # it onward is an aborted call, not prose, so drop it rather than render raw markup.
    for opener in ("<tool_call>", "<atem:invoke", "<atem:function_calls>", "<|tool_call_start|>"):
        dangling = cleaned.find(opener)
        if dangling != -1:
            cleaned = cleaned[:dangling]
    return [_as_openai(name, args) for name, args in calls], cleaned.strip()


def _flatten_content(content):
    """OpenAI allows ``content`` to be a list of typed parts rather than a plain string
    (``[{"type": "text", "text": "..."}, {"type": "image_url", ...}]``) — this is what many
    coding agents / OpenAI SDKs send. The text targets we serve (and their chat templates)
    expect a string; a list reaches the template unchanged and blows up inside it as
    ``'list object' has no attribute 'startswith'``. So join the text parts (dropping
    non-text parts — images/audio have no place in the text path) and hand the template a
    string. A plain string / ``None`` is returned unchanged.
    """
    if not isinstance(content, list):
        return content
    parts = []
    for p in content:
        if isinstance(p, dict) and isinstance(p.get("text"), str):
            parts.append(p["text"])
        elif isinstance(p, str):
            parts.append(p)
    return "\n".join(parts)


def normalize_tool_messages(messages: list[dict]) -> list[dict]:
    """Make an OpenAI message history renderable by the model chat templates:

      * list-valued ``content`` (OpenAI structured content parts) -> concatenated text
        (see :func:`_flatten_content`);
      * assistant ``function.arguments`` JSON strings -> dicts (templates iterate a mapping);
      * ``content: null`` -> ``""`` (OpenAI allows null content on tool-call messages, but the
        Qwen3 / Gemma-4 templates assume a string and error on ``None``).
    """
    out = []
    for m in messages:
        m = dict(m)
        if "content" in m:
            m["content"] = _flatten_content(m["content"])
        if m.get("content", "") is None:
            m["content"] = ""
        tcs = m.get("tool_calls")
        if tcs:
            new = []
            for tc in tcs:
                tc = dict(tc)
                fn = dict(tc.get("function", {}))
                a = fn.get("arguments")
                if isinstance(a, str):
                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                        fn["arguments"] = json.loads(a)
                tc["function"] = fn
                new.append(tc)
            m["tool_calls"] = new
        out.append(m)
    return out
