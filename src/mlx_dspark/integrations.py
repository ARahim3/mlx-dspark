"""Config for pointing coding agents at a running ``mlx-dspark serve``.

The server already speaks both the OpenAI and Anthropic wire formats (see ``server.py`` and
``anthropic_api.py``), so most of an agent's setup is "here is the base URL and the model id".
The awkward part is that every agent spells that differently — an env block, a TOML provider,
a JSON provider, a settings file — and getting one field wrong fails silently. This module
produces the exact, complete config for each, so the app (and the CLI) can hand a user
something that works on the first try instead of a base URL and good luck.

Nothing here launches anything or writes to disk; it returns descriptors. The app decides
whether to copy them to the clipboard, write a project file, or just display them, and the
Claude Code launcher (``mlx-dspark claude``) stays the residue-free way to actually run one.

Model-free and side-effect-free by design, so it is unit-testable without a server.
"""

from __future__ import annotations

# Fields an agent needs, derived once from the server's /health.


def _openai_base(base_url: str) -> str:
    return base_url.rstrip("/") + "/v1"


def anthropic_env(base_url: str, model: str, api_key: str | None) -> dict[str, str]:
    """The mlx-dspark-specific slice of what ``claude_env`` sets — just the variables a user
    would paste, not a whole copied environment. Kept in sync with ``cli.claude_env`` (the
    launcher is the source of truth; this mirrors its Anthropic/Claude-Code keys)."""
    token = api_key or "mlx-dspark"
    label = f"{model} (mlx-dspark)"
    env = {
        "ANTHROPIC_BASE_URL": base_url.rstrip("/"),
        "ANTHROPIC_AUTH_TOKEN": token,
        "ANTHROPIC_MODEL": model,
        # The haiku slot is load-bearing: Claude Code sends background tasks (titles, etc.)
        # to it, and without a mapping those ask for a model this server never heard of.
        "ANTHROPIC_SMALL_FAST_MODEL": model,
    }
    for alias in ("OPUS", "SONNET", "HAIKU", "FABLE"):
        env[f"ANTHROPIC_DEFAULT_{alias}_MODEL"] = model
        env[f"ANTHROPIC_DEFAULT_{alias}_MODEL_NAME"] = label
    return env


def integrations(base_url: str, model: str, api_key: str | None) -> list[dict]:
    """Every supported agent, each with a ready-to-use config.

    Each descriptor:
      ``id`` / ``name``      — stable key and display name
      ``protocol``           — "anthropic" or "openai" (which wire format it will use)
      ``summary``            — one line on what this agent is
      ``setup``              — ordered steps, each {kind, ...}:
                                 kind "command" — run this in a shell
                                 kind "file"    — write ``content`` to ``path``
                                 kind "env"     — export these variables
                                 kind "fields"  — fill these into the agent's own settings UI
      ``note``               — caveat worth showing inline (optional)
    """
    openai = _openai_base(base_url)
    key = api_key or "mlx-dspark"

    return [
        {
            "id": "claude-code",
            "name": "Claude Code",
            "protocol": "anthropic",
            "summary": "Anthropic's CLI, running on your local model.",
            "setup": [
                {"kind": "command",
                 "label": "Launch it scoped to this session (nothing is left behind)",
                 "command": _claude_command(base_url, api_key)},
                {"kind": "env",
                 "label": "…or set these yourself",
                 "env": anthropic_env(base_url, model, api_key)},
            ],
            "note": "`mlx-dspark claude` configures only the launched process — your other "
                    "Claude Code sessions and your login are untouched.",
        },
        {
            "id": "codex",
            "name": "Codex CLI",
            "protocol": "openai",
            "summary": "OpenAI's coding CLI, via its custom-provider config.",
            "setup": [
                {"kind": "file",
                 "label": "Add a provider to ~/.codex/config.toml",
                 "path": "~/.codex/config.toml",
                 "content": _codex_toml(openai, model, key)},
                {"kind": "command",
                 "label": "Then run",
                 "command": "codex --profile mlx-dspark"},
            ],
        },
        {
            "id": "opencode",
            "name": "OpenCode",
            "protocol": "openai",
            "summary": "The open-source coding agent, via an OpenAI-compatible provider.",
            "setup": [
                {"kind": "file",
                 "label": "Add to opencode.json (project root or ~/.config/opencode/)",
                 "path": "opencode.json",
                 "content": _opencode_json(openai, model, key)},
            ],
        },
        {
            "id": "pi",
            "name": "pi",
            "protocol": "openai",
            "summary": "The pi coding agent, via a custom provider in models.json.",
            "setup": [
                {"kind": "file",
                 "label": "Add to ~/.pi/models.json",
                 "path": "~/.pi/models.json",
                 "content": _pi_json(openai, model, key)},
            ],
            "note": "pi's system prompt is small (~1.5k tokens vs Claude Code's ~18–26k), so "
                    "it feels much faster on a local model — prefill dominates the clock.",
        },
        {
            "id": "openai-compatible",
            "name": "Cline · Continue · Zed · Open WebUI",
            "protocol": "openai",
            "summary": "Anything that takes an OpenAI base URL. Paste these into its settings.",
            "setup": [
                {"kind": "fields",
                 "label": "Provider settings",
                 "fields": {
                     "Base URL": openai,
                     "API key": key,
                     "Model": model,
                 }},
            ],
        },
    ]


def _claude_command(base_url: str, api_key: str | None) -> str:
    default = base_url.rstrip("/") == "http://127.0.0.1:8080"
    parts = ["mlx-dspark claude"]
    if not default:
        parts.append(f"--url {base_url.rstrip('/')}")
    if api_key:
        parts.append(f"--api-key {api_key}")
    return " ".join(parts)


def _codex_toml(openai_base: str, model: str, key: str) -> str:
    return (
        '[model_providers.mlx-dspark]\n'
        'name = "mlx-dspark"\n'
        f'base_url = "{openai_base}"\n'
        'wire_api = "chat"\n'
        'env_key = "MLX_DSPARK_KEY"\n\n'
        '[profiles.mlx-dspark]\n'
        'model_provider = "mlx-dspark"\n'
        f'model = "{model}"\n'
        f'# export MLX_DSPARK_KEY={key}\n'
    )


def _opencode_json(openai_base: str, model: str, key: str) -> str:
    import json

    return json.dumps({
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "mlx-dspark": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "mlx-dspark",
                "options": {"baseURL": openai_base, "apiKey": key},
                "models": {model: {"name": model}},
            }
        },
    }, indent=2)


def _pi_json(openai_base: str, model: str, key: str) -> str:
    import json

    return json.dumps({
        "providers": {
            "mlx-dspark": {
                "type": "openai",
                "base_url": openai_base,
                "api_key": key,
                "models": [{"id": model, "name": f"{model} (mlx-dspark)"}],
            }
        }
    }, indent=2)
