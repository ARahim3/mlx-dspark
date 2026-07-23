"""Agent integration config — model-free (pure config generation)."""

import json

from mlx_dspark import integrations as I


def rows(base="http://127.0.0.1:8080", model="Qwen3-8B-8bit", key=None):
    return I.integrations(base, model, key)


class TestCoverage:
    def test_covers_the_agents_the_readme_names(self):
        ids = {r["id"] for r in rows()}
        assert {"claude-code", "codex", "opencode", "pi", "openai-compatible"} <= ids

    def test_every_row_has_at_least_one_setup_step(self):
        for r in rows():
            assert r["setup"], f"{r['id']} has no setup steps"

    def test_protocol_is_anthropic_only_for_claude_code(self):
        by_id = {r["id"]: r for r in rows()}
        assert by_id["claude-code"]["protocol"] == "anthropic"
        for other in ("codex", "opencode", "pi", "openai-compatible"):
            assert by_id[other]["protocol"] == "openai"


class TestClaudeCode:
    def test_env_sets_base_url_auth_and_the_haiku_slot(self):
        env = I.anthropic_env("http://127.0.0.1:8080", "Qwen3-8B-8bit", None)
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8080"
        # Without an auth token Claude Code keeps billing the saved login — must always be set.
        assert env["ANTHROPIC_AUTH_TOKEN"] == "mlx-dspark"
        # The haiku slot is where Claude Code sends background tasks; unmapped, they ask for a
        # model this server never heard of.
        assert env["ANTHROPIC_SMALL_FAST_MODEL"] == "Qwen3-8B-8bit"
        assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "Qwen3-8B-8bit"

    def test_api_key_flows_into_the_token(self):
        env = I.anthropic_env("http://x", "m", "secret")
        assert env["ANTHROPIC_AUTH_TOKEN"] == "secret"

    def test_command_is_bare_for_the_default_url(self):
        cc = next(r for r in rows() if r["id"] == "claude-code")
        command = cc["setup"][0]["command"]
        assert command == "mlx-dspark claude"

    def test_command_carries_url_and_key_when_non_default(self):
        cc = next(r for r in rows(base="http://192.168.1.9:9000", key="abc")
                  if r["id"] == "claude-code")
        command = cc["setup"][0]["command"]
        assert "--url http://192.168.1.9:9000" in command
        assert "--api-key abc" in command


class TestOpenAIAgents:
    def test_all_openai_agents_use_the_v1_base(self):
        for r in rows():
            if r["protocol"] != "openai":
                continue
            blob = json.dumps(r["setup"])
            # the /v1 suffix is the single most common thing to get wrong
            assert "http://127.0.0.1:8080/v1" in blob, f"{r['id']} missing /v1 base"

    def test_codex_toml_is_valid_and_names_the_model(self):
        codex = next(r for r in rows() if r["id"] == "codex")
        toml = codex["setup"][0]["content"]
        assert "[model_providers.mlx-dspark]" in toml
        assert 'model = "Qwen3-8B-8bit"' in toml
        assert codex["setup"][0]["path"] == "~/.codex/config.toml"

    def test_opencode_and_pi_emit_valid_json(self):
        for agent_id in ("opencode", "pi"):
            agent = next(r for r in rows() if r["id"] == agent_id)
            content = agent["setup"][0]["content"]
            parsed = json.loads(content)              # must parse
            assert "Qwen3-8B-8bit" in json.dumps(parsed)

    def test_openai_compatible_exposes_the_three_fields_a_gui_asks_for(self):
        generic = next(r for r in rows() if r["id"] == "openai-compatible")
        fields = generic["setup"][0]["fields"]
        assert fields["Base URL"] == "http://127.0.0.1:8080/v1"
        assert fields["Model"] == "Qwen3-8B-8bit"
        assert fields["API key"] == "mlx-dspark"       # placeholder when none set


class TestKeyPlaceholder:
    def test_missing_key_becomes_a_placeholder_not_empty(self):
        """An empty api key field makes an OpenAI client send no Authorization header, which
        some servers reject outright — a placeholder is safer than blank."""
        generic = next(r for r in rows(key=None) if r["id"] == "openai-compatible")
        assert generic["setup"][0]["fields"]["API key"] == "mlx-dspark"
