---
title: CLI reference
description: Every subcommand and flag, generated from the command-line parser itself, so it can't drift from the code.
toc_depth: 2
---

For a guided tour, see [Command line](/use/command-line/). Most flags have measured defaults: when a flag says *unset*, the engine uses the value calibrated for this Mac and model, or the value measured best for this pair.

{% set fallback = {
  "--prompt": "The prompt. It goes through the model's chat template unless `--no-chat-template` is set.",
  "--max-new-tokens": "Maximum tokens to generate.",
  "--seed": "Random seed for sampled decoding (`--temperature` above 0).",
  "--confidence-threshold": "DSpark: stop drafting early in a round when the confidence head's cumulative survival estimate drops below this (0 = off). Pays only where the verify curve still rises inside the cap; see Draft caps.",
  "--drafter-bits": "Quantize the drafter to this many bits at load (default 4). Doesn't change acceptance, only drafter speed.",
  "--no-chat-template": "Send the prompt as raw text, without the model's chat template.",
  "--no-stream": "Print the answer once at the end instead of streaming it.",
  "--host": "Interface to bind (default 127.0.0.1). Use 0.0.0.0 with `--api-key` to serve other devices.",
  "--port": "Port to listen on (default 8080).",
  "--drafter": "Drafter repo or path, overriding auto-resolution."
} %}
{% for name in cli_commands %}
{% set c = cli[name] %}
## `mlx-dspark {{ name }}` {#{{ name }}}

{% set about = {
  "serve": "The API server: OpenAI Chat Completions and Completions, Anthropic Messages, and OpenAI Responses on one port. See [Run a server](/use/server/).",
  "generate": "One prompt, one answer, streamed to the terminal. Also what a bare `mlx-dspark --prompt …` runs.",
  "benchmark": "A warm, reproducible speed sweep on this Mac: plain decoding against the speculative modes, over a chat, a code and a math prompt.",
  "models": "List the measured models whose drafters resolve automatically, with their memory.",
  "doctor": "Check the environment: Apple Silicon, the MLX stack, memory, and every folder searched for models."
} %}
{% if about.get(name) %}
{{ about[name] }}
{% elif c.description %}<p>{{ c.description | ticks }}</p>{% endif %}
{% if c.options %}
<dl class="flags">
{% for o in c.options %}
{% set fid = name ~ '-' ~ (o.flags[0] | replace('--', '') ) %}
<div class="flag" id="{{ fid }}">
<dt><a href="#{{ fid }}"><code>{{ o.flags | join(', ') }}{% if o.metavar %} {{ o.metavar }}{% endif %}</code></a>{% if o.choices %} <span class="flag-meta">one of {{ o.choices | join(', ') }}</span>{% endif %}{% if o.default is not none and o.default != '' %} <span class="flag-meta">default {{ o.default }}</span>{% endif %}</dt>
<dd>{{ (o.help or fallback.get(o.flags[0], '')) | ticks }}</dd>
</div>
{% endfor %}
</dl>
{% else %}
<p>No options.</p>
{% endif %}
{% if c.epilog %}<p>{{ c.epilog | ticks }}</p>{% endif %}

{% endfor %}
