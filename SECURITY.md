# Security policy

## Reporting a vulnerability

Please **do not** open a public issue for a security problem. Use GitHub's private
reporting: **Security → "Report a vulnerability"** on
https://github.com/ARahim3/mlx-dspark/security/advisories. Reports are acknowledged there and
fixed in a patch release; credit goes to the reporter unless they ask otherwise.

## Threat model (what the server is, and is not, hardened against)

`mlx-dspark serve` is a **local, single-user inference server**. By default it binds
`127.0.0.1` and has no authentication. Anyone who can reach the port can generate text, read
telemetry, and — with the hot-swap server — **load any model repo or local path** via
`POST /admin/load`. Treat the port as equivalent to a shell for the user running it:

- Bind to `0.0.0.0` only together with `--api-key`, and only on a network you trust.
- With `--api-key` set, **every route except `GET /health` requires the key** (`Authorization:
  Bearer <key>` or `x-api-key`). `GET /admin/integrations` returns ready-to-paste agent
  configurations that contain the key — which is why it, like every other route, is behind it.
- **Model-supplied code is refused by default.** A checkpoint can ship Python that
  mlx-lm / transformers would import (`config.json: model_file`, `auto_map` in the model,
  tokenizer or processor configs). `load_target` refuses such checkpoints with the list of
  offending keys. To run one you trust anyway, start with `--trust-remote-code` (or set
  `MLX_DSPARK_TRUST_REMOTE_CODE=1`); there is deliberately no per-request override on
  `/admin/load`, so a client of an authenticated server cannot opt the server into it.
- The drafter loaders (`load_drafter`, `load_dflash`, the GGUF / NVFP4 converters) read
  safetensors, GGUF and JSON only — they never execute checkpoint code.

## Supported versions

Fixes land on the latest release only; there are no backport branches.
