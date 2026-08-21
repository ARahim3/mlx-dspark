# Contributing to mlx-dspark

Thanks for your interest. mlx-dspark is a small project with a deliberate roadmap, and the
fastest way to get something landed is to line up with how it's built. A few notes up front so
nobody wastes an evening:

## Open an issue first (for anything bigger than a bug fix)

Features, new endpoints, new modes, app screens: **please open an issue before writing the
code.** The engine has a planned direction and several things are already in progress locally,
so a feature PR can land on top of work that's about to ship differently. An issue takes five
minutes and gets you a clear yes / not-now / "here's how it should plug in" before you invest.

Bug fixes with a reproduction don't need an issue — open the PR.

## What makes a PR easy to merge

- **Lossless stays lossless.** The target verifies every token; fp-tie differences are fine,
  greedy/quality shortcuts are not. Say explicitly how your change preserves that.
- **Measured, not reasoned.** Performance claims need a *paired* A/B on one machine, warm,
  with the mlx version and chip named — between-run noise on Apple Silicon is ~±14%, so
  "it should be faster" and single cold runs don't count. Microbench wins have overstated
  end-to-end results here more than once; the end-to-end number is the one that matters.
- **Tests are model-free.** `python -m pytest tests/ -q` runs in CI with no weights; add
  coverage at that level (the fake-engine patterns in `tests/test_server.py` are the template).
  `ruff check src tests` must pass.
- **Server endpoint first.** Anything the Mac app shows must exist as an HTTP endpoint the
  CLI can also use; Swift stays a rendering client.
- **Small and single-purpose.** One fix or one feature per PR, with the *why* in the
  description (a failing case, a measurement, a client that needs the field).

## What to expect from the maintainer

- This is a one-person side project next to a day job, so I can vanish for days at a time.
  You will get a response — even if it's "not now" — once I'm back at it; a quiet week
  means busy, not ignored.
- Some PRs will be taken over, rebased, or re-implemented to fit in-flight work. When that
  happens you'll be credited in the CHANGELOG and release notes — the idea and the report are
  the valuable part.
- Reference implementations ("here's how it could work, rewrite as you see fit") are welcome
  as draft PRs and are read as such.

## Reporting issues

The most useful reports carry: chip + RAM, `mlx-dspark doctor --json` output, the exact
`serve` flags or app version, the model/drafter pair, and — for speed reports — the
`x_mlx_dspark` block from a response (it has decode tok/s, accept length, cap, cached tokens,
and the roofline ratio for *your* machine). `GET /machine` is a good paste too.
