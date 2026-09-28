# The docs site

Source for **https://arahim.dev/mlx-dspark/**. A small custom builder
(`scripts/build_docs.py`: Markdown + Jinja + Pygments, no framework) renders this folder into
`build/site/`; `.github/workflows/docs.yml` publishes it to GitHub Pages on every push to `main`
that touches it.

```bash
uv pip install -e ".[docs]"                  # markdown, pygments, jinja2
python scripts/build_docs.py --serve         # build, then preview at http://127.0.0.1:8000
python scripts/build_docs.py --check         # validate the data only
```

## Layout

| Path | What |
|---|---|
| `site.toml` | Site settings and the sidebar navigation (the order pages appear in). |
| `content/**.md` | One file per page. Front matter: `title`, `description`, optional `nav_title`, `scripts: widgets` (loads the interactive widgets), `layout: wide` (data tables, no right-hand contents), `toc_depth: 2`, `include: FILE`. |
| `data/models.json` | The measured numbers for every registry row. Model pages (`/models/<slug>/`), the models table, the results tables, the picker, the hero race and the chart all render from it. |
| `data/cli.json` | A snapshot of the CLI's argparse parsers; the CLI reference renders from it. |
| `theme/templates/` | `base.html` (chrome), `page.html` (docs layout), `home.html`, `model.html`, `macros.html`, `404.html`. |
| `theme/static/` | `css/site.css` (the whole design system), `js/site.js` (theme, nav, search, tabs, copy), `js/widgets.js` (race, picker, chart, cost model), fonts, media. |
| `demo.gif` | The README's demo; not used by the site (it uses the sharper MP4s in `theme/static/media/`). |
| `social/` | The social preview card: `card.html` is the source, rendered to `theme/static/img/social-card.png` (the pages' `og:image`) and `github-social-preview.png` (upload in the repo's Settings › Social preview). The render commands are at the top of `card.html`. |

Pages are Markdown with Jinja on top, so a page can loop over `models`, `variants`, `cli` or
`registry`. Fenced code blocks are never touched by Jinja. Write links site-absolute
(`/use/server/`); the builder rewrites them relative to each page, so the site works under
any base path.

## Keeping it true

- **A new registry row needs an entry in `data/models.json`**, or the build fails (and so does
  `tests/test_docs_data.py`). Repo, RAM and drafters come from `REGISTRY` and must not be
  repeated there; the file carries only what the registry doesn't: measured numbers per
  arm (`baseline`, `speedup`, `content` chat/code/math, `accept`, `cap`, `tps`), when and how
  they were measured, and the model page's prose (`notes`, Markdown).
- **A CLI flag change needs `python scripts/build_docs.py --refresh-cli`**, or
  `tests/test_docs_data.py` fails.
- Every number on the site comes from the README's measured tables or the CHANGELOG. When a
  model is re-measured, update `data/models.json` and the README tables together.

## Design notes

One visual idea runs through the site: a **draft** is a dashed periwinkle ghost, a **verified**
thing is solid ink (the hero race, the round diagram, the bars, the 404). The feel is soft:
DM Sans at light-to-semibold weights (headlines pair a semibold line with a light gray one),
a near-white page, soft shadows, and the race shown as a dark editor window. Monaspace Neon
for code. Both fonts self-hosted under the SIL OFL. Chart colors (chat / code / math) were validated for colour
vision deficiency in both themes; every chart also uses shape, so colour never carries meaning
alone.
