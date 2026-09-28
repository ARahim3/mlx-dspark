#!/usr/bin/env python3
"""Build the mlx-dspark docs site: docs/ -> build/site/ (static HTML for GitHub Pages).

    python scripts/build_docs.py                 # build into build/site/
    python scripts/build_docs.py --serve         # build, then serve on http://127.0.0.1:8000
    python scripts/build_docs.py --check         # validate the data only (CI), write nothing
    python scripts/build_docs.py --refresh-cli   # re-snapshot the CLI flags (needs mlx-dspark installed)

Model-free and mlx-free on purpose, so it runs on any CI runner:

- the model registry is read straight out of ``src/mlx_dspark/load.py`` with ``ast`` (no import),
  and joined with ``docs/data/models.json`` (the measured numbers). The build FAILS when the two
  disagree — a registry row without site data, or site data for a row that no longer exists —
  so "registry == measured pairs == the models page" is enforced mechanically.
- the CLI reference is rendered from ``docs/data/cli.json``, a snapshot of the argparse parsers.
  ``tests/test_docs_data.py`` fails when that snapshot drifts from the live CLI, and
  ``--refresh-cli`` rewrites it.

Deps: markdown, pygments, jinja2 (``pip install -e ".[docs]"``).
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import re
import shutil
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
CONTENT = DOCS / "content"
THEME = DOCS / "theme"
DATA = DOCS / "data"
DEFAULT_OUT = ROOT / "build" / "site"


class BuildError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------- data


def read_registry() -> list[dict]:
    """``REGISTRY`` from load.py without importing it (no mlx needed)."""
    tree = ast.parse((ROOT / "src" / "mlx_dspark" / "load.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "REGISTRY" for t in node.targets):
            return ast.literal_eval(node.value)
    raise BuildError("REGISTRY not found in src/mlx_dspark/load.py")


def read_version() -> str:
    text = (ROOT / "src" / "mlx_dspark" / "__init__.py").read_text()
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.M)
    return m.group(1) if m else "?"


def _ram_gb(ram: str) -> float | None:
    m = re.search(r"([\d.]+)\s*GB", ram or "")
    return float(m.group(1)) if m else None


def load_models(registry: list[dict]) -> dict:
    """models.json joined with the registry, validated both ways."""
    doc = json.loads((DATA / "models.json").read_text())
    by_id = {r["id"]: r for r in registry}
    seen: set[str] = set()
    errors: list[str] = []
    for model in doc["models"]:
        for v in model["variants"]:
            rid = v.get("registry")
            if rid:
                if rid not in by_id:
                    errors.append(f"models.json: {model['slug']} variant names registry id "
                                  f"{rid!r}, which is not in REGISTRY")
                    continue
                if rid in seen:
                    errors.append(f"models.json: registry id {rid!r} is used twice")
                seen.add(rid)
                row = by_id[rid]
                # the registry is the source of truth for repos and RAM — never duplicated
                for key in ("target", "ram"):
                    if key in v and v[key] != row[key]:
                        errors.append(f"models.json: {rid}.{key} = {v[key]!r} disagrees with "
                                      f"REGISTRY ({row[key]!r}); drop it from models.json")
                v["target"] = row["target"]
                v["ram"] = row["ram"]
                v["registry_speedup"] = row.get("speedup")
                v["default_mode"] = row.get("mode", "dspark")
                v["lookup_drafts"] = row.get("lookup_drafts", True)
                v["drafters"] = {k: row[k] for k in ("dspark", "dflash") if row.get(k)}
            else:
                for key in ("target", "ram"):
                    if key not in v:
                        errors.append(f"models.json: {model['slug']} unregistered variant "
                                      f"needs an explicit {key!r}")
                v.setdefault("default_mode", "dspark")
                v.setdefault("drafters", {})
            if v.get("ram_label"):
                # a registry RAM string can cover several quants ("~26 GB (4-bit) / ~40 GB
                # (8-bit)"); the page shows this variant's own share
                v["ram"] = v["ram_label"]
            v["ram_gb"] = _ram_gb(v.get("ram", ""))
            if "ram_gb_override" in v:
                v["ram_gb"] = v["ram_gb_override"]
            v["served_id"] = v["target"].rstrip("/").split("/")[-1]
            arms = v.get("arms") or []
            defaults = [a for a in arms if a.get("default")]
            if arms and len(defaults) != 1:
                errors.append(f"models.json: {model['slug']} {v['quant']}: exactly one arm must "
                              f"be marked default (found {len(defaults)})")
            for a in arms:
                if a.get("baseline") and a.get("speedup") and not a.get("tps"):
                    a["tps"] = round(a["baseline"] * a["speedup"], 1)
                c = a.get("content") or {}
                if a.get("baseline") and c:
                    lo, hi = min(c.values()), max(c.values())
                    a["tps_range"] = [round(a["baseline"] * lo), round(a["baseline"] * hi)]
            v["best"] = defaults[0] if defaults else None
    missing = sorted(set(by_id) - seen)
    if missing:
        errors.append("REGISTRY rows with no entry in docs/data/models.json: "
                      + ", ".join(missing) + " — add their measured numbers there")
    if errors:
        raise BuildError("\n".join(errors))
    return doc


def load_cli() -> dict:
    return json.loads((DATA / "cli.json").read_text())


# The subcommands in the order the reference page shows them.
CLI_COMMANDS = ("serve", "claude", "generate", "benchmark", "models", "doctor")


def capture_cli() -> dict:
    """Snapshot every subcommand's argparse parser (flags, help, defaults) as plain data."""
    from mlx_dspark import cli as cli_mod

    class _Captured(Exception):
        def __init__(self, parser):
            self.parser = parser

    def _raise(self, *a, **k):
        raise _Captured(self)

    out = {}
    orig = (argparse.ArgumentParser.parse_args, argparse.ArgumentParser.parse_known_args)
    argparse.ArgumentParser.parse_args = _raise
    argparse.ArgumentParser.parse_known_args = _raise
    try:
        for name in CLI_COMMANDS:
            try:
                getattr(cli_mod, f"cmd_{name}")([])
            except _Captured as c:
                out[name] = _parser_data(c.parser)
            else:
                raise BuildError(f"cmd_{name} never parsed its arguments")
    finally:
        argparse.ArgumentParser.parse_args, argparse.ArgumentParser.parse_known_args = orig
    return out


def _parser_data(p: argparse.ArgumentParser) -> dict:
    opts = []
    for a in p._actions:
        if isinstance(a, argparse._HelpAction):
            continue
        is_flag = a.nargs == 0
        default = a.default
        if default is argparse.SUPPRESS or (is_flag and default in (False, None)):
            default = None
        opts.append({
            "flags": list(a.option_strings),
            "metavar": None if is_flag else (a.metavar or a.dest.upper()),
            "choices": list(a.choices) if a.choices else None,
            "default": default if isinstance(default, (str, int, float, bool)) else None,
            "help": " ".join((a.help or "").split()) if a.help != argparse.SUPPRESS else "",
        })
    return {"prog": p.prog, "description": p.description or "", "epilog": p.epilog or "",
            "options": opts}


# ------------------------------------------------------------------------------- markdown

FENCE_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<fence>```|~~~)(?P<lang>[\w+-]*)[ \t]*(?P<attrs>[^\n]*)\n"
    r"(?P<code>.*?)\n(?P=indent)(?P=fence)[ \t]*$",
    re.M | re.S)
LANG_ALIASES = {"sh": "bash", "shell": "bash", "zsh": "bash", "py": "python", "": "text",
                "jsonc": "json"}


def highlight(code: str, lang: str) -> str:
    from pygments import highlight as pyg
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import get_lexer_by_name
    from pygments.util import ClassNotFound

    lang = LANG_ALIASES.get(lang, lang)
    try:
        lexer = get_lexer_by_name(lang)
    except ClassNotFound:
        lexer = get_lexer_by_name("text")
    return pyg(code, lexer, HtmlFormatter(nowrap=True))


def code_block(code: str, lang: str, title: str | None = None) -> str:
    lang_label = {"bash": "shell", "text": "", "": ""}.get(LANG_ALIASES.get(lang, lang),
                                                            LANG_ALIASES.get(lang, lang))
    head = ""
    if title:
        head = f'<div class="code-title">{html.escape(title)}</div>'
    return (f'<div class="code" data-lang="{html.escape(lang_label)}">{head}'
            f'<pre><code>{highlight(code, lang)}</code></pre>'
            f'<button class="copy" type="button" aria-label="Copy to clipboard">'
            f'<span>Copy</span></button></div>')


def stash_fences(text: str, stash: list[str]) -> str:
    """Replace fenced code blocks with placeholders so neither Jinja nor Markdown touches
    them; they come back as highlighted HTML after Markdown has run."""
    def repl(m: re.Match) -> str:
        indent = m.group("indent")
        code = m.group("code")
        if indent:
            code = "\n".join(line[len(indent):] if line.startswith(indent) else line
                             for line in code.split("\n"))
        title = None
        t = re.search(r'title="([^"]*)"', m.group("attrs") or "")
        if t:
            title = t.group(1)
        stash.append(code_block(code, m.group("lang"), title))
        return f"{indent}@@CODE{len(stash) - 1}@@"
    return FENCE_RE.sub(repl, text)


def unstash(html_text: str, stash: list[str]) -> str:
    html_text = re.sub(r"<p>\s*@@CODE(\d+)@@\s*</p>", lambda m: stash[int(m.group(1))], html_text)
    return re.sub(r"@@CODE(\d+)@@", lambda m: stash[int(m.group(1))], html_text)


def split_front_matter(text: str) -> tuple[dict, str]:
    meta: dict = {}
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            for line in text[4:end].splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip().strip('"')
            text = text[end + 5:]
    return meta, text


def make_markdown():
    import markdown

    return markdown.Markdown(
        extensions=["tables", "attr_list", "md_in_html", "footnotes", "def_list", "admonition",
                    "sane_lists", "toc"],
        extension_configs={"toc": {"permalink": "#", "permalink_class": "anchor",
                                   "permalink_title": "Link to this section", "toc_depth": "2-3"}},
        output_format="html",
    )


def render_markdown(text: str, jenv, ctx: dict) -> tuple[str, list[dict]]:
    stash: list[str] = []
    text = stash_fences(text, stash)
    text = jenv.from_string(text).render(**ctx)
    md = make_markdown()
    body = md.convert(text)
    body = unstash(body, stash)
    body = re.sub(r"<table>", '<div class="table-wrap"><table>', body)
    body = re.sub(r"</table>", "</table></div>", body)
    toc = [{"id": t["id"], "name": _strip_tags(t["name"]),
            "children": [{"id": c["id"], "name": _strip_tags(c["name"])} for c in t["children"]]}
           for t in md.toc_tokens]
    # toc_tokens start at the shallowest heading present; pages have no body h1
    return body, toc


def _strip_tags(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s))


def relativize(html_text: str, root: str) -> str:
    """Site-absolute links (/start/install/) -> relative to the page, so the site works
    under any base path (a GitHub project page, a custom domain, or a local server)."""
    return re.sub(r'(href|src|srcset|poster|content|data-src|data-root)="/(?!/)',
                  lambda m: f'{m.group(1)}="{root}', html_text)


# ------------------------------------------------------------------------------ search

def search_sections(html_text: str, page_title: str, url: str) -> list[dict]:
    """Split a rendered page into h2/h3 sections for the client-side search index. CLI flags
    (``<div class="flag" id=…>``) get an entry each, so ``--kv-bits`` finds its own line."""
    out = []
    for fm in re.finditer(r'<div class="flag" id="([^"]+)">(.*?)</div>', html_text, flags=re.S):
        dt = re.search(r"<code>(.*?)</code>", fm.group(2), flags=re.S)
        dd = re.search(r"<dd>(.*?)</dd>", fm.group(2), flags=re.S)
        cmd = fm.group(1).split("-", 1)[0]
        out.append({"p": f"{page_title}: mlx-dspark {cmd}",
                    "h": _strip_tags(dt.group(1)) if dt else fm.group(1),
                    "u": f"{url}#{fm.group(1)}",
                    "x": re.sub(r"\s+", " ", _strip_tags(dd.group(1))).strip()[:400] if dd else ""})
    html_text = re.sub(r'<dl class="flags">.*?</dl>', "", html_text, flags=re.S)
    parts = re.split(r'(<h[23][^>]*id="[^"]+"[^>]*>.*?</h[23]>)', html_text, flags=re.S)
    heading, anchor = page_title, ""
    for part in parts:
        hm = re.match(r'<h[23][^>]*id="([^"]+)"[^>]*>(.*?)</h[23]>', part, flags=re.S)
        if hm:
            anchor = hm.group(1)
            heading = _strip_tags(re.sub(r'<a class="anchor".*?</a>', "", hm.group(2))).strip()
            continue
        text = _strip_tags(re.sub(r"<(script|style)[^>]*>.*?</\1>", "", part, flags=re.S))
        text = re.sub(r"\s+", " ", text).strip()
        if not text and not anchor:
            continue
        out.append({"p": page_title, "h": heading if heading != page_title else "",
                    "u": url + (f"#{anchor}" if anchor else ""), "x": text[:600]})
    return out


# ------------------------------------------------------------------------------- build

def fmt_x(v) -> str:
    return "—" if v is None else f"{v:.2f}×"


def build(out: Path, check_only: bool = False) -> None:
    import jinja2

    site = tomllib.loads((DOCS / "site.toml").read_text())
    registry = read_registry()
    models = load_models(registry)
    cli = load_cli()
    missing = [c for c in CLI_COMMANDS if c not in cli]
    if missing:
        raise BuildError(f"docs/data/cli.json lacks {missing}; run --refresh-cli")
    version = read_version()

    # nav: resolve every page's title from its front matter
    pages: dict[str, dict] = {}
    nav = []
    for section in site["nav"]:
        items = []
        for slug in section["pages"]:
            src = CONTENT / f"{slug}.md"
            if not src.exists():
                raise BuildError(f"nav lists {slug}, but {src.relative_to(ROOT)} is missing")
            meta, body = split_front_matter(src.read_text())
            if meta.get("include"):
                # a page that wraps a repo file (the changelog): drop that file's own H1
                included = (ROOT / meta["include"]).read_text()
                body += "\n\n" + re.sub(r"\A# .*\n", "", included)
            url = f"/{slug.removesuffix('/index').removesuffix('index')}/".replace("//", "/")
            page = {"slug": slug, "url": url, "title": meta.get("title", slug),
                    "nav_title": meta.get("nav_title", meta.get("title", slug)),
                    "description": meta.get("description", ""), "body": body,
                    "section": section["title"], "meta": meta}
            pages[slug] = page
            items.append(page)
        nav.append({"title": section["title"], "pages": items})
    order = [p for s in nav for p in s["pages"]]
    for i, p in enumerate(order):
        p["prev"] = order[i - 1] if i > 0 else None
        p["next"] = order[i + 1] if i + 1 < len(order) else None
    if check_only:
        print(f"ok: {len(registry)} registry rows, {len(models['models'])} models, "
              f"{len(order)} pages, {len(cli)} CLI commands")
        return

    jenv = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(THEME / "templates")),
        autoescape=jinja2.select_autoescape(["html"]),
        comment_start_string="{##", comment_end_string="##}",
        trim_blocks=True, lstrip_blocks=True,
    )
    jenv.filters["x"] = fmt_x
    # "</" can't appear inside an inline <script> block
    jenv.filters["json"] = lambda v: json.dumps(v, ensure_ascii=False).replace("</", "<\\/")
    jenv.globals["highlight_block"] = code_block
    # CLI help text: escape, then render `backtick spans` as code
    jenv.filters["ticks"] = lambda s: re.sub(
        r"`([^`]+)`", r"<code>\1</code>", html.escape(s or "", quote=False))
    # content markdown is rendered by a sibling env that sees the same macros
    menv = jenv.overlay(autoescape=False)

    all_models = models["models"]
    variants = [dict(v, model=m) for m in all_models for v in m["variants"]]
    ctx_base = {"site": site, "nav": nav, "version": version, "models": all_models,
                "variants": variants, "registry": registry, "cli": cli,
                "cli_commands": CLI_COMMANDS, "stamp": models.get("stamp", {})}

    # client-side data: the picker, race, charts and builders all read this one blob
    client_models = []
    for m in all_models:
        for v in m["variants"]:
            client_models.append({
                "slug": m["slug"], "name": m["name"], "vendor": m.get("vendor"),
                "params": m.get("params"), "params_b": m.get("params_b"),
                "arch": m.get("arch"), "kind": m.get("kind"), "tools": m.get("tools", False),
                "thinking": m.get("thinking", False), "quant": v["quant"],
                "target": v["target"], "served": v["served_id"], "ram": v["ram"],
                "ram_gb": v["ram_gb"], "mode": v["default_mode"],
                "recommended": v.get("recommended", True),
                "caveat": v.get("caveat") or m.get("caveat"),
                "arms": v.get("arms", []), "measured": v.get("measured"),
                "picker_note": v.get("picker_note"),
            })
    ctx_base["client_models"] = client_models

    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(THEME / "static", out / "static")
    for extra in site.get("copy", []):
        src, dst = ROOT / extra["from"], out / extra["to"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst)

    index: list[dict] = []

    def write(url: str, html_text: str) -> None:
        depth = url.strip("/").count("/") + (1 if url.strip("/") else 0)
        root = "../" * depth
        html_text = relativize(html_text, root)
        dest = out / url.strip("/") / "index.html" if url.strip("/") else out / "index.html"
        if url.endswith(".html"):
            dest = out / url.strip("/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(html_text)

    # content pages
    for page in order:
        ctx = dict(ctx_base, page=page)
        body, toc = render_markdown(page["body"], menv, ctx)
        if page["meta"].get("toc_depth") == "2":
            toc = [dict(t, children=[]) for t in toc]
        page["html"], page["toc"] = body, toc
        tpl = jenv.get_template(page["meta"].get("template", "page.html"))
        write(page["url"], tpl.render(**ctx, content=body, toc=toc))
        index.extend(search_sections(body, page["title"], page["url"]))

    # one page per model
    for m in all_models:
        notes_html, toc = render_markdown(m.get("notes", ""), menv, ctx_base) if m.get(
            "notes") else ("", [])
        url = f"/models/{m['slug']}/"
        page = {"title": m["name"], "url": url, "section": "Models",
                "description": m.get("summary", ""), "slug": f"models/{m['slug']}"}
        html_text = jenv.get_template("model.html").render(
            **ctx_base, page=page, model=m, notes=notes_html, toc=toc,
            edit_path="docs/data/models.json")
        write(url, html_text)
        index.append({"p": m["name"], "h": "", "u": url,
                      "x": f"{m.get('summary', '')} {' '.join(v['target'] for v in m['variants'])}"})
        index.extend(search_sections(notes_html, m["name"], url))

    # home + 404
    home = {"title": site["site"]["name"], "url": "/", "section": "",
            "description": site["site"]["description"], "slug": ""}
    write("/", jenv.get_template("home.html").render(**ctx_base, page=home))
    nf = {"title": "Page not found", "url": "/404.html", "section": "", "description": "",
          "slug": "404"}
    # 404 is served from any depth, so its links must be absolute to the site root
    html404 = jenv.get_template("404.html").render(**ctx_base, page=nf)
    base = site["site"].get("base_path", "/")
    (out / "404.html").write_text(re.sub(r'(href|src|poster|data-root)="/(?!/)',
                                         lambda m: f'{m.group(1)}="{base}', html404))

    (out / "static" / "search-index.json").write_text(
        json.dumps(index, ensure_ascii=False, separators=(",", ":")))
    urls = ["/"] + [p["url"] for p in order] + [f"/models/{m['slug']}/" for m in all_models]
    site_url = site["site"]["url"].rstrip("/")
    (out / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "".join(f"  <url><loc>{site_url}{u}</loc></url>\n" for u in urls)
        + "</urlset>\n")
    (out / ".nojekyll").write_text("")
    print(f"built {len(urls)} pages -> {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true", help="validate data only; write nothing")
    ap.add_argument("--refresh-cli", action="store_true",
                    help="re-snapshot docs/data/cli.json from the installed CLI, then build")
    ap.add_argument("--serve", action="store_true", help="serve the built site on :8000")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    try:
        if args.refresh_cli:
            data = capture_cli()
            (DATA / "cli.json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
            print(f"wrote docs/data/cli.json ({len(data)} commands)")
        build(args.out, check_only=args.check)
    except BuildError as e:
        sys.exit(f"docs build failed:\n{e}")
    if args.serve and not args.check:
        import functools
        import http.server

        handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                    directory=str(args.out))
        print(f"serving on http://127.0.0.1:{args.port}/")
        http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
