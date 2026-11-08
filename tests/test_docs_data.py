"""The docs site is generated from the code; these keep its inputs honest (model-free).

- ``docs/data/cli.json`` is a snapshot of the argparse parsers the CLI reference renders from.
  If a flag is added, removed or re-documented, refresh it:
  ``python scripts/build_docs.py --refresh-cli``.
- ``docs/data/models.json`` must cover exactly the rows of ``REGISTRY`` (the build enforces
  this too; the test catches it in the main CI job, before a docs deploy would).
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# The sdist ships tests/ but not docs/, and the builder needs tomllib (3.11+).
pytestmark = pytest.mark.skipif(
    not (ROOT / "docs" / "data").is_dir() or sys.version_info < (3, 11),
    reason="docs sources not present (sdist) or Python < 3.11")


def _builder():
    spec = importlib.util.spec_from_file_location("build_docs", ROOT / "scripts" / "build_docs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_snapshot_matches_the_live_parsers():
    b = _builder()
    snapshot = json.loads((ROOT / "docs" / "data" / "cli.json").read_text())
    live = json.loads(json.dumps(b.capture_cli()))  # same JSON round-trip the snapshot took
    assert live == snapshot, (
        "docs/data/cli.json is stale: run `python scripts/build_docs.py --refresh-cli`")


def test_models_json_covers_the_registry():
    from mlx_dspark.load import REGISTRY

    b = _builder()
    assert b.read_registry() == REGISTRY  # the ast read sees what the import sees
    b.load_models(REGISTRY)  # raises BuildError on any mismatch


def test_every_default_arm_has_the_numbers_the_site_charts():
    from mlx_dspark.load import REGISTRY

    doc = _builder().load_models(REGISTRY)
    for model in doc["models"]:
        for v in model["variants"]:
            a = v["best"]
            assert a is not None, f"{model['slug']} {v['quant']}: no default arm"
            for key in ("baseline", "speedup"):
                assert isinstance(a.get(key), (int, float)), f"{model['slug']}: {key}"
            assert set(a["content"]) == {"chat", "code", "math"}, model["slug"]


@pytest.mark.parametrize("page", sorted((ROOT / "docs" / "content").rglob("*.md")))
def test_content_pages_have_titles(page):
    text = page.read_text()
    assert text.startswith("---\n") and "\ntitle:" in text.split("\n---\n", 1)[0], page.name


# ---- table wrappers: one per table, never a stray </div> (the 2026-11-08 layout break) ----

def test_wrap_tables_wraps_bare_tables_once_and_leaves_hand_wrapped_ones_alone():
    b = _builder()
    bare = "<p>a</p>\n<table>\n<tr><td>1</td></tr>\n</table>\n<p>b</p>"
    out = b.wrap_tables(bare)
    assert out.count('<div class="table-wrap">') == 1 and out.count("</div>") == 1
    hand = '<div class="table-wrap"><table class="num">\n<tr><td>1</td></tr>\n</table></div>\n<p>b</p>'
    assert b.wrap_tables(hand) == hand                       # already wrapped: untouched
    both = hand + "\n" + bare
    out = b.wrap_tables(both)
    assert out.count('<div class="table-wrap">') == 2 and out.count("</div>") == 2
    assert "</table></div></div>" not in out


def test_built_pages_keep_their_divs_balanced(tmp_path):
    """Every generated page opens as many <div> as it closes — a stray close pushes the rest of
    the page out of the content column (what happened to results / models / long-context /
    prefill between the site's launch and 2026-11-08)."""
    import re

    b = _builder()
    b.build(tmp_path)
    pages = sorted(tmp_path.rglob("index.html"))
    assert pages, "the build produced no pages"
    for page in pages:
        html = page.read_text(encoding="utf-8")
        opened = len(re.findall(r"<div\b", html))
        closed = html.count("</div>")
        assert opened == closed, f"{page.relative_to(tmp_path)}: {opened} <div> vs {closed} </div>"
        assert "</table></div></div>" not in html, page.relative_to(tmp_path)
