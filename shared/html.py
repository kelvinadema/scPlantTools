#!/usr/bin/env python3

"""
Shared scaffolding for the pipeline's self-contained HTML reports.

Three stages now write one: mapping's per-sample QC report, annotation's
stage-1 review page, and (next) refinement's comparison page. They share a
stylesheet, a page skeleton and the handful of formatters that decide how an
unknown value renders -- which is not cosmetic. A number the mapping report
shows as `&mdash;` and the review page shows as `0` are two different claims
about the data, and the only way those stay in agreement is to format them in
one place.

Extracted verbatim from `mapping/report.py`, which now imports from
here. Everything the reports are constrained by holds unchanged:

  - **stdlib only.** jinja2 is importable in the mapping environment purely as
    a CellBender transitive; depending on it would be an undeclared dependency,
    and the annotation environment has no reason to carry it at all.
  - **no CDN, no external asset.** Figures are inlined as data URIs so a report
    survives being copied off the cluster, which is the main way anyone
    actually reads one.
  - **light and dark.** The pages are read in both.
"""

import base64
import html

from pathlib import Path


_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin: 0; padding: 2rem 1.5rem 4rem; font: 15px/1.6 -apple-system, BlinkMacSystemFont,
       "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; color: #1a1a1a; background: #fff;
       max-width: 1100px; margin-inline: auto; }
h1 { font-size: 1.6rem; margin: 0 0 .25rem; letter-spacing: -.01em; }
h2 { font-size: 1.15rem; margin: 2.5rem 0 .75rem; padding-bottom: .4rem;
     border-bottom: 1px solid #e5e5e5; letter-spacing: -.01em; }
h3 { font-size: .95rem; margin: 1.5rem 0 .5rem; color: #444; font-weight: 600; }
.sub { color: #666; margin: 0 0 1.5rem; font-size: .9rem; }
.cards { display: flex; flex-wrap: wrap; gap: .75rem; margin: 1rem 0 1.5rem; }
.card { flex: 1 1 150px; border: 1px solid #e5e5e5; border-radius: 6px; padding: .75rem .9rem; }
.card .k { font-size: .72rem; text-transform: uppercase; letter-spacing: .05em; color: #777; }
.card .v { font-size: 1.35rem; font-weight: 600; margin-top: .15rem; }
.card .v small { font-size: .8rem; font-weight: 400; color: #777; }
table { border-collapse: collapse; width: 100%; margin: .5rem 0 1rem; font-size: .88rem; }
th, td { text-align: left; padding: .45rem .6rem; border-bottom: 1px solid #ededed; }
th { font-weight: 600; color: #555; background: #fafafa; white-space: nowrap; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
figure { margin: 1rem 0 1.5rem; }
figure img { width: 100%; height: auto; border: 1px solid #e5e5e5; border-radius: 6px; }
figcaption { font-size: .8rem; color: #666; margin-top: .4rem; }
.note { font-size: .85rem; color: #777; background: #fafafa; border-left: 3px solid #ddd;
        padding: .6rem .8rem; margin: .75rem 0; border-radius: 0 4px 4px 0; }
.banner { font-size: .9rem; color: #7a4a00; background: #fff6e5; border: 1px solid #f0d38a;
          padding: .8rem 1rem; margin: 1rem 0 1.5rem; border-radius: 6px; }
.banner ul { margin: .4rem 0; padding-left: 1.3rem; }
.banner p { margin: .5rem 0 0; }
.missing { color: #999; font-style: italic; }
.bar { background: #eee; border-radius: 3px; height: 1.5rem; position: relative; overflow: hidden; }
.bar > span { display: block; height: 100%; background: #4a7ebb; }
.pill { display: inline-block; padding: .1rem .5rem; border-radius: 999px; font-size: .75rem;
        font-weight: 600; }
.pill.ok { background: #e3f2e3; color: #256029; }
.pill.bad { background: #fdecea; color: #9b1c1c; }
.pill.advisory { background: #eaf0fa; color: #2c4f8a; font-weight: 500; }
details { margin: .5rem 0; }
summary { cursor: pointer; font-size: .85rem; color: #555; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .82em;
       background: #f4f4f4; padding: .1em .35em; border-radius: 3px; word-break: break-all; }
a { color: #2c5aa0; }
@media (prefers-color-scheme: dark) {
  body { background: #16181c; color: #e6e6e6; }
  h2 { border-bottom-color: #2e3238; }
  h3 { color: #b9b9b9; }
  .sub, .card .k, figcaption, .note, summary { color: #9aa0a6; }
  .card, figure img { border-color: #2e3238; }
  th { background: #1e2126; color: #b9b9b9; }
  th, td { border-bottom-color: #262a30; }
  .note { background: #1a1d21; border-left-color: #3a3f46; }
  .banner { background: #2b2210; border-color: #6b5420; color: #e8c777; }
  .bar { background: #262a30; }
  code { background: #22262c; }
  a { color: #7aa7e6; }
  .pill.advisory { background: #1e2a3d; color: #9dbdf0; }
}
"""

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""


def _esc(value) -> str:
    """HTML-escape a value, rendering unknowns explicitly rather than as blanks."""
    if value is None or value == '':
        return '<span class="missing">unknown</span>'
    return html.escape(str(value))


def _fmt_int(value) -> str:
    """Format an integer with thousands separators, or an em dash if unknown.

    Args:
        value: The value.

    Returns:
        str: HTML-safe text.
    """
    try:
        return '{:,}'.format(int(value))
    except (TypeError, ValueError):
        return '&mdash;'


def _fmt_float(value, digits: int=3) -> str:
    """Format a float to `digits` places, or an em dash if unknown.

    Args:
        value: The value.
        digits (int): Decimal places.

    Returns:
        str: HTML-safe text.
    """
    try:
        if value is None:
            return '&mdash;'
        return '{:.{d}f}'.format(float(value), d=digits)
    except (TypeError, ValueError):
        return '&mdash;'


def _embed_png(path: Path) -> str:
    """
    Inline a PNG as a data URI so the report survives being copied off the cluster.

    Returns '' when the file is absent: QCPlotter logs and swallows plot failures,
    so a missing figure is a normal outcome rather than an error.
    """
    try:
        data = Path(path).read_bytes()
    except OSError:
        return ''
    return 'data:image/png;base64,{}'.format(base64.b64encode(data).decode('ascii'))


def _figure(path: Path, caption: str) -> str:
    """A figure block, or a "not available" note when the file is absent.

    Args:
        path (Path): The image.
        caption (str): Caption text.

    Returns:
        str: HTML fragment.
    """
    uri = _embed_png(path)
    if not uri:
        return '<p class="missing">Figure not available: {}</p>'.format(html.escape(Path(path).name))
    return ('<figure><img alt="{alt}" src="{uri}">'
            '<figcaption>{cap}</figcaption></figure>').format(
        alt=html.escape(caption), uri=uri, cap=html.escape(caption))
