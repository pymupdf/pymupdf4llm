"""Canonical text source for chunking: render one LayoutBox to markdown.

This module is the single place where the chunking package calls the
markdown renderers in ``document_layout``.  ``box_to_markdown`` mirrors the
per-box dispatch of ``ParsedDocument.to_markdown`` so that chunk text is the
same markdown the document renderer emits for that box — upstream renderer
signature drift is absorbed here and nowhere else.

Documented deviations from ``to_markdown``:

- picture/formula boxes without text are not rendered here at all; the
  chunker represents them with a placeholder (see ``figure_placeholder``).
- formula boxes *with* textlines are rendered via ``text_to_md``.
  ``to_markdown`` drops that text (it only emits the image reference), but
  dropping it from chunks would lose indexable content.
- image files / base64 payloads are never inlined into chunk text; images
  travel in metadata.
"""

from ..document_layout import (
    create_list_item_levels,
    footnote_to_md,
    list_item_to_md,
    picture_text_to_md,
    section_hdr_to_md,
    text_to_md,
    title_to_md,
)

__all__ = [
    "box_to_markdown",
    "create_list_item_levels",
    "extract_table_headers",
    "figure_placeholder",
    "table_content",
]


def table_content(box):
    """Adapter for ``box.table`` — the only place chunking reads its schema.

    Returns ``(markdown, html)`` where at most one side is a string,
    mirroring the ``to_markdown`` branching: when the parse ran with HTML
    table rendering, the table dict carries the canonical rendering under
    ``"html"`` and ``markdown`` is ``None``; otherwise ``markdown`` is the
    canonical rendering (possibly ``""`` for a degenerate table) and
    ``html`` is ``None``.

    Routing uses ``is not None`` on purpose: a degenerate table carries
    ``markdown == ""`` and must still route as markdown, never as html.
    """
    table = box.table if isinstance(box.table, dict) else {}
    html = table.get("html")
    if html is None:
        # Defensive: accept per-table entries even if the aggregate "html"
        # key is absent (D13 — header info lives in these renderings).
        per_table = table.get("html_tables")
        if per_table:
            html = "\n\n".join(
                t.get("html", "") for t in per_table if isinstance(t, dict)
            ) or None
    if html is not None:
        return None, html
    return table.get("markdown"), None


def extract_table_headers(html):
    """Header cell texts from a table's HTML rendering (D13).

    Header identity comes exclusively from ``<th>`` elements emitted by the
    table engine — no local header heuristics.  Returns ``[]`` when the
    rendering carries no ``<th>`` (including all markdown-mode parses).
    """
    if not html:
        return []

    from html.parser import HTMLParser

    class _THCollector(HTMLParser):
        def __init__(self):
            super().__init__()
            self._depth = 0
            self._parts = []
            self.headers = []

        def handle_starttag(self, tag, attrs):
            if tag == "th":
                self._depth += 1
                self._parts = []

        def handle_endtag(self, tag):
            if tag == "th" and self._depth:
                self._depth -= 1
                self.headers.append("".join(self._parts).strip())

        def handle_data(self, data):
            if self._depth:
                self._parts.append(data)

    collector = _THCollector()
    collector.feed(html)
    return collector.headers


def _copy_textlines(textlines):
    """Copy textlines deep enough to isolate renderer side effects.

    Some renderers mutate their input in place (``footnote_to_md`` extends
    the first line's span list, ``list_item_to_md`` rewrites/pops spans), so
    rendering the same box twice — e.g. ``to_chunks`` followed by
    ``to_markdown`` — would duplicate text.  Rendering from a copy keeps
    ``box_to_markdown`` free of side effects on the parsed document.
    """
    return [
        {**tl, "spans": [dict(s) for s in tl.get("spans", [])]}
        for tl in textlines
    ]


def figure_placeholder(fig_number, box):
    """Stable text placeholder for a figure/formula box without text."""
    w, h = int(box.x1 - box.x0), int(box.y1 - box.y0)
    return f"[Figure f{fig_number}: {w}x{h}]"


def box_to_markdown(page, box, box_idx, list_item_levels=None, ignore_code=False):
    """Render one LayoutBox to markdown, mirroring ``to_markdown`` dispatch.

    Args:
        page: the PageLayout owning the box (``full_ocred`` is honored).
        box: the LayoutBox to render.
        box_idx: index of the box on its page (list-item level lookup).
        list_item_levels: optional precomputed ``create_list_item_levels``
            result for the page; computed on demand when omitted.
        ignore_code: suppress code-block styling (as in ``to_markdown``).

    Returns the markdown string for the box ("" when there is nothing to
    render).  Trailing block separators ("\\n\\n") are preserved as emitted
    by the renderers; callers strip as needed.
    """
    btype = box.boxclass
    ignore_code = ignore_code or page.full_ocred

    if btype in ("table", "table-fallback"):
        markdown, html = table_content(box)
        if html is not None:
            return html
        table_text = markdown or ""
        if page.full_ocred:
            # remove code style if page was OCR'd
            table_text = table_text.replace("`", "")
        if table_text.strip():
            return table_text
        # degenerate/legacy table without a rendering: fall back to its text
        textlines = getattr(box, "textlines", None)
        if textlines:
            return text_to_md(_copy_textlines(textlines), ignore_code=ignore_code)
        return ""

    textlines = getattr(box, "textlines", None)
    if not textlines:
        return ""
    textlines = _copy_textlines(textlines)

    if btype == "picture":
        return picture_text_to_md(textlines, ignore_code=ignore_code)
    if btype == "formula":
        # to_markdown drops formula text; keep it (documented deviation)
        return text_to_md(textlines, ignore_code=ignore_code)
    if btype == "title":
        return title_to_md(box.header_level, textlines)
    if btype == "section-header":
        return section_hdr_to_md(box.header_level, textlines)
    if btype == "list-item":
        if list_item_levels is None:
            list_item_levels = create_list_item_levels(page.boxes)
        return list_item_to_md(textlines, list_item_levels.get(box_idx, 1))
    if btype == "footnote":
        return footnote_to_md(textlines)
    return text_to_md(textlines, ignore_code=ignore_code)
