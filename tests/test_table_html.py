import inspect
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import pymupdf
import pymupdf4llm
from pymupdf4llm.helpers import document_layout
from pymupdf4llm.helpers.table_html import page_html_tables
from pymupdf4llm.helpers.table_html.reconstruct import to_html


g_root = os.path.normpath(f"{__file__}/../..")
TABLE_PDF = os.path.join(g_root, "tests", "test_sce_150_1.pdf")


def test_to_html_is_live_only_public_api():
    signature = inspect.signature(to_html)
    assert list(signature.parameters) == ["pdf", "page_index"]


def test_page_html_tables_uses_core_union_find_tables():
    original_find_tables = pymupdf.Page.find_tables
    calls = []

    def wrapped_find_tables(self, *args, **kwargs):
        calls.append(kwargs.copy())
        return original_find_tables(self, *args, **kwargs)

    pymupdf.Page.find_tables = wrapped_find_tables
    try:
        doc = pymupdf.open(TABLE_PDF)
        try:
            tables = page_html_tables(doc[0])
        finally:
            doc.close()
    finally:
        pymupdf.Page.find_tables = original_find_tables

    assert len(tables) == 2
    # find_tables(union=True, refine=True) is called exactly once at the Page
    # level; the line-based candidate pass runs via the module-level find_tables
    # (not a Page.find_tables call), so it is not counted here.
    assert len(calls) == 1
    assert calls[0]["use_layout"] is True
    assert calls[0]["union"] is True
    assert calls[0]["refine"] is True


def test_to_markdown_table_output_html_uses_layout_path():
    original_parse_document = document_layout.parse_document
    calls = []

    def wrapped_parse_document(*args, **kwargs):
        calls.append(kwargs.copy())
        return original_parse_document(*args, **kwargs)

    document_layout.parse_document = wrapped_parse_document
    pymupdf4llm.use_layout(True)
    try:
        md = pymupdf4llm.to_markdown(
            TABLE_PDF,
            pages=[0],
            table_output="html",
            use_ocr=False,
        )
    finally:
        document_layout.parse_document = original_parse_document
    assert md.count("<table") == 2
    assert "| --- |" not in md
    assert calls
    assert calls[0]["render_html_tables"] is True


def test_to_json_table_output_html_uses_layout_path():
    original_parse_document = document_layout.parse_document
    calls = []

    def wrapped_parse_document(*args, **kwargs):
        calls.append(kwargs.copy())
        return original_parse_document(*args, **kwargs)

    document_layout.parse_document = wrapped_parse_document
    pymupdf4llm.use_layout(True)
    try:
        js = pymupdf4llm.to_json(
            TABLE_PDF,
            pages=[0],
            table_output="html",
            use_ocr=False,
        )
    finally:
        document_layout.parse_document = original_parse_document

    data = json.loads(js)
    table_boxes = [
        box
        for page in data["pages"]
        for box in page["boxes"]
        if box["boxclass"] == "table"
    ]
    assert calls
    assert calls[0]["render_html_tables"] is True
    assert any(box["table"].get("html") for box in table_boxes)


def test_layout_html_env_does_not_enable_table_html(monkeypatch):
    monkeypatch.setenv("PYMUPDF_LAYOUT_HTML_TABLES", "1")
    pymupdf4llm.use_layout(True)

    md = pymupdf4llm.to_markdown(
        TABLE_PDF,
        pages=[0],
        use_ocr=False,
    )

    assert "<table" not in md


def test_table_html_parallel_smoke():
    expected = to_html(TABLE_PDF, 0)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: to_html(TABLE_PDF, 0), range(16)))

    assert results == [expected] * 16


_TABLE_TAG_RE = re.compile(r"<table.*?</table>", re.S)


def test_to_json_html_tables_match_to_markdown():
    """VALUE parity: every <table>...</table> emitted into html-mode markdown
    must be byte-identical to (and in the same order as) the html the
    same-mode JSON reports for its table boxes."""
    md = pymupdf4llm.to_markdown(
        TABLE_PDF,
        pages=[0],
        table_output="html",
        use_ocr=False,
    )
    js = pymupdf4llm.to_json(
        TABLE_PDF,
        pages=[0],
        table_output="html",
        use_ocr=False,
    )
    data = json.loads(js)
    table_boxes = [
        box
        for page in data["pages"]
        for box in page["boxes"]
        if box["boxclass"] == "table"
    ]
    json_html = "\n\n".join(
        box["table"]["html"] for box in table_boxes if box["table"].get("html")
    )

    md_tables = _TABLE_TAG_RE.findall(md)
    json_tables = _TABLE_TAG_RE.findall(json_html)

    assert md_tables  # guard against both-empty passes
    assert md_tables == json_tables


_CELL_TAG_RE = re.compile(r"<(?:td|th)\b([^>]*)>")
_COLSPAN_RE = re.compile(r'colspan="(\d+)"')
_ROW_TAG_RE = re.compile(r"<tr\b[^>]*>.*?</tr>", re.S)


def _table_row_width(row_html):
    """Row width in grid columns, summing each cell's colspan (default 1)."""
    width = 0
    for attrs in _CELL_TAG_RE.findall(row_html):
        m = _COLSPAN_RE.search(attrs)
        width += int(m.group(1)) if m else 1
    return width


def test_to_json_html_mode_grid_fields_consistent():
    """row_count/col_count/cells/extract reported next to html must describe the
    SAME grid the html shows; markdown stays unset (html authoritative)."""
    js = pymupdf4llm.to_json(
        TABLE_PDF,
        pages=[0],
        table_output="html",
        use_ocr=False,
    )
    data = json.loads(js)
    table_boxes = [
        box
        for page in data["pages"]
        for box in page["boxes"]
        if box["boxclass"] == "table"
    ]
    assert table_boxes  # guard against an empty (vacuously-passing) run

    for box in table_boxes:
        tbl = box["table"]
        html = tbl["html"]
        rows = _ROW_TAG_RE.findall(html)

        assert len(rows) == tbl["row_count"]
        assert max((_table_row_width(row) for row in rows), default=0) == tbl["col_count"]

        # Shape, not full occupancy: individual cells may legitimately be
        # None (grid gap / covered by a span), but row/col dimensions must
        # match row_count/col_count.
        cells = tbl["cells"]
        assert len(cells) == tbl["row_count"]
        assert all(len(row) == tbl["col_count"] for row in cells)

        # extract is the post-span cell-text matrix (row-major, None for a
        # span-covered slot / grid gap), same shape as cells. markdown stays None
        # (html authoritative in this mode).
        extract = tbl["extract"]
        assert extract is not None
        assert len(extract) == tbl["row_count"]
        assert all(len(row) == tbl["col_count"] for row in extract)
        assert tbl["markdown"] is None


def test_table_output_html_no_layout_falls_back_to_rag_path(monkeypatch):
    """When the layout engine is unavailable, table_output="html" must still
    work end-to-end by falling back to the legacy pymupdf_rag path (which has
    its own independent table_output="html" wiring)."""
    monkeypatch.setattr(pymupdf4llm, "_use_layout", False)

    md = pymupdf4llm.to_markdown(TABLE_PDF, table_output="html")

    assert "<table" in md


def _build_body_table_body_doc():
    """One page: a body paragraph, a small bordered table, another body
    paragraph -- used to check to_markdown never drops/duplicates the
    surrounding body text just because a table is present on the page."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_textbox(
        pymupdf.Rect(72, 72, 540, 140),
        "ALPHAMARK is a short introductory paragraph that appears before the table on this page.",
        fontsize=11,
    )
    table_rect = pymupdf.Rect(72, 200, 540, 340)
    cells = pymupdf.make_table(table_rect, rows=4, cols=3)
    for row in cells:
        for cell in row:
            page.draw_rect(cell)
    for i, row in enumerate(cells):
        for j, cell in enumerate(row):
            page.insert_textbox(
                cell,
                f"R{i}C{j}",
                align=pymupdf.TEXT_ALIGN_CENTER,
                fontsize=10,
            )
    page.insert_textbox(
        pymupdf.Rect(72, 400, 540, 470),
        "OMEGAMARK is a short closing paragraph that appears after the table on this page.",
        fontsize=11,
    )
    page.clean_contents()
    return doc


def test_body_text_preserved_around_tables():
    """No-loss/no-duplication property for the reading-order rewrite that
    carves table regions out of surrounding body text
    (document_layout._split_text_box_around_tables): body text above and
    below a table must survive exactly once, in both plain markdown and html
    table_output modes."""
    pymupdf4llm.use_layout(True)
    doc = _build_body_table_body_doc()
    pdfdata = doc.tobytes()
    doc.close()

    doc_plain = pymupdf.open("pdf", pdfdata)
    try:
        md_plain = pymupdf4llm.to_markdown(doc_plain, use_ocr=False)
    finally:
        doc_plain.close()

    doc_html = pymupdf.open("pdf", pdfdata)
    try:
        md_html = pymupdf4llm.to_markdown(doc_html, table_output="html", use_ocr=False)
    finally:
        doc_html.close()

    for marker in ("ALPHAMARK", "OMEGAMARK"):
        assert md_plain.count(marker) == 1
        assert md_html.count(marker) == 1
