"""Regression tests for four related table-extraction bugs, all found while
diagnosing table corruption in a real Slovak public-procurement PDF, and
all fixed on this branch.

Fix 1 (legacy, non-AI-layout path -- src/helpers/pymupdf_rag.py):
`to_markdown()` locates tables via `page.find_tables(strategy=table_strategy)`
with the hardcoded default `table_strategy == "lines_strict"`, which only
trusts genuinely ruled vector lines (PyMuPDF's own table.py drops any
solid-fill rectangle whose width AND height both exceed the snap tolerance
-- see `clean_graphics()` in pymupdf's table.py). A table whose cell
boundaries are drawn purely as solid background-color fills (row/column
shading, no ruled grid at all -- exactly what the real repro document does)
is therefore invisible to "lines_strict": `find_tables()` returns *zero*
tables for the whole region, not a degraded one. The table's text is then
not even kept as plain paragraphs: it sits inside a vector-graphic cluster,
whose text the legacy path drops, so it is missing from the output
entirely. The fix lets callers opt into a retry by
passing a sequence of strategies, e.g. `("lines_strict", "lines")`: each
is tried in order until one finds a `row_count >= 2 and col_count >= 2`
table. "lines" still requires vector graphics but accepts fill-derived
edges too. A plain "lines_strict" request keeps its meaning and never
falls back.

Fix 2 (AI-layout path -- src/helpers/document_layout.py, `get_table_details`):
The Layout model's own row-0 boundary can be an outlier that swallows
unrelated content sitting just above the real table (a page title, a
running footer) into the table's first row, because that content happens
to fall inside the model's proposed table bbox. The fix re-clusters row 0's
own text geometrically (via `get_raw_lines(require_x_continuity=True)`,
the same guard used in the legacy path to stop cross-column splicing) and
drops row 0 from the grid -- emitting it as a leading plain-text block
instead -- only when it is BOTH sparse (fewer reconstructed lines than
table columns) AND either fragmented across multiple lines or a single
line that crosses an interior column boundary. Sparsity alone would
false-positive on a genuine, single-value title-like row that never
crosses a column; boundary-crossing alone would false-positive on a
genuine multi-line column-header row where two adjacent header labels
happen to be typeset with a small gap and cluster into one line that
crosses their shared boundary. Both conditions are required jointly.

Fix 3 (AI-layout path -- src/helpers/document_layout.py, `get_table_details`):
Fix 2's sparsity check still false-positived on a genuine ONE-line header
row (no wrapping at all) whenever just two of its labels happened to sit
close enough together to cluster into one crossing line -- e.g. the very
document that motivated Fix 2, "Por. c." / "Typ vozidla" / "Technicka
specifikacia": the first two glue into one line 42.12pt apart (just under
the ~42.24pt clustering threshold), dropping row 0 to 2 reconstructed
lines for 3 columns, even though every column genuinely has its own
label. The root defect: `_cluster_rawdict_lines` clusters purely by
inter-span x-gap, with zero awareness of the table's own v_lines -- it
cannot distinguish "two words in one cell" from "two different columns'
headers sitting close together". The fix splits a reconstructed line
back apart wherever an interior v_line falls strictly in the GAP between
two of its spans (never through the middle of one indivisible span --
that's still the genuine swallowed-title case from Fix 2), *before* the
sparsity/crossing checks run. This corrects Fix 2's false positive for
the right reason without weakening it: a real leaked title is still one
atomic span with no inter-span gap to split at, so it is still correctly
excluded; a genuine multi-line wrapped header (Fix 2's own
counter-example) still has its glued pair split apart, only now for the
right reason instead of by the coincidence of already having enough
other sub-lines to avoid the sparsity threshold.

Fix 4 (AI-layout path -- src/helpers/document_layout.py, `get_table_details`):
Fix 2/3's row-0 decision was still made at the *row* level (sparse count vs.
column count, plus whether any one line crosses a boundary), which cannot
express a genuine merged/colspan header cell: a real colspan cell crossing
a boundary makes the row "look" foreign under the old check even though it
belongs in the grid, and Markdown has no colspan syntax to render it with
anyway. The fix reclassifies each row-0 line (after Fix 3's split)
independently by how much of the table's own width and column grid it
covers: a line covering (nearly) the whole table width, or all columns, is
still definitely foreign (a title/running header) and forces the whole row
out as plain text, same as before. A line crossing more than one column
without reaching that threshold is a real colspan cell -- IF everything
else in row 0, taken together, still covers every column (so nothing is
missing, it's genuinely just spread across cells of different widths); its
own text is then duplicated into every column its bbox touches instead of
being dropped or cut in half. But if such a crossing line's own columns,
combined with row 0's other content, still leave some column completely
untouched, the row reads as foreign after all (real-world titles are
routinely split into several disconnected pieces -- a left title, a
centered subtitle, a right-aligned reference number -- each individually
falling short of the "whole table width" threshold, yet together leaving
gaps a genuine header never would) and the whole row is hoisted out as
before. A line that never crosses a boundary at all is untouched by any of
this and is extracted normally by the per-cell loop below, regardless of
how sparse row 0 otherwise is (a lone single-column note with nothing else
in row 0 is legitimate content, not a foreign row).

No true end-to-end test exists for these AI-layout fixes through `pymupdf4llm.use_layout(True)`:
that path needs PyMuPDF's real trained Layout model (and typically a GPU),
which isn't practically invokable in this environment, so `get_table_details()`
is exercised directly instead -- see the label/value splice test's aside
about `insert_text`-generated geometry not reliably triggering MuPDF's own
block-fusion behavior for a similar reason.
"""

from types import SimpleNamespace

import pymupdf
import pymupdf4llm
import pytest
from pymupdf4llm.helpers.document_layout import (
    get_table_details,
    text_to_md,
    text_to_text,
)


@pytest.fixture(autouse=True)
def _reset_layout_mode():
    # Some other test modules (e.g. test_137.py) call use_layout(True) and
    # can leave that global toggle set for the rest of the pytest session
    # if they fail before resetting it; be explicit so these tests' outcome
    # doesn't depend on run order. Restore whatever was in effect before
    # this test afterward -- this toggle is process-global (it even flips
    # pymupdf._get_layout, see test_table_grid_repair_real_fixtures.py's
    # _find_table()), so leaving it at False here would silently break any
    # later test in the same session that needs the real layout engine.
    prev = pymupdf4llm._use_layout
    pymupdf4llm.use_layout(False)
    yield
    pymupdf4llm.use_layout(prev)


# ---------------------------------------------------------------------------
# Fix 1: legacy path, fill-only ("borderless") table falls back from
# "lines_strict" to "lines" instead of being lost to paragraph text.
# ---------------------------------------------------------------------------


def _make_fill_only_table_pdf(nrows=3, ncols=3, col_w=90.0, row_h=20.0, x0=100.0, y0=100.0):
    """A table with NO ruled border/grid lines at all: every cell boundary
    is implied only by adjacent solid-fill background rectangles (like
    alternating row/column shading), mirroring the real repro document
    (a budget table using colored fill bands, no stroked lines whatsoever).
    Confirmed experimentally against PyMuPDF's own table-detection code
    (table.py's `clean_graphics()`): a solid fill is dropped entirely under
    "lines_strict" once both its width AND height exceed the snap
    tolerance -- true here, since each cell fill is a full cell-sized
    rectangle, not a thin simulated line -- but is kept (and its edges used
    to reconstruct the grid) under "lines".
    """
    doc = pymupdf.open()
    page = doc.new_page()
    shape = page.new_shape()
    colors = [(0.92, 0.92, 0.92), (1.0, 1.0, 1.0)]
    for r in range(nrows):
        for c in range(ncols):
            rx0 = x0 + c * col_w
            ry0 = y0 + r * row_h
            rect = pymupdf.Rect(rx0, ry0, rx0 + col_w, ry0 + row_h)
            shape.draw_rect(rect)
            shape.finish(fill=colors[(r + c) % 2], color=None, width=0)
    shape.commit()
    for r in range(nrows):
        for c in range(ncols):
            page.insert_text(
                (x0 + c * col_w + 5, y0 + r * row_h + 14),
                f"Cell{r}_{c}",
                fontsize=9,
            )
    return doc


def test_find_tables_lines_strict_misses_fill_only_table_but_lines_finds_it():
    """Documents the underlying PyMuPDF fact the fix relies on: a
    fill-only table (no ruled lines) is invisible to "lines_strict" but
    detected fine by "lines"."""
    doc = _make_fill_only_table_pdf()
    page = doc[0]

    strict_tables = page.find_tables(strategy="lines_strict").tables
    assert strict_tables == []

    lines_tables = [
        t for t in page.find_tables(strategy="lines").tables
        if t.row_count >= 2 and t.col_count >= 2
    ]
    assert len(lines_tables) == 1
    assert lines_tables[0].row_count == 3
    assert lines_tables[0].col_count == 3
    doc.close()


_STRICT_THEN_LINES = ("lines_strict", "lines")


def test_to_markdown_recovers_fill_only_table():
    """End-to-end: exercises the opt-in fallback retry in pymupdf_rag.py's
    to_markdown(). Without the retry, this table's content is missing from
    the output; with it, assert the content landed inside an actual
    markdown table, not merely somewhere in the page text."""
    doc = _make_fill_only_table_pdf()
    md = pymupdf4llm.to_markdown(doc, table_strategy=_STRICT_THEN_LINES)

    for r in range(3):
        for c in range(3):
            assert f"Cell{r}_{c}" in md, f"Cell{r}_{c} missing from output:\n{md!r}"

    assert "|Cell0_0|Cell0_1|Cell0_2|" in md, (
        f"table content was not extracted as a markdown table:\n{md!r}"
    )
    doc.close()


def test_explicit_lines_strict_does_not_fall_back_to_lines():
    """An explicit (or default) "lines_strict" request keeps its meaning:
    the fill-only table is not detected, rather than being silently
    re-detected with "lines"."""
    doc = _make_fill_only_table_pdf()
    for md in (
        pymupdf4llm.to_markdown(doc),
        pymupdf4llm.to_markdown(doc, table_strategy="lines_strict"),
    ):
        assert "|Cell0_0|" not in md, f"lines_strict fell back to another strategy:\n{md!r}"
    doc.close()


def test_fallback_retry_does_not_run_when_the_first_strategy_finds_a_table(monkeypatch):
    """The retry only happens when the earlier strategy finds nothing, so
    a page that "lines_strict" handles is searched exactly once."""
    doc = _make_fill_only_table_pdf()
    calls = []
    orig = pymupdf.Page.find_tables

    def spy(self, *args, **kwargs):
        calls.append(kwargs.get("strategy"))
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(pymupdf.Page, "find_tables", spy)
    pymupdf4llm.to_markdown(doc, table_strategy=("lines", "lines_strict"))
    assert calls == ["lines"]
    doc.close()


def _make_bar_chart_pdf():
    """Filled bars standing on a ruled axis, labelled underneath."""
    doc = pymupdf.open()
    page = doc.new_page()
    shape = page.new_shape()
    shape.draw_line((100, 400), (100, 200))
    shape.draw_line((100, 400), (400, 400))
    shape.finish(color=(0, 0, 0), width=1)
    for i, h in enumerate([120, 80, 160, 60, 140]):
        x = 120 + i * 55
        shape.draw_rect(pymupdf.Rect(x, 400 - h, x + 35, 400))
        shape.finish(fill=(0.3, 0.5, 0.8), color=None, width=0)
    shape.commit()
    page.insert_text((100, 190), "Revenue by quarter", fontsize=11)
    for i in range(5):
        page.insert_text((120 + i * 55, 415), f"Q{i}", fontsize=9)
    return doc


def _make_decorative_fills_pdf():
    """A title banner, a subtitle band and two side-by-side callout boxes,
    all drawn as adjacent fills with text inside them."""
    doc = pymupdf.open()
    page = doc.new_page()
    shape = page.new_shape()
    for rect, color in [
        ((72, 72, 540, 110), (0.1, 0.2, 0.4)),
        ((72, 110, 540, 130), (0.9, 0.9, 0.9)),
        ((72, 200, 300, 300), (0.95, 0.95, 0.8)),
        ((300, 200, 540, 300), (0.8, 0.95, 0.95)),
    ]:
        shape.draw_rect(pymupdf.Rect(rect))
        shape.finish(fill=color, color=None, width=0)
    shape.commit()
    page.insert_text((80, 95), "Annual Report 2025", fontsize=16)
    page.insert_text((80, 124), "Subtitle band text", fontsize=9)
    page.insert_text((80, 220), "Left callout box text here", fontsize=9)
    page.insert_text((80, 240), "second line left", fontsize=9)
    page.insert_text((310, 220), "Right callout box text", fontsize=9)
    return doc


def test_fallback_does_not_read_charts_or_decorative_fills_as_tables():
    """False-positive check for the opt-in retry: neither a bar chart nor
    decorative banner/callout fills become a table under "lines"."""
    for make in (_make_bar_chart_pdf, _make_decorative_fills_pdf):
        doc = make()
        md = pymupdf4llm.to_markdown(doc, table_strategy=_STRICT_THEN_LINES)
        assert "|---" not in md, f"{make.__name__} was read as a table:\n{md!r}"
        doc.close()


# ---------------------------------------------------------------------------
# Fix 2: AI-layout path, get_table_details()'s row-0 swallowed-header
# exclusion logic.
# ---------------------------------------------------------------------------


def _char(c, x0, y0, x1, y1):
    return {"c": c, "bbox": (x0, y0, x1, y1)}


def _span(text, x0, y0, x1, y1, size=9.0):
    """A RAWDICT-format span: char-level, with "chars" (each carrying "c"
    and "bbox") rather than DICT format's flat "text" field -- this is what
    get_table_details()/_cluster_rawdict_lines()/utils.extract_cells() all
    expect. Character boxes are evenly divided across the span's width;
    good enough for the >50%-overlap check extract_cells() does per char."""
    n = len(text)
    cw = (x1 - x0) / max(n, 1)
    chars = [_char(ch, x0 + i * cw, y0, x0 + (i + 1) * cw, y1) for i, ch in enumerate(text)]
    return {
        "bbox": (x0, y0, x1, y1),
        "size": size,
        "flags": 0,
        "char_flags": 0,
        "font": "Helvetica",
        "alpha": 255,
        "chars": chars,
    }


def _line(spans, x0, y0, x1, y1, direction=(1.0, 0.0)):
    return {"bbox": (x0, y0, x1, y1), "dir": direction, "spans": spans}


def _block(lines, x0, y0, x1, y1):
    return {"type": 0, "bbox": (x0, y0, x1, y1), "lines": lines}


def _text_block(text, x0, y0, x1, y1, size=9.0):
    """One block containing one line containing one span -- the common case
    below: a single piece of text at a known bbox."""
    span = _span(text, x0, y0, x1, y1, size=size)
    return _block([_line([span], x0, y0, x1, y1)], x0, y0, x1, y1)


def _make_tab_dict(x0, y0, x1, y1, interior_v_abs, interior_h_abs):
    """Build the (tab_dict, grid) get_table_details() expects. Takes
    ABSOLUTE interior column/row boundary coordinates (not the x0/y0-
    relative offsets grid.v_lines/h_lines actually carry) purely so the
    tests below can be written in the same absolute coordinates as the
    text placement -- the conversion happens here."""
    grid = SimpleNamespace(
        v_lines=[v - x0 for v in interior_v_abs],
        h_lines=[h - y0 for h in interior_h_abs],
    )
    return {"group_bbox": (x0, y0, x1, y1), "table_grid": grid}


def _data_row_blocks(x0, y_start, col_w, row_h, ncols, nrows, size=9.0):
    """A plain grid of "R{row}C{col}" cells, one per column per row,
    entirely inside their own column -- the normal table body below any
    row-0 header/title under test."""
    blocks = []
    for r in range(nrows):
        ry0 = y_start + r * row_h
        for c in range(ncols):
            cx0 = x0 + c * col_w
            blocks.append(_text_block(f"R{r}C{c}", cx0 + 5, ry0 + 3, cx0 + 45, ry0 + 3 + size))
    return blocks


def _extract_flat(det):
    return [cell for row in det.extract for cell in row]


def _excluded_text(det):
    """Joined plain text of det.excluded_textlines -- the real "text" box
    the caller inserts ahead of the table (see document_layout.py's
    preceding_text_box) now carries this content, not det.markdown."""
    return " ".join(
        s["text"] for tl in (det.excluded_textlines or []) for s in tl["spans"]
    )


def test_swallowed_single_line_title_is_excluded_from_row_0():
    """Row 0 contains ONE reconstructed text line whose x-range crosses an
    interior column boundary, and the row is sparse (1 line for 3
    columns). This is the case a naive "row 0 has >1 line" check would
    miss entirely, since it never fragments into multiple lines -- only
    the row0_spans_column check catches it."""
    x0, y0 = 100.0, 100.0
    col_w, ncols = 100.0, 3
    row0_h, data_row_h, nrows_data = 30.0, 20.0, 3
    x1 = x0 + col_w * ncols
    y_after_row0 = y0 + row0_h
    y1 = y_after_row0 + data_row_h * nrows_data

    tab_dict = _make_tab_dict(
        x0, y0, x1, y1,
        interior_v_abs=[x0 + col_w, x0 + 2 * col_w],
        interior_h_abs=[y_after_row0 + data_row_h * i for i in range(nrows_data)],
    )
    title_block = _text_block(
        "PRACOVNY BALIK 3-2 Project Title Long Text", x0 + 10, y0 + 5, x1 - 20, y0 + 25
    )
    blocks = [title_block] + _data_row_blocks(
        x0, y_after_row0, col_w, data_row_h, ncols, nrows_data
    )

    det = get_table_details(tab_dict, blocks)

    assert det.col_count == 3
    assert det.row_count == 3  # row 0's boundary was dropped
    assert "PRACOVNY BALIK 3-2 Project Title Long Text" in _excluded_text(det)
    assert not det.markdown.startswith("PRACOVNY")
    assert det.extract == [
        [f"R{r}C{c}" for c in range(3)] for r in range(3)
    ]
    for cell in _extract_flat(det):
        assert "PRACOVNY" not in cell and "BALIK" not in cell


def test_foreign_looking_row_0_is_kept_when_it_is_the_tables_only_row():
    """Regression test for the single-row guard: if the grid has exactly
    one row and that row's content looks foreign (sparse, crossing a
    column boundary -- the same signal that triggers exclusion when a
    real data row exists underneath), it must NOT be hoisted out. Doing
    so would leave zero rows, which the markdown renderer cannot handle,
    and there is no data row left to even benefit from the exclusion."""
    x0, y0 = 100.0, 100.0
    col_w, ncols = 100.0, 3
    row0_h = 20.0
    x1 = x0 + col_w * ncols
    y1 = y0 + row0_h

    tab_dict = _make_tab_dict(
        x0, y0, x1, y1,
        interior_v_abs=[x0 + col_w, x0 + 2 * col_w],
        interior_h_abs=[],  # no interior row boundary -- exactly one row
    )
    # Same sparse, boundary-crossing shape used elsewhere in this file to
    # trigger the leak classification.
    title_block = _text_block(
        "PRACOVNY BALIK 3-2 Project Title Long Text", x0 + 10, y0 + 5, x1 - 20, y0 + 15
    )

    det = get_table_details(tab_dict, [title_block])

    assert det.col_count == 3
    assert det.row_count == 1  # never dropped to 0
    assert not det.excluded_textlines
    assert "PRACOVNY" in " ".join(_extract_flat(det))


def test_swallowed_multi_line_title_and_footer_is_excluded_from_row_0():
    """Row 0 fragments into 2 separate visually-distinct lines (a title
    line and an unrelated running footer line at a different y), still
    sparse relative to the column count (2 lines for 3 columns)."""
    x0, y0 = 100.0, 100.0
    col_w, ncols = 100.0, 3
    row0_h, data_row_h, nrows_data = 24.0, 20.0, 2
    x1 = x0 + col_w * ncols
    y_after_row0 = y0 + row0_h
    y1 = y_after_row0 + data_row_h * nrows_data

    tab_dict = _make_tab_dict(
        x0, y0, x1, y1,
        interior_v_abs=[x0 + col_w, x0 + 2 * col_w],
        interior_h_abs=[y_after_row0 + data_row_h * i for i in range(nrows_data)],
    )
    title_block = _text_block("PROJECT TITLE HERE", x0 + 10, y0 + 2, x0 + 250, y0 + 12)
    footer_block = _text_block(
        "Page 5 of 12 -- running footer text", x0 + 10, y0 + 14, x0 + 260, y0 + 24
    )
    blocks = [title_block, footer_block] + _data_row_blocks(
        x0, y_after_row0, col_w, data_row_h, ncols, nrows_data
    )

    det = get_table_details(tab_dict, blocks)

    assert det.col_count == 3
    assert det.row_count == 2  # row 0's boundary was dropped
    excluded_text = _excluded_text(det)
    assert "PROJECT TITLE HERE" in excluded_text
    assert "Page 5 of 12 -- running footer text" in excluded_text
    assert not det.markdown.startswith("PROJECT")
    assert det.extract == [
        [f"R{r}C{c}" for c in range(3)] for r in range(2)
    ]
    for cell in _extract_flat(det):
        assert "PROJECT" not in cell and "footer" not in cell


def test_excluded_textlines_render_via_text_to_md_and_text_to_text():
    """Regression test for the reviewer-flagged content loss: the caller
    (document_layout.py's per-box loop) renders det.excluded_textlines as a
    genuine "text" LayoutBox via text_to_md()/text_to_text() -- the same
    functions used for every other text box -- so this content must
    survive in BOTH to_markdown() and to_text() output, not just
    `.markdown`. Also confirms spans from the title and the footer (two
    originally separate text blocks) come out separated by whitespace, not
    glued together, since text_to_md/text_to_text insert a space after
    every span regardless of which original line/block it came from."""
    x0, y0 = 100.0, 100.0
    col_w, ncols = 100.0, 3
    row0_h, data_row_h, nrows_data = 24.0, 20.0, 2
    x1 = x0 + col_w * ncols
    y_after_row0 = y0 + row0_h
    y1 = y_after_row0 + data_row_h * nrows_data

    tab_dict = _make_tab_dict(
        x0, y0, x1, y1,
        interior_v_abs=[x0 + col_w, x0 + 2 * col_w],
        interior_h_abs=[y_after_row0 + data_row_h * i for i in range(nrows_data)],
    )
    title_block = _text_block("PROJECT TITLE HERE", x0 + 10, y0 + 2, x0 + 250, y0 + 12)
    footer_block = _text_block(
        "Page 5 of 12 -- running footer text", x0 + 10, y0 + 14, x0 + 260, y0 + 24
    )
    blocks = [title_block, footer_block] + _data_row_blocks(
        x0, y_after_row0, col_w, data_row_h, ncols, nrows_data
    )

    det = get_table_details(tab_dict, blocks)
    assert det.excluded_textlines

    md = text_to_md(det.excluded_textlines)
    text = text_to_text(det.excluded_textlines)
    for rendered in (md, text):
        assert "PROJECT TITLE HERE" in rendered
        assert "Page 5 of 12" in rendered
        # Never glued across the title/footer boundary with no separator.
        assert "HEREPage" not in rendered


def test_genuine_multiline_header_row_with_glued_adjacent_labels_is_not_excluded():
    """The false-positive case a boundary-crossing-only check would trip
    on: row 0 is a genuine multi-line column-header row (each column's
    header wraps to 2 lines), and two adjacent header fragments in
    different columns happen to sit close enough together to cluster into
    ONE reconstructed line that crosses their shared column boundary
    (e.g. "Jednotkova cena bez DPH" / "Celkove opravnene vydavky" --
    real column headers from the source document, typeset with a small
    gap between them). Row 0 as a whole still produces close to one
    reconstructed line per column (5 lines for 3 columns, since each
    column wraps to 2 except the two that fused into 1) -- NOT sparse --
    so it must stay part of the table grid despite the crossing line."""
    x0, y0 = 100.0, 100.0
    col_w, ncols = 200.0, 3
    row0_h, data_row_h, nrows_data = 24.0, 20.0, 2
    x1 = x0 + col_w * ncols
    y_after_row0 = y0 + row0_h
    y1 = y_after_row0 + data_row_h * nrows_data
    boundary_1_2 = x0 + col_w * 2  # shared boundary between column 1 and 2

    tab_dict = _make_tab_dict(
        x0, y0, x1, y1,
        interior_v_abs=[x0 + col_w, boundary_1_2],
        interior_h_abs=[y_after_row0 + data_row_h * i for i in range(nrows_data)],
    )
    sub1_y0, sub1_y1 = y0 + 2, y0 + 2 + 9.0
    sub2_y0, sub2_y1 = y0 + 14, y0 + 14 + 9.0
    header_blocks = [
        # column 0: 2-line header, entirely inside column 0
        _text_block("Header0", x0 + 5, sub1_y0, x0 + 95, sub1_y1),
        _text_block("Sub0", x0 + 5, sub2_y0, x0 + 80, sub2_y1),
        # columns 1+2 first sub-line: glued together across the boundary
        # with only a 4pt gap -- clusters into one crossing line.
        _text_block("Jednotkova cena bez DPH", x0 + col_w + 5, sub1_y0, boundary_1_2 - 2, sub1_y1),
        _text_block("Celkove opravnene", boundary_1_2 + 2, sub1_y0, x0 + 2.9 * col_w, sub1_y1),
        # columns 1, 2 second sub-line: each entirely inside its own column.
        _text_block("Sub1", x0 + col_w + 5, sub2_y0, x0 + col_w + 100, sub2_y1),
        _text_block("Sub2", boundary_1_2 + 5, sub2_y0, boundary_1_2 + 100, sub2_y1),
    ]
    blocks = header_blocks + _data_row_blocks(
        x0, y_after_row0, col_w, data_row_h, ncols, nrows_data
    )

    det = get_table_details(tab_dict, blocks)

    assert det.col_count == 3
    assert det.row_count == 3  # row 0 kept -- NOT hoisted out
    assert det.extract[0] == [
        "Header0\nSub0",
        "Jednotkova cena bez DPH\nSub1",
        "Celkove opravnene\nSub2",
    ]
    # The crossing line's two halves must still land in their own,
    # correct columns -- the transient clustering used only to decide
    # whether to exclude row 0 must not affect actual cell assignment,
    # which is done independently via per-char bbox overlap.
    assert "Celkove" not in det.extract[0][1]
    assert "Jednotkova" not in det.extract[0][2]
    assert not det.markdown.startswith("Jednotkova")



def test_genuine_single_line_header_row_where_glued_labels_make_it_falsely_sparse_is_not_excluded():
    """Regression test for the real-world false positive this discriminator
    still had (found in a Slovak public-procurement PDF: "Por. c." / "Typ
    vozidla" / "Technicka specifikacia"). Unlike the wrapped 2-line header
    above, row 0 here is a genuine ONE-line, one-value-per-column header --
    but two ADJACENT labels ("Por. c." and "Typ vozidla") are typeset close
    enough together (42.12pt apart, just under the ~42.24pt clustering
    threshold at their font size) that `_cluster_rawdict_lines` merges them
    into a single reconstructed line straddling their shared column
    boundary, while the third label sits far enough away to stay separate.
    That alone drops the line count to 2 for 3 columns -- sparse -- even
    though every column genuinely has its own label; `_cluster_rawdict_lines`
    has no notion of the table's own v_lines at all, so it cannot tell "two
    words in one cell" apart from "two different columns' headers that
    happen to sit close together".

    The fix: before judging sparsity, split any reconstructed line wherever
    an interior v_line falls strictly in the GAP between two of its spans
    (not through the middle of one indivisible span -- that's the separate,
    still-valid swallowed-title case above). That turns the false 2-lines-
    for-3-columns count back into the true 3-lines-for-3-columns, so this
    genuine header is no longer hoisted out as leading plain text."""
    x0, y0 = 100.0, 100.0
    col_w, ncols = 100.0, 3
    row0_h, data_row_h, nrows_data = 20.0, 20.0, 2
    x1 = x0 + col_w * ncols  # 400.0
    y_after_row0 = y0 + row0_h  # 120.0
    y1 = y_after_row0 + data_row_h * nrows_data
    boundary_0_1 = x0 + col_w  # 200.0
    boundary_1_2 = x0 + col_w * 2  # 300.0

    tab_dict = _make_tab_dict(
        x0, y0, x1, y1,
        interior_v_abs=[boundary_0_1, boundary_1_2],
        interior_h_abs=[y_after_row0 + data_row_h * i for i in range(nrows_data)],
    )
    header_blocks = [
        # Columns 0/1's labels are 20pt apart -- comfortably under the
        # clustering threshold (max(5, 9*4)=36 at this test font's default
        # 9pt size) -- so they merge into one line straddling x=200.
        _text_block("Label1", x0 + 5, y0 + 5, boundary_0_1 - 10, y0 + 13),
        _text_block("Label2", boundary_0_1 + 10, y0 + 5, boundary_1_2 - 5, y0 + 13),
        # Column 2's label sits far enough from column 1's (45pt) that it
        # stays its own reconstructed line even before any fix.
        _text_block("Label3", boundary_1_2 + 40, y0 + 5, x1 - 5, y0 + 13),
    ]
    blocks = header_blocks + _data_row_blocks(
        x0, y_after_row0, col_w, data_row_h, ncols, nrows_data
    )

    det = get_table_details(tab_dict, blocks)

    assert det.col_count == 3
    assert det.row_count == 3  # row 0 kept intact, never hoisted out
    assert det.extract == [
        ["Label1", "Label2", "Label3"],
        ["R0C0", "R0C1", "R0C2"],
        ["R1C0", "R1C1", "R1C2"],
    ]
    assert not det.markdown.startswith("Label1")


def test_genuine_colspan_header_cell_is_duplicated_into_every_column_it_spans():
    """A real merged/colspan-style header cell -- ONE indivisible span (no
    inter-span gap to split at, unlike the glued-adjacent-labels case above)
    that genuinely covers 2 of 4 columns but not (nearly) the table's whole
    width -- must stay part of the table grid rather than being hoisted out
    as plain text (it isn't a leaked title: it doesn't cover enough of the
    table's width for that), and since Markdown tables have no colspan, the
    only faithful rendering is to repeat its own text into every column its
    own bbox visually spans, not silently drop it in just one of them."""
    x0, y0 = 100.0, 100.0
    col_w, ncols = 100.0, 4
    row0_h, data_row_h, nrows_data = 20.0, 20.0, 1
    x1 = x0 + col_w * ncols  # 500.0
    y_after_row0 = y0 + row0_h
    y1 = y_after_row0 + data_row_h * nrows_data
    boundary_1_2 = x0 + col_w * 2  # 300.0 -- the boundary the merged cell straddles

    tab_dict = _make_tab_dict(
        x0, y0, x1, y1,
        interior_v_abs=[x0 + col_w, boundary_1_2, x0 + col_w * 3],
        interior_h_abs=[y_after_row0 + data_row_h * i for i in range(nrows_data)],
    )
    header_blocks = [
        _text_block("Label0", x0 + 5, y0 + 5, x0 + col_w - 10, y0 + 13),
        # Column 1 and column 2's shared boundary (x=300) falls INSIDE this
        # one span, not in a gap between two spans -- there is nothing to
        # split it back apart at, unlike the false-merge case above.
        _text_block("Combined Header", x0 + col_w + 10, y0 + 5, boundary_1_2 + col_w - 10, y0 + 13),
        _text_block("Label3", x0 + col_w * 3 + 10, y0 + 5, x1 - 10, y0 + 13),
    ]
    blocks = header_blocks + _data_row_blocks(
        x0, y_after_row0, col_w, data_row_h, ncols, nrows_data
    )

    det = get_table_details(tab_dict, blocks)

    assert det.col_count == 4
    assert det.row_count == 1 + nrows_data  # row 0 kept, not hoisted out
    assert det.extract[0] == ["Label0", "Combined Header", "Combined Header", "Label3"]
    assert det.markdown.count("Combined Header") == 2


def test_single_fragment_title_crossing_several_but_not_all_columns_is_still_excluded():
    """Regression test for a real-world case (found in this branch's own
    repro PDF's second table) that a naive "does this line alone cover
    (nearly) the whole table width" leak check misses: a real leaked
    running title, this time reduced to ONE single row-0 line (not several
    disconnected fragments), that crosses several interior boundaries but
    -- because the table is wide and the title is short relative to it --
    covers well under the width-fraction leak threshold and well under
    all of the table's columns. This looks superficially like the genuine
    colspan cell above (one line crossing more than one column boundary,
    not reaching the leak threshold), but the crucial difference is that
    NOTHING else exists in row 0 to fill the columns this line doesn't
    touch -- a real colspan header's sibling cells fill every remaining
    column; here, several columns on both sides are left completely
    empty, which is what a leaked title does and a genuine header
    doesn't. So it must still be hoisted out as leading plain text."""
    x0, y0 = 100.0, 100.0
    col_w, ncols = 100.0, 9
    row0_h, data_row_h, nrows_data = 20.0, 20.0, 1
    x1 = x0 + col_w * ncols  # 1000.0
    y_after_row0 = y0 + row0_h
    y1 = y_after_row0 + data_row_h * nrows_data

    tab_dict = _make_tab_dict(
        x0, y0, x1, y1,
        interior_v_abs=[x0 + col_w * i for i in range(1, ncols)],
        interior_h_abs=[y_after_row0 + data_row_h * i for i in range(nrows_data)],
    )
    # Crosses columns 1..6 (6 of 9 -- well under the whole table) but
    # columns 0, 7 and 8 are left with no row-0 content at all.
    title_block = _text_block(
        "PRACOVNY BALIK: 3-2 Project Title Spanning The Middle",
        x0 + col_w + 10, y0 + 5, x0 + col_w * 7 - 10, y0 + 13,
    )
    blocks = [title_block] + _data_row_blocks(
        x0, y_after_row0, col_w, data_row_h, ncols, nrows_data
    )

    det = get_table_details(tab_dict, blocks)

    assert det.col_count == 9
    assert det.row_count == nrows_data  # row 0's boundary was dropped
    assert "PRACOVNY BALIK: 3-2 Project Title Spanning The Middle" in _excluded_text(det)
    assert not det.markdown.startswith("PRACOVNY")
    for cell in _extract_flat(det):
        assert "PRACOVNY" not in cell and "BALIK" not in cell


def test_single_column_header_in_a_very_wide_column_is_not_treated_as_a_leak():
    """A genuine single-column header cell that never crosses any interior
    boundary must not be misclassified as a leaked title just because its
    own column happens to be a large fraction of the table's total width
    (e.g. a wide "description" column next to a narrow "price" column):
    the width-fraction leak test is only meaningful for a line that is
    actually crossing a boundary -- width alone, with no crossing, is
    exactly what an ordinary wide column's header looks like."""
    x0, y0 = 100.0, 100.0
    row0_h, data_row_h, nrows_data = 20.0, 20.0, 1
    wide_col_w, narrow_col_w = 800.0, 100.0
    boundary = x0 + wide_col_w  # 900.0
    x1 = boundary + narrow_col_w  # 1000.0 -- wide column is 80% of table width
    y_after_row0 = y0 + row0_h
    y1 = y_after_row0 + data_row_h * nrows_data

    tab_dict = _make_tab_dict(
        x0, y0, x1, y1,
        interior_v_abs=[boundary],
        interior_h_abs=[y_after_row0 + data_row_h * i for i in range(nrows_data)],
    )
    header_blocks = [
        _text_block(
            "Description of the item being delivered",
            x0 + 5, y0 + 5, boundary - 10, y0 + 13,
        ),
        _text_block("Cena", boundary + 5, y0 + 5, x1 - 5, y0 + 13),
    ]
    data_blocks = [
        _text_block("Widget", x0 + 5, y_after_row0 + 3, boundary - 10, y_after_row0 + 12),
        _text_block("9.99", boundary + 5, y_after_row0 + 3, x1 - 5, y_after_row0 + 12),
    ]
    blocks = header_blocks + data_blocks

    det = get_table_details(tab_dict, blocks)

    assert det.col_count == 2
    assert det.row_count == 1 + nrows_data  # row 0 kept, not hoisted out
    assert det.extract[0] == ["Description of the item being delivered", "Cena"]
    assert not det.markdown.startswith("Description")


def test_single_column_table_header_row_is_not_excluded():
    """A table with only one column has no interior boundary at all, so
    every row-0 line trivially "touches all columns" (there is only one)
    -- that must not be read as automatic proof of a leak, or every
    single-column table would lose its header row."""
    x0, y0 = 100.0, 100.0
    col_w, ncols = 200.0, 1
    row0_h, data_row_h, nrows_data = 20.0, 20.0, 2
    x1 = x0 + col_w * ncols
    y_after_row0 = y0 + row0_h
    y1 = y_after_row0 + data_row_h * nrows_data

    tab_dict = _make_tab_dict(
        x0, y0, x1, y1,
        interior_v_abs=[],
        interior_h_abs=[y_after_row0 + data_row_h * i for i in range(nrows_data)],
    )
    header_block = _text_block("Header", x0 + 5, y0 + 5, x1 - 5, y0 + 13)
    blocks = [header_block] + _data_row_blocks(
        x0, y_after_row0, col_w, data_row_h, ncols, nrows_data
    )

    det = get_table_details(tab_dict, blocks)

    assert det.col_count == 1
    assert det.row_count == 1 + nrows_data  # row 0 kept, not hoisted out
    assert det.extract[0] == ["Header"]
    assert not det.markdown.startswith("Header\n\n")


def test_sparse_single_value_row_not_crossing_a_boundary_is_not_excluded():
    """Sanity check that sparsity alone isn't sufficient to exclude row 0:
    a single short line (1 line for 3 columns -- sparse) that sits
    entirely inside one column, crossing no interior boundary, must NOT
    be excluded -- both a fragmentation/crossing signal AND sparsity are
    required jointly, not sparsity alone."""
    x0, y0 = 100.0, 100.0
    col_w, ncols = 100.0, 3
    row0_h, data_row_h, nrows_data = 24.0, 20.0, 2
    x1 = x0 + col_w * ncols
    y_after_row0 = y0 + row0_h
    y1 = y_after_row0 + data_row_h * nrows_data

    tab_dict = _make_tab_dict(
        x0, y0, x1, y1,
        interior_v_abs=[x0 + col_w, x0 + 2 * col_w],
        interior_h_abs=[y_after_row0 + data_row_h * i for i in range(nrows_data)],
    )
    note_block = _text_block("Note:", x0 + 10, y0 + 5, x0 + 60, y0 + 15)
    blocks = [note_block] + _data_row_blocks(
        x0, y_after_row0, col_w, data_row_h, ncols, nrows_data
    )

    det = get_table_details(tab_dict, blocks)

    assert det.col_count == 3
    assert det.row_count == 3  # row 0 kept
    assert det.extract[0] == ["Note:", "", ""]
    assert det.markdown is not None
    assert not det.markdown.startswith("Note:\n\n")
