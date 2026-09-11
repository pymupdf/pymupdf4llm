"""Regression tests for two related table-extraction bugs, both found while
diagnosing table corruption in a real Slovak public-procurement PDF, and
both fixed on this branch.

Fix 1 (legacy, non-AI-layout path -- src/helpers/pymupdf_rag.py):
`to_markdown()` locates tables via `page.find_tables(strategy=table_strategy)`
with the hardcoded default `table_strategy == "lines_strict"`, which only
trusts genuinely ruled vector lines (PyMuPDF's own table.py drops any
solid-fill rectangle whose width AND height both exceed the snap tolerance
-- see `clean_graphics()` in pymupdf's table.py). A table whose cell
boundaries are drawn purely as solid background-color fills (row/column
shading, no ruled grid at all -- exactly what the real repro document does)
is therefore invisible to "lines_strict": `find_tables()` returns *zero*
tables for the whole region, not a degraded one, and the table's content is
lost into loose paragraph text. The fix retries with the more lenient
"lines" strategy (which still requires vector graphics, but accepts
fill-derived edges too) whenever the first "lines_strict" call comes back
with no `row_count >= 2 and col_count >= 2` table, before giving up.

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

No true end-to-end test exists for Fix 2 through `pymupdf4llm.use_layout(True)`:
that path needs PyMuPDF's real trained Layout model (and typically a GPU),
which isn't practically invokable in this environment, so `get_table_details()`
is exercised directly instead -- see the label/value splice test's aside
about `insert_text`-generated geometry not reliably triggering MuPDF's own
block-fusion behavior for a similar reason.
"""

from types import SimpleNamespace

import pymupdf
import pytest

import pymupdf4llm
from pymupdf4llm.helpers.document_layout import get_table_details


@pytest.fixture(autouse=True)
def _reset_layout_mode():
    # Some other test modules (e.g. test_137.py) call use_layout(True) and
    # can leave that global toggle set for the rest of the pytest session
    # if they fail before resetting it; be explicit so these tests' outcome
    # doesn't depend on run order.
    pymupdf4llm.use_layout(False)


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


def test_to_markdown_recovers_fill_only_table_instead_of_losing_it_to_paragraph_text():
    """End-to-end: exercises the actual fallback retry logic in
    pymupdf_rag.py's to_markdown(), which defaults to table_strategy=
    "lines_strict". Without the retry, this table's content would still
    appear in the output, but as loose paragraph text rather than a table
    -- so also assert the content landed inside an actual markdown table,
    not merely somewhere in the page text."""
    doc = _make_fill_only_table_pdf()
    md = pymupdf4llm.to_markdown(doc)

    for r in range(3):
        for c in range(3):
            assert f"Cell{r}_{c}" in md, f"Cell{r}_{c} missing from output:\n{md!r}"

    assert "|Cell0_0|Cell0_1|Cell0_2|" in md, (
        f"table content was not extracted as a markdown table:\n{md!r}"
    )
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
    assert det.markdown.startswith("PRACOVNY BALIK 3-2 Project Title Long Text")
    assert det.extract == [
        [f"R{r}C{c}" for c in range(3)] for r in range(3)
    ]
    for cell in _extract_flat(det):
        assert "PRACOVNY" not in cell and "BALIK" not in cell


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
    assert det.markdown.startswith(
        "PROJECT TITLE HERE\nPage 5 of 12 -- running footer text"
    )
    assert det.extract == [
        [f"R{r}C{c}" for c in range(3)] for r in range(2)
    ]
    for cell in _extract_flat(det):
        assert "PROJECT" not in cell and "footer" not in cell


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
    so it must stay part of the table grid despite the crossing line
    (Fix 2's row-0 exclusion does not hoist it out as leading text).

    Separately, the grid-gap repair pass (wired in after this fix) then
    acts on the row's own internal geometry and splits it into two grid
    rows -- an accepted cosmetic over-split, not a regression of Fix 2;
    see the row_count/extract assertions below."""
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
    # The grid-gap repair pass's accepted cosmetic header over-split (see
    # table_grid_repair.py's module docstring) now splits this synthetic
    # header row into two rows. Column separation is still verified below.
    assert det.row_count == 4  # row 0 kept, but grid-gap repair splits it in two
    assert det.extract == [
        ["Header0", "Jednotkova cena bez DPH", "Celkove opravnene"],
        ["Sub0", "Sub1", "Sub2"],
        ["R0C0", "R0C1", "R0C2"],
        ["R1C0", "R1C1", "R1C2"],
    ]
    # The crossing line's two halves must still land in their own,
    # correct columns -- the transient clustering used only to decide
    # whether to exclude row 0 must not affect actual cell assignment,
    # which is done independently via per-char bbox overlap. This holds
    # regardless of the grid-gap repair's row split above.
    for row in det.extract:
        assert "Celkove" not in row[1]
        assert "Jednotkova" not in row[2]
    assert not det.markdown.startswith("Jednotkova")


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
