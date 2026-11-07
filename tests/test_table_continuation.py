"""Regression tests for spurious `|---|` header separators splitting a
single logical table into pieces when pymupdf.layout's GNN model detects
it as several independent "table" boxes (either several boxes on the same
page, or one table box ending at the bottom of a page and another picking
up at the top of the next).

`get_layout_locked()`/`parse_document()` process each page independently
with no state shared across pages, and previously `table_to_markdown()`
unconditionally treated every table box's row 0 as a header, emitting a
`|---|` GFM separator after it -- even when that row 0 was actually just
the next chunk of an already-started table's body. This is a different
bug from the one fixed in 3bbb653 (a swallowed *foreign* row -- a page
title/footer -- sitting above a genuine table's own header); here every
row in every box is genuine table content, just split across boxes by the
model.

Deciding that a box continues the previous table takes two steps:

- `_is_table_continuation()` (used by `parse_document()`, which tracks
  state across consecutive layout boxes -- including across page
  boundaries, skipping page-header/-footer furniture) is a geometric gate:
  near-identical OUTER bounding-box x0/x1 plus vertical contiguity
  (adjacent on the same page, or the previous table ran to the bottom of
  its page and this one starts at the top of the next). It uses the outer
  bbox rather than pymupdf.layout's interior column grid because on real
  multi-box tables that grid drifts 10+ points between boxes of the very
  same table and can even disagree on the number of columns.
- `get_table_details(prev_table=...)` then has to fit the box's text into
  the previous box's column grid: no text crossing one of its interior
  boundaries, no two cells side by side within one of its columns, and --
  unless the two grids already agree -- every one of its columns filled.
  Matching outer bounds alone would also merge an unrelated table stacked
  directly below (e.g. a 4-column table under a 2-column one). A
  continuation is extracted with the previous grid, and a header row
  repeated at the top of the box is dropped when the rows are appended.

Every box keeps a complete Markdown table. `ParsedDocument.to_markdown()`
appends a continuation box's rows to the previous table (no header, no
blank line) only when that table is the last thing it emitted -- not
after an emitted page header/footer, a page separator, or in a new page
chunk.

As with 3bbb653, there is no practical way to exercise `parse_document()`
end-to-end here (it needs PyMuPDF's real trained Layout model, typically
GPU-bound) -- so the pure geometry-decision functions and
`get_table_details()`'s row-emission behavior are exercised directly.
"""

from types import SimpleNamespace

import pymupdf
import pymupdf4llm
from pymupdf4llm.helpers import utils
from pymupdf4llm.helpers.document_layout import (
    LayoutBox,
    PageLayout,
    ParsedDocument,
    _is_table_continuation,
    _table_continuation_state,
    get_table_details,
)


def _char(c, x0, y0, x1, y1):
    return {"c": c, "bbox": (x0, y0, x1, y1)}


def _span(text, x0, y0, x1, y1, size=9.0):
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
    span = _span(text, x0, y0, x1, y1, size=size)
    return _block([_line([span], x0, y0, x1, y1)], x0, y0, x1, y1)


def _make_tab_dict(x0, y0, x1, y1, interior_v_abs, interior_h_abs):
    grid = SimpleNamespace(
        v_lines=[v - x0 for v in interior_v_abs],
        h_lines=[h - y0 for h in interior_h_abs],
    )
    return {"group_bbox": (x0, y0, x1, y1), "table_grid": grid}


def _data_row_blocks(x0, y_start, col_w, row_h, ncols, nrows, size=9.0):
    blocks = []
    for r in range(nrows):
        ry0 = y_start + r * row_h
        for c in range(ncols):
            cx0 = x0 + c * col_w
            blocks.append(_text_block(f"R{r}C{c}", cx0 + 5, ry0 + 3, cx0 + 45, ry0 + 3 + size))
    return blocks


def _extract_flat(det):
    return [cell for row in det.extract for cell in row]


PAGE_HEIGHT = 792.0  # US Letter


# ---------------------------------------------------------------------------
# table_to_markdown(skip_header=...)
# ---------------------------------------------------------------------------


def test_table_to_markdown_default_emits_header_and_separator():
    md = utils.table_to_markdown([["A", "B"], ["1", "2"]])
    assert md == "|A|B|\n|---|---|\n|1|2|\n\n"


def test_table_to_markdown_skip_header_omits_header_and_separator():
    md = utils.table_to_markdown([["1", "2"], ["3", "4"]], skip_header=True)
    assert "---" not in md
    assert md == "|1|2|\n|3|4|\n\n"


# ---------------------------------------------------------------------------
# _is_table_continuation(): pure geometry decision
# ---------------------------------------------------------------------------


def _tab_dict_at(x0, y0, x1, y1, col_w, ncols):
    interior_v = [x0 + col_w * c for c in range(1, ncols)]
    return _make_tab_dict(x0, y0, x1, y1, interior_v_abs=interior_v, interior_h_abs=[])


def test_same_page_adjacent_matching_columns_is_continuation():
    prev_tab = _tab_dict_at(100.0, 100.0, 500.0, 300.0, col_w=100.0, ncols=4)
    prev_state = _table_continuation_state(prev_tab, page_number=3, page_height=PAGE_HEIGHT)

    cur_tab = _tab_dict_at(100.0, 300.0, 500.0, 450.0, col_w=100.0, ncols=4)
    assert _is_table_continuation(prev_state, cur_tab, page_number=3, page_height=PAGE_HEIGHT)


def test_different_interior_column_count_but_same_outer_bbox_is_continuation():
    # The model's interior column grid can disagree on column count and
    # position between boxes of the very same logical table, but the outer
    # bbox x0/x1 -- the signal this geometric gate keys off -- matches, so
    # the box remains a candidate. Whether its content fits the previous
    # grid is decided by get_table_details() (tests below).
    prev_tab = _tab_dict_at(100.0, 100.0, 500.0, 300.0, col_w=100.0, ncols=4)
    prev_state = _table_continuation_state(prev_tab, page_number=3, page_height=PAGE_HEIGHT)

    cur_tab = _tab_dict_at(100.0, 300.0, 500.0, 450.0, col_w=133.33, ncols=3)
    assert _is_table_continuation(prev_state, cur_tab, page_number=3, page_height=PAGE_HEIGHT)


def test_shifted_outer_bbox_is_not_continuation():
    prev_tab = _tab_dict_at(100.0, 100.0, 500.0, 300.0, col_w=100.0, ncols=4)
    prev_state = _table_continuation_state(prev_tab, page_number=3, page_height=PAGE_HEIGHT)

    # Outer bbox x0/x1 shifted by 40pt (well past TABLE_CONTINUATION_X_TOLERANCE)
    # -- an unrelated table that merely happens to have the same column count.
    cur_tab = _tab_dict_at(140.0, 300.0, 540.0, 450.0, col_w=100.0, ncols=4)
    assert not _is_table_continuation(prev_state, cur_tab, page_number=3, page_height=PAGE_HEIGHT)


def test_large_same_page_vertical_gap_is_not_continuation():
    prev_tab = _tab_dict_at(100.0, 100.0, 500.0, 300.0, col_w=100.0, ncols=4)
    prev_state = _table_continuation_state(prev_tab, page_number=3, page_height=PAGE_HEIGHT)

    # 200pt gap -- unrelated content (a heading, a paragraph) plausibly sits
    # between the two boxes even though neither was recorded as breaking it
    # in this pure-function test.
    cur_tab = _tab_dict_at(100.0, 500.0, 500.0, 650.0, col_w=100.0, ncols=4)
    assert not _is_table_continuation(prev_state, cur_tab, page_number=3, page_height=PAGE_HEIGHT)


def test_bottom_of_page_to_top_of_next_page_is_continuation():
    prev_tab = _tab_dict_at(100.0, 600.0, 500.0, PAGE_HEIGHT - 20.0, col_w=100.0, ncols=4)
    prev_state = _table_continuation_state(prev_tab, page_number=5, page_height=PAGE_HEIGHT)

    cur_tab = _tab_dict_at(100.0, 60.0, 500.0, 300.0, col_w=100.0, ncols=4)
    assert _is_table_continuation(prev_state, cur_tab, page_number=6, page_height=PAGE_HEIGHT)


def test_cross_page_but_not_near_page_edges_is_not_continuation():
    # Previous table box ends mid-page (nowhere near the bottom margin), so
    # even though the next page's table starts near the top, this looks
    # like two separate, unrelated tables rather than one split by a page
    # break.
    prev_tab = _tab_dict_at(100.0, 100.0, 500.0, 300.0, col_w=100.0, ncols=4)
    prev_state = _table_continuation_state(prev_tab, page_number=5, page_height=PAGE_HEIGHT)

    cur_tab = _tab_dict_at(100.0, 60.0, 500.0, 300.0, col_w=100.0, ncols=4)
    assert not _is_table_continuation(prev_state, cur_tab, page_number=6, page_height=PAGE_HEIGHT)


def test_skipped_page_is_not_continuation():
    prev_tab = _tab_dict_at(100.0, 600.0, 500.0, PAGE_HEIGHT - 20.0, col_w=100.0, ncols=4)
    prev_state = _table_continuation_state(prev_tab, page_number=5, page_height=PAGE_HEIGHT)

    cur_tab = _tab_dict_at(100.0, 60.0, 500.0, 300.0, col_w=100.0, ncols=4)
    # page 7 -- page 6 was skipped (e.g. filtered out of the run) -- not a
    # direct continuation.
    assert not _is_table_continuation(prev_state, cur_tab, page_number=7, page_height=PAGE_HEIGHT)


def test_no_previous_table_is_not_continuation():
    cur_tab = _tab_dict_at(100.0, 100.0, 500.0, 300.0, col_w=100.0, ncols=4)
    assert not _is_table_continuation(None, cur_tab, page_number=1, page_height=PAGE_HEIGHT)


# ---------------------------------------------------------------------------
# get_table_details(prev_table=...): a box that lines up with the previous
# table is only a continuation if its content fits that table's column
# grid; it is then extracted with that grid.
# ---------------------------------------------------------------------------


def _grid_table(x0, y0, col_w, ncols, row_h, nrows, v_abs=None, rows=None, prefix="R"):
    """(tab_dict, blocks) for a table of `nrows` rows. Each row's cells
    are `rows[r]` (one string per column) or "{prefix}{r}C{c}". `v_abs`
    overrides the interior column boundaries the model predicted, to
    simulate its grid noise between boxes of one table."""
    x1 = x0 + col_w * ncols
    y1 = y0 + row_h * nrows
    if v_abs is None:
        v_abs = [x0 + col_w * c for c in range(1, ncols)]
    tab_dict = _make_tab_dict(
        x0, y0, x1, y1,
        interior_v_abs=v_abs,
        interior_h_abs=[y0 + row_h * r for r in range(1, nrows)],
    )
    blocks = []
    for r in range(nrows):
        ry0 = y0 + r * row_h
        texts = rows[r] if rows else [f"{prefix}{r}C{c}" for c in range(ncols)]
        for c, text in enumerate(texts):
            cx0 = x0 + c * col_w
            blocks.append(_text_block(text, cx0 + 5, ry0 + 3, cx0 + 45, ry0 + 12))
    return tab_dict, blocks


def _state_after(tab_dict, blocks, page_number=1, prev_table=None):
    """Extract a box like parse_document() does and return its details plus
    the continuation state the next box is compared against."""
    det = get_table_details(tab_dict, blocks, prev_table=prev_table)
    state = _table_continuation_state(tab_dict, page_number, PAGE_HEIGHT, det)
    return det, state


def test_continuation_with_matching_grid_keeps_all_rows():
    first, first_blocks = _grid_table(100.0, 100.0, 100.0, 3, 20.0, 2)
    _, prev = _state_after(first, first_blocks)
    second, second_blocks = _grid_table(100.0, 140.0, 100.0, 3, 20.0, 3, prefix="S")
    assert _is_table_continuation(prev, second, page_number=1, page_height=PAGE_HEIGHT)

    det = get_table_details(second, second_blocks, prev_table=prev)

    assert det.is_continuation
    assert det.row_count == 3
    assert det.extract == [[f"S{r}C{c}" for c in range(3)] for r in range(3)]
    assert det.continuation_markdown == (
        "|S0C0|S0C1|S0C2|\n|S1C0|S1C1|S1C2|\n|S2C0|S2C1|S2C2|\n\n"
    )
    # standalone form, for when the renderer cannot join it
    assert det.markdown.startswith("|S0C0|S0C1|S0C2|\n|---|---|---|\n")


def test_table_without_previous_table_is_not_a_continuation():
    tab_dict, blocks = _grid_table(100.0, 100.0, 100.0, 3, 20.0, 3)

    det = get_table_details(tab_dict, blocks)

    assert not det.is_continuation
    assert det.continuation_markdown is None
    assert "|---|---|---|" in det.markdown


def test_under_split_continuation_grid_is_reconciled_to_previous_columns():
    """The model merged two columns of the continuation box into one (3
    predicted columns instead of 4): its rows are re-extracted with the
    previous box's 4 columns, so every cell lands in its own column."""
    first, first_blocks = _grid_table(100.0, 100.0, 100.0, 4, 20.0, 2)
    _, prev = _state_after(first, first_blocks)
    second, second_blocks = _grid_table(
        100.0, 140.0, 100.0, 4, 20.0, 2, v_abs=[200.0, 300.0], prefix="S"
    )
    assert _is_table_continuation(prev, second, page_number=1, page_height=PAGE_HEIGHT)

    det = get_table_details(second, second_blocks, prev_table=prev)

    assert det.is_continuation
    assert det.col_count == 4
    assert det.extract == [[f"S{r}C{c}" for c in range(4)] for r in range(2)]
    assert det.continuation_markdown == "|S0C0|S0C1|S0C2|S0C3|\n|S1C0|S1C1|S1C2|S1C3|\n\n"


def test_over_split_continuation_grid_is_reconciled_to_previous_columns():
    """The model put a spurious boundary through the middle column of the
    continuation box (4 predicted columns instead of 3), cutting its text
    in half: the previous box's 3 columns keep each cell's text whole."""
    first, first_blocks = _grid_table(100.0, 100.0, 100.0, 3, 20.0, 2)
    _, prev = _state_after(first, first_blocks)
    second, second_blocks = _grid_table(
        100.0, 140.0, 100.0, 3, 20.0, 2, v_abs=[200.0, 225.0, 300.0], prefix="S"
    )

    det = get_table_details(second, second_blocks, prev_table=prev)

    assert det.is_continuation
    assert det.col_count == 3
    assert det.extract == [[f"S{r}C{c}" for c in range(3)] for r in range(2)]


def test_four_column_table_under_two_column_table_is_not_a_continuation():
    """Same outer bounds, directly below: geometrically a continuation
    candidate, but two of its cells sit side by side in each of the
    previous table's columns, so it is a separate table."""
    first, first_blocks = _grid_table(100.0, 100.0, 200.0, 2, 20.0, 2)
    _, prev = _state_after(first, first_blocks)
    second, second_blocks = _grid_table(100.0, 140.0, 100.0, 4, 20.0, 2)
    assert _is_table_continuation(prev, second, page_number=1, page_height=PAGE_HEIGHT)

    det = get_table_details(second, second_blocks, prev_table=prev)

    assert not det.is_continuation
    assert det.continuation_markdown is None
    assert det.col_count == 4
    assert det.markdown.startswith("|R0C0|R0C1|R0C2|R0C3|\n|---|---|---|---|\n")


def test_two_column_table_under_four_column_table_is_not_a_continuation():
    """Each of its cells falls into a distinct column of the previous
    table, but it leaves half of those columns empty: no evidence that it
    belongs to that grid."""
    first, first_blocks = _grid_table(100.0, 100.0, 100.0, 4, 20.0, 2)
    _, prev = _state_after(first, first_blocks)
    second, second_blocks = _grid_table(100.0, 140.0, 200.0, 2, 20.0, 2)

    det = get_table_details(second, second_blocks, prev_table=prev)

    assert not det.is_continuation
    assert det.col_count == 2
    assert "|---|---|" in det.markdown


def test_text_crossing_a_previous_column_boundary_is_not_a_continuation():
    first, first_blocks = _grid_table(100.0, 100.0, 100.0, 3, 20.0, 2)
    _, prev = _state_after(first, first_blocks)
    # a single-column box whose text runs straight across both of the
    # previous table's interior boundaries
    second = _make_tab_dict(100.0, 140.0, 400.0, 180.0, interior_v_abs=[], interior_h_abs=[160.0])
    second_blocks = [
        _text_block("one long sentence spanning the table", 105.0, 143.0, 390.0, 152.0),
        _text_block("and another one right below it here", 105.0, 163.0, 390.0, 172.0),
    ]

    det = get_table_details(second, second_blocks, prev_table=prev)

    assert not det.is_continuation


def test_repeated_header_row_is_dropped_when_joined_but_kept_standalone():
    header = ["Name", "Qty", "Price"]
    first, first_blocks = _grid_table(
        100.0, 680.0, 100.0, 3, 20.0, 3, rows=[header, ["a", "1", "10"], ["b", "2", "20"]]
    )
    _, prev = _state_after(first, first_blocks, page_number=1)
    second, second_blocks = _grid_table(
        100.0, 60.0, 100.0, 3, 20.0, 2, rows=[header, ["c", "3", "30"]]
    )
    assert _is_table_continuation(prev, second, page_number=2, page_height=PAGE_HEIGHT)

    det = get_table_details(second, second_blocks, prev_table=prev)

    assert det.is_continuation
    assert det.continuation_markdown == "|c|3|30|\n\n"
    assert det.markdown == "|Name|Qty|Price|\n|---|---|---|\n|c|3|30|\n\n"


def test_continuation_chain_keeps_using_the_first_boxs_grid_and_header():
    header = ["Name", "Qty", "Price"]
    first, first_blocks = _grid_table(
        100.0, 700.0, 100.0, 3, 20.0, 2, rows=[header, ["a", "1", "10"]]
    )
    _, state = _state_after(first, first_blocks, page_number=1)
    # same page 2: the model over-split this box's middle column
    second, second_blocks = _grid_table(
        100.0, 60.0, 100.0, 3, 20.0, 1, v_abs=[200.0, 225.0, 300.0], rows=[["b", "2", "20"]]
    )
    det2, state = _state_after(second, second_blocks, page_number=2, prev_table=state)
    assert det2.is_continuation
    third, third_blocks = _grid_table(
        100.0, 80.0, 100.0, 3, 20.0, 2, rows=[header, ["c", "3", "30"]]
    )
    assert _is_table_continuation(state, third, page_number=2, page_height=PAGE_HEIGHT)

    det = get_table_details(third, third_blocks, prev_table=state)

    assert det.is_continuation
    assert det.v_lines == [100.0, 200.0, 300.0, 400.0]
    assert det.continuation_markdown == "|c|3|30|\n\n"


def test_continuation_suppresses_swallowed_header_exclusion():
    """A continuation box's row 0 must never be run through the
    swallowed-foreign-header exclusion -- it is a genuine body row of the
    already-started table, not a title/footer that leaked into the box,
    even if it happens to look sparse."""
    x0, y0 = 100.0, 100.0
    col_w, ncols = 100.0, 3
    row0_h, data_row_h, nrows_data = 30.0, 20.0, 2
    x1 = x0 + col_w * ncols
    y_after_row0 = y0 + row0_h
    y1 = y_after_row0 + data_row_h * nrows_data

    tab_dict = _make_tab_dict(
        x0, y0, x1, y1,
        interior_v_abs=[x0 + col_w, x0 + 2 * col_w],
        interior_h_abs=[y_after_row0 + data_row_h * i for i in range(nrows_data)],
    )
    # Sparse, single-line, column-crossing row 0 -- exactly the shape the
    # exclusion targets in a table's first box.
    sparse_row = _text_block(
        "spolu 448000 eur", x0 + 10, y0 + 5, x1 - 20, y0 + 25
    )
    blocks = [sparse_row] + _data_row_blocks(
        x0, y_after_row0, col_w, data_row_h, ncols, nrows_data
    )
    first, first_blocks = _grid_table(x0, 40.0, col_w, ncols, 20.0, 3)
    _, prev = _state_after(first, first_blocks)

    det = get_table_details(tab_dict, blocks, prev_table=prev)

    assert det.is_continuation  # grids agree
    assert det.row_count == 3  # row 0 kept as a body row, not hoisted out
    assert not det.excluded_textlines
    # Split across its row's cells by column, like any other body row. The
    # per-char column split can land mid-word, so compare with whitespace
    # stripped rather than expecting exact words to survive intact.
    row0_chars = "".join(det.extract[0]).replace(" ", "")
    assert row0_chars == "spolu448000eur"
    assert "---" not in det.continuation_markdown


# ---------------------------------------------------------------------------
# ParsedDocument.to_markdown(): a continuation box is appended to the
# previous table (no header, no blank-line gap -- GFM tables end at the
# first blank line) only when that table is the last thing emitted.
# Otherwise it is emitted as the complete table it also carries.
# ---------------------------------------------------------------------------

FIRST_MD = "|A|B|\n|---|---|\n|1|2|\n\n"
SECOND_FULL_MD = "|3|4|\n|---|---|\n|5|6|\n\n"
SECOND_JOIN_MD = "|3|4|\n|5|6|\n\n"


def _table_layout_box(markdown, continuation_markdown=None):
    return LayoutBox(
        x0=100.0, y0=100.0, x1=400.0, y1=200.0, boxclass="table",
        table={
            "bbox": [100.0, 100.0, 400.0, 200.0],
            "row_count": 2,
            "col_count": 2,
            "cells": None,
            "extract": None,
            "markdown": markdown,
            "is_continuation": continuation_markdown is not None,
            "continuation_markdown": continuation_markdown,
        },
    )


def _furniture_box(boxclass, text):
    span = {
        "text": text,
        "bbox": (100.0, 20.0, 300.0, 30.0),
        "size": 9.0,
        "flags": 0,
        "char_flags": 0,
        "font": "Helvetica",
        "alpha": 255,
        "block": 0,
        "line": 0,
    }
    return LayoutBox(
        x0=100.0, y0=20.0, x1=300.0, y1=30.0, boxclass=boxclass,
        textlines=[{"bbox": pymupdf.Rect(span["bbox"]), "spans": [span]}],
    )


def _first():
    return _table_layout_box(FIRST_MD)


def _second():
    return _table_layout_box(SECOND_FULL_MD, SECOND_JOIN_MD)


def _doc(*pages):
    return ParsedDocument(
        page_count=len(pages),
        toc=[],
        metadata={},
        pages=[
            PageLayout(page_number=n, width=612.0, height=792.0, boxes=list(boxes))
            for n, boxes in enumerate(pages, 1)
        ],
    )


def test_continuation_box_joins_previous_table_on_same_page():
    md = _doc([_first(), _second()]).to_markdown()

    assert "|1|2|\n|3|4|\n|5|6|" in md
    assert md.count("|---|") == 1


def test_continuation_box_joins_previous_table_across_pages():
    # The continuation box is the first box on its page, so the gap to
    # close is at the tail of the previous page's already-flushed text.
    md = _doc([_first()], [_second()]).to_markdown()

    assert "|1|2|\n|3|4|\n|5|6|" in md
    assert md.count("|---|") == 1


def test_page_chunks_continuation_page_starts_with_a_complete_table():
    chunks = _doc([_first()], [_second()]).to_markdown(page_chunks=True)

    assert chunks[0]["text"].startswith("|A|B|\n|---|---|\n|1|2|")
    assert chunks[1]["text"].startswith("|3|4|\n|---|---|\n|5|6|")


def test_page_separator_between_boxes_keeps_a_complete_table():
    md = _doc([_first()], [_second()]).to_markdown(page_separators=True)

    assert "|3|4|\n|---|---|\n|5|6|" in md
    assert "|1|2|\n|3|4|" not in md


def test_emitted_page_header_between_boxes_keeps_a_complete_table():
    pages = ([_first()], [_furniture_box("page-header", "Running header"), _second()])

    md = _doc(*pages).to_markdown()
    assert "Running header" in md
    assert "|3|4|\n|---|---|\n|5|6|" in md

    # with the header suppressed, nothing comes between the two tables
    md = _doc(*pages).to_markdown(header=False)
    assert "|1|2|\n|3|4|\n|5|6|" in md


def test_emitted_page_footer_between_boxes_keeps_a_complete_table():
    pages = ([_first(), _furniture_box("page-footer", "Page 1 of 2")], [_second()])

    md = _doc(*pages).to_markdown()
    assert "Page 1 of 2" in md
    assert "|3|4|\n|---|---|\n|5|6|" in md

    md = _doc(*pages).to_markdown(footer=False)
    assert "|1|2|\n|3|4|\n|5|6|" in md


def test_text_box_between_tables_keeps_a_complete_table():
    md = _doc([_first(), _furniture_box("text", "Some paragraph"), _second()]).to_markdown()

    assert "|3|4|\n|---|---|\n|5|6|" in md


def test_joined_box_holding_only_a_repeated_header_adds_no_rows():
    repeated = _table_layout_box("|A|B|\n|---|---|\n\n", "")

    md = _doc([_first()], [repeated, _second()]).to_markdown()

    assert md.count("|A|B|") == 1
    assert "|1|2|\n|3|4|\n|5|6|" in md
