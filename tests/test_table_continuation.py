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

The fix adds `_is_table_continuation()` (used by `parse_document()`, which
tracks state across consecutive layout boxes -- including across page
boundaries, skipping page-header/-footer furniture) to detect when a table
box is a direct continuation of the immediately preceding one: near-
identical OUTER bounding-box x0/x1 (each box's overall left/right table
edge), plus vertical contiguity (adjacent on the same page, or the previous
table ran to the bottom of its page and this one starts at the top of the
next). `get_table_details()` then threads that signal through to
`table_to_markdown()`'s new `skip_header` argument so a continuation box's
own row 0 is emitted as a plain body row, with no header treatment and no
`|---|` separator.

Column matching deliberately uses each box's outer x0/x1, not
pymupdf.layout's interior column grid (`table_grid.v_lines`): on real
multi-box tables the model's interior column-boundary detection is noisy
enough between boxes of the very same table to drift 10+ points and even
disagree on the number of columns -- that per-box grid noise is a separate,
out-of-scope bug ("Bug 2", grid boundary under-prediction) -- while the
outer bbox stays essentially fixed across genuine continuations.

As with 3bbb653, there is no practical way to exercise `parse_document()`
end-to-end here (it needs PyMuPDF's real trained Layout model, typically
GPU-bound) -- so the pure geometry-decision functions and
`get_table_details()`'s row-emission behavior are exercised directly.
"""

from types import SimpleNamespace

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
    # This is the real-world case the bbox-based redesign exists for: the
    # model's interior column grid disagrees on column count/position
    # between boxes of the very same logical table (out-of-scope "Bug 2"
    # noise), but the outer bbox x0/x1 -- the signal this function actually
    # keys off -- matches, so it's still recognized as a continuation.
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
# get_table_details(is_continuation=True): row 0 is emitted as a plain body
# row, not a header with a `|---|` separator.
# ---------------------------------------------------------------------------


def test_continuation_table_has_no_header_separator_and_keeps_all_rows():
    x0, y0 = 100.0, 100.0
    col_w, ncols = 100.0, 3
    row_h, nrows = 20.0, 3
    x1 = x0 + col_w * ncols
    y1 = y0 + row_h * nrows

    tab_dict = _make_tab_dict(
        x0, y0, x1, y1,
        interior_v_abs=[x0 + col_w, x0 + 2 * col_w],
        interior_h_abs=[y0 + row_h * i for i in range(1, nrows)],
    )
    blocks = _data_row_blocks(x0, y0, col_w, row_h, ncols, nrows)

    det = get_table_details(tab_dict, blocks, is_continuation=True)

    assert det.row_count == 3
    assert "---" not in det.markdown
    assert det.markdown.startswith("|R0C0|R0C1|R0C2|\n")
    assert det.extract == [[f"R{r}C{c}" for c in range(3)] for r in range(3)]


def test_non_continuation_table_still_gets_header_separator():
    x0, y0 = 100.0, 100.0
    col_w, ncols = 100.0, 3
    row_h, nrows = 20.0, 3
    x1 = x0 + col_w * ncols
    y1 = y0 + row_h * nrows

    tab_dict = _make_tab_dict(
        x0, y0, x1, y1,
        interior_v_abs=[x0 + col_w, x0 + 2 * col_w],
        interior_h_abs=[y0 + row_h * i for i in range(1, nrows)],
    )
    blocks = _data_row_blocks(x0, y0, col_w, row_h, ncols, nrows)

    det = get_table_details(tab_dict, blocks, is_continuation=False)

    assert "|---|---|---|" in det.markdown


def test_continuation_flag_suppresses_swallowed_header_exclusion():
    """A continuation box's row 0 must never be run through the
    swallowed-foreign-header exclusion (3bbb653) either -- it is a genuine
    body row of the already-started table, not a title/footer that leaked
    into the box, even if it happens to look sparse."""
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
    # Sparse, single-line, column-crossing row 0 -- exactly the shape
    # 3bbb653's exclusion targets when is_continuation is False.
    sparse_row = _text_block(
        "spolu 448000 eur", x0 + 10, y0 + 5, x1 - 20, y0 + 25
    )
    blocks = [sparse_row] + _data_row_blocks(
        x0, y_after_row0, col_w, data_row_h, ncols, nrows_data
    )

    det = get_table_details(tab_dict, blocks, is_continuation=True)

    assert det.row_count == 3  # row 0 kept as a body row, not hoisted out
    # Split across its row's cells by column, like any other body row --
    # not hoisted out whole as a plain-text block ahead of the table. The
    # per-char column split can land mid-word, so compare with whitespace
    # stripped rather than expecting exact words to survive intact.
    row0_chars = "".join(det.extract[0]).replace(" ", "")
    assert row0_chars == "spolu448000eur"
    assert "---" not in det.markdown
    assert not det.markdown.startswith("spolu 448000 eur\n\n")


# ---------------------------------------------------------------------------
# ParsedDocument.to_markdown(): a continuation box's rows must be joined
# directly onto the previous table box's last row -- no blank-line gap.
# GFM tables end at the first blank line, so even with skip_header already
# dropping the second box's own header/separator, leaving the blank line
# that to_markdown() otherwise inserts between every box would still visibly
# split the table in two.
# ---------------------------------------------------------------------------


def _table_layout_box(markdown, is_continuation):
    return LayoutBox(
        x0=100.0, y0=100.0, x1=400.0, y1=200.0, boxclass="table",
        table={
            "bbox": [100.0, 100.0, 400.0, 200.0],
            "row_count": 1,
            "col_count": 2,
            "cells": None,
            "extract": None,
            "markdown": markdown,
            "is_continuation": is_continuation,
        },
    )


def test_continuation_table_box_joins_without_blank_line_in_to_markdown():
    first = _table_layout_box("|A|B|\n|---|---|\n|1|2|\n\n", is_continuation=False)
    second = _table_layout_box("|3|4|\n\n", is_continuation=True)
    page = PageLayout(page_number=1, width=612.0, height=792.0, boxes=[first, second])
    doc = ParsedDocument(pages=[page])

    md = doc.to_markdown()

    assert "|1|2|\n|3|4|" in md
    assert "|1|2|\n\n|3|4|" not in md


def test_cross_page_continuation_table_box_joins_without_blank_line_in_to_markdown():
    # Real-world shape from 6634064.pdf's WP 3-2 budget table: the
    # continuation box is the *first* box on the next page, so md_string
    # (this page's accumulator) is still empty when it's processed -- the
    # blank-line gap to close sits at the tail of document_output (the
    # previous page's already-flushed text), not in the current md_string.
    first = _table_layout_box("|A|B|\n|---|---|\n|1|2|\n\n", is_continuation=False)
    page1 = PageLayout(page_number=1, width=612.0, height=792.0, boxes=[first])

    second = _table_layout_box("|3|4|\n\n", is_continuation=True)
    page2 = PageLayout(page_number=2, width=612.0, height=792.0, boxes=[second])

    doc = ParsedDocument(pages=[page1, page2])

    md = doc.to_markdown()

    assert "|1|2|\n|3|4|" in md
    assert "|1|2|\n\n|3|4|" not in md
