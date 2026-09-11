"""tests/test_table_grid_repair.py — unit tests for the grid-gap-repair
algorithm. Fast, synthetic-fixture tests only; no PyMuPDF Layout model or
real PDF is required (see test_table_header_swallowing.py for why: the
Layout model needs a GPU and isn't practically invokable in this
environment, so the algorithm is exercised directly against hand-built
RAWDICT-shaped blocks instead)."""
import pymupdf
import pytest

from pymupdf4llm.helpers.table_grid_repair import (
    band_rects,
    boundaries_of,
    evaluate_gap,
    merge_overlapping,
    repair_grid_gaps,
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


def test_merge_overlapping_merges_touching_row_rects_but_keeps_gapped_ones_apart():
    rects = [
        pymupdf.Rect(0, 0, 10, 5),
        pymupdf.Rect(0, 4.8, 10, 9),   # overlaps the first by 0.2pt on the row axis
        pymupdf.Rect(0, 20, 10, 25),   # far away -- stays separate
    ]
    merged = merge_overlapping(rects, axis="row")
    assert len(merged) == 2
    assert merged[0].y0 == 0 and merged[0].y1 == 9
    assert merged[1].y0 == 20 and merged[1].y1 == 25


def test_boundaries_of_row_axis_is_midpoint_between_consecutive_bands():
    strip = [pymupdf.Rect(0, 0, 10, 5), pymupdf.Rect(0, 9, 10, 14)]
    assert boundaries_of(strip, axis="row") == [7.0]


def test_evaluate_gap_no_defect_when_only_one_strip_shows_multiple_bands():
    # strip 0 (x in [0,50]) legitimately wraps to two lines; strip 1
    # (x in [50,100]) is a normal single-line value at the same height as
    # strip 0's *first* line only -- so strip 1 never shows >=2 bands.
    # This is the "one column legitimately wraps" false-positive shape the
    # corroboration gate must suppress.
    blocks = [
        _text_block("wrapped line one", 5, 0, 45, 5),
        _text_block("wrapped line two", 5, 9, 45, 14),
        _text_block("single value", 55, 0, 95, 5),
    ]
    result = evaluate_gap(blocks, axis="row", gap_lo=0, gap_hi=14, cross_lines=[0, 50, 100])
    assert result["is_defect"] is False
    assert result["new_boundaries"] == []


def test_evaluate_gap_finds_defect_when_two_strips_corroborate():
    # Both strip 0 and strip 1 independently show 2 bands at matching
    # offsets -- two real rows genuinely fused into one grid row.
    blocks = [
        _text_block("R0C0", 5, 0, 45, 5),
        _text_block("R1C0", 5, 9, 45, 14),
        _text_block("R0C1", 55, 0, 95, 5),
        _text_block("R1C1", 55, 9, 95, 14),
    ]
    result = evaluate_gap(blocks, axis="row", gap_lo=0, gap_hi=14, cross_lines=[0, 50, 100])
    assert result["is_defect"] is True
    assert len(result["new_boundaries"]) == 1
    assert 5 < result["new_boundaries"][0] < 9
    assert result["support"] == [2]


def test_evaluate_gap_reference_strip_rule_ignores_higher_band_count_strip():
    # strip 0 has 3 bands (a header-only column that also happens to
    # subdivide further); strip 1 has 2 bands at a *different* internal
    # split point. Per the reference-strip rule, only strip 1 (the
    # minimum-band corroborating strip) may source boundary candidates.
    blocks = [
        _text_block("A", 5, 0, 45, 3),
        _text_block("B", 5, 6, 45, 9),
        _text_block("C", 5, 12, 45, 15),
        _text_block("X", 55, 0, 95, 5),
        _text_block("Y", 55, 10, 95, 15),
    ]
    result = evaluate_gap(blocks, axis="row", gap_lo=0, gap_hi=15, cross_lines=[0, 50, 100])
    assert result["is_defect"] is True
    assert len(result["new_boundaries"]) == 1
    assert 5 < result["new_boundaries"][0] < 10


def test_evaluate_gap_rejects_boundary_that_would_bisect_a_single_line_header_cell():
    # Real-document regression (WP-3-3 table header): strip 0 and strip 2
    # each wrap their header label to 2 lines, coincidentally corroborating
    # a boundary at y~7 -- but strip 1 ("Kategória výdavkov") is a single
    # unwrapped line straddling that exact midpoint (y 5 to 9). Splitting
    # there would bisect strip 1's line rather than separate two rows.
    blocks = [
        _text_block("Jednotkova cena bez DPH", 5, 0, 45, 5),
        _text_block("(EUR)", 5, 9, 45, 14),
        _text_block("Kategoria vydavkov", 55, 5, 95, 9),
        _text_block("Celkove opravnene vydavky", 105, 0, 145, 5),
        _text_block("(EUR)", 105, 9, 145, 14),
    ]
    result = evaluate_gap(
        blocks, axis="row", gap_lo=0, gap_hi=14, cross_lines=[0, 50, 100, 150]
    )
    assert result["is_defect"] is False
    assert result["new_boundaries"] == []


def test_evaluate_gap_keeps_other_corroborated_boundaries_when_one_is_rejected():
    # Same shape as the rejection case above, but with a second, independent
    # gap-worth of corroboration far enough away that only the first
    # candidate collides with strip 1's single line -- the second candidate
    # must still survive.
    blocks = [
        _text_block("Jednotkova cena bez DPH", 5, 0, 45, 5),
        _text_block("(EUR)", 5, 9, 45, 14),
        _text_block("Kategoria vydavkov", 55, 5, 95, 9),
        _text_block("Celkove opravnene vydavky", 105, 0, 145, 5),
        _text_block("(EUR)", 105, 9, 145, 14),
        _text_block("R0C3", 155, 20, 195, 25),
        _text_block("R1C3", 155, 29, 195, 34),
        _text_block("R0C4", 205, 20, 245, 25),
        _text_block("R1C4", 205, 29, 245, 34),
    ]
    result = evaluate_gap(
        blocks, axis="row", gap_lo=0, gap_hi=34, cross_lines=[0, 50, 100, 150, 200, 250]
    )
    assert result["is_defect"] is True
    assert len(result["new_boundaries"]) == 1
    assert 25 < result["new_boundaries"][0] < 29


def test_repair_grid_gaps_extends_and_sorts_h_lines_for_a_fused_row():
    blocks = [
        _text_block("R0C0", 5, 0, 45, 5),
        _text_block("R1C0", 5, 9, 45, 14),
        _text_block("R0C1", 55, 0, 95, 5),
        _text_block("R1C1", 55, 9, 95, 14),
    ]
    h_lines = [0, 14, 28]  # one fused row [0,14] and one normal row [14,28]
    v_lines = [0, 50, 100]
    new_h, new_v = repair_grid_gaps(blocks, h_lines, v_lines)
    assert len(new_h) == 4  # one new interior boundary inserted
    assert new_h[0] == 0 and new_h[-1] == 28
    assert new_v == v_lines  # no column defect in this fixture
