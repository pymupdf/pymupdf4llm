"""tests/test_table_grid_repair_integration.py -- confirms grid-gap repair
is actually wired into get_table_details() end to end, using the same
synthetic tab_dict / RAWDICT-block builder pattern as
test_table_header_swallowing.py."""
from types import SimpleNamespace

from pymupdf4llm.helpers.document_layout import get_table_details


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


def test_get_table_details_repairs_a_fused_row():
    # A 2-column, notionally-1-row table where the model's grid drew one
    # row [20,34], but the text inside actually shows two independently
    # corroborating bands per column at y~[20,25] and y~[29,34] -- a
    # genuine fused-row defect.
    x0, y0, x1, y1 = 0.0, 0.0, 100.0, 34.0
    header_row = [
        # NOTE: x-coordinates widened from the plan's literal (5,45)/(55,95)
        # -- that left only a 10pt gap between H0/H1, well under
        # get_raw_lines' require_x_continuity merge threshold
        # (max(5.0, size*4.0) == 36pt for size 9.0 text), so the two spans
        # spliced into ONE reconstructed line ("H0H1") that crosses the
        # interior column boundary (v=50) -- exactly the signature the
        # pre-existing (unrelated, correctly-behaving) row-0
        # swallowed-header heuristic in get_table_details() treats as a
        # false title/footer and drops from the grid, which has nothing to
        # do with grid-gap repair. Widening the gap here (>36pt) keeps
        # H0/H1 as two distinct lines so this test actually exercises the
        # grid-gap-repair wiring.
        _text_block("H0", 5, 0, 20, 5),
        _text_block("H1", 80, 0, 95, 5),
    ]
    fused_row = [
        _text_block("R0C0", 5, 20, 45, 25),
        _text_block("R1C0", 5, 29, 45, 34),
        _text_block("R0C1", 55, 20, 95, 25),
        _text_block("R1C1", 55, 29, 95, 34),
    ]
    blocks = header_row + fused_row
    # Two grid rows before repair: [0,10] (header) and [10,34] (fused body).
    tab_dict = _make_tab_dict(x0, y0, x1, y1, interior_v_abs=[50.0], interior_h_abs=[10.0])
    det = get_table_details(tab_dict, blocks)
    assert det.row_count == 3  # header + the two repaired body rows
    assert det.col_count == 2


def test_get_table_details_no_repair_when_grid_is_already_correct():
    x0, y0, x1, y1 = 0.0, 0.0, 100.0, 20.0
    blocks = [
        # See the widened-gap note in the previous test -- same fix, same
        # reason (avoid a false row-0 swallowed-header exclusion that is
        # unrelated to grid-gap repair).
        _text_block("H0", 5, 0, 20, 5),
        _text_block("H1", 80, 0, 95, 5),
        _text_block("R0C0", 5, 12, 45, 17),
        _text_block("R0C1", 55, 12, 95, 17),
    ]
    tab_dict = _make_tab_dict(x0, y0, x1, y1, interior_v_abs=[50.0], interior_h_abs=[10.0])
    det = get_table_details(tab_dict, blocks)
    assert det.row_count == 2
    assert det.col_count == 2
