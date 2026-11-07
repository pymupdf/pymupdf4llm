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
    # The model's grid drew one body row [10,34]; the text inside shows two
    # rows, each corroborated by both columns.
    x0, y0, x1, y1 = 0.0, 0.0, 100.0, 34.0
    # H0/H1 sit far enough apart to stay two separate lines for the row-0
    # classification, so it keeps the header row as is.
    header_row = [
        _text_block("H0", 5, 0, 20, 5),
        _text_block("H1", 80, 0, 95, 5),
    ]
    fused_row = [
        _text_block("R0C0", 5, 20, 45, 25),
        _text_block("R1C0", 5, 29, 45, 34),
        _text_block("R0C1", 55, 20, 95, 25),
        _text_block("R1C1", 55, 29, 95, 34),
    ]
    tab_dict = _make_tab_dict(x0, y0, x1, y1, interior_v_abs=[50.0], interior_h_abs=[10.0])
    det = get_table_details(tab_dict, header_row + fused_row)
    assert det.row_count == 3
    assert det.col_count == 2
    assert det.extract == [["H0", "H1"], ["R0C0", "R0C1"], ["R1C0", "R1C1"]]


def test_colspan_header_lands_in_its_columns_after_a_column_is_inserted_before_it():
    # Column 0 of the model's grid holds two columns (two headers, two
    # values). The repair inserts their boundary, which shifts the columns
    # of the colspan header "Combined Header" from (1, 2) to (2, 3); its
    # text must follow it there.
    x0, y0, x1, y1 = 100.0, 100.0, 500.0, 140.0
    tab_dict = _make_tab_dict(
        x0, y0, x1, y1, interior_v_abs=[200.0, 300.0, 400.0], interior_h_abs=[120.0]
    )
    blocks = [
        _text_block("LabelA", 105, 105, 135, 113),
        _text_block("LabelB", 175, 105, 195, 113),
        _text_block("Combined Header", 210, 105, 390, 113),
        _text_block("Label3", 410, 105, 490, 113),
        _text_block("A0", 105, 125, 125, 133),
        _text_block("B0", 175, 125, 195, 133),
        _text_block("R0C1", 205, 125, 245, 133),
        _text_block("R0C2", 305, 125, 345, 133),
        _text_block("R0C3", 405, 125, 445, 133),
    ]
    det = get_table_details(tab_dict, blocks)
    assert det.col_count == 5
    assert det.row_count == 2
    assert det.extract == [
        ["LabelA", "LabelB", "Combined Header", "Combined Header", "Label3"],
        ["A0", "B0", "R0C1", "R0C2", "R0C3"],
    ]


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
