"""tests/test_table_grid_repair.py -- unit tests for the grid-gap repair
(table_grid_repair.py), on hand-built RAWDICT-shaped blocks so that no
Layout model is needed. Every case states the grid the model produced and
the boundaries that must (or must not) be inserted into it."""

from pymupdf4llm.helpers.table_grid_repair import (
    TableText,
    gap_boundaries,
    repair_grid_gaps,
)


def _span(text, x0, y0, x1, y1, size=9.0, origin=None, asc=None, desc=None):
    n = len(text)
    cw = (x1 - x0) / max(n, 1)
    chars = []
    for i, ch in enumerate(text):
        char = {"c": ch, "bbox": (x0 + i * cw, y0, x0 + (i + 1) * cw, y1)}
        if origin is not None:
            char["origin"] = (x0 + i * cw, origin)
        chars.append(char)
    span = {
        "bbox": (x0, y0, x1, y1),
        "size": size,
        "flags": 0,
        "char_flags": 0,
        "font": "Helvetica",
        "alpha": 255,
        "chars": chars,
    }
    if asc is not None:
        span["ascender"], span["descender"] = asc, desc
    return span


def _text_block(text, x0, y0, x1, y1, size=9.0):
    span = _span(text, x0, y0, x1, y1, size=size)
    line = {"bbox": (x0, y0, x1, y1), "dir": (1.0, 0.0), "spans": [span]}
    return {"type": 0, "bbox": (x0, y0, x1, y1), "lines": [line]}


def _lines(texts, x0, x1, y0, height=9.0, pitch=10.0):
    """A cell wrapped to several lines, `pitch` apart (normal leading: the
    line boxes nearly touch)."""
    return [
        _text_block(t, x0, y0 + k * pitch, x1, y0 + k * pitch + height)
        for k, t in enumerate(texts)
    ]


def _new(old, new):
    return sorted(set(new) - set(old))


# ---------------------------------------------------------------- rows


def test_fused_row_is_split_between_its_two_rows():
    # The model drew one row [0,40]; two rows of two cells sit inside it,
    # 11pt apart (cell padding), not one line pitch apart.
    blocks = [
        _text_block("R0C0", 5, 5, 45, 14),
        _text_block("R0C1", 55, 5, 95, 14),
        _text_block("R1C0", 5, 25, 45, 34),
        _text_block("R1C1", 55, 25, 95, 34),
    ]
    h, v = [0, 40], [0, 50, 100]
    new_h, new_v = repair_grid_gaps(blocks, h, v)
    assert _new(h, new_h) == [19.5]
    assert new_v == v


def test_legitimately_wrapped_cells_stay_one_row():
    # Two columns each wrap to two lines at the same offsets (one line
    # pitch apart) and a third column has one line at the top. This is one
    # row; the two wrapped cells are not two corroborating rows.
    blocks = (
        _lines(["Long label that", "wraps once"], 5, 45, 5)
        + _lines(["Another wrapped", "description"], 55, 95, 5)
        + [_text_block("42", 105, 5, 145, 14)]
    )
    h, v = [0, 30], [0, 50, 100, 150]
    assert repair_grid_gaps(blocks, h, v) == (h, v)


def test_one_cell_with_widely_spaced_lines_is_not_enough():
    # Only one cell has text on both sides of the whitespace: one strip is
    # no corroboration, however wide the gap.
    blocks = [
        _text_block("top", 5, 5, 45, 14),
        _text_block("bottom", 5, 30, 45, 39),
        _text_block("value", 55, 5, 95, 14),
    ]
    h, v = [0, 45], [0, 50, 100]
    assert repair_grid_gaps(blocks, h, v) == (h, v)


def test_boundary_never_cuts_a_single_line_cell():
    # Two header cells wrap to two lines (wide apart, so they would
    # corroborate a boundary), but the middle header cell is one line
    # sitting exactly across that whitespace.
    blocks = [
        _text_block("Unit price", 5, 2, 45, 11),
        _text_block("(EUR)", 5, 22, 45, 31),
        _text_block("Category of expense", 55, 12, 95, 21),
        _text_block("Total eligible", 105, 2, 145, 11),
        _text_block("(EUR)", 105, 22, 145, 31),
    ]
    h, v = [0, 35], [0, 50, 100, 150]
    assert repair_grid_gaps(blocks, h, v) == (h, v)


def test_boundary_never_cuts_a_multi_line_cell():
    # The midpoint between the two short rows (y=24.5) falls inside a
    # wrapped paragraph in the third column; the boundary must go into the
    # whitespace that is free in every cell (y 14..20), not through the
    # paragraph.
    blocks = (
        [
            _text_block("0,00", 5, 5, 45, 14),
            _text_block("0,00", 55, 5, 95, 14),
            _text_block("Item", 5, 35, 45, 44),
            _text_block("100,00", 55, 35, 95, 44),
            _text_block("0,00", 105, 5, 145, 14),
        ]
        + _lines(["Paragraph one", "two", "three", "four"], 105, 145, 20)
    )
    h, v = [0, 60], [0, 50, 100, 150]
    new_h, _ = repair_grid_gaps(blocks, h, v)
    assert _new(h, new_h) == [17.0]


def test_text_is_never_split_or_duplicated_by_inserted_boundaries():
    blocks = (
        [
            _text_block("0,00", 5, 5, 45, 14),
            _text_block("0,00", 55, 5, 95, 14),
            _text_block("Item", 5, 35, 45, 44),
            _text_block("100,00", 55, 35, 95, 44),
        ]
        + _lines(["Paragraph one", "two", "three", "four"], 105, 145, 20)
    )
    h, v = [0, 60], [0, 50, 100, 150]
    new_h, new_v = repair_grid_gaps(blocks, h, v)
    old_text, new_text = TableText(blocks, h, v), TableText(blocks, new_h, new_v)
    count = lambda t: sum(len(chars) for chars in t._chars.values())
    assert count(new_text) == count(old_text)
    # every line's characters land in a single cell of the repaired grid
    for block in blocks:
        for line in block["lines"]:
            cells = {
                new_text._cell_of(
                    (c["bbox"][0] + c["bbox"][2]) / 2, (c["bbox"][1] + c["bbox"][3]) / 2
                )
                for span in line["spans"]
                for c in span["chars"]
            }
            assert len(cells) == 1


def test_glyph_tight_boxes_of_wrapped_lines_do_not_look_like_rows():
    # TEXT_ACCURATE_BBOXES makes glyph boxes tight around the ink, so two
    # wrapped lines' boxes are several points apart. The font box (baseline
    # + ascender/descender) still shows normal leading, so nothing splits.
    def tight_line(text, x0, x1, baseline):
        span = _span(text, x0, baseline - 5, x1, baseline, origin=baseline, asc=0.9, desc=-0.2)
        line = {"bbox": span["bbox"], "dir": (1.0, 0.0), "spans": [span]}
        return {"type": 0, "bbox": span["bbox"], "lines": [line]}

    blocks = [
        tight_line("wrapped", 5, 45, 12),
        tight_line("lines", 5, 45, 22),
        tight_line("also", 55, 95, 12),
        tight_line("wrapped", 55, 95, 22),
    ]
    h, v = [0, 30], [0, 50, 100]
    assert repair_grid_gaps(blocks, h, v) == (h, v)


# ------------------------------------------------------------- columns


def _currency_column(header):
    """An 'Amount' column: left-aligned currency symbol, right-aligned value,
    one grid row per value (so three rows would corroborate a split)."""
    blocks = [header]
    for k, amount in enumerate(["1,234.00", "56.70", "890.12"]):
        y = 20 + k * 15
        blocks.append(_text_block("$", 5, y, 10, y + 9))
        blocks.append(_text_block(amount, 60, y, 95, y + 9))
    return blocks


def test_currency_symbol_and_amount_stay_one_column_under_a_centred_header():
    blocks = _currency_column(_text_block("Amount", 30, 2, 70, 11))
    h, v = [0, 15, 32, 47, 65], [0, 100]
    assert repair_grid_gaps(blocks, h, v) == (h, v)


def test_currency_symbol_and_amount_stay_one_column_under_a_right_header():
    # The header does not cross the whitespace, but it has text on one side
    # only: the column has one header, so it is one column.
    blocks = _currency_column(_text_block("Amount", 65, 2, 95, 11))
    h, v = [0, 15, 32, 47, 65], [0, 100]
    assert repair_grid_gaps(blocks, h, v) == (h, v)


def test_fused_column_with_two_headers_is_split():
    blocks = [
        _text_block("Eligible", 5, 2, 40, 11),
        _text_block("Claimed", 60, 2, 95, 11),
        _text_block("0,00", 20, 20, 40, 29),
        _text_block("0,00", 75, 20, 95, 29),
        _text_block("0,00", 20, 35, 40, 44),
        _text_block("0,00", 75, 35, 95, 44),
    ]
    h, v = [0, 15, 32, 50], [0, 100]
    new_h, new_v = repair_grid_gaps(blocks, h, v)
    assert new_h == h
    assert _new(v, new_v) == [50.0]


def test_columns_are_not_repaired_when_disabled():
    blocks = [
        _text_block("Eligible", 5, 2, 40, 11),
        _text_block("Claimed", 60, 2, 95, 11),
        _text_block("0,00", 20, 20, 40, 29),
        _text_block("0,00", 75, 20, 95, 29),
    ]
    h, v = [0, 15, 32], [0, 100]
    assert repair_grid_gaps(blocks, h, v, columns=False) == (h, v)


# --------------------------------------------------------------- misc


def test_gap_boundaries_needs_the_first_strip_only_when_asked():
    strips = [[(0, 10)], [(0, 10), (30, 40)], [(0, 10), (30, 40)]]
    assert gap_boundaries(strips, 5) == [20.0]
    assert gap_boundaries(strips, 5, require_first_strip=True) == []


def test_degenerate_inputs_are_returned_unchanged():
    assert repair_grid_gaps([], [0, 10], [0, 10]) == ([0, 10], [0, 10])
    assert repair_grid_gaps(None, [0, 10], [0, 10]) == ([0, 10], [0, 10])
    assert repair_grid_gaps([_text_block("x", 1, 1, 5, 5)], [0], [0, 10]) == ([0], [0, 10])


def test_text_outside_the_table_is_ignored():
    blocks = [
        _text_block("above", 5, -30, 45, -21),
        _text_block("R0C0", 5, 5, 45, 14),
        _text_block("R0C1", 55, 5, 95, 14),
    ]
    h, v = [0, 20], [0, 50, 100]
    assert repair_grid_gaps(blocks, h, v) == (h, v)
