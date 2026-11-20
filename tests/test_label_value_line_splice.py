"""Regression test for the "label/value splice" bug: when the underlying
PDF's native block detection fuses a wrapped label (e.g. a 2-line quoted
term in a glossary/definition-list row) together with its neighbouring
short value cell into a single block -- because they partially share a
y-range -- get_raw_lines()'s span-to-line clustering used to interleave
them purely by y-position, splicing the value sentence into the middle of
the label and stranding the label's second line afterwards.

Confirmed on a real-world document (Slovak legal-contract "Definície"
glossary annex): a row whose label wraps to 2 lines and whose 1-line value
sits beside it got rendered as:

    "Term Label" + value sentence + "Continuation"

instead of the correct "Term Label" + "Continuation" (one label), then the
value sentence.

The fix is two separate opt-in `get_raw_lines()` flags:
`require_x_continuity=True` stops the value from being joined into the
label's line, and `reorder_columns=True` emits the block's side-by-side
columns one after the other so the label's two lines stay adjacent. Both
are purely geometric -- no text-content, font-weight, or punctuation
dependence -- so it also covers this same failure mode for any language or
document type with a similar label/value layout, not just this one
document. It is off by default because some callers (e.g.
`get_text_lines(ocr=True)`'s rudimentary table reconstruction) intentionally
rely on same-row, different-x spans being joined into one line.
"""

import pymupdf
import pytest
from pymupdf4llm.helpers.get_text_lines import get_raw_lines


def _span(text, x0, y0, x1, y1, size=9.0):
    return {
        "text": text,
        "bbox": (x0, y0, x1, y1),
        "size": size,
        "flags": 0,
        "char_flags": 0,
        "font": "Helvetica",
        "alpha": 255,
    }


def _line(spans, x0, y0, x1, y1):
    return {"bbox": (x0, y0, x1, y1), "dir": (1.0, 0.0), "spans": spans}


def _fused_label_value_block():
    """One block whose 3 lines mirror the real bug's geometry: a 2-line
    label at x0=131.5 (rows ~1 and ~3) and a 1-line value at x0=230.7
    (row ~1, sharing a y-band with the label's FIRST line only)."""
    label_l1 = _line([_span("Term Label ", 131.5, 319.8, 192.1, 327.8)], 131.5, 319.8, 192.1, 327.8)
    label_l2 = _line([_span("Continuation", 131.5, 332.9, 173.5, 340.2)], 131.5, 332.9, 173.5, 340.2)
    value_l = _line(
        [_span("Its short value clause here;", 230.7, 319.8, 410.2, 328.7)],
        230.7, 319.8, 410.2, 328.7,
    )
    bbox = (131.5, 319.8, 410.2, 340.2)
    return {"type": 0, "bbox": bbox, "lines": [label_l1, label_l2, value_l]}


def _single_column_block():
    """A genuine multi-line paragraph: all lines left-aligned at the same
    x0 -- must be completely unaffected by require_x_continuity."""
    l1 = _line([_span("Wrapped paragraph line one ", 92.0, 400.0, 300.0, 410.0)], 92.0, 400.0, 300.0, 410.0)
    l2 = _line([_span("line two continues here ", 92.0, 412.0, 280.0, 422.0)], 92.0, 412.0, 280.0, 422.0)
    l3 = _line([_span("and line three ends it.", 92.0, 424.0, 250.0, 434.0)], 92.0, 424.0, 250.0, 434.0)
    bbox = (92.0, 400.0, 300.0, 434.0)
    return {"type": 0, "bbox": bbox, "lines": [l1, l2, l3]}


def _extract_texts(nlines):
    return ["".join(s["text"] for s in spans) for _, spans in nlines]


def test_require_x_continuity_false_reproduces_the_original_splice():
    """Documents the bug: default behaviour (flag off) still splices."""
    blocks = [_fused_label_value_block()]
    clip = pymupdf.Rect(0, 0, 600, 800)
    nlines = get_raw_lines(blocks=blocks, clip=clip, require_x_continuity=False)
    texts = _extract_texts(nlines)
    # The value sentence gets fused onto the same line as the label's
    # first line (same y-band), splicing it into the middle.
    assert any("Term Label" in t and "value clause" in t for t in texts)


def test_require_x_continuity_alone_does_not_reorder_lines():
    """The continuity check only decides which spans form a line; it must
    not change line order. Without reorder_columns the value is a separate
    line but still sits between the label's two lines (y order)."""
    blocks = [_fused_label_value_block()]
    clip = pymupdf.Rect(0, 0, 600, 800)
    texts = _extract_texts(get_raw_lines(blocks=blocks, clip=clip, require_x_continuity=True))
    assert texts == ["Term Label ", "Its short value clause here;", "Continuation"]


def test_require_x_continuity_and_reorder_columns_fix_the_splice():
    blocks = [_fused_label_value_block()]
    clip = pymupdf.Rect(0, 0, 600, 800)
    nlines = get_raw_lines(
        blocks=blocks, clip=clip, require_x_continuity=True, reorder_columns=True
    )
    texts = _extract_texts(nlines)

    label_idx = next(i for i, t in enumerate(texts) if "Term Label" in t)
    cont_idx = next(i for i, t in enumerate(texts) if "Continuation" in t)
    value_idx = next(i for i, t in enumerate(texts) if "value clause" in t)

    # The label's own text must never be spliced with the value.
    assert "value clause" not in texts[label_idx]
    assert "value clause" not in texts[cont_idx]
    # Label line 1 must be immediately followed by label line 2 (its own
    # continuation), and the value must come after the complete label --
    # column-major within the block, not interleaved by y-position.
    assert cont_idx == label_idx + 1
    assert value_idx > cont_idx


def test_single_column_multiline_block_is_unaffected():
    """A genuine wrapped paragraph (all lines at the same x0) must produce
    the exact same result whether or not require_x_continuity is set."""
    clip = pymupdf.Rect(0, 0, 600, 800)
    without = _extract_texts(
        get_raw_lines(blocks=[_single_column_block()], clip=clip, require_x_continuity=False)
    )
    withx = _extract_texts(
        get_raw_lines(
            blocks=[_single_column_block()], clip=clip,
            require_x_continuity=True, reorder_columns=True,
        )
    )
    assert without == withx == [
        "Wrapped paragraph line one ",
        "line two continues here ",
        "and line three ends it.",
    ]


def _indented_first_line_block():
    """A genuine wrapped paragraph whose first line is indented (e.g. a
    typical prose paragraph opening). All lines' x-ranges overlap heavily
    (they share the same wrap width), even though the first line's x0 sits
    well to the right of the other lines' x0 -- this must never be read as
    a second column."""
    l1 = _line([_span("    Indented opening line of the ", 120.0, 400.0, 300.0, 410.0)], 120.0, 400.0, 300.0, 410.0)
    l2 = _line([_span("paragraph continues here without ", 92.0, 412.0, 300.0, 422.0)], 92.0, 412.0, 300.0, 422.0)
    l3 = _line([_span("any indentation on wrapped lines.", 92.0, 424.0, 290.0, 434.0)], 92.0, 424.0, 290.0, 434.0)
    bbox = (92.0, 400.0, 300.0, 434.0)
    return {"type": 0, "bbox": bbox, "lines": [l1, l2, l3]}


def _centred_paragraph_block():
    """A genuine centred paragraph: each line's x0 (and x1) varies by
    design because the line is centred around a shared midpoint, not
    because it belongs to a different column. Ranges overlap around that
    shared centre, so this must also never be read as multiple columns."""
    l1 = _line([_span("Short title", 180.0, 400.0, 220.0, 410.0)], 180.0, 400.0, 220.0, 410.0)
    l2 = _line([_span("A rather longer centred subtitle line", 100.0, 412.0, 300.0, 422.0)], 100.0, 412.0, 300.0, 422.0)
    l3 = _line([_span("Mid-length line", 140.0, 424.0, 260.0, 434.0)], 140.0, 424.0, 260.0, 434.0)
    bbox = (100.0, 400.0, 300.0, 434.0)
    return {"type": 0, "bbox": bbox, "lines": [l1, l2, l3]}


def test_indented_first_line_paragraph_retains_reading_order():
    """Regression test for the reviewer-flagged false positive: an
    indented first line must not be reordered after the rest of the
    paragraph just because its x0 differs -- its x-range still overlaps
    the other lines', so this is one column, not two."""
    clip = pymupdf.Rect(0, 0, 600, 800)
    texts = _extract_texts(
        get_raw_lines(
            blocks=[_indented_first_line_block()], clip=clip,
            require_x_continuity=True, reorder_columns=True,
        )
    )
    assert texts == [
        "    Indented opening line of the ",
        "paragraph continues here without ",
        "any indentation on wrapped lines.",
    ]


def test_centred_paragraph_retains_reading_order():
    """Regression test for the reviewer-flagged false positive: centred
    lines legitimately vary in x0/x1 line to line; their ranges overlap
    around the shared centre, so this must stay one column, top to
    bottom, not be split and reordered."""
    clip = pymupdf.Rect(0, 0, 600, 800)
    texts = _extract_texts(
        get_raw_lines(
            blocks=[_centred_paragraph_block()], clip=clip,
            require_x_continuity=True, reorder_columns=True,
        )
    )
    assert texts == [
        "Short title",
        "A rather longer centred subtitle line",
        "Mid-length line",
    ]


def _staggered_lines_block():
    """Lines whose x-ranges are disjoint but which never share a row: a
    right-aligned date line above left-aligned text, then another
    right-aligned line (e.g. a letter heading or alternating dialogue).
    Disjoint x-ranges alone are not columns -- this must read top to
    bottom."""
    l1 = _line([_span("4 October 2026", 420.0, 400.0, 500.0, 410.0)], 420.0, 400.0, 500.0, 410.0)
    l2 = _line([_span("Dear Sir or Madam,", 92.0, 412.0, 200.0, 422.0)], 92.0, 412.0, 200.0, 422.0)
    l3 = _line([_span("Ref. 12/345", 430.0, 424.0, 500.0, 434.0)], 430.0, 424.0, 500.0, 434.0)
    bbox = (92.0, 400.0, 500.0, 434.0)
    return {"type": 0, "bbox": bbox, "lines": [l1, l2, l3]}


def test_staggered_lines_without_a_shared_row_retain_reading_order():
    """Reordering needs side-by-side evidence (a shared row), not just
    disjoint x-ranges; otherwise the left line would be moved first."""
    clip = pymupdf.Rect(0, 0, 600, 800)
    texts = _extract_texts(
        get_raw_lines(
            blocks=[_staggered_lines_block()], clip=clip,
            require_x_continuity=True, reorder_columns=True,
        )
    )
    assert texts == ["4 October 2026", "Dear Sir or Madam,", "Ref. 12/345"]


@pytest.fixture
def legacy_layout_mode():
    """Run with the legacy (non-layout) path and restore the previous
    process-wide layout mode on teardown, also when the test fails."""
    import pymupdf4llm

    prev_use_layout = pymupdf4llm._use_layout
    pymupdf4llm.use_layout(False)
    yield
    pymupdf4llm.use_layout(prev_use_layout)


def test_end_to_end_glossary_page_via_to_markdown(legacy_layout_mode):
    """End-to-end sanity check through the public to_markdown() API on a
    synthetic multi-row glossary-style page. Uses insert_text-generated
    geometry (this does not reliably reproduce MuPDF's native block-fusion
    trigger -- see module docstring's unit tests above for that -- but
    confirms the opt-in flag doesn't break normal extraction end-to-end)."""
    import pymupdf4llm

    doc = pymupdf.open()
    page = doc.new_page()
    y = 140.0
    for i in range(4):
        page.insert_text((131.5, y), "Term Label", fontsize=11)
        page.insert_text((131.5, y + 13), f"Row {i}", fontsize=11)
        page.insert_text((230.7, y), f"value clause number {i} here", fontsize=11)
        y += 32.0

    md = pymupdf4llm.to_markdown(doc)
    for i in range(4):
        assert f"Row {i}" in md
        assert f"value clause number {i}" in md
