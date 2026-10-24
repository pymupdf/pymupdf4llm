"""Tests for include_clipped_text: recovering table cell text cut off by a clip path."""
import pymupdf
import pymupdf4llm  # noqa: F401
from pymupdf4llm.helpers.document_layout import get_clipped_texts, parse_document

COLUMNS = (20, 170, 320, 500)
ROWS = [
    ["Name", "Address", "Phone"],
    ["Alex Example", "ADDRESS", "+44 1234 567890"],
    ["Sam Sample", "7 Example Road", "+44 9876 543210"],
    ["Kim Person", "12 High Street", "+44 5555 123456"],
    ["Lee Someone", "3 Market Lane", "+44 7777 654321"],
]


def table_pdf(rows, clips=None, offsets=None):
    """Create a ruled 3-column table.

    clips: {(row, col): width} clips that cell's text to the given width.
    offsets: {(row, col): x} overrides the x position of that cell's text.
    """
    clips, offsets = clips or {}, offsets or {}
    doc = pymupdf.open()
    page = doc.new_page(width=520, height=60 + 30 * len(rows))
    top = page.rect.height - 30
    bottom = top - 30 * len(rows)
    ops = [f"{COLUMNS[0]} {top - 30 * i} m {COLUMNS[-1]} {top - 30 * i} l S"
           for i in range(len(rows) + 1)]
    ops += [f"{x} {top} m {x} {bottom} l S" for x in COLUMNS]
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            x = offsets.get((r, c), COLUMNS[c] + 5)
            y = top - 30 * r - 20
            clip = ""
            if (r, c) in clips:
                clip = f"{COLUMNS[c]} {y - 10} {clips[r, c]} 30 re W n "
            ops.append(f"q {clip}BT /helv 11 Tf 1 0 0 1 {x} {y} Tm ({text}) Tj ET Q")
    xref = doc.get_new_xref()
    doc.update_object(xref, "<<>>")
    doc.update_stream(xref, "\n".join(ops).encode())
    doc.xref_set_key(page.xref, "Contents", f"{xref} 0 R")
    doc.xref_set_key(
        page.xref,
        "Resources",
        "<</Font<</helv<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>>>",
    )
    return pymupdf.open("pdf", doc.tobytes())


def address_pdf(address, width):
    rows = [list(row) for row in ROWS]
    rows[1][1] = address
    return table_pdf(rows, clips={(1, 1): width})


def table_extract(doc, **kwargs):
    parsed = parse_document(doc, use_ocr=False, **kwargs)
    tables = [b.table for b in parsed.pages[0].boxes if b.boxclass == "table"]
    assert len(tables) == 1
    return tables[0]["extract"]


def test_clipped_cell_text_is_recovered():
    with address_pdf("PO Box 7140, Naples", 80) as doc:
        assert table_extract(doc)[1] == ["Alex Example", "PO Box 7140, N", "+44 1234 567890"]
        assert table_extract(doc, include_clipped_text=True)[1] == [
            "Alex Example", "PO Box 7140, Naples", "+44 1234 567890",
        ]
        md = doc.to_markdown(use_ocr=False, include_clipped_text=True)
        assert md.count("PO Box 7140, Naples") == 1
        assert "|Alex Example|PO Box 7140, Naples|+44 1234 567890|" in md


def test_overflow_does_not_leak_into_neighbor_cell():
    # The full address extends into the Phone column. In the clipped
    # extraction, MuPDF merges the visible address and the phone number
    # into one span; the recovered text must still stay in the Address cell.
    with address_pdf("PO Box 7140, Naples, Campania", 150) as doc:
        before = table_extract(doc)
        after = table_extract(doc, include_clipped_text=True)
        assert before[1][1] == "PO Box 7140, Naples, Campa"
        assert after[1] == ["Alex Example", "PO Box 7140, Naples, Campania", "+44 1234 567890"]
        assert after[2:] == before[2:]


def test_ambiguous_owner_is_left_unchanged():
    # The clipped text continues exactly where "WXYZ" starts in the next
    # cell, so the unclipped extraction joins both into one span. Its
    # visible parts are in two cells: keep the default output.
    hidden = "ABCDEFGHIJKLMNOPQRSTUV"
    rows = [list(row) for row in ROWS]
    rows[1][:2] = [hidden, "WXYZ"]
    x = COLUMNS[0] + 5 + pymupdf.get_text_length(hidden, "helv", 11)
    with table_pdf(rows, clips={(1, 0): 28}, offsets={(1, 1): x}) as doc:
        assert table_extract(doc, include_clipped_text=True) == table_extract(doc)


def test_default_output_is_unchanged():
    with address_pdf("PO Box 7140, Naples", 80) as doc:
        default = doc.to_markdown(use_ocr=False)
        assert "|Alex Example|PO Box 7140, N|+44 1234 567890|" in default
        assert doc.to_markdown(use_ocr=False, include_clipped_text=False) == default
        doc.to_markdown(use_ocr=False, include_clipped_text=True)
        assert doc.to_markdown(use_ocr=False) == default


def test_glyph_shared_by_two_sources_is_not_recovered():
    # Visible "AB" matches stored "ABC", but another stored span has the same
    # "B" glyph, so the visible text cannot be assigned to one source.
    def blocks(*spans):
        return [{"type": 0, "lines": [{"spans": [
            {"font": "helv", "size": 11, "chars": [
                {"origin": (x + 5.0 * i, 10.0), "c": c} for i, c in enumerate(text)
            ]}
            for text, x in spans
        ]}]}]

    assert get_clipped_texts(blocks(("AB", 0)), blocks(("ABC", 0)))
    assert not get_clipped_texts(blocks(("AB", 0)), blocks(("ABC", 0), ("BX", 5)))
