import pytest
import pymupdf
import pymupdf4llm


@pytest.mark.parametrize("layout", [False, True])
@pytest.mark.parametrize("text", ["Highlighted", "Highlighted words"])
def test_partial_highlight_preserves_word(layout, text):
    previous_layout = pymupdf4llm._use_layout
    pymupdf4llm.use_layout(layout)
    try:
        with pymupdf.open() as doc:
            page = doc.new_page(width=400, height=100)
            page.draw_rect(pymupdf.Rect(18, 38, 62, 58), color=None, fill=(1, 1, 0))
            page.insert_text((20, 53), text, fontname="helv", fontsize=14)
            assert page.get_text().strip() == text
            result = pymupdf4llm.to_markdown(doc).strip()
            assert result == "<mark>Highlig</mark>" + text[len("Highlig"):]
    finally:
        pymupdf4llm.use_layout(previous_layout)


@pytest.mark.parametrize(
    "left,right,gap,line,block,same_style,expected",
    [
        ("High", "lighted", 0, 0, 0, False, "Highlighted"),
        ("High", "lighted", 0, 0, 0, True, "High lighted"),
        ("High ", "lighted", 0, 0, 0, False, "High lighted"),
        ("High", " lighted", 0, 0, 0, False, "High lighted"),
        ("High", "lighted", 3, 0, 0, False, "High lighted"),
        ("High", "lighted", 0, 1, 0, False, "High lighted"),
        ("High", "lighted", 0, 0, 1, False, "High lighted"),
        ("High", "lighted", 0.00001, 0, 0, False, "Highlighted"),
    ],
)
def test_styled_spans_preserve_real_separators(left, right, gap, line, block, same_style, expected):
    from pymupdf4llm.helpers.document_layout import get_styled_text

    highlight = pymupdf.mupdf.FZ_STEXT_HIGHLIGHT
    spans = [
        {"text": left, "bbox": (0, 0, 20, 10), "flags": 0,
         "char_flags": highlight, "block": 0, "line": 0},
        {"text": right, "bbox": (20 + gap, 0, 50, 10), "flags": 0,
         "char_flags": highlight if same_style else 0, "block": block, "line": line},
    ]
    styled, _ = get_styled_text(spans)
    assert styled.replace("<mark>", "").replace("</mark>", "").strip() == expected
    assert styled.count("<mark>") == 1
    assert styled.count("</mark>") == 1


def test_legacy_header_does_not_split_highlighted_word():
    previous_layout = pymupdf4llm._use_layout
    pymupdf4llm.use_layout(False)
    try:
        with pymupdf.open() as doc:
            page = doc.new_page(width=400, height=100)
            page.draw_rect(pymupdf.Rect(18, 38, 62, 58), color=None, fill=(1, 1, 0))
            page.insert_text((20, 53), "Highlighted", fontname="helv", fontsize=14)
            result = pymupdf4llm.to_markdown(doc, hdr_info=lambda span, page=None: "# ")
            assert result.strip() == "# Highlighted"
    finally:
        pymupdf4llm.use_layout(previous_layout)
