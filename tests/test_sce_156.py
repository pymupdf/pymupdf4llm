import pymupdf4llm
from pathlib import Path
from pymupdf4llm import pymupdf

def test_sce_156_1():
    """Do not recommend OCR if Devanagari text exists on page."""
    filename = Path(__file__).parent / "test_sce_156.pdf"
    # Original file in sce issue 156
    # https://github.com/user-attachments/files/29286923/hindi.pdf
    expected = (
        (Path(__file__).parent / "test_sce_156.expected.md").read_bytes().decode()
    )
    expected = expected.replace("\r", "")  # For github windows.
    md = pymupdf4llm.to_markdown(
        filename,
        write_images=False,
        embed_images=False,
        header=False,
        footer=False,
        use_ocr=True,  # do not recommend OCR if Devanagari is present
    )
    assert md == expected


def test_sce_156_2():
    """Never EXECUTE OCR if Devanagari text exists on page."""
    filename = Path(__file__).parent / "test_sce_156.pdf"
    # Original file in sce issue 156
    # https://github.com/user-attachments/files/29286923/hindi.pdf
    expected = (
        (Path(__file__).parent / "test_sce_156.expected.md").read_bytes().decode()
    )
    expected = expected.replace("\r", "")  # For github windows.
    md = pymupdf4llm.to_markdown(
        filename,
        write_images=False,
        embed_images=False,
        header=False,
        footer=False,
        force_ocr=True,  # no OCR - even if forced
    )
    assert md == expected


def test_sce_156_3():
    """Never EXECUTE OCR if Devanagari text exists on page."""
    filename = Path(__file__).parent / "test_sce_156.pdf"
    # Original file in sce issue 156
    # https://github.com/user-attachments/files/29286923/hindi.pdf
    analyze_page = pymupdf4llm.helpers.utils.analyze_page
    doc = pymupdf.open(filename)
    page = doc[0]
    analysis = analyze_page(page)
    assert analysis["needs_ocr"] is False
