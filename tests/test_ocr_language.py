"""Parameter "ocr_language" must reach the recognition engine.

The tests use stubs, so they need neither RapidOCR nor Tesseract.
"""

import pymupdf

from pymupdf4llm.ocr import exec_ocr_interface

BOX = [[10, 10], [100, 10], [100, 40], [10, 40]]


def _page_with_graphics():
    """A page without text whose OCR image is not empty."""
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=120)
    page.draw_rect((20, 40, 200, 70), color=(0, 0, 0), fill=(0, 0, 0))
    return doc, page


def test_tesseract_recognition_gets_language(monkeypatch):
    """RapidOCR detection + Tesseract recognition uses the requested language."""
    doc, page = _page_with_graphics()
    monkeypatch.setattr(exec_ocr_interface, "TESSDATA", "tessdata")
    languages = []

    def get_text(pixmap, irect, language="eng"):
        languages.append(language)
        return "text"

    monkeypatch.setattr(exec_ocr_interface, "get_text", get_text)
    exec_ocr_interface.exec_ocr_detection(
        page, lambda img: [(BOX, 1.0)], dpi=72, language="eng+kor"
    )
    assert languages == ["eng+kor"]
    assert "text" in page.get_text()
