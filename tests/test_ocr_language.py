"""Parameter "ocr_language" must reach the recognition engine.

Most tests use stubs, so they need neither rapidocr, Tesseract nor a network
connection. The end-to-end test downloads the RapidOCR Korean model and only
runs if PYMUPDF4LLM_TEST_RAPIDOCR_DOWNLOAD=1.
"""

import importlib.util
import os
import sys
import types

import pymupdf
import pytest

import pymupdf4llm
import pymupdf4llm.ocr
from pymupdf4llm.ocr import exec_ocr_interface

BOX = [[10, 10], [100, 10], [100, 40], [10, 40]]
DEFAULT_PARAMS = {
    "Global.use_det": True,
    "Global.use_cls": True,
    "Global.use_rec": True,
}
KOREAN_PARAMS = {
    **DEFAULT_PARAMS,
    "Global.use_cls": False,
    "Rec.lang_type": "korean",
    "Rec.ocr_version": "PP-OCRv5",
    "Rec.model_type": "mobile",
}


def _page_with_graphics():
    """A page without text whose OCR image is not empty."""
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=120)
    page.draw_rect((20, 40, 200, 70), color=(0, 0, 0), fill=(0, 0, 0))
    return doc, page


@pytest.fixture
def backend(monkeypatch):
    """A private copy of rapidocr_universal_backend with a fake rapidocr.

    fake.created lists the "params" of every engine which was created.
    """
    fake = types.ModuleType("rapidocr")
    fake.created = []

    class RapidOCR:
        def __init__(self, params=None):
            fake.created.append(params)

        def __call__(self, img):
            return types.SimpleNamespace(boxes=[BOX], txts=["text"], scores=[0.9])

    fake.RapidOCR = RapidOCR
    fake.LangRec = fake.OCRVersion = fake.ModelType = str
    monkeypatch.setitem(sys.modules, "rapidocr", fake)
    path = os.path.join(
        os.path.dirname(pymupdf4llm.ocr.__file__), "rapidocr_universal_backend.py"
    )
    spec = importlib.util.spec_from_file_location("_test_rapidocr_backend", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, fake


def test_language_selects_the_recognition_model(backend):
    module, fake = backend
    # Languages which the default model reads use the one default engine.
    for language in (None, "", "eng", "jpn", "deu+fra"):
        assert module.full_ocr(None, language=language) == [(BOX, "text", 0.9)]
    assert module.full_ocr(None) == [(BOX, "text", 0.9)]
    assert fake.created == [DEFAULT_PARAMS]
    # One more engine for Korean, whatever the position of its code.
    for language in ("kor", "eng+kor", "kor+eng"):
        assert module.full_ocr(None, language=language) == [(BOX, "text", 0.9)]
    assert fake.created == [DEFAULT_PARAMS, KOREAN_PARAMS]
    # Detection does not depend on the language.
    assert module.det_only(None) == [(BOX, 0.9)]
    assert len(fake.created) == 3
    module.reset_engines()
    module.full_ocr(None, language="kor")
    assert fake.created[3:] == [KOREAN_PARAMS]


def test_rapidocr_plugin_passes_language(monkeypatch):
    """Only the rapidocr 3.x backend gets the language."""
    from pymupdf4llm.ocr import rapidocr_api

    doc, page = _page_with_graphics()
    calls = []

    def full_ocr(img, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(rapidocr_api, "full_ocr", full_ocr)
    monkeypatch.setattr(rapidocr_api, "rapidocr_backend", "rapidocr")
    rapidocr_api.exec_ocr(page, dpi=72, language="eng+kor")
    monkeypatch.setattr(rapidocr_api, "rapidocr_backend", "rapidocr_onnxruntime")
    rapidocr_api.exec_ocr(page, dpi=72, language="eng+kor")
    assert calls == [{"language": "eng+kor"}, {}]


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


KOREAN_LINES = [
    "한국어 영수증 합계 금액",
    "서울특별시 강남구 테헤란로 123",
    "합계 금액 12,500원 부가세 포함",
    "Total amount 12,500 KRW",
]


def _hangul(text):
    return sum(1 for c in text if 0xAC00 <= ord(c) <= 0xD7A3)


@pytest.mark.skipif(
    os.environ.get("PYMUPDF4LLM_TEST_RAPIDOCR_DOWNLOAD") != "1",
    reason="downloads a RapidOCR model, set PYMUPDF4LLM_TEST_RAPIDOCR_DOWNLOAD=1",
)
def test_korean_end_to_end():
    pytest.importorskip("onnxruntime")
    from pymupdf4llm.ocr import rapidocr_api

    if rapidocr_api.rapidocr_backend != "rapidocr":
        pytest.skip("needs rapidocr 3.x")
    if not pymupdf4llm._use_layout:
        pytest.skip("OCR requires PyMuPDF Layout")

    # An image of Korean text, without a text layer.
    src = pymupdf.open()
    src_page = src.new_page(width=595, height=400)
    writer = pymupdf.TextWriter(src_page.rect)
    font = pymupdf.Font("korea")
    for i, line in enumerate(KOREAN_LINES):
        writer.append((50, 80 + 50 * i), line, font=font, fontsize=22)
    writer.write_text(src_page)
    pix = src_page.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=400)
    page.insert_image(page.rect, pixmap=pix)

    # Use the RapidOCR plugin explicitly: with Tesseract installed,
    # the default selection is RapidOCR detection + Tesseract recognition.
    md = pymupdf4llm.to_markdown(
        doc,
        force_ocr=True,
        ocr_language="eng+kor",
        ocr_function=rapidocr_api.exec_ocr,
    )
    # The default model reads no Hangul at all. With the text line direction
    # classifier, the third line was turned upside down and lost.
    assert _hangul(md) >= 0.9 * _hangul("".join(KOREAN_LINES))
    assert "서울특별시" in md
    assert "부가세" in md
    assert "Total amount" in md
