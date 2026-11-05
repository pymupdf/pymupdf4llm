"""Regression tests for missing-OCR-engine handling in parse_document().

Simulates an environment where no OCR engine (RapidOCR, Tesseract, ...)
is available and asserts that parse_document() surfaces the problem
instead of silently returning empty output for scanned documents.

No test PDF file is required: fixtures are generated in memory.
"""

import contextlib
import io

import pymupdf

from pymupdf4llm.helpers import document_layout as dl
from pymupdf4llm.ocr import OCRMode


def _make_pdf():
    """Create a small in-memory PDF instead of shipping a test file."""
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=300)
    page.insert_text((72, 100), "Hello OCR engine test")
    return doc


class _NoEngine:
    """Context manager: pretend that select_ocr_function() finds nothing."""

    def __enter__(self):
        self._saved = dl.select_ocr_function
        dl.select_ocr_function = lambda: None
        return self

    def __exit__(self, *exc):
        dl.select_ocr_function = self._saved


def test_missing_engine_prints_warning():
    """Default OCR mode without an engine must warn, not fail silently."""
    with _NoEngine():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            parsed = dl.parse_document(_make_pdf())
        assert "No OCR engine available" in buf.getvalue()
        # serialization must still work without crashing.
        assert isinstance(parsed.to_markdown(), str)


def test_missing_engine_force_ocr_raises():
    """force_ocr=True without an engine must raise ValueError."""
    with _NoEngine():
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                dl.parse_document(_make_pdf(), force_ocr=True)
        except ValueError as e:
            assert "no OCR engine" in str(e)
        else:
            raise AssertionError("expected ValueError was not raised")


def test_missing_engine_never_mode_no_warning():
    """use_ocr=NEVER never asked for OCR: no warning expected."""
    with _NoEngine():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dl.parse_document(_make_pdf(), use_ocr=OCRMode.NEVER)
        assert "No OCR engine available" not in buf.getvalue()


def test_info_messages_cleared_between_parses():
    """INFO_MESSAGES must be emptied after each parse_document() call.

    (truncate() only works after seek(0): truncating at the end of the
    buffer is a no-op and would leak stale messages into later parses.)
    """
    with _NoEngine():
        dl.INFO_MESSAGES.write("stale message")
        with contextlib.redirect_stdout(io.StringIO()):
            dl.parse_document(_make_pdf(), use_ocr=OCRMode.NEVER)
        assert dl.INFO_MESSAGES.getvalue() == ""


if __name__ == "__main__":
    # Allow running without pytest: python tests/test_ocr_engine_missing.py
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"{_name}: OK")
    print("all OK")
