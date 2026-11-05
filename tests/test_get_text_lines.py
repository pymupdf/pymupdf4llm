"""Regression tests for invisible text handling in the plain-text helper."""

import pymupdf
import pytest

from pymupdf4llm.helpers.get_text_lines import get_text_lines


@pytest.mark.parametrize("with_textpage", [False, True])
@pytest.mark.parametrize("with_visible_text", [False, True])
def test_get_text_lines_ignore_invisible(with_textpage, with_visible_text):
    with pymupdf.open() as doc:
        page = doc.new_page()
        if with_visible_text:
            page.insert_text((72, 72), "Visible text")
        # Rendering mode 3 reproduces an invisible OCR text layer.
        page.insert_text((72, 100), "OCR text", render_mode=3)
        kwargs = {"textpage": page.get_textpage()} if with_textpage else {}
        visible_text = "\nVisible text\n\n" if with_visible_text else ""

        assert get_text_lines(page, **kwargs) == visible_text
        assert get_text_lines(page, ignore_invisible=True, **kwargs) == visible_text

        expected = (
            "\nVisible text\n\nOCR text\n\n"
            if with_visible_text else "\nOCR text\n\n"
        )
        assert get_text_lines(page, ignore_invisible=False, **kwargs) == expected


def test_get_text_lines_ignore_invisible_respects_clip():
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), "Inside clip", render_mode=3)
        page.insert_text((72, 150), "Outside clip", render_mode=3)

        text = get_text_lines(
            page, clip=(0, 0, 300, 100), ignore_invisible=False
        )

        assert text == "\nInside clip\n\n"
