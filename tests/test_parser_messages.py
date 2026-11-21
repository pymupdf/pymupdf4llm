import os

import pymupdf
import pymupdf4llm  # noqa: F401
from pymupdf4llm.helpers import document_layout

PDF = os.path.normpath(f"{__file__}/../test_sce_150_1.pdf")


def test_parser_messages_are_not_repeated(monkeypatch):
    messages = []
    monkeypatch.setattr(pymupdf, "message", messages.append)
    with pymupdf.open(PDF) as doc:
        print("first call message", file=document_layout.INFO_MESSAGES)
        doc.to_markdown(pages=[0], use_ocr=False)
        assert "first call message" in messages
        messages.clear()
        doc.to_markdown(pages=[0], use_ocr=False)
        assert messages == []
