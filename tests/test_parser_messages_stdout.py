import io
import os

import pymupdf
import pymupdf4llm  # noqa: F401
from pymupdf4llm.helpers import document_layout

PDF = os.path.normpath(f"{__file__}/../test_sce_150_1.pdf")


def test_parser_messages_follow_set_messages(monkeypatch, capsys):
    # With messages redirected away from stdout, nothing of the parser
    # messages block may still be printed to stdout.
    redirected = io.StringIO()
    monkeypatch.setattr(pymupdf, "_g_out_message", redirected)
    with pymupdf.open(PDF) as doc:
        print("test message", file=document_layout.INFO_MESSAGES)
        md = doc.to_markdown(pages=[0], use_ocr=False)
    assert md
    assert "test message" in redirected.getvalue()
    assert capsys.readouterr().out == ""
