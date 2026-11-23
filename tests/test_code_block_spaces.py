import re

import pymupdf
import pymupdf4llm
from pymupdf4llm.helpers import document_layout
from pymupdf4llm.helpers.get_text_lines import get_raw_lines

CODE = [
    "def area(w, h):",
    "    return w * h",
]


def _make_document():
    """Page with mono-spaced text that has no space characters.

    Every word is written as a separate text object at its column. MuPDF
    then derives the spaces from the gaps between the words.
    """
    fontsize = 10
    char_width = pymupdf.get_text_length("x", fontname="cour", fontsize=fontsize)
    document = pymupdf.open()
    page = document.new_page()
    for lno, line in enumerate(CODE):
        for word in re.finditer(r"\S+", line):
            point = (72 + word.start() * char_width, 100 + lno * 12)
            page.insert_text(point, word.group(), fontname="cour", fontsize=fontsize)
    return document


def _textlines(page):
    textpage = page.get_textpage(flags=document_layout.FLAGS)
    return [
        {"bbox": line[0], "spans": line[1]}
        for line in get_raw_lines(textpage=textpage, clip=page.rect)
    ]


def test_code_block_keeps_spaces():
    # Spaces and indentation of a code block must survive when they only
    # exist as gaps between the spans.
    with _make_document() as document:
        textlines = _textlines(document[0])
        assert document_layout.is_monospaced(textlines)
        expected = "\n".join(CODE)
        assert document_layout.code_block_to_md(textlines) == f"```\n{expected}\n```\n\n"
        assert document_layout.code_block_to_text(textlines) == f"{expected}\n\n\n"


def test_code_block_keeps_spaces_to_markdown():
    with _make_document() as document:
        md = pymupdf4llm.to_markdown(document)
    expected = "\n".join(CODE)
    assert f"```\n{expected}\n```" in md, md


def test_code_block_adds_no_spaces():
    # Stored spaces stay as they are, touching spans get no space between them.
    fontsize = 10
    char_width = pymupdf.get_text_length("x", fontname="cour", fontsize=fontsize)
    with pymupdf.open() as document:
        page = document.new_page()
        x = 72
        for text, fontname in (("int ", "cour"), ("main", "cobo"), ("(void) {", "cour")):
            page.insert_text((x, 100), text, fontname=fontname, fontsize=fontsize)
            x += len(text) * char_width
        textlines = _textlines(page)
        assert len(textlines[0]["spans"]) == 3
        assert document_layout.code_block_to_md(textlines) == "```\nint main(void) {\n```\n\n"
