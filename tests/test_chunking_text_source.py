"""W0-1 DoD: box_to_markdown mirrors ParsedDocument.to_markdown per box.

For every box class with a markdown rendering, the chunker's canonical text
(box_to_markdown) must equal the segment to_markdown emits for that box,
modulo whitespace normalization.  Excluded by design:

- picture/formula boxes without text (placeholder territory, D9)
- formula boxes with text (to_markdown drops the text; the chunker keeps it)
- inline image references (images travel in metadata, not chunk text)
"""

import os
import re

import pytest

from pymupdf4llm.helpers.document_layout import parse_document
from pymupdf4llm.helpers.chunking.text_source import (
    box_to_markdown,
    create_list_item_levels,
)

HERE = os.path.dirname(os.path.abspath(__file__))

_IMG_RE = re.compile(r"!\[\]\([^)]*\)")


def _norm(s):
    return re.sub(r"\s+", " ", s).strip()


def _iter_box_segments(doc):
    """Yield (page, box, box_idx, md_segment) using page_chunks offsets."""
    pages_md = doc.to_markdown(page_chunks=True)
    for page, chunk in zip(doc.pages, pages_md):
        text = chunk["text"]
        for pb, box in zip(chunk["page_boxes"], page.boxes):
            start, stop = pb["pos"]
            yield page, box, pb["index"], text[start:stop]


@pytest.mark.parametrize(
    "pdf",
    [
        os.path.join(HERE, "test_370.pdf"),
        os.path.join(HERE, "..", "examples", "country-capitals",
                     "national-capitals.pdf"),
    ],
)
def test_box_text_matches_to_markdown(pdf):
    # Two parses: some renderers mutate textlines in place, so the reference
    # rendering and the chunker rendering each get a pristine document.
    ref_doc = parse_document(pdf)
    our_doc = parse_document(pdf)

    ours_by_box = {}
    for page in our_doc.pages:
        lil = create_list_item_levels(page.boxes)
        for box_idx, box in enumerate(page.boxes):
            ours_by_box[(page.page_number, box_idx)] = box_to_markdown(
                page, box, box_idx, lil
            )

    compared = 0
    mismatches = []
    for page, box, box_idx, seg in _iter_box_segments(ref_doc):
        btype = box.boxclass
        if btype in ("picture", "formula"):
            continue  # placeholder / documented deviation
        ours = ours_by_box[(page.page_number, box_idx)]
        compared += 1
        if _norm(ours) != _norm(_IMG_RE.sub("", seg)):
            mismatches.append((page.page_number, box_idx, btype, seg, ours))

    assert compared > 0
    assert not mismatches, "\n".join(
        f"p{p} b{b} {t}\n  to_md: {_norm(s)[:150]!r}\n  ours : {_norm(o)[:150]!r}"
        for p, b, t, s, o in mismatches[:10]
    )


def test_box_to_markdown_is_side_effect_free():
    """Rendering via box_to_markdown must not change a later to_markdown."""
    pdf = os.path.join(HERE, "test_370.pdf")
    doc = parse_document(pdf)
    for page in doc.pages:
        lil = create_list_item_levels(page.boxes)
        for box_idx, box in enumerate(page.boxes):
            box_to_markdown(page, box, box_idx, lil)
    md_after_chunk_render = doc.to_markdown()

    md_pristine = parse_document(pdf).to_markdown()
    assert md_after_chunk_render == md_pristine
