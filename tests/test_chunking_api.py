"""P0 API surface and regression tests for the layout-aware chunker.

- public symbol snapshot (to_chunks unification, Chunk rename)
- kwargs router: internal engine flags are rejected, aliases still work
- chunk text containment in to_markdown (whitespace-normalized)
- pages= partial parse keeps original page numbers in chunk addresses
- OCR'd document still yields chunk text (invisible-text-layer case is
  handled by the 1.28 OCR pipeline; the v1 parse_document guard was
  dropped on that basis)
"""

import os
import re

import pytest

import pymupdf4llm
from pymupdf4llm.helpers import chunking

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "test_370.pdf")
OCR_PDF = os.path.join(HERE, "test_ocr_loremipsum_svg.pdf")


def _norm(s):
    return re.sub(r"\s+", " ", s).strip()


# ── public surface ──────────────────────────────────────────────────

def test_public_symbols_snapshot():
    # package level
    for name in ("to_chunks", "to_markdown", "to_json", "to_text"):
        assert callable(getattr(pymupdf4llm, name)), name

    # chunking package level
    for name in (
        "to_chunks", "Chunk", "ChunkMetadata",
        "ChunkedDocument", "Element", "TableChunk", "FigureChunk",
        "SectionChunk", "SentenceUnit",
    ):
        assert hasattr(chunking, name), name



def test_internal_parse_flags_rejected():
    with pytest.raises(TypeError):
        pymupdf4llm.to_chunks(PDF, render_html_tables=True)


def test_edge_threshold_is_public():
    # edge_threshold is a public parse_document kwarg since table_output
    # landed upstream; the router must pass it through, not reject it.
    chunks = pymupdf4llm.to_chunks(PDF, pages=[0], edge_threshold=0.55)
    assert chunks


def test_table_output_router():
    # explicit default routes like the implicit one
    chunks = pymupdf4llm.to_chunks(PDF, pages=[0], table_output="markdown")
    assert chunks
    with pytest.raises(ValueError):
        pymupdf4llm.to_chunks(PDF, pages=[0], table_output="csv")
    # html opt-in must not crash even without the improved PyMuPDF table
    # model: the parse layer degrades to markdown tables with a warning.
    chunks = pymupdf4llm.to_chunks(PDF, pages=[0], table_output="html")
    assert chunks
    # table_output is a substrate (parse-level) option, not accepted by reassemble_chunks
    with pytest.raises(ValueError):
        chunks.reassemble_chunks(table_output="markdown")


def test_kwargs_router_aliases():
    # dpi → image_dpi and chunking kwargs must both route without error
    chunks = pymupdf4llm.to_chunks(PDF, pages=[0], dpi=96, max_tokens=200)
    assert chunks


# ── W0-4: text equivalence regression ──────────────────────────────

def test_chunk_text_contained_in_markdown():
    chunks = pymupdf4llm.to_chunks(PDF)
    md = _norm(pymupdf4llm.to_markdown(PDF))
    missing = []
    for c in chunks:
        for para in c.text.split("\n"):
            p = _norm(para)
            if p and not p.startswith("[Figure") and p not in md:
                missing.append(p[:120])
    assert not missing, missing[:5]


def test_ocr_document_yields_chunk_text():
    chunks = pymupdf4llm.to_chunks(OCR_PDF)
    assert chunks
    assert any(_norm(c.text) for c in chunks)


# ── W0-5: pages= partial parse keeps original addresses ────────────

def test_partial_parse_preserves_page_numbers():
    chunks = pymupdf4llm.to_chunks(PDF, pages=[2, 3])
    assert chunks
    pages_seen = set()
    for c in chunks:
        pages_seen.add(c.metadata.page_start)
        pages_seen.add(c.metadata.page_end)
        for eid in c.metadata.element_ids:
            pages_seen.add(int(eid.split(".")[0][1:]))
    # pages= is 0-based; page addresses are original 1-based page numbers
    assert pages_seen <= {3, 4}
    assert pages_seen
