"""HTML table mode integration tests for the chunking surface.

These run only against a PyMuPDF that carries the layout-union table model
(find_tables(use_layout=..., union=..., refine=...)); on a stock build the
whole module is skipped and table_output="html" degrades to markdown tables
(covered by test_chunking_api.test_table_output_router).
"""

import inspect
import os
import re

import pytest

import pymupdf
import pymupdf4llm

# Page.find_tables may be a (self, **kwargs) forwarding wrapper, so probe
# the underlying implementation for the layout-union model.
HAS_IMPROVED_TABLES = "use_layout" in inspect.signature(
    pymupdf.table.find_tables
).parameters

pytestmark = pytest.mark.skipif(
    not HAS_IMPROVED_TABLES,
    reason="requires PyMuPDF with the layout-union table model",
)

g_root = os.path.abspath(f"{__file__}/../..")
TABLE_PDF = os.path.join(g_root, "tests", "test_sce_150_1.pdf")

_WS = re.compile(r"\s+")


def _norm(s):
    return _WS.sub(" ", s).strip()


@pytest.fixture(scope="module")
def html_doc():
    return pymupdf4llm.to_chunks(TABLE_PDF, table_output="html")


def test_html_mode_table_contract(html_doc):
    # D15: html is canonical in html mode, markdown must stay None.
    tables = html_doc.tables
    assert tables, "fixture is expected to contain at least one table"
    for t in tables:
        assert t.html is not None and "<table" in t.html
        assert t.markdown is None
        assert t.text == t.html


def test_html_mode_headers_from_th(html_doc):
    # D13: headers come only from <th> cells in the engine HTML —
    # populated iff the engine emitted <th>, never from own heuristics.
    for t in html_doc.tables:
        if "<th" in t.html:
            assert t.headers
            for h in t.headers:
                assert isinstance(h, str)
        else:
            assert t.headers == []


def test_md_mode_headers_stay_empty():
    # D13: markdown mode carries no header metadata even on the improved
    # engine; headers must remain [].
    cdoc = pymupdf4llm.to_chunks(TABLE_PDF)
    for t in cdoc.tables:
        assert t.html is None
        assert t.markdown is not None
        assert t.headers == []


def test_html_mode_chunk_text_matches_markdown(html_doc):
    # Chunk text contract holds in html mode too: every chunk paragraph
    # appears in to_markdown(table_output="html") output.
    md = _norm(pymupdf4llm.to_markdown(TABLE_PDF, table_output="html"))
    missing = []
    for c in html_doc:
        for para in c.text.split("\n"):
            p = _norm(para)
            if p and not p.startswith("[Figure") and p not in md:
                missing.append(p[:120])
    assert not missing, missing[:5]
