"""P1 tests: ChunkedDocument, views, id contracts, reassemble_chunks, diagnostics.

The html-mode paths (TableChunk.html, headers from <th>) run on canned
dicts only — real html tables need the improved PyMuPDF and are gated by
W0-0-final integration tests.
"""

import json
import os
import re

import pytest

import pymupdf4llm
from pymupdf4llm.helpers.chunking import ChunkedDocument, SectionNode
from pymupdf4llm.helpers.chunking.text_source import (
    extract_table_headers,
    table_content,
)
from pymupdf4llm.helpers.chunking.token_utils import TokenCounter

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "test_370.pdf")

_ID_RES = {
    "chunk": re.compile(r"^c\d+$"),
    "table": re.compile(r"^t\d+$"),
    "figure": re.compile(r"^f\d+$"),
    "section": re.compile(r"^s\d+$"),
    "element": re.compile(r"^p\d+\.b\d+$"),
}


@pytest.fixture(scope="module")
def cd():
    return pymupdf4llm.to_chunks(PDF)


# ── id format = public contract (snapshot) ──────────────────────────

def test_id_formats(cd):
    assert all(_ID_RES["chunk"].match(c.id) for c in cd)
    assert all(_ID_RES["table"].match(t.id) for t in cd.tables)
    assert all(_ID_RES["figure"].match(f.id) for f in cd.figures)
    assert all(_ID_RES["section"].match(s.id) for s in cd.sections)
    assert cd.elements and all(_ID_RES["element"].match(e.id) for e in cd.elements)
    # first element of a 1-based page numbering
    assert cd.elements[0].id == "p1.b0"
    # pre-rename alias stays readable
    assert cd[0].chunk_id == cd[0].id


# ── Sequence protocol + lazy text ───────────────────────────────────

def test_sequence_protocol(cd):
    assert isinstance(cd, ChunkedDocument)
    assert len(cd) > 0
    assert cd[0].id == "c0"
    assert [c.id for c in cd[:2]] == ["c0", "c1"]
    assert list(iter(cd))[-1] is cd[len(cd) - 1]
    assert cd.chunks == tuple(cd)
    assert cd.text  # lazy join
    assert cd[0].text in cd.text


def test_params_read_only(cd):
    with pytest.raises(TypeError):
        cd.params["max_tokens"] = 1


# ── t/f/s ↔ chunk round trip ────────────────────────────────────────

def test_views_round_trip(cd):
    chunk_ids = {c.id for c in cd}

    for t in cd.tables:
        assert t.chunk_id in chunk_ids
        assert t.id in cd.get(t.chunk_id).metadata.table_ids
        assert cd.get(t.element_id) is not None
        assert t.section_id == cd.get(t.chunk_id).metadata.section_id

    for f in cd.figures:
        assert f.chunk_id in chunk_ids
        assert f.id in cd.get(f.chunk_id).metadata.figure_ids
        assert f.section_id == cd.get(f.chunk_id).metadata.section_id
        # placeholder figures carry their id in the placeholder text
        if f.placeholder:
            assert f.placeholder.startswith(f"[Figure {f.id}:")

    for s in cd.sections:
        assert s.child_chunk_ids, s.id
        for cid in s.child_chunk_ids:
            assert cid in chunk_ids

    for c in cd:
        for tid in c.metadata.table_ids:
            assert cd.get(tid).chunk_id == c.id
        for fid in c.metadata.figure_ids:
            assert cd.get(fid).chunk_id == c.id
        if c.metadata.section_id:
            assert c.id in cd.get(c.metadata.section_id).child_chunk_ids


def test_section_fields_and_lazy_text(cd):
    assert cd.sections
    total_elements = len(cd.elements)
    for s in cd.sections:
        assert s.heading_element_id and _ID_RES["element"].match(s.heading_element_id)
        assert s.path and s.path[-1] == s.title
        lo, hi = s.element_span
        assert 0 <= lo < hi <= total_elements
        # the heading element opens its own span
        assert cd.elements[lo].id == s.heading_element_id
        assert s.token_count >= 0
    # lazy section text assembles from the registry
    assert any(s.text for s in cd.sections)


def test_hierarchy_tree(cd):
    root = cd.hierarchy
    assert isinstance(root, SectionNode)
    assert root.level == 0 and root.section_id is None
    assert root.children  # document has sections

    seen = []

    def _walk(node):
        for child in node.children:
            assert child.level > node.level or node is root
            seen.append(child.section_id)
            _walk(child)

    _walk(root)
    assert set(seen) == {s.id for s in cd.sections}


def test_element_registry_keeps_header_footers(cd):
    # D8: excluded header/footer boxes remain addressable as elements
    hf = [e for e in cd.elements if e.is_header_footer]
    assert cd.diagnostics["header_footer_excluded"] > 0
    assert hf and any(e.text for e in hf)


def test_get_semantics(cd):
    assert cd.get(cd[0].id) is cd[0]
    with pytest.raises(KeyError):
        cd.get("c999999")
    with pytest.raises(KeyError):
        cd.get("x123")
    assert cd.get("c999999", None) is None
    assert cd.get("x123", "fallback") == "fallback"


def test_contextualize(cd):
    c = cd[0]
    assert cd.contextualize(c) == c.contextual_text
    assert "[Content]" in c.contextual_text


# ── content_hash (D17) + serialization ──────────────────────────────

def test_content_hash_and_to_dicts(cd):
    c = cd[0]
    h1 = c.content_hash
    assert re.match(r"^[0-9a-f]{64}$", h1)
    assert c.content_hash == h1  # cached

    dicts = cd.to_dicts()
    assert len(dicts) == len(cd)
    d0 = dicts[0]
    assert d0["id"] == "c0"
    assert d0["content_hash"] == h1
    assert "contextual_text" in d0

    meta = d0["metadata"]
    for key in ("page_start", "page_end", "bboxes", "chunk_type",
                "chunk_types", "section_id", "section_path", "token_count",
                "element_ids", "table_ids", "figure_ids", "lists", "ocr",
                "file_path", "page_count"):
        assert key in meta, key
    # internals must not leak into the payload
    for key in ("box_indices", "sent_ids", "toc_items", "is_table_related",
                "chunk_type_hint"):
        assert key not in meta, key
    assert meta["token_count"] > 0
    assert meta["ocr"] is False

    lean = cd.to_dicts(include_contextual=False)
    assert "contextual_text" not in lean[0]

    parsed = json.loads(cd.to_json())
    assert parsed[0]["content_hash"] == h1


# ── reassemble_chunks (2-tier policy, D11) ─────────────────────────────────────

def test_reassemble_chunks_same_params_is_identity(cd):
    again = cd.reassemble_chunks()
    assert len(again) == len(cd)
    assert [c.text for c in again] == [c.text for c in cd]
    assert [c.content_hash for c in again] == [c.content_hash for c in cd]


def test_reassemble_chunks_new_budget_changes_chunks(cd):
    small = cd.reassemble_chunks(max_tokens=120)
    assert len(small) >= len(cd)
    # same content, different boundaries
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    assert norm(" ".join(c.text for c in small)) == norm(" ".join(c.text for c in cd))


def test_reassemble_chunks_rejects_non_assembly_params(cd):
    for bad in (
        {"pages": [0]},                          # parse tier
        {"extract_images": True},                # parse tier
        {"sentence_splitter": "multilingual"},   # substrate tier
        {"weights": {"w_box": 1.0}},             # substrate tier
        {"header_footer_mode": "keep"},          # substrate tier
        {"tokenizer": "cl100k_base"},            # substrate tier
    ):
        with pytest.raises(ValueError):
            cd.reassemble_chunks(**bad)


# ── diagnostics (D16) ───────────────────────────────────────────────

def test_diagnostics_shape(cd):
    d = cd.diagnostics
    for key in ("chunk_count", "element_count", "table_count", "figure_count",
                "section_count", "page_count", "pages_without_chunks",
                "zero_chunk_causes", "figures_without_text",
                "degenerate_tables", "header_footer_excluded"):
        assert key in d, key
    assert d["chunk_count"] == len(cd)
    assert d["zero_chunk_causes"] == []
    for fid in d["figures_without_text"]:
        assert not cd.get(fid).has_text


# ── token counter contract ──────────────────────────────────────────

def test_unknown_tiktoken_encoding_is_loud():
    pytest.importorskip("tiktoken")
    with pytest.raises(ValueError):
        TokenCounter("no-such-encoding")


# ── table adapter contracts (canned; html mode dormant on this base) ─

class _Box:
    def __init__(self, table):
        self.table = table


def test_table_content_routing():
    # markdown mode
    assert table_content(_Box({"markdown": "| a |"})) == ("| a |", None)
    # degenerate table: markdown == "" must still route as markdown
    assert table_content(_Box({"markdown": ""})) == ("", None)
    # html mode: markdown=None, html canonical
    md, html = table_content(_Box({"markdown": None, "html": "<table></table>"}))
    assert md is None and html == "<table></table>"
    # per-table entries accepted even without the aggregate "html" key (D13)
    md, html = table_content(_Box({
        "markdown": None,
        "html_tables": [{"html": "<table><tr><th>H</th></tr></table>"}],
    }))
    assert md is None and "<th>H</th>" in html
    # no table dict at all
    assert table_content(_Box(None)) == (None, None)


def test_extract_table_headers_canned():
    html = ("<table><tr><th>Name</th><th>Qty <b>(kg)</b></th></tr>"
            "<tr><td>x</td><td>1</td></tr></table>")
    assert extract_table_headers(html) == ["Name", "Qty (kg)"]
    # markdown-mode parses have no html → provably []
    assert extract_table_headers(None) == []
    assert extract_table_headers("<table><tr><td>a</td></tr></table>") == []


def test_section_path_from_headings_without_bookmarks():
    """D18: section_path derives from layout headings, not the PDF TOC.

    national-capitals.pdf has no bookmarks; before D18 its section_path
    was empty even though the sections view carried the title heading.
    """
    pdf = os.path.join(HERE, "..", "examples", "country-capitals",
                       "national-capitals.pdf")
    cd = pymupdf4llm.to_chunks(pdf)
    assert cd.sections, "expected the title heading to open a section"
    s0 = cd.sections[0]
    owned = cd.get(s0.child_chunk_ids[0])
    assert owned.metadata.section_path == s0.path
    assert s0.path and s0.path[-1] == s0.title
    assert f"[Section] {' > '.join(s0.path)}" in owned.contextual_text
    # reassemble_chunks keeps the derivation (serializer runs per assembly)
    for c in cd.reassemble_chunks(max_tokens=1200):
        if c.metadata.section_id == s0.id:
            assert c.metadata.section_path == s0.path
            break
    else:
        raise AssertionError("no reassembled chunk owned by s0")


def test_heading_depth_from_engine_levels(cd):
    """D20: section depth follows the engine's font-statistics levels.

    test_370.pdf carries section-header boxes at levels 1-4; the section
    paths and the hierarchy tree must nest accordingly (not flatten to a
    single level as the old TOC-or-2 fallback did).
    """
    levels = {s.level for s in cd.sections}
    assert len(levels) >= 3, f"expected nested levels, got {levels}"
    deepest = max(cd.sections, key=lambda s: s.level)
    assert len(deepest.path) == deepest.level
    assert deepest.path[:-1], "deep section must carry its ancestor titles"

    def depth(node):
        return 1 + max((depth(c) for c in node.children), default=0)
    assert depth(cd.hierarchy) >= 4  # root + >=3 nested section levels

    # a chunk owned by a deep section cites the full nested path
    if deepest.child_chunk_ids:
        c = cd.get(deepest.child_chunk_ids[0])
        assert c.metadata.section_path == deepest.path
