"""ChunkedDocument tests: views, id contracts, reassemble_chunks, diagnostics.

The html-mode paths (TableChunk.html, headers from <th>) run on canned
dicts only — real html tables need the improved PyMuPDF and are gated by
W0-0-final integration tests.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import pymupdf
import pytest

import pymupdf4llm
from pymupdf4llm.helpers import chunking
from pymupdf4llm.helpers.chunking import (
    Chunk,
    ChunkedDocument,
    SectionNode,
    SentenceUnit,
)
from pymupdf4llm.helpers.chunking.chunk_assembler import ChunkAssembler
from pymupdf4llm.helpers.chunking.sentence_builder import (
    _SENT_END_EN,
    _SENT_END_MULTI,
    SentenceBuilder,
    _split_sentence_text,
)
from pymupdf4llm.helpers.chunking.serializer import _group_list_items
from pymupdf4llm.helpers.chunking.text_source import (
    box_to_markdown,
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
        # chunk_id is None only for a table box that rendered to nothing
        assert t.chunk_id is None or t.chunk_id in chunk_ids
        if t.chunk_id:
            assert t.id in cd.get(t.chunk_id).metadata.table_ids
            assert t.section_id == cd.get(t.chunk_id).metadata.section_id
        assert cd.get(t.element_id) is not None

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
            # a figure whose text spans several chunks is listed by all of
            # them; its own chunk_id names the first one
            assert cd.get(fid) is not None
        if c.metadata.section_id:
            assert c.id in cd.get(c.metadata.section_id).child_chunk_ids

    for f in cd.figures:
        holders = [c.id for c in cd if f.id in c.metadata.figure_ids]
        assert holders and f.chunk_id == holders[0]


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


def test_tagged_content(cd):
    c = cd[0]
    assert "[Markdown]" in c.tagged_content
    assert c.text in c.tagged_content


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
    assert "tagged_content" in d0

    meta = d0["metadata"]
    for key in ("page_start", "page_end", "bboxes", "types",
                "section_id", "section_path", "token_count",
                "element_ids", "table_ids", "figure_ids", "lists", "ocr",
                "file_path", "page_count"):
        assert key in meta, key
    # internals must not leak into the payload
    for key in ("box_indices", "sent_ids", "toc_items", "is_table_related",
                "primary_type"):
        assert key not in meta, key
    assert meta["token_count"] > 0
    assert meta["ocr"] is False

    lean = cd.to_dicts(include_tagged=False)
    assert "tagged_content" not in lean[0]

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
    assert f"[Section] {' > '.join(s0.path)}" in owned.tagged_content
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


# ════════════════════════════════════════════════════════════════════
# Review regressions
#
# Each case below reproduces one reported defect on the smallest input
# that shows it.  Layout-shaped inputs (an empty middle page, a table
# that renders to nothing, a list continuing across a page break) are
# built here instead of searched for in a PDF, so the case stays exact.
# ════════════════════════════════════════════════════════════════════

@dataclass
class _LayoutBox:
    """LayoutBox stand-in (only the attributes chunking reads)."""
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 300.0
    y1: float = 20.0
    boxclass: str = "text"
    image: Optional[bytes] = None
    table: Optional[dict] = None
    textlines: Optional[list] = None
    header_level: Optional[int] = 1
    max_fontsize: Optional[float] = None


@dataclass
class _PageLayout:
    page_number: int = 1
    width: float = 612.0
    height: float = 792.0
    boxes: list = field(default_factory=list)
    full_ocred: bool = False
    text_ocred: bool = False
    fulltext: Optional[list] = None
    words: Optional[list] = None
    links: Optional[list] = None


@dataclass
class _ParsedDoc:
    filename: Optional[str] = "synthetic.pdf"
    page_count: int = 1
    toc: list = field(default_factory=list)
    pages: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _textlines(text, size=10.0, x0=0.0, x1=300.0, y0=0.0):
    """One textline per line of *text*, in the shape the renderers expect."""
    lines = []
    y = y0
    for line_no, line_text in enumerate(text.split("\n")):
        y1 = y + size + 2
        lines.append({
            "bbox": pymupdf.Rect(x0, y, x1, y1),
            "spans": [{
                "text": line_text, "bbox": (x0, y, x1, y1),
                "font": "Helvetica", "size": size, "flags": 0,
                "char_flags": 0, "origin": (x0, y1 - 2),
                "block": 0, "line": line_no,
            }],
        })
        y += size + 4
    return lines


def _unit(sent_id, text, *, page=1, box=0, tokens=None, boxclass="text",
          bbox=(0.0, 0.0, 300.0, 20.0), **hints):
    return SentenceUnit(
        sent_id=sent_id, text=text, norm_text=text.lower(), page_no=page,
        box_index=box, boxclass=boxclass, bbox=bbox,
        token_count=tokens if tokens is not None else max(1, len(text) // 4),
        font_size_dominant=10.0, **hints,
    )


def _nows(text):
    return "".join(text.split())


# ── 1.2  sentence splitting must not drop characters ────────────────

_QUOTE_CASES = [
    'He said, "Yes." Then he left.',
    'She replied, "No!" He nodded. [1] Later they agreed.',
    "The result (see Fig. 1.) was clear. Another sentence follows.",
    "Values differ [a]. Others match.",
    "Ends with a quote. 'Quoted!' And continues.",
]


@pytest.mark.parametrize("pattern", [_SENT_END_EN, _SENT_END_MULTI])
@pytest.mark.parametrize("text", _QUOTE_CASES)
def test_sentence_split_keeps_every_non_whitespace_character(pattern, text):
    """Splitting may drop whitespace, never text (both splitters)."""
    pieces = _split_sentence_text(text, pattern)
    assert _nows(" ".join(pieces)) == _nows(text)


@pytest.mark.parametrize("pattern", [_SENT_END_EN, _SENT_END_MULTI])
def test_closing_quote_stays_with_its_sentence(pattern):
    assert _split_sentence_text('He said, "Yes." Then he left.', pattern) == [
        'He said, "Yes."', "Then he left."]


def test_box_sentences_reproduce_the_rendered_markdown():
    """The units of one box must carry the box's whole rendering."""
    source = ('He said, "Yes." Then he left. '
              'She asked, "Why?" Nobody answered.')
    doc = _ParsedDoc(pages=[_PageLayout(boxes=[_LayoutBox(textlines=_textlines(source))])])
    units = SentenceBuilder().build_from_document(doc)
    rendered = box_to_markdown(doc.pages[0], doc.pages[0].boxes[0], 0).strip()

    assert len(units) == 4
    assert _nows(" ".join(u.text for u in units)) == _nows(rendered)
    assert units[0].text.endswith('"Yes."')


# ── 1.3  splitting an oversized chunk must respect the budget ───────

def test_split_keeps_every_chunk_within_budget():
    """Carrying a shared box forward must not push the next chunk over.

    Review repro: units (box_index, tokens) = (0,10) (1,10) (2,40)
    (2,40) (2,40) with max_tokens=100.  Assembly keeps the box-2 run
    together, so the split step has to break inside the box rather than
    carry all of it into the next chunk.
    """
    specs = [(0, 10), (1, 10), (2, 40), (2, 40), (2, 40)]
    units = [_unit(i, "x" * (tokens * 4), box=box, tokens=tokens)
             for i, (box, tokens) in enumerate(specs)]

    assembler = ChunkAssembler(max_tokens=100, min_tokens=0)
    proto = assembler.assemble(units, [0.0] * (len(units) - 1))
    assert [p.token_count for p in proto] == [140]      # one oversized chunk

    refined = assembler.refine(proto)
    assert all(p.token_count <= 100 for p in refined), \
        [p.token_count for p in refined]
    # nothing dropped or duplicated by the split
    assert [s.sent_id for p in refined for s in p._sentences] == \
        [u.sent_id for u in units]


# ── 1.1  framework exports: id scope and chunk_id round trip ────────

def _second_document(cd):
    """A different ChunkedDocument reusing the same (document-local) ids."""
    return ChunkedDocument([Chunk(id=c.id, text=c.text) for c in cd[:2]])


def test_llama_export_ids_can_be_document_scoped(cd):
    pytest.importorskip("llama_index.core")
    other = _second_document(cd)

    unscoped = cd.to_llama_nodes() + other.to_llama_nodes()
    assert len({n.id_ for n in unscoped}) < len(unscoped)   # ids are local

    scoped = cd.to_llama_nodes(doc_id="docA") + other.to_llama_nodes(doc_id="docB")
    assert len({n.id_ for n in scoped}) == len(scoped)
    assert scoped[0].id_ == f"docA:{cd[0].id}"

    for node in cd.to_llama_nodes(doc_id="docA"):
        assert cd.get(node.metadata["chunk_id"]).text == node.text


def test_langchain_export_ids_can_be_document_scoped(cd):
    pytest.importorskip("langchain_core")
    other = _second_document(cd)

    unscoped = cd.to_langchain_documents() + other.to_langchain_documents()
    assert len({d.id for d in unscoped}) < len(unscoped)

    scoped = (cd.to_langchain_documents(doc_id="docA")
              + other.to_langchain_documents(doc_id="docB"))
    assert len({d.id for d in scoped}) == len(scoped)
    assert scoped[0].id == f"docA:{cd[0].id}"

    for doc in cd.to_langchain_documents(doc_id="docA"):
        assert cd.get(doc.metadata["chunk_id"]).text == doc.page_content


# ── §3  diagnostics: empty pages, empty tables ──────────────────────

def _page_gap_document():
    """Pages 1 and 3 carry text; page 2 is empty and was OCR'd."""
    return _ParsedDoc(page_count=3, pages=[
        _PageLayout(page_number=1, boxes=[
            _LayoutBox(y0=700.0, y1=720.0,
                 textlines=_textlines("First page body.", y0=700.0))]),
        _PageLayout(page_number=2, boxes=[], full_ocred=True),
        _PageLayout(page_number=3, boxes=[
            _LayoutBox(y0=20.0, y1=40.0,
                 textlines=_textlines("Third page body.", y0=20.0))]),
    ])


def test_empty_middle_page_is_reported_as_uncovered():
    cd = chunking.to_chunks(_page_gap_document())
    chunk = cd[0]
    # one chunk spanning the gap: the span says 1-3, the content does not
    assert (chunk.metadata.page_start, chunk.metadata.page_end) == (1, 3)
    assert cd.diagnostics["pages_without_chunks"] == [2]


def test_ocr_flag_follows_contributing_pages():
    cd = chunking.to_chunks(_page_gap_document())
    # page 2 is the OCR'd one and it put nothing into the chunk
    assert not any(c.metadata.ocr for c in cd)


def test_degenerate_table_stays_addressable_and_reported():
    """A table box that renders to nothing must not vanish silently."""
    doc = _ParsedDoc(pages=[_PageLayout(boxes=[
        _LayoutBox(boxclass="table", y0=100.0, y1=140.0,
             table={"markdown": "|a|b|\n|---|---|\n|1|2|"}),
        _LayoutBox(boxclass="table", y0=200.0, y1=240.0, table={"markdown": ""}),
        _LayoutBox(y0=300.0, y1=320.0,
             textlines=_textlines("Body text.", y0=300.0)),
    ])])
    cd = chunking.to_chunks(doc)

    assert [t.element_id for t in cd.tables] == ["p1.b0", "p1.b1"]
    empty = cd.tables[1]
    assert empty.chunk_id is None and empty.text == ""
    assert cd.get(empty.id) is empty
    assert cd.get(empty.element_id).text == ""
    assert cd.diagnostics["degenerate_tables"] == [empty.id]
    # the rendered table keeps its chunk link and its id ordering
    assert cd.tables[0].chunk_id
    assert cd.get(cd.tables[0].chunk_id).metadata.table_ids == [cd.tables[0].id]


# ── §3  provenance: figures across chunks, lists across pages ───────

def test_figure_split_across_chunks_links_every_chunk():
    """Every chunk holding part of a figure's text carries its id."""
    body = " ".join(f"{word} " * 12 for word in ("Alpha.", "Bravo.", "Charlie."))
    doc = _ParsedDoc(pages=[_PageLayout(boxes=[
        _LayoutBox(boxclass="formula", y0=100.0, y1=300.0,
             textlines=_textlines(body, y0=100.0))])])
    cd = chunking.to_chunks(doc, max_tokens=25, min_tokens=0)

    assert len(cd) > 1
    figure = cd.figures[0]
    holders = [c.id for c in cd if figure.id in c.metadata.figure_ids]
    assert holders == [c.id for c in cd]        # complete back-links
    assert figure.chunk_id == holders[0]        # primary link is the first


def test_multi_page_list_keeps_a_bbox_per_page():
    items = [
        _unit(0, "- first", page=1, boxclass="list-item",
              bbox=(50.0, 700.0, 300.0, 712.0), is_list_item=True),
        _unit(1, "- second", page=2, boxclass="list-item",
              bbox=(50.0, 20.0, 300.0, 32.0), is_list_item=True),
    ]
    group, = _group_list_items(items)

    assert group["bboxes"] == [(1, 50.0, 700.0, 300.0, 712.0),
                               (2, 50.0, 20.0, 300.0, 32.0)]
    assert [i["page"] for i in group["items"]] == [1, 2]
    # no rectangle that exists on neither page
    assert all(len(b) == 5 for b in group["bboxes"])


# ── §3  min_tokens and section-start protection ─────────────────────

def _two_paragraph_chunks(assembler, tokens=(150, 150)):
    units = [
        _unit(0, "A" * (tokens[0] * 4), box=0, tokens=tokens[0],
              bbox=(0.0, 100.0, 300.0, 200.0)),
        _unit(1, "B" * (tokens[1] * 4), box=1, tokens=tokens[1],
              bbox=(0.0, 300.0, 300.0, 400.0)),
    ]
    return [assembler._make_proto_chunk(i, [u]) for i, u in enumerate(units)]


def test_min_tokens_controls_budget_merge():
    """Two self-sufficient chunks are only merged below the floor."""
    def merged(min_tokens):
        assembler = ChunkAssembler(max_tokens=400, min_tokens=min_tokens)
        return assembler.refine(_two_paragraph_chunks(assembler))

    assert [c.token_count for c in merged(0)] == [300]     # floor disabled
    assert [c.token_count for c in merged(120)] == [150, 150]
    assert [c.token_count for c in merged(200)] == [300]   # both below floor


def test_small_chunk_still_merges_into_its_neighbour():
    assembler = ChunkAssembler(max_tokens=400, min_tokens=120)
    protos = _two_paragraph_chunks(assembler, tokens=(50, 150))
    assert [c.token_count for c in assembler.refine(protos)] == [200]


def test_semantic_merge_does_not_cross_a_section_start():
    """A lone parent heading must not swallow the next section's head."""
    assembler = ChunkAssembler(max_tokens=400, min_tokens=0,
                               respect_section_starts=True)
    parent = _unit(0, "# Parent", boxclass="title", tokens=5,
                   bbox=(0.0, 100.0, 300.0, 120.0),
                   is_heading_hint=True, heading_level_hint=1)
    child = _unit(1, "## Child", boxclass="section-header", box=1, tokens=5,
                  bbox=(0.0, 140.0, 300.0, 160.0),
                  is_heading_hint=True, heading_level_hint=2)
    body = _unit(2, "Child body text.", box=2, tokens=20,
                 bbox=(0.0, 170.0, 300.0, 200.0))

    kept = assembler.refine([assembler._make_proto_chunk(0, [parent]),
                             assembler._make_proto_chunk(1, [child, body])])
    assert [[s.text for s in c._sentences] for c in kept] == [
        ["# Parent"], ["## Child", "Child body text."]]

    # a heading followed by plain content still merges
    paragraph = _unit(1, "Body text follows.", box=1, tokens=20,
                      bbox=(0.0, 130.0, 300.0, 160.0))
    joined = assembler.refine([assembler._make_proto_chunk(0, [parent]),
                               assembler._make_proto_chunk(1, [paragraph])])
    assert len(joined) == 1

    # ... and without the guard the old greedy behaviour is available
    loose = ChunkAssembler(max_tokens=400, min_tokens=0,
                           respect_section_starts=False)
    assert len(loose.refine([loose._make_proto_chunk(0, [parent]),
                             loose._make_proto_chunk(1, [child, body])])) == 1


# ── §3  parameter validation ────────────────────────────────────────

_INVALID_PARAMS = [
    {"max_tokens": 0},
    {"max_tokens": -5},
    {"min_tokens": -1},
    {"header_footer_mode": "exlcude"},
    {"table_mode": "isolated"},
    {"sentence_splitter": "multi"},
]


@pytest.mark.parametrize("params", _INVALID_PARAMS)
def test_to_chunks_rejects_invalid_values(params):
    name = next(iter(params))
    with pytest.raises(ValueError, match=name):
        chunking.to_chunks(_page_gap_document(), **params)


@pytest.mark.parametrize("params", [{"max_tokens": 0}, {"min_tokens": -1},
                                    {"table_mode": "isolated"}])
def test_reassemble_chunks_rejects_invalid_values(cd, params):
    name = next(iter(params))
    with pytest.raises(ValueError, match=name):
        cd.reassemble_chunks(**params)
