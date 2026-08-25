"""Layout-Aware Chunking for PyMuPDF4LLM.

Layout-only chunker using PDF-native signals (box boundaries, font
changes, vertical gaps, page breaks).  Entry point: :func:`to_chunks`,
returning a :class:`ChunkedDocument`.
"""

from .models import (
    Chunk,
    ChunkMetadata,
    Element,
    FigureChunk,
    ProtoChunk,
    SectionChunk,
    SectionNode,
    SentenceUnit,
    TableChunk,
    element_id,
)
from .chunked_document import ChunkedDocument, REASSEMBLY_PARAMS
from .sentence_builder import SentenceBuilder
from .boundary_scorer import BoundaryScorer
from .chunk_assembler import ChunkAssembler
from .serializer import ChunkSerializer
from .token_utils import TokenCounter

# Default weights (layout signals only)
WEIGHTS_LAYOUT = {
    "w_box": 0.35,
    "w_class": 0.50,
    "w_page": 0.15,
    "w_gap": 0.30,
    "w_hgap": 0.30,
    "w_font": 0.25,
    "w_head": 0.45,
    "w_foot": 0.20,
    "w_list": 0.40,
    "w_table": 0.60,
    "w_caption": 0.80,
}

DEFAULTS = {
    # assembly tier (accepted by reassemble_chunks)
    "max_tokens": 400,
    "min_tokens": 120,
    "breakpoint_threshold": 0.5,
    "merge_small_chunks": True,
    "table_mode": "preserve",
    "respect_section_starts": True,
    # substrate tier (fixed per to_chunks call)
    "header_footer_mode": "exclude",
    "sentence_splitter": "default",
    "tokenizer": None,
    "weights": None,
}


def to_chunks(parsed_doc, **kwargs):
    """Chunk a ParsedDocument; returns a :class:`ChunkedDocument`."""
    unknown = set(kwargs) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown to_chunks parameters: {sorted(unknown)}")
    params = {**DEFAULTS, **kwargs}

    # Step A: Box → SentenceUnit (all units, header/footer included — the
    # element registry must keep everything the parser saw)
    builder = SentenceBuilder(splitter=params["sentence_splitter"])
    all_units = builder.build_from_document(parsed_doc)

    return _chunk_units(parsed_doc, all_units, params)


def _chunk_units(parsed_doc, all_units, params):
    """Steps B–E from prebuilt units (shared by to_chunks and reassemble_chunks)."""
    elements = _build_elements(parsed_doc, all_units)

    # Header/footer handling
    hf_mode = params["header_footer_mode"]
    if hf_mode == "auto":
        builder = SentenceBuilder(splitter=params["sentence_splitter"])
        repeated = builder.detect_repeated_headers_footers(parsed_doc)

        def _is_repeated(s):
            if (s.page_no, s.box_index) in repeated:
                return True
            return any((s.page_no, bi) in repeated
                       for bi in s._source_box_indices)

        units = [s for s in all_units if not _is_repeated(s)]
    elif hf_mode == "exclude":
        units = [s for s in all_units if not s.is_header_footer]
    else:  # "include"
        units = list(all_units)
    hf_excluded = len(all_units) - len(units)

    # Renumber so unit_range is contiguous over the working list, and
    # precompute token counts once (assembly only sums; D12)
    counter = TokenCounter(params["tokenizer"])
    for i, u in enumerate(units):
        u.sent_id = i
        u.token_count = counter.count(u.text)

    chunks, tables, figures, sections = [], [], [], []
    if units:
        # Step B: Boundary scoring (layout-only)
        scorer = BoundaryScorer(weights=params["weights"] or WEIGHTS_LAYOUT)
        scores = scorer.score_all(units)

        # Steps C+D: Chunk assembly + refinement
        assembler = ChunkAssembler(
            max_tokens=params["max_tokens"],
            min_tokens=params["min_tokens"],
            threshold=params["breakpoint_threshold"],
            table_mode=params["table_mode"],
            merge_small_chunks=params["merge_small_chunks"],
            respect_section_starts=params["respect_section_starts"],
        )
        proto_chunks = assembler.assemble(units, scores)
        proto_chunks = assembler.refine(proto_chunks)

        # Step E: Serialization + views
        serializer = ChunkSerializer(doc=parsed_doc)
        chunks, tables, figures, sections = serializer.serialize(proto_chunks)

    return ChunkedDocument(
        chunks,
        elements=elements,
        tables=tables,
        figures=figures,
        sections=sections,
        params=params,
        doc=parsed_doc,
        all_units=all_units,
        header_footer_excluded=hf_excluded,
    )


def _build_elements(parsed_doc, all_units):
    """Element registry: one entry per layout box, everything preserved (D8).

    Text comes from the units already rendered for the box; boxes that
    produced no unit (nothing renders) appear with text "".  When repeated
    header/footer units were merged, the merged text lives on the first
    source box's element.
    """
    texts_by_box = {}
    for u in all_units:
        texts_by_box.setdefault((u.page_no, u.box_index), []).append(u.text)

    elements = []
    for page in parsed_doc.pages:
        for box_idx, box in enumerate(page.boxes):
            key = (page.page_number, box_idx)
            texts = texts_by_box.get(key, [])
            elements.append(Element(
                id=element_id(*key),
                page=key[0],
                box=box_idx,
                boxclass=box.boxclass,
                bbox=(box.x0, box.y0, box.x1, box.y1),
                text="\n".join(texts),
                is_header_footer=box.boxclass in ("page-header", "page-footer"),
            ))
    return elements
