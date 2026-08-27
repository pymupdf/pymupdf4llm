"""Step E: Serialize ProtoChunks into Chunks, plus table/figure/section views.

The serializer materializes chunk text once (assembly never joins text),
builds the tables/figures/sections views with bidirectional links
(chunk.metadata.table_ids/figure_ids/section_id <-> view.chunk_id(s)),
and attaches citation metadata (section_path, pages, element ids).

section_path derives from the layout-detected heading structure — it is
the innermost owning section's path (D18). It never consults the PDF TOC,
so it stays populated and consistent with the sections view on documents
without bookmarks.
"""

import re

from .models import (
    Chunk,
    ChunkMetadata,
    FigureChunk,
    ProtoChunk,
    SectionChunk,
    TableChunk,
    element_id,
)
from .text_source import extract_table_headers

# Markdown decoration around section titles (both ends: "# **Title**")
_MD_DECOR_RE = re.compile(r'^[#>*_`\s]+')
_MD_DECOR_TRAIL_RE = re.compile(r'[*_`\s]+$')


class ChunkSerializer:
    """Converts ProtoChunks into Chunks and builds the document views."""

    def __init__(self, doc):
        self.doc = doc
        self._ocr_pages = {
            p.page_number for p in doc.pages if getattr(p, "full_ocred", False)
        }

    def serialize(self, proto_chunks: list[ProtoChunk]):
        """Convert ProtoChunks to Chunks.

        Returns (chunks, tables, figures, sections) where tables/figures/
        sections are the document views, bidirectionally linked to chunks.
        """
        if not proto_chunks:
            return [], [], [], []

        chunks = [self._make_chunk(pc) for pc in proto_chunks]
        tables, figures, sections = self._build_views(proto_chunks, chunks)

        # section_path is known only after the views pass assigned each
        # chunk its innermost section, so contextual text renders last.
        for pc, chunk in zip(proto_chunks, chunks):
            chunk.tagged_content = self._build_tagged_content(
                pc, chunk.metadata.section_path)

        return chunks, tables, figures, sections

    def _make_chunk(self, pc: ProtoChunk) -> Chunk:
        """Create a Chunk from a ProtoChunk (text materialized here, once)."""
        pc.text = "\n".join(s.text for s in pc._sentences)

        metadata = ChunkMetadata(
            page_start=pc.page_start,
            page_end=pc.page_end,
            element_ids=[element_id(p, b) for p, b in pc.box_indices],
            types=pc.types or [pc.primary_type or "paragraph"],
            token_count=pc.token_count,
            bboxes=pc.bboxes,
            lists=_group_list_items(pc._sentences),
            ocr=any(p in self._ocr_pages
                    for p in range(pc.page_start, pc.page_end + 1)),
            file_path=self.doc.filename,
            page_count=self.doc.page_count,
        )

        return Chunk(
            id=f"c{pc.chunk_id}",
            text=pc.text,
            tagged_content="",   # rendered in serialize() after views
            metadata=metadata,
        )

    # ── Views (tables / figures / sections) ─────────────────────────

    def _build_views(self, proto_chunks, chunks):
        """Build t/f/s views with bidirectional chunk links."""
        tables = []
        figures = []
        sections = []
        seen_figures = {}       # figure_number -> FigureChunk
        open_sections = []      # stack of (level, SectionChunk)

        # Global element index per (page, box) address, for element_span.
        box_offset = {}
        total_boxes = 0
        for page in self.doc.pages:
            box_offset[page.page_number] = total_boxes
            total_boxes += len(page.boxes)

        def _link(section, chunk):
            if chunk.id not in section.child_chunk_ids:
                section.child_chunk_ids.append(chunk.id)
                section.token_count += chunk.metadata.token_count

        for pc, chunk in zip(proto_chunks, chunks):
            for idx, s in enumerate(pc._sentences):
                if s.is_table_content:
                    cap = _nearest_caption(pc._sentences, idx, "table")
                    table = TableChunk(
                        id=f"t{len(tables)}",
                        chunk_id=chunk.id,
                        element_id=element_id(s.page_no, s.box_index),
                        page=s.page_no,
                        bbox=s.bbox,
                        markdown=s.table_markdown,
                        html=s.table_html,
                        headers=extract_table_headers(s.table_html),
                        caption=cap.text if cap else None,
                        caption_element_id=(
                            element_id(cap.page_no, cap.box_index) if cap else None),
                        token_count=s.token_count,
                    )
                    tables.append(table)
                    chunk.metadata.table_ids.append(table.id)

                elif s.is_figure_related and s.figure_number is not None:
                    fig = seen_figures.get(s.figure_number)
                    if fig is None:
                        placeholder = s.text if s.text.startswith("[Figure") else ""
                        cap = _nearest_caption(pc._sentences, idx, "figure")
                        fig = FigureChunk(
                            id=f"f{s.figure_number}",
                            chunk_id=chunk.id,
                            element_id=element_id(s.page_no, s.box_index),
                            page=s.page_no,
                            bbox=s.bbox,
                            boxclass=s.boxclass,
                            ocr_text=None if placeholder else s.text,
                            placeholder=placeholder,
                            caption=cap.text if cap else None,
                            caption_element_id=(
                                element_id(cap.page_no, cap.box_index) if cap else None),
                            image=s.image_data,
                        )
                        seen_figures[s.figure_number] = fig
                        figures.append(fig)
                        chunk.metadata.figure_ids.append(fig.id)
                    else:
                        # further sentences of the same figure box
                        if not s.text.startswith("[Figure"):
                            fig.ocr_text = ((fig.ocr_text + "\n" + s.text).strip()
                                            if fig.ocr_text else s.text)

                elif s.is_heading_hint:
                    level = s.heading_level_hint or 1
                    while open_sections and open_sections[-1][0] >= level:
                        open_sections.pop()
                    title = _MD_DECOR_TRAIL_RE.sub(
                        "", _MD_DECOR_RE.sub("", s.text)).strip()
                    section = SectionChunk(
                        id=f"s{len(sections)}",
                        title=title,
                        level=level,
                        page_start=s.page_no,
                        page_end=s.page_no,
                        heading_element_id=element_id(s.page_no, s.box_index),
                        path=[sec.title for _l, sec in open_sections] + [title],
                        element_span=(
                            box_offset.get(s.page_no, 0) + s.box_index,
                            total_boxes,
                        ),
                    )
                    sections.append(section)
                    # the chunk carrying the heading belongs to the section,
                    # even if the section closes within this same chunk
                    _link(section, chunk)
                    open_sections.append((level, section))

            # every open section contains this chunk; innermost wins the link
            for _level, section in open_sections:
                _link(section, chunk)
                section.page_end = max(section.page_end, pc.page_end)
            if open_sections:
                innermost = open_sections[-1][1]
                chunk.metadata.section_id = innermost.id
                chunk.metadata.section_path = list(innermost.path)

        # Close element spans: a section runs until the next heading at the
        # same or a shallower level (else to the end of the document).
        for i, sec in enumerate(sections):
            end = total_boxes
            for later in sections[i + 1:]:
                if later.level <= sec.level:
                    end = later.element_span[0]
                    break
            sec.element_span = (sec.element_span[0], end)

        # Tables/figures inherit the section of their owning chunk.
        chunk_by_id = {c.id: c for c in chunks}
        for view in (tables, figures):
            for item in view:
                owner = chunk_by_id.get(item.chunk_id)
                if owner is not None:
                    item.section_id = owner.metadata.section_id

        return tables, figures, sections

    # ── Contextual text ─────────────────────────────────────────────

    def _build_tagged_content(self, pc: ProtoChunk, hierarchy: list[str]) -> str:
        """Build context-enriched text for embedding."""
        parts = []

        if hierarchy:
            parts.append(f"[Section] {' > '.join(hierarchy)}")

        if pc.page_start == pc.page_end:
            parts.append(f"[Page] {pc.page_start}")
        else:
            parts.append(f"[Pages] {pc.page_start}-{pc.page_end}")

        non_para = [t for t in (pc.types or []) if t != "paragraph"]
        if non_para:
            parts.append(f"[Type] {', '.join(non_para)}")

        parts.append(f"[Markdown]\n{pc.text}")

        return "\n".join(parts)


# ── Module-level helpers ────────────────────────────────────────────

def _nearest_caption(sents, idx, target_type):
    """Nearest caption unit to sents[idx] targeting *target_type* (or untyped)."""
    best = None
    best_dist = None
    for j, s in enumerate(sents):
        if not s.is_caption:
            continue
        if s.caption_target_type not in (None, target_type):
            continue
        dist = abs(j - idx)
        if best_dist is None or dist < best_dist:
            best, best_dist = s, dist
    return best


def _group_list_items(sents) -> list[dict]:
    """Group consecutive list-item sentences into logical list groups.

    Each group gets a union bbox and an ordered items list.
    Non-list sentences break the current group, so two separate
    runs of list-items produce two distinct list groups.

    Returns: [{"items": [{"text": str, "bbox": tuple}], "bbox": (page, x0, y0, x1, y1)}]
    """
    groups = []
    current_items = []

    for s in sents:
        if s.is_list_item:
            current_items.append(s)
        else:
            if current_items:
                groups.append(_finalize_list_group(current_items))
                current_items = []

    if current_items:
        groups.append(_finalize_list_group(current_items))

    return groups


def _finalize_list_group(items) -> dict:
    """Build a list group dict from consecutive list-item sentences."""
    entries = [{"text": s.text, "bbox": s.bbox} for s in items]
    # Union bbox with page (use first item's page)
    page = items[0].page_no
    x0 = min(s.bbox[0] for s in items)
    y0 = min(s.bbox[1] for s in items)
    x1 = max(s.bbox[2] for s in items)
    y1 = max(s.bbox[3] for s in items)
    return {"items": entries, "bbox": (page, x0, y0, x1, y1)}

