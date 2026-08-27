"""Data models for the chunking pipeline.

ID formats are a public contract (stable across versions, snapshot-tested):

- element:  ``p{page}.b{box}``  (1-based original page number, 0-based box
  index on that page).  Element ids are scoped to (document bytes, package
  version, parse options): parse modes that re-split boxes (e.g. HTML table
  rendering) produce different box indices, so ids must not be cached
  across parse-option changes.
- chunk:    ``c{n}``
- table:    ``t{n}``
- figure:   ``f{n}``  (same numbering as ``[Figure f{n}: WxH]`` placeholders)
- section:  ``s{n}``
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional


def element_id(page: int, box: int) -> str:
    """Public element id: 'p{page}.b{box}'."""
    return f"p{page}.b{box}"


@dataclass
class SentenceUnit:
    """A sentence-level unit extracted from a LayoutBox.

    For table/figure boxes, this represents the entire box content as one unit.
    """
    sent_id: int
    text: str
    norm_text: str

    # Location
    page_no: int
    box_index: int
    boxclass: str
    bbox: tuple  # (x0, y0, x1, y1)

    # Token count (precomputed once by the pipeline; see CumulativeCounter)
    token_count: int = 0

    # Font metrics (from dominant span)
    font_size_dominant: float = 0.0
    font_flags_dominant: int = 0

    # Vertical spacing
    line_gap_before: Optional[float] = None
    line_gap_after: Optional[float] = None

    # Structure hints (derived from LayoutBox.boxclass)
    is_heading_hint: bool = False
    heading_level_hint: Optional[int] = None
    is_list_item: bool = False
    is_table_content: bool = False
    is_figure_related: bool = False
    is_footnote: bool = False
    is_header_footer: bool = False
    is_caption: bool = False
    caption_target_type: Optional[str] = None  # "table", "figure", None

    # Original table content (when is_table_content=True). Exactly one of
    # markdown/html is set, matching the parse mode (html mode carries
    # markdown=None); the unit's text holds the canonical rendering.
    table_markdown: Optional[str] = None
    table_html: Optional[str] = None

    # Document-wide figure sequence number (when is_figure_related=True)
    figure_number: Optional[int] = None

    # Image bytes (when is_figure_related=True and extract_images=True)
    image_data: Optional[bytes] = field(default=None, repr=False)

    # Tracks original box indices when multiple HF units are merged
    _source_box_indices: list = field(default_factory=list, repr=False)


@dataclass
class ProtoChunk:
    """An intermediate chunk before split/merge refinement.

    Holds a contiguous run of units; ``text`` is only materialized at
    serialization time (``unit_range`` + precomputed token sums keep
    assembly free of repeated joining/tokenizing).
    """
    chunk_id: int
    sent_ids: list = field(default_factory=list)
    text: str = ""
    token_count: int = 0
    unit_range: tuple = (0, 0)  # [start, end) over the working unit list

    # Location
    page_start: int = 0
    page_end: int = 0
    box_indices: list = field(default_factory=list)  # [(page_no, box_index), ...]
    bboxes: list = field(default_factory=list)  # list of (page, x0, y0, x1, y1)

    # Structure
    primary_type: Optional[str] = None  # primary type: paragraph, table, list, figure, ...
    types: list = field(default_factory=list)  # all types present, e.g. ["heading", "table"]

    # References to SentenceUnits (kept for split/merge)
    _sentences: list = field(default_factory=list, repr=False)


@dataclass(frozen=True)
class Element:
    """One layout box, chunk-addressable evidence with canonical text.

    The registry keeps every box — including header/footer boxes excluded
    from chunk text — so nothing the parser saw is unreachable (D8).
    """
    id: str        # element_id(page, box) — public contract
    page: int
    box: int
    boxclass: str
    bbox: tuple    # (x0, y0, x1, y1)
    text: str      # canonical markdown rendering ("" when nothing renders)
    is_header_footer: bool = False


@dataclass
class TableChunk:
    """A table exposed in the tables view; links back to its chunk.

    ``markdown`` and ``html`` are mutually exclusive per parse mode: in
    HTML table mode the parser carries markdown=None and html is the
    canonical rendering; otherwise markdown is canonical (possibly ``""``
    for a degenerate table) and html is None (D15).
    """
    id: str                    # "t{n}"
    chunk_id: Optional[str]    # owning chunk id ("c{n}"), None if unchunked
    element_id: str
    page: int
    bbox: tuple
    markdown: Optional[str] = None
    html: Optional[str] = None
    headers: list = field(default_factory=list)  # header cell texts (from <th>)
    caption: Optional[str] = None
    caption_element_id: Optional[str] = None
    section_id: Optional[str] = None             # "s{n}" of innermost section
    token_count: int = 0

    @property
    def text(self) -> str:
        """Canonical rendering for the parse mode."""
        if self.html is not None:
            return self.html
        return self.markdown or ""


@dataclass
class FigureChunk:
    """A figure/formula exposed in the figures view; links back to its chunk."""
    id: str                    # "f{n}"
    chunk_id: Optional[str]
    element_id: str
    page: int
    bbox: tuple                # re-render address for vision pipelines
    boxclass: str = "picture"
    ocr_text: Optional[str] = None  # text extracted inside the figure (None → OCR candidate)
    placeholder: str = ""      # "[Figure f{n}: WxH]" when the figure has no text
    caption: Optional[str] = None
    caption_element_id: Optional[str] = None
    section_id: Optional[str] = None
    image: Optional[bytes] = field(default=None, repr=False)

    @property
    def has_text(self) -> bool:
        return bool(self.ocr_text and self.ocr_text.strip())


@dataclass
class SectionChunk:
    """A section (heading span) exposed in the sections view."""
    id: str                    # "s{n}"
    title: str
    level: int
    page_start: int
    page_end: int
    heading_element_id: Optional[str] = None
    path: list = field(default_factory=list)       # titles, root → self
    element_span: tuple = (0, 0)                   # [start, end) into ChunkedDocument.elements
    child_chunk_ids: list = field(default_factory=list)  # chunks under this section
    token_count: int = 0                           # sum over child chunks

    _elements: tuple = field(default=(), repr=False, compare=False)

    @property
    def text(self) -> str:
        """Section body text, assembled lazily from the element registry."""
        return "\n\n".join(
            e.text for e in self._elements
            if e.text and not e.is_header_footer
        )


@dataclass
class SectionNode:
    """A node in ChunkedDocument.hierarchy (root has level 0, no section)."""
    title: str = ""
    level: int = 0
    section_id: Optional[str] = None
    children: list = field(default_factory=list)


@dataclass
class ChunkMetadata:
    """Metadata attached to a final chunk (payload-ready: every field is
    json-safe and lands in ``to_dicts()`` as-is)."""
    page_start: int = 0
    page_end: int = 0
    bboxes: list = field(default_factory=list)  # list of (page, x0, y0, x1, y1)
    types: list = field(default_factory=list)   # element types present, in order
    section_id: Optional[str] = None                 # "s{n}" of innermost section
    section_path: list = field(default_factory=list)  # human-facing citation path
    token_count: int = 0
    element_ids: list = field(default_factory=list)   # ["p{page}.b{box}", ...]

    # Bidirectional links into the document views
    table_ids: list = field(default_factory=list)    # ["t0", ...]
    figure_ids: list = field(default_factory=list)   # ["f1", ...]

    # Logical list groups within this chunk
    lists: list = field(default_factory=list)

    # Source pages went through OCR
    ocr: bool = False

    # Document provenance
    file_path: Optional[str] = None
    page_count: Optional[int] = None


_WS_RE = re.compile(r"\s+")


@dataclass
class Chunk:
    """A finalized, retrieval-ready chunk."""
    id: str                    # "c{n}" (budget-local: changes across reassemble_chunks)
    text: str
    tagged_content: str = ""
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)

    _content_hash: Optional[str] = field(default=None, repr=False, compare=False)

    @property
    def content_hash(self) -> str:
        """Lazy sha256 of whitespace-normalized text (D17).

        Stable across runs for identical content; whitespace-normalized so
        that byte-identical content is not reported as changed by
        rendering-neutral whitespace drift.
        """
        if self._content_hash is None:
            norm = _WS_RE.sub(" ", self.text).strip()
            self._content_hash = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        return self._content_hash


def caption_matches_element(caption, element) -> bool:
    """Check if a caption SentenceUnit matches an element SentenceUnit (one direction).

    Returns True if caption targets the element type, or if target_type is None
    and element is table/figure/list.
    """
    if not caption.is_caption:
        return False
    if not (element.is_table_content or element.is_figure_related or element.is_list_item):
        return False
    if caption.caption_target_type is None:
        return True
    if caption.caption_target_type == "table" and element.is_table_content:
        return True
    if caption.caption_target_type == "figure" and element.is_figure_related:
        return True
    return False


def horizontal_overlap_ratio(bbox_a: tuple, bbox_b: tuple) -> float:
    """Compute horizontal overlap ratio between two bboxes.

    Returns overlap / min_width, or 1.0 if either bbox has zero width.
    """
    ax0, _, ax1, _ = bbox_a
    bx0, _, bx1, _ = bbox_b
    min_width = min(ax1 - ax0, bx1 - bx0)
    if min_width <= 0:
        return 1.0
    overlap = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    return overlap / min_width


def union_bbox(bboxes) -> tuple:
    """Compute the union bounding box from an iterable of (x0, y0, x1, y1) tuples."""
    x0 = y0 = float('inf')
    x1 = y1 = float('-inf')
    for bx0, by0, bx1, by1 in bboxes:
        x0 = min(x0, bx0)
        y0 = min(y0, by0)
        x1 = max(x1, bx1)
        y1 = max(y1, by1)
    if x0 == float('inf'):
        return (0, 0, 0, 0)
    return (x0, y0, x1, y1)


def _has_shared_edge(bbox_a: tuple, bbox_b: tuple, tol: float) -> bool:
    """Check if two bboxes share a matching edge *pair*.

    Returns True when (x0 AND x1) or (y0 AND y1) both match within *tol*.
    A single edge match (e.g. only x1) is not enough — it would cause
    chain-reaction grouping across spatially unrelated boxes.
    """
    ax0, ay0, ax1, ay1 = bbox_a
    bx0, by0, bx1, by1 = bbox_b
    x_pair = abs(ax0 - bx0) <= tol and abs(ax1 - bx1) <= tol
    y_pair = abs(ay0 - by0) <= tol and abs(ay1 - by1) <= tol
    return x_pair or y_pair


def group_bboxes(bboxes, tolerance: float = 10.0) -> list[tuple]:
    """Group sequential bboxes that share a matching edge pair.

    Only the *last* (most recent) group is checked for each incoming box.
    If it doesn't match, a new group is created.  This respects reading
    order: once a spatially different region appears (e.g. an indented
    code block), subsequent boxes continue from that break rather than
    jumping back to an earlier group.

    *tolerance* defaults to 10 PDF points (~3.5 mm at 72 dpi).
    """
    boxes = list(bboxes)
    if not boxes:
        return []
    if len(boxes) == 1:
        return list(boxes)

    groups: list[list[tuple]] = [[boxes[0]]]
    for box in boxes[1:]:
        last_union = union_bbox(iter(groups[-1]))
        if _has_shared_edge(last_union, box, tolerance):
            groups[-1].append(box)
        else:
            groups.append([box])

    return [union_bbox(iter(g)) for g in groups]
