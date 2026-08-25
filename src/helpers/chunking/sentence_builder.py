"""Step A: Convert LayoutBox textlines into SentenceUnits."""

import re
from collections import Counter, defaultdict
from typing import Optional

from .models import SentenceUnit, union_bbox
from .text_source import (
    box_to_markdown,
    create_list_item_levels,
    figure_placeholder,
    table_content,
)

# Sentence-ending patterns
_SENT_END_EN = re.compile(
    r'(?<=[.!?])'        # after sentence-ending punctuation
    r'(?:\s*["\'\)\]]*)'  # optional closing quotes/brackets
    r'\s+'                # followed by whitespace
    r'(?=[A-Z"\'\(\[])'   # before uppercase or opening quote/bracket
)

_SENT_END_MULTI = re.compile(
    r'(?<=[.!?。！？])'
    r'(?:\s*["\'\)\]」』]*)'
    r'\s*'
    r'(?=[A-Z가-힣ㄱ-ㅎㅏ-ㅣ一-鿿"\'\(\[「『]|$)'
)

# Whitespace normalization
_MULTI_SPACE = re.compile(r'[ \t]+')

# Leading markdown decoration (heading marks, styles, quotes)
_MD_DECORATION = re.compile(r'^[#>*_`\s]+')

# Caption detection patterns
_CAPTION_PATTERNS = [
    re.compile(r'^(?:figure|fig\.?\s*)\s*\d+(?:[.\-]\d+)*', re.IGNORECASE),
    re.compile(r'^(?:table|tbl\.?\s*)\s*\d+(?:[.\-]\d+)*', re.IGNORECASE),
    re.compile(r'^\(\s*[a-z0-9]\s*\)', re.IGNORECASE),
    re.compile(r'^(?:source|notes?)\s*:', re.IGNORECASE),
]


def _detect_caption(text: str) -> tuple[bool, Optional[str]]:
    """Detect if text is a caption and determine target type.

    Returns (is_caption, target_type) where target_type is "figure", "table", or None.
    """
    stripped = text.strip()
    if not stripped:
        return False, None

    for pat in _CAPTION_PATTERNS:
        if pat.search(stripped):
            low = stripped.lower()
            if low.startswith(("figure", "fig")):
                return True, "figure"
            if low.startswith(("table", "tbl")):
                return True, "table"
            return True, None

    return False, None


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace."""
    t = text.strip().lower()
    t = _MULTI_SPACE.sub(' ', t)
    return t


def _strip_md_decoration(text: str) -> str:
    """Strip leading markdown decoration for pattern matching (captions, TOC)."""
    return _MD_DECORATION.sub('', text)


def _get_dominant_font(textlines: list[dict]) -> tuple[float, int]:
    """Get the most common (font_size, font_flags) from spans by character count."""
    font_counter = Counter()
    for tl in textlines:
        for span in tl.get("spans", []):
            text = span.get("text", "")
            char_count = len(text.strip())
            if char_count > 0:
                size = round(span.get("size", 0), 1)
                flags = span.get("flags", 0)
                font_counter[(size, flags)] += char_count

    if not font_counter:
        return 0.0, 0

    (size, flags), _ = font_counter.most_common(1)[0]
    return size, flags


def _compute_bbox_union(textlines: list[dict]) -> tuple:
    """Compute the union bounding box of all textlines."""
    def _iter_bboxes():
        for tl in textlines:
            bbox = tl.get("bbox")
            if bbox is None:
                continue
            try:
                yield (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            except (TypeError, IndexError):
                continue
    return union_bbox(_iter_bboxes())


def _boxclass_to_hints(boxclass: str, toc=None, page_no=None, text=None,
                       header_level=None):
    """Map LayoutBox.boxclass to structure hint flags.

    Returns only non-default values (SentenceUnit defaults handle the rest).

    Heading depth comes first from the engine's font-statistics assignment
    (LayoutBox.header_level, the same value the markdown renderers use for
    #/##/### depth), so the section hierarchy nests the way the rendered
    text does. The TOC match is a fallback for boxes the engine left
    unleveled.
    """
    if boxclass == "title":
        return {"is_heading_hint": True,
                "heading_level_hint": header_level or 1}
    if boxclass == "section-header":
        level = header_level or (
            _match_toc_level(toc, page_no, text) if toc and text else None)
        return {"is_heading_hint": True, "heading_level_hint": level or 2}
    if boxclass == "list-item":
        return {"is_list_item": True}
    if boxclass in ("table", "table-fallback"):
        return {"is_table_content": True}
    if boxclass in ("picture", "formula"):
        return {"is_figure_related": True}
    if boxclass == "footnote":
        return {"is_footnote": True}
    if boxclass in ("page-header", "page-footer"):
        return {"is_header_footer": True}
    if boxclass == "caption":
        hints = {"is_caption": True}
        if text:
            _, target = _detect_caption(text)
            if target is not None:
                hints["caption_target_type"] = target
        return hints
    return {}


def _match_toc_level(toc: list, page_no: int, text: str) -> Optional[int]:
    """Try to match text against TOC entries for the given page to determine heading level."""
    if not toc or not text:
        return None

    norm = _normalize_text(text)
    if not norm:
        return None

    for entry in toc:
        if len(entry) < 3:
            continue
        level, title, toc_page = entry[0], entry[1], entry[2]
        if toc_page != page_no:
            continue
        toc_norm = _normalize_text(str(title))
        if not toc_norm:
            continue
        if norm.startswith(toc_norm) or toc_norm.startswith(norm):
            return level

    return None


class SentenceBuilder:
    """Converts ParsedDocument's LayoutBoxes into SentenceUnits."""

    def __init__(self, splitter: str = "default"):
        self.splitter = splitter
        self._figure_count = 0
        if splitter == "multilingual":
            self._split_re = _SENT_END_MULTI
        else:
            self._split_re = _SENT_END_EN

    def build_from_document(self, doc) -> list[SentenceUnit]:
        """Build SentenceUnits from all pages/boxes in a ParsedDocument."""
        units = []
        sent_id = 0
        prev_unit = None
        self._figure_count = 0

        for page in doc.pages:
            list_item_levels = create_list_item_levels(page.boxes)
            for box_idx, box in enumerate(page.boxes):
                new_units = self._process_box(
                    page=page,
                    box_idx=box_idx,
                    box=box,
                    sent_id_start=sent_id,
                    toc=doc.toc,
                    prev_unit=prev_unit,
                    list_item_levels=list_item_levels,
                )
                for u in new_units:
                    units.append(u)
                    sent_id += 1
                    prev_unit = u

        # Merge same-page header/footer units that share a y-band
        units = self._merge_same_page_hf_units(units)
        units = self._renumber_sent_ids(units)

        for i in range(len(units) - 1):
            if units[i + 1].line_gap_before is not None and units[i].page_no == units[i + 1].page_no:
                units[i].line_gap_after = units[i + 1].line_gap_before

        return units

    def _process_box(self, page, box_idx, box, sent_id_start, toc, prev_unit,
                     list_item_levels=None) -> list[SentenceUnit]:
        """Process a single LayoutBox into SentenceUnits."""
        boxclass = box.boxclass

        if boxclass in ("table", "table-fallback"):
            return self._table_as_unit(page, box_idx, box, sent_id_start, toc)

        if boxclass in ("picture", "formula"):
            # One document-wide figure number per figure box; it is the "f{n}"
            # id in the figures view and in text placeholders.
            self._figure_count += 1
            if not box.textlines:
                return self._figure_as_unit(page, box_idx, box, sent_id_start, toc)
            units = self._split_sentences(page, box_idx, box, sent_id_start, toc,
                                          prev_unit, list_item_levels)
            for u in units:
                u.figure_number = self._figure_count
            return units

        if box.textlines:
            return self._split_sentences(page, box_idx, box, sent_id_start, toc,
                                         prev_unit, list_item_levels)

        return []

    def _table_as_unit(self, page, box_idx, box, sent_id_start, toc) -> list[SentenceUnit]:
        """Create a single SentenceUnit for a table box."""
        table_md, table_html = table_content(box)
        text = box_to_markdown(page, box, box_idx).strip()
        if not text:
            return []

        hints = _boxclass_to_hints(box.boxclass, toc, page.page_number, text,
                                    header_level=getattr(box, 'header_level', None))
        return [SentenceUnit(
            sent_id=sent_id_start,
            text=text,
            norm_text=_normalize_text(text),
            page_no=page.page_number,
            box_index=box_idx,
            boxclass=box.boxclass,
            bbox=(box.x0, box.y0, box.x1, box.y1),
            table_markdown=table_md or None,
            table_html=table_html,
            **hints,
        )]

    def _figure_as_unit(self, page, box_idx, box, sent_id_start, toc) -> list[SentenceUnit]:
        """Create a single SentenceUnit for a figure/formula without text."""
        text = figure_placeholder(self._figure_count, box)
        hints = _boxclass_to_hints(box.boxclass, toc, page.page_number, text,
                                    header_level=getattr(box, 'header_level', None))
        image_data = box.image if isinstance(box.image, bytes) else None
        return [SentenceUnit(
            sent_id=sent_id_start,
            text=text,
            norm_text=_normalize_text(text),
            page_no=page.page_number,
            box_index=box_idx,
            boxclass=box.boxclass,
            bbox=(box.x0, box.y0, box.x1, box.y1),
            figure_number=self._figure_count,
            image_data=image_data,
            **hints,
        )]

    def _split_sentences(self, page, box_idx, box, sent_id_start, toc, prev_unit,
                         list_item_levels=None) -> list[SentenceUnit]:
        """Split a box's rendered markdown into sentence-level SentenceUnits."""
        textlines = box.textlines
        if not textlines:
            return []

        # Canonical text: the same markdown to_markdown emits for this box.
        # The renderers join spans with proper spacing and resolve hyphenation,
        # so no line-break collapsing is needed here.
        joined = box_to_markdown(page, box, box_idx, list_item_levels).strip()
        if not joined:
            return []

        font_size, font_flags = _get_dominant_font(textlines)
        first_gap = None
        if prev_unit and prev_unit.page_no == page.page_number:
            if textlines[0].get("bbox") is not None:
                first_gap = float(textlines[0]["bbox"][1]) - prev_unit.bbox[3]

        # Match patterns against text without markdown decoration
        plain = _strip_md_decoration(joined)
        hints = _boxclass_to_hints(box.boxclass, toc, page.page_number, plain,
                                    header_level=getattr(box, 'header_level', None))

        # Detect captions in non-caption boxclass text boxes
        if not hints.get("is_caption") and box.boxclass not in (
            "title", "section-header", "table", "table-fallback",
            "picture", "formula", "footnote", "page-header", "page-footer",
        ):
            is_cap, cap_target = _detect_caption(plain)
            if is_cap:
                hints["is_caption"] = True
                hints["caption_target_type"] = cap_target

        # Check if this should be kept as a single unit (no sentence splitting).
        # Multi-block markdown (code blocks, multi-item lists, multi-note
        # footnotes) contains newlines and must never be sentence-split.
        keep_single = (
            hints.get("is_heading_hint") or hints.get("is_footnote")
            or hints.get("is_header_footer") or hints.get("is_caption")
            or hints.get("is_list_item")
            or "\n" in joined
        )

        if not keep_single:
            sentences = self._split_re.split(joined)
            sentences = [s.strip() for s in sentences if s.strip()]
            if not sentences:
                return []
            keep_single = len(sentences) == 1

        if keep_single:
            bbox = _compute_bbox_union(textlines)
            return [SentenceUnit(
                sent_id=sent_id_start,
                text=joined,
                norm_text=_normalize_text(joined),
                page_no=page.page_number,
                box_index=box_idx,
                boxclass=box.boxclass,
                bbox=bbox,
                font_size_dominant=font_size,
                font_flags_dominant=font_flags,
                line_gap_before=first_gap,
                **hints,
            )]

        box_bbox = _compute_bbox_union(textlines)
        units = []
        for i, sent_text in enumerate(sentences):
            units.append(SentenceUnit(
                sent_id=sent_id_start + i,
                text=sent_text,
                norm_text=_normalize_text(sent_text),
                page_no=page.page_number,
                box_index=box_idx,
                boxclass=box.boxclass,
                bbox=box_bbox,  # approximate: use whole box bbox
                font_size_dominant=font_size,
                font_flags_dominant=font_flags,
                line_gap_before=first_gap if i == 0 else None,
                **hints,
            ))

        return units

    def detect_repeated_headers_footers(self, doc) -> set[tuple[int, int]]:
        """Detect header/footer boxes that repeat across pages.

        Returns set of (page_no, box_index) tuples to exclude.
        """
        if not doc.pages or len(doc.pages) < 3:
            return set()

        hf_texts = defaultdict(list)  # (boxclass, y_bucket, text_norm) -> [(page_no, box_idx)]

        for page in doc.pages:
            for box_idx, box in enumerate(page.boxes):
                if box.boxclass in ("page-header", "page-footer") and box.textlines:
                    text = _normalize_text(box_to_markdown(page, box, box_idx))
                    if text:
                        y_bucket = round(box.y0 / 10) * 10  # bucket by ~10pt
                        hf_texts[(box.boxclass, y_bucket, text)].append((page.page_number, box_idx))

        threshold = max(2, len(doc.pages) * 0.5)
        repeated = set()

        for _key, locations in hf_texts.items():
            if len(locations) >= threshold:
                repeated.update(locations)

        return repeated

    # ── Header/Footer same-line merge ─────────────────────────────────

    def _merge_same_page_hf_units(self, units: list[SentenceUnit],
                                   y_tolerance: float = 20.0) -> list[SentenceUnit]:
        """Merge consecutive same-page, same-boxclass HF units sharing a y-band."""
        if not units:
            return units

        result = []
        i = 0
        while i < len(units):
            u = units[i]
            if not u.is_header_footer:
                result.append(u)
                i += 1
                continue

            # Collect consecutive HF units on the same page with the same boxclass
            group = [u]
            j = i + 1
            while j < len(units):
                nxt = units[j]
                if (nxt.is_header_footer
                        and nxt.page_no == u.page_no
                        and nxt.boxclass == u.boxclass):
                    group.append(nxt)
                    j += 1
                else:
                    break

            if len(group) == 1:
                result.append(u)
            else:
                # Group by y-band and merge each band
                bands = self._group_by_y_band(group, y_tolerance)
                for band in bands:
                    result.append(self._merge_hf_group(band))
            i = j

        return result

    @staticmethod
    def _group_by_y_band(units: list[SentenceUnit],
                         y_tolerance: float) -> list[list[SentenceUnit]]:
        """Group units by y-center proximity (greedy clustering)."""
        sorted_units = sorted(units, key=lambda u: (u.bbox[1] + u.bbox[3]) / 2)
        bands: list[list[SentenceUnit]] = []
        for u in sorted_units:
            yc = (u.bbox[1] + u.bbox[3]) / 2
            if bands:
                last_yc = (bands[-1][-1].bbox[1] + bands[-1][-1].bbox[3]) / 2
                if abs(yc - last_yc) <= y_tolerance:
                    bands[-1].append(u)
                    continue
            bands.append([u])
        return bands

    @staticmethod
    def _merge_hf_group(group: list[SentenceUnit]) -> SentenceUnit:
        """Merge a group of HF units into a single unit (left-to-right order)."""
        if len(group) == 1:
            return group[0]

        sorted_group = sorted(group, key=lambda u: u.bbox[0])  # sort by x0
        text = " | ".join(u.text for u in sorted_group)

        bbox = union_bbox(u.bbox for u in sorted_group)

        # Use font info from the longest-text unit
        longest = max(sorted_group, key=lambda u: len(u.text))

        # Collect all source box indices
        source_indices = []
        for u in sorted_group:
            source_indices.append(u.box_index)
            source_indices.extend(u._source_box_indices)

        merged = SentenceUnit(
            sent_id=sorted_group[0].sent_id,
            text=text,
            norm_text=_normalize_text(text),
            page_no=sorted_group[0].page_no,
            box_index=sorted_group[0].box_index,
            boxclass=sorted_group[0].boxclass,
            bbox=bbox,
            font_size_dominant=longest.font_size_dominant,
            font_flags_dominant=longest.font_flags_dominant,
            line_gap_before=sorted_group[0].line_gap_before,
            is_header_footer=True,
        )
        merged._source_box_indices = source_indices
        return merged

    @staticmethod
    def _renumber_sent_ids(units: list[SentenceUnit]) -> list[SentenceUnit]:
        """Renumber sent_ids sequentially from 0."""
        for i, u in enumerate(units):
            u.sent_id = i
        return units
