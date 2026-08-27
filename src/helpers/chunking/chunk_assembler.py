"""Steps C+D: Assemble initial chunks and refine via split/merge.

Token budgeting works on per-unit token counts precomputed once by the
pipeline and summed here (D12 error contract: the sum of unit counts may
differ from tokenizing the joined text by up to the number of joins).
Assembly never re-tokenizes or re-joins chunk text; ``ProtoChunk.text``
stays empty until serialization.
"""

from .models import SentenceUnit, ProtoChunk, union_bbox, group_bboxes, horizontal_overlap_ratio, caption_matches_element

_ELEMENT_TYPES = frozenset({"table", "figure", "list"})


class ChunkAssembler:
    """Assembles SentenceUnits into ProtoChunks based on boundary scores,
    then refines by splitting oversized and merging undersized chunks.
    """

    def __init__(
        self,
        max_tokens: int = 400,
        min_tokens: int = 120,
        threshold: float = 0.5,
        table_mode: str = "preserve",
        merge_small_chunks: bool = True,
        respect_section_starts: bool = True,
    ):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.threshold = threshold
        self.table_mode = table_mode
        self.merge_small = merge_small_chunks
        self.respect_section_starts = respect_section_starts

    # ── Step C: Initial assembly ─────────────────────────────────────

    def assemble(self, sents: list[SentenceUnit], scores: list[float]) -> list[ProtoChunk]:
        """Form initial chunks from sentences + boundary scores.

        A new chunk starts when:
        - break_score > threshold, OR
        - structural break (heading, type transition), OR
        - adding the next sentence would exceed max_tokens
        """
        if not sents:
            return []

        chunks = []
        current_sents = [sents[0]]
        current_tokens = sents[0].token_count

        for i, score in enumerate(scores):
            next_sent = sents[i + 1]

            # Break conditions (lazy: skip expensive checks if earlier one triggers)
            do_break = score > self.threshold
            if not do_break:
                do_break = self._is_structural_break(current_sents[-1], next_sent)
            if not do_break:
                do_break = current_tokens + next_sent.token_count > self.max_tokens

            # Never split sentences from the same box — they share a bbox
            if do_break and (next_sent.box_index == current_sents[-1].box_index
                             and next_sent.page_no == current_sents[-1].page_no):
                do_break = False

            if do_break:
                chunks.append(self._make_proto_chunk(len(chunks), current_sents))
                current_sents = [next_sent]
                current_tokens = next_sent.token_count
            else:
                current_sents.append(next_sent)
                current_tokens += next_sent.token_count

        # Flush remaining
        if current_sents:
            chunks.append(self._make_proto_chunk(len(chunks), current_sents))

        return chunks

    # ── Step D: Refine ───────────────────────────────────────────────

    def refine(self, chunks: list[ProtoChunk]) -> list[ProtoChunk]:
        """Refine chunks via split and multi-depth merge.

        1. Split  — break oversized chunks at sentence boundaries
        2. Semantic merge — heading+content, caption+element
        3. Budget merge  — greedily combine within max_tokens
        """
        # Step 1: Split oversized chunks
        split: list[ProtoChunk] = []
        for chunk in chunks:
            if chunk.token_count > self.max_tokens:
                split.extend(self._split_chunk(chunk))
            else:
                split.append(chunk)

        if not self.merge_small:
            return self._renumber(split)

        # Step 2: Semantic merge — preserve heading+content and caption+element pairs
        semantic = self._merge_semantic(split)

        # Step 3: Budget merge — greedily combine within max_tokens
        merged = self._merge_budget(semantic)
        return self._renumber(merged)

    # ── Structural break (used by assemble) ──────────────────────────

    def _is_structural_break(self, prev: SentenceUnit, next_sent: SentenceUnit) -> bool:
        """Check if there's a forced structural break between units."""
        # Caption-element pairs should NOT be structurally separated
        if self._is_caption_element_pair(prev, next_sent):
            return False
        # Heading followed by table/figure on same page → keep together
        if self._is_heading_element_pair(prev, next_sent):
            return False
        if next_sent.is_header_footer != prev.is_header_footer:
            return True
        if next_sent.is_table_content != prev.is_table_content:
            return True
        if next_sent.is_figure_related != prev.is_figure_related:
            return True
        if next_sent.is_heading_hint:
            return True
        return False

    @staticmethod
    def _is_heading_element_pair(prev: SentenceUnit, next_sent: SentenceUnit) -> bool:
        """Check if prev is a heading and next is its associated element."""
        if not prev.is_heading_hint:
            return False
        if prev.page_no != next_sent.page_no:
            return False
        if not (next_sent.is_table_content or next_sent.is_figure_related):
            return False
        return ChunkAssembler._has_horizontal_overlap(prev.bbox, next_sent.bbox)

    @staticmethod
    def _is_caption_element_pair(a: SentenceUnit, b: SentenceUnit) -> bool:
        """Check if a and b form a caption-element pair on the same page."""
        if a.page_no != b.page_no:
            return False
        if not ChunkAssembler._has_horizontal_overlap(a.bbox, b.bbox):
            return False
        return caption_matches_element(a, b) or caption_matches_element(b, a)

    # ── Split ────────────────────────────────────────────────────────

    def _split_chunk(self, chunk: ProtoChunk) -> list[ProtoChunk]:
        """Split an oversized chunk at sentence boundaries.

        Prefers splitting at box boundaries (where box_index changes)
        to avoid the same box's bbox appearing in two chunks.
        """
        sents = chunk._sentences
        if not sents:
            return [chunk]

        if chunk.primary_type == "table" and self.table_mode == "preserve":
            return [chunk]

        result = []
        current = []
        current_tokens = 0

        for sent in sents:
            if sent.token_count > self.max_tokens:
                if current:
                    result.append(self._make_proto_chunk(0, current))
                    current = []
                    current_tokens = 0
                result.append(self._make_proto_chunk(0, [sent]))
                continue

            if current_tokens + sent.token_count > self.max_tokens and current:
                # Prefer splitting at a box boundary: if the new sentence
                # shares box_index with the last sentence in current, move
                # those shared sentences to the next chunk.
                split_point = len(current)
                if current and sent.box_index == current[-1].box_index:
                    # Find where this box_index started in current
                    bi = sent.box_index
                    pg = sent.page_no
                    j = len(current) - 1
                    while j > 0 and current[j].box_index == bi and current[j].page_no == pg:
                        j -= 1
                    if j > 0:  # don't empty the entire current
                        split_point = j + 1

                keep = current[:split_point]
                carry = current[split_point:]
                if keep:
                    result.append(self._make_proto_chunk(0, keep))
                current = carry + [sent]
                current_tokens = sum(s.token_count for s in current)
            else:
                current.append(sent)
                current_tokens += sent.token_count

        if current:
            result.append(self._make_proto_chunk(0, current))

        return result

    # ── Semantic merge ───────────────────────────────────────────────

    def _merge_semantic(self, chunks: list[ProtoChunk]) -> list[ProtoChunk]:
        """Merge semantic pairs in a single forward pass.

        Pairs: heading + following content, caption + adjacent element.
        Semantic pairs always produce a single union bbox (not a list).
        """
        if len(chunks) <= 1:
            return chunks

        result: list[ProtoChunk] = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]

            if i + 1 < len(chunks) and self._is_semantic_pair(chunk, chunks[i + 1]):
                # Greedily consume all consecutive semantic partners
                combined = list(chunk._sentences)
                j = i + 1
                while j < len(chunks) and self._is_semantic_pair(
                    self._make_proto_chunk(0, combined), chunks[j]
                ):
                    combined.extend(chunks[j]._sentences)
                    j += 1
                pc = self._make_proto_chunk(0, combined)
                # Semantic pairs are tightly related — union per page
                if len(pc.bboxes) > 1:
                    pc.bboxes = _union_per_page(pc.bboxes)
                result.append(pc)
                i = j
            else:
                result.append(chunk)
                i += 1

        return result

    def _is_semantic_pair(self, a: ProtoChunk, b: ProtoChunk) -> bool:
        """Check if (a, b) form a semantic pair that should stay together."""
        if a.primary_type == "header_footer" or b.primary_type == "header_footer":
            return False
        if a.token_count + b.token_count > self.max_tokens:
            return False
        if abs(a.page_end - b.page_start) > 1:
            return False
        if not self._any_bbox_overlap(a.bboxes, b.bboxes):
            return False

        # heading + following content (any non-heading type)
        if a.primary_type == "heading" and b.primary_type != "heading":
            return True

        # consecutive list chunks → merge into one logical list
        if a.primary_type == "list" and b.primary_type == "list":
            return True

        # caption + element (caption first)
        a_is_caption = bool(a._sentences) and all(s.is_caption for s in a._sentences)
        if a_is_caption and b.primary_type in _ELEMENT_TYPES:
            return self._caption_target_matches(a, b.primary_type)

        # element + caption (element first)
        b_is_caption = bool(b._sentences) and all(s.is_caption for s in b._sentences)
        if a.primary_type in _ELEMENT_TYPES and b_is_caption:
            return self._caption_target_matches(b, a.primary_type)

        return False

    @staticmethod
    def _caption_target_matches(caption_chunk: ProtoChunk, element_type: str) -> bool:
        """Check if caption's target type matches the element type."""
        target_types = {
            s.caption_target_type for s in caption_chunk._sentences
            if s.caption_target_type
        }
        if not target_types:
            return True  # no specific target → match any element
        return element_type in target_types

    # ── Budget merge ─────────────────────────────────────────────────

    def _merge_budget(self, chunks: list[ProtoChunk]) -> list[ProtoChunk]:
        """Greedily merge adjacent chunks that fit within max_tokens.

        Preserves bbox structure from earlier phases: instead of rebuilding
        bboxes from scratch, concatenates the bbox lists of both chunks.
        This keeps semantic-merge union bboxes intact.
        """
        if len(chunks) <= 1:
            return chunks

        merged = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = merged[-1]
            curr = chunks[i]

            if self._can_budget_merge(prev, curr):
                combined_sents = prev._sentences + curr._sentences
                pc = self._make_proto_chunk(0, combined_sents)
                # Re-group at chunk-level bboxes (not sentence-level) so that
                # semantic-merge unions are preserved as atomic inputs while
                # compatible bboxes (matching x0+x1 or y0+y1 pair) still merge.
                pc.bboxes = _regroup_page_bboxes(prev.bboxes + curr.bboxes)
                merged[-1] = pc
            else:
                merged.append(curr)

        return merged

    def _can_budget_merge(self, a: ProtoChunk, b: ProtoChunk) -> bool:
        """Check if two adjacent chunks can be merged within budget.

        Spatial proximity (bbox overlap, page distance) is intentionally
        NOT checked here — budget merge is purely about filling token
        capacity with sequentially adjacent chunks.
        """
        if a.primary_type == "header_footer" or b.primary_type == "header_footer":
            return False
        if b.primary_type == "heading":
            return False
        # A chunk opening with a heading is a section start; merging it
        # backward would attach a section's head to the previous section's
        # tail. (primary_type is "heading" only for lone headings, so
        # the check above does not cover heading+content chunks — D18.)
        if (self.respect_section_starts and b._sentences
                and b._sentences[0].is_heading_hint):
            return False
        # table_mode="isolate": table chunks stay separate in budget merge
        # (semantic merge heading+table / caption+table is still allowed)
        if self.table_mode == "isolate":
            if a.primary_type == "table" or b.primary_type == "table":
                return False
        if a.token_count + b.token_count > self.max_tokens:
            return False
        return True

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _has_horizontal_overlap(bbox_a: tuple, bbox_b: tuple, min_ratio: float = 0.3) -> bool:
        """Check if two single bboxes have sufficient horizontal overlap."""
        return horizontal_overlap_ratio(bbox_a, bbox_b) >= min_ratio

    @staticmethod
    def _any_bbox_overlap(bboxes_a: list, bboxes_b: list, min_ratio: float = 0.3) -> bool:
        """Check if any bbox from list a overlaps horizontally with any from list b.

        Handles both 4-tuple (x0,y0,x1,y1) and 5-tuple (page,x0,y0,x1,y1).
        """
        for ba in bboxes_a:
            ba4 = ba[1:] if len(ba) == 5 else ba
            for bb in bboxes_b:
                bb4 = bb[1:] if len(bb) == 5 else bb
                if horizontal_overlap_ratio(ba4, bb4) >= min_ratio:
                    return True
        return False

    def _make_proto_chunk(self, chunk_id: int, sents: list[SentenceUnit]) -> ProtoChunk:
        """Create a ProtoChunk from a contiguous run of SentenceUnits.

        Text is NOT materialized here (serialization does that once);
        token_count is the sum of precomputed unit counts (D12).
        """
        primary_type, all_types = self._infer_chunk_types(sents)

        bboxes = _group_bboxes_by_page(sents)

        box_indices = list(dict.fromkeys(
            (s.page_no, s.box_index) for s in sents
        ))

        return ProtoChunk(
            chunk_id=chunk_id,
            sent_ids=[s.sent_id for s in sents],
            token_count=sum(s.token_count for s in sents),
            unit_range=(sents[0].sent_id, sents[-1].sent_id + 1),
            page_start=sents[0].page_no,
            page_end=sents[-1].page_no,
            box_indices=box_indices,
            bboxes=bboxes,
            primary_type=primary_type,
            types=all_types,
            _sentences=list(sents),
        )

    @staticmethod
    def _infer_chunk_types(sents: list[SentenceUnit]) -> tuple[str, list[str]]:
        """Collect all content types and determine the primary type.

        Returns (primary_type, all_types) where all_types preserves
        reading order and primary_type is the dominant element type.
        """
        # Collect types in reading order (deduplicated, stable)
        seen = set()
        all_types = []
        for s in sents:
            t = None
            if s.is_header_footer:
                t = "header_footer"
            elif s.is_table_content:
                t = "table"
            elif s.is_figure_related:
                t = "figure"
            elif s.is_list_item:
                t = "list"
            elif s.is_heading_hint:
                t = "heading"
            elif s.is_caption:
                t = "caption"
            elif s.is_footnote:
                t = "footnote"
            else:
                t = "paragraph"
            if t not in seen:
                seen.add(t)
                all_types.append(t)

        # Primary type: first element type wins, then heading, then paragraph
        _ELEMENT_PRIORITY = ("table", "figure", "list")
        for et in _ELEMENT_PRIORITY:
            if et in seen:
                return et, all_types
        if "header_footer" in seen:
            return "header_footer", all_types
        if "heading" in seen and len(sents) == 1:
            return "heading", all_types
        if "footnote" in seen:
            return "footnote", all_types
        return "paragraph", all_types

    @staticmethod
    def _renumber(chunks: list[ProtoChunk]) -> list[ProtoChunk]:
        """Renumber chunk IDs sequentially."""
        for i, chunk in enumerate(chunks):
            chunk.chunk_id = i
        return chunks


def _group_bboxes_by_page(sents) -> list[tuple]:
    """Group sentence bboxes by page, then by edge-pair within each page.

    Consecutive list-items are pre-merged into a single union bbox before
    edge-pair grouping, since individual list-item x1 varies by text length
    while they share the same x0 (indentation level).

    Returns list of (page, x0, y0, x1, y1) 5-tuples.
    """
    from collections import defaultdict

    # Pre-merge consecutive list-items into union bboxes
    merged_entries = []  # (page_no, bbox)
    list_run = []
    for s in sents:
        if s.is_list_item:
            list_run.append(s)
        else:
            if list_run:
                merged_entries.extend(_flush_list_run(list_run))
                list_run = []
            merged_entries.append((s.page_no, s.bbox))
    if list_run:
        merged_entries.extend(_flush_list_run(list_run))

    # Group by page, then edge-pair
    page_bboxes = defaultdict(list)
    for page_no, bbox in merged_entries:
        page_bboxes[page_no].append(bbox)

    result = []
    for page_no in sorted(page_bboxes.keys()):
        grouped = group_bboxes(iter(page_bboxes[page_no]))
        for bbox in grouped:
            result.append((page_no, *bbox))
    return result


def _flush_list_run(items) -> list[tuple]:
    """Union consecutive list-item sentences into one bbox per page."""
    from collections import defaultdict
    by_page = defaultdict(list)
    for s in items:
        by_page[s.page_no].append(s.bbox)
    result = []
    for page_no in sorted(by_page.keys()):
        bboxes = by_page[page_no]
        x0 = min(b[0] for b in bboxes)
        y0 = min(b[1] for b in bboxes)
        x1 = max(b[2] for b in bboxes)
        y1 = max(b[3] for b in bboxes)
        result.append((page_no, (x0, y0, x1, y1)))
    return result


def _union_per_page(page_bboxes: list[tuple]) -> list[tuple]:
    """Union all bboxes on each page into a single bbox per page."""
    from collections import defaultdict
    by_page = defaultdict(list)
    for pb in page_bboxes:
        page, x0, y0, x1, y1 = pb
        by_page[page].append((x0, y0, x1, y1))

    result = []
    for page_no in sorted(by_page.keys()):
        u = union_bbox(iter(by_page[page_no]))
        result.append((page_no, *u))
    return result


def _regroup_page_bboxes(page_bboxes: list[tuple]) -> list[tuple]:
    """Re-group 5-tuple bboxes (page, x0, y0, x1, y1) by page.

    Preserves page info while merging compatible bboxes within each page.
    """
    from collections import defaultdict
    by_page = defaultdict(list)
    for pb in page_bboxes:
        page, x0, y0, x1, y1 = pb
        by_page[page].append((x0, y0, x1, y1))

    result = []
    for page_no in sorted(by_page.keys()):
        grouped = group_bboxes(iter(by_page[page_no]))
        for bbox in grouped:
            result.append((page_no, *bbox))
    return result
