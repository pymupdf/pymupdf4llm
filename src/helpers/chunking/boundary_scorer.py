"""Step B: Compute boundary scores between adjacent SentenceUnits."""

import statistics

from .models import SentenceUnit, horizontal_overlap_ratio, caption_matches_element


class BoundaryScorer:
    """Scores the likelihood of a chunk boundary between adjacent sentence pairs.

    Layout-only: PDF-native signals (boxes, classes, gaps, fonts, structure
    hints). Higher score = more likely to break here.
    """

    def __init__(self, weights: dict):
        self.weights = weights
        self._median_gap: float = 12.0
        self._median_font: float = 10.0

    def score_all(self, sents: list[SentenceUnit]) -> list[float]:
        """Compute break scores for all adjacent pairs.

        Returns list of len(sents)-1 scores.
        """
        if len(sents) < 2:
            return []

        gaps = [s.line_gap_before for s in sents if s.line_gap_before is not None and s.line_gap_before > 0]
        self._median_gap = statistics.median(gaps) if gaps else 12.0

        sizes = [s.font_size_dominant for s in sents if s.font_size_dominant > 0]
        self._median_font = statistics.median(sizes) if sizes else 10.0

        scores = []
        for i in range(len(sents) - 1):
            scores.append(self.score_pair(sents[i], sents[i + 1]))

        return scores

    def score_pair(self, left: SentenceUnit, right: SentenceUnit) -> float:
        """Compute break score for a single adjacent pair."""
        w = self.weights
        score = 0.0

        # Layout signals
        box_changed = (left.box_index != right.box_index) or (left.page_no != right.page_no)
        score += w.get("w_box", 0) * float(box_changed)

        score += w.get("w_class", 0) * float(left.boxclass != right.boxclass)
        score += w.get("w_page", 0) * float(left.page_no != right.page_no)
        score += w.get("w_gap", 0) * self._compute_gap_jump(right)
        score += w.get("w_font", 0) * self._compute_font_jump(left, right)

        # Structure hint signals
        score += w.get("w_head", 0) * float(right.is_heading_hint and not left.is_heading_hint)
        score += w.get("w_foot", 0) * float(not left.is_footnote and right.is_footnote)
        score += w.get("w_hgap", 0) * self._compute_hgap_jump(left, right)

        # Suppression signals (negative = keep together)
        score -= w.get("w_list", 0) * float(left.is_list_item and right.is_list_item and not box_changed)
        score -= w.get("w_table", 0) * float(left.is_table_content and right.is_table_content)
        score -= w.get("w_caption", 0) * float(self._is_caption_adjacent(left, right))

        return score

    def _compute_gap_jump(self, right: SentenceUnit) -> float:
        """Normalized vertical gap signal (0.0-1.0)."""
        gap = right.line_gap_before
        if gap is None or self._median_gap <= 0:
            return 0.0
        ratio = gap / self._median_gap
        if ratio <= 1.5:
            return 0.0
        return min(1.0, (ratio - 1.5) / 2.0)

    def _compute_hgap_jump(self, left: SentenceUnit, right: SentenceUnit) -> float:
        """Horizontal non-overlap signal for multi-column detection (0.0-1.0)."""
        if left.page_no != right.page_no:
            return 0.0

        ratio = horizontal_overlap_ratio(left.bbox, right.bbox)
        if ratio >= 0.5:
            return 0.0
        # 0.5 → 0.0, 0.0 → 1.0
        return 1.0 - ratio * 2.0

    def _is_caption_adjacent(self, left: SentenceUnit, right: SentenceUnit) -> bool:
        """Check if a caption is adjacent to its target element."""
        if left.page_no != right.page_no:
            return False
        return caption_matches_element(left, right) or caption_matches_element(right, left)

    def _compute_font_jump(self, left: SentenceUnit, right: SentenceUnit) -> float:
        """Font size change signal (0.0-1.0)."""
        if left.font_size_dominant <= 0 or right.font_size_dominant <= 0:
            return 0.0

        diff = abs(left.font_size_dominant - right.font_size_dominant)
        if diff < 0.5:
            return 0.0

        if self._median_font > 0:
            ratio = diff / self._median_font
            return min(1.0, ratio)
        return min(1.0, diff / 10.0)
