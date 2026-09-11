"""Grid-gap repair for tables detected via PyMuPDF's Layout model.

The Layout model's table grid can under-count rows or columns when two
adjacent cells' text visually fuses during RAWDICT extraction (e.g. a
wrapped multi-line label sitting flush against its neighbor with no ruled
line between them). This module re-clusters the text inside each INTERIOR
grid gap, purely geometrically -- never by parsing header text or column
semantics, so the detector works the same for any language -- and inserts
any additional boundary it can corroborate with independent evidence.

"Language-agnostic" means the detector never looks at what the text SAYS
(no header-keyword matching, no locale-specific parsing) -- only at where
text sits and how many lines it wraps to, so the same logic applies
uniformly regardless of the document's language. The "corroboration gate"
(see `evaluate_gap`) is the requirement that at least two independent
cross-axis strips within a gap show >=2 bands before the gap is treated as
a real fusion defect; a single strip with many bands while every other
strip has at most one band is instead the "one column/row legitimately
wraps to many lines" false-positive signature, not a missing boundary, so
it is rejected.

`evaluate_gap` also rejects any candidate boundary that would fall
strictly inside a single-band strip's own band -- a strip with exactly
one band across the gap has one real, unambiguous physical line there,
and cutting through it would bisect that line's text rather than
separate two rows/columns (found via a real document's WP-3-3 table
header: two neighboring header cells legitimately wrapped to 2-3 lines
each, corroborating a boundary that fell mid-line through a third,
single-line header cell -- "Kategória výdavkov" -- splicing its
characters across both sides of the inserted line). Multi-band strips are
exempt from this check, since their own bands are already known to not
align exactly with every candidate (see the second known limitation
below).

Two known, deliberately accepted residual limitations remain:

  - Any row where two DIFFERENT columns independently wrap to the same
    line count at a coincidentally matching y-offset, with no OTHER
    column's single unwrapped line straddling that offset, can still pass
    the corroboration gate and the single-band check above, producing a
    cosmetic false-positive split (no data loss, since nothing gets
    bisected -- just an extra row/column boundary with plausible-looking
    but ultimately spurious content on each side).
  - When a header row's own band count is HIGHER than the data rows'
    minimum band count (e.g. one header-only column that is empty in every
    data row), the header-only boundary is not sourced, since the
    reference-strip rule below only trusts minimum-band strips -- the
    repair can be a safe partial fix rather than fully correct.
"""
import logging
import statistics

import pymupdf

logger = logging.getLogger(__name__)

from pymupdf4llm.helpers.get_text_lines import get_raw_lines

CONSISTENCY_TOL = 6.0


def _cluster_rawdict_lines(table_blocks, clip):
    """Cluster RAWDICT-format text (char-level, used for exact cell-boundary
    extraction elsewhere) into visual lines via get_raw_lines(), which
    expects DICT-shaped spans (a "text" field; RAWDICT spans have "chars"
    instead). Builds fresh block/line/span dict copies scoped to `clip` --
    never mutates table_blocks itself, since that list is shared and reused
    for every cell's char-level text extraction across the whole table.
    """
    converted_blocks = []
    for block in table_blocks:
        if not pymupdf.Rect(block["bbox"]).intersects(clip):
            continue
        new_lines = []
        for line in block["lines"]:
            if not pymupdf.Rect(line["bbox"]).intersects(clip):
                continue
            new_spans = []
            for span in line["spans"]:
                if not pymupdf.Rect(span["bbox"]).intersects(clip):
                    continue
                text = "".join(c["c"] for c in span.get("chars", ()))
                if not text:
                    continue
                new_spans.append({**span, "text": text})
            if new_spans:
                new_lines.append({**line, "spans": new_spans})
        if new_lines:
            converted_blocks.append({**block, "lines": new_lines})
    if not converted_blocks:
        return []
    return get_raw_lines(
        textpage=None,
        blocks=converted_blocks,
        clip=clip,
        require_x_continuity=True,
    )


def merge_overlapping(rects, axis):
    """Lines from get_raw_lines can be split apart purely because of an
    x-discontinuity (require_x_continuity=True) even when they sit at
    (almost) the same y -- or, symmetrically, a single wrapped-text line's
    sub-lines all start near the same x. Neither case is a real band along
    `axis`; merge any rects whose extent along that axis overlaps before
    counting bands."""
    if not rects:
        return []
    lo_attr, hi_attr = ("y0", "y1") if axis == "row" else ("x0", "x1")
    rects = sorted(rects, key=lambda r: getattr(r, lo_attr))
    merged = [rects[0]]
    for r in rects[1:]:
        last = merged[-1]
        if getattr(r, lo_attr) < getattr(last, hi_attr) + 0.5:
            merged[-1] = last | r
        else:
            merged.append(r)
    return merged


def band_rects(table_blocks, clip, axis):
    lines = _cluster_rawdict_lines(table_blocks, clip)
    rects = [rect for rect, _spans in lines]
    return merge_overlapping(rects, axis)


def boundaries_of(strip, axis):
    out = []
    for a, b in zip(strip[:-1], strip[1:]):
        out.append((a.y1 + b.y0) / 2.0 if axis == "row" else (a.x1 + b.x0) / 2.0)
    return out


def evaluate_gap(table_blocks, axis, gap_lo, gap_hi, cross_lines):
    """Evaluate one interior grid gap (a row gap between two h_lines, or a
    column gap between two v_lines) for a missing-boundary defect.

    `cross_lines` are the grid lines on the OTHER axis (v_lines for a row
    gap, h_lines for a column gap); each pair of adjacent cross_lines
    defines one narrow strip clipped out of the gap and evaluated
    independently."""
    strips = []
    for lo, hi in zip(cross_lines[:-1], cross_lines[1:]):
        clip = (
            pymupdf.Rect(lo, gap_lo, hi, gap_hi)
            if axis == "row"
            else pymupdf.Rect(gap_lo, lo, gap_hi, hi)
        )
        strips.append(band_rects(table_blocks, clip, axis))

    band_counts = [len(s) for s in strips]
    multi = [s for s, n in zip(strips, band_counts) if n >= 2]

    # Corroboration gate: a single strip with many bands while every other
    # strip has at most one band is the "one column/row legitimately
    # wraps to many lines" false-positive signature -- not a real fusion.
    # Require at least two independent strips to show >=2 bands before
    # treating the gap as a candidate defect.
    if len(multi) < 2:
        return {"is_defect": False, "band_count": max(band_counts, default=0), "new_boundaries": []}

    # Text wrapping only ever ADDS bands to a strip, it never removes real
    # distinguishing bands -- so the strip(s) with the fewest (but still
    # >=2) bands among the corroborating set are the least polluted by
    # wrap artifacts. Source candidate boundary positions only from those
    # minimum-count strips.
    min_bands = min(len(s) for s in multi)
    reference_strips = [s for s in multi if len(s) == min_bands]
    candidates = sorted(c for s in reference_strips for c in boundaries_of(s, axis))

    clusters = [[candidates[0]]]
    support = [1]
    for c in candidates[1:]:
        if c - clusters[-1][-1] <= CONSISTENCY_TOL:
            clusters[-1].append(c)
            support[-1] += 1
        else:
            clusters.append([c])
            support.append(1)

    # A strip with exactly one band is a column/row that does NOT wrap here --
    # its band is one real, unambiguous physical line. A candidate boundary
    # that falls strictly inside such a band would bisect that line (the
    # WP-3-3 header regression: a short label like "Kategória výdavkov" sits
    # as a single line straddling the midpoint two OTHER columns' wrapped
    # text corroborated on). Multi-band strips are exempt from this check --
    # their bands are already known to not align exactly with every
    # candidate (see the "partial fix" residual limitation in the module
    # docstring), so requiring alignment there would defeat real corroborated
    # repairs (see test_evaluate_gap_reference_strip_rule_ignores_higher_band_count_strip).
    single_bands = [s[0] for s in strips if len(s) == 1]
    lo_attr, hi_attr = ("y0", "y1") if axis == "row" else ("x0", "x1")

    def _bisects_a_single_line(candidate):
        return any(
            getattr(b, lo_attr) < candidate < getattr(b, hi_attr) for b in single_bands
        )

    kept = [
        (statistics.mean(cl), sup)
        for cl, sup in zip(clusters, support)
        if not _bisects_a_single_line(statistics.mean(cl))
    ]
    if not kept:
        return {"is_defect": False, "band_count": max(band_counts, default=0), "new_boundaries": []}

    new_boundaries = [b for b, _ in kept]
    kept_support = [s for _, s in kept]

    return {
        "is_defect": True,
        "band_count": max(band_counts, default=0),
        "new_boundaries": new_boundaries,
        "support": kept_support,
    }


def repair_grid_gaps(table_blocks, h_lines, v_lines):
    """Evaluate every interior row and column gap of a table's detected
    grid and insert any additional boundaries this pass finds corroborating
    geometric evidence for. Returns (new_h_lines, new_v_lines) -- both
    sorted, with the outer edges (h_lines[0]/[-1], v_lines[0]/[-1])
    unchanged, since this only repairs INTERIOR gaps."""
    table_rect = pymupdf.Rect(v_lines[0], h_lines[0], v_lines[-1], h_lines[-1])
    table_blocks = [
        b for b in table_blocks if pymupdf.Rect(b["bbox"]).intersects(table_rect)
    ]

    new_h = list(h_lines)
    for i in range(len(h_lines) - 1):
        result = evaluate_gap(table_blocks, "row", h_lines[i], h_lines[i + 1], v_lines)
        if result["is_defect"]:
            logger.debug(
                "repaired row gap idx=%d band_count=%d new_boundaries=%s support=%s",
                i, result["band_count"], result["new_boundaries"], result.get("support"),
            )
            new_h.extend(result["new_boundaries"])

    new_v = list(v_lines)
    for i in range(len(v_lines) - 1):
        result = evaluate_gap(table_blocks, "col", v_lines[i], v_lines[i + 1], h_lines)
        if result["is_defect"]:
            logger.debug(
                "repaired col gap idx=%d band_count=%d new_boundaries=%s support=%s",
                i, result["band_count"], result["new_boundaries"], result.get("support"),
            )
            new_v.extend(result["new_boundaries"])

    new_h.sort()
    new_v.sort()
    return new_h, new_v
