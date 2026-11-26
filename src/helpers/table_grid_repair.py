"""Grid-gap repair for tables detected via PyMuPDF's Layout model.

The Layout model's table grid can under-count rows or columns: two table
rows (or columns) end up inside one grid cell band, and their text is then
merged into single cells ("0,00<br>0,00"). This module inserts the missing
boundaries, purely from where the table's text sits -- never from what it
says, so it behaves the same for any language.

A boundary is only inserted into a stretch of whitespace inside one grid
row (or column) that satisfies all of these:

  - No character of the table crosses it, in ANY cell of that row (or
    column). A boundary therefore never cuts through a line of text, and
    every character ends up in exactly one cell.
  - It is wider than ordinary line spacing (rows) or word spacing
    (columns), so the lines of one wrapped cell are never pulled apart:
    wrapped lines sit about one line pitch apart with almost no gap
    between their boxes, while separate table rows also have the cells'
    padding between them.
  - At least two distinct cells have text on both sides of it. One cell
    with several lines, or several runs of text, is never enough.
  - For a column boundary only: the topmost cell with any text in that
    column (normally its header) also has text on both sides. A column
    whose values are split into two runs -- an accounting-style currency
    column with a left-aligned symbol and a right-aligned amount -- has a
    single header above it and is left alone.

The text is indexed once per table: every character is assigned to the
grid cell containing its centre, and each cell's text is reduced to the
intervals it occupies along each axis.

Known, accepted limitations: rows that were laid out with no more space
between them than between wrapped lines look exactly like one wrapped row
and are not split; and a column split without a header row above it (or
whose header spans both parts) is not repaired.
"""

import bisect
import itertools
import statistics
from collections import defaultdict

# A gap between two rows of text must be at least this fraction of the
# table's typical line height. Wrapped lines inside one cell are typically
# separated by 0-20% of their height; separate table rows by more.
ROW_GAP_MIN_FRACTION = 0.4

# Characters on one line closer than max(COLUMN_GAP_MIN, COLUMN_GAP_EM *
# font size) belong to the same run of text. Word spacing, even in
# justified text, stays well below this; neighbouring table columns are
# often only a little wider apart.
COLUMN_GAP_EM = 1.5
COLUMN_GAP_MIN = 3.0

# Distinct cells that must have text on both sides of a new boundary.
MIN_SUPPORT = 2


def _merge(intervals, tol):
    """Merge (lo, hi) intervals whose gap is at most `tol`."""
    merged = []
    for lo, hi in sorted(intervals):
        if merged and lo - merged[-1][1] <= tol:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [tuple(m) for m in merged]


class TableText:
    """The table's characters, indexed once by the grid cell holding each
    character's centre. `bands(row, col, axis)` gives the intervals a cell's
    text occupies along the y-axis ("row") or x-axis ("col")."""

    def __init__(self, table_blocks, h_lines, v_lines):
        self.h_lines = list(h_lines)
        self.v_lines = list(v_lines)
        self._chars = defaultdict(list)  # (row, col) -> [(x0, y0, x1, y1, size)]
        heights = []
        for block in table_blocks or ():
            for line in block.get("lines", ()):
                for span in line.get("spans", ()):
                    size = span.get("size", 0)
                    # Vertical extent from the font box (baseline, ascender,
                    # descender) rather than the glyph box: with
                    # TEXT_ACCURATE_BBOXES glyph boxes are tight around each
                    # glyph's ink, which makes wrapped lines look further
                    # apart than they are.
                    asc = span.get("ascender", 1.0) * size
                    desc = span.get("descender", 0.0) * size
                    for char in span.get("chars", ()):
                        if char["c"].isspace():
                            continue
                        cx0, cy0, cx1, cy1 = char["bbox"]
                        if "origin" in char and size:
                            cy0, cy1 = char["origin"][1] - asc, char["origin"][1] - desc
                        cell = self._cell_of((cx0 + cx1) / 2, (cy0 + cy1) / 2)
                        if cell is None:
                            continue
                        self._chars[cell].append((cx0, cy0, cx1, cy1, size))
                        heights.append(cy1 - cy0)
        self.line_height = statistics.median(heights) if heights else 0.0

    def _cell_of(self, x, y):
        h, v = self.h_lines, self.v_lines
        if not (h[0] <= y <= h[-1] and v[0] <= x <= v[-1]):
            return None
        row = min(max(bisect.bisect_right(h, y) - 1, 0), len(h) - 2)
        col = min(max(bisect.bisect_right(v, x) - 1, 0), len(v) - 2)
        return row, col

    def bands(self, row, col, axis):
        chars = self._chars.get((row, col), ())
        if axis == "row":
            return _merge(((c[1], c[3]) for c in chars), 0.0)
        tol = max((max(COLUMN_GAP_MIN, c[4] * COLUMN_GAP_EM) for c in chars), default=0)
        # Group the characters into visual lines (overlapping y-ranges) and
        # split each line into runs at horizontal whitespace wider than
        # `tol`; then merge the runs of all lines.
        runs = []
        line, line_y1 = [], None
        for c in sorted(chars, key=lambda c: c[1]):
            if line and c[1] > line_y1:
                runs.extend(_merge(line, tol))
                line = []
            line_y1 = c[3] if not line else max(line_y1, c[3])
            line.append((c[0], c[2]))
        runs.extend(_merge(line, tol))
        return _merge(runs, 0.0)


def gap_boundaries(strips, min_width, require_first_strip=False):
    """New boundaries inside one grid row (or column).

    `strips` holds, for each cell of that row (or column) in order, the
    intervals its text occupies along the axis being split. Returns the
    midpoints of the whitespace intervals that no strip's text crosses,
    that are at least `min_width` wide, and that have text on both sides in
    at least MIN_SUPPORT strips (and, with `require_first_strip`, in the
    first non-empty strip)."""
    occupied = _merge((b for strip in strips for b in strip), 0.0)
    nonempty = [strip for strip in strips if strip]
    boundaries = []
    for (_, a), (b, _) in itertools.pairwise(occupied):
        if b - a < min_width:
            continue
        supporters = [
            strip
            for strip in nonempty
            if strip[0][0] < a and strip[-1][1] > b
        ]
        if len(supporters) < MIN_SUPPORT:
            continue
        if require_first_strip and supporters[0] is not nonempty[0]:
            continue
        boundaries.append((a + b) / 2)
    return boundaries


def repair_grid_gaps(table_blocks, h_lines, v_lines, columns=True):
    """Insert the missing interior row and column boundaries of a table grid
    (see the module docstring). With `columns=False` only rows are repaired.
    Returns (new_h_lines, new_v_lines), sorted, with the outer edges
    unchanged."""
    if len(h_lines) < 2 or len(v_lines) < 2:
        return list(h_lines), list(v_lines)
    text = TableText(table_blocks, h_lines, v_lines)
    nrows, ncols = len(h_lines) - 1, len(v_lines) - 1

    new_h = list(h_lines)
    min_row_gap = ROW_GAP_MIN_FRACTION * text.line_height
    for i in range(nrows):
        strips = [text.bands(i, j, "row") for j in range(ncols)]
        new_h.extend(gap_boundaries(strips, max(min_row_gap, 0.5)))

    new_v = list(v_lines)
    for j in range(ncols if columns else 0):
        strips = [text.bands(i, j, "col") for i in range(nrows)]
        new_v.extend(gap_boundaries(strips, COLUMN_GAP_MIN, require_first_strip=True))

    return sorted(new_h), sorted(new_v)
