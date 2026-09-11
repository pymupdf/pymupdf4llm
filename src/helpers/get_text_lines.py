"""
This script accepts a PDF document filename and converts it to a text file.


Dependencies
-------------
PyMuPDF v1.24.2 or later

Copyright and License
----------------------
Copyright 2024 Artifex Software, Inc.
License GNU Affero GPL 3.0
"""

import sys

import pymupdf
from pymupdf4llm.helpers.utils import (
    is_white,
    almost_in_bbox,
    are_disjoint,
    bbox_is_empty,
    TYPE3_FONT_NAME,
)


def get_raw_lines(
    textpage=None,
    blocks=None,
    clip=None,
    tolerance=3,
    ignore_invisible=True,
    only_horizontal=True,
    require_x_continuity=False,
):
    """Extract the text spans from a TextPage in natural reading sequence.

    All spans roughly on the same line are joined to generate an improved line.
    This copes with MuPDF's algorithm that generates new lines also for spans
    whose horizontal distance is larger than some threshold.

    Result is a sorted list of line objects that consist of the recomputed line
    boundary box and the sorted list of spans in that line.

    This result can then easily be converted e.g. to plain text and other
    formats like Markdown or JSON.

    Args:
        textpage: TextPage object. Can be None if blocks are given.
        blocks: (list) if given, use these blocks instead of extracting them
              from the TextPage. This allows to re-use blocks extracted
              by the caller.
        clip: (Rect) specifies a sub-rectangle of the textpage rect (which in
              turn may be based on a sub-rectangle of the full page).
        tolerance: (float) put spans on the same line if their top or bottom
              coordinate differ by no more than this value.
        ignore_invisible: (bool) if True, invisible text is ignored. This may
              have been set to False for pages with OCR text.
        require_x_continuity: (bool) if True, two spans that are vertically
              close enough to join (per `tolerance`) are only actually joined
              into one synthesized line if they are also horizontally
              contiguous -- i.e. not separated by a gap much larger than a
              normal inter-word space. Without this, a source block whose
              lines occupy two distinct, disjoint x-ranges (most commonly
              when the underlying PDF/MuPDF block detection has mistakenly
              fused two side-by-side columns into a single block, e.g. a
              wrapped label sharing a block with an unrelated value cell
              because they partially share a y-range) gets its unrelated
              same-row content spliced into one output line. Off by default
              because some callers (e.g. `get_text_lines(ocr=True)`'s table
              reconstruction) intentionally rely on wide same-row gaps being
              preserved within one line to recover table columns.

    Returns:
        A sorted list of items (rect, [spans]), each representing one line. The
        spans are sorted left to right. Span dictionaries have been changed:
        - "bbox" has been converted to a Rect object
        - "line" (new) the line number in TextPage.extractDICT
        - "block" (new) the block number in TextPage.extractDICT
        This allows to detect where MuPDF has generated line breaks to indicate
        large inter-span distances.
    """
    y_delta = tolerance  # allowable vertical coordinate deviation
    # Purely geometric threshold for "same visual line" horizontal continuity:
    # normal inter-word/inter-span gaps are a small fraction of the font size;
    # a gap of several font-sizes strongly indicates a different column, not
    # a continuation of the same line. Floored so tiny fonts don't produce a
    # near-zero threshold.
    _X_GAP_EM_MULTIPLIER = 4.0
    _X_GAP_MIN = 5.0

    def x_gap(rect_a, rect_b):
        """Horizontal gap between two rects; 0 if they overlap in x."""
        return max(0.0, rect_b.x0 - rect_a.x1, rect_a.x0 - rect_b.x1)

    def sanitize_spans(line):
        """Sort and join the spans in a re-synthesized line.

        The PDF may contain "broken" text with words cut into pieces.
        This funtion joins spans representing the particles and sorts them
        left to right.

        Arg:
            A list of spans - as derived from TextPage.extractDICT()
        Returns:
            A list of sorted, and potentially cleaned-up spans
        """
        # sort ascending horizontally
        line.sort(key=lambda s: s["bbox"].x0)
        # join spans, delete duplicates
        # underline differences are being ignored
        for i in range(len(line) - 1, 0, -1):  # iterate back to front
            s0 = line[i - 1]  # preceding span
            s1 = line[i]  # this span
            # "delta" depends on the font size. Spans  will be joined if
            # no more than 10% of the font size separates them and important
            # attributes are the same.
            delta = s1["size"] * 0.1
            if s0["bbox"].x1 + delta < s1["bbox"].x0 or (
                s0["flags"],
                s0["char_flags"] & ~2,
                # s0["size"],
            ) != (
                s1["flags"],
                s1["char_flags"] & ~2,
                # s1["size"],
            ):
                continue  # no joining
            # We need to join bbox and text of two consecutive spans
            # Sometimes, spans may also be duplicated.
            if s0["text"] != s1["text"] or s0["bbox"] != s1["bbox"]:
                s0["text"] += s1["text"]
            s0["bbox"] |= s1["bbox"]  # join boundary boxes
            del line[i]  # delete the joined-in span
            line[i - 1] = s0  # update the span
        return line

    if not isinstance(textpage, pymupdf.TextPage) and blocks is None:
        raise ValueError("Either textpage or blocks must be provided.")

    if clip is None and textpage is not None:  # use TextPage rect if not provided
        clip = textpage.rect
    # extract text blocks - if bbox is not empty
    if blocks is None:
        blocks = [
            b
            for b in textpage.extractDICT()["blocks"]
            if b["type"] == 0 and not bbox_is_empty(b["bbox"])
        ]
    spans = []  # all spans in TextPage here
    for bno, b in enumerate(blocks):  # the numbered blocks
        if are_disjoint(b["bbox"], clip):
            continue
        for lno, line in enumerate(b["lines"]):  # the numbered lines
            if are_disjoint(line["bbox"], clip):
                continue
            line_dir = line["dir"]
            if (
                only_horizontal and abs(1 - line_dir[0]) > 1e-3
            ):  # only accept horizontal text
                continue
            for sno, s in enumerate(line["spans"]):  # the numered spans
                if is_white(s["text"]):
                    # ignore white text if not a Type3 font
                    continue
                # Ignore invisible text. Type 3 font text is never invisible.
                if (
                    not s["font"].startswith(TYPE3_FONT_NAME)
                    and s["alpha"] == 0
                    and ignore_invisible
                ):
                    continue
                sbbox = pymupdf.Rect(s["bbox"])  # span bbox as a Rect
                # the y-coords of the line bbox wrap the y-coords of
                # all spans. Therefore use the line's y coordinates.
                sbbox.y0 = line["bbox"][1]
                sbbox.y1 = line["bbox"][3]
                s["bbox"] = sbbox  # update with the Rect version
                if not almost_in_bbox(s["bbox"], clip, portion=0.51):
                    # if not mostly inside clip
                    continue
                s["line"] = lno
                s["block"] = bno
                s["dir"] = line_dir
                spans.append(s)

    if not spans:  # no text at all
        return []

    spans.sort(key=lambda s: (-s["dir"][0], s["bbox"].y1))  # sort spans by bottom coord
    nlines = []  # final result
    line = [spans[0]]  # collects spans with fitting vertical coordinates
    lrect = spans[0]["bbox"]  # rectangle joined from span rectangles

    for s in spans[1:]:  # walk through the spans
        sbbox = s["bbox"]  # this bbox
        sbbox0 = line[-1]["bbox"]  # previous bbox
        # if any of top or bottom coordinates are close enough, join...
        y_ok = abs(sbbox.y1 - sbbox0.y1) <= y_delta or abs(sbbox.y0 - sbbox0.y0) <= y_delta
        if y_ok and require_x_continuity:
            # Check continuity against the whole accumulated line rect, not
            # just the last-appended span -- spans are sorted globally by
            # y1, so "last appended" isn't necessarily the one geometrically
            # nearest in x.
            max_gap = max(_X_GAP_MIN, s.get("size", 0) * _X_GAP_EM_MULTIPLIER)
            y_ok = x_gap(lrect, sbbox) <= max_gap
        if y_ok:
            line.append(s)  # append to this line
            lrect |= sbbox  # extend line rectangle
            continue

        # end of current line, sort its spans from left to right
        line = sanitize_spans(line)

        # append line rect and its spans to final output
        nlines.append([lrect, line])

        line = [s]  # start next line
        lrect = sbbox  # initialize its rectangle

    # need to append last line in the same way
    line = sanitize_spans(line)
    nlines.append([lrect, line])

    if require_x_continuity:
        nlines = _reorder_multi_column_lines_within_block(nlines)

    return nlines


# Line rects within one source block are normally left-aligned at (nearly)
# the same x0. A gap this much larger indicates the block's lines actually
# occupy two distinct columns (see require_x_continuity above for why that
# can happen), not an ordinary nested-indent jitter within one column.
_BLOCK_COLUMN_X_GAP = 20.0


def _reorder_multi_column_lines_within_block(nlines):
    """Within any single source block whose synthesized lines still span
    more than one x-cluster after the horizontal-continuity fix above (i.e.
    a block whose *lines* -- not just individual same-row spans -- occupy
    two disjoint x-ranges), re-emit that block's lines in column-major
    order: each column's lines top-to-bottom, left column before right --
    instead of the default single sort-by-y order, which would otherwise
    still interleave the two columns row by row.

    Purely geometric (line rect x0 proximity only). A block whose lines are
    all left-aligned at (nearly) the same x0 -- the normal, single-column
    case -- yields one cluster and is left untouched. Only reorders
    positions already occupied by entries from the same block; the
    relative position of different blocks in `nlines` is unchanged.
    """
    if not nlines:
        return nlines

    block_positions = {}
    for i, (_, spans) in enumerate(nlines):
        bno = spans[0]["block"] if spans else None
        block_positions.setdefault(bno, []).append(i)

    result = list(nlines)
    for bno, positions in block_positions.items():
        if bno is None or len(positions) < 2:
            continue
        entries = [nlines[i] for i in positions]

        order_by_x0 = sorted(range(len(entries)), key=lambda i: entries[i][0].x0)
        clusters = [[order_by_x0[0]]]
        for i in order_by_x0[1:]:
            prev = clusters[-1][-1]
            if entries[i][0].x0 - entries[prev][0].x0 <= _BLOCK_COLUMN_X_GAP:
                clusters[-1].append(i)
            else:
                clusters.append([i])
        if len(clusters) < 2:
            continue  # single column -- nothing to reorder

        clusters.sort(key=lambda c: min(entries[i][0].x0 for i in c))
        new_order = []
        for cluster in clusters:
            new_order.extend(sorted(cluster, key=lambda i: entries[i][0].y0))

        for pos, i in zip(positions, new_order):
            result[pos] = entries[i]

    return result


def get_text_lines(page, *, textpage=None, clip=None, sep="\t", tolerance=3, ocr=False):
    """Extract text by line keeping natural reading sequence.

    Notes:
        Internally uses "dict" to select lines and their spans.
        Returns plain text. If originally separate MuPDF lines in fact have
        (approximatly) the same baseline, they are joined into one line using
        the 'sep' character(s).
        This method can be used to extract text in reading sequence - even in
        cases of text replaced by way of redaction annotations.

    Args:
        page: (pymupdf.Page)
        textpage: (TextPage) if None a temporary one is created.
        clip: (rect-like) only consider spans inside this area
        sep: (str) use this string when joining multiple MuPDF lines.
    Returns:
        String of plain text in reading sequence.
    """
    textflags = pymupdf.TEXT_MEDIABOX_CLIP
    page.remove_rotation()
    prect = page.rect if not clip else pymupdf.Rect(clip)  # area to consider

    sep = sep if sep == "|" else ""

    # make a TextPage if required
    if textpage is None:
        if ocr is False:
            tp = page.get_textpage(clip=prect, flags=textflags)
        else:
            tp = page.get_textpage_ocr(dpi=300, full=True)
    else:
        tp = textpage

    lines = get_raw_lines(tp, clip=prect, tolerance=tolerance)

    if not textpage:  # delete temp TextPage
        tp = None

    if not lines:
        return ""

    # Compose final text
    alltext = ""

    if not ocr:
        prev_bno = -1  # number of previous text block
        for lrect, line in lines:  # iterate through lines
            # insert extra line break if a different block
            bno = line[0]["block"]  # block number of this line
            if bno != prev_bno:
                alltext += "\n"
            prev_bno = bno

            line_no = line[0]["line"]  # store the line number of previous span
            for s in line:  # walk over the spans in the line
                lno = s["line"]
                stext = s["text"]
                if line_no == lno:
                    alltext += stext
                else:
                    alltext += sep + stext
                line_no = lno
            alltext += "\n"  # append line break after a line
        alltext += "\n"  # append line break at end of block
        return alltext

    """
    For OCR output, we try a rudimentary table recognition.
    """
    rows = []
    xvalues = []
    col_count = 0  # just to calm down the linter
    for lrect, line in lines:
        # if only 1 span in line and no columns identified yet...
        if len(line) == 1 and not xvalues:
            alltext += line[0]["text"] + "\n\n\n"
            continue
        # multiple spans in line and no columns identified yet
        elif not xvalues:  # define column borders
            xvalues = [s["bbox"].x0 for s in line] + [line[-1]["bbox"].x1]
            col_count = len(line)  # number of columns
        row = [""] * col_count
        for r, l in line:
            for i in range(len(xvalues) - 1):
                x0, x1 = xvalues[i], xvalues[i + 1]
                if abs(r.x0 - x0) <= 3 or abs(r.x1 - x1) <= 3:
                    row[i] = l
        rows.append(row)
    if rows:
        row = "|" + "|".join(rows[0]) + "|\n"
        alltext += row
        alltext += "|---" * len(rows[0]) + "|\n"
        for row in rows[1:]:
            alltext += "|" + "|".join(row) + "|\n"
        alltext += "\n"
    return alltext


if __name__ == "__main__":
    import pathlib

    filename = sys.argv[1]
    doc = pymupdf.open(filename)
    text = ""
    for page in doc:
        text += get_text_lines(page, sep=" ") + "\n" + chr(12) + "\n"
    pathlib.Path(f"{doc.name}.txt").write_bytes(text.encode())
