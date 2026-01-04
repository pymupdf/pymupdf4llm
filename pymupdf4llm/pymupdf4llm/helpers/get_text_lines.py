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
    outside_bbox,
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

    if clip is None:  # use TextPage rect if not provided
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
        if outside_bbox(b["bbox"], clip):
            continue
        for lno, line in enumerate(b["lines"]):  # the numbered lines
            if outside_bbox(line["bbox"], clip):
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
                    s["font"] != TYPE3_FONT_NAME
                    and s["alpha"] == 0
                    and ignore_invisible
                ):
                    continue
                if not almost_in_bbox(s["bbox"], clip):  # if not in clip
                    continue
                sbbox = pymupdf.Rect(s["bbox"])  # span bbox as a Rect
                if s["flags"] & 1:  # if a superscript, modify bbox
                    # with that of the preceding or following span
                    i = 1 if sno == 0 else sno - 1
                    if len(line["spans"]) > i:
                        neighbor = line["spans"][i]
                        sbbox.y1 = neighbor["bbox"][3]
                    s["text"] = f"[{s['text']}]"
                s["bbox"] = sbbox  # update with the Rect version
                # include line/block numbers to facilitate separator insertion
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
        if abs(sbbox.y1 - sbbox0.y1) <= y_delta or abs(sbbox.y0 - sbbox0.y0) <= y_delta:
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

    return nlines


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

    xsep = sep if sep == "|" else ""

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
