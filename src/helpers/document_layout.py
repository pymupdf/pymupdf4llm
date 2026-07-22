import base64
import io
import json
import math
import threading
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union
import textwrap

import pymupdf
import tabulate
from pymupdf import mupdf
from pymupdf4llm.helpers import utils
from pymupdf4llm.helpers.get_text_lines import get_raw_lines
from pymupdf4llm.ocr import OCRMode

try:
    from tqdm import tqdm as ProgressBar
except ImportError:
    from pymupdf4llm.helpers.progress import ProgressBar

from dataclasses import dataclass

pymupdf.TOOLS.unset_quad_corrections(True)
_LAYOUT_LOCK = threading.RLock()

INFO_MESSAGES = io.StringIO()
GRAPHICS_TEXT = "\n![](%s)\n"

FLAGS = (
    0
    | pymupdf.TEXT_COLLECT_STYLES
    | pymupdf.TEXT_COLLECT_VECTORS
    | pymupdf.TEXT_PRESERVE_IMAGES
    | pymupdf.TEXT_ACCURATE_BBOXES
    | pymupdf.TEXT_MEDIABOX_CLIP
    | pymupdf.TEXT_IGNORE_ACTUALTEXT
)
BULLETS = tuple(utils.BULLETS)


def get_layout_locked(page: pymupdf.Page, **kwargs):
    """Serialize PyMuPDF layout inference, which uses process-global state."""
    with _LAYOUT_LOCK:
        return page.get_layout(**kwargs)


def get_table_details(tab_dict, table_blocks):
    """Create a TableDetails object.

    The table dictionary is as returned by the Layout module with option
    "return_raw=True".
    """
    tab_det = TableDetails()
    tab_det.bbox = tab_dict["group_bbox"]  # bounding box
    x0, y0, x1, y1 = tab_det.bbox
    grid = tab_dict.get("table_grid")  # Layout's GridPrediction object
    cells = []  # cell bounding boxes
    extract = []  # cell text content
    md_cells = []  # cell markdown content
    h_lines = [y0] + [h + y0 for h in grid.h_lines] + [y1]
    v_lines = [x0] + [v + x0 for v in grid.v_lines] + [x1]
    tab_det.row_count = len(h_lines) - 1
    tab_det.col_count = len(v_lines) - 1
    for i in range(tab_det.row_count):
        row = []
        text_row = []
        md_row = []
        for j in range(tab_det.col_count):
            cell_bbox = (v_lines[j], h_lines[i], v_lines[j + 1], h_lines[i + 1])
            row.append(cell_bbox)
            text = utils.extract_cells(
                table_blocks, cell_bbox, markdown=False, ocrpage=False
            )
            text_row.append(text)
            md_text = utils.extract_cells(
                table_blocks, cell_bbox, markdown=True, ocrpage=False
            )
            md_row.append(md_text)
        cells.append(row)
        extract.append(text_row)
        md_cells.append(md_row)
    tab_det.cells = cells
    tab_det.extract = extract
    tab_det.markdown = utils.table_to_markdown(md_cells)
    return tab_det


def wrap_table_for_tabulate(table, max_width=100, min_col_width=10):
    """
    Pre-wraps a table (List[List[str]]) so that tabulate cannot produce
    absurdly wide tables. Each column gets a width budget based on max_width.
    """
    if not table:
        return table

    # Number of columns
    num_cols = max(len(row) for row in table)

    # Distribute width evenly
    base_width = max(min_col_width, max_width // num_cols)
    col_widths = [base_width] * num_cols

    wrapped_table = []

    for row in table:
        new_row = []
        for col_idx, cell in enumerate(row):
            cell = cell or ""
            width = col_widths[col_idx]

            # Wrap the cell text
            lines = textwrap.wrap(cell, width=width) or [""]
            new_row.append("\n".join(lines))

        wrapped_table.append(new_row)

    return wrapped_table


def make_page_chunk(doc, page, text, string_lengths) -> Dict:
    """Create a page chunk dictionary for output.

    Args:
        doc: the ParsedDocument object
        page: the PageLayout object
        text: the page text string

    Returns:
        dict: page chunk dictionary
    """
    assert len(page.boxes) == len(string_lengths)
    chunk = defaultdict(lambda: None)
    page_tocs = [t for t in doc.toc if t[-1] == page.page_number]
    chunk["metadata"] = doc.metadata | {
        "file_path": doc.filename,
        "page_count": doc.page_count,
        "page_number": page.page_number,
    }

    chunk["toc_items"] = page_tocs
    page_boxes = []
    for i in range(len(page.boxes)):
        b = page.boxes[i]
        start = string_lengths[i - 1] if i > 0 else 0
        stop = string_lengths[i]
        page_boxes.append(
            {
                "index": i,
                "class": b.boxclass,
                "bbox": tuple(pymupdf.IRect(b.x0, b.y0, b.x1, b.y1)),
                "pos": (start, stop),
            }
        )
    chunk["page_boxes"] = page_boxes
    chunk["text"] = text
    return chunk


def omit_if_pua_char(text):
    """Check if character is in the Private Use Area (PUA) of Unicode."""
    if len(text) != 1:  # only single characters are checked
        return text
    o = ord(text)
    if (
        (0xE000 <= o <= 0xF8FF)
        or (0xF0000 <= o <= 0xFFFFD)
        or (0x100000 <= o <= 0x10FFFD)
    ):
        return ""
    return text


def create_list_item_levels(layout_info):
    """Map the layout box number of each list-item to its hierarchy level.

    Args:
        layout_info (list): the bbox list "page.layout_information"

    Returns:
        dict: {bbox sequence number: level} where level is 1 for top-level.
    """
    segments = []  # list of item segments
    segment = []  # current segment

    # Create segments of contiguous list items. Each non-list-item finishes
    # the current segment. Also, two list-items in a row belonging to different
    # page text columns end the segment after the first item.
    for i, item in enumerate(layout_info):
        if item.boxclass != "list-item":  # bbox class is no list-item
            if segment:  # end and save the current segment
                segments.append(segment)
                segment = []
            continue
        if segment:  # check if we need to end the current segment
            _, prev_item = segment[-1]
            if item.x0 > prev_item.x1 or item.y1 < prev_item.y0:
                # end and save the current segment
                segments.append(segment)
                segment = []
        segment.append((i, item))  # append item to segment
    if segment:
        segments.append(segment)  # append last segment

    item_dict = {}  # dictionary of item index -> (level
    if not segments:  # no list items found
        return item_dict

    # walk through segments and assign levels
    for i, s in enumerate(segments):
        if not s:  # skip empty segments
            continue
        s.sort(key=lambda x: x[1].x0)  # sort by x0 coordinate of the bbox

        # list of leveled items in the segment: (idx, bbox, level)
        # first item has level 1
        leveled_items = [(s[0][0], s[0][1], 1)]
        for idx, bbox in s[1:]:
            prev_idx, prev_bbox, prev_lvl = leveled_items[-1]
            # x0 coordinate increased by more than 10 points: increase level
            if bbox.x0 > prev_bbox.x0 + 10:
                curr_lvl = prev_lvl + 1
                leveled_items.append((idx, bbox, curr_lvl))
            else:
                leveled_items.append((idx, bbox, prev_lvl))
        for idx, bbox, lvl in leveled_items:
            item_dict[idx] = lvl
    return item_dict


def is_monospaced(textlines):
    """Detect text bboxes with all mono-spaced lines.

    Returns True if all lines are mono-spaced.
    Used to output code blocks.
    """
    line_count = len(textlines)
    mono = 0

    for l in textlines:
        all_mono = all(
            bool(
                s["flags"] & pymupdf.TEXT_FONT_MONOSPACED
                and not utils.is_ocr_text(s)
            )
            for s in l["spans"]
            if not s["text"].isspace()
        )
        if all_mono:
            mono += 1
    return mono == line_count


def is_superscripted(line):
    spans = line["spans"]
    line_bbox = line["bbox"]
    if not spans:
        return False
    span0 = spans[0]
    if span0["flags"] & 1:  # check for superscript flag
        return True
    if len(spans) < 2:  # single span line: skip
        return False
    if span0["origin"][1] < spans[1]["origin"][1] and span0["size"] < spans[1]["size"]:
        return True
    return False


def get_plain_text(spans):
    """Output text without any markdown or other styling.
    Parameter is a list of span dictionaries. The spans may come from
    one or more original "textlines" items.
    Returns the text string of the boundary box.
    """
    output = ""
    for i, s in enumerate(spans):
        superscript = s["flags"] & 1
        span_text = s["text"].strip()  # remove leading/trailing spaces
        if superscript:
            # enclose superscripted text in brackets if first span
            if i == 0:
                span_text = f"[{span_text}] "
            elif output.endswith(" "):
                output = output[:-1]
        # resolve hyphenation
        if output.endswith("- ") and len(output.split()[-1]) > 2:
            output = output[:-2]
        output += span_text + " "
    return output


def list_item_to_text(textlines, level) -> str:
    """
    Convert "list-item" bboxes to text.
    """
    if not textlines:
        return ""
    indent = "   " * (level - 1)  # indentation based on level
    output = indent
    line = textlines[0]
    x0 = line["bbox"][0]  # left of first line
    spans = line["spans"]
    span0 = line["spans"][0]
    span0_text = span0["text"].strip()

    if not omit_if_pua_char(span0_text):
        spans.pop(0)
        if spans:
            x0 = spans[0]["bbox"][0]

    for line in textlines[1:]:
        this_x0 = line["bbox"][0]
        if this_x0 < x0 - 2:
            line_output = get_plain_text(spans)
            output += line_output
            output = output.rstrip() + f"\n\n{indent}"
            spans = line["spans"]
            if not omit_if_pua_char(spans[0]["text"].strip()):
                spans.pop(0)
        else:
            spans.extend(line["spans"])
        x0 = this_x0  # store this left coordinate
    line_output = get_plain_text(spans)
    output += line_output

    return output.rstrip() + "\n\n"


def footnote_to_text(textlines) -> str:
    """
    Convert "footnote" bboxes to text.
    """
    if not textlines:
        return ""
    # we render footnotes as blockquotes
    output = "> "
    line = textlines[0]
    spans = line["spans"]

    for line in textlines[1:]:
        # superscripted line starts a new footnote line
        if is_superscripted(line):
            line_output = get_plain_text(spans)
            output += line_output
            output = output.rstrip() + "\n\n> "
            spans = line["spans"]
        else:
            spans.extend(line["spans"])
    line_output = get_plain_text(spans)
    output += line_output

    return output.rstrip() + "\n\n"


def code_block_to_text(textlines):
    """Output a code block in plain text format.

    Basic difference is that lines are separated by line breaks.
    """
    output = ""
    for line in textlines:
        line_text = ""
        for s in line["spans"]:
            span_text = s["text"]
            line_text += span_text
        output += line_text.rstrip() + "\n"
    output += "\n\n"
    return output


def text_to_text(textlines, ignore_code: bool = False):
    """
    Convert "text" bboxes to plain text, as well as boxclasses
    not specifically handled elsewhere.
    The text of all spans of all lines is written without line breaks.
    At the end, two newlines are added to separate from the next block.
    """
    if not textlines:
        return ""
    if is_superscripted(textlines[0]):  # check for superscript
        # handle mis-classified text boundary box
        return footnote_to_text(textlines)
    # handle completely mnonospaced textlines as code block
    if not ignore_code and is_monospaced(textlines):
        return code_block_to_text(textlines)

    spans = []
    for l in textlines:
        for s in l["spans"]:
            assert isinstance(s, dict)
            spans.append(s)
    output = get_plain_text(spans)
    return output + "\n\n"


def picture_text_to_text(textlines, ignore_code: bool = False, clip=None):
    """Convert text extracted from images to plain text format.

    In case text has been written inside a picture bbxox, we want to output it
    in some form. Because we cannot be sure about the formatting we simply
    write it line by line wrapped by markers.
    """
    if not textlines:
        return "\n"
    output = "----- Start of picture text -----\n"
    for tl in textlines:
        line_text = " ".join([s["text"] for s in tl["spans"]])
        output += line_text.rstrip() + "\n"
    output += "----- End of picture text -----\n"
    return output + "\n"


def fallback_text_to_text(textlines, ignore_code: bool = False, clip=None):
    """Convert text extracted from unrecognized tables.

    We hope for some sort of table structure being present in the text spans:
    The maximum span count in the lines is assumed to equal column count.
    """
    span_count = max(len(tl["spans"]) for tl in textlines)
    lines = []
    output = ""
    for tl in textlines:
        spans = tl["spans"]
        # prepare a row with empty strings in each cell
        line = [""] * span_count
        if len(spans) < span_count and spans[0]["bbox"][0] > clip[0] + 10:
            i = 1
        else:
            i = 0
        for j, s in enumerate(spans, start=i):
            line[j] = f'{s["text"].strip()} '
        lines.append(line)
    tab_text = tabulate.tabulate(
        lines,
        tablefmt="grid",
        disable_numparse=True,
        maxcolwidths=int(100 / span_count),
    )
    output += tab_text + "\n"
    return output + "\n"


def get_styled_text(spans):
    """Output text with markdown style codes based on font properties.
    Parameter is a list of span dictionaries. The spans may come from
    one or more original "textlines" items.
    Returns the text string and the suffix for continuing styles.
    The text string always ends with the suffix and a space
    """
    output = ""
    prefix = ""
    suffix = ""
    old_line = 0
    old_block = 0

    for i, s in enumerate(spans):
        # decode font flags and char_flags properties
        superscript = s["flags"] & pymupdf.TEXT_FONT_SUPERSCRIPT
        mono = s["flags"] & pymupdf.TEXT_FONT_MONOSPACED and not utils.is_ocr_text(s)
        bold = (
            s["flags"] & pymupdf.TEXT_FONT_BOLD
            or s["char_flags"] & pymupdf.mupdf.FZ_STEXT_BOLD
        )
        italic = s["flags"] & pymupdf.TEXT_FONT_ITALIC
        strikeout = s["char_flags"] & pymupdf.mupdf.FZ_STEXT_STRIKEOUT
        underline = s["char_flags"] & pymupdf.mupdf.FZ_STEXT_UNDERLINE
        highlight = s["char_flags"] & pymupdf.mupdf.FZ_STEXT_HIGHLIGHT

        # compute styling prefix and suffix
        prefix = []
        suffix = []

        if superscript:
            prefix.append("<sup>")
            suffix.append("</sup>")

        if bold:
            prefix.append("**")
            suffix.append("**")

        if italic:
            prefix.append("_")
            suffix.append("_")

        if strikeout:
            prefix.append("~~")
            suffix.append("~~")

        if underline:
            prefix.append("<u>")
            suffix.append("</u>")

        if highlight:
            prefix.append("<mark>")
            suffix.append("</mark>")

        if mono:
            prefix.append("`")
            suffix.append("`")

        prefix = "".join(prefix)
        suffix = "".join(reversed(suffix))

        span_text = s["text"].strip()  # remove leading/trailing spaces
        # convert intersecting link to markdown syntax
        # ltext = resolve_links(parms.links, s)
        # ltext = ""  # TODO: implement link resolution
        # if ltext:
        #     text = f"{hdr_string}{prefix}{ltext}{suffix} "
        # else:
        #     text = f"{prefix}{span_text}{suffix} "
        text = f"{prefix}{span_text}{suffix} "
        # Extend output string taking care of styles staying the same.
        if output.endswith(f"{suffix} "):
            output = output[: -len(suffix) - 1]
            # resolve hyphenation if old_block and old_line are not the same
            if (
                1
                and (old_block, old_line) != (s["block"], s["line"])
                and output.endswith("-")
                and len(output.split()[-1]) > 2
            ):
                output = output[:-1]
                text = span_text + suffix + " "
            elif superscript:
                text = span_text + suffix + " "
            else:
                text = " " + span_text + suffix + " "

        old_line = s["line"]
        old_block = s["block"]
        if superscript:
            output = output.rstrip(" ")
        output += text
    return output, suffix


def list_item_to_md(textlines, level):
    """
    Convert "list-item" bboxes to markdown.
    The first line is prefixed with "- ". Subsequent lines are appended
    without line break if their rectangle does not start to the left
    of the previous line.
    Otherwise, a linebreak and "- " are added to the output string.
    2 units of tolerance is used to avoid spurious line breaks.

    This post-layout heuristics helps cover cases where more than
    one list item is contained in a single bbox.
    """

    if not textlines:
        return ""
    indent = "   " * (level - 1)  # indentation based on level
    line = textlines[0]
    x0 = line["bbox"][0]  # left of first line
    spans = line["spans"]
    span0 = line["spans"][0]
    span0_text = span0["text"].strip()

    starter = "- "
    if utils.startswith_bullet(span0_text):
        span0_text = span0_text[1:].strip()
        line["spans"][0]["text"] = span0_text
    elif span0_text.endswith(".") and span0_text[:-1].isdigit():
        starter = ""
    elif " " in span0_text:
        first_word = span0_text.split(" ")[0]
        if first_word.endswith(".") and first_word[:-1].isdigit():
            starter = ""

    if not omit_if_pua_char(span0["text"].strip()):
        # bullet was a PUA char: remove it
        spans.pop(0)
        if spans:
            x0 = spans[0]["bbox"][0]

    output = indent + starter
    for line in textlines[1:]:
        this_x0 = line["bbox"][0]
        if this_x0 < x0 - 2:
            line_output, suffix = get_styled_text(spans)
            output += line_output + f"\n\n{indent}{starter}"
            spans = line["spans"]
            if not omit_if_pua_char(spans[0]["text"].strip()):
                spans.pop(0)
        else:
            spans.extend(line["spans"])
        x0 = this_x0  # store this left coordinate
    line_output, suffix = get_styled_text(spans)
    output += line_output

    return output + "\n\n"


def footnote_to_md(textlines):
    """
    Convert "footnote" bboxes to markdown.
    The first line is prefixed with "> ". Subsequent lines are appended
    without line break if they do not start with a superscript.
    Otherwise, a linebreak and "> " are added to the output string.

    This post-layout heuristics helps cover cases where more than
    one list item is contained in a single bbox.
    """
    if not textlines:
        return ""
    line = textlines[0]
    spans = line["spans"]
    output = "> "
    for line in textlines[1:]:
        if is_superscripted(line):
            line_output, suffix = get_styled_text(spans)
            output += line_output + "\n\n> "
            spans = line["spans"]
        else:
            spans.extend(line["spans"])
    line_output, suffix = get_styled_text(spans)
    output += line_output

    return output + "\n\n"


def section_hdr_to_md(header_level, textlines):
    """
    Convert "section-header" bboxes to markdown.
    """
    spans = []
    for l in textlines:
        for s in l["spans"]:
            assert isinstance(s, dict)
            spans.append(s)
    output, suffix = get_styled_text(spans)
    return f"{'#' * header_level} {output}\n\n"


def title_to_md(header_level, textlines):
    """
    Convert "title" bboxes to markdown.
    The line text itself is handled like normal text.
    TODO: Consider joining with section_hdr.
    """
    spans = []
    for l in textlines:
        for s in l["spans"]:
            assert isinstance(s, dict)
            spans.append(s)
    output, suffix = get_styled_text(spans)
    return f"{'#' * header_level} {output}\n\n"


def code_block_to_md(textlines):
    """Output a code block in markdown format."""
    output = "```\n"
    for line in textlines:
        line_text = ""
        for s in line["spans"]:
            span_text = s["text"]
            line_text += span_text
        output += line_text.rstrip() + "\n"
    output += "```\n\n"
    return output


def text_to_md(textlines, ignore_code: bool = False):
    """
    Convert "text" bboxes to markdown, as well as other boxclasses
    not specifically handled elsewhere.
    The line text is written without line breaks. At the end,
    two newlines are added to separate from the next block.
    """
    if not textlines:
        return ""
    if is_superscripted(textlines[0]):
        # exec advanced superscript detector
        return footnote_to_md(textlines)
    if not ignore_code and is_monospaced(textlines):
        return code_block_to_md(textlines)

    spans = []
    for l in textlines:
        for s in l["spans"]:
            assert isinstance(s, dict)
            spans.append(s)
    output, suffix = get_styled_text(spans)
    return output + "\n\n"


def picture_text_to_md(textlines, ignore_code: bool = False, clip=None):
    """Convert text extracted from images to plain text format.

    In case text has been written inside a picture bbxox, we want to output it
    in some form. Because we cannot be sure about the formatting we simply
    write it line by line wrapped by markers.
    """
    if not textlines:
        return "\n"
    output = "<!-- Start of picture text -->\n"
    for tl in textlines:
        line_text = " ".join([s["text"] for s in tl["spans"]])
        output += line_text.rstrip() + "<br>"
    output += "<!-- End of picture text -->\n"
    return output + "\n"


def fallback_text_to_md(textlines, ignore_code: bool = False, clip=None):
    """
    Convert text extracted from images to markdown format.
    """
    span_count = max(len(tl["spans"]) for tl in textlines)
    output = "<!-- Start of picture text -->\n"
    output += "|" * (span_count + 1) + "\n"
    output += "|" + "|".join(["---"] * span_count) + "|\n"
    for tl in textlines:
        ltext = "|" + "|".join([s["text"].strip() for s in tl["spans"]]) + "|\n"
        output += ltext
    output += "\n<!-- End of picture text -->\n"
    return output + "\n"


def _rect_area(rect) -> float:
    return max(0.0, float(rect.x1 - rect.x0)) * max(0.0, float(rect.y1 - rect.y0))


def _html_table_meta(table_item) -> Dict:
    rect = pymupdf.Rect(table_item[0])
    return {
        "bbox": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
        "html": table_item[1],
        "rows": int(table_item[2]) if len(table_item) > 2 and table_item[2] is not None else None,
        "cols": int(table_item[3]) if len(table_item) > 3 and table_item[3] is not None else None,
        "cells": table_item[4] if len(table_item) > 4 else None,
        "extract": table_item[5] if len(table_item) > 5 else None,
    }


def _assign_html_tables_to_boxes(layout_boxes, html_tables, threshold: float = 0.5):
    """Assign table_html output to layout table boxes, adding unmatched boxes."""
    by_box: Dict[tuple, List[Dict]] = {}
    if not html_tables:
        return layout_boxes, by_box

    table_boxes = [
        (tuple(pymupdf.IRect(box[:4])), pymupdf.Rect(box[:4]))
        for box in layout_boxes
        if len(box) >= 5 and box[4] == "table"
    ]
    augmented_boxes = list(layout_boxes)

    for table_item in html_tables:
        meta = _html_table_meta(table_item)
        table_rect = pymupdf.Rect(meta["bbox"])
        table_area = _rect_area(table_rect)
        best_key = None
        best_score = 0.0
        if table_area > 0:
            for key, box_rect in table_boxes:
                inter = table_rect & box_rect
                if inter.is_empty:
                    continue
                score = _rect_area(inter) / table_area
                if score > best_score:
                    best_key = key
                    best_score = score
        if best_key is None or best_score < threshold:
            synthetic = (table_rect.x0, table_rect.y0, table_rect.x1, table_rect.y1, "table")
            augmented_boxes.append(synthetic)
            best_key = tuple(pymupdf.IRect(table_rect))
            table_boxes.append((best_key, table_rect))
        by_box.setdefault(best_key, []).append(meta)

    for items in by_box.values():
        items.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return augmented_boxes, by_box


def _line_claimed_by_table(line_bbox, table_rects) -> bool:
    line_rect = pymupdf.Rect(line_bbox)
    line_area = _rect_area(line_rect)
    if line_area <= 0:
        return False
    center = pymupdf.Point(
        (line_rect.x0 + line_rect.x1) * 0.5,
        (line_rect.y0 + line_rect.y1) * 0.5,
    )
    for table_rect in table_rects:
        if center in table_rect:
            return True
        inter = line_rect & table_rect
        if not inter.is_empty and _rect_area(inter) / line_area >= 0.5:
            return True
    return False


def _union_line_rects(lines):
    rect = pymupdf.Rect(lines[0]["bbox"])
    for line in lines[1:]:
        rect |= pymupdf.Rect(line["bbox"])
    return rect


def _split_text_box_around_tables(box, fulltext, table_rects):
    box_rect = pymupdf.Rect(box[:4])
    if not any(box_rect.intersects(table_rect) for table_rect in table_rects):
        return [box], {}

    try:
        lines = [
            {"bbox": l[0], "spans": l[1]}
            for l in get_raw_lines(
                textpage=None,
                blocks=fulltext,
                clip=box_rect,
                ignore_invisible=False,
            )
        ]
    except Exception:
        return [box], {}
    if not lines:
        return [box], {}

    split_boxes = []
    textlines_by_box = {}
    claimed = [_line_claimed_by_table(line["bbox"], table_rects) for line in lines]
    start = None
    for index, is_claimed in enumerate(claimed + [True]):
        if not is_claimed and start is None:
            start = index
        elif is_claimed and start is not None:
            end = index - 1
            y0 = box_rect.y0 if start == 0 else pymupdf.Rect(lines[start - 1]["bbox"]).y1
            y1 = (
                box_rect.y1
                if end == len(lines) - 1
                else pymupdf.Rect(lines[end + 1]["bbox"]).y0
            )
            if y1 <= y0:
                rect = _union_line_rects(lines[start : end + 1])
                y0, y1 = rect.y0, rect.y1
            split_box = (box_rect.x0, y0, box_rect.x1, y1, box[4])
            split_boxes.append(split_box)
            textlines_by_box[tuple(pymupdf.IRect(split_box[:4]))] = lines[start : end + 1]
            start = None
    return split_boxes, textlines_by_box


def normalize_layout_boxes(layout_boxes, html_tables, fulltext):
    """Build opt-in HTML-table layout boxes before reading-order sorting."""
    normalized_boxes, html_tables_by_box = _assign_html_tables_to_boxes(
        layout_boxes,
        html_tables,
    )
    if not html_tables_by_box:
        return normalized_boxes, html_tables_by_box, {}

    table_rects = [
        pymupdf.Rect(item["bbox"])
        for items in html_tables_by_box.values()
        for item in items
    ]
    output_boxes = []
    textlines_by_box = {}
    for box in normalized_boxes:
        if len(box) < 5 or box[4] in ("table", "picture", "formula"):
            output_boxes.append(box)
            continue
        split_boxes, split_textlines = _split_text_box_around_tables(
            box,
            fulltext,
            table_rects,
        )
        output_boxes.extend(split_boxes)
        textlines_by_box.update(split_textlines)
    return output_boxes, html_tables_by_box, textlines_by_box


@dataclass
class TableDetails:
    bbox: tuple = None
    row_count: int = None
    col_count: int = None
    cells: list = None  # list of list of cell bbox coordinates
    extract: list = None  # list of list of cell plain text content
    markdown: str = None  # table markdown content


@dataclass
class LayoutBox:
    x0: float
    y0: float
    x1: float
    y1: float
    boxclass: str  # e.g. 'text', 'picture', 'table', etc.

    # if boxclass == 'picture' or 'formula', store image bytes
    image: Optional[bytes] = None

    # if boxclass == 'table'
    table: Optional[Dict] = None

    # text line information for text-type boxclasses
    max_fontsize: Optional[int] = None
    header_level: Optional[int] = 0  # one of 1..6 for title/section-header
    textlines: Optional[List[Dict]] = None


@dataclass
class PageLayout:
    page_number: int
    width: float
    height: float
    boxes: List[LayoutBox]
    full_ocred: bool = False  # whether the page is an OCR'd page
    fulltext: Optional[List[Dict]] = None  # full page text in extractDICT format
    words: Optional[List[Dict]] = None  # list of words with bbox
    links: Optional[List[Dict]] = None


@dataclass
class ParsedDocument:
    filename: Optional[str] = None  # source file name
    page_count: int = None
    toc: Optional[List[List]] = None  # e.g. [{'title': 'Intro', 'page': 1}]
    pages: List[PageLayout] = None
    metadata: Optional[Dict] = None
    from_bytes: bool = False  # whether loaded from bytes
    image_dpi: int = 150  # image resolution
    image_format: str = "png"  # 'png' or 'jpg'
    image_path: str = ""  # path to save images
    use_ocr: OCRMode = OCRMode.SELECT_KEEP_OLD  # if beneficial invoke OCR

    def to_markdown(
        self,
        header: bool = True,
        footer: bool = True,
        write_images: bool = False,
        embed_images: bool = False,
        ignore_code: bool = False,
        show_progress: bool = False,
        page_separators: bool = False,
        page_chunks: bool = False,
        **kwargs,
    ) -> Union[str, List[Dict]]:
        """
        Serialize ParsedDocument to markdown text.
        """
        if page_chunks:
            document_output = []
        else:
            document_output = ""

        if show_progress and len(self.pages) > 5:
            print(f"Generating markdown text...")
            this_iterator = ProgressBar(self.pages)
        else:
            this_iterator = self.pages
        for page in this_iterator:
            md_string = ""
            string_lengths = []
            # Make a mapping: box number -> list item hierarchy level
            list_item_levels = create_list_item_levels(page.boxes)

            for i, box in enumerate(page.boxes):
                clip = pymupdf.IRect(box.x0, box.y0, box.x1, box.y1)
                btype = box.boxclass

                # skip headers/footers if requested
                if btype == "page-header" and header is False:
                    string_lengths.append(len(md_string))
                    continue
                if btype == "page-footer" and footer is False:
                    string_lengths.append(len(md_string))
                    continue

                # pictures and formulas: either write image file or embed
                if btype in ("picture", "formula"):
                    if isinstance(box.image, str):
                        md_string += GRAPHICS_TEXT % box.image + "\n\n"
                    elif isinstance(box.image, bytes):
                        # make a base64 encoded string of the image
                        data = base64.b64encode(box.image).decode()
                        data = f"data:image/{self.image_format};base64," + data
                        md_string += GRAPHICS_TEXT % data + "\n\n"
                    else:
                        md_string += f"\n\n"

                    # output text in image if requested
                    if box.textlines:
                        if btype == "picture":
                            md_string += picture_text_to_md(
                                box.textlines,
                                ignore_code=ignore_code or page.full_ocred,
                                clip=clip,
                            )
                    string_lengths.append(len(md_string))
                    continue
                if btype == "table":
                    if box.table.get("html"):
                        md_string += box.table["html"] + "\n\n"
                        string_lengths.append(len(md_string))
                        continue
                    table_text = box.table["markdown"]
                    if page.full_ocred:
                        # remove code style if page was OCR'd
                        table_text = table_text.replace("`", "")
                    md_string += table_text + "\n\n"
                    string_lengths.append(len(md_string))
                    continue
                if not hasattr(box, "textlines"):
                    print(f"Warning: box {btype} has no textlines")
                    string_lengths.append(len(md_string))
                    continue
                if btype == "title":
                    md_string += title_to_md(box.header_level, box.textlines)
                    string_lengths.append(len(md_string))
                elif btype == "section-header":
                    md_string += section_hdr_to_md(box.header_level, box.textlines)
                    string_lengths.append(len(md_string))
                elif btype == "list-item":
                    md_string += list_item_to_md(box.textlines, list_item_levels[i])
                    string_lengths.append(len(md_string))
                elif btype == "footnote":
                    md_string += footnote_to_md(box.textlines)
                    string_lengths.append(len(md_string))
                else:  # treat as normal MD text
                    md_string += text_to_md(
                        box.textlines, ignore_code=ignore_code or page.full_ocred
                    )
                    string_lengths.append(len(md_string))
            if page_separators:
                md_string += f"--- end of {page.page_number=} ---\n\n"
            if not page_chunks:
                document_output += md_string
            else:
                chunk = make_page_chunk(self, page, md_string, string_lengths)
                document_output.append(chunk)
        return document_output

    def to_json(self, show_progress=False) -> str:
        # Serialize to JSON
        _ = show_progress
        class LayoutEncoder(json.JSONEncoder):
            def default(self, s):
                if isinstance(s, (bytes, bytearray)):
                    return base64.b64encode(s).decode()
                if isinstance(
                    s,
                    (
                        pymupdf.Rect,
                        pymupdf.Point,
                        pymupdf.Matrix,
                        pymupdf.IRect,
                        pymupdf.Quad,
                    ),
                ):
                    return list(s)
                if hasattr(s, "__dict__"):
                    return s.__dict__
                return super().default(s)

        js = json.dumps(self, cls=LayoutEncoder, ensure_ascii=False)
        return js

    def to_text(
        self,
        header: bool = True,
        footer: bool = True,
        ignore_code: bool = False,
        show_progress: bool = False,
        page_chunks: bool = False,
        table_format: str = "grid",
        table_max_width: int = 100,
        table_min_col_width: int = 10,
        **kwargs,
    ) -> Union[str, List[Dict]]:
        """
        Serialize ParsedDocument to plain text. Optionally omit page headers or footers.
        """
        if table_format not in tabulate.tabulate_formats:
            print(f"Warning: invalid table format '{table_format}', using 'grid'.")
            table_format = "grid"

        if page_chunks:
            document_output = []
        else:
            document_output = ""

        if show_progress and len(self.pages) > 5:
            print(f"Generating plain text ..")
            this_iterator = ProgressBar(self.pages)
        else:
            this_iterator = self.pages
        for page in this_iterator:
            text_string = ""
            string_lengths = []
            list_item_levels = create_list_item_levels(page.boxes)
            for i, box in enumerate(page.boxes):
                clip = pymupdf.IRect(box.x0, box.y0, box.x1, box.y1)
                btype = box.boxclass
                if btype == "page-header" and header is False:
                    string_lengths.append(len(text_string))
                    continue
                if btype == "page-footer" and footer is False:
                    string_lengths.append(len(text_string))
                    continue
                if btype in ("picture", "formula"):
                    if box.textlines and btype == "picture":
                        text_string += picture_text_to_text(
                            box.textlines,
                            ignore_code=ignore_code or page.full_ocred,
                            clip=clip,
                        )
                    string_lengths.append(len(text_string))

                elif btype == "table":
                    wrapped_table = wrap_table_for_tabulate(
                        box.table["extract"],
                        max_width=table_max_width,
                        min_col_width=table_min_col_width,
                    )
                    text_string += (
                        tabulate.tabulate(
                            wrapped_table, disable_numparse=True, tablefmt=table_format
                        )
                        + "\n\n"
                    )
                    string_lengths.append(len(text_string))

                elif btype == "list-item":
                    text_string += list_item_to_text(box.textlines, list_item_levels[i])
                    string_lengths.append(len(text_string))

                elif btype == "footnote":
                    text_string += footnote_to_text(box.textlines)
                    string_lengths.append(len(text_string))

                else:  # handle other cases as normal text
                    text_string += text_to_text(
                        box.textlines, ignore_code=ignore_code or page.full_ocred
                    )
                    string_lengths.append(len(text_string))

            if not page_chunks:
                document_output += text_string
            else:
                chunk = make_page_chunk(self, page, text_string, string_lengths)
                document_output.append(chunk)
        return document_output


def select_ocr_function():
    """Check availability of OCR tools and language data.

    Return the best OCR function available or None.
    """
    tessdata = None
    rapidocr_available = False
    paddleocr_available = False
    try:
        tessdata = pymupdf.get_tessdata()
    except:
        tessdata = None

    try:
        import rapidocr_onnxruntime

        rapidocr_available = True
        paddleocr_available = True
    except:
        pass
    if {tessdata, rapidocr_available, paddleocr_available} == {None, False, False}:
        return None
    if tessdata:
        if rapidocr_available:
            from pymupdf4llm.ocr import rapidtess_api

            print(
                "Using RapidOCR and Tesseract for OCR processing.",
                file=INFO_MESSAGES,
            )
            return rapidtess_api.exec_ocr
        elif paddleocr_available:
            from pymupdf4llm.ocr import paddletess_api

            print(
                "Using PaddleOCR and Tesseract for OCR processing.", file=INFO_MESSAGES
            )
            return paddletess_api.exec_ocr
        else:
            from pymupdf4llm.ocr import tesseract_api

            print("Using Tesseract for OCR processing.", file=INFO_MESSAGES)
            return tesseract_api.exec_ocr
    else:
        if rapidocr_available:
            from pymupdf4llm.ocr import rapidocr_api

            print("Using RapidOCR for OCR processing.", file=INFO_MESSAGES)
            return rapidocr_api.exec_ocr
        elif paddleocr_available:
            from pymupdf4llm.ocr import paddleocr_api

            print("Using PaddleOCR for OCR processing.", file=INFO_MESSAGES)
            return paddleocr_api.exec_ocr


def update_header_tags(pages, header_fontsizes):
    """Update title/section-header boxes with HTML header tags."""
    # List of up to 6 integer font sizes in descending order
    header_fontsizes = sorted(header_fontsizes, reverse=True)[:6]
    for page in pages:
        for box in page.boxes:
            if box.boxclass in ("title", "section-header"):
                if box.max_fontsize >= header_fontsizes[-1]:
                    box.header_level = header_fontsizes.index(box.max_fontsize) + 1
                else:
                    box.header_level = 6


def make_ocr_decision(page, use_ocr):
    """Decide whether to OCR a page.

    Returns a tuple (needs_ocr, ocr_spans) where needs_ocr is a boolean
    indicating whether OCR is needed, and ocr_spans is the number of
    existing OCR spans on the page (if any).
    """
    # OCR not desired at all
    if use_ocr == OCRMode.NEVER:
        return False, 0

    page_analysis = utils.analyze_page(page)

    needs_ocr = page_analysis.get("needs_ocr", False)
    # may be > 0 even if needs_ocr is False:
    ocr_spans = page_analysis.get("ocr_spans", 0)

    if ocr_spans and use_ocr in (OCRMode.FORCE_KEEP_OLD, OCRMode.SELECT_KEEP_OLD):
        # return False if old OCR should be kept
        return False, ocr_spans

    return needs_ocr, 0


def parse_document(
    doc,
    filename="",
    image_dpi=150,
    ocr_dpi=300,
    image_format="png",
    image_path="",
    pages=None,
    show_progress=False,
    embed_images=False,
    write_images=False,
    force_text=False,
    use_ocr=OCRMode.SELECT_KEEP_OLD,
    force_ocr=False,
    ocr_language="eng",
    ocr_function=None,
    render_html_tables=None,
    edge_threshold=None,
) -> ParsedDocument:
    if isinstance(doc, pymupdf.Document):
        mydoc = doc
    else:
        mydoc = pymupdf.open(doc)

    if mydoc.metadata["format"] == "Image":
        # Re-open as PDF to ensure we can successfully OCR the image.
        data = mydoc.convert_to_pdf()
        mydoc.close()
        mydoc = pymupdf.open(stream=data)

    if mydoc.is_pdf:
        # Remove StructTreeRoot to avoid possible performance degradation.
        # This package will not use the structure tree anyway.
        mypdf = pymupdf._as_pdf_document(mydoc)
        root = mupdf.pdf_dict_get(mupdf.pdf_trailer(mypdf), pymupdf.PDF_NAME("Root"))
        root.pdf_dict_del(pymupdf.PDF_NAME("StructTreeRoot"))
    else:
        use_ocr = OCRMode.NEVER
        if force_ocr:
            print(
                "Warning: OCR disabled because document is no PDF.",
                file=INFO_MESSAGES,
            )
        force_ocr = False

    if embed_images and write_images:
        raise ValueError("Cannot both embed and write images.")

    # collect font sizes of title and section_header
    header_fontsizes = set()

    document = ParsedDocument()
    document.filename = mydoc.name if mydoc.name else filename
    document.toc = mydoc.get_toc(simple=True)
    document.page_count = mydoc.page_count
    document.metadata = mydoc.metadata
    document.form_fields = utils.extract_form_fields_with_pages(mydoc)
    document.image_dpi = image_dpi
    document.image_format = image_format
    document.image_path = image_path
    document.pages = []
    document.force_text = force_text
    document.embed_images = embed_images
    document.write_images = write_images

    if force_ocr:
        use_ocr = OCRMode.FORCE_KEEP_OLD

    if use_ocr:
        if callable(ocr_function):
            document.use_ocr = use_ocr
        else:
            ocr_function = select_ocr_function()
            if callable(ocr_function):
                document.use_ocr = use_ocr
            else:
                document.use_ocr = OCRMode.NEVER
    else:
        document.use_ocr = OCRMode.NEVER

    if not callable(ocr_function):
        if document.use_ocr in (
            OCRMode.FORCE_DROP_OLD,
            OCRMode.FORCE_KEEP_OLD,
        ):
            raise ValueError("Force OCR is True but no OCR engine available.")
        if document.use_ocr != OCRMode.NEVER:
            print("Warning: No OCR engine available, OCR disabled.")
            document.use_ocr = OCRMode.NEVER

    if pages is None:
        page_filter = range(mydoc.page_count)
    elif isinstance(pages, int):
        while pages < 0:
            pages += mydoc.page_count
        page_filter = [pages]
    elif not hasattr(pages, "__getitem__"):
        raise ValueError("'pages' parameter must be an int, or a sequence of ints")
    else:
        page_filter = sorted(set(pages))

    if (
        not all(isinstance(p, int) for p in page_filter)
        or page_filter[-1] >= mydoc.page_count
    ):
        raise ValueError(
            f"'pages' parameter must be None, int, or a sequence of ints < {mydoc.page_count}."
        )

    if show_progress and len(page_filter) >= 5:
        print(f"Parsing {len(page_filter)} pages of '{document.filename}'...")
        page_filter = ProgressBar(page_filter)

    for pno in page_filter:
        page = mydoc.load_page(pno)
        page.remove_rotation()
        page_full_ocred = False
        PAGE_ANALYSIS = {}
        OCR_SPANS = 0
        needs_ocr, OCR_SPANS = make_ocr_decision(page, document.use_ocr)

        if needs_ocr:
            # execute OCR for the page replacing any previous OCR spans
            ocr_function(
                page,
                dpi=ocr_dpi,
                language=ocr_language,
                keep_ocr_text=False,
            )
            print(f"OCR on {page.number=}/{page.number+1}.", file=INFO_MESSAGES)

        textpage = page.get_textpage(flags=FLAGS, clip=pymupdf.INFINITE_RECT())
        blocks = textpage.extractDICT()["blocks"]

        # Execute the Layout module AFTER any OCR.
        layout_kwargs = {"return_raw": True}
        if edge_threshold is not None:
            layout_kwargs["edge_threshold"] = edge_threshold
        get_layout_locked(page, **layout_kwargs)

        # Optionally render tables as HTML, reusing this raw GNN layout
        # (get_layout is guarded to reuse it, so no second GNN pass).
        # Save/restore layout_information so table_html's internal normalization
        # does not disturb the layout path below.
        page_html_tables_list = None
        _render_html_tables = bool(render_html_tables)
        if _render_html_tables:
            _saved_raw_layout = page.layout_information
            try:
                from pymupdf4llm.helpers.table_html import page_html_tables
                page_html_tables_list = list(page_html_tables(page))
            except Exception as exc:
                # HTML table rendering failed on this page -> fall back to the
                # layout table path, but surface the reason (flushed to
                # pymupdf.message at the end of parse_document) rather than
                # swallowing it silently.
                print(
                    f"Warning: HTML table engine failed on page {page.number}: "
                    f"{type(exc).__name__}: {exc}",
                    file=INFO_MESSAGES,
                )
                page_html_tables_list = None
            finally:
                page.layout_information = _saved_raw_layout

        # Dictionary with details for all tables. Key is the bounding box
        # tuple, value is the original Layout info per table.
        table_infos = {}
        html_tables_by_box = {}
        textlines_by_box = {}

        new_layout_info = []  # will contain Layout boxes in non-"raw" format
        for b in page.layout_information:
            bbox = tuple(b["group_bbox"] + [b["class_name"]])
            new_layout_info.append(bbox)

            # store table info for later use in table extraction
            # we use the bounding box tuple as key for later matching
            if b["class_name"] == "table":
                key = tuple(pymupdf.IRect(b["group_bbox"]))
                table_infos[key] = b

        if _render_html_tables and page_html_tables_list:
            new_layout_info, html_tables_by_box, textlines_by_box = normalize_layout_boxes(
                new_layout_info,
                page_html_tables_list,
                fulltext=[b for b in blocks if b["type"] == 0],
            )

        page.layout_information = new_layout_info
        # Determine if any tables are present after HTML-table normalization.
        # Synthetic find_tables proposals are first-class table boxes on the
        # opt-in path, so this must be based on the normalized boxes.
        tables_exist = any(
            len(b) >= 5 and b[4] == "table" for b in page.layout_information
        )
        if not OCR_SPANS:  # some cleaning if no old OCR spans
            utils.clean_pictures(page, blocks)
            utils.add_image_orphans(page, blocks)

        # execute our own reading order function
        page.layout_information = utils.find_reading_order(
            page.rect, blocks, page.layout_information
        )
        fulltext = [b for b in blocks if b["type"] == 0]
        if tables_exist:
            table_blocks = [
                b for b in textpage.extractRAWDICT()["blocks"] if b["type"] == 0
            ]
        else:
            table_blocks = None

        words = []  # not yet activated
        links = [l for l in page.get_links() if l["kind"] == pymupdf.LINK_URI]
        pagelayout = PageLayout(
            page_number=page.number + 1,
            width=page.rect.width,
            height=page.rect.height,
            boxes=[],
            full_ocred=page_full_ocred,
            fulltext=fulltext,
            words=words,
            links=links,
        )
        for box in page.layout_information:
            layoutbox = LayoutBox(*box)
            clip = pymupdf.Rect(box[:4])

            if layoutbox.boxclass in ("picture", "formula"):
                if document.embed_images or document.write_images:
                    pix = page.get_pixmap(clip=clip, dpi=document.image_dpi)
                    irect = pymupdf.IRect(pix.irect)  # guard against empty images
                    if not irect.is_empty:
                        if document.embed_images:
                            layoutbox.image = pix.tobytes(document.image_format)
                        elif document.write_images:
                            img_filename = f"{document.filename}-{page.number+1:04d}-{len(pagelayout.boxes):02d}.{document.image_format}"
                            md_filename, save_img_filename = utils.md_path(
                                document.image_path, img_filename
                            )
                            layoutbox.image = md_filename
                            pix.save(save_img_filename)
                    else:
                        layoutbox.image = None
                else:
                    layoutbox.image = None
                if layoutbox.boxclass == "picture" and document.force_text:
                    # extract any text within the image box
                    layoutbox.textlines = [
                        {"bbox": l[0], "spans": l[1]}
                        for l in get_raw_lines(
                            textpage=None,
                            blocks=pagelayout.fulltext,
                            clip=clip,
                            ignore_invisible=False,
                            only_horizontal=False,
                        )
                    ]

            elif layoutbox.boxclass == "table":
                search_key = (layoutbox.x0, layoutbox.y0, layoutbox.x1, layoutbox.y1)
                html_tables = html_tables_by_box.get(tuple(pymupdf.IRect(clip)), [])

                if html_tables:
                    # Opt-in HTML mode: this box's table(s) were detected and
                    # rendered by table_html. Its reconstructed grid -- not the
                    # layout GNN grid or a best-fit rematch -- is the source of
                    # truth, so row_count/col_count/cells/extract describe the
                    # SAME grid the html shows: `cells` the post-span cell bbox
                    # matrix and `extract` the parallel plain-text matrix (both
                    # None for span-covered slots). `markdown` stays None --
                    # `html` is authoritative in this mode (to_markdown emits
                    # box.table["html"] directly).
                    single = html_tables[0] if len(html_tables) == 1 else None
                    layoutbox.table = {
                        "bbox": [layoutbox.x0, layoutbox.y0, layoutbox.x1, layoutbox.y1],
                        "row_count": single.get("rows") if single else None,
                        "col_count": single.get("cols") if single else None,
                        "cells": single.get("cells") if single else None,
                        "extract": single.get("extract") if single else None,
                        "markdown": None,
                        "html_tables": html_tables,
                        "html": "\n\n".join(item["html"] for item in html_tables),
                    }
                else:
                    # Non-HTML path (to_text, or a layout table box table_html did
                    # not render): keep the layout-grid extraction that feeds
                    # markdown/text. Because of intermediate processing the bbox
                    # might not match the original exactly, so take the best fit.
                    tab_details = None
                    if table_infos and table_blocks is not None:
                        key = max(table_infos.keys(), key=lambda k: utils.iou(k, search_key))
                        tab_dict = table_infos.get(key)
                        tab_details = get_table_details(tab_dict, table_blocks)

                    if tab_details is not None:
                        layoutbox.table = {
                            "bbox": list(tab_details.bbox),
                            "row_count": tab_details.row_count,
                            "col_count": tab_details.col_count,
                            "cells": tab_details.cells,
                            "extract": tab_details.extract,
                            "markdown": tab_details.markdown,
                        }
                    else:
                        layoutbox.table = {
                            "bbox": [layoutbox.x0, layoutbox.y0, layoutbox.x1, layoutbox.y1],
                            "row_count": None,
                            "col_count": None,
                            "cells": None,
                            "extract": None,
                            "markdown": "",
                        }

            else:
                # Handle text-like box classes:
                # Extract text line information within the box.
                # Each line is represented as its bbox and a list of spans.
                layoutbox.textlines = textlines_by_box.get(tuple(pymupdf.IRect(clip)))
                if layoutbox.textlines is None:
                    layoutbox.textlines = [
                        {"bbox": l[0], "spans": l[1]}
                        for l in get_raw_lines(
                            textpage=None,
                            blocks=pagelayout.fulltext,
                            clip=clip,
                            ignore_invisible=False,
                        )
                    ]
                # For each title/section_header compute and store the maximum
                # font size, to be used as a signal for header "#" prefix
                if layoutbox.boxclass in ("title", "section-header"):
                    max_fontsize = 0
                    for line in layoutbox.textlines:
                        for span in line["spans"]:
                            size = round(span["size"])
                            max_fontsize = max(max_fontsize, size)
                    header_fontsizes.add(max_fontsize)
                    layoutbox.max_fontsize = max_fontsize

            pagelayout.boxes.append(layoutbox)
        document.pages.append(pagelayout)
    if mydoc != doc:
        mydoc.close()
    msg_text = INFO_MESSAGES.getvalue()
    if msg_text:
        pymupdf.message("=== Document parser messages ===")
        pymupdf.message(msg_text)
    INFO_MESSAGES.truncate(0)  # empty the file-like object
    # Update title/section-header boxes with html header tags
    update_header_tags(document.pages, header_fontsizes)
    return document


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path

    filename = sys.argv[1]
    pdoc = parse_document(filename)
    # Path(filename).with_suffix(".json").write_text(pdoc.to_json())
    # Path(filename).with_suffix(".txt").write_text(pdoc.to_text(footer=False))
    md = pdoc.to_markdown(write_images=False, header=False, footer=False)
    Path(filename).with_suffix(".md").write_text(md)
