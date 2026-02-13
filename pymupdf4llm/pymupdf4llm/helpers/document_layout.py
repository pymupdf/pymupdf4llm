import base64
import io
import json
import os
import math
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union
import textwrap

import pymupdf
import tabulate
from pymupdf import mupdf
from pymupdf4llm.helpers import utils, check_ocr
from pymupdf4llm.helpers.get_text_lines import get_raw_lines


try:
    from tqdm import tqdm as ProgressBar
except ImportError:
    from pymupdf4llm.helpers.progress import ProgressBar

pymupdf.TOOLS.unset_quad_corrections(True)

INFO_MESSAGES = io.StringIO()
GRAPHICS_TEXT = "\n![](%s)\n"
OCR_FONTNAME = "GlyphLessFont"  # if encountered do not use "code" style
FLAGS = (
    0
    | pymupdf.TEXT_COLLECT_STYLES
    | pymupdf.TEXT_COLLECT_VECTORS
    | pymupdf.TEXT_PRESERVE_IMAGES
    | pymupdf.TEXT_ACCURATE_BBOXES
    | pymupdf.TEXT_MEDIABOX_CLIP
)
BULLETS = tuple(utils.BULLETS)


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
                "class": b.boxclass if b.boxclass != "table-fallback" else "table",
                "bbox": (
                    math.floor(b.x0),
                    math.floor(b.y0),
                    math.ceil(b.x1),
                    math.ceil(b.y1),
                ),
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
    This may be used to output code blocks.
    """
    line_count = len(textlines)
    mono = 0

    for l in textlines:
        all_mono = all(
            bool(s["flags"] & 8 and s["font"] != OCR_FONTNAME) for s in l["spans"]
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
    """
    Convert text extracted from images to plain text format.
    """
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
            line[j] = s["text"].strip()
        lines.append(line)
    tab_text = tabulate.tabulate(
        lines,
        tablefmt="grid",
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
    old_line = 0
    old_block = 0
    suffix = ""
    for i, s in enumerate(spans):
        # decode font properties
        prefix = ""
        superscript = s["flags"] & 1
        mono = s["flags"] & 8 and s["font"] != OCR_FONTNAME
        bold = s["flags"] & 16 or s["char_flags"] & 8
        italic = s["flags"] & 2
        strikeout = s["char_flags"] & 1

        # compute styling prefix and suffix
        if mono:
            prefix = "`" + prefix
        if bold:
            prefix = "**" + prefix
        if italic:
            prefix = "_" + prefix
        if strikeout:
            prefix = "~~" + prefix

        suffix = "".join(reversed(prefix))  # reverse of prefix

        span_text = s["text"].strip()  # remove leading/trailing spaces
        # convert intersecting link to markdown syntax
        # ltext = resolve_links(parms.links, s)
        ltext = ""  # TODO: implement link resolution
        if ltext:
            text = f"{hdr_string}{prefix}{ltext}{suffix} "
        else:
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


def section_hdr_to_md(textlines):
    """
    Convert "section-header" bboxes to markdown.
    This is treated as a level 2 header (##).
    The line text itself is handled like normal text.
    """
    spans = []
    for l in textlines:
        for s in l["spans"]:
            assert isinstance(s, dict)
            spans.append(s)
    output, suffix = get_styled_text(spans)
    return f"## {output}\n\n"


def title_to_md(textlines):
    """
    Convert "title" bboxes to markdown.
    This is treated as a level 1 header (#).
    The line text itself is handled like normal text.
    """
    spans = []
    for l in textlines:
        for s in l["spans"]:
            assert isinstance(s, dict)
            spans.append(s)
    output, suffix = get_styled_text(spans)
    return f"# {output}\n\n"


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
    """
    Convert text extracted from images to markdown format.
    """
    output = "**----- Start of picture text -----**<br>\n"
    for tl in textlines:
        line_text = " ".join([s["text"] for s in tl["spans"]])
        output += line_text.rstrip() + "<br>"
    output += "**----- End of picture text -----**<br>\n"
    return output + "\n\n"


def fallback_text_to_md(textlines, ignore_code: bool = False, clip=None):
    """
    Convert text extracted from images to markdown format.
    """
    span_count = max(len(tl["spans"]) for tl in textlines)
    output = "**----- Start of picture text -----**<br>\n"
    output += "|" * (span_count + 1) + "\n"
    output += "|" + "|".join(["---"] * span_count) + "|\n"
    for tl in textlines:
        ltext = "|" + "|".join([s["text"].strip() for s in tl["spans"]]) + "|\n"
        output += ltext
    output += "\n**----- End of picture text -----**<br>\n"
    return output + "\n\n"


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
    textlines: Optional[List[Dict]] = None


@dataclass
class PageLayout:
    page_number: int
    width: float
    height: float
    boxes: List[LayoutBox]
    full_ocred: bool = False  # whether the page is an OCR page
    text_ocred: bool = False  # whether the page text only is OCR'd
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
    use_ocr: bool = True  # if beneficial invoke OCR

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
                if btype in ("picture", "formula", "table-fallback"):
                    if isinstance(box.image, str):
                        md_string += GRAPHICS_TEXT % box.image + "\n\n"
                    elif isinstance(box.image, bytes):
                        # make a base64 encoded string of the image
                        data = base64.b64encode(box.image).decode()
                        data = f"data:image/{self.image_format};base64," + data
                        md_string += GRAPHICS_TEXT % data + "\n\n"
                    else:
                        md_string += f"**==> picture [{clip.width} x {clip.height}] intentionally omitted <==**\n\n"

                    # output text in image if requested
                    if box.textlines:
                        if btype == "picture":
                            md_string += picture_text_to_md(
                                box.textlines,
                                ignore_code=ignore_code or page.full_ocred,
                                clip=clip,
                            )
                        elif btype == "table-fallback":
                            md_string += fallback_text_to_md(
                                box.textlines,
                                ignore_code=ignore_code or page.full_ocred,
                                clip=clip,
                            )
                    string_lengths.append(len(md_string))
                    continue
                if btype == "table":
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
                    md_string += title_to_md(box.textlines)
                    string_lengths.append(len(md_string))
                elif btype == "section-header":
                    md_string += section_hdr_to_md(box.textlines)
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
                return self.super().default(s)

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
                if btype in ("picture", "formula", "table-fallback"):
                    text_string += f"==> picture [{clip.width} x {clip.height}] <==\n\n"
                    if box.textlines:
                        if btype == "picture":
                            text_string += picture_text_to_text(
                                box.textlines,
                                ignore_code=ignore_code or page.full_ocred,
                                clip=clip,
                            )
                        elif btype == "table-fallback":
                            text_string += fallback_text_to_text(
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
                        tabulate.tabulate(wrapped_table, tablefmt=table_format) + "\n\n"
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
    use_ocr=True,
    force_ocr=False,
    ocr_language="eng",
    ocr_function=None,
) -> ParsedDocument:
    if isinstance(doc, pymupdf.Document):
        mydoc = doc
    else:
        mydoc = pymupdf.open(doc)

    if mydoc.is_pdf:
        # Remove StructTreeRoot to avoid possible performance degradation.
        # We will not use the structure tree anyway.
        mypdf = pymupdf._as_pdf_document(mydoc)
        root = mupdf.pdf_dict_get(mupdf.pdf_trailer(mypdf), pymupdf.PDF_NAME("Root"))
        root.pdf_dict_del(pymupdf.PDF_NAME("StructTreeRoot"))

    TESSDATA = None
    if embed_images and write_images:
        raise ValueError("Cannot both embed and write images.")
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

    if use_ocr:
        if callable(ocr_function):
            document.use_ocr = True
        else:
            try:
                reason = "Tesseract language data not found"
                TESSDATA = pymupdf.get_tessdata()
                document.use_ocr = True
            except Exception as e:
                print(f"OCR disabled because {reason}.")
                document.use_ocr = False
    else:
        document.use_ocr = False

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
            f"'pages' parameter must be None, int, or a sequence of ints less than {mydoc.page_count}."
        )

    if show_progress and len(page_filter) > 5:
        print(f"Parsing {len(page_filter)} pages of '{document.filename}'...")
        page_filter = ProgressBar(page_filter)

    for pno in page_filter:
        page = mydoc.load_page(pno)
        page.remove_rotation()
        page_full_ocred = False
        page_text_ocred = False

        if force_ocr:
            ocr_function(page, dpi=ocr_dpi)
            # do not OCR again and indicate presence of OCR text
            decision = {"should_ocr": False, "has_ocr_text": True}
            page_full_ocred = True

        textpage = page.get_textpage(flags=FLAGS, clip=pymupdf.INFINITE_RECT())
        blocks = textpage.extractDICT()["blocks"]

        if not force_ocr:
            # check if this page should be OCR'd
            if document.use_ocr:
                decision = check_ocr.should_ocr_page(
                    page,
                    dpi=ocr_dpi,
                    blocks=blocks,
                )
            else:
                decision = {"should_ocr": False}
                page_analysis = utils.analyze_page(page, blocks)
                decision["has_ocr_text"] = page_analysis["ocr_spans"] > 0

        if decision["has_ocr_text"]:  # prevent MD styling if OCR'd
            page_full_ocred = True

        if decision["should_ocr"]:
            # We should be OCR: check full-page vs. text-only
            if decision.get("pixmap"):
                pix = decision["pixmap"]  # retrieve the Pixmap
                if not callable(ocr_function):
                    # use built-in OCR
                    pdf_data = pix.pdfocr_tobytes(
                        language=ocr_language,
                        tessdata=TESSDATA,
                    )  # OCR it
                    ocr_pdf = pymupdf.open("pdf", pdf_data)  # get the OCR'd PDF
                    ocr_page = ocr_pdf[0]  # this is its OCR'd page
                    # remove everything except the text
                    ocr_page.add_redact_annot(ocr_page.rect)
                    ocr_page.apply_redactions(
                        images=pymupdf.PDF_REDACT_IMAGE_REMOVE,
                        graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                        text=pymupdf.PDF_REDACT_TEXT_NONE,
                    )
                    # copy text over to original page
                    page.show_pdf_page(page.rect, ocr_pdf, 0)
                    ocr_pdf.close()  # discard temporary OCR PDF
                    del ocr_pdf
                else:
                    # custom OCR function: will directly modify the page
                    ocr_function(page, pixmap=pix)

                print(
                    f"Full-page OCR on {page.number=}/{page.number+1}.",
                    file=INFO_MESSAGES,
                )
                textpage = page.get_textpage(flags=FLAGS, clip=pymupdf.INFINITE_RECT())
                blocks = textpage.extractDICT()["blocks"]
                page_full_ocred = True
            else:
                print(
                    f"Text-only OCR on {page.number=}/{page.number+1}.",
                    file=INFO_MESSAGES,
                )
                blocks = check_ocr.repair_blocks(blocks, page)
                page_text_ocred = True

        page.get_layout()
        # Determine if any tables are present. If False, we skip any table-related efforts.
        tables_exist = any(b for b in page.layout_information if b[4] == "table")
        if not page_full_ocred:
            utils.clean_pictures(page, blocks)
            utils.add_image_orphans(page, blocks)
            if tables_exist:
                utils.clean_tables(page, blocks)
        page.layout_information = utils.find_reading_order(
            page.rect, blocks, page.layout_information
        )

        # identify vector graphics to help find tables
        if tables_exist and not page_full_ocred:
            all_lines, all_boxes = utils.complete_table_structure(page)
        else:
            all_lines, all_boxes = [], []
        if tables_exist:
            tbf = page.find_tables(
                strategy="lines_strict", add_lines=all_lines, add_boxes=all_boxes
            )
        else:
            tbf = None
        fulltext = [b for b in blocks if b["type"] == 0]
        if tables_exist:
            # tables are present on page:
            if not (page_full_ocred or page_text_ocred):
                # we need the by-character extraction if no OCR
                table_blocks = [
                    b for b in textpage.extractRAWDICT()["blocks"] if b["type"] == 0
                ]
            else:
                table_blocks = fulltext
        else:
            table_blocks = None

        words = []  # not yet activated
        links = page.get_links()
        pagelayout = PageLayout(
            page_number=page.number + 1,
            width=page.rect.width,
            height=page.rect.height,
            boxes=[],
            full_ocred=page_full_ocred,
            text_ocred=page_text_ocred,
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
                            ignore_invisible=not pagelayout.full_ocred,
                            only_horizontal=False,
                        )
                    ]

            elif layoutbox.boxclass == "table":
                # This is either a table detected by native TableFinder or by
                # MuPDF's table structure recognition (which may fail).
                # If the structure was not detected, we output an image.
                # A table is represented as a dict with bbox, row_count,
                # col_count, cells, extract (2D list of cell texts), and the
                # markdown string.

                try:  # guard against table structure detection failure
                    table = [
                        tab
                        for tab in tbf.tables
                        if pymupdf.table._iou(tab.bbox, clip) > 0.6
                    ][0]
                    cells = [[c for c in row.cells] for row in table.rows]
                    row_count = table.row_count
                    if table.header.external:  # if the header ioutside table
                        cells.insert(0, table.header.cells)  # insert a row
                        row_count += 1  # increase row count

                    layoutbox.table = {
                        "bbox": list(table.bbox),
                        "row_count": row_count,
                        "col_count": table.col_count,
                        "cells": cells,
                    }

                    layoutbox.table["extract"] = utils.table_extract(
                        table_blocks,
                        layoutbox,
                        ocrpage=(pagelayout.full_ocred or pagelayout.text_ocred),
                    )

                    layoutbox.table["markdown"] = utils.table_to_markdown(
                        table_blocks,
                        layoutbox,
                        ocrpage=(pagelayout.full_ocred or pagelayout.text_ocred),
                        markdown=True,
                    )

                except Exception as e:
                    # print(f"table detection error '{e}' on page {page.number+1}")
                    layoutbox.boxclass = "table-fallback"
                    # table structure not detected: treat like an image
                    if document.embed_images or document.write_images:
                        pix = page.get_pixmap(clip=clip, dpi=document.image_dpi)
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
                    layoutbox.textlines = [
                        {"bbox": l[0], "spans": l[1]}
                        for l in get_raw_lines(
                            textpage=None,
                            blocks=pagelayout.fulltext,
                            clip=clip,
                            ignore_invisible=not pagelayout.full_ocred,
                        )
                    ]
                    if layoutbox.textlines and (
                        len(layoutbox.textlines) == 1
                        or max(len(l["spans"]) for l in layoutbox.textlines) < 2
                    ):
                        # treat as text if only one line or only one span per line:
                        layoutbox.boxclass = "text"
            else:
                # Handle text-like box classes:
                # Extract text line information within the box.
                # Each line is represented as its bbox and a list of spans.
                layoutbox.textlines = [
                    {"bbox": l[0], "spans": l[1]}
                    for l in get_raw_lines(
                        textpage=None,
                        blocks=pagelayout.fulltext,
                        clip=clip,
                        ignore_invisible=not pagelayout.full_ocred,
                    )
                ]
            pagelayout.boxes.append(layoutbox)
        document.pages.append(pagelayout)
    if mydoc != doc:
        mydoc.close()
    msg_text = INFO_MESSAGES.getvalue()
    if msg_text:
        pymupdf.message("=== Document parser messages ===")
        pymupdf.message(msg_text)
        INFO_MESSAGES.truncate(0)  # empty the file-like object
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
