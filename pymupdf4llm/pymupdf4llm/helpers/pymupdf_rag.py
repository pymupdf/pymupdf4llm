"""This script accepts a PDF document filename and converts it to a text file
in Markdown format, compatible with the GitHub standard.

It must be invoked with the filename like this:

python pymupdf_rag.py input.pdf [-pages PAGES]

The "PAGES" parameter is a string (containing no spaces) of comma-separated
page numbers to consider. Each item is either a single page number or a
number range "m-n". Use "N" to address the document's last page number.
Example: "-pages 2-15,40,43-N"

It will produce a markdown text file called "input.md".

Text will be sorted in Western reading order. Any table will be included in
the text in markdwn format as well.

Dependencies
-------------
PyMuPDF v1.25.5 or later

Copyright and License
----------------------
Copyright (C) 2024-2025 Artifex Software, Inc.

PyMuPDF4LLM is free software: you can redistribute it and/or modify it under the
terms of the GNU Affero General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option)
any later version.

Alternative licensing terms are available from the licensor.
For commercial licensing, see <https://www.artifex.com/> or contact
Artifex Software, Inc., 39 Mesa Street, Suite 108A, San Francisco,
CA 94129, USA, for further information.
"""

import os
import re
import string
import unicodedata
from binascii import b2a_base64
from collections import defaultdict
from dataclasses import dataclass

import pymupdf
from pymupdf import mupdf
from pymupdf4llm.helpers.get_text_lines import get_raw_lines
from pymupdf4llm.helpers.multi_column import column_boxes
from pymupdf4llm.helpers.utils import (
    BULLETS,
    _merge_single_letter_word_splits,
    _normalize_table_br_tags,
    extract_cells,
    REPLACEMENT_CHARACTER,
    startswith_bullet,
    is_white,
    bbox_is_empty,
    almost_in_bbox,
    outside_bbox,
    bbox_in_bbox,
    intersect_rects,
)

try:
    from tqdm import tqdm as ProgressBar
except ImportError:
    from pymupdf4llm.helpers.progress import ProgressBar

pymupdf.TOOLS.unset_quad_corrections(True)

GRAPHICS_TEXT = "\n![](%s)\n"

# configuration constants for ASCII table rendering
# These can be monkeypatched in tests to exercise width-limiting logic.
MAX_TOTAL_WIDTH = 120  # maximum allowed width for the entire ASCII table
MIN_COL_WIDTH = 7      # smallest width any column may have


class IdentifyHeaders:
    """Compute data for identifying header text.

    All non-white text from all selected pages is extracted and its font size
    noted as a rounded value.
    The most frequent font size (and all smaller ones) is taken as body text
    font size.
    Larger font sizes are mapped to strings of multiples of '#', the header
    tag in Markdown, which in turn is Markdown's representation of HTML's
    header tags <h1> to <h6>.
    Larger font sizes than body text but smaller than the <h6> font size are
    represented as <h6>.
    """

    def __init__(
        self,
        doc: str,
        pages: list = None,
        body_limit: float = 12,  # force this to be body text
        max_levels: int = 6,  # accept this many header levels
    ):
        """Read all text and make a dictionary of fontsizes.

        Args:
            doc: PDF document or filename
            pages: consider these page numbers only
            body_limit: treat text with larger font size as a header
        """
        if not isinstance(max_levels, int) or max_levels not in range(1, 7):
            raise ValueError("max_levels must be an integer between 1 and 6")
        if isinstance(doc, pymupdf.Document):
            mydoc = doc
        else:
            mydoc = pymupdf.open(doc)


        if mydoc.is_pdf:
            # remove StructTreeRoot to avoid possible performance degradation
            mypdf = pymupdf._as_pdf_document(mydoc)
            root = mupdf.pdf_dict_get(
                mupdf.pdf_trailer(mypdf), pymupdf.PDF_NAME("Root")
            )
            root.pdf_dict_del(pymupdf.PDF_NAME("StructTreeRoot"))

            pages = range(mydoc.page_count)

        fontsizes = defaultdict(int)
        for pno in pages:
            page = mydoc.load_page(pno)
            blocks = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)["blocks"]
            for span in [
                s
                for b in blocks
                for l in b["lines"]
                for s in l["spans"]
                if not is_white(s["text"])
            ]:
                fontsz = round(span["size"])
                fontsizes[fontsz] += len(span["text"].strip())

        if mydoc != doc:
            # if opened here, close it now
            mydoc.close()

        self.header_id = {}
        temp = sorted(
            [(k, v) for k, v in fontsizes.items()], key=lambda i: (i[1], i[0])
        )
        if temp:
            # most frequent font size
            self.body_limit = max(body_limit, temp[-1][0])
        else:
            self.body_limit = body_limit

        # identify up to 6 font sizes as header candidates
        sizes = sorted(
            [f for f in fontsizes.keys() if f > self.body_limit],
            reverse=True,
        )[:max_levels]

        # make the header tag dictionary
        for i, size in enumerate(sizes, start=1):
            self.header_id[size] = "#" * i + " "
        if self.header_id.keys():
            self.body_limit = min(self.header_id.keys()) - 1

    def get_header_id(self, span: dict, page=None) -> str:
        """Return appropriate markdown header prefix.

        Given a text span from a "dict"/"rawdict" extraction, determine the
        markdown header prefix string of 0 to n concatenated '#' characters.
        """
        fontsize = round(span["size"])
        if fontsize <= self.body_limit:
            return ""
        hdr_id = self.header_id.get(fontsize, "")
        return hdr_id


class TocHeaders:
    """Compute data for identifying header text.

    This is an alternative to IdentifyHeaders. Instead of running through the
    full document to identify font sizes, it uses the document's Table Of
    Contents (TOC) to identify headers on pages.
    Like IdentifyHeaders, this also is no guarantee to find headers, but it
    represents a good chance for appropriately built documents. In such cases,
    this method can be very much faster and more accurate, because we can
    directly use the hierarchy level of TOC items to ientify the header level.
    Examples where this works very well are the Adobe PDF documents.
    """

    def __init__(self, doc: str):
        """Read and store the TOC of the document."""
        if isinstance(doc, pymupdf.Document):
            mydoc = doc
        else:
            mydoc = pymupdf.open(doc)

        self.TOC = doc.get_toc()
        if mydoc != doc:
            # if opened here, close it now
            mydoc.close()

    def get_header_id(self, span: dict, page=None) -> str:
        """Return appropriate markdown header prefix.

        Given a text span from a "dict"/"rawdict" extraction, determine the
        markdown header prefix string of 0 to n concatenated '#' characters.
        """
        if not page:
            return ""
        # check if this page has TOC entries with an actual title
        my_toc = [t for t in self.TOC if t[1] and t[-1] == page.number + 1]
        if not my_toc:  # no TOC items present on this page
            return ""
        # Check if the span matches a TOC entry. This must be done in the
        # most forgiving way: exact matches are rare animals.
        text = span["text"].strip()  # remove leading and trailing whitespace
        for t in my_toc:
            title = t[1].strip()  # title of TOC entry
            lvl = t[0]  # level of TOC entry
            if text.startswith(title) or title.startswith(text):
                # found a match: return the header tag
                return "#" * lvl + " "
        return ""


# store relevant parameters here
@dataclass
class Parameters:
    pass


def wrap_text_by_bbox(text, cell_rect, textpage=None, avg_font_size=None):
    """Wrap text to fit within cell width based on bbox dimensions.
    
    Args:
        text: Text to wrap
        cell_rect: pymupdf.Rect with cell dimensions
        textpage: Optional TextPage to extract font size from
        avg_font_size: Optional average font size (if not provided, will estimate)
    
    Returns:
        Text with line breaks inserted to fit cell width
    """
    if not text or not cell_rect or cell_rect.is_empty:
        return text
    
    # Estimate font size if not provided
    if avg_font_size is None:
        if textpage:
            # Try to extract font size from text in the cell
            try:
                blocks = textpage.extractDICT(clip=cell_rect)["blocks"]
                font_sizes = []
                for block in blocks:
                    if block.get("type") == 0:  # text block
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                if span.get("text", "").strip():
                                    font_sizes.append(span.get("size", 0))
                if font_sizes:
                    avg_font_size = sum(font_sizes) / len(font_sizes)
                else:
                    avg_font_size = 10  # default fallback
            except:
                avg_font_size = 10  # default fallback
        else:
            avg_font_size = 10  # default fallback
    
    # Estimate character width: typically 0.5-0.6 * font_size for proportional fonts
    # Using 0.55 as a reasonable average
    char_width = avg_font_size * 0.55
    if char_width <= 0:
        char_width = 5.5  # fallback
    
    # Calculate max characters per line (leave some margin)
    cell_width = cell_rect.width
    max_chars_per_line = max(10, int(cell_width / char_width) - 2)  # -2 for margin
    
    if len(text) <= max_chars_per_line:
        return text
    
    # Break text into lines
    lines = []
    current_pos = 0
    dynamic_max_chars = max_chars_per_line  # Allow dynamic adjustment
    
    while current_pos < len(text):
        # If remaining text fits in one line
        if current_pos + dynamic_max_chars >= len(text):
            lines.append(text[current_pos:])
            break
        
        # Find break point
        break_pos = current_pos + dynamic_max_chars
        
        # If break point is in the middle of a word, go back to find a space
        if break_pos < len(text) and text[break_pos] != ' ':
            # Look backwards for a space
            space_pos = text.rfind(' ', current_pos, break_pos + 1)
            if space_pos > current_pos:
                # Found a space, break there
                break_pos = space_pos + 1  # Include the space, will be stripped
            else:
                # No space found - word is too long for current line width
                # Find the end of the current word
                word_end = break_pos
                while word_end < len(text) and text[word_end] != ' ':
                    word_end += 1
                
                # Calculate how many characters the word needs
                word_length = word_end - current_pos
                
                # If word is longer than current max, increase column width to fit the entire word
                if word_length > dynamic_max_chars:
                    # Increase the column width to accommodate the entire word
                    dynamic_max_chars = word_length
                
                # Break at the end of the word (which may now fit due to increased width)
                break_pos = word_end
        
        # Extract line and move forward
        line = text[current_pos:break_pos].rstrip()
        if line:
            lines.append(line)
        current_pos = break_pos
        
        # Skip leading spaces on next line
        while current_pos < len(text) and text[current_pos] == ' ':
            current_pos += 1
        
        # Reset dynamic_max_chars for next line (but keep it if it was increased)
        # This allows subsequent lines to also benefit from the increased width
        # if they contain similarly long words
    
    return '\n'.join(lines)

def matrix_to_ascii(matrix):
    """Convert a matrix (list of lists) into a simple ASCII table.
    
    This version enforces a STRICT maximum width to prevent line wrapping
    in text editors, ensuring visual alignment even at 100% zoom.
    """
    # helper used to clean up cell contents before any width calculations.
    # removes HTML <br> tags, collapses real newlines, merges single-letter
    # splits and normalizes whitespace.  This guarantees that ASCII layout is
    # computed on the same normalized text seen in Markdown output.
    def _sanitize(text: str) -> str:
        if not text:
            return ""
        txt = _normalize_table_br_tags(str(text))
        # detect cases where a real newline broke a word and the first
        # letter of the continuation should join the previous word while
        # the remainder becomes a new word.  Example:
        #   "stretchin\ngvibration" -> "stretching vibration"
        txt = re.sub(r"([A-Za-z])\n([A-Za-z])", r"\1\2 ", txt)
        txt = txt.replace("\n", " ")
        txt = txt.replace("\u00a0", " ").replace("\t", " ")
        txt = _merge_single_letter_word_splits(txt)
        txt = re.sub(r"[ \t]+", " ", txt)

        return txt.strip()

    import textwrap
    import math
    from functools import lru_cache

    if not matrix:
        return ""

    @lru_cache(maxsize=512)
    def _display_width(text: str) -> int:
        """Return terminal-like display width for alignment.
        
        Cached to avoid recalculating for the same text.
        """
        try:
            from wcwidth import wcwidth  # type: ignore
        except Exception:
            wcwidth = None

        width = 0
        for ch in text:
            if ch == "\n":
                continue
            if wcwidth:
                w = wcwidth(ch)
                width += w if w > 0 else 1
                continue
            if unicodedata.combining(ch):
                continue
            # Treat Ambiguous width as wide to match terminal rendering for symbols like ℃, Ⅱ.
            width += 2 if unicodedata.east_asian_width(ch) in ("W", "F", "A") else 1
        return width

    def _pad_display(text: str, width: int) -> str:
        # Pad by display width to keep column alignment consistent.
        pad = width - _display_width(text)
        if pad <= 0:
            return text
        return text + (" " * pad)

    def _max_word_width(text: str) -> int:
        words = text.split()
        return max((_display_width(w) for w in words), default=0)

    def _max_line_width(text: str) -> int:
        lines = text.split("\n")
        return max((_display_width(l) for l in lines), default=0)

    def _wrap_by_display_width(text: str, width: int) -> list:
        if width <= 0:
            return [text] if text else [""]
        words = text.split(" ")
        lines = []
        current = ""
        for word in words:
            if not current:
                current = word
                continue
            trial = f"{current} {word}"
            if _display_width(trial) <= width:
                current = trial
            else:
                lines.append(current)
                current = word
        if current or not lines:
            lines.append(current)
        return lines

    row_cells = []
    row_texts = []
    max_cols = 0

    for row in matrix:
        current_row_cells = []
        current_row_texts = []
        for cell in row:
            current_row_cells.append(cell)
            if isinstance(cell, dict):
                if cell.get("is_merged") and cell.get("merged_from"):
                    text = ""
                else:
                    raw_text = str(cell.get("text", "") or "")
                    text = _sanitize(raw_text)
            else:
                raw_text = "" if cell is None else str(cell)
                text = _sanitize(raw_text)
            current_row_texts.append(text)
        row_cells.append(current_row_cells)
        row_texts.append(current_row_texts)
        max_cols = max(max_cols, len(current_row_texts))

    if max_cols == 0:
        return ""
    
    border_overhead = max_cols + 1
    available_text_space = MAX_TOTAL_WIDTH - border_overhead

    # Determine "Natural" required width for each column (longest word/line)
    # For colspan cells, distribute required width across the spanned columns.
    natural_widths = [0] * max_cols
    for r_idx, row in enumerate(row_texts):
        for idx_col in range(max_cols):
            text = row[idx_col] if idx_col < len(row) else ""
            if not text:
                continue
            cell = row_cells[r_idx][idx_col] if idx_col < len(row_cells[r_idx]) else None
            if isinstance(cell, dict) and cell.get("is_merged") and cell.get("merged_from"):
                continue

            colspan = 1
            if isinstance(cell, dict):
                colspan = max(1, int(cell.get("colspan", 1)))
            colspan = min(colspan, max_cols - idx_col)

            # Check longest unbroken word to avoid ugly splits if possible
            max_word = _max_word_width(text)
            # Also check longest existing line (if pre-formatted)
            max_line = _max_line_width(text)
            required = max(max_word, max_line)

            # Account for merged column separators when distributing width
            required_no_separators = max(0, required - (colspan - 1))
            per_col = int(math.ceil(required_no_separators / colspan)) if colspan > 1 else required_no_separators

            for k in range(idx_col, idx_col + colspan):
                natural_widths[k] = max(natural_widths[k], per_col)

    # Determine Final Widths.
    # To avoid breaking words, keep columns at least their natural width.
    final_widths = [max(w, MIN_COL_WIDTH) for w in natural_widths]

    col_widths = final_widths

    # enforce strict maximum table width by shrinking columns if required
    border_overhead = max_cols + 1
    available_text_space = MAX_TOTAL_WIDTH - border_overhead
    total = sum(col_widths)
    if total > available_text_space:
        # greedily reduce the widest column until the table fits
        while total > available_text_space:
            idx = max(
                (i for i, w in enumerate(col_widths) if w > MIN_COL_WIDTH),
                default=None,
            )
            if idx is None:
                break
            col_widths[idx] -= 1
            total -= 1

    # wrap text into cells; widen columns when a wrap would split a word
    changed = True
    while changed:
        changed = False
        for r_idx in range(len(row_texts)):
            for c_idx in range(max_cols):
                text = row_texts[r_idx][c_idx]
                cell = row_cells[r_idx][c_idx]
                if not text:
                    continue

                # Determine effective width for this specific cell (handling merged cols)
                colspan = 1
                if isinstance(cell, dict):
                    colspan = max(1, int(cell.get("colspan", 1)))
                
                eff_width = 0
                if c_idx + colspan <= max_cols:
                    for k in range(c_idx, c_idx + colspan):
                        eff_width += col_widths[k]
                    eff_width += (colspan - 1)
                else:
                    eff_width = col_widths[c_idx]

                wrap_width = max(1, eff_width)

                paragraphs = text.split("\n")
                all_lines = []
                for p in paragraphs:
                    if not p.strip():
                        all_lines.append("")
                        continue
                    lines = _wrap_by_display_width(p, wrap_width)
                    all_lines.extend(lines)

                # Record wrapped text
                row_texts[r_idx][c_idx] = "\n".join(all_lines)

                # check if any wrapped line exceeded our budget
                used = max((_display_width(l) for l in all_lines), default=0)
                if used > wrap_width:
                    extra = used - wrap_width
                    if colspan == 1:
                        col_widths[c_idx] += extra
                    else:
                        per = int(math.ceil(extra / colspan))
                        for k in range(c_idx, c_idx + colspan):
                            col_widths[k] += per
                    changed = True
        # if we increased any column widths, rerun the wrapping to ensure consistency


    def build_content_line(row_index, line_in_cell=0):
        parts = []
        col = 0
        while col < max_cols:
            cell = row_cells[row_index][col]
            text = row_texts[row_index][col]
            
            # Check overlap from above (Rowspan)
            covered_by_above = None
            if row_index > 0:
                r_up = row_index - 1
                while r_up >= 0:
                    c_check = row_cells[r_up][col]
                    if isinstance(c_check, dict) and not c_check.get("is_merged"):
                        r_span = c_check.get("rowspan", 1)
                        if (r_up + r_span) > row_index:
                            covered_by_above = c_check
                        break # Found the physical cell governing this column
                    r_up -= 1
            
            # Determine content to display
            display_text = ""
            if covered_by_above:
                display_text = ""
            elif isinstance(cell, dict) and cell.get("is_merged") and cell.get("merged_from"):
                pass 
            else:
                # Primary cell
                if "\n" in text:
                    lines = text.split("\n")
                    if line_in_cell < len(lines):
                        display_text = lines[line_in_cell]
                else:
                    if line_in_cell == 0:
                        display_text = text

            # Handle Colspan
            colspan = 1
            if isinstance(cell, dict):
                colspan = max(1, int(cell.get("colspan", 1)))
            colspan = min(colspan, max_cols - col)

            # Calculate total display width
            total_width = 0
            for k in range(col, col + colspan):
                total_width += col_widths[k]
            total_width += (colspan - 1)

            segment = _pad_display(display_text, total_width) + "|"
            parts.append(segment)
            
            col += colspan

        return "|" + "".join(parts)

    def build_separator_line(row_index):
        parts = []
        col = 0
        while col < max_cols:
            cell = row_cells[row_index][col]
            
            is_crossing = False
            
            # Find primary to check rowspan
            primary = cell
            if isinstance(cell, dict) and cell.get("is_merged") and cell.get("merged_from"):
                p_row, p_col = cell.get("merged_from")
                if p_row < len(row_cells) and p_col < len(row_cells[p_row]):
                    primary = row_cells[p_row][p_col]
            
            if isinstance(primary, dict):
                start = primary.get("row", row_index)
                span = primary.get("rowspan", 1)
                if start + span > row_index + 1:
                    is_crossing = True
            
            colspan = 1
            if isinstance(cell, dict):
                colspan = max(1, int(cell.get("colspan", 1)))
            colspan = min(colspan, max_cols - col)
            
            width = 0
            for k in range(col, col + colspan):
                width += col_widths[k]
            width += (colspan - 1)
            
            char = " " if is_crossing else "-"
            parts.append((char * width) + "|")
            
            col += colspan
            
        return "|" + "".join(parts)

    def get_max_lines(row_index):
        m = 1
        for t in row_texts[row_index]:
            m = max(m, len(t.split('\n')))
        return m

    output = []
    
    # Top Border
    # Calculate exact length based on columns
    total_table_width = sum(col_widths) + max_cols + 1 # widths + N separators + 1 start
    output.append("-" * total_table_width)
    
    for r in range(len(row_texts)):
        lines = get_max_lines(r)
        for l in range(lines):
            output.append(build_content_line(r, l))
        
        # Bottom separator for this row
        if r < len(row_texts) - 1:
            output.append(build_separator_line(r))
            
    output.append("-" * total_table_width)

    # Ensure all lines have exactly the same display width.
    max_len = max(_display_width(line) for line in output) if output else 0
    normalized_output = []
    for line in output:
        if set(line) == {"-"}:
            normalized_output.append("-" * max_len)
            continue
        pad = max_len - _display_width(line)
        if pad > 0:
            line = line + (" " * pad)
        normalized_output.append(line)

    return "\n".join(normalized_output)

def merge_split_tables(tables, y_gap_factor: float = 1.5, x_tolerance_factor: float = 0.05):
    """Merge table metadata entries that are parts of the same logical table.

    Sometimes `page.find_tables()` returns one physical table in the PDF
    as two or more independent tables (for example, when there is a small
    whitespace or a short text line between blocks of the same table). This
    would otherwise lead to reading them as separate tables.

    This helper applies a simple heuristic to detect such cases and join
    them into one logical table:
    - same number of columns;
    - horizontally well aligned (x0 / x1 very similar);
    - small vertical distance relative to the average row height.
    """

    if not tables:
        return tables

    # Sort tables from top to bottom
    indices = list(range(len(tables)))
    indices.sort(key=lambda i: (tables[i]["bbox"][1], tables[i]["bbox"][0]))

    merged_tables = []
    used = set()

    for idx in indices:
        if idx in used:
            continue

        base = tables[idx]
        base_bbox = pymupdf.Rect(base["bbox"])
        base_cols = base.get("columns", 0)
        base_matrix = [row[:] for row in (base.get("matrix") or [])]
        base_rows = base.get("rows", 0) or len(base_matrix)

        # Average row height to calibrate the acceptable vertical gap
        avg_row_height = (
            base_bbox.height / base_rows if base_rows > 0 and base_bbox.height > 0 else 0
        )
        x_tol = max(5, base_bbox.width * x_tolerance_factor)

        for j in indices:
            if j == idx or j in used:
                continue

            other = tables[j]
            if other.get("columns", 0) != base_cols:
                continue

            other_bbox = pymupdf.Rect(other["bbox"])

            # Same horizontal span (same x-limits with a small tolerance)
            if (
                abs(base_bbox.x0 - other_bbox.x0) > x_tol
                or abs(base_bbox.x1 - other_bbox.x1) > x_tol
            ):
                continue

            # Vertical distance between bounding boxes (if they do not overlap)
            if other_bbox.y0 >= base_bbox.y1:
                gap = other_bbox.y0 - base_bbox.y1
            elif base_bbox.y0 >= other_bbox.y1:
                gap = base_bbox.y0 - other_bbox.y1
            else:
                gap = 0  # vertical overlap

            # Check if gap is small enough (first condition: small vertical distance)
            gap_is_small = False
            if avg_row_height <= 0:
                # Without a good estimate for row height, only accept very small gaps
                gap_is_small = gap <= 5
            else:
                gap_is_small = gap <= avg_row_height * y_gap_factor

            # If gap is not small, don't merge (skip this table)
            if not gap_is_small:
                continue

            # If we get here, the gap is small enough and columns are aligned.
            # This is sufficient to merge. Headers check is only to avoid duplicating header row.
            other_matrix = other.get("matrix") or []
            if not other_matrix:
                used.add(j)
                continue

            # Check headers only to avoid duplicating the header row, not to block merging
            # If gap is small and columns align, we merge regardless of headers
            rows_to_add = other_matrix
            if base_matrix and other_matrix:
                header1 = [c.get("text", "").strip() if isinstance(c, dict) else str(c).strip() if c else "" for c in base_matrix[0]]
                header2 = [c.get("text", "").strip() if isinstance(c, dict) else str(c).strip() if c else "" for c in other_matrix[0]]
                
                # Check if second table has a header (at least one non-empty cell)
                has_header2 = any(h2 for h2 in header2)
                
                if has_header2:
                    # If second table has a header, check if it matches the first table's header
                    if header1 and header2:
                        h1 = " | ".join(header1).lower()
                        h2 = " | ".join(header2).lower()
                        if h1 == h2:
                            # Headers match, avoid duplicating the header row
                            rows_to_add = other_matrix[1:]
                        # If headers don't match, still merge but keep the header row
                        # (gap is small and columns align, so it's likely the same table)
                    # If can't compare headers properly, still merge (keep all rows)
                # If has_header2 is False, we merge (keep all rows)

            # Adjust row / column indices for the newly added rows
            row_offset = len(base_matrix)
            for r_idx, row in enumerate(rows_to_add):
                for c_idx, cell in enumerate(row):
                    if isinstance(cell, dict):
                        cell["row"] = row_offset + r_idx
                        cell["col"] = c_idx

            base_matrix.extend(rows_to_add)
            base_rows = len(base_matrix)
            base_bbox |= other_bbox
            used.add(j)

        # Update the base table definition after merging all parts
        new_table = dict(base)
        new_table["bbox"] = (base_bbox.x0, base_bbox.y0, base_bbox.x1, base_bbox.y1)
        new_table["rows"] = base_rows
        # Keep public keys "matrix" and "matrix_ascii" for backward compatibility
        new_table["matrix"] = base_matrix
        # Prefer to build the ASCII matrix from the already-normalized
        # Markdown representation when available (this ensures ASCII uses
        # the same cleaned text as the Markdown table). Fall back to
        # collapsing newlines in the original matrix if Markdown parsing
        # fails.
        sanitized = None
        md_text = base.get("markdown", "") or ""
        if md_text:
            try:
                # collect non-empty lines
                md_lines = [l for l in md_text.splitlines() if l.strip()]
                sep_idx = None
                for i in range(len(md_lines) - 1):
                    # detect the header separator line e.g. |---|---|
                    if re.match(r"^\|?[-:\s|]+\|?$", md_lines[i + 1]):
                        sep_idx = i + 1
                        break
                if sep_idx is not None:
                    header_line = md_lines[sep_idx - 1] if sep_idx > 0 else ""
                    data_lines = md_lines[sep_idx + 1 :]
                    parsed_rows = []
                    if header_line:
                        header_parts = [
                            p.strip() for p in header_line.strip().strip("|").split("|")
                        ]
                        parsed_rows.append(header_parts)
                    for dl in data_lines:
                        # split on '|' and strip
                        parts = [p.strip() for p in dl.strip().strip("|").split("|")]
                        parsed_rows.append(parts)
                    # verify column count matches
                    if parsed_rows:
                        cols = max(len(r) for r in parsed_rows)
                        if cols == base.get("columns", cols):
                            sanitized = []
                            for r_idx, prow in enumerate(parsed_rows):
                                srow = []
                                for c_idx in range(cols):
                                    txt = prow[c_idx] if c_idx < len(prow) else ""
                                    cell_dict = {
                                        "text": txt,
                                        "row": r_idx,
                                        "col": c_idx,
                                        "rowspan": 1,
                                        "colspan": 1,
                                        "bbox": None,
                                        "is_merged": False,
                                        "merged_from": None,
                                    }
                                    srow.append(cell_dict)
                                sanitized.append(srow)
            except Exception:
                sanitized = None

        if sanitized is None:
            # fallback: collapse newlines inside cell text
            try:
                sanitized = []
                for row in base_matrix:
                    srow = []
                    for cell in row:
                        if isinstance(cell, dict):
                            c = dict(cell)
                            txt = c.get("text", "") or ""
                            c["text"] = normalize_table_text(txt, keep_newlines=False)
                            srow.append(c)
                        else:
                            srow.append(normalize_table_text(str(cell) if cell is not None else "", keep_newlines=False))
                    sanitized.append(srow)
            except Exception:
                sanitized = base_matrix

        new_table["matrix_ascii"] = matrix_to_ascii(sanitized)

        merged_tables.append(new_table)
        used.add(idx)

    return merged_tables


def refine_boxes(boxes, enlarge=0):
    """Join any rectangles with a pairwise non-empty overlap.

    Accepts and returns a list of Rect items.
    Note that rectangles that only "touch" each other (common point or edge)
    are not considered as overlapping.
    Use a positive "enlarge" parameter to enlarge rectangle by these many
    points in every direction.

    """
    delta = (-enlarge, -enlarge, enlarge, enlarge)
    new_rects = []
    prects = boxes[:]

    while prects:  # the algorithm will empty this list
        r = +prects[0] + delta  # copy of first rectangle
        repeat = True  # initialize condition
        while repeat:
            repeat = False  # set false as default
            for i in range(len(prects) - 1, 0, -1):  # from back to front
                if r.intersects(prects[i].irect):  # enlarge first rect with this
                    r |= prects[i]
                    del prects[i]  # delete this rect
                    repeat = True  # indicate must try again

        # first rect now includes all overlaps
        new_rects.append(r)
        del prects[0]

    new_rects = sorted(set(new_rects), key=lambda r: (r.x0, r.y0))
    return new_rects


def is_significant(box, paths):
    """Check whether the rectangle "box" contains 'signifiant' drawings.

    This means that some path is contained in the "interior" of box.
    To this end, we build a sub-box of 90% of the original box and check
    whether this still contains drawing paths.
    """
    if box.width > box.height:
        d = box.width * 0.025
    else:
        d = box.height * 0.025
    nbox = box + (d, d, -d, -d)  # nbox covers 90% of box interior
    # paths contained in, but not equal to box:
    my_paths = [p for p in paths if p["rect"] in box and p["rect"] != box]
    widths = set(round(p["rect"].width) for p in my_paths) | {round(box.width)}
    heights = set(round(p["rect"].height) for p in my_paths) | {round(box.height)}
    if len(widths) == 1 or len(heights) == 1:
        return False  # all paths are horizontal or vertical lines / rectangles
    for p in my_paths:
        rect = p["rect"]
        if not (
            bbox_is_empty(rect) or bbox_is_empty(intersect_rects(rect, nbox))
        ):  # intersects interior: significant!
            return True
        # Remaining case: a horizontal or vertical line
        # horizontal line:
        if (
            1
            and rect.y0 == rect.y1
            and nbox.y0 <= rect.y0 <= nbox.y1
            and rect.x0 < nbox.x1
            and rect.x1 > nbox.x0
        ):
            pass  # return True
        # vertical line
        if (
            1
            and rect.x0 == rect.x1
            and nbox.x0 <= rect.x0 <= nbox.x1
            and rect.y0 < nbox.y1
            and rect.y1 > nbox.y0
        ):
            pass  # return True
    return False


def to_json(*args, **kwargs):
    raise NotImplementedError(
        "Function 'to_json' is only available in PyMuPDF-Layout mode"
    )


def to_text(*args, **kwargs):
    raise NotImplementedError(
        "Function 'to_text' is only available in PyMuPDF-Layout mode"
    )


def to_markdown(
    doc,
    *,
    pages=None,
    hdr_info=None,
    write_images=False,
    embed_images=False,
    ignore_images=False,
    ignore_graphics=False,
    detect_bg_color=True,
    image_path="",
    image_format="png",
    image_size_limit=0.05,
    filename=None,
    force_text=True,
    page_chunks=False,
    page_separators=False,
    margins=0,
    dpi=150,
    page_width=612,
    page_height=None,
    table_strategy="lines_strict",
    graphics_limit=None,
    fontsize_limit=3,
    ignore_code=False,
    extract_words=False,
    show_progress=False,
    use_glyphs=False,
    ignore_alpha=False,
    table_format="ascii",
    **kwargs,
) -> str:
    """Process the document and return the text of the selected pages.

    Args:
        doc: pymupdf.Document or string.
        pages: list of page numbers to consider (0-based).
        hdr_info: callable or object having method 'get_hdr_info'.
        write_images: (bool) save images / graphics as files.
        embed_images: (bool) embed images in markdown text (base64 encoded)
        image_path: (str) store images in this folder.
        image_format: (str) use this image format. Choose a supported one.
        force_text: (bool) output text despite of image background.
        page_chunks: (bool) whether to segment output by page.
        page_separators: (bool) whether to include page separators in output.
        margins: omit content overlapping margin areas.
        dpi: (int) desired resolution for generated images.
        page_width: (float) assumption if page layout is variable.
        page_height: (float) assumption if page layout is variable.
        table_strategy: choose table detection strategy
        graphics_limit: (int) if vector graphics count exceeds this, ignore all.
        ignore_code: (bool) suppress code-like formatting (mono-space fonts)
        extract_words: (bool, False) include "words"-like output in page chunks
        show_progress: (bool, False) print progress as each page is processed.
        use_glyphs: (bool, False) replace the Invalid Unicode by glyph numbers.
        ignore_alpha: (bool, True) ignore text with alpha = 0 (transparent).

    """
    if kwargs.keys():
        print(f"Warning - arguments ignored in legacy mode: {set(kwargs.keys())}.")

    if write_images is False and embed_images is False and force_text is False:
        raise ValueError("Images and text on images cannot both be suppressed.")
    if embed_images is True:
        write_images = False
        image_path = ""
    if not 0 <= image_size_limit < 1:
        raise ValueError("'image_size_limit' must be non-negative and less than 1.")
    DPI = dpi
    IGNORE_CODE = ignore_code
    IMG_EXTENSION = image_format
    EXTRACT_WORDS = extract_words
    if EXTRACT_WORDS is True:
        page_chunks = True
        ignore_code = True
    IMG_PATH = image_path
    if IMG_PATH and write_images is True and not os.path.exists(IMG_PATH):
        os.makedirs(IMG_PATH, exist_ok=True)

    if not isinstance(doc, pymupdf.Document):
        doc = pymupdf.open(doc)

    FILENAME = doc.name if filename is None else filename
    GRAPHICS_LIMIT = graphics_limit
    FONTSIZE_LIMIT = fontsize_limit
    IGNORE_IMAGES = ignore_images
    IGNORE_GRAPHICS = ignore_graphics
    DETECT_BG_COLOR = detect_bg_color
    if table_format not in ("ascii", "markdown"):
        print(f"Warning: invalid table_format '{table_format}', using 'ascii'.")
        table_format = "ascii"
    if doc.is_form_pdf or (doc.is_pdf and doc.has_annots()):
        doc.bake()

    # for reflowable documents allow making 1 page for the whole document
    if doc.is_reflowable:
        if hasattr(page_height, "__float__"):
            # accept user page dimensions
            doc.layout(width=page_width, height=page_height)
        else:
            # no page height limit given: make 1 page for whole document
            doc.layout(width=page_width, height=792)
            page_count = doc.page_count
            height = 792 * page_count  # height that covers full document
            doc.layout(width=page_width, height=height)

    if pages is None:  # use all pages if no selection given
        pages = list(range(doc.page_count))

    if hasattr(margins, "__float__"):
        margins = [margins] * 4
    if len(margins) == 2:
        margins = (0, margins[0], 0, margins[1])
    if len(margins) != 4:
        raise ValueError("margins must be one, two or four floats")
    elif not all(hasattr(m, "__float__") for m in margins):
        raise ValueError("margin values must be floats")

    # If "hdr_info" is not an object with a method "get_header_id", scan the
    # document and use font sizes as header level indicators.
    if callable(hdr_info):
        get_header_id = hdr_info
    elif hasattr(hdr_info, "get_header_id") and callable(hdr_info.get_header_id):
        get_header_id = hdr_info.get_header_id
    elif hdr_info is False:
        get_header_id = lambda s, page=None: ""
    else:
        hdr_info = IdentifyHeaders(doc)
        get_header_id = hdr_info.get_header_id

    def max_header_id(spans, page):
        hdr_ids = sorted(
            [l for l in set([len(get_header_id(s, page=page)) for s in spans]) if l > 0]
        )
        if not hdr_ids:
            return ""
        return "#" * (hdr_ids[0] - 1) + " "

    def resolve_links(links, span):
        """Accept a span and return a markdown link string.

        Args:
            links: a list as returned by page.get_links()
            span: a span dictionary as returned by page.get_text("dict")

        Returns:
            None or a string representing the link in MD format.
        """
        bbox = pymupdf.Rect(span["bbox"])  # span bbox
        # a link should overlap at least 70% of the span
        for link in links:
            hot = link["from"]  # the hot area of the link
            middle = (hot.tl + hot.br) / 2  # middle point of hot area
            if not middle in bbox:
                continue  # does not touch the bbox
            text = f'[{span["text"].strip()}]({link["uri"]})'
            return text

    def _collapse_table_spaces(value: str) -> str:
        """Collapse excessive spaces while keeping newlines intact."""
        if not value:
            return value
        # Collapse multiple spaces/tabs while keeping newlines intact.
        value = re.sub(r"[ \t]+", " ", value)
        return value

    def normalize_table_text(text, *, keep_newlines=False):
        """Normalize whitespace for table cell text to avoid layout breaks."""
        if not text:
            return text
        value = _normalize_table_br_tags(text)
        if not keep_newlines:
            value = value.replace("\n", " ")
            value = _collapse_table_spaces(value).strip()
            return value
        # Keep line breaks but normalize spaces within each line
        value = _collapse_table_spaces(value)
        value = "\n".join(line.strip() for line in value.split("\n"))
        return value

    def save_image(parms, rect, i):
        """Optionally render the rect part of a page.

        We will ignore images that are empty or that have an edge smaller
        than x% of the corresponding page edge."""
        page = parms.page
        if (
            rect.width < page.rect.width * image_size_limit
            or rect.height < page.rect.height * image_size_limit
        ):
            return ""
        if write_images is True or embed_images is True:
            pix = page.get_pixmap(clip=rect, dpi=DPI)
        else:
            return ""
        if pix.height <= 0 or pix.width <= 0:
            return ""

        if write_images is True:
            filename = os.path.basename(parms.filename).replace(" ", "-")
            image_filename = os.path.join(
                IMG_PATH, f"{filename}-{page.number}-{i}.{IMG_EXTENSION}"
            )
            pix.save(image_filename)
            return image_filename.replace("\\", "/")
        elif embed_images is True:
            # make a base64 encoded string of the image
            data = b2a_base64(pix.tobytes(IMG_EXTENSION)).decode()
            data = f"data:image/{IMG_EXTENSION};base64," + data
            return data
        return ""

    def write_text(
        parms,
        clip: pymupdf.Rect,
        tables=True,
        images=True,
        force_text=force_text,
        base_offset: int = 0,
    ):
        """Output the text found inside the given clip.

        This is an alternative for plain text in that it outputs
        text enriched with markdown styling.
        The logic is capable of recognizing headers, body text, code blocks,
        inline code, bold, italic and bold-italic styling.
        There is also some effort for list supported (ordered / unordered) in
        that typical characters are replaced by respective markdown characters.

        'tables'/'images' indicate whether this execution should output these
        objects.
        """

        if clip is None:
            clip = parms.clip
        out_string = []
        out_string_ascii = []

        # This is a list of tuples (linerect, spanlist)
        nlines = get_raw_lines(
            parms.textpage,
            clip=clip,
            tolerance=3,
            ignore_invisible=not parms.accept_invisible,
        )
        nlines = [
            l for l in nlines if outside_all_bboxes(l[0], parms.tab_rects.values())
        ]

        parms.line_rects.extend([l[0] for l in nlines])  # store line rectangles

        prev_lrect = None  # previous line rectangle
        prev_bno = -1  # previous block number of line
        code = False  # mode indicator: outputting code
        prev_hdr_string = None

        for lrect, spans in nlines:
            # there may be tables or images inside the text block: skip them
            if not outside_all_bboxes(lrect, parms.img_rects):
                continue

            # ------------------------------------------------------------
            # Pick up tables ABOVE this text block
            # ------------------------------------------------------------
            if tables:
                tab_candidates = [
                    (i, tab_rect)
                    for i, tab_rect in parms.tab_rects.items()
                    if tab_rect.y1 <= lrect.y0
                    and i not in parms.written_tables
                    and (
                        0
                        or lrect.x0 <= tab_rect.x0 < lrect.x1
                        or lrect.x0 < tab_rect.x1 <= lrect.x1
                        or tab_rect.x0 <= lrect.x0 < lrect.x1 <= tab_rect.x1
                    )
                ]
                for i, _ in tab_candidates:
                    table_md = _normalize_table_br_tags(
                        parms.tabs[i].to_markdown(clean=False)
                    )
                    table_ascii = parms.tables_by_tab[i].get("matrix_ascii", "")
                    md_segment = "\n" + table_md + "\n"
                    ascii_segment = "\n" + (table_ascii or "") + "\n"
                    span_start = base_offset + len(out_string) + 1
                    span_end = span_start + len(table_md)
                    parms.tables_by_tab[i]["markdown_span"] = (span_start, span_end)
                    out_string += md_segment
                    out_string_ascii += ascii_segment
                    if EXTRACT_WORDS:
                        # for "words" extraction, add table cells as line rects
                        cells = sorted(
                            set(
                                [
                                    pymupdf.Rect(c)
                                    for c in parms.tabs[i].header.cells
                                    + parms.tabs[i].cells
                                    if c is not None
                                ]
                            ),
                            key=lambda c: (c.y1, c.x0),
                        )
                        parms.line_rects.extend(cells)
                    parms.written_tables.append(i)
                    prev_hdr_string = None

            # ------------------------------------------------------------
            # Pick up images / graphics ABOVE this text block
            # ------------------------------------------------------------
            if images:
                for i in range(len(parms.img_rects)):
                    if i in parms.written_images:
                        continue
                    r = parms.img_rects[i]
                    if max(r.y0, lrect.y0) < min(r.y1, lrect.y1) and (
                        0
                        or lrect.x0 <= r.x0 < lrect.x1
                        or lrect.x0 < r.x1 <= lrect.x1
                        or r.x0 <= lrect.x0 < lrect.x1 <= r.x1
                    ):
                        pathname = save_image(parms, r, i)
                        if pathname:
                            img_md = GRAPHICS_TEXT % pathname
                            out_string.append(img_md)
                            out_string_ascii.append(img_md)

                        # recursive invocation
                        if force_text is True:
                            img_txt = write_text(
                                parms,
                                r,
                                tables=False,
                                images=False,
                                force_text=True,
                            )

                            if not is_white(img_txt):
                                out_string.append(img_txt)
                                out_string_ascii.append(img_txt)
                        parms.written_images.append(i)
                        prev_hdr_string = None

            parms.line_rects.append(lrect)
            # if line rect is far away from the previous one, add a line break
            if (
                len(parms.line_rects) > 1
                and lrect.y1 - parms.line_rects[-2].y1 > lrect.height * 1.5
            ):
                out_string.append("\n")
                out_string_ascii.append("\n")
            # make text string for the full line
            text = " ".join([s["text"] for s in spans]).strip()

            # full line strikeout?
            all_strikeout = all([s["char_flags"] & 1 for s in spans])
            # full line italic?
            all_italic = all([s["flags"] & 2 for s in spans])
            # full line bold?
            all_bold = all([(s["flags"] & 16) or (s["char_flags"] & 8) for s in spans])
            # full line mono-spaced?
            all_mono = all([s["flags"] & 8 for s in spans])

            # if line is a header, this will return multiple "#" characters,
            # otherwise an empty string
            hdr_string = max_header_id(spans, page=parms.page)  # a header?

            if hdr_string:  # if a header line skip the rest
                if all_mono:
                    text = "`" + text + "`"
                if all_italic:
                    text = "_" + text + "_"
                if all_bold:
                    text = "**" + text + "**"
                if all_strikeout:
                    text = "~~" + text + "~~"
                if hdr_string != prev_hdr_string:
                    out_string.append(hdr_string + text + "\n")
                    out_string_ascii.append(hdr_string + text + "\n")
                else:
                    # Remove trailing newline from last item if present
                    if out_string and out_string[-1].endswith("\n"):
                        out_string[-1] = out_string[-1].rstrip("\n")
                    if out_string_ascii and out_string_ascii[-1].endswith("\n"):
                        out_string_ascii[-1] = out_string_ascii[-1].rstrip("\n")
                    out_string.append(" " + text + "\n")
                    out_string_ascii.append(" " + text + "\n")
                prev_hdr_string = hdr_string
                continue

            prev_hdr_string = hdr_string

            # start or extend a code block
            if all_mono and not IGNORE_CODE:
                if not code:
                    out_string.append("```\n")
                    out_string_ascii.append("```\n")
                    code = True
                delta = int((lrect.x0 - clip.x0) / (spans[0]["size"] * 0.5))
                indent = " " * delta

                out_string.append(indent + text + "\n")
                out_string_ascii.append(indent + text + "\n")
                continue

            if code and not all_mono:
                out_string.append("```\n")
                out_string_ascii.append("```\n")
                code = False

            span0 = spans[0]
            bno = span0["block"]
            if bno != prev_bno:
                out_string.append("\n")
                out_string_ascii.append("\n")
                prev_bno = bno

            if (
                prev_lrect
                and lrect.y1 - prev_lrect.y1 > lrect.height * 1.5
                or span0["text"].startswith("[")
                or startswith_bullet(span0["text"])
                or span0["flags"] & 1  # superscript?
            ):
                out_string.append("\n")
                out_string_ascii.append("\n")
            prev_lrect = lrect

            if code:
                out_string.append("```\n")
                out_string_ascii.append("```\n")
                code = False

            for i, s in enumerate(spans):  # iterate spans of the line
                # decode font properties
                mono = s["flags"] & 8
                bold = s["flags"] & 16 or s["char_flags"] & 8
                italic = s["flags"] & 2
                strikeout = s["char_flags"] & 1

                prefix = ""
                suffix = ""
                if mono:
                    prefix = "`" + prefix
                    suffix += "`"
                if bold:
                    prefix = "**" + prefix
                    suffix += "**"
                if italic:
                    prefix = "_" + prefix
                    suffix += "_"
                if strikeout:
                    prefix = "~~" + prefix
                    suffix += "~~"

                # convert intersecting link to markdown syntax
                ltext = resolve_links(parms.links, s)
                if ltext:
                    text = f"{hdr_string}{prefix}{ltext}{suffix} "
                else:
                    text = f"{hdr_string}{prefix}{s['text'].strip()}{suffix} "
                if startswith_bullet(text):
                    text = "- " + text[1:]
                    text = re.sub(r' +', ' ', text)
                    dist = span0["bbox"][0] - clip.x0
                    cwidth = (span0["bbox"][2] - span0["bbox"][0]) / len(span0["text"])
                    if cwidth == 0.0:
                        cwidth = span0["size"] * 0.5
                    text = " " * int(round(dist / cwidth)) + text

                out_string.append(text)
                out_string_ascii.append(text)
            if not code:
                out_string.append("\n")
                out_string_ascii.append("\n")
        out_string.append("\n")
        out_string_ascii.append("\n")
        if code:
            out_string.append("```\n")  # switch of code mode
            out_string_ascii.append("```\n")
            code = False
        out_string.append("\n\n")
        out_string_ascii.append("\n\n")
        # Join lists into strings for efficient concatenation
        out_str = "".join(out_string)
        out_str_ascii = "".join(out_string_ascii)
        out_str = re.sub(r" +", " ", out_str)
        return (
            out_str.replace(" \n", "\n").replace("\n\n\n", "\n\n"),
            out_str_ascii,
        )

    def bbox_in_any_bbox(rect, rect_list):
        """Check if rect is contained in a rect of the list."""
        return any(bbox_in_bbox(rect, r) for r in rect_list)

    def outside_all_bboxes(rect, rect_list):
        """Check if rect is outside all rects in the list."""
        return all(outside_bbox(rect, r) for r in rect_list)

    def almost_in_any_bbox(rect, rect_list):
        """Check if middle of rect is contained in a rect of the list."""
        # enlarge rect somewhat
        enlarged = [rect[0] - 1, rect[1] - 1, rect[2] + 1, rect[3] + 1]
        return any(almost_in_bbox(enlarged, r, portion=0.5) for r in rect_list)

    def output_tables(parms, text_rect):
        """Output tables above given text rectangle."""
        this_md = []  # markdown list for efficient concatenation
        this_ascii = []
        base_offset = len(parms.md_string)
        if text_rect is not None:  # select tables above the text block
            for i, trect in sorted(
                [j for j in parms.tab_rects.items() if j[1].y1 <= text_rect.y0],
                key=lambda j: (j[1].y1, j[1].x0),
            ):
                if i in parms.written_tables:
                    continue
                table_md = _normalize_table_br_tags(
                    parms.tabs[i].to_markdown(clean=False)
                )
                table_ascii = parms.tables_by_tab[i].get("matrix_ascii", "")
                span_start = base_offset + sum(len(m) for m in this_md)
                span_end = span_start + len(table_md)
                parms.tables_by_tab[i]["markdown_span"] = (span_start, span_end)
                this_md.append(table_md)
                this_md.append("\n")
                this_ascii.append(table_ascii or "")
                this_ascii.append("\n")
                if EXTRACT_WORDS:
                    # for "words" extraction, add table cells as line rects
                    cells = sorted(
                        set(
                            [
                                pymupdf.Rect(c)
                                for c in parms.tabs[i].header.cells
                                + parms.tabs[i].cells
                                if c is not None
                            ]
                        ),
                        key=lambda c: (c.y1, c.x0),
                    )
                    parms.line_rects.extend(cells)
                parms.written_tables.append(i)  # do not touch this table twice

        else:  # output all remaining tables
            for i, trect in parms.tab_rects.items():
                if i in parms.written_tables:
                    continue
                table_md = _normalize_table_br_tags(
                    parms.tabs[i].to_markdown(clean=False)
                )
                table_ascii = parms.tables_by_tab[i].get("matrix_ascii", "")
                span_start = base_offset + sum(len(m) for m in this_md)
                span_end = span_start + len(table_md)
                parms.tables_by_tab[i]["markdown_span"] = (span_start, span_end)
                this_md.append(table_md)
                this_md.append("\n")
                this_ascii.append(table_ascii or "")
                this_ascii.append("\n")
                if EXTRACT_WORDS:
                    # for "words" extraction, add table cells as line rects
                    cells = sorted(
                        set(
                            [
                                pymupdf.Rect(c)
                                for c in parms.tabs[i].header.cells
                                + parms.tabs[i].cells
                                if c is not None
                            ]
                        ),
                        key=lambda c: (c.y1, c.x0),
                    )
                    parms.line_rects.extend(cells)
                parms.written_tables.append(i)  # do not touch this table twice
        return "".join(this_md), "".join(this_ascii)

    def output_images(parms, text_rect, force_text):
        """Output images and graphics above text rectangle."""
        if not parms.img_rects:
            return "", ""
        this_md = []  # markdown list for efficient concatenation
        this_ascii = []
        if text_rect is not None:  # select images above the text block
            for i, img_rect in enumerate(parms.img_rects):
                if img_rect.y0 > text_rect.y0:
                    continue
                if img_rect.x0 >= text_rect.x1 or img_rect.x1 <= text_rect.x0:
                    continue
                if i in parms.written_images:
                    continue
                pathname = save_image(parms, img_rect, i)
                parms.written_images.append(i)  # do not touch this image twice
                if pathname:
                    img_md = GRAPHICS_TEXT % pathname
                    this_md.append(img_md)
                    this_ascii.append(img_md)
                if force_text:
                    img_txt = write_text(
                        parms,
                        img_rect,
                        tables=False,  # we have no tables here
                        images=False,  # we have no other images here
                        force_text=True,
                        base_offset=0,
                    )
                    if not is_white(img_txt[0]):  # was there text at all?
                        this_md.append(img_txt[0])
                        this_ascii.append(img_txt[1])
        else:  # output all remaining images
            for i, img_rect in enumerate(parms.img_rects):
                if i in parms.written_images:
                    continue
                pathname = save_image(parms, img_rect, i)
                parms.written_images.append(i)  # do not touch this image twice
                if pathname:
                    img_md = GRAPHICS_TEXT % pathname
                    this_md.append(img_md)
                    this_ascii.append(img_md)
                if force_text:
                    img_txt = write_text(
                        parms,
                        img_rect,
                        tables=False,  # we have no tables here
                        images=False,  # we have no other images here
                        force_text=True,
                        base_offset=0,
                    )
                    if not is_white(img_txt[0]):
                        this_md.append(img_txt[0])
                        this_ascii.append(img_txt[1])

        return "".join(this_md), "".join(this_ascii)

    def page_is_ocr(page):
        """Check if page exclusivley contains OCR text.

        For this to be true, all text must be written as "ignore-text".
        """
        try:
            text_types = set([b[0] for b in page.get_bboxlog() if "text" in b[0]])
            if text_types == {"ignore-text"}:
                return True
        except:
            pass
        return False

    def get_bg_color(page):
        """Determine the background color of the page.

        The function returns a PDF RGB color triple or None.
        We check the color of 10 x 10 pixel areas in the four corners of the
        page. If they are unicolor and of the same color, we assume this to
        be the background color.
        """
        pix = page.get_pixmap(
            clip=(page.rect.x0, page.rect.y0, page.rect.x0 + 10, page.rect.y0 + 10)
        )
        if not pix.samples or not pix.is_unicolor:
            return None
        pixel_ul = pix.pixel(0, 0)  # upper left color
        pix = page.get_pixmap(
            clip=(page.rect.x1 - 10, page.rect.y0, page.rect.x1, page.rect.y0 + 10)
        )
        if not pix.samples or not pix.is_unicolor:
            return None
        pixel_ur = pix.pixel(0, 0)  # upper right color
        if not pixel_ul == pixel_ur:
            return None
        pix = page.get_pixmap(
            clip=(page.rect.x0, page.rect.y1 - 10, page.rect.x0 + 10, page.rect.y1)
        )
        if not pix.samples or not pix.is_unicolor:
            return None
        pixel_ll = pix.pixel(0, 0)  # lower left color
        if not pixel_ul == pixel_ll:
            return None
        pix = page.get_pixmap(
            clip=(page.rect.x1 - 10, page.rect.y1 - 10, page.rect.x1, page.rect.y1)
        )
        if not pix.samples or not pix.is_unicolor:
            return None
        pixel_lr = pix.pixel(0, 0)  # lower right color
        if not pixel_ul == pixel_lr:
            return None
        return (pixel_ul[0] / 255, pixel_ul[1] / 255, pixel_ul[2] / 255)

    def get_metadata(doc, pno):
        meta = doc.metadata.copy()
        meta["file_path"] = FILENAME
        meta["page_count"] = doc.page_count
        meta["page"] = pno + 1
        return meta

    def sort_words(words: list) -> list:
        """Reorder words in lines.

        The argument list must be presorted by bottom, then left coordinates.

        Words with similar top / bottom coordinates are assumed to belong to
        the same line and will be sorted left to right within that line.
        """
        if not words:
            return []
        nwords = []
        line = [words[0]]
        lrect = pymupdf.Rect(words[0][:4])
        for w in words[1:]:
            if abs(w[1] - lrect.y0) <= 3 or abs(w[3] - lrect.y1) <= 3:
                line.append(w)
                lrect |= w[:4]
            else:
                line.sort(key=lambda w: w[0])
                nwords.extend(line)
                line = [w]
                lrect = pymupdf.Rect(w[:4])
        line.sort(key=lambda w: w[0])
        nwords.extend(line)
        return nwords

    def get_page_output(
        doc, pno, margins, textflags, FILENAME, IGNORE_IMAGES, IGNORE_GRAPHICS
    ):
        """Process one page.

        Args:
            doc: pymupdf.Document
            pno: 0-based page number
            textflags: text extraction flag bits

        Returns:
            Markdown string of page content and image, table and vector
            graphics information.
        """
        page = doc[pno]
        page.remove_rotation()  # make sure we work on rotation=0
        parms = Parameters()  # all page information
        parms.page = page
        parms.filename = FILENAME
        parms.md_string = ""
        parms.md_string_ascii = ""
        parms.images = []
        parms.tables = []
        parms.graphics = []
        parms.words = []
        parms.line_rects = []
        parms.accept_invisible = (
            page_is_ocr(page) or ignore_alpha
        )  # accept invisible text

        # determine background color
        parms.bg_color = None if not DETECT_BG_COLOR else get_bg_color(page)

        left, top, right, bottom = margins
        parms.clip = page.rect + (left, top, -right, -bottom)

        # extract external links on page
        parms.links = [l for l in page.get_links() if l["kind"] == pymupdf.LINK_URI]

        # extract annotation rectangles on page
        parms.annot_rects = [a.rect for a in page.annots()]

        # make a TextPage for all later extractions
        parms.textpage = page.get_textpage(flags=textflags, clip=parms.clip)

        # extract images on page
        if not IGNORE_IMAGES:
            img_info = page.get_image_info()
        else:
            img_info = []
        for i in range(len(img_info)):
            img_info[i]["bbox"] = pymupdf.Rect(img_info[i]["bbox"])

        # filter out images that are too small or outside the clip
        img_info = [
            i
            for i in img_info
            if i["bbox"].width >= image_size_limit * parms.clip.width
            and i["bbox"].height >= image_size_limit * parms.clip.height
            and i["bbox"].intersects(parms.clip)
            and i["bbox"].width > 3
            and i["bbox"].height > 3
        ]

        # sort descending by image area size
        img_info.sort(key=lambda i: abs(i["bbox"]), reverse=True)

        # subset of images truly inside the clip
        if img_info:
            img_max_size = abs(parms.clip) * 0.9
            sane = [i for i in img_info if abs(i["bbox"] & parms.clip) < img_max_size]
            if len(sane) < len(img_info):  # found some
                img_info = sane  # use those images instead
                # output full page image
                name = save_image(parms, parms.clip, "full")
                if name:
                    parms.md_string += GRAPHICS_TEXT % name

        img_info = img_info[:30]  # only accept the largest up to 30 images
        # run from back to front (= small to large)
        for i in range(len(img_info) - 1, 0, -1):
            r = img_info[i]["bbox"]
            if r.is_empty:
                del img_info[i]
                continue
            for j in range(i):  # image areas larger than r
                if r in img_info[j]["bbox"]:
                    del img_info[i]  # contained in some larger image
                    break
        parms.images = img_info

        parms.img_rects = [i["bbox"] for i in parms.images]

        # catch too-many-graphics situation
        graphics_count = len([b for b in page.get_bboxlog() if "path" in b[0]])
        if GRAPHICS_LIMIT and graphics_count > GRAPHICS_LIMIT:
            IGNORE_GRAPHICS = True
            too_many_graphics = True
        else:
            too_many_graphics = False

        # Locate all tables on page
        parms.written_tables = []  # stores already written tables
        omitted_table_rects = []
        parms.tabs = []
        if IGNORE_GRAPHICS or not table_strategy:
            # do not try to extract tables
            pass
        else:
            tabs = page.find_tables(clip=parms.clip, strategy=table_strategy)
            for t in tabs.tables:
                # remove tables with too few rows or columns
                if t.row_count < 2 or t.col_count < 2:
                    omitted_table_rects.append(pymupdf.Rect(t.bbox))
                    continue
                parms.tabs.append(t)
            parms.tabs.sort(key=lambda t: (t.bbox[0], t.bbox[1]))

        # Make a list of table boundary boxes.
        # Must include the header bbox (which may exist outside tab.bbox)
        tab_rects = {}
        for i, t in enumerate(parms.tabs):
            tab_rects[i] = pymupdf.Rect(t.bbox) | pymupdf.Rect(t.header.bbox)
            # Extract matrix (list of lists) from table using extract_cells
            # to preserve spaces and formatting
            try:
                # Try to access table structure
                if hasattr(t, "table") and t.table:
                    table_dict = t.table
                    row_count = table_dict.get("row_count", t.row_count)
                    col_count = table_dict.get("col_count", t.col_count)
                    cell_boxes = table_dict.get("cells", [])
                else:
                    # Fallback: use rows attribute
                    row_count = t.row_count
                    col_count = t.col_count
                    # Build cell_boxes from rows
                    cell_boxes = []
                    # Include header if it exists (PyMuPDF stores header separately)
                    if hasattr(t, "header") and t.header is not None:
                        row_cells = []
                        for c in t.header.cells:
                            if c is None:
                                row_cells.append(None)
                            elif hasattr(c, "bbox"):
                                row_cells.append(c.bbox)
                            elif isinstance(c, (tuple, list)) and len(c) >= 4:
                                row_cells.append(c)
                            else:
                                row_cells.append(None)
                        cell_boxes.append(row_cells)
                    # Add regular rows
                    for row in t.rows:
                        row_cells = []
                        for c in row.cells:
                            if c is None:
                                row_cells.append(None)
                            elif hasattr(c, "bbox"):
                                row_cells.append(c.bbox)
                            elif isinstance(c, (tuple, list)) and len(c) >= 4:
                                row_cells.append(c)
                            else:
                                row_cells.append(None)
                        cell_boxes.append(row_cells)
                
                # Calculate cell dimensions to detect merged cells
                # First, collect all non-empty cell bboxes
                all_cell_bboxes = []
                cell_rects_map = {}  # Map (row, col) to cell rect
                for row_idx, row in enumerate(cell_boxes):
                    for col_idx, cell in enumerate(row):
                        if cell is not None:
                            cell_rect = pymupdf.Rect(cell) if not isinstance(cell, pymupdf.Rect) else cell
                            if not cell_rect.is_empty:
                                all_cell_bboxes.append(cell_rect)
                                cell_rects_map[(row_idx, col_idx)] = cell_rect
                
                # Calculate average cell width and height from non-empty cells
                # Use a more robust method: find the most common cell size
                avg_cell_width = 0
                avg_cell_height = 0
                if all_cell_bboxes:
                    # Group similar widths/heights to find the most common size
                    # This helps avoid outliers from merged cells
                    widths = [c.width for c in all_cell_bboxes]
                    heights = [c.height for c in all_cell_bboxes]
                    
                    # Sort and use median
                    sorted_widths = sorted(widths)
                    sorted_heights = sorted(heights)
                    mid = len(sorted_widths) // 2
                    avg_cell_width = sorted_widths[mid] if sorted_widths else 0
                    avg_cell_height = sorted_heights[mid] if sorted_heights else 0
                    
                    # If we have enough cells, use a smaller percentile to avoid merged cell outliers
                    if len(sorted_widths) > 3:
                        # Use 25th percentile for more accurate base cell size
                        quartile_idx = len(sorted_widths) // 4
                        avg_cell_width = sorted_widths[quartile_idx]
                        avg_cell_height = sorted_heights[quartile_idx]
                    
                    # Calculate row and column boundaries to help detect merged cells
                    # Group cells by approximate row/column positions
                    row_y_positions = sorted(set([round(c.y0, 1) for c in all_cell_bboxes]))
                    col_x_positions = sorted(set([round(c.x0, 1) for c in all_cell_bboxes]))
                    
                    # Calculate typical row height and column width from spacing
                    if len(row_y_positions) > 1:
                        row_heights = [row_y_positions[i+1] - row_y_positions[i] 
                                      for i in range(len(row_y_positions)-1)]
                        if row_heights:
                            avg_row_height_from_spacing = sorted(row_heights)[len(row_heights)//2]
                            # Use the smaller of the two estimates to avoid merged cell bias
                            if avg_row_height_from_spacing > 0:
                                avg_cell_height = min(avg_cell_height, avg_row_height_from_spacing) if avg_cell_height > 0 else avg_row_height_from_spacing
                    
                    if len(col_x_positions) > 1:
                        col_widths = [col_x_positions[i+1] - col_x_positions[i] 
                                      for i in range(len(col_x_positions)-1)]
                        if col_widths:
                            avg_col_width_from_spacing = sorted(col_widths)[len(col_widths)//2]
                            # Use the smaller of the two estimates to avoid merged cell bias
                            if avg_col_width_from_spacing > 0:
                                avg_cell_width = min(avg_cell_width, avg_col_width_from_spacing) if avg_cell_width > 0 else avg_col_width_from_spacing
                
                # Initialize matrix with None
                matrix = [[None for _ in range(col_count)] for _ in range(row_count)]
                # Extract text from each cell using extract_cells
                for row_idx, row in enumerate(cell_boxes):
                    if row_idx >= row_count:
                        break
                    for col_idx, cell in enumerate(row):
                        if col_idx >= col_count:
                            break
                        if cell is not None:
                            try:
                                cell_rect = pymupdf.Rect(cell) if not isinstance(cell, pymupdf.Rect) else cell
                                cell_text = extract_cells(
                                    parms.textpage, cell, markdown=False
                                )
                                # Normalize whitespace/HTML breaks before wrapping
                                cell_text = normalize_table_text(cell_text)
                                
                                # Wrap text to fit cell width based on bbox
                                if cell_text and not cell_rect.is_empty:
                                    cell_text = wrap_text_by_bbox(
                                        cell_text, cell_rect, parms.textpage
                                    )
                                
                                # Calculate rowspan and colspan using multiple detection methods
                                rowspan = 1
                                colspan = 1
                                
                                # Method 1: Analyze cell size compared to average
                                if avg_cell_height > 0 and avg_cell_width > 0:
                                    height_ratio = cell_rect.height / avg_cell_height
                                    width_ratio = cell_rect.width / avg_cell_width
                                    
                                    # Use a threshold of 1.2x to detect merged cells (more sensitive)
                                    if height_ratio > 1.2:
                                        rowspan = max(1, round(height_ratio))
                                    if width_ratio > 1.2:
                                        colspan = max(1, round(width_ratio))
                                
                                # Method 2: Check for overlapping cells that indicate merging
                                # Look for cells in adjacent positions that significantly overlap
                                # This detects merged cells even when size-based detection fails
                                
                                # Check vertical merging (rowspan)
                                # Look for cells in rows below that overlap significantly with this cell
                                for check_row in range(row_idx + 1, row_count):
                                    check_pos = (check_row, col_idx)
                                    if check_pos in cell_rects_map:
                                        other_rect = cell_rects_map[check_pos]
                                        # Calculate overlap
                                        intersection = cell_rect & other_rect
                                        if not intersection.is_empty:
                                            # Check if cells overlap significantly (indicating they're part of the same merged cell)
                                            overlap_area = abs(intersection)
                                            cell_area = abs(cell_rect)
                                            other_area = abs(other_rect)
                                            
                                            # If overlap is significant (>70% of smaller cell), they're likely merged
                                            if cell_area > 0 and other_area > 0:
                                                overlap_ratio_smaller = overlap_area / min(cell_area, other_area)
                                                # Also check if the other cell is mostly contained within this cell's vertical span
                                                vertical_overlap_ratio = intersection.height / min(cell_rect.height, other_rect.height) if min(cell_rect.height, other_rect.height) > 0 else 0
                                                
                                                if overlap_ratio_smaller > 0.7 or vertical_overlap_ratio > 0.8:
                                                    # Cells are merged vertically
                                                    # Calculate how many rows this cell spans
                                                    estimated_rowspan = check_row - row_idx + 1
                                                    rowspan = max(rowspan, estimated_rowspan)
                                                    break
                                    else:
                                        # No physical cell at this position - check if it's None in cell_boxes
                                        if (check_row < len(cell_boxes) and 
                                            col_idx < len(cell_boxes[check_row]) and
                                            cell_boxes[check_row][col_idx] is None):
                                            # Empty cell position - check if this cell's height suggests it spans here
                                            if avg_cell_height > 0:
                                                # Estimate rows based on height
                                                estimated_rows = round(cell_rect.height / avg_cell_height)
                                                if estimated_rows > rowspan:
                                                    # Verify no physical cells in between
                                                    can_span = True
                                                    for verify_row in range(row_idx + 1, min(row_idx + estimated_rows, row_count)):
                                                        if (verify_row, col_idx) in cell_rects_map:
                                                            can_span = False
                                                            break
                                                    if can_span:
                                                        rowspan = max(rowspan, min(estimated_rows, row_count - row_idx))
                                
                                # Check horizontal merging (colspan)
                                # Look for cells in columns to the right that overlap significantly
                                for check_col in range(col_idx + 1, col_count):
                                    check_pos = (row_idx, check_col)
                                    if check_pos in cell_rects_map:
                                        other_rect = cell_rects_map[check_pos]
                                        # Calculate overlap
                                        intersection = cell_rect & other_rect
                                        if not intersection.is_empty:
                                            # Check if cells overlap significantly
                                            overlap_area = abs(intersection)
                                            cell_area = abs(cell_rect)
                                            other_area = abs(other_rect)
                                            
                                            if cell_area > 0 and other_area > 0:
                                                overlap_ratio_smaller = overlap_area / min(cell_area, other_area)
                                                # Also check horizontal overlap
                                                horizontal_overlap_ratio = intersection.width / min(cell_rect.width, other_rect.width) if min(cell_rect.width, other_rect.width) > 0 else 0
                                                
                                                if overlap_ratio_smaller > 0.7 or horizontal_overlap_ratio > 0.8:
                                                    # Cells are merged horizontally
                                                    estimated_colspan = check_col - col_idx + 1
                                                    colspan = max(colspan, estimated_colspan)
                                                    break
                                    else:
                                        # No physical cell at this position
                                        if (row_idx < len(cell_boxes) and 
                                            check_col < len(cell_boxes[row_idx]) and
                                            cell_boxes[row_idx][check_col] is None):
                                            # Empty cell position - check if this cell's width suggests it spans here
                                            if avg_cell_width > 0:
                                                estimated_cols = round(cell_rect.width / avg_cell_width)
                                                if estimated_cols > colspan:
                                                    # Verify no physical cells in between
                                                    can_span = True
                                                    for verify_col in range(col_idx + 1, min(col_idx + estimated_cols, col_count)):
                                                        if (row_idx, verify_col) in cell_rects_map:
                                                            can_span = False
                                                            break
                                                    if can_span:
                                                        colspan = max(colspan, min(estimated_cols, col_count - col_idx))
                                
                                # Method 3: Verify merged cells by checking if positions that should be covered are empty
                                # Refine rowspan by checking for separate physical cells
                                if rowspan > 1:
                                    for check_row in range(row_idx + 1, min(row_idx + rowspan, row_count)):
                                        if (check_row, col_idx) in cell_rects_map:
                                            other_rect = cell_rects_map[(check_row, col_idx)]
                                            # If there's a separate physical cell that doesn't overlap significantly,
                                            # reduce rowspan
                                            intersection = cell_rect & other_rect
                                            if intersection.is_empty or abs(intersection) / min(abs(cell_rect), abs(other_rect)) < 0.5:
                                                # Separate cell found, reduce rowspan
                                                rowspan = check_row - row_idx
                                                break
                                
                                # Refine colspan by checking for separate physical cells
                                if colspan > 1:
                                    for check_col in range(col_idx + 1, min(col_idx + colspan, col_count)):
                                        if (row_idx, check_col) in cell_rects_map:
                                            other_rect = cell_rects_map[(row_idx, check_col)]
                                            # If there's a separate physical cell that doesn't overlap significantly,
                                            # reduce colspan
                                            intersection = cell_rect & other_rect
                                            if intersection.is_empty or abs(intersection) / min(abs(cell_rect), abs(other_rect)) < 0.5:
                                                # Separate cell found, reduce colspan
                                                colspan = check_col - col_idx
                                                break
                                
                                # Cap at reasonable values
                                rowspan = min(rowspan, row_count - row_idx)
                                colspan = min(colspan, col_count - col_idx)
                                
                                # Create cell dictionary with all information
                                cell_dict = {
                                    "text": cell_text if cell_text else "",
                                    "row": row_idx,
                                    "col": col_idx,
                                    "rowspan": rowspan,
                                    "colspan": colspan,
                                    "bbox": tuple(cell_rect) if not cell_rect.is_empty else None,
                                    "is_merged": False,  # This is the primary cell
                                    "merged_from": None,  # No parent cell
                                }
                                matrix[row_idx][col_idx] = cell_dict
                                
                                # Fill all positions covered by this merged cell
                                # If cell is merged, mark all covered positions
                                if rowspan > 1 or colspan > 1:
                                    for r_offset in range(rowspan):
                                        for c_offset in range(colspan):
                                            covered_row = row_idx + r_offset
                                            covered_col = col_idx + c_offset
                                            
                                            # Skip the primary cell position (already filled)
                                            if r_offset == 0 and c_offset == 0:
                                                continue
                                            
                                            # Only fill if within bounds and not already filled
                                            if (covered_row < row_count and 
                                                covered_col < col_count and
                                                matrix[covered_row][covered_col] is None):
                                                # Create a merged cell reference
                                                merged_cell_dict = {
                                                    "text": cell_text if cell_text else "",  # Same text as primary
                                                    "row": covered_row,
                                                    "col": covered_col,
                                                    "rowspan": 1,  # This position itself is 1x1
                                                    "colspan": 1,
                                                    "bbox": tuple(cell_rect) if not cell_rect.is_empty else None,  # Same bbox
                                                    "is_merged": True,  # This is a merged position
                                                    "merged_from": (row_idx, col_idx),  # Reference to primary cell
                                                    "primary_row": row_idx,  # Row of primary cell
                                                    "primary_col": col_idx,  # Col of primary cell
                                                }
                                                matrix[covered_row][covered_col] = merged_cell_dict
                            except Exception:
                                # Create empty cell dict on error
                                cell_dict = {
                                    "text": "",
                                    "row": row_idx,
                                    "col": col_idx,
                                    "rowspan": 1,
                                    "colspan": 1,
                                    "bbox": None,
                                    "is_merged": False,
                                    "merged_from": None,
                                }
                                matrix[row_idx][col_idx] = cell_dict
                
                # Detect empty cells (None) that should be merged with cells above them
                # This handles cases where PyMuPDF returns None for merged cell positions
                # Build a map of estimated cell positions based on existing cells and cell_boxes
                estimated_cell_positions = {}
                for row_idx in range(row_count):
                    for col_idx in range(col_count):
                        if (row_idx, col_idx) in cell_rects_map:
                            estimated_cell_positions[(row_idx, col_idx)] = cell_rects_map[(row_idx, col_idx)]
                        elif row_idx < len(cell_boxes) and col_idx < len(cell_boxes[row_idx]):
                            cell_data = cell_boxes[row_idx][col_idx]
                            if cell_data is not None:
                                estimated_cell_positions[(row_idx, col_idx)] = pymupdf.Rect(cell_data) if not isinstance(cell_data, pymupdf.Rect) else cell_data
                
                # Estimate positions for empty cells based on adjacent cells
                for row_idx in range(row_count):
                    for col_idx in range(col_count):
                        if (row_idx, col_idx) not in estimated_cell_positions:
                            # Try to estimate from adjacent cells
                            # Look for cells in same row (left/right) or same column (above)
                            estimated_rect = None
                            
                            # Check left neighbor
                            if col_idx > 0 and (row_idx, col_idx - 1) in estimated_cell_positions:
                                left_rect = estimated_cell_positions[(row_idx, col_idx - 1)]
                                # Estimate width from average cell width
                                if avg_cell_width > 0:
                                    estimated_rect = pymupdf.Rect(
                                        left_rect.x1, left_rect.y0,
                                        left_rect.x1 + avg_cell_width, left_rect.y1
                                    )
                            
                            # Check right neighbor
                            if estimated_rect is None and col_idx < col_count - 1 and (row_idx, col_idx + 1) in estimated_cell_positions:
                                right_rect = estimated_cell_positions[(row_idx, col_idx + 1)]
                                if avg_cell_width > 0:
                                    estimated_rect = pymupdf.Rect(
                                        right_rect.x0 - avg_cell_width, right_rect.y0,
                                        right_rect.x0, right_rect.y1
                                    )
                            
                            # Check cell above
                            if estimated_rect is None and row_idx > 0 and (row_idx - 1, col_idx) in estimated_cell_positions:
                                above_rect = estimated_cell_positions[(row_idx - 1, col_idx)]
                                if avg_cell_height > 0:
                                    estimated_rect = pymupdf.Rect(
                                        above_rect.x0, above_rect.y1,
                                        above_rect.x1, above_rect.y1 + avg_cell_height
                                    )
                            
                            if estimated_rect is not None:
                                estimated_cell_positions[(row_idx, col_idx)] = estimated_rect
                
                # Now detect empty cells that should be merged - check ALL rows including row 0
                # This is important for detecting merged cells that start at the top of the table
                for row_idx in range(row_count):  # Start from row 0, not row 1
                    for col_idx in range(col_count):
                        if matrix[row_idx][col_idx] is None:
                            # Check if this empty cell should be merged with a cell above or to the left
                            merged_with = None
                            empty_pos = (row_idx, col_idx)
                            
                            # Get estimated bbox for empty cell
                            empty_cell_bbox = estimated_cell_positions.get(empty_pos)
                            
                            # First, check for cells above that might span to this position
                            if row_idx > 0:
                                for check_row in range(row_idx - 1, -1, -1):  # Check rows above, from closest to furthest
                                    check_cell = matrix[check_row][col_idx]
                                    if check_cell is not None and isinstance(check_cell, dict) and not check_cell.get("is_merged", False):
                                        # Check if cell above has rowspan that would cover this row
                                        check_rowspan = check_cell.get("rowspan", 1)
                                        if check_rowspan > 1:
                                            rows_covered = check_row + check_rowspan
                                            if rows_covered > row_idx:
                                                # This cell spans to cover the empty cell position
                                                merged_with = check_cell
                                                break
                                        
                                        # Also check bbox overlap if available
                                        if empty_cell_bbox is not None and check_cell.get("bbox"):
                                            check_bbox = pymupdf.Rect(check_cell["bbox"])
                                            # Check if the cell above vertically spans to cover this empty cell
                                            # Allow tolerance for alignment
                                            tolerance = avg_cell_height * 0.15 if avg_cell_height > 0 else 3
                                            
                                            # Check vertical overlap - cell above should extend down to cover this position
                                            vertical_overlap = check_bbox.y1 >= empty_cell_bbox.y0 - tolerance
                                            horizontal_alignment = (abs(check_bbox.x0 - empty_cell_bbox.x0) < tolerance * 2 and
                                                                   abs(check_bbox.x1 - empty_cell_bbox.x1) < tolerance * 2)
                                            
                                            if vertical_overlap and horizontal_alignment:
                                                # The cell above covers this position - it's a merged cell
                                                merged_with = check_cell
                                                break
                            
                            # If not merged vertically, check for horizontal merging (colspan from left)
                            if merged_with is None and col_idx > 0:
                                for check_col in range(col_idx - 1, -1, -1):
                                    check_cell = matrix[row_idx][check_col]
                                    if check_cell is not None and isinstance(check_cell, dict) and not check_cell.get("is_merged", False):
                                        # Check if cell to the left has colspan that would cover this column
                                        check_colspan = check_cell.get("colspan", 1)
                                        if check_colspan > 1:
                                            cols_covered = check_col + check_colspan
                                            if cols_covered > col_idx:
                                                # This cell spans to cover the empty cell position
                                                merged_with = check_cell
                                                break
                                        
                                        # Also check bbox overlap if available
                                        if empty_cell_bbox is not None and check_cell.get("bbox"):
                                            check_bbox = pymupdf.Rect(check_cell["bbox"])
                                            # Check if the cell to the left horizontally spans to cover this empty cell
                                            tolerance = avg_cell_width * 0.15 if avg_cell_width > 0 else 3
                                            
                                            # Check horizontal overlap
                                            horizontal_overlap = check_bbox.x1 >= empty_cell_bbox.x0 - tolerance
                                            vertical_alignment = (abs(check_bbox.y0 - empty_cell_bbox.y0) < tolerance * 2 and
                                                                 abs(check_bbox.y1 - empty_cell_bbox.y1) < tolerance * 2)
                                            
                                            if horizontal_overlap and vertical_alignment:
                                                # The cell to the left covers this position - it's a merged cell
                                                merged_with = check_cell
                                                break
                            
                            if merged_with is not None:
                                # Update rowspan/colspan of the primary cell if needed
                                primary_row = merged_with["row"]
                                primary_col = merged_with["col"]
                                primary_cell = matrix[primary_row][primary_col]
                                if primary_cell is not None and isinstance(primary_cell, dict):
                                    # Check if we need to update rowspan (vertical merge)
                                    if row_idx > primary_row:
                                        current_rowspan = primary_cell.get("rowspan", 1)
                                        required_rowspan = row_idx - primary_row + 1
                                        if required_rowspan > current_rowspan:
                                            primary_cell["rowspan"] = required_rowspan
                                            # Update the matrix reference
                                            matrix[primary_row][primary_col] = primary_cell
                                    
                                    # Check if we need to update colspan (horizontal merge)
                                    if col_idx > primary_col:
                                        current_colspan = primary_cell.get("colspan", 1)
                                        required_colspan = col_idx - primary_col + 1
                                        if required_colspan > current_colspan:
                                            primary_cell["colspan"] = required_colspan
                                            # Update the matrix reference
                                            matrix[primary_row][primary_col] = primary_cell
                                
                                # Mark as merged cell
                                matrix[row_idx][col_idx] = {
                                    "text": merged_with.get("text", ""),
                                    "row": row_idx,
                                    "col": col_idx,
                                    "rowspan": 1,
                                    "colspan": 1,
                                    "bbox": merged_with.get("bbox"),
                                    "is_merged": True,
                                    "merged_from": (merged_with["row"], merged_with["col"]),
                                    "primary_row": merged_with["row"],
                                    "primary_col": merged_with["col"],
                                }
                
                # Replace remaining None with empty cell dictionaries for consistency
                for row_idx in range(row_count):
                    for col_idx in range(col_count):
                        if matrix[row_idx][col_idx] is None:
                            matrix[row_idx][col_idx] = {
                                "text": "",
                                "row": row_idx,
                                "col": col_idx,
                                "rowspan": 1,
                                "colspan": 1,
                                "bbox": None,
                                "is_merged": False,
                                "merged_from": None,
                            }
            except Exception as e:
                # Fallback to t.extract() if something goes wrong
                try:
                    extracted = t.extract()
                    # Convert simple matrix to rich format
                    matrix = []
                    for row_idx, row in enumerate(extracted):
                        matrix_row = []
                        for col_idx, cell in enumerate(row):
                            cell_text = cell if cell is not None else ""
                            cell_dict = {
                                "text": cell_text if isinstance(cell_text, str) else str(cell_text),
                                "row": row_idx,
                                "col": col_idx,
                                "rowspan": 1,
                                "colspan": 1,
                                "bbox": None,
                                "is_merged": False,
                                "merged_from": None,
                            }
                            matrix_row.append(cell_dict)
                        matrix.append(matrix_row)
                except Exception:
                    matrix = []
            # Final cleanup: normalize any remaining HTML line breaks in table text
            for row in matrix:
                for cell in row:
                    if isinstance(cell, dict) and cell.get("text"):
                        cell["text"] = normalize_table_text(
                            cell["text"], keep_newlines=True
                        )
            # Extract markdown representation
            try:
                markdown = _normalize_table_br_tags(t.to_markdown(clean=False))
            except Exception:
                markdown = ""
            # attempt to rebuild a sanitized matrix from the markdown
            # string generated by MuPDF.  The markdown output has already had
            # line breaks and hyphenation repaired by MuPDF's internal
            # algorithm, so using it as the source for ASCII text guarantees
            # the two representations will match.
            sanitized = None
            if markdown:
                try:
                    md_lines = [l for l in markdown.splitlines() if l.strip()]
                    sep_idx = None
                    for j in range(len(md_lines) - 1):
                        if re.match(r"^\|?[-:\s|]+\|?$", md_lines[j + 1]):
                            sep_idx = j + 1
                            break
                    if sep_idx is not None:
                        header_line = md_lines[sep_idx - 1] if sep_idx > 0 else ""
                        data_lines = md_lines[sep_idx + 1 :]
                        parsed_rows = []
                        if header_line:
                            header_parts = [
                                p.strip()
                                for p in header_line.strip().strip("|").split("|")
                            ]
                            parsed_rows.append(header_parts)
                        for dl in data_lines:
                            parts = [p.strip() for p in dl.strip().strip("|").split("|")]
                            parsed_rows.append(parts)
                        if parsed_rows:
                            cols = max(len(r) for r in parsed_rows)
                            if cols == t.col_count:
                                sanitized = []
                                for r_idx, prow in enumerate(parsed_rows):
                                    srow = []
                                    for c_idx in range(cols):
                                        txt = prow[c_idx] if c_idx < len(prow) else ""
                                        cell_dict = {
                                            "text": txt,
                                            "row": r_idx,
                                            "col": c_idx,
                                            "rowspan": 1,
                                            "colspan": 1,
                                            "bbox": None,
                                            "is_merged": False,
                                            "merged_from": None,
                                        }
                                        srow.append(cell_dict)
                                    sanitized.append(srow)
                except Exception:
                    sanitized = None
            if sanitized is None:
                # fallback: just normalise each cell text (collapse newlines)
                try:
                    sanitized = []
                    for row in matrix:
                        srow = []
                        for cell in row:
                            if isinstance(cell, dict):
                                c = dict(cell)
                                txt = c.get("text", "") or ""
                                c["text"] = normalize_table_text(txt, keep_newlines=False)
                                srow.append(c)
                            else:
                                srow.append(normalize_table_text(str(cell) if cell is not None else "", keep_newlines=False))
                        sanitized.append(srow)
                except Exception:
                    sanitized = matrix

            tab_dict = {
                "bbox": tuple(tab_rects[i]),
                "rows": t.row_count,
                "columns": t.col_count,
                "matrix": matrix,
                "markdown": markdown,
                # Optional representation in a simple ASCII table
                "matrix_ascii": matrix_to_ascii(sanitized),
            }
            parms.tables.append(tab_dict)
        # Keep a per-table list aligned to tabs before merging.
        parms.tables_by_tab = list(parms.tables)
        # After building the list of tables for this page, merge those that
        # are parts of the same logical table (same columns, aligned and close).
        parms.tables = merge_split_tables(parms.tables)
        parms.tab_rects = tab_rects
        # list of table rectangles
        parms.tab_rects0 = list(tab_rects.values())

        # Select paths not intersecting any table.
        # Ignore full page graphics.
        # Ignore fill paths having the background color.
        if not IGNORE_GRAPHICS:
            paths = [
                p
                for p in page.get_drawings()
                if p["rect"] in parms.clip
                and p["rect"].width < parms.clip.width
                and p["rect"].height < parms.clip.height
                and (p["rect"].width > 3 or p["rect"].height > 3)
                and not (p["type"] == "f" and p["fill"] == parms.bg_color)
                and outside_all_bboxes(p["rect"], parms.tab_rects0)
                and outside_all_bboxes(p["rect"], parms.annot_rects)
            ]
        else:
            paths = []
        # catch too-many-graphics situation
        if IGNORE_GRAPHICS:
            paths = []

        # We also ignore vector graphics that only represent
        # "text emphasizing sugar".
        vg_clusters0 = []  # worthwhile vector graphics go here

        # walk through all vector graphics outside any table
        clusters = page.cluster_drawings(drawings=paths)
        for bbox in clusters:
            if is_significant(bbox, paths):
                vg_clusters0.append(bbox)

        # remove paths that are not in some relevant graphic
        parms.actual_paths = [
            p for p in paths if bbox_in_any_bbox(p["rect"], vg_clusters0)
        ]

        # also add image rectangles to the list and vice versa
        vg_clusters0.extend(parms.img_rects)
        parms.img_rects.extend(vg_clusters0)
        parms.img_rects = sorted(set(parms.img_rects), key=lambda r: (r.y1, r.x0))
        parms.written_images = []
        # these may no longer be pairwise disjoint:
        # remove area overlaps by joining into larger rects
        parms.vg_clusters0 = refine_boxes(vg_clusters0)

        parms.vg_clusters = dict((i, r) for i, r in enumerate(parms.vg_clusters0))
        block_count = len(parms.textpage.extractBLOCKS())
        if block_count > 0:
            char_density = len(parms.textpage.extractTEXT()) / block_count
        else:
            char_density = 0
        # identify text bboxes on page, avoiding tables, images and graphics
        if too_many_graphics and char_density < 20:
            # This page has too many isolated text pieces for meaningful
            # layout analysis. Treat whole page as one text block.
            text_rects = [parms.clip]
        else:
            text_rects = column_boxes(
                parms.page,
                paths=parms.actual_paths,
                no_image_text=not force_text,
                textpage=parms.textpage,
                avoid=parms.tab_rects0 + parms.vg_clusters0,
                footer_margin=margins[3],
                header_margin=margins[1],
                ignore_images=IGNORE_IMAGES,
            )

        for text_rect in text_rects:
            # output tables above this rectangle
            md_tables, ascii_tables = output_tables(parms, text_rect)
            md_images, ascii_images = output_images(parms, text_rect, force_text)
            parms.md_string += md_tables + md_images
            parms.md_string_ascii += ascii_tables + ascii_images

            # output text inside this rectangle
            md_text, ascii_text = write_text(
                parms,
                text_rect,
                force_text=force_text,
                images=True,
                tables=True,
                base_offset=len(parms.md_string),
            )
            parms.md_string += md_text
            parms.md_string_ascii += ascii_text

        md_tables, ascii_tables = output_tables(parms, None)
        md_images, ascii_images = output_images(parms, None, force_text)
        parms.md_string += md_tables + md_images
        parms.md_string_ascii += ascii_tables + ascii_images

        while parms.md_string.startswith("\n"):
            parms.md_string = parms.md_string[1:]

        parms.md_string = parms.md_string.replace(chr(0), REPLACEMENT_CHARACTER)
        while parms.md_string_ascii.startswith("\n"):
            parms.md_string_ascii = parms.md_string_ascii[1:]
        parms.md_string_ascii = parms.md_string_ascii.replace(chr(0), REPLACEMENT_CHARACTER)

        if EXTRACT_WORDS is True:
            # output words in sequence compliant with Markdown text
            rawwords = parms.textpage.extractWORDS()
            rawwords.sort(key=lambda w: (w[3], w[0]))

            words = []
            for lrect in parms.line_rects:
                lwords = []
                for w in rawwords:
                    wrect = pymupdf.Rect(w[:4])
                    if wrect in lrect:
                        lwords.append(w)
                words.extend(sort_words(lwords))

            # remove word duplicates without spoiling the sequence
            # duplicates may occur for multiple reasons
            nwords = []  # words w/o duplicates
            for w in words:
                if w not in nwords:
                    nwords.append(w)
            words = nwords

        else:
            words = []
        parms.words = words
        if page_separators:
            # add page separators to output
            parms.md_string += f"\n\n--- end of page={parms.page.number} ---\n\n"
        return parms

    if page_chunks is False:
        document_output = []  # Use list for efficient string concatenation
    else:
        document_output = []

    # read the Table of Contents
    toc = doc.get_toc()

    # Text extraction flags:
    # omit clipped text, collect styles, use accurate bounding boxes
    textflags = (
        0
        | pymupdf.TEXT_MEDIABOX_CLIP
        # | pymupdf.TEXT_ACCURATE_BBOXES
        | pymupdf.TEXT_COLLECT_STYLES
    )
    pymupdf.table.FLAGS = (
        0
        | pymupdf.TEXTFLAGS_TEXT
        | pymupdf.TEXT_COLLECT_STYLES
        # | pymupdf.TEXT_ACCURATE_BBOXES
        | pymupdf.TEXT_MEDIABOX_CLIP
    )
    # optionally replace REPLACEMENT_CHARACTER by glyph number
    if use_glyphs and hasattr(mupdf, "FZ_STEXT_USE_GID_FOR_UNKNOWN_UNICODE"):
        mupdf.FZ_STEXT_USE_GID_FOR_UNKNOWN_UNICODE = 1

    if show_progress:
        print(f"Processing {FILENAME}...")
        pages = ProgressBar(pages)
    for pno in pages:
        parms = get_page_output(
            doc,
            pno,
            margins,
            textflags,
            FILENAME,
            IGNORE_IMAGES,
            IGNORE_GRAPHICS,
        )
        if page_chunks is False:
            if table_format == "ascii":
                document_output.append(parms.md_string_ascii)
            else:
                document_output.append(parms.md_string)
        else:
            # build subet of TOC for this page
            page_tocs = [t for t in toc if t[-1] == pno + 1]

            metadata = get_metadata(doc, pno)
            if table_format == "ascii":
                page_text = parms.md_string_ascii
            else:
                page_text = parms.md_string
            document_output.append(
                {
                    "metadata": metadata,
                    "toc_items": page_tocs,
                    "tables": parms.tables,
                    "images": parms.images,
                    "graphics": parms.graphics,
                    "text": page_text,
                    "text_markdown": parms.md_string,
                    "text_ascii": parms.md_string_ascii,
                    "words": parms.words,
                }
            )
        del parms

    # Join the list of strings efficiently if not using page chunks
    if page_chunks is False:
        return "".join(document_output)
    return document_output


def _remove_overlapping_images(img_info):
    """Remove images that are completely contained within larger images.
    
    Assumes img_info is sorted by area (largest first).
    This avoids duplicating the same removal logic across multiple functions.
    """
    img_info.sort(key=lambda i: abs(i["bbox"]), reverse=True)
    i = len(img_info) - 1
    while i > 0:
        r = img_info[i]["bbox"]
        if r.is_empty:
            del img_info[i]
            i -= 1
            continue
        # Check if this image is contained in any larger image
        for j in range(i):
            if r in img_info[j]["bbox"]:
                del img_info[i]
                break
        i -= 1
    return img_info


def extract_images_on_page_simple(page, parms, image_size_limit):
    img_info = page.get_image_info()
    # Update bboxes in-place (single pass)
    for item in img_info:
        item["bbox"] = pymupdf.Rect(item["bbox"]) & parms.clip
    
    # Remove overlapping images
    img_info = _remove_overlapping_images(img_info)
    return img_info


def filter_small_images(page, parms, image_size_limit):
    img_info = []
    for item in page.get_image_info():
        r = pymupdf.Rect(item["bbox"]) & parms.clip
        if r.is_empty or (
            max(r.width / page.rect.width, r.height / page.rect.height)
            < image_size_limit
        ):
            continue
        item["bbox"] = r
        img_info.append(item)
    return img_info


def extract_images_on_page_simple_drop(page, parms, image_size_limit):
    img_info = filter_small_images(page, parms, image_size_limit)
    # Remove overlapping images
    img_info = _remove_overlapping_images(img_info)
    return img_info


if __name__ == "__main__":
    import pathlib
    import sys
    import time

    try:
        filename = sys.argv[1]
    except IndexError:
        print(f"Usage:\npython {os.path.basename(__file__)} input.pdf")
        sys.exit()

    t0 = time.perf_counter()

    doc = pymupdf.open(filename)
    parms = sys.argv[2:]
    pages = range(doc.page_count)
    if len(parms) == 2 and parms[0] == "-pages":
        pages = []
        pages_spec = parms[1].replace("N", f"{doc.page_count}")
        for spec in pages_spec.split(","):
            if "-" in spec:
                start, end = map(int, spec.split("-"))
                pages.extend(range(start - 1, end))
            else:
                pages.append(int(spec) - 1)

        wrong_pages = set([n + 1 for n in pages if n >= doc.page_count][:4])
        if wrong_pages:
            sys.exit(f"Page number(s) {wrong_pages} not in '{doc}'.")
    md_string = to_markdown(
        doc,
        pages=pages,
    )
    FILENAME = doc.name
    outname = FILENAME + ".md"
    pathlib.Path(outname).write_bytes(md_string.encode())
    t1 = time.perf_counter()
    print(f"Markdown creation time for {FILENAME=} {round(t1-t0,2)} sec.")
