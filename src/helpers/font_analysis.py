"""Font reliability analysis for PDF text extraction.

Detects fonts with potentially incorrect ToUnicode mappings that can cause
silent text corruption during extraction.  When a font's ToUnicode CMap
maps symbol glyphs to wrong Unicode code points, every PDF text extractor
(PyMuPDF, pypdf, pdfplumber, poppler) will faithfully reproduce the
*wrong* characters -- the corruption is in the font metadata, not the
extractor.  This module inspects font metadata and flags pages where
text extraction may be unreliable, so callers can fall back to OCR or
vision-based extraction for those pages.

Typical usage::

    import pymupdf
    from pymupdf4llm.helpers.font_analysis import analyze_fonts

    doc = pymupdf.open("paper.pdf")
    report = analyze_fonts(doc)
    for page_report in report:
        if page_report["unreliable"]:
            print(f"Page {page_report['page']}: use VLM/OCR")
            for w in page_report["warnings"]:
                print(f"  - {w['message']}")

Copyright (C) 2024-2025 Artifex Software, Inc.
License: GNU AGPL v3 (see LICENSE).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pymupdf

# ---------------------------------------------------------------------------
# Adobe Glyph List (AGL) subset -- the ~180 most commonly encountered
# glyph names for math, Greek, arrows, and typographic symbols.
# Full list: https://github.com/adobe-type-tools/agl-aglfn
# ---------------------------------------------------------------------------
_AGL: Dict[str, int] = {
    # Greek uppercase
    "Alpha": 0x0391, "Beta": 0x0392, "Gamma": 0x0393, "Delta": 0x0394,
    "Epsilon": 0x0395, "Zeta": 0x0396, "Eta": 0x0397, "Theta": 0x0398,
    "Iota": 0x0399, "Kappa": 0x039A, "Lambda": 0x039B, "Mu": 0x039C,
    "Nu": 0x039D, "Xi": 0x039E, "Omicron": 0x039F, "Pi": 0x03A0,
    "Rho": 0x03A1, "Sigma": 0x03A3, "Tau": 0x03A4, "Upsilon": 0x03A5,
    "Phi": 0x03A6, "Chi": 0x03A7, "Psi": 0x03A8, "Omega": 0x03A9,
    # Greek lowercase
    "alpha": 0x03B1, "beta": 0x03B2, "gamma": 0x03B3, "delta": 0x03B4,
    "epsilon": 0x03B5, "zeta": 0x03B6, "eta": 0x03B7, "theta": 0x03B8,
    "iota": 0x03B9, "kappa": 0x03BA, "lambda": 0x03BB, "mu": 0x03BC,
    "nu": 0x03BD, "xi": 0x03BE, "omicron": 0x03BF, "pi": 0x03C0,
    "rho": 0x03C1, "sigma": 0x03C3, "tau": 0x03C4, "upsilon": 0x03C5,
    "phi": 0x03C6, "chi": 0x03C7, "psi": 0x03C8, "omega": 0x03C9,
    "sigma1": 0x03C2, "phi1": 0x03D5, "theta1": 0x03D1,
    # Arrows
    "arrowleft": 0x2190, "arrowup": 0x2191, "arrowright": 0x2192,
    "arrowdown": 0x2193, "arrowboth": 0x2194, "arrowupdn": 0x2195,
    "arrowdblright": 0x21D2, "arrowdblleft": 0x21D0,
    "arrowdblboth": 0x21D4, "arrowdblup": 0x21D1, "arrowdbldown": 0x21D3,
    # Math operators
    "plus": 0x002B, "minus": 0x2212, "multiply": 0x00D7, "divide": 0x00F7,
    "equal": 0x003D, "notequal": 0x2260, "lessequal": 0x2264,
    "greaterequal": 0x2265, "plusminus": 0x00B1, "approxequal": 0x2248,
    "infinity": 0x221E, "integral": 0x222B, "summation": 0x2211,
    "product": 0x220F, "radical": 0x221A, "proportional": 0x221D,
    "partialdiff": 0x2202, "nabla": 0x2207, "emptyset": 0x2205,
    "element": 0x2208, "notelement": 0x2209, "suchthat": 0x220B,
    "intersection": 0x2229, "union": 0x222A, "propersubset": 0x2282,
    "propersuperset": 0x2283, "reflexsubset": 0x2286,
    "reflexsuperset": 0x2287, "circleplus": 0x2295,
    "circlemultiply": 0x2297, "perpendicular": 0x22A5,
    "dotmath": 0x22C5, "logicaland": 0x2227, "logicalor": 0x2228,
    "logicalnot": 0x00AC, "therefore": 0x2234, "because": 0x2235,
    "equivalence": 0x2261, "congruent": 0x2245,
    # Typographic
    "endash": 0x2013, "emdash": 0x2014, "bullet": 0x2022,
    "ellipsis": 0x2026, "degree": 0x00B0, "dagger": 0x2020,
    "daggerdbl": 0x2021, "section": 0x00A7, "paragraph": 0x00B6,
    "copyright": 0x00A9, "registered": 0x00AE, "trademark": 0x2122,
    "fi": 0x0066_0069, "fl": 0x0066_006C,
    # Misc symbols
    "heart": 0x2665, "diamond": 0x2666, "club": 0x2663, "spade": 0x2660,
    "lozenge": 0x25CA, "dieresis": 0x00A8, "acute": 0x00B4,
    "grave": 0x0060, "circumflex": 0x005E, "tilde": 0x007E,
    "ring": 0x02DA, "cedilla": 0x00B8, "dotaccent": 0x02D9,
    "hungarumlaut": 0x02DD, "ogonek": 0x02DB, "caron": 0x02C7,
    "breve": 0x02D8, "macron": 0x00AF,
    # Sub/superscripts in some fonts
    "zero": 0x0030, "one": 0x0031, "two": 0x0032, "three": 0x0033,
    "four": 0x0034, "five": 0x0035, "six": 0x0036, "seven": 0x0037,
    "eight": 0x0038, "nine": 0x0039,
}

# Unicode categories considered "symbol" or "math" (not normal text)
_SYMBOL_CATEGORIES = {"Sm", "So", "Sk", "No"}

# Font names that hint at symbol/math content
_SYMBOL_FONT_PATTERNS = re.compile(
    r"(?i)(symbol|math|cmsy|cmmi|cmex|msam|msbm|stix|cambria.?math"
    r"|euclid|zapf.?dingbat|wingding|webding|mt.?extra"
    r"|lucida.?math|asana.?math|xits.?math|latin.?modern.?math)",
)


# ---------------------------------------------------------------------------
# CMap parser (minimal, handles beginbfchar / beginbfrange)
# ---------------------------------------------------------------------------
_HEX_RE = re.compile(r"<([0-9A-Fa-f]+)>")


def _parse_tounicode_cmap(stream_bytes: bytes) -> Dict[int, int]:
    """Parse a ToUnicode CMap and return {source_code: unicode_codepoint}.

    Only handles the beginbfchar / beginbfrange operators that account for
    virtually all ToUnicode maps in practice.
    """
    try:
        text = stream_bytes.decode("latin-1")
    except Exception:
        return {}

    mapping: Dict[int, int] = {}

    # beginbfchar sections: pairs of <src> <dst>
    for section in re.finditer(
        r"beginbfchar\s*(.*?)\s*endbfchar", text, re.S
    ):
        hexvals = _HEX_RE.findall(section.group(1))
        for i in range(0, len(hexvals) - 1, 2):
            src = int(hexvals[i], 16)
            dst = int(hexvals[i + 1], 16)
            mapping[src] = dst

    # beginbfrange sections: triples of <start> <end> <dst_start>
    for section in re.finditer(
        r"beginbfrange\s*(.*?)\s*endbfrange", text, re.S
    ):
        hexvals = _HEX_RE.findall(section.group(1))
        for i in range(0, len(hexvals) - 2, 3):
            start = int(hexvals[i], 16)
            end = int(hexvals[i + 1], 16)
            dst = int(hexvals[i + 2], 16)
            for code in range(start, end + 1):
                mapping[code] = dst + (code - start)

    return mapping


# ---------------------------------------------------------------------------
# Glyph-name resolution
# ---------------------------------------------------------------------------

def _resolve_glyph_name(name: str) -> Optional[int]:
    """Resolve a glyph name to a Unicode code point using the AGL.

    Returns None if the name cannot be resolved.
    """
    if not name or name in ("", ".notdef", ".null"):
        return None

    # Direct AGL lookup
    if name in _AGL:
        return _AGL[name]

    # uni<XXXX> convention (e.g. "uni2192" -> 0x2192)
    m = re.match(r"^uni([0-9A-Fa-f]{4,6})$", name)
    if m:
        return int(m.group(1), 16)

    # u<XXXX> convention
    m = re.match(r"^u([0-9A-Fa-f]{4,6})$", name)
    if m:
        return int(m.group(1), 16)

    return None


# ---------------------------------------------------------------------------
# Font inspection
# ---------------------------------------------------------------------------

@dataclass
class FontWarning:
    """A single warning about a font's text extraction reliability."""
    font_name: str
    font_xref: int
    category: str  # "tounicode_conflict", "duplicate_mapping", "missing_tounicode", "symbol_to_ascii"
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class PageFontReport:
    """Font reliability report for a single page."""
    page: int
    warnings: List[FontWarning] = field(default_factory=list)

    @property
    def unreliable(self) -> bool:
        return len(self.warnings) > 0


def _get_tounicode_stream(doc: pymupdf.Document, font_xref: int) -> Optional[bytes]:
    """Read the ToUnicode CMap stream for a font, if it has one."""
    try:
        tu_key = doc.xref_get_key(font_xref, "ToUnicode")
    except Exception:
        return None

    if tu_key[0] == "null":
        return None

    # ToUnicode is an indirect reference like "14 0 R"
    m = re.search(r"(\d+)\s+\d+\s+R", tu_key[1])
    if not m:
        return None

    tu_xref = int(m.group(1))
    try:
        return doc.xref_stream(tu_xref)
    except Exception:
        return None


def _get_encoding_differences(doc: pymupdf.Document, font_xref: int) -> Dict[int, str]:
    """Extract the Encoding/Differences array: {char_code: glyph_name}."""
    differences: Dict[int, str] = {}
    try:
        enc_key = doc.xref_get_key(font_xref, "Encoding")
    except Exception:
        return differences

    # Encoding can be a name ("/WinAnsiEncoding") or an indirect ref to a dict
    if enc_key[0] == "name":
        return differences  # standard encoding, no Differences to parse

    if enc_key[0] in ("null",):
        return differences

    # Try to get the Encoding dict (might be inline or indirect)
    enc_text = ""
    if enc_key[0] == "xref":
        m = re.search(r"(\d+)\s+\d+\s+R", enc_key[1])
        if m:
            try:
                enc_text = doc.xref_object(int(m.group(1)))
            except Exception:
                return differences
    elif enc_key[0] == "dict":
        enc_text = enc_key[1]

    # Parse the Differences array
    diff_match = re.search(r"/Differences\s*\[(.*?)\]", enc_text, re.S)
    if not diff_match:
        return differences

    tokens = diff_match.group(1).split()
    code = 0
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if tok.isdigit():
            code = int(tok)
        elif tok.startswith("/"):
            differences[code] = tok[1:]  # strip leading /
            code += 1

    return differences


def _is_symbol_font(font_name: str) -> bool:
    """Heuristic: does the font name suggest symbol/math content?"""
    return bool(_SYMBOL_FONT_PATTERNS.search(font_name))


def _check_tounicode_conflicts(
    tounicode_map: Dict[int, int],
    glyph_names: Dict[int, str],
    font_name: str,
    font_xref: int,
) -> List[FontWarning]:
    """Check for conflicts between ToUnicode and glyph-name resolutions."""
    warnings = []
    conflicts = []

    for code, glyph_name in glyph_names.items():
        if code not in tounicode_map:
            continue
        expected = _resolve_glyph_name(glyph_name)
        if expected is None:
            continue
        actual = tounicode_map[code]
        if actual != expected:
            conflicts.append({
                "char_code": code,
                "glyph_name": glyph_name,
                "tounicode_says": actual,
                "glyph_name_says": expected,
            })

    if conflicts:
        warnings.append(FontWarning(
            font_name=font_name,
            font_xref=font_xref,
            category="tounicode_conflict",
            message=(
                f"Font '{font_name}': ToUnicode disagrees with glyph names "
                f"for {len(conflicts)} character(s). Text extraction may "
                f"produce wrong characters for these glyphs."
            ),
            details={"conflicts": conflicts},
        ))

    return warnings


def _check_duplicate_mappings(
    tounicode_map: Dict[int, int],
    font_name: str,
    font_xref: int,
) -> List[FontWarning]:
    """Check for multiple source codes mapping to the same Unicode code point."""
    warnings = []

    # Count how many source codes map to each target
    target_counts: Dict[int, List[int]] = defaultdict(list)
    for src, dst in tounicode_map.items():
        target_counts[dst].append(src)

    # Filter to targets that are non-whitespace and have multiple sources,
    # excluding the common case of small-caps/all-caps fonts where uppercase
    # and lowercase source codes map to the same target (e.g., 'M' and 'm'
    # both mapping to 'm' in a small-caps font).
    duplicates = {}
    for dst, sources in target_counts.items():
        if len(sources) <= 1 or dst <= 0x20:
            continue
        # Exclude case-folding pairs: if the sources differ only by case
        # (upper vs lower of the same letter), it's likely a small-caps font
        if len(sources) == 2:
            a, b = sources
            if abs(a - b) == 0x20 and 0x41 <= min(a, b) <= 0x5A:
                continue
        try:
            cat = unicodedata.category(chr(dst))
        except (ValueError, OverflowError):
            cat = "??"
        if dst < 0x80 or cat not in _SYMBOL_CATEGORIES:
            duplicates[dst] = sources

    if duplicates:
        warnings.append(FontWarning(
            font_name=font_name,
            font_xref=font_xref,
            category="duplicate_mapping",
            message=(
                f"Font '{font_name}': {len(duplicates)} Unicode code point(s) "
                f"are targeted by multiple source codes. This often indicates "
                f"a broken ToUnicode CMap."
            ),
            details={"duplicates": {
                f"U+{dst:04X}": [f"0x{s:04X}" for s in sources]
                for dst, sources in duplicates.items()
            }},
        ))

    return warnings


def _check_symbol_to_ascii(
    tounicode_map: Dict[int, int],
    font_name: str,
    font_xref: int,
) -> List[FontWarning]:
    """Flag symbol/math fonts whose ToUnicode maps to basic ASCII.

    When a font with a symbol-like name maps most of its glyphs to
    U+0020..U+007E, the ToUnicode is likely wrong -- real symbol fonts
    should map to their actual Unicode code points.
    """
    if not _is_symbol_font(font_name):
        return []

    warnings = []
    ascii_count = sum(1 for v in tounicode_map.values() if 0x20 < v < 0x7F)
    total = len(tounicode_map)

    if total >= 3 and ascii_count / total > 0.7:
        warnings.append(FontWarning(
            font_name=font_name,
            font_xref=font_xref,
            category="symbol_to_ascii",
            message=(
                f"Font '{font_name}' appears to be a symbol/math font but "
                f"its ToUnicode maps {ascii_count}/{total} glyphs to basic "
                f"ASCII. This is a strong indicator of incorrect mappings."
            ),
            details={"ascii_ratio": ascii_count / total},
        ))

    return warnings


# Unicode blocks that contain symbols, math, Greek, and arrows
_SYMBOL_RANGES = [
    (0x0370, 0x03FF),  # Greek and Coptic
    (0x2000, 0x206F),  # General Punctuation (some symbols)
    (0x2100, 0x214F),  # Letterlike Symbols
    (0x2190, 0x21FF),  # Arrows
    (0x2200, 0x22FF),  # Mathematical Operators
    (0x2300, 0x23FF),  # Miscellaneous Technical
    (0x2500, 0x257F),  # Box Drawing
    (0x2580, 0x259F),  # Block Elements
    (0x25A0, 0x25FF),  # Geometric Shapes
    (0x2600, 0x26FF),  # Miscellaneous Symbols
    (0x2700, 0x27BF),  # Dingbats
    (0x27C0, 0x27EF),  # Misc Mathematical Symbols-A
    (0x2980, 0x29FF),  # Misc Mathematical Symbols-B
    (0x2A00, 0x2AFF),  # Supplemental Mathematical Operators
]


def _is_symbol_codepoint(cp: int) -> bool:
    """Is this code point in a symbol/math/Greek Unicode block?"""
    for lo, hi in _SYMBOL_RANGES:
        if lo <= cp <= hi:
            return True
    return False


def _is_basic_latin(cp: int) -> bool:
    """Is this a basic Latin letter or digit (the target of wrong mappings)?"""
    return (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A) or (0x0030 <= cp <= 0x0039)


def _check_category_mismatch(
    tounicode_map: Dict[int, int],
    font_name: str,
    font_xref: int,
) -> List[FontWarning]:
    """Detect source codes from symbol/Greek/math ranges mapped to basic Latin.

    This is the most reliable detector for the "wrong-character" corruption
    family.  In a correct ToUnicode CMap, a source code for a Greek letter
    or arrow should map to the corresponding Unicode symbol, not to a basic
    Latin letter.  For example, source 0x2192 (rightwards arrow position)
    should map to U+2192, not to U+006F ('o').

    This check works on CIDFonts (Type0/Identity-H) where the source codes
    are CIDs that semantically correspond to Unicode positions, and also on
    any font where the ToUnicode source-to-target pair shows a category
    mismatch.
    """
    warnings = []
    mismatches = []

    for src, dst in tounicode_map.items():
        if _is_symbol_codepoint(src) and _is_basic_latin(dst):
            try:
                src_name = unicodedata.name(chr(src), f"U+{src:04X}")
            except (ValueError, OverflowError):
                src_name = f"U+{src:04X}"
            mismatches.append({
                "source_code": f"0x{src:04X}",
                "source_name": src_name,
                "maps_to": f"U+{dst:04X} ('{chr(dst)}')",
                "expected_near": f"U+{src:04X}",
            })

    if mismatches:
        warnings.append(FontWarning(
            font_name=font_name,
            font_xref=font_xref,
            category="category_mismatch",
            message=(
                f"Font '{font_name}': {len(mismatches)} symbol/math/Greek "
                f"source code(s) map to basic Latin letters via ToUnicode. "
                f"This is a strong indicator that the font's character map "
                f"is incorrect and text extraction will produce wrong characters."
            ),
            details={"mismatches": mismatches},
        ))

    return warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_fonts(
    doc: Union[str, pymupdf.Document],
    pages: Optional[Sequence[int]] = None,
) -> List[PageFontReport]:
    """Analyze font reliability for text extraction across PDF pages.

    For each page, inspects every embedded font's ToUnicode CMap and
    encoding metadata to detect potential text corruption sources:

    - **category_mismatch**: source codes from symbol/math/Greek Unicode
      ranges are mapped to basic Latin letters via ToUnicode.  This is
      the primary indicator of the "wrong-character" corruption family
      and works reliably on CIDFonts (Type0/Identity-H) where glyph
      names are unavailable.
    - **tounicode_conflict**: the ToUnicode CMap disagrees with the font's
      glyph names (resolved via the Adobe Glyph List).
    - **duplicate_mapping**: multiple character codes map to the same
      Unicode code point, suggesting a broken CMap.
    - **symbol_to_ascii**: a symbol/math font maps most glyphs to basic
      ASCII, which almost always indicates incorrect ToUnicode.

    Args:
        doc: A file path or an open ``pymupdf.Document``.
        pages: Page indices to analyze (default: all pages).

    Returns:
        A list of :class:`PageFontReport`, one per analyzed page.
        Each report has an ``unreliable`` property (True if any
        warnings were raised) and a ``warnings`` list with structured
        :class:`FontWarning` objects.

    Example::

        report = analyze_fonts("document.pdf")
        unreliable_pages = [r.page for r in report if r.unreliable]
    """
    if isinstance(doc, (str, bytes)):
        doc = pymupdf.open(doc)
        should_close = True
    else:
        should_close = False

    try:
        return _analyze_fonts_impl(doc, pages)
    finally:
        if should_close:
            doc.close()


def _analyze_fonts_impl(
    doc: pymupdf.Document,
    pages: Optional[Sequence[int]],
) -> List[PageFontReport]:
    if pages is None:
        pages = range(doc.page_count)

    reports = []
    seen_fonts: Dict[int, List[FontWarning]] = {}

    for page_idx in pages:
        page = doc.load_page(page_idx)
        page_report = PageFontReport(page=page_idx)
        fonts = page.get_fonts(full=True)

        for font_info in fonts:
            xref, ext, ftype, basename, name, encoding, _ref = font_info

            if xref in seen_fonts:
                page_report.warnings.extend(seen_fonts[xref])
                continue

            font_warnings: List[FontWarning] = []
            font_name = basename or name or f"xref-{xref}"

            # 1. Read the ToUnicode CMap
            tu_stream = _get_tounicode_stream(doc, xref)
            tounicode_map = _parse_tounicode_cmap(tu_stream) if tu_stream else {}

            # 2. Read glyph names from the Encoding/Differences
            glyph_names = _get_encoding_differences(doc, xref)

            # 3. Check for category mismatch (symbol source -> Latin target)
            if tounicode_map:
                font_warnings.extend(_check_category_mismatch(
                    tounicode_map, font_name, xref,
                ))

            # 4. Check for ToUnicode vs glyph-name conflicts
            if tounicode_map and glyph_names:
                font_warnings.extend(_check_tounicode_conflicts(
                    tounicode_map, glyph_names, font_name, xref,
                ))

            # 5. Check for duplicate ToUnicode mappings
            if tounicode_map:
                font_warnings.extend(_check_duplicate_mappings(
                    tounicode_map, font_name, xref,
                ))

            # 6. Check for symbol fonts mapped to ASCII
            if tounicode_map:
                font_warnings.extend(_check_symbol_to_ascii(
                    tounicode_map, font_name, xref,
                ))

            seen_fonts[xref] = font_warnings
            page_report.warnings.extend(font_warnings)

        reports.append(page_report)

    return reports
