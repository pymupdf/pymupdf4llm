"""Tests for the font_analysis module.

These tests create synthetic PDFs with known-good and known-bad ToUnicode
CMaps and verify that the analyzer flags the correct issues without false
positives on clean PDFs.
"""

import os
import tempfile

import pymupdf
import pytest

from pymupdf4llm.helpers.font_analysis import (
    FontWarning,
    PageFontReport,
    analyze_fonts,
    _parse_tounicode_cmap,
    _resolve_glyph_name,
    _is_symbol_codepoint,
)


# ---------------------------------------------------------------------------
# Helpers for creating synthetic test PDFs
# ---------------------------------------------------------------------------

def _make_tounicode_cmap(mappings):
    """Build a ToUnicode CMap stream from {char_code: unicode_codepoint}."""
    lines = [
        "/CIDInit /ProcSet findresource begin",
        "12 dict begin",
        "begincmap",
        "/CIDSystemInfo",
        "<< /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def",
        "/CMapName /Adobe-Identity-UCS def",
        "/CMapType 2 def",
        "1 begincodespacerange",
        "<0000> <FFFF>",
        "endcodespacerange",
        f"{len(mappings)} beginbfchar",
    ]
    for code, uni in sorted(mappings.items()):
        lines.append(f"<{code:04X}> <{uni:04X}>")
    lines += [
        "endbfchar",
        "endcmap",
        "CMapName currentdict /CMap defineresource pop",
        "end",
        "end",
    ]
    return "\n".join(lines)


def _inject_tounicode(doc, font_xref, cmap_text):
    """Inject a ToUnicode CMap stream into a font."""
    xref = doc.get_new_xref()
    stream_bytes = cmap_text.encode("latin-1")
    obj_str = f"<< /Length {len(stream_bytes)} >>"
    doc.update_object(xref, obj_str)
    doc.update_stream(xref, stream_bytes)
    doc.xref_set_key(font_xref, "ToUnicode", f"{xref} 0 R")
    return xref


@pytest.fixture
def broken_tounicode_pdf(tmp_path):
    """PDF with a Symbol font whose ToUnicode maps symbols to wrong ASCII."""
    path = str(tmp_path / "broken_tounicode.pdf")

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Normal text.", fontsize=12, fontname="helv")
    tw = pymupdf.TextWriter(page.rect)
    tw.append(
        (72, 180),
        chr(0x2192) + chr(0x0394) + chr(0x03B1),
        font=pymupdf.Font("symb"),
        fontsize=14,
    )
    tw.write_text(page)
    doc.save(path)
    doc.close()

    doc = pymupdf.open(path)
    page = doc[0]
    for font_info in page.get_fonts(full=True):
        xref, _ext, _ftype, basename, name, _enc, _ref = font_info
        if "ymb" in basename.lower() or "ymb" in name.lower():
            broken_cmap = _make_tounicode_cmap({
                0x2192: 0x006F,  # arrow -> 'o' (WRONG)
                0x0394: 0x00A8,  # Delta -> dieresis (WRONG)
                0x03B1: 0x0061,  # alpha -> 'a' (WRONG)
            })
            _inject_tounicode(doc, xref, broken_cmap)
    doc.saveIncr()
    doc.close()
    return path


@pytest.fixture
def duplicate_tounicode_pdf(tmp_path):
    """PDF where multiple char codes map to the same Unicode code point."""
    path = str(tmp_path / "duplicate_tounicode.pdf")

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Text with duplicate mappings.", fontsize=12, fontname="helv")
    doc.save(path)
    doc.close()

    doc = pymupdf.open(path)
    page = doc[0]
    for font_info in page.get_fonts(full=True):
        xref = font_info[0]
        dup_cmap = _make_tounicode_cmap({
            0x0054: 0x006F,  # T -> o (WRONG)
            0x0065: 0x006F,  # e -> o (WRONG)
            0x0078: 0x006F,  # x -> o (WRONG)
            0x0074: 0x0074,  # t -> t (correct)
            0x0020: 0x0020,  # space -> space (correct)
        })
        _inject_tounicode(doc, xref, dup_cmap)
        break
    doc.saveIncr()
    doc.close()
    return path


@pytest.fixture
def clean_pdf(tmp_path):
    """Normal PDF with correct fonts - should produce zero warnings."""
    path = str(tmp_path / "clean.pdf")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Clean PDF with standard fonts.", fontsize=12, fontname="helv")
    page.insert_text((72, 130), "No mapping issues here.", fontsize=12, fontname="tiro")
    doc.save(path)
    doc.close()
    return path


# ---------------------------------------------------------------------------
# CMap parser tests
# ---------------------------------------------------------------------------

class TestCMapParser:
    def test_parse_bfchar(self):
        cmap = _make_tounicode_cmap({0x0041: 0x0042, 0x0043: 0x0044})
        mapping = _parse_tounicode_cmap(cmap.encode("latin-1"))
        assert mapping == {0x0041: 0x0042, 0x0043: 0x0044}

    def test_parse_empty(self):
        assert _parse_tounicode_cmap(b"") == {}
        assert _parse_tounicode_cmap(b"some random text") == {}

    def test_parse_bfrange(self):
        cmap_text = (
            "1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n"
            "1 beginbfrange\n<0041> <0043> <0061>\nendbfrange"
        )
        mapping = _parse_tounicode_cmap(cmap_text.encode("latin-1"))
        assert mapping == {0x0041: 0x0061, 0x0042: 0x0062, 0x0043: 0x0063}


# ---------------------------------------------------------------------------
# Glyph name resolution tests
# ---------------------------------------------------------------------------

class TestGlyphNameResolution:
    def test_agl_lookup(self):
        assert _resolve_glyph_name("arrowright") == 0x2192
        assert _resolve_glyph_name("Delta") == 0x0394
        assert _resolve_glyph_name("alpha") == 0x03B1

    def test_uni_prefix(self):
        assert _resolve_glyph_name("uni2192") == 0x2192
        assert _resolve_glyph_name("uni03B1") == 0x03B1

    def test_u_prefix(self):
        assert _resolve_glyph_name("u2192") == 0x2192

    def test_notdef(self):
        assert _resolve_glyph_name(".notdef") is None
        assert _resolve_glyph_name("") is None


# ---------------------------------------------------------------------------
# Symbol codepoint detection tests
# ---------------------------------------------------------------------------

class TestSymbolDetection:
    def test_arrows(self):
        assert _is_symbol_codepoint(0x2192)  # rightwards arrow
        assert _is_symbol_codepoint(0x2190)  # leftwards arrow

    def test_greek(self):
        assert _is_symbol_codepoint(0x0394)  # Delta
        assert _is_symbol_codepoint(0x03B1)  # alpha

    def test_math(self):
        assert _is_symbol_codepoint(0x2200)  # for all
        assert _is_symbol_codepoint(0x222B)  # integral

    def test_latin_not_symbol(self):
        assert not _is_symbol_codepoint(0x0041)  # A
        assert not _is_symbol_codepoint(0x006F)  # o


# ---------------------------------------------------------------------------
# Integration tests with synthetic PDFs
# ---------------------------------------------------------------------------

class TestBrokenToUnicode:
    def test_detects_category_mismatch(self, broken_tounicode_pdf):
        reports = analyze_fonts(broken_tounicode_pdf)
        assert len(reports) == 1
        assert reports[0].unreliable

        categories = [w.category for w in reports[0].warnings]
        assert "category_mismatch" in categories

    def test_mismatch_details(self, broken_tounicode_pdf):
        reports = analyze_fonts(broken_tounicode_pdf)
        mismatch_warnings = [
            w for w in reports[0].warnings if w.category == "category_mismatch"
        ]
        assert len(mismatch_warnings) == 1
        mismatches = mismatch_warnings[0].details["mismatches"]
        source_codes = {m["source_code"] for m in mismatches}
        assert "0x03B1" in source_codes  # alpha -> a
        assert "0x2192" in source_codes  # arrow -> o


class TestDuplicateMapping:
    def test_detects_duplicates(self, duplicate_tounicode_pdf):
        reports = analyze_fonts(duplicate_tounicode_pdf)
        assert len(reports) == 1
        assert reports[0].unreliable

        categories = [w.category for w in reports[0].warnings]
        assert "duplicate_mapping" in categories

    def test_duplicate_details(self, duplicate_tounicode_pdf):
        reports = analyze_fonts(duplicate_tounicode_pdf)
        dup_warnings = [
            w for w in reports[0].warnings if w.category == "duplicate_mapping"
        ]
        assert len(dup_warnings) == 1
        duplicates = dup_warnings[0].details["duplicates"]
        assert "U+006F" in duplicates
        assert len(duplicates["U+006F"]) == 3


class TestCleanPDF:
    def test_no_warnings(self, clean_pdf):
        reports = analyze_fonts(clean_pdf)
        assert len(reports) == 1
        assert not reports[0].unreliable
        assert len(reports[0].warnings) == 0


class TestAPIAcceptance:
    def test_accepts_path_string(self, clean_pdf):
        reports = analyze_fonts(clean_pdf)
        assert isinstance(reports, list)
        assert all(isinstance(r, PageFontReport) for r in reports)

    def test_accepts_document_object(self, clean_pdf):
        doc = pymupdf.open(clean_pdf)
        reports = analyze_fonts(doc)
        assert isinstance(reports, list)
        doc.close()

    def test_page_selection(self, clean_pdf):
        reports = analyze_fonts(clean_pdf, pages=[0])
        assert len(reports) == 1
        assert reports[0].page == 0

    def test_empty_pages(self, clean_pdf):
        reports = analyze_fonts(clean_pdf, pages=[])
        assert len(reports) == 0
