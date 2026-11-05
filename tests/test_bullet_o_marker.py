"""Regression test for the 'o' bullet-marker over-detection bug.

BULLETS (helpers/utils.py) includes ASCII 'o', kept for genuine
outline-style "o" sub-bullets. But 'o' is also a real one-letter word or
article in several languages (e.g. Slovak/Czech/Polish/Russian "o" =
"about", Portuguese "o" = "the", Spanish "o" = "or"), so a lone 'o'
starting a wrapped line -- with nothing but an ordinary single space after
it -- was previously always treated as a bullet marker and rewritten to
"- ", silently discarding the word, regardless of which language's text
it appeared in.

The fix (`_has_wide_marker_gap` in helpers/utils.py) only trusts a
letter-shaped BULLETS marker (currently just 'o' -- see
`AMBIGUOUS_BULLETS`, derived from Unicode General Category rather than a
hand-picked list of specific characters/languages) as a bullet when the
gap after it is geometrically wide, as in a real hanging-indent/tab-stop
list layout -- not when it's followed by one ordinary inter-word space.
This is a purely geometric check with no language or content assumptions.
"""

import unicodedata

import pymupdf
import pytest

import pymupdf4llm
from pymupdf4llm.helpers.utils import AMBIGUOUS_BULLETS, BULLETS


def test_ambiguous_bullets_is_derived_from_unicode_letter_category():
    """AMBIGUOUS_BULLETS must track BULLETS automatically via Unicode
    General Category, not a hand-picked list of specific characters --
    otherwise a future BULLETS addition (e.g. a Cyrillic/Greek lookalike,
    or another single-letter outline marker) could silently reintroduce
    this bug for markers nobody thought to special-case."""
    expected = frozenset(c for c in BULLETS if unicodedata.category(c).startswith("L"))
    assert AMBIGUOUS_BULLETS == expected
    assert "o" in AMBIGUOUS_BULLETS
    # Sanity check: every non-letter marker (dashes, asterisks, geometric
    # shapes, private-use glyphs, ...) must NOT be treated as ambiguous.
    assert "-" not in AMBIGUOUS_BULLETS
    assert "*" not in AMBIGUOUS_BULLETS
    assert "•" not in AMBIGUOUS_BULLETS


@pytest.fixture(autouse=True)
def _reset_layout_mode():
    # Some other test modules (e.g. test_137.py) call use_layout(True) and
    # can leave that global toggle set for the rest of the pytest session
    # if they fail before resetting it; be explicit so these tests' outcome
    # doesn't depend on run order.
    pymupdf4llm.use_layout(False)


_DEJAVU_SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _insert_text_with_diacritics(page, point, text, fontsize=11):
    """insert_text() with the base14 Helvetica default silently maps
    unsupported glyphs (Slovak/Czech diacritics included) to a placeholder
    dot, which would make this test pass or fail for the wrong reason.
    Embed a Unicode-capable font so the PDF's actual text layer matches the
    intended characters."""
    font = pymupdf.Font(fontfile=_DEJAVU_SANS)
    page.insert_font(fontname="F0", fontbuffer=font.buffer)
    page.insert_text(point, text, fontsize=fontsize, fontname="F0")


def _make_prose_with_lone_o_pdf(path):
    """A normal paragraph where a line wraps starting with the word 'o'
    followed by a single ordinary space -- must NOT be treated as a bullet."""
    doc = pymupdf.open()
    page = doc.new_page()
    _insert_text_with_diacritics(
        page, (72, 100), "Predávajúci sa zaväzuje podať návrh na vykonanie zápisu"
    )
    _insert_text_with_diacritics(
        page, (72, 115), "o predaji časti podniku v obchodnom registri."
    )
    doc.save(path)
    doc.close()


def _make_wide_gap_o_bullet_pdf(path):
    """A genuine hanging-indent list using 'o' as a marker, with a wide
    tab-stop-style gap before the item text -- must still be detected as
    a bullet and converted to '- '."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Top-level item.", fontsize=11)
    page.insert_text((90, 120), "o                       First sub-item using a tab-stop gap.", fontsize=11)
    page.insert_text((90, 140), "o                       Second sub-item using a tab-stop gap.", fontsize=11)
    doc.save(path)
    doc.close()


def test_lone_o_word_is_not_treated_as_bullet(tmp_path):
    pdf_path = str(tmp_path / "prose_o.pdf")
    _make_prose_with_lone_o_pdf(pdf_path)
    md = pymupdf4llm.to_markdown(pdf_path)
    assert "o predaji" in md, f"word 'o' was dropped/corrupted:\n{md!r}"
    assert " - predaji" not in md


def test_wide_gap_o_marker_is_still_a_bullet(tmp_path):
    pdf_path = str(tmp_path / "o_bullets.pdf")
    _make_wide_gap_o_bullet_pdf(pdf_path)
    md = pymupdf4llm.to_markdown(pdf_path)
    assert "First sub-item using a tab-stop gap." in md
    assert "Second sub-item using a tab-stop gap." in md
    assert "- " in md, f"genuine 'o' bullet list was not converted to '- ':\n{md!r}"


def test_symbol_bullets_are_unaffected(tmp_path):
    """Sanity check: the fix must not touch detection for unambiguous
    symbol bullets, which never call the new geometric check."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "• First bulleted item.", fontsize=11)
    page.insert_text((72, 120), "• Second bulleted item.", fontsize=11)
    pdf_path = str(tmp_path / "symbol_bullets.pdf")
    doc.save(pdf_path)
    doc.close()

    md = pymupdf4llm.to_markdown(pdf_path)
    assert "First bulleted item." in md
    assert "Second bulleted item." in md
