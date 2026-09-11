"""tests/test_table_grid_repair_real_fixtures.py -- validates the
grid-gap-repair code against three real table fixtures (WP-3-1,
WP-3-2-head, WP-3-2-tail) extracted from a real Slovak public-procurement
budget document, the case this algorithm's design was validated against.

Requires the real PyMuPDF Layout model (pymupdf.layout), which needs a
GPU-backed environment to run -- skips cleanly everywhere else.
"""
import pathlib

import pytest

pymupdf = pytest.importorskip("pymupdf")
pytest.importorskip("pymupdf.layout")
import pymupdf.layout  # noqa: F401  (wires up page.get_layout)

from pymupdf4llm.helpers.table_grid_repair import repair_grid_gaps

REPRO_PDF = pathlib.Path(__file__).parent / "mr_repro_table_grid_gap_repair_FROM_REAL_DOCUMENT.pdf"
pytestmark = pytest.mark.skipif(not REPRO_PDF.exists(), reason="real repro PDF not present in this environment")


def _get_table_blocks(page):
    textpage = page.get_textpage(clip=pymupdf.INFINITE_RECT())
    return [b for b in textpage.extractRAWDICT()["blocks"] if b["type"] == 0]


def _find_table(page, bbox_hint, tol=2.0):
    # Other test modules mutate the process-global `pymupdf._get_layout`
    # singleton via `pymupdf4llm.use_layout(False)` and never restore it
    # (e.g. test_bullet_o_marker.py's autouse `_reset_layout_mode` fixture,
    # and test_label_value_line_splice.py's module-level call). Once that
    # happens, `pymupdf.Page.get_layout()`'s `if _get_layout:` guard is
    # false, so `page.layout_information` is silently never (re)assigned
    # and stays at its class default of None -- causing
    # "TypeError: 'NoneType' object is not iterable" below, only when
    # this file runs after one of those modules in the same pytest
    # session. `activate()` is idempotent (no-ops if already active), so
    # reactivate defensively right before use instead of relying on the
    # module-level `import pymupdf.layout` above having the last word.
    pymupdf.layout.activate()
    page.get_layout(return_raw=True)
    for b in page.layout_information:
        if b["class_name"] != "table" or not b["table_grid"]:
            continue
        gb = b["group_bbox"]
        if all(abs(gb[i] - bbox_hint[i]) < tol for i in range(4)):
            return b
    raise RuntimeError(f"no table matching {bbox_hint} on page {page.number}")


def _repaired_counts(doc, page_idx, bbox_hint):
    page = doc[page_idx]
    tab = _find_table(page, bbox_hint)
    x0, y0, x1, y1 = tab["group_bbox"]
    grid = tab["table_grid"]
    h_lines = [y0] + [h + y0 for h in grid.h_lines] + [y1]
    v_lines = [x0] + [v + x0 for v in grid.v_lines] + [x1]
    table_blocks = _get_table_blocks(page)
    new_h, new_v = repair_grid_gaps(table_blocks, h_lines, v_lines)
    return len(new_h) - 1, len(new_v) - 1


@pytest.fixture(scope="module")
def doc():
    return pymupdf.open(str(REPRO_PDF))


def test_wp_3_1_repairs_to_14_rows_8_cols(doc):
    # Was 15 rows before the single-band-bisection guard (see
    # table_grid_repair.py's module docstring): the header's "Jednotková
    # cena bez DPH" / "(EUR)" wrap corroborated a row split that bisected
    # the neighboring "Kategória výdavkov" header cell's own single,
    # unwrapped line, corrupting its text. The guard keeps the header as
    # one row (14 total) and the header text intact.
    rows, cols = _repaired_counts(doc, 0, (52.08, 92.29, 715.94, 272.58))
    assert rows == 14
    assert cols == 8


def test_wp_3_2_head_repairs_to_6_rows(doc):
    # Was 7 rows before the single-band-bisection guard -- same header
    # shape and same fix as test_wp_3_1_repairs_to_14_rows_8_cols above.
    rows, _cols = _repaired_counts(doc, 0, (52.1, 285.9, 779.5, 486.3))
    assert rows == 6


def test_wp_3_2_tail_repairs_to_8_rows_8_cols(doc):
    rows, cols = _repaired_counts(doc, 1, (52.0, 65.3, 778.1, 411.4))
    assert rows == 8
    assert cols == 8  # known out-of-scope gap vs. doc's expected 9 -- see plan Background, limitation 3
