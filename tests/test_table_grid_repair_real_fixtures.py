"""tests/test_table_grid_repair_real_fixtures.py -- grid-gap repair on three
real tables (WP-3-1, WP-3-2-head, WP-3-2-tail) from a Slovak public-funding
budget document, run through get_table_details() exactly as
parse_document() does (same textpage flags), with assertions on the
extracted cells rather than only on row/column counts.

Requires the PyMuPDF Layout model (pymupdf.layout); skips without it.
"""

import pathlib

import pytest

pymupdf = pytest.importorskip("pymupdf")
pytest.importorskip("pymupdf.layout")

from pymupdf4llm.helpers import document_layout
from pymupdf4llm.helpers.table_grid_repair import repair_grid_gaps

REPRO_PDF = (
    pathlib.Path(__file__).parent
    / "mr_repro_table_grid_gap_repair_FROM_REAL_DOCUMENT.pdf"
)
pytestmark = pytest.mark.skipif(
    not REPRO_PDF.exists(), reason="real repro PDF not present"
)

WP_3_1 = (0, (52.08, 92.29))
WP_3_2_HEAD = (0, (52.1, 285.9))
WP_3_2_TAIL = (1, (52.0, 65.3))


@pytest.fixture(scope="module")
def doc():
    with pymupdf.open(str(REPRO_PDF)) as d:
        yield d


def _table(doc, where):
    pno, (x0, y0) = where
    page = doc[pno]
    # Other test modules switch the process-global layout hook off via
    # pymupdf4llm.use_layout(False); activate() is idempotent.
    pymupdf.layout.activate()  # imported by importorskip above
    page.get_layout(return_raw=True)
    tab_dict = next(
        b
        for b in page.layout_information
        if b["class_name"] == "table"
        and b["table_grid"]
        and abs(b["group_bbox"][0] - x0) < 2
        and abs(b["group_bbox"][1] - y0) < 2
    )
    textpage = page.get_textpage(
        flags=document_layout.FLAGS, clip=pymupdf.INFINITE_RECT()
    )
    table_blocks = [b for b in textpage.extractRAWDICT()["blocks"] if b["type"] == 0]
    return tab_dict, table_blocks


def _grid(tab_dict):
    x0, y0, x1, y1 = tab_dict["group_bbox"]
    grid = tab_dict["table_grid"]
    h = [y0] + [y + y0 for y in grid.h_lines] + [y1]
    v = [x0] + [x + x0 for x in grid.v_lines] + [x1]
    return h, v


def _assert_no_fused_values(det):
    for row in det.extract:
        for cell in row:
            assert "0,00\n0,00" not in (cell or ""), det.extract


def test_wp_3_1_fused_header_columns_and_empty_rows_are_split(doc):
    det = document_layout.get_table_details(*_table(doc, WP_3_1))
    assert det.col_count == 9
    header = next(r for r in det.extract if r[0] == "Názov výdavku")
    # Three headers the model's grid had fused into one column.
    assert header[5].startswith("Celkové oprávnené výdavky")
    assert header[6] == "Intenzita pomoci"
    assert header[7].startswith("Nárokované oprávnené")
    empty_rows = [r for r in det.extract if r[0] == "" and "0,00" in r]
    assert len(empty_rows) == 9
    for row in empty_rows:
        assert row == ["", "", "", "", "", "0,00", "", "0,00", ""]
    assert det.extract[-1][0] == "SPOLU - pracovný balík 3-1"
    _assert_no_fused_values(det)


def test_wp_3_2_head_is_left_unchanged(doc):
    tab_dict, table_blocks = _table(doc, WP_3_2_HEAD)
    h, v = _grid(tab_dict)
    assert repair_grid_gaps(table_blocks, h, v) == (h, v)


def test_wp_3_2_tail_fused_rows_are_split_and_wrapped_cells_kept(doc):
    det = document_layout.get_table_details(*_table(doc, WP_3_2_TAIL))
    assert det.row_count == 8
    assert det.col_count == 8
    assert [r[0] for r in det.extract] == [
        "PM/Compliance",
        "",
        "Zmluvný výskum",
        "",
        "HW vybavenie 1 – odpisy",
        "",
        det.extract[6][0],  # the multi-line "Paušálna sadzba ..." label
        "SPOLU - pracovný balík 3-2",
    ]
    for i in (1, 3, 5):
        assert det.extract[i][5] == "0,00" and det.extract[i][7] == "0,00"
    # The wrapped description paragraph stays whole, in its own row's cell.
    hw_note = det.extract[4][7]
    assert hw_note.startswith("170 400,00\nProjekt využíva existujúce")
    assert "zahrnuli prostredníctvom oprávnených odpisov" in hw_note
    # The three-line label is one cell, not three rows.
    label = det.extract[6][0]
    assert label.startswith("Paušálna sadzba - ostatné výdavky")
    assert label.rstrip().endswith("ak relevantné")
    assert det.extract[7][5] == "2 199 840,00"
    _assert_no_fused_values(det)
