from __future__ import annotations
"""HTML table reconstruction helpers for pymupdf4llm.

Builds the per-page table payload consumed by the markdown/JSON renderers
(:func:`page_html_tables`) and a standalone page-to-HTML convenience
(:func:`to_html`), both driven by ``page.find_tables(use_layout=True,
union=True, refine=True)``.
"""
import pymupdf
from .core import html_document


def _placement_grid_matrices(placements) -> tuple[int, int, list, list]:
    """Derive ``(row_count, col_count, cells, extract)`` from a placement grid.

    ``placements`` is a ``Table.placements`` row-major grid of ``SpanCell``
    colspan/rowspan cells. ``row_count`` is the ``<tr>`` count; ``col_count`` the
    column extent after resolving spans (the HTML column count). ``cells`` and
    ``extract`` are the ``row_count x col_count`` post-span bbox and plain-text
    matrices, ``None`` where a span covers a slot or the grid has a gap."""
    row_count = len(placements)
    # Column extent after resolving colspan/rowspan == the HTML <td>/<th> column
    # count the matrices below are shaped to.
    occupied = set()
    col_count = 0
    for row_idx, row in enumerate(placements):
        col_idx = 0
        for cell in row:
            while (row_idx, col_idx) in occupied:
                col_idx += 1
            for dr in range(cell.rowspan):
                for dc in range(cell.colspan):
                    occupied.add((row_idx + dr, col_idx + dc))
            col_idx += cell.colspan
            col_count = max(col_count, col_idx)
    # Expand the ragged post-span grid into row_count x col_count bbox and text
    # matrices, ``None`` where a span covers a slot (or a grid gap).
    bbox_grid = [[None] * col_count for _ in range(row_count)]
    text_grid = [[None] * col_count for _ in range(row_count)]
    covered = set()
    for row_idx, row in enumerate(placements):
        col_idx = 0
        for cell in row:
            while (row_idx, col_idx) in covered:
                col_idx += 1
            if col_idx >= col_count:
                break
            bbox_grid[row_idx][col_idx] = list(cell.bbox) if cell.bbox is not None else None
            text_grid[row_idx][col_idx] = cell.text
            for dr in range(cell.rowspan):
                for dc in range(cell.colspan):
                    if dr or dc:
                        covered.add((row_idx + dr, col_idx + dc))
            col_idx += cell.colspan
    return row_count, col_count, bbox_grid, text_grid


def to_html(pdf, page_index=0):
    """Reconstruct the tables on one PDF page and return them as an HTML string.

    pdf        : PDF file path (str/Path) or an already-open pymupdf.Document.
    page_index : 0-based page number.
    returns    : concatenated <table>...</table> HTML (empty string if no tables).

    When ``pdf`` is an already-open Document, the target page is derotated in
    place via ``page.remove_rotation()`` (mutates the caller-owned page).
    """
    owns_doc = not isinstance(pdf, pymupdf.Document)
    doc = pymupdf.open(pdf) if owns_doc else pdf
    try:
        page = doc[page_index]
        page.remove_rotation()
        tf = page.find_tables(use_layout=True, union=True, refine=True)
        tables = [tab.to_html() for tab in (getattr(tf, "tables", None) or [])]
    finally:
        if owns_doc:
            doc.close()
    return html_document([{"html": h} for h in tables])


def page_html_tables(page: pymupdf.Page) -> list[tuple[pymupdf.Rect, str, int, int, list, list]]:
    """Reconstruct one already-open page's tables as payload tuples.

    Returns one ``(bbox, html, rows, cols, cells, extract)`` tuple per table, in
    reading order, for the markdown/JSON renderers to drive table emission,
    reading order and body-text exclusion:

    * ``bbox``    -- ``tab.bbox`` (a grid-ref table keeps its reported layout box);
    * ``html``    -- ``tab.to_html()``;
    * ``rows``/``cols``/``cells``/``extract`` -- the reconstructed grid the
      ``html`` shows (see :func:`_placement_grid_matrices`): ``extract`` is the
      per-cell plain-text matrix (``None`` for span-covered slots / grid gaps)
      and ``cells`` the matching post-span bbox matrix, both ``rows x cols``.

    The caller must remove page rotation before calling. Not thread-safe on a
    shared Page: core caches word/vector extraction as attributes on the given
    ``page``, so concurrent calls must each use their own ``pymupdf.Page``.
    """
    tf = page.find_tables(use_layout=True, union=True, refine=True)
    result = []
    for tab in (getattr(tf, "tables", None) or []):
        row_count, col_count, cells, extract = _placement_grid_matrices(tab.placements)
        result.append(
            (
                pymupdf.Rect(tab.bbox),
                tab.to_html(),
                row_count,
                col_count,
                cells,
                extract,
            )
        )
    return result
