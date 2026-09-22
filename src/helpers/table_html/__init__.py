"""HTML table reconstruction for pymupdf4llm.

Public entry points:

* `page_html_tables(page)` -- the per-page `(bbox, html, rows, cols, cells,
  extract)` payload consumed by `pymupdf_rag.to_markdown(..., table_output="html")`
  and `document_layout`;
* `to_html(pdf, page_index)` -- a standalone page -> `<table>` HTML convenience.
"""

from __future__ import annotations

from .reconstruct import to_html, page_html_tables

__all__ = ["to_html", "page_html_tables"]
