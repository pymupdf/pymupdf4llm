from __future__ import annotations
"""Document-level HTML join used by :func:`reconstruct.to_html`."""
from typing import Any


def html_document(tables: list[dict[str, Any]]) -> str:
    """Concatenate per-table HTML fragments into one document string.

    Joins each entry's ``"html"`` (skipping empties) with a blank line."""
    return "\n\n".join(str(table.get("html") or "") for table in tables if table.get("html"))
