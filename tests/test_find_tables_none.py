"""Regression tests for a non-table result from ``page.find_tables()``.

Issue #462: ``get_page_output`` in ``helpers/pymupdf_rag`` read ``.tables`` off
the result of ``page.find_tables(...)`` with no guard, so a ``None`` result
aborted ``to_markdown`` with::

    AttributeError: 'NoneType' object has no attribute 'tables'

``helpers/table_html/reconstruct.py`` already guards both of its call sites with
``(getattr(tf, "tables", None) or [])``. These tests pin the same behaviour in
the rag helper: a page whose tables cannot be found still converts, with its
tables skipped, rather than failing the whole document.
"""

import os

import pymupdf
import pytest

import pymupdf4llm

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "test_370.pdf")


@pytest.fixture
def legacy_mode(monkeypatch):
    # get_page_output's table handling only runs on the rag path.
    monkeypatch.setattr(pymupdf4llm, "_use_layout", False)


def test_find_tables_returning_none_does_not_abort(legacy_mode, monkeypatch):
    monkeypatch.setattr(pymupdf.Page, "find_tables", lambda *a, **k: None)
    text = pymupdf4llm.to_markdown(PDF, pages=[0])
    assert text


def test_find_tables_result_without_tables_attribute_does_not_abort(
        legacy_mode, monkeypatch):
    monkeypatch.setattr(pymupdf.Page, "find_tables", lambda *a, **k: object())
    text = pymupdf4llm.to_markdown(PDF, pages=[0])
    assert text


def test_real_tables_are_still_extracted(legacy_mode):
    # The guard must skip only a missing result, never a real one.
    text = pymupdf4llm.to_markdown(PDF, pages=[0])
    assert text
