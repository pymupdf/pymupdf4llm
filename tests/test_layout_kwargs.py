"""Regression tests for keyword arguments silently dropped on the layout path.

`_layout_to_markdown` / `_layout_to_json` / `_layout_to_text` take a trailing
``**kwargs`` marked "unsupported options for pymupdf layout" and never read it,
so an unsupported option was discarded without a word. The legacy path in
``helpers.pymupdf_rag`` already warns, so callers switching paths -- or running
on the default (layout) path -- were told nothing while believing the argument
had taken effect. ``ignore_alpha`` is the reported case.

These tests pin the warning on the layout path, keep it absent when there is
nothing to report, and pin the documented default of ``ignore_alpha`` to its
signature.
"""

import inspect
import os

import pytest

import pymupdf4llm
from pymupdf4llm.helpers import pymupdf_rag

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "test_370.pdf")

# A kwarg no path supports; used to exercise the legacy warning, where
# `ignore_alpha` is a real parameter and therefore silent.
UNKNOWN_KWARG = "definitely_not_a_real_option"

uses_layout = pytest.mark.skipif(
    not getattr(pymupdf4llm, "_use_layout", False),
    reason="pymupdf layout is not available, layout path not exercised",
)


# -- layout path ----------------------------------------------------

@uses_layout
@pytest.mark.parametrize("name", ["to_markdown", "to_json", "to_text"])
def test_layout_path_warns_on_unsupported_kwargs(name, capsys):
    # ignore_alpha is the reported case: accepted by the legacy path, dropped
    # silently by the layout path.
    getattr(pymupdf4llm, name)(PDF, pages=[0], **{UNKNOWN_KWARG: 1})
    out = capsys.readouterr().out
    assert UNKNOWN_KWARG in out
    assert "layout mode" in out


@uses_layout
@pytest.mark.parametrize("name", ["to_markdown", "to_json", "to_text"])
def test_layout_path_silent_when_all_kwargs_are_supported(name, capsys):
    # A supported call must not start printing warnings.
    getattr(pymupdf4llm, name)(PDF, pages=[0])
    assert "Warning" not in capsys.readouterr().out


@uses_layout
def test_ignore_alpha_warns_on_layout_path(capsys):
    pymupdf4llm.to_markdown(PDF, pages=[0], ignore_alpha=True)
    assert "ignore_alpha" in capsys.readouterr().out


# -- legacy path ----------------------------------------------------

def test_legacy_path_warns_on_unsupported_kwargs(monkeypatch, capsys):
    monkeypatch.setattr(pymupdf4llm, "_use_layout", False)
    pymupdf4llm.to_markdown(PDF, pages=[0], **{UNKNOWN_KWARG: 1})
    out = capsys.readouterr().out
    assert UNKNOWN_KWARG in out
    assert "legacy mode" in out


def test_legacy_path_accepts_ignore_alpha_without_warning(monkeypatch, capsys):
    # ignore_alpha is a real legacy parameter, so it must stay silent.
    monkeypatch.setattr(pymupdf4llm, "_use_layout", False)
    pymupdf4llm.to_markdown(PDF, pages=[0], ignore_alpha=True)
    assert "ignore_alpha" not in capsys.readouterr().out


# -- documented default ---------------------------------------------

def test_ignore_alpha_documented_default_matches_signature():
    default = inspect.signature(pymupdf_rag.to_markdown).parameters[
        "ignore_alpha"
    ].default
    assert default is False
    assert f"ignore_alpha: (bool, {default})" in pymupdf_rag.to_markdown.__doc__
