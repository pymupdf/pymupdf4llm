import os

import pytest

import pymupdf4llm

HERE = os.path.dirname(os.path.abspath(__file__))
PDFS = [os.path.join(HERE, "test_370.pdf"), os.path.join(HERE, "test_137.pdf")]


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"streaming": True},
        {"persistent": True},
    ],
    ids=["pool", "streaming", "persistent"],
)
def test_convert_batch_multiworker_without_out_dir(kwargs):
    """Process-pool strategies must work when no out_dir is given.

    Previously doc_dir=None was sent to workers as the string "None", so each
    worker tried to write None/run-log.txt and every document failed.
    """
    results = list(
        pymupdf4llm.convert_batch(PDFS, workers=2, backend="json", **kwargs)
    )
    assert len(results) == len(PDFS)
    for r in results:
        assert r.success, f"{r.input}: {r.error!r}"
    assert not os.path.exists("None")
