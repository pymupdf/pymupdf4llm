"""Test that cluster_stripes handles dividers with no boxes below them.

Regression test for ValueError: min() iterable argument is empty
in cluster_stripes() — related to #320, #329.

Uses real box geometry captured from a Parade PS8468E datasheet page
that reliably triggers the bug.
"""

import os
import sys
import json
import pymupdf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pymupdf4llm.helpers.utils import cluster_stripes  # noqa: E402


def test_cluster_stripes_no_crash_on_bottom_divider():
    """cluster_stripes must not crash when no boxes exist below a divider."""
    # Minimal reproduction: boxes where the lowest box's bottom + vertical_gap
    # exceeds all box tops, making min() receive an empty generator.
    # Captured from a real PDF (Parade PS8468E datasheet, page with table).
    boxes = [
        (50.0, 157.0, 529.0, 646.32),       # large picture
        (72.24, 93.58, 521.48, 102.76),      # text line
        (70.2, 65.76, 106.56, 93.24),        # small picture (logo)
        (70.08, 127.66, 171.44, 135.62),     # section header
        (50.0, 647.32, 529.0, 647.0),        # degenerate table-fallback
    ]
    joined_boxes = pymupdf.Rect(50.0, 65.76, 529.0, 647.0)
    vectors = []
    vertical_gap = 12.63

    # Should not raise ValueError
    result = cluster_stripes(boxes, joined_boxes, vectors, vertical_gap)
    assert result is not None


if __name__ == "__main__":
    test_cluster_stripes_no_crash_on_bottom_divider()
    print("PASS")
