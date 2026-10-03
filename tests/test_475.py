import pymupdf

from pymupdf4llm.helpers import multi_column


def test_475_cache_key_survives_address_reuse():
    """`_in_bbox_using_cache` used to key its cache on id(bb). A freed Rect's
    address gets reused almost immediately in CPython, and column_boxes()
    drops and replaces Rect objects constantly (nblocks[j] = temp), so a
    later, unrelated Rect can inherit another rectangle's cached answer.
    This forces that exact collision deterministically.
    """
    container = pymupdf.Rect(0, 0, 10, 10)
    bboxes = [container]

    outside = pymupdf.Rect(500, 500, 600, 600)  # not inside `container`
    cache = {}
    # Stale entry a collision would leave behind: some earlier, unrelated
    # rect that WAS inside its container, at the address `outside` now reuses.
    cache[f"{id(outside)}_{id(bboxes)}"] = 1

    result = multi_column._in_bbox_using_cache(outside, bboxes, cache)

    assert result == 0


def test_475_column_boxes_is_stable_across_repeated_calls():
    """End-to-end: a page with enough short-lived block merges to plausibly
    trigger address reuse should still extract identically every time.
    """
    doc = pymupdf.open()
    page = doc.new_page()
    for i in range(40):
        page.insert_text((60 + (i % 4) * 5, 70 + i * 12), f"filler line {i} of body text", fontsize=9)
    page.insert_text((60, 640), "Heading One", fontsize=16)
    page.insert_text((60, 665), "Body text under heading one, several words long.", fontsize=10)
    page.insert_text((60, 720), "Heading Two", fontsize=16)
    page.insert_text((60, 745), "Body text under heading two, several words long too.", fontsize=10)
    data = doc.tobytes()

    results = set()
    for _ in range(20):
        d = pymupdf.open("pdf", data)
        rects = multi_column.column_boxes(d[0])
        results.add(tuple(tuple(round(v, 1) for v in r) for r in rects))
        d.close()

    assert len(results) == 1, f"column_boxes produced {len(results)} distinct results across 20 identical calls"
