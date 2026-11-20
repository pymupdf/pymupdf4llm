import pymupdf

from pymupdf4llm.helpers import multi_column


def _page_with_image_between_two_text_blocks():
    doc = pymupdf.open()
    page = doc.new_page()

    pix = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 50, 50), False)
    pix.set_rect(pix.irect, (200, 0, 0))
    img_bytes = pix.tobytes("png")

    page.insert_text((60, 100), "Block One heading text here", fontsize=14)
    page.insert_image(pymupdf.Rect(60, 130, 300, 300), stream=img_bytes)
    page.insert_text((60, 320), "Block Two heading text here", fontsize=14)

    data = doc.tobytes()
    return pymupdf.open("pdf", data)


def test_474_can_extend_does_not_join_across_an_image():
    """can_extend() only checked bboxlist and vert_bboxes, never img_bboxes,
    so a text block could be joined to another across an entire intervening
    image. Fixes #474.
    """
    doc = _page_with_image_between_two_text_blocks()
    page = doc[0]

    rects = multi_column.column_boxes(page)

    texts = [page.get_text(clip=r) for r in rects]
    one = [t for t in texts if "Block One" in t]
    two = [t for t in texts if "Block Two" in t]
    assert one and two, f"expected both blocks present, got: {texts}"
    assert one != two, "Block One and Block Two were joined into the same rect"


def test_474_no_image_text_false_keeps_old_behaviour():
    """no_image_text=False opts out of the image-avoidance check entirely,
    same as it already opts out of the image-text exclusion it's named for.
    """
    doc = _page_with_image_between_two_text_blocks()
    page = doc[0]

    rects = multi_column.column_boxes(page, no_image_text=False)

    texts = [page.get_text(clip=r) for r in rects]
    assert any("Block One" in t and "Block Two" in t for t in texts)
