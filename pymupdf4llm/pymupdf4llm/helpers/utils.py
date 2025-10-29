import pymupdf


def table_cleaner(page, blocks, tbbox):
    """Clean the table bbox 'tbbox'.

    'blocks' is the TextPage.extractDict()["blocks"] list.

    This function must be used AFTER clean_pictures() so we know that tbbox
    is complete in terms of includable vectors.

    We check whether the table bbox contains non-rect ("tilted") vectors
    and determine which part of tbbox they cover. If this is too large, we
    re-classify tbbox as a picture.
    Else we check whether the tilted vectors only cover some upper part of the
    result. In that case we separate the top part as a picture and keep
    the remining area as a table.
    """
    bbox = pymupdf.Rect(tbbox[:4])

    # All vectors inside tbbox. Checking for the top-left corner is enough.
    all_vectors = [
        (pymupdf.IRect(b["bbox"]), b["isrect"])
        for b in blocks
        if b["type"] == 3 and b["bbox"][:2] in bbox
    ]
    tilt_vectors = [v for v in all_vectors if not v[1]]
    # Early exit if no tilted vectors
    if not tilt_vectors:
        return None, None

    y0 = min([b[0].y0 for b in tilt_vectors])
    y1 = max([b[0].y1 for b in tilt_vectors])
    x0 = min([b[0].x0 for b in tilt_vectors])
    x1 = max([b[0].x1 for b in tilt_vectors])

    # Rectangle containing all non-rectangle vectors inside the table bbox
    tilted = pymupdf.Rect(x0, y0, x1, y1)

    # if it covers most of the table bbox, we convert to picture
    if tilted.width >= bbox.width * 0.8 and tilted.height >= bbox.height * 0.8:
        return tbbox[:4] + ["picture"], None

    # Extract text spans. Needed for completing the potential picture area.
    span_rects = [
        s["bbox"]
        for b in blocks
        if b["type"] == 0
        for l in b["lines"]
        for s in l["spans"]
        if s["bbox"] in bbox
    ]

    # Check if non-rect vectors cover some acceptable upper part of tbbox.
    if (
        1
        and tilted.y1 - bbox.y0 <= bbox.height * 0.3  # 30% of tbbox height
        and tilted.width >= bbox.width * 0.7  # at least 80% of tbbox width
    ):
        tilted.y1 += 2  # add some buffer at the bottom

        # include any text that is part of the picture area
        for r in span_rects:
            if tilted.intersects(r):
                tilted |= r

        picture_box = [bbox.x0, bbox.y0, bbox.x1, tilted.y1, "picture"]
        table_box = [bbox.x0, tilted.y1 + 1, bbox.x1, bbox.y1, "table"]
        return picture_box, table_box
    return None, None


def clean_tables(page, blocks):
    for i in range(len(page.layout_information)):
        if page.layout_information[i][4] != "table":
            continue
        # re-classify some corner cases as "text"
        # the layout bbox as a Rect
        bbox = pymupdf.Rect(page.layout_information[i][:4])

        # lines in this bbox
        lines = [
            l for b in blocks if b["type"] == 0 for l in b["lines"] if l["bbox"] in bbox
        ]
        y_vals0 = sorted(set(round(l["bbox"][3]) for l in lines))
        y_vals = [y_vals0[0]]
        for y in y_vals0[1:]:
            if y - y_vals[-1] > 3:
                y_vals.append(y)
        if len(y_vals) < 2:  # too few distinct line bottoms
            # too few text lines to be a table
            page.layout_information[i][4] = "text"
            continue
        # our table minimum dimension, rows x cols, is 2 x 2
        mx_same_baseline = 1
        for y in y_vals:
            count = len([l for l in lines if abs(y - l["bbox"][3]) <= 3])
            if count > mx_same_baseline:
                mx_same_baseline = count
                break
        if mx_same_baseline < 2:
            # too few text columns to be a table
            page.layout_information[i][4] = "text"
            continue
        rc1, rc2 = table_cleaner(page, blocks, page.layout_information[i])
        if rc1:
            if not rc2:
                page.layout_information[i] = rc1
            else:
                page.layout_information[i] = rc2
                page.layout_information.insert(i, rc1)
                i += 1
    return


def clean_pictures(page, blocks):
    """Extend picture / formula / table bboxes.

    Join layout boxes with intersecting text, image, vectors.

    'blocks' is the TextPage.extractDict()["blocks"] list.
    """
    # all layout boxes
    all_bboxes = [pymupdf.Rect(b[:4]) for b in page.layout_information]

    for i in range(len(all_bboxes)):
        if page.layout_information[i][4] not in ("picture", "formula", "table"):
            # no eligible layout box
            continue

        # get its Rect object
        bbox = pymupdf.Rect(page.layout_information[i][:4])
        for b in blocks:
            if b["type"] not in (0, 1, 3):
                continue
            block_bbox = pymupdf.IRect(b["bbox"])
            if b["type"] == 3 and block_bbox.is_empty:
                block_bbox += (-1, -1, 1, 1)
            if bbox.intersects(block_bbox) and not any(
                bb.intersects(block_bbox) for j, bb in enumerate(all_bboxes) if j != i
            ):
                bbox |= block_bbox
        page.layout_information[i] = list(bbox) + [page.layout_information[i][4]]


def add_image_orphans(page, blocks):
    """Add orphan images as layout boxes of class 'picture'.

    'blocks' is the TextPage.extractDict()["blocks"] list.
    """
    # all layout boxes
    all_bboxes = [pymupdf.Rect(b[:4]) for b in page.layout_information]
    area_limit = abs(page.rect) * 0.9
    images = []
    for img in page.get_image_info():
        r = page.rect & img["bbox"]
        if r.is_empty or abs(r) >= area_limit:
            continue
        images.append(r)

    paths = []
    for b in blocks:
        if b["type"] != 3:
            continue
        r = page.rect & b["bbox"]
        if abs(r) >= area_limit:
            continue
        if r.width < 3 and r.height < 3:
            continue
        r_low_limit = 0.1 * abs(r)
        r_hi_limit = 0.8 * abs(r)
        if any(abs(r & bb) > r_low_limit for bb in all_bboxes):
            continue
        if any(abs(r & i) > r_hi_limit for i in images):
            continue
        paths.append({"rect": r})

    vectors = page.cluster_drawings(drawings=paths, x_tolerance=20, y_tolerance=20)
    vectors = [v for v in vectors if v.width > 30 and v.height > 30]

    imgs = sorted(images + vectors, key=lambda r: abs(r), reverse=True)

    filtered_imgs = []
    for r in imgs:
        if not any(r in fr for fr in filtered_imgs):
            filtered_imgs.append(r)

    for r in filtered_imgs:
        if not any(abs(r & bbox) > 0.1 * abs(r) for bbox in all_bboxes):
            page.layout_information.append(list(r) + ["picture"])
            all_bboxes.append(r)
    return


"""
Determine reading order of layout boxes on a document page.

Layout boxes are defined as classified bounding boxes, with class info as
provided by pymupdf_layout. Each box is a tuple (x0, y0, x1, y1, "class").

The main function is "find_reading_order()".
"""


def cluster_stripes(boxes, vertical_gap: float = 12):
    """
    Divide page into horizontal stripes based on vertical gaps.

    Args:
        boxes (list): List of bounding boxes, each defined as (x0, y0, x1, y1).
        vertical_gap (float): Minimum vertical gap to separate stripes.

    Returns:
        List of disjoint horizontal stripes. Each stripe is a list of boxes.
    """
    # Sort top to bottom
    sorted_boxes = sorted(boxes, key=lambda b: b[1])
    stripes = []
    if not sorted_boxes:
        return stripes
    current_stripe = [sorted_boxes[0]]

    for box in sorted_boxes[1:]:
        prev_bottom = max(b[3] for b in current_stripe)
        if box[1] - prev_bottom > vertical_gap:
            stripes.append(current_stripe)
            current_stripe = [box]
        else:
            current_stripe.append(box)

    stripes.append(current_stripe)
    return stripes


def cluster_columns_in_stripe(stripe: list):
    """
    Within a stripe, group boxes into columns based on horizontal proximity.

    Args:
        stripe (list): List of boxes within a stripe.

    Returns:
        list: List of columns, each column is a list of boxes.
    """
    # Sort left to right
    sorted_boxes = sorted(stripe, key=lambda b: b[0])
    columns = []
    current_column = [sorted_boxes[0]]

    for box in sorted_boxes[1:]:
        prev_right = max([b[2] for b in current_column])
        if box[0] - prev_right >= -1:
            columns.append(sorted(current_column, key=lambda b: b[3]))
            current_column = [box]
        else:
            current_column.append(box)

    columns.append(sorted(current_column, key=lambda b: b[3]))
    return columns


def compute_reading_order(boxes, vertical_gap: float = 12):
    """
    Compute reading order of boxes delivered by PyMuPDF-Layout.

    Args:
        boxes (list): List of bounding boxes.
        vertical_gap (float): Minimum vertical gap to separate stripes.

    Returns:
        list: List of boxes in reading order.
    """
    # compute adequate vertical_gap based height of union of bboxes
    temp = pymupdf.EMPTY_RECT()
    for b in boxes:
        temp |= pymupdf.Rect(b[:4])
    this_vertical_gap = vertical_gap * temp.height / 800
    stripes = cluster_stripes(boxes, vertical_gap=this_vertical_gap)
    ordered = []
    for stripe in stripes:
        columns = cluster_columns_in_stripe(stripe)
        for col in columns:
            ordered.extend(col)
    return ordered


def find_reading_order(boxes, vertical_gap: float = 12) -> list:
    """Given page layout information, return the boxes in reading order.

    Args:
        boxes: List of classified bounding boxes with class info as defined
               by pymupdf_layout: (x0, y0, x1, y1, "class").
        vertical_gap: Minimum vertical gap to separate stripes. The default
                      value of 12 works well for most documents.

    Returns:
        List of boxes in reading order.
    """

    def is_contained(inner, outer) -> bool:
        """Check if inner box is fully contained within outer box."""
        return (
            1
            and outer[0] <= inner[0]
            and outer[1] <= inner[1]
            and outer[2] >= inner[2]
            and outer[3] >= inner[3]
            and inner != outer
        )

    def filter_contained(boxes) -> list:
        """Remove boxes that are fully contained within another box."""
        # Sort boxes by descending area
        sorted_boxes = sorted(
            boxes, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True
        )
        result = []
        for r in sorted_boxes:
            if not any(is_contained(r, other) for other in result):
                result.append(r)
        return result

    """
    We expect being passed raw 'layout_information' as provided by
    pymupdf_layout. We separate page headers and footers from the
    body, bring body boxes into reading order and concatenate the final list.
    """
    filtered = filter_contained(boxes)  # remove nested boxes first
    page_headers = []  # for page headers
    page_footers = []  # for page footers
    body_boxes = []  # for main body boxes

    # separate boxes by type
    for box in filtered:
        x0, y0, x1, y1, bclass = box
        if bclass == "page-header":
            page_headers.append(box)
        elif bclass == "page-footer":
            page_footers.append(box)
        else:
            body_boxes.append(box)

    # bring body into reading order
    ordered = compute_reading_order(body_boxes, vertical_gap=vertical_gap)

    # Final full boxes list. We do simple sorts for non-body boxes.
    final = (
        sorted(page_headers, key=lambda r: (r[1], r[0]))
        + ordered
        + sorted(page_footers, key=lambda r: (r[1], r[0]))
    )
    return final
