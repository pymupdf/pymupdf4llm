import pymupdf

white_spaces = set([chr(i) for i in range(33)]) | {0xA0, 0x2002, 0x2003, 0x2009, 0x202F}


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

        # ignore vectors that significantly overlap layout bboxes
        if any(abs(r & bb) > min(r_low_limit, abs(bb) * 0.1) for bb in all_bboxes):
            continue
        # ignore vectors that are mostly covered by images
        if any(abs(r & i) > r_hi_limit for i in images):
            continue
        paths.append({"rect": r})

    # make vector clusters, select only sufficiently large ones
    vectors = page.cluster_drawings(drawings=paths, x_tolerance=20, y_tolerance=20)
    vectors = [v for v in vectors if v.width > 30 and v.height > 30]

    # resolve mutual containment of images and vectors
    imgs = sorted(images + vectors, key=lambda r: abs(r), reverse=True)

    filtered_imgs = []
    for r in imgs:
        if not any(r in fr for fr in filtered_imgs):
            filtered_imgs.append(r)

    for r in filtered_imgs:
        # add picture orphans that do not significantly overlap layout boxes
        if not any(abs(r & bbox) > 0.1 * min(abs(r), abs(bbox)) for bbox in all_bboxes):
            page.layout_information.append(list(r) + ["picture"])
            all_bboxes.append(r)
    return


"""
Determine reading order of layout boxes on a document page.

Layout boxes are defined as classified bounding boxes, with class info as
provided by pymupdf_layout. Each box is a tuple (x0, y0, x1, y1, "class").

The main function is "find_reading_order()".
"""


def cluster_stripes(boxes, joined_boxes, vectors, vertical_gap=12):
    """
    Divide page into horizontal stripes based on vertical gaps.

    Args:
        boxes (list): List of bounding boxes.
        vertical_gap (float): Minimum vertical gap to separate stripes.

    Returns:
        List of disjoint horizontal stripes. Each stripe is a list of boxes.
    """

    def is_multi_column_layout(boxes):
        """Check if the boxes have a clean multi-column layout.

        Used to early exit from stripe clustering.
        """
        sorted_boxes = sorted(boxes, key=lambda b: b[0])
        columns = []
        current_column = [sorted_boxes[0]]
        for box in sorted_boxes[1:]:
            prev_right = max([b[2] for b in current_column])
            if box[0] - prev_right > 3:
                columns.append(current_column)
                current_column = [box]
            else:
                current_column.append(box)
        columns.append(current_column)
        return len(columns) > 1

    def divider(y, box, vertical_gap):
        """Create a rectangle of box width and vertical_gap height below y."""
        r = pymupdf.Rect(box[0], y, box[2], y + vertical_gap)
        return r

    # Sort top to bottom
    sorted_boxes = sorted(boxes, key=lambda b: b[3])
    stripes = []

    # exit if no boxes
    if not sorted_boxes:
        return stripes

    # Exit if clean multi-column layout: treat full page as single stripe.
    if is_multi_column_layout(boxes):
        return [boxes]

    # y-borders of horizontal stripes
    y_values = {joined_boxes.y1}
    for box in sorted_boxes:
        # find empty horizontal dividers of minimum height 'vertical_gap'
        y = box[3]
        if y >= joined_boxes.y1:
            continue
        div = divider(y, joined_boxes, vertical_gap)
        if not any(div.intersects(pymupdf.Rect(b[:4])) for b in boxes):
            # look for next bbox below the divider
            y0 = min(b[1] for b in sorted_boxes if b[1] >= div.y1)
            div.y1 = y0  # divider has this bottom now
            inter_count = 0  # counts intersections with vectors

            # if divider is fully contained in more than one vector's stripe
            # we don't consider it.
            for vr in vectors:
                if div.intersects(vr) and vr.y0 <= div.y0 and div.y1 <= vr.y1:
                    inter_count += 1
            if inter_count <= 1:
                y_values.add(div.y1)
    y_values = sorted(y_values)
    current_stripe = []
    for y in y_values:
        while sorted_boxes and sorted_boxes[0][3] <= y:
            current_stripe.append(sorted_boxes.pop(0))
        if current_stripe:
            stripes.append(current_stripe)
            current_stripe = []
    return stripes


def cluster_columns_in_stripe(stripe):
    """
    Within a stripe, group boxes into columns based on horizontal proximity.

    We use a small horizontal gap threshold to decide when a new column starts.

    Args:
        stripe (list): List of boxes we look at here.

    Returns:
        list: List of columns, each column is a list of boxes.
    """
    HORIZONTAL_GAP = 1  # allowable gap to start a new column
    # Sort left to right
    sorted_boxes = sorted(stripe, key=lambda b: b[0])
    columns = []
    current_column = [sorted_boxes[0]]

    for box in sorted_boxes[1:]:
        prev_right = max([b[2] for b in current_column])
        if box[0] - prev_right > HORIZONTAL_GAP:
            columns.append(sorted(current_column, key=lambda b: b[1]))
            current_column = [box]
        else:
            current_column.append(box)

    columns.append(sorted(current_column, key=lambda b: b[1]))
    return columns


def compute_reading_order(boxes, joined_boxes, vectors, vertical_gap=12):
    """
    Compute reading order of boxes delivered by PyMuPDF-Layout.

    Args:
        boxes (list): List of bounding boxes.
        vertical_gap (float): Minimum vertical gap to separate stripes.

    Returns:
        list: List of boxes in reading order.
    """
    stripes = cluster_stripes(
        boxes,
        joined_boxes,
        vectors,
        vertical_gap=vertical_gap,
    )
    ordered = []
    for stripe in stripes:
        columns = cluster_columns_in_stripe(stripe)
        for col in columns:
            ordered.extend(col)
    return ordered


def find_reading_order(page_rect, blocks, boxes, vertical_gap: float = 12) -> list:
    """Given page layout information, return the boxes in reading order.

    Args:
        boxes: List of classified bounding boxes with class info as defined
               by pymupdf_layout: (x0, y0, x1, y1, "class").
        vertical_gap: Minimum vertical gap to separate stripes. The default
                      value of 36 works well for most documents. It roughly
                      corresponds to 2 -3 text line heights

    Returns:
        List of boxes in reading order.
    """

    # compute adequate vertical_gap based on the height the page rectangle
    this_vertical_gap = vertical_gap * page_rect.height / 800

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

    # Handle edge case: pages with no body content (only headers/footers or empty)
    # This can occur with scanned images, blank pages, or pages with only headers/footers
    if not body_boxes:
        # Return only headers and footers sorted by position
        final = (
            sorted(page_headers, key=lambda r: (r[1], r[0]))
            + sorted(page_footers, key=lambda r: (r[1], r[0]))
        )
        return final

    # compute joined boxes of body
    joined_boxes = pymupdf.Rect(
        min(b[0] for b in body_boxes),
        min(b[1] for b in body_boxes),
        max(b[2] for b in body_boxes),
        max(b[3] for b in body_boxes),
    )

    # extract vectors contained in the TextPage
    min_bbox_height = min(b[3] - b[1] for b in body_boxes)
    vectors = [
        pymupdf.Rect(b["bbox"])
        for b in blocks
        if b["bbox"][3] - b["bbox"][1] >= min_bbox_height and b["bbox"] in joined_boxes
    ]
    # bring body into reading order
    ordered = compute_reading_order(
        body_boxes,
        joined_boxes,
        vectors,
        vertical_gap=this_vertical_gap,
    )
    # Final full boxes list. We do simple sorts for non-body boxes.
    final = (
        sorted(page_headers, key=lambda r: (r[1], r[0]))
        + ordered
        + sorted(page_footers, key=lambda r: (r[1], r[0]))
    )
    return final


def simplify_vectors(vectors):
    """Join vectors that are horizontally adjacent and vertically aligned."""
    Y_TOLERANCE = 1  # allowable top / bottom  difference
    new_vectors = []
    if not vectors:
        return new_vectors
    new_vectors = [vectors[0]]
    for v in vectors[1:]:
        last_v = new_vectors[-1]
        if (
            1
            and abs(v["bbox"][1] - last_v["bbox"][1]) < Y_TOLERANCE
            and abs(v["bbox"][3] - last_v["bbox"][3]) < Y_TOLERANCE
            and v["bbox"][0] <= last_v["bbox"][2] + 1
        ):
            # merge horizontally
            new_bbox = [
                min(v["bbox"][0], last_v["bbox"][0]),
                min(v["bbox"][1], last_v["bbox"][1]),
                max(v["bbox"][2], last_v["bbox"][2]),
                max(v["bbox"][3], last_v["bbox"][3]),
            ]
            last_v["bbox"] = new_bbox
        else:
            new_vectors.append(v)
    return new_vectors


def find_virtual_lines(page, table_bbox, words, vectors, link_rects):
    """Return virtual lines for a given table bbox.

    This utility looks for:
    * horizontal non-stroke vectors and uses their top and bottom edges
      as virtual lines. Should work for tables with alternating row colors.
    * horizontal thin lines and uses their left x-coordinate as column
      borders.
    """

    def make_vertical(table_bbox, line_bbox, word_boxes):
        # default top and bottom point of vertical line
        top = line_bbox.tl - (2, 0)
        bottom = pymupdf.Point(top.x, table_bbox.y1)

        # check if this cuts through any word boxes below and adjust bottom y
        my_wboxes = sorted(
            [
                wr
                for wr in word_boxes
                if wr.y0 >= top.y and wr.y1 <= bottom.y and wr.x0 < top.x < wr.x1
            ],
            key=lambda r: r.y1,
        )
        if my_wboxes:  # if so, adjust bottom y
            bottom.y = my_wboxes[0].y0

        # same check above
        my_wboxes = sorted(
            [
                wr
                for wr in word_boxes
                if wr.y0 >= table_bbox.y0 and wr.y1 <= top.y and wr.x0 < top.x < wr.x1
            ],
            key=lambda r: r.y1,
        )
        if my_wboxes:  # if so, adjust top y
            top.y = my_wboxes[-1].y1
        else:  # else we can start at top of table
            top.y = table_bbox.y0

        # extender = [((table_bbox.x0, top.y), (table_bbox.x1, top.y)), (top, bottom)]
        extender = [(top, bottom)]
        return extender

    word_boxes = sorted(
        [
            pymupdf.Rect(w[:4])
            for w in words
            if (w[3] - w[1]) > 5 and table_bbox.contains(w[:4])
        ],
        key=lambda r: r.y1,
    )

    all_lines = []
    all_boxes = []
    for v in vectors:
        vbbox = pymupdf.Rect(v["bbox"]).normalize()
        vbbox += (0, -0.5, 0, 0.5)  # expand vertically a bit
        vbbox &= table_bbox
        if vbbox.is_empty:
            continue
        if not v["stroked"] and vbbox.height >= 5 and vbbox.width > 20:
            all_lines.append((vbbox.tl, vbbox.tr))
            all_lines.append((vbbox.bl, vbbox.br))
            continue
        if (
            vbbox.width > 20
            and vbbox.height <= 3
            and not any(vbbox.intersects(lr) for lr in link_rects)
        ):  # horizontal line
            lines = make_vertical(table_bbox, vbbox, word_boxes)
            for line in lines:
                all_lines.append(line)

    return all_lines, all_boxes


def complete_table_structure(page):
    """Add virtual lines for "table" layout bboxes

    Iterate through all "table" layout boxes on the page's layout_information
    and return virtual lines and boxes that can help detect table structures.

    Returns:
        lists of virtual lines and boxes for the page's TableFinder.
    """
    all_lines = []
    all_boxes = []
    textpage = page.get_textpage(
        flags=pymupdf.TEXT_ACCURATE_BBOXES
        | pymupdf.TEXT_COLLECT_VECTORS
        | pymupdf.TEXT_COLLECT_STYLES
    )
    words = page.get_text("words", textpage=textpage)
    vectors = sorted(
        [b for b in textpage.extractDICT()["blocks"] if b["type"] == 3 and b["isrect"]],
        key=lambda v: (v["bbox"][3], v["bbox"][0]),
    )
    vectors = simplify_vectors(vectors)
    link_rects = [l["from"] for l in page.get_links()]
    for b in page.layout_information:
        if b[-1] != "table":
            continue
        table_bbox = pymupdf.Rect(b[:4])
        all_boxes.append(table_bbox)
        lines, boxes = find_virtual_lines(
            page,
            table_bbox,
            words,
            vectors,
            link_rects,
        )
        all_lines.extend(lines)
        all_boxes.extend(boxes)

    return all_lines, all_boxes


def extract_cells(textpage, cell, markdown=False):
    """Extract text from a rect-like 'cell' as plain or MD styled text.

    This function should ultimately be used to extract text from a table cell.
    Markdown output will only work correctly if extraction flag bit
    TEXT_COLLECT_STYLES is set.

    Args:
        textpage: A PyMuPDF TextPage object. Must have been created with
            TEXTFLAGS_TEXT | TEXT_COLLECT_STYLES.
        cell: A tuple (x0, y0, x1, y1) defining the cell's bbox.
        markdown: If True, return text formatted for Markdown.

    Returns:
        A string with the text extracted from the cell.
    """

    def outside_cell(bbox, cell):
        return (
            0
            or bbox[0] >= cell[2]
            or bbox[2] <= cell[0]
            or bbox[1] >= cell[3]
            or bbox[3] <= cell[1]
        )

    text = ""
    for block in textpage.extractRAWDICT()["blocks"]:
        if block["type"] != 0:
            continue
        if outside_cell(block["bbox"], cell):
            continue
        for line in block["lines"]:
            if outside_cell(line["bbox"], cell):
                continue
            if text:  # must be a new line in the cell
                text += "<br>" if markdown else "\n"

            # strikeout detection only works with horizontal text
            horizontal = line["dir"] == (0, 1) or line["dir"] == (1, 0)

            for span in line["spans"]:
                if outside_cell(span["bbox"], cell):
                    continue
                # only include chars with more than 50% bbox overlap
                span_text = ""
                for char in span["chars"]:
                    this_char = char["c"]
                    bbox = pymupdf.Rect(char["bbox"])
                    if abs(bbox & cell) > 0.5 * abs(bbox):
                        span_text += this_char
                    elif this_char in white_spaces:
                        span_text += " "

                if not span_text:
                    continue  # skip empty span

                if not markdown:  # no MD styling
                    text += span_text
                    continue

                prefix = ""
                suffix = ""
                if horizontal and span["char_flags"] & pymupdf.mupdf.FZ_STEXT_STRIKEOUT:
                    prefix += "~~"
                    suffix = "~~" + suffix
                if span["char_flags"] & pymupdf.mupdf.FZ_STEXT_BOLD:
                    prefix += "**"
                    suffix = "**" + suffix
                if span["flags"] & pymupdf.TEXT_FONT_ITALIC:
                    prefix += "_"
                    suffix = "_" + suffix
                if span["flags"] & pymupdf.TEXT_FONT_MONOSPACED:
                    prefix += "`"
                    suffix = "`" + suffix

                if len(span["chars"]) > 2:
                    span_text = span_text.rstrip()

                # if span continues previous styling: extend cell text
                if (ls := len(suffix)) and text.endswith(suffix):
                    text = text[:-ls] + span_text + suffix
                else:  # append the span with new styling
                    if not span_text.strip():
                        text += " "
                    else:
                        text += prefix + span_text + suffix
    text = text.replace("$<br>", "$ ").replace(" $ <br>", "$ ")
    return text.strip()


def table_to_markdown(textpage, table_item, markdown=True):
    output = ""
    table = table_item.table
    row_count = table["row_count"]
    col_count = table["col_count"]
    cell_boxes = table["cells"]
    # make empty cell text list
    cells = [[None for i in range(col_count)] for j in range(row_count)]

    # fill None cells with extracted text
    # for rows, copy content from left to right
    for j in range(row_count):
        for i in range(col_count - 1):
            if cells[j][i + 1] is None:
                cells[j][i + 1] = cells[j][i]

    # for columns, copy top to bottom
    for i in range(col_count):
        for j in range(row_count - 1):
            if cells[j + 1][i] is None:
                cells[j + 1][i] = cells[j][i]

    for i, row in enumerate(cell_boxes):
        for j, cell in enumerate(row):
            if cell is not None:
                cells[i][j] = extract_cells(
                    textpage, cell_boxes[i][j], markdown=markdown
                )
    for i, name in enumerate(cells[0]):
        if name is None:
            if i > 0:
                cells[0][i] = cells[0][i - 1]
            else:
                cells[0][i] = ""

    header = "|" + "|".join(cells[0]) + "|\n"
    output += header
    # insert GitHub header line separator
    output += "|" + "|".join("---" for i in range(col_count)) + "|\n"

    # skip first row in details if header is part of the table
    j = 1  # if self.header.external else 1

    # iterate over detail rows
    for row in cells[j:]:
        line = "|"
        for i, cell in enumerate(row):
            # replace None cells with empty string
            # use HTML line break tag
            if cell is None:
                cell = ""
            line += cell + "|"
        line += "\n"
        output += line
    return output + "\n"


def table_extract(textpage, table_item):
    table = table_item.table
    row_count = table["row_count"]
    col_count = table["col_count"]
    cell_boxes = table["cells"]
    # make empty cell text list
    cells = [[None for i in range(col_count)] for j in range(row_count)]

    for i, row in enumerate(cell_boxes):
        for j, cell in enumerate(row):
            if cell is not None:
                cells[i][j] = extract_cells(textpage, cell_boxes[i][j], markdown=False)

    return cells
