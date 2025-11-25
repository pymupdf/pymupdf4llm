import cv2
import numpy as np
import pymupdf  # PyMuPDF
from pymupdf4llm.helpers.utils import WHITE_CHARS, analyze_page

FLAGS = (
    0
    | pymupdf.TEXT_COLLECT_STYLES
    | pymupdf.TEXT_COLLECT_VECTORS
    | pymupdf.TEXT_PRESERVE_IMAGES
    | pymupdf.TEXT_ACCURATE_BBOXES
    # | pymupdf.TEXT_MEDIABOX_CLIP
)
REPLACEMENT_CHARACTER = chr(0xFFFD)
"""
--------------------------------------------------------------------------
Start of OpenCV example features for potential later use.
--------------------------------------------------------------------------
"""


# def detect_qr_codes(img):
#     """Currently not used by PyMuPDF4LLM."""
#     detector = cv2.QRCodeDetector()
#     data, points, _ = detector.detectAndDecode(img)

#     if points is not None and data:
#         pts = points[0].astype(int)
#         return {"data": data, "bbox": pts.tolist()}
#     return None


# def detect_barcodes(img):
#     """Currently not used by PyMuPDF4LLM."""
#     try:
#         from pyzbar.pyzbar import decode as barcode_decode
#     except ImportError:
#         raise ImportError("pyzbar is required for barcode detection")
#     gray = img
#     barcodes = barcode_decode(gray)
#     results = []

#     for barcode in barcodes:
#         results.append(
#             {
#                 "type": barcode.type,
#                 "data": barcode.data.decode("utf-8"),
#                 "bbox": [(p.x, p.y) for p in barcode.polygon],
#             }
#         )
#     return results


# def detect_lines(img, min_length=50, max_gap=10, matrix=pymupdf.Identity):
#     """Currently not used by PyMuPDF4LLM."""
#     gray = img
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#     pix_lines = cv2.HoughLinesP(
#         edges,
#         1,
#         np.pi / 180,
#         threshold=100,
#         minLineLength=min_length,
#         maxLineGap=max_gap,
#     )
#     lines = []
#     for np_linesr in pix_lines:
#         for r in np_linesr:
#             p0 = pymupdf.Point(r[0], r[1]) * matrix
#             p1 = pymupdf.Point(r[2], r[3]) * matrix
#             lines.append((p0, p1))
#     return lines  # array of (point1, point2)


# def detect_curves(img, matrix=pymupdf.Identity):
#     """Currently not used by PyMuPDF4LLM."""
#     gray = img
#     _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

#     curves = []
#     for cnt in contours:
#         if len(cnt) > 5:
#             ellipse = cv2.fitEllipse(cnt)
#             curves.append(ellipse)
#     return curves


# def detect_rectangles(img, min_area=1000, matrix=pymupdf.Identity):
#     """Currently not used by PyMuPDF4LLM."""
#     gray = img
#     _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     rectangles = []
#     for cnt in contours:
#         approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
#         if len(approx) == 4 and cv2.contourArea(cnt) > min_area:
#             r = pymupdf.Rect(approx) * matrix
#             rectangles.append(r)
#     return rectangles


"""
--------------------------------------------------------------------------
End of OpenCV example features
--------------------------------------------------------------------------
"""


def get_span_ocr(page, bbox, dpi=300):
    """Return OCR'd span text using Tesseract.

    Args:
        page: pymupdf Page
        bbox: pymupdf Rect or its sequence
        dpi: resolution for OCR image
    Returns:
        The OCR-ed text of the bbox.
    """
    # Step 1: Make a high-resolution image of the bbox.
    pix = page.get_pixmap(dpi=dpi, clip=bbox)
    ocrpdf = pymupdf.open("pdf", pix.pdfocr_tobytes())
    ocrpage = ocrpdf[0]
    text = ocrpage.get_text()
    text = text.replace("\n", " ").strip()  # get rid of line breaks
    return text


def repair_blocks(input_blocks, page):
    """Repair text blocks with missing glyphs using OCR.

    TODO: Support non-linear block structure.
    """
    repaired_blocks = []
    for block in input_blocks:
        if block["type"] != 0:  # accept non-text blocks as is
            repaired_blocks.append(block)
            continue

        for line in block["lines"]:
            spans = line["spans"]
            for span in spans:
                if "chars" in span:
                    span_text = "".join([c["c"] for c in span["chars"]])
                else:
                    span_text = span["text"]
                if not REPLACEMENT_CHARACTER in span_text:
                    continue
                span_text_len = len(span_text)
                new_text = get_span_ocr(page, span["bbox"])[:span_text_len]
                if "chars" in span:
                    # rebuild chars array
                    new_chars = []
                    for i in range(min(len(new_text), len(span["chars"]))):
                        cdict = span["chars"][i].copy()
                        cdict["c"] = new_text[i]
                        new_chars.append(cdict)
                    span["chars"] = new_chars
                else:
                    span["text"] = new_text
        repaired_blocks.append(block)
    return repaired_blocks


def get_page_image(page, dpi=150, covered=None):
    """Determine whether the page contains text worthwhile to OCR.

    Args:
        page: PyMuPDF Page object
        dpi: DPI used for rasterization
        covered: area to consider for text presence
    Returns:
        A tuple containing cv2 image of covered area, the full-page
        transformation matrix, and the full-page pixmap.
    """
    if covered is None:
        covered = page.rect
    covered = covered.irect
    pix = page.get_pixmap(dpi=dpi)
    matrix = pymupdf.Rect(pix.irect).torect(page.rect)

    # make a sub-pixmap of the covered area
    pix_covered = pymupdf.Pixmap(pymupdf.csRGB, covered)
    pix_covered.copy(pix, covered)  # copy over covered area
    # convert to numpy array
    img = np.frombuffer(pix_covered.samples, dtype=np.uint8).reshape(
        pix_covered.height, pix_covered.width, pix_covered.n
    )
    # cv2 needs the gray image version of this
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray, matrix, pix


def should_ocr_page(
    page,
    dpi=150,
    edge_thresh=0.02,
    vector_thresh=0.9,
    image_coverage_thresh=0.9,
    text_readability_thresh=0.9,
    blocks=None,
):
    """
    Decide whether a PyMuPDF page should be OCR'd.

    Parameters:
        page: PyMuPDF page object
        dpi: DPI used for rasterization
        edge_thresh: minimum edge density to suggest text presence
        vector_thresh: minimum number of vector paths to suggest glyph simulation
        image_coverage_thresh: fraction of page area covered by images to trigger OCR
        text_readability_thresh: fraction of readable characters to skip OCR
        blocks: output of page.get_text("dict") if already available

    Returns:
        dict with decision and diagnostic flags
    """
    decision = {
        "should_ocr": False,
        "has_ocr_text": False,
        "has_text": False,
        "readable_text": False,
        "image_covers_page": False,
        "has_vector_chars": False,
        "transform": pymupdf.Identity,
        "pixmap": None,
        "edge_density": 0.0,
    }
    page_rect = page.rect
    page_area = abs(page_rect)  # size of the full page

    # Analyze the page
    analysis = analyze_page(page, blocks=blocks)

    # return if page is completely blank
    if analysis["covered"].is_empty:
        decision["should_ocr"] = False
        return decision

    # return if page has been OCR'd already
    if analysis["ocr_spans"] > 0:
        decision["has_ocr_text"] = True
        decision["should_ocr"] = False
        return decision

    # preset OCR if very little text area exists
    if (
        1
        and analysis["txt_area"] < 0.05
        and analysis["chars_total"] < 200
        and analysis["txt_joins"] < 0.3
    ):
        # less than 5% text area in covered area
        if analysis["vec_area"] >= vector_thresh:
            decision["should_ocr"] = True
            decision["has_vector_chars"] = True
        if analysis["img_area"] >= image_coverage_thresh:
            decision["should_ocr"] = True
            decision["image_covers_page"] = True

    elif analysis["chars_total"] >= 200:
        decision["has_text"] = True
        readability = 1 - analysis["chars_bad"] / analysis["chars_total"]
        if readability >= text_readability_thresh:
            decision["readable_text"] = True
            decision["should_ocr"] = False
        else:
            decision["readable_text"] = False
            decision["should_ocr"] = True

    if decision["should_ocr"] is False:
        return decision

    if decision["readable_text"] is False and decision["has_text"] is True:
        return decision

    # We need OCR and do a final check for potential text presence
    assert decision["should_ocr"] is True

    if not decision["has_text"]:
        # Rasterize and analyze edge density
        img, matrix, pix = get_page_image(page, dpi=dpi, covered=analysis["covered"])

        # Analyze edge density
        edges = cv2.Canny(img, 100, 200)
        decision["edge_density"] = float(np.sum(edges > 0) / edges.size)
        if decision["edge_density"] <= edge_thresh:
            # this seems to be a non-text picture page
            decision["should_ocr"] = False
        else:
            decision["should_ocr"] = True
            decision["transform"] = matrix
            decision["pixmap"] = pix

    if decision["should_ocr"]:
        decision["transform"] = matrix
        decision["pixmap"] = pix
    return decision
