import numpy as np
import pymupdf
from pymupdf4llm.helpers.utils import WHITE_CHARS, analyze_page, bbox_is_empty
from pymupdf4llm.helpers.image_quality import analyze_image

FLAGS = (
    0
    | pymupdf.TEXT_COLLECT_STYLES
    | pymupdf.TEXT_COLLECT_VECTORS
    | pymupdf.TEXT_PRESERVE_IMAGES
    | pymupdf.TEXT_ACCURATE_BBOXES
    # | pymupdf.TEXT_MEDIABOX_CLIP
)
REPLACEMENT_CHARACTER = chr(0xFFFD)


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


def repair_blocks(input_blocks, page, dpi=300):
    """Repair text blocks with missing glyphs using Tesseract OCR.

    TODO: Support non-linear / nested TextPage block structure.
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
                new_text = get_span_ocr(page, span["bbox"], dpi=dpi)[:span_text_len]
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
        dpi: DPI used for rasterization *if* we decide to OCR
        covered: area to consider for text presence
    Returns:
        The full-page transformation matrix, the full-page pixmap and a
        boolean indicating whether the page is photo-like (True) or
        text-like (False).
    """
    if covered is None:
        covered = page.rect
    covered = covered.irect
    # Make a gray pixmap of the covered area
    pix_covered = page.get_pixmap(colorspace=pymupdf.csGRAY, clip=covered)
    # convert to numpy array
    gray = np.frombuffer(pix_covered.samples, dtype=np.uint8).reshape(
        pix_covered.height,
        pix_covered.width,
    )

    # Run photo checks
    scores = analyze_image(gray)
    score = scores["score"]
    if score >= 3:
        pix = None
        matrix = pymupdf.Identity
        photo = True
    else:
        pix = page.get_pixmap(dpi=dpi)
        matrix = pymupdf.Rect(pix.irect).torect(page.rect)
        photo = False

    return matrix, pix, photo


def should_ocr_page(
    page,
    dpi=150,
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
    }
    page_rect = page.rect
    page_area = abs(page_rect)  # size of the full page

    # Analyze the page
    analysis = analyze_page(page, blocks=blocks)

    # return if page is completely blank
    if bbox_is_empty(analysis["covered"]):
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
        # Rasterize and check for photo versus text-heaviness
        matrix, pix, photo = get_page_image(page, dpi=dpi, covered=analysis["covered"])

        if photo:
            # this seems to be a non-text picture page
            decision["should_ocr"] = False
            decision["pixmap"] = None
        else:
            decision["should_ocr"] = True
            decision["transform"] = matrix
            decision["pixmap"] = pix

    return decision
