"""
This callback function performs OCR on the given page using RapidOCR.

It is intended to be used by extraction methods of PyMuPDF4LLM, like
"to_markdown()".
Its purpose is to detect AND recognize text on page regions where there
is no legible text. Regions with legible text are ignore and left unchanged.

Recognized text is inserted in the page as standard extractable text
using MuPDF's universal Fallback Font.

This non-intrusive text augmentation approach ensures that the detected
text does not interfere with existing content like text, images or vector
graphics. This may also speed up OCR when standard text is present.

The package "rapidocr_onnxruntime" must be installed and available.
It effectively uses the latest PaddlePaddle OCR models, which are optimized
for speed and accuracy. This is currently also the only working way to use
"PaddleOCR" on any platform.
"""

from rapidocr_onnxruntime import RapidOCR
import pymupdf
import numpy as np
from .get_culled_pixmap import get_pixmap
from .check_legal_text import contains_unsafe

FONT = pymupdf.Font("cjk")  # this is the "Droid Sans Fallback" font
FONTNAME = "myfont"  # its reference name in the page
REPLACEMENT_UNICODE = chr(0xFFFD)  # Unicode Replacement Character
STROKED_TEXT = pymupdf.mupdf.FZ_STEXT_STROKED
FILLED_TEXT = pymupdf.mupdf.FZ_STEXT_FILLED


def ocr_text(span) -> bool:
    if (span["char_flags"] & STROKED_TEXT) or (span["char_flags"] & FILLED_TEXT):
        return False
    return True


ENGINE = RapidOCR()

# pass any keyword arguments to RapidOCR when calling exec_ocr()
KWARGS = {}


def exec_ocr(page, dpi=300, pixmap=None, language="eng", keep_ocr_text=False):
    """This callback function performs OCR on the given page.

    The pixmap parameter is deprecated and ignored.
    The keep_ocr parameter is ignored. If this plugin is called,
    existing OCR text will ALWAYS be removed and replaced with new OCR text.
    """

    def adjust_width(text, fontsize, rect):
        """Compute matrix to adjust text width.

        We must ensure that inserted text has the width of the rectangle.
        The computed matrix will do this scaling.
        """
        tl = FONT.text_length(text, fontsize)
        if tl > 0:
            mat = pymupdf.Matrix(rect.width / tl, 1)
        else:
            mat = pymupdf.Matrix(1, 1)
        return mat

    """
    We ensure that legible extractable text is excluded from OCR. We render
    the page without "good" text and perform OCR on the rest.
    """
    displaylist = page.get_displaylist()
    stextpage = displaylist.get_textpage(flags=pymupdf.TEXT_ACCURATE_BBOXES)
    textpage = pymupdf.TextPage(stextpage)
    text_blocks = textpage.extractDICT()["blocks"]

    # get bboxes with multiple text categories on page
    spans = []  # spans with legible text
    fffd_spans = []  # spans containing U+FFFD characters.
    ocr_spans = []  # spans with previously OCRed text
    unsafe_span_count = 0
    for b in text_blocks:
        for l in b["lines"]:
            for s in l["spans"]:
                if ocr_text(s):
                    ocr_spans.append(s["bbox"])
                elif REPLACEMENT_UNICODE in s["text"]:
                    fffd_spans.append(s["bbox"])
                elif contains_unsafe(s["text"]):
                    unsafe_span_count += 1
                else:
                    # for removal good text regions
                    spans.append(s["bbox"])

    if ocr_spans and keep_ocr_text:
        # If there are already OCR spans and the user wants to keep them, we skip OCR.
        # This is because we cannot distinguish between "good" text and "bad" OCR text.
        return

    if unsafe_span_count > 0:
        # If there are spans with unsafe characters, we skip OCR.
        # This is because the OCR engine may not handle them correctly.
        return

    # make a Pixmap without "good" text
    pix = get_pixmap(displaylist, dpi=dpi, rects=spans)

    # Converts ENGINE box coordinates to page coordinates
    matrix = pymupdf.Rect(pix.irect).torect(page.rect)

    # make numpy array from pixmap
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height,
        pix.width,
        3,
    )

    # Execute RapidOCR
    result, _ = ENGINE(img)
    if not result:
        return

    if any(contains_unsafe(text) for _, text, _ in result):
        # If any recognized text contains unsafe characters, we skip OCR
        # entirely. This is because we have no way yet to make correct
        # insertions of such text into the page.
        return

    # Remove all OCR and illegible spans from the page.
    # The OCR engine will restore them according to its best ability.
    redaction_rects = fffd_spans + ocr_spans
    if redaction_rects:
        for sbbox in redaction_rects:
            page.add_redact_annot(sbbox)
        page.apply_redactions(
            images=pymupdf.PDF_REDACT_IMAGE_NONE,
            graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
            text=pymupdf.PDF_REDACT_TEXT_REMOVE,
        )

    # insert the font into the page if not already present
    page.insert_font(fontname=FONTNAME, fontbuffer=FONT.buffer)

    # Insert recognized text
    for box, text, conf in result:
        # PaddleOCR box: 4 point-likes (tl, tr, br, bl)
        rect = (
            pymupdf.Rect(
                min(p[0] for p in box),
                min(p[1] for p in box),
                max(p[0] for p in box),
                max(p[1] for p in box),
            )
            * matrix
        )
        if not text.strip():
            continue

        fontsize = rect.height
        # Text width scaling matrix ensures text width = box width
        mat = adjust_width(text, fontsize, rect)

        page.insert_text(
            rect.bl + (0, -0.2 * fontsize),
            text,
            fontsize=fontsize,
            fontname=FONTNAME,
            morph=(rect.bl, mat),
        )
