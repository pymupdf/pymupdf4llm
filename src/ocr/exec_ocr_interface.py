import inspect

import numpy as np
import pymupdf

from .get_culled_pixmap import get_pixmap

TESSDATA = pymupdf.get_tessdata()
FONT = pymupdf.Font("cjk")  # this is the "Droid Sans Fallback" font
FONTNAME = "myfont"  # its reference name in the page
REPLACEMENT_UNICODE = chr(0xFFFD)  # Unicode Replacement Character
STROKED_TEXT = pymupdf.mupdf.FZ_STEXT_STROKED
FILLED_TEXT = pymupdf.mupdf.FZ_STEXT_FILLED


def ocr_text(span) -> bool:
    if (span["char_flags"] & STROKED_TEXT) or (span["char_flags"] & FILLED_TEXT):
        return False
    return True


def adjust_width(text, fontsize, rect):
    """Compute matrix to adjust text width.

    We must ensure that inserted text has the width of the rectangle.
    The computed matrix will do this scaling.
    """
    tl = FONT.text_length(text, fontsize)
    if tl > 0:
        return pymupdf.Matrix(rect.width / tl, 1)
    return pymupdf.Matrix(1, 1)


# prepare for more advanced use of Tesseract by checking a function signature
sig = inspect.signature(pymupdf.Pixmap.pdfocr_tobytes)
USE_TESS_OPTIONS = "options" in sig.parameters


def get_text(pixmap, irect, language="eng"):
    """Use Tesseract to extract text from a given bounding box of the pixmap.

    The irect is expected to contain one line only, so we use
    tessedit_pageseg_mode=7.
    """
    if irect.is_empty:
        return ""
    my_irect = irect
    # these options ensure a much improved Tesseract behavior
    options = "tessedit_pageseg_mode=7,preserve_interword_spaces=1"
    this_pix = pymupdf.Pixmap(pymupdf.csRGB, my_irect)
    this_pix.copy(pixmap, my_irect)
    if USE_TESS_OPTIONS:
        # use options if pymupdf already provides this
        data = this_pix.pdfocr_tobytes(
            language=language,
            tessdata=TESSDATA,
            options=options,
        )
    else:
        data = this_pix.pdfocr_tobytes(
            language=language,
            tessdata=TESSDATA,
        )
    doc = pymupdf.open("pdf", data)
    page = doc[0]
    return page.get_text().strip()


def exec_ocr_detection(page, det_only, dpi=150, language="eng", keep_ocr_text=False):
    """This callback function performs OCR on the given page.

    It uses the "detection-only" function of some OCR engine. This function is
    expected to identify text regions and return bounding boxes.
    The actual text recognition is performed by Tesseract OCR, which is expected to be
    installed and available in the system path.
    """

    if TESSDATA is None:
        raise RuntimeError("Tesseract unavailable.")

    if not callable(det_only):
        raise RuntimeError("OCR engine unavailable - no detection callable provided.")

    """
    We ensure that legible extractable text is excluded from OCR. We render
    the page without "good" text and perform OCR on the rest.
    """
    displaylist = page.get_displaylist()
    stextpage = displaylist.get_textpage(flags=pymupdf.TEXT_ACCURATE_BBOXES)
    textpage = pymupdf.TextPage(stextpage)
    text_blocks = textpage.extractDICT()["blocks"]

    # get bboxes with multiple text categories on page
    spans = []  # bboxes with good text
    fffd_spans = []  # boxes with illegible text
    ocr_spans = []  # boxes with old OCR text
    for b in text_blocks:
        for l in b["lines"]:
            for s in l["spans"]:
                if ocr_text(s):
                    ocr_spans.append(s["bbox"])
                elif REPLACEMENT_UNICODE in s["text"]:
                    fffd_spans.append(s["bbox"])
                else:
                    # for removal of good text regions
                    spans.append(s["bbox"])
    if ocr_spans and keep_ocr_text:
        # If there are already OCR spans and the user wants to keep them, we skip OCR.
        # This is because we cannot distinguish between "good" text and "bad" OCR text.
        return

    # make a Pixmap without "good" text
    pix, empty = get_pixmap(displaylist, dpi=dpi, rects=spans, empty_threshold=250)
    if empty:
        return  # nothing to OCR, the page is empty after removing good text

    # For converting ENGINE box coordinates to page coordinates
    matrix = pymupdf.Rect(pix.irect).torect(page.rect)

    # make numpy array from pixmap
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height,
        pix.width,
        3,
    )
    """
    This calls the detection-only function of the OCR engine.
    """
    result = det_only(img)
    if not result:
        return
    if len(result[0]) != 2:
        raise RuntimeError(
            "Detection-only function must return a list of (box, score) tuples."
        )

    # Remove all OCR spans and spans containing a U+FFFD.
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

    # Execute Tesseract's text Recognizer
    # List of Tesseract text results
    tess_results = []

    for box, score in result:
        irect = pymupdf.IRect(
            min(p[0] for p in box),
            min(p[1] for p in box),
            max(p[0] for p in box),
            max(p[1] for p in box),
        )
        text = get_text(pix, irect)
        tess_results.append((irect, text))

    if not tess_results:
        return

    # insert the OCR font into the page
    page.insert_font(fontname=FONTNAME, fontbuffer=FONT.buffer)

    for irect, text in tess_results:
        # this is the line box
        rect = pymupdf.Rect(irect) * matrix

        # this matrix will ensure text width = rect width
        mat = adjust_width(text, rect.height, rect)

        # Insert one line of text. Insertion point is the bottom-left box
        # corner adjusted slightly upwards to account for the descender. Note
        # that the original is unknown, so descender -0.2 is best guess only.
        # Also true for the font size: guessed to be rectangle height.
        # NOTE: Guesses could be improved by checking actual text content for
        # the presence of descenders and uppercase letters.
        page.insert_text(
            rect.bl + (0, -0.2 * rect.height),  # insertion point
            text,  # text to render
            fontsize=rect.height,  # take this as font size
            fontname=FONTNAME,  # fallback font
            morph=(rect.bl, mat),  # adjust width to fit the line box
        )


def exec_ocr_full(page, full_ocr, dpi=150, language=None, keep_ocr_text=False):
    """OCR callback with flexible OCR engine backend."""

    if not callable(full_ocr):
        raise RuntimeError("OCR engine is unavailable - no callable provided.")

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
    for b in text_blocks:
        for l in b["lines"]:
            for s in l["spans"]:
                if ocr_text(s):
                    ocr_spans.append(s["bbox"])
                elif REPLACEMENT_UNICODE in s["text"]:
                    fffd_spans.append(s["bbox"])
                else:
                    # for removal good text regions
                    spans.append(s["bbox"])

    if ocr_spans and keep_ocr_text:
        # If there are already OCR spans and the user wants to keep them, we skip OCR.
        # This is because we cannot distinguish between "good" text and "bad" OCR text.
        return

    # make a Pixmap without "good" text
    pix, empty = get_pixmap(displaylist, dpi=dpi, rects=spans, empty_threshold=250)
    if empty:
        return  # nothing to OCR, the page is empty after removing good text

    # Converts ENGINE box coordinates to page coordinates
    matrix = pymupdf.Rect(pix.irect).torect(page.rect)

    # make numpy array from pixmap
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height,
        pix.width,
        3,
    )

    """
    Call the OCR engine to provide full detection and recognition results.
    """
    result = full_ocr(img)
    if not result:
        return
    if len(result[0]) != 3:
        raise RuntimeError(
            "Full OCR function must return a list of (box, text, score) tuples."
        )
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
