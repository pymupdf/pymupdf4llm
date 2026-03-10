"""
This callback function performs OCR on the given page using a combination
of RapidOCR and Tesseract OCR.

It is intended to be used by extraction methods of PyMuPDF4LLM, like
"to_markdown()".
Its purpose is to detect text regions using RapidOCR and recognize text
within those regions using Tesseract OCR.
Regions with legible text are ignored and left unchanged.

Recognized text is inserted in the page as standard extractable text
using MuPDF's universal Fallback Font.

This non-intrusive text augmentation approach ensures that the detected
text does not interfere with existing content like text, images or vector
graphics. This may also speed up OCR when standard text is present.

The combination of RapidOCR and Tesseract OCR combines the strengths of both
engines: RapidOCR provides fast and accurate text box detection, while
Tesseract OCR provides accurate text recognition within those boxes. This
approach produces much better overall results and is also faster than using
RapidOCR for both purposes.

The package "rapidocr_onnxruntime" must be installed and available.
It effectively uses the latest PaddlePaddle OCR models, which are optimized
for speed and accuracy. This is currently also the only working way to use
"PaddleOCR" on any platform.

Tesseract OCR must also be available for this function to work.
"""

import inspect
from rapidocr_onnxruntime import RapidOCR
import pymupdf
import numpy as np

TESSDATA = pymupdf.get_tessdata()
if TESSDATA is None:
    pymupdf.message(
        "Warning: Tesseract OCR is not available. No OCR text will be extracted."
    )

FONT = pymupdf.Font("cjk")  # this is the "Droid Sans Fallback" font
FONTNAME = "myfont"  # its reference name in the page
REPLACEMENT_UNICODE = chr(0xFFFD)  # Unicode Replacement Character


def ocr_text(span) -> bool:
    if not (span["char_flags"] & 32) and not (span["char_flags"] & 16):
        return True
    return False


ENGINE = RapidOCR()

# prepare for more advanced use of Tesseract by checking a function signature
sig = inspect.signature(pymupdf.Pixmap.pdfocr_tobytes)
if "options" in sig.parameters:
    USE_TESS_OPTIONS = True
else:
    USE_TESS_OPTIONS = False


def get_text(pixmap, irect, language="eng"):
    """Use Tesseract to extract text from a given bounding box of the pixmap.

    The irect is expected to contain one line only, so we use
    tessedit_pageseg_mode=7.
    """
    my_irect = irect + (-2, -2, 2, 2)
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


def exec_ocr(page, dpi=300, pixmap=None, language="eng", keep_ocr_text=False):
    """This callback function performs OCR on the given page.

    It uses RapidOCR for text region detection and Tesseract OCR for text
    recognition in each identified region (boundary box).

    If a Pixmap is provided, the DPI parameter is ignored. Otherwise, an RGB
    Pixmap is created from the page at the specified DPI.
    The DPI parameter is also used if extractable text is present.
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

    if TESSDATA is None:
        # cannot perform OCR without Tesseract, so just
        return

    """
    We ensure that legible extractable text is excluded from OCR. If present
    on page we make a temporary copy without such text and perform OCR
    on that copy.
    """
    text_blocks = page.get_text("dict", flags=pymupdf.TEXT_ACCURATE_BBOXES)["blocks"]
    # get bboxes with significant legible text on page
    spans = []
    fffd_spans = []
    for b in text_blocks:
        for l in b["lines"]:
            for s in l["spans"]:
                if ocr_text(s):
                    if keep_ocr_text:
                        spans.append(s["bbox"])
                    else:
                        fffd_spans.append(s["bbox"])
                    continue
                if not REPLACEMENT_UNICODE in s["text"]:
                    spans.append(s["bbox"])
                else:
                    fffd_spans.append(s["bbox"])

    if spans:
        temp_pdf = pymupdf.open()  # create a temporary PDF in memory
        # insert the page
        temp_pdf.insert_pdf(
            page.parent,
            from_page=page.number,
            to_page=page.number,
        )
        temp_page = temp_pdf[0]
        for sbbox in spans:
            # add redaction annotation for each text span
            temp_page.add_redact_annot(sbbox)

        # remove text
        temp_page.apply_redactions(
            images=pymupdf.PDF_REDACT_IMAGE_NONE,
            graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
            text=pymupdf.PDF_REDACT_TEXT_REMOVE,
        )
        # make pixmap from the page where text is removed
        pixmap = temp_page.get_pixmap(dpi=dpi)

    # make pixmap if not provided
    if pixmap is None:
        pixmap = page.get_pixmap(dpi=dpi)

    # make numpy array from pixmap
    img = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )

    if fffd_spans:
        # if there are spans with U+FFFD, we add redaction annotations for them
        # and apply redactions to the pixmap to remove them before OCR
        for sbbox in fffd_spans:
            page.add_redact_annot(sbbox)
        page.apply_redactions(
            images=pymupdf.PDF_REDACT_IMAGE_NONE,
            graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
            text=pymupdf.PDF_REDACT_TEXT_REMOVE,
        )
    # for converting box coordinates to page coordinates
    matrix = pymupdf.Rect(pixmap.irect).torect(page.rect)
    # t0 = time.perf_counter()
    # Execute the ENGINE's bbox Detector
    boxes, _ = ENGINE.text_detector(img)
    # t1 = time.perf_counter()
    # Execute Tesseract's text Recognizer
    # List of Tesseract text results
    tess_results = []
    for box in boxes:
        # top-left, top-right, bottom-right, bottom-left
        tl, tr, br, bl = box
        irect = pymupdf.IRect(tl[0], tl[1], br[0], br[1])
        text = get_text(pixmap, irect)  # execute Tesseract OCR on the line box
        tess_results.append((irect, text))
    if not tess_results:  # guard against no text found
        return
    # t2 = time.perf_counter()
    # print(f"RapidOCR detection time: {t1 - t0:.2f} seconds")
    # print(f"Tesseract OCR time: {t2 - t1:.2f} seconds")
    # insert the OCR font into the page
    page.insert_font(fontname=FONTNAME, fontbuffer=FONT.buffer)

    for irect, text in tess_results:
        # this is the line box
        rect = pymupdf.Rect(irect) * matrix

        # this matrix will adjust the rendered text width to fit text box
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
            # render_mode=0,
            morph=(rect.bl, mat),  # adjust width to fit the line box
        )
