"""
This callable performs OCR on a given PyMuPDF page.
It is intended to be used by extraction methods of PyMuPDF4LLM, like
"to_markdown()".

It combines two OCR engines in an effort to achieve the best of both: text box
accuracy and text accuracy.
The DETECTION of text boxes is performed by RapidOCR, while text RECOGNITION
within each box is performed by Tesseract OCR.

Contrary to intuition, this combination is MUCH faster than RapidOCR alone
while also giving much better text recognition results.

Detected text is inserted into the page as the "OCR text layer" and other
page content is erased using Redaction Annotations.
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
FONT = pymupdf.Font("cjk")  # this is the "Droid Sans Fallback Regular" font
FONTNAME = "myfont"  # its reference name in the page

ENGINE = RapidOCR()

# prepare for more advanced use of Tesseract by checking a function signature
sig = inspect.signature(pymupdf.Pixmap.pdfocr_tobytes)
if "options" in sig.parameters:
    USE_TESS_OPTIONS = True
else:
    USE_TESS_OPTIONS = False


def get_text(pixmap, irect):
    """Use Tesseract to extract text from a given bounding box of the pixmap.

    The irect is expected to contain one line only, so we use
    tessedit_pageseg_mode=7.
    """
    # these options ensure a much improved Tesseract behavior
    options = "tessedit_pageseg_mode=7,preserve_interword_spaces=1"
    this_pix = pymupdf.Pixmap(pymupdf.csRGB, irect)
    this_pix.copy(pixmap, irect)

    if USE_TESS_OPTIONS:  # use options if pymupdf already provides this
        data = this_pix.pdfocr_tobytes(options=options)
    else:
        data = this_pix.pdfocr_tobytes()
    doc = pymupdf.open("pdf", data)
    page = doc[0]
    return page.get_text().strip()


def exec_ocr(page, dpi=300, pixmap=None):
    """This callback function performs OCR on the given page using RapidOCR.

    It is intended to be used by extraction methods of PyMuPDF4LLM, like
    "to_markdown()". The Layout plugin expects standard extractable text to be
    present on the page. So, this function inserts the detected text using
    a fallback font provided by MuPDF, "Droid Sans Fallback". This font
    supports a wide range of Unicode characters, including Chinese, and is
    included with MuPDF. The font is inserted into the page if not already
    present, and then used to render the detected text.

    If a Pixmap is provided, the DPI parameter is ignored. Otherwise, an RGB
    Pixmap is created from the page at the specified DPI.

    OCR will be performed requesting line and potentially word boxes. If word
    boxes are detected, each word will be rendered separately within its line.
    """

    def adjust_width(text, fontsize, rect):
        """Compute matrix to adjust the width of the rendered text."""
        tl = FONT.text_length(text, fontsize)
        mat = pymupdf.Matrix(rect.width / tl, 1)
        return mat

    if TESSDATA is None:
        return

    # make pixmap if not provided
    pix = pixmap
    if pix is None:
        pix = page.get_pixmap(dpi=dpi)

    # make numpy array from pixmap
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, pix.width, pix.n)

    # for converting box coordinates to page coordinates
    matrix = pymupdf.Rect(pix.irect).torect(page.rect)

    # Execute RapidOCR's text box detector
    boxes, _ = ENGINE.text_detector(img)

    # Blank out the page before inserting the OCR text layer. This provides
    # the layout plugin with a clean slate to work with.
    page.add_redact_annot(page.rect)  # add a redaction annotation to the whole page
    page.apply_redactions(
        images=pymupdf.PDF_REDACT_IMAGE_REMOVE,
        graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
        text=pymupdf.PDF_REDACT_TEXT_REMOVE,
    )

    # insert the OCR font into the page
    page.insert_font(fontname=FONTNAME, fontbuffer=FONT.buffer)

    for i, box in enumerate(boxes):
        tl, tr, br, bl = box  # top-left, top-right, bottom-right, bottom-left
        irect = pymupdf.IRect(tl[0], tl[1], br[0], br[1])
        text = get_text(pix, irect)  # execute Tesseract OCR on the line box
        if not text:  # guard against no text found
            continue
        # this is the line box
        rect = pymupdf.Rect(tl[0], tl[1], br[0], br[1]) * matrix

        # this matrix will adjust the rendered text width to fit text box
        mat = adjust_width(text, rect.height, rect)

        # Insert one line of text. Insertion point is the bottom-left box
        # corner adjusted slightly upwards to account for the descender. Note
        # that the original is unknown, so descender -0.2 is best guess only.
        # Similar is true for the font size which we take as the line box height.
        # NOTE: Guesses could be improved by checking actual text content for
        # the presence of descenders and uppercase letters.
        page.insert_text(
            rect.bl + (0, -0.2 * rect.height),  # insertion point
            text,  # text to render
            fontsize=rect.height,  # take this as font size
            fontname=FONTNAME,  # fallback font
            # render_mode=0,  # text is invisible but still extractable
            morph=(rect.bl, mat),  # adjust width to fit the line box
        )
