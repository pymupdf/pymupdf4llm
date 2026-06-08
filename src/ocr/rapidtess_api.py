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
import pymupdf
import numpy as np
from rapidocr_onnxruntime import RapidOCR
from .get_culled_pixmap import get_pixmap

class RapidOCR_DetOnly(RapidOCR):
    """
    A future-proof, detection-only variant of RapidOCR.

    This class ensures:
    - No recognition models are loaded.
    - No recognition or classification steps are executed.
    - Stable public API for downstream consumers.
    - Defensive behavior against internal API changes in RapidOCR.
    - Drop-in replacement for the standard RapidOCR class.

    Intended for production use in environments where only text
    detection (bounding boxes) is required and recognition overhead
    must be avoided.
    """

    def __init__(self):
        # Call base class constructor (loads all models by default)
        super().__init__()

        # Remove recognition-related components if they exist.
        # This prevents recognition from being executed even if
        # RapidOCR changes internal behavior in future versions.
        for attr in ("text_recognizer", "text_classifier", "text_recognizer_session"):
            if hasattr(self, attr):
                setattr(self, attr, None)

        # If RapidOCR introduces flags in future versions, disable them.
        for flag in ("use_rec", "use_cls"):
            if hasattr(self, flag):
                setattr(self, flag, False)

    def __call__(self, img):
        """
        Execute detection only.

        Always returns a tuple: (boxes, scores)

        This method is defensive: if RapidOCR changes the return
        format of text_detector(), a clear error is raised instead
        of silently producing incorrect results.
        """
        # Ensure the detector exists
        if not hasattr(self, "text_detector") or self.text_detector is None:
            raise RuntimeError("RapidOCR_DetOnly: No text_detector available.")

        # Run detection
        result = self.text_detector(img)

        # Validate return format (RapidOCR may change this in the future)
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            boxes, scores = result[0], result[1]
        else:
            raise RuntimeError(
                f"RapidOCR_DetOnly: Unexpected return format from text_detector: {type(result)}"
            )

        return boxes, scores

    def detect(self, img):
        """
        Stable public API for detection.

        Downstream users should call this method instead of __call__().
        This ensures API stability even if internal behavior changes.
        """
        return self.__call__(img)


TESSDATA = pymupdf.get_tessdata()
if TESSDATA is None:
    pymupdf.message(
        "Warning: Tesseract OCR is not available. No OCR text will be extracted."
    )

FONT = pymupdf.Font("cjk")  # this is the "Droid Sans Fallback" font
FONTNAME = "myfont"  # its reference name in the page
REPLACEMENT_UNICODE = chr(0xFFFD)  # Unicode Replacement Character
STROKED_TEXT = pymupdf.mupdf.FZ_STEXT_STROKED
FILLED_TEXT = pymupdf.mupdf.FZ_STEXT_FILLED


def ocr_text(span) -> bool:
    if (span["char_flags"] & STROKED_TEXT) or (span["char_flags"] & FILLED_TEXT):
        return False
    return True


ENGINE = RapidOCR_DetOnly()

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
    if irect.is_empty:
        return ""
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
    recognition in each identified region (bounding box).

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

    if TESSDATA is None:
        # cannot perform OCR without Tesseract, so just
        return

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
    pix = get_pixmap(displaylist, dpi=dpi, rects=spans)

    # For converting ENGINE box coordinates to page coordinates
    matrix = pymupdf.Rect(pix.irect).torect(page.rect)

    # make numpy array from pixmap
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height,
        pix.width,
        3,
    )

    # Execute ENGINE's Detector
    boxes, score = ENGINE.detect(img)

    if boxes is None or not len(boxes):  # nothing detected
        return

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
    for box in boxes:
        irect = pymupdf.IRect(min(p[0] for p in box),
                              min(p[1] for p in box),
                              max(p[0] for p in box),
                              max(p[1] for p in box))
        text = get_text(pix, irect)  # execute Tesseract OCR on the line box
        tess_results.append((irect, text))
    if not tess_results:  # guard against no text found
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
