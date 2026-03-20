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

FONT = pymupdf.Font("cjk")  # this is the "Droid Sans Fallback" font
FONTNAME = "myfont"  # its reference name in the page
REPLACEMENT_UNICODE = chr(0xFFFD)  # Unicode Replacement Character


def ocr_text(span) -> bool:
    if not (span["char_flags"] & 32) and not (span["char_flags"] & 16):
        return True
    return False


ENGINE = RapidOCR()

# pass any keyword arguments to RapidOCR when calling exec_ocr()
KWARGS = {}


def exec_ocr(page, dpi=300, pixmap=None, language="eng", keep_ocr_text=False):
    """Perform OCR on the given page and insert recognized text.

    If a Pixmap is provided, the DPI parameter is ignored. Otherwise, an RGB
    Pixmap is created from the page at the specified DPI.
    The DPI value is also used if extractable text is present.
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

    text_blocks = page.get_text("dict", flags=pymupdf.TEXT_ACCURATE_BBOXES)["blocks"]
    # get bboxes with legible significant text on page
    spans = []  # spans with legible text
    fffd_spans = []  # spans containing U+FFFD characters.
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

    # for converting box coordinates to page coordinates
    matrix = pymupdf.Rect(pixmap.irect).torect(page.rect)

    # Execute RapidOCR, passing any additional keyword arguments.
    # t0 = time.perf_counter()
    result = ENGINE.__call__(img, **KWARGS)
    # t1 = time.perf_counter()
    # print(f"OCR execution time: {t1-t0:.2f} seconds")

    if fffd_spans:
        # FFFD spans should have been recognized by OCR.
        # We therefore remove them here.
        for sbbox in fffd_spans:
            page.add_redact_annot(sbbox)
        page.apply_redactions(
            images=pymupdf.PDF_REDACT_IMAGE_NONE,
            graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
            text=pymupdf.PDF_REDACT_TEXT_REMOVE,
        )
    # insert the font into the page if not already present
    page.insert_font(fontname=FONTNAME, fontbuffer=FONT.buffer)

    # Insert recognized text
    lines = result[0]
    confs = result[1]
    for line in lines:
        if not line:
            continue

        box, text, conf = line

        # PaddleOCR box: 4 points (tl, tr, br, bl)
        tl, tr, br, bl = box
        rect = pymupdf.Rect(tl[0], tl[1], br[0], br[1]) * matrix

        if not text.strip():
            continue

        fontsize = rect.height
        mat = adjust_width(text, fontsize, rect)

        page.insert_text(
            rect.bl + (0, -0.2 * fontsize),
            text,
            fontsize=fontsize,
            fontname=FONTNAME,
            render_mode=0,  # standard PDF text rendering
            morph=(rect.bl, mat),
        )
