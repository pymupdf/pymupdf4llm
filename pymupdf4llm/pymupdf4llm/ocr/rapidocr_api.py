import rapidocr
from rapidocr import RapidOCR
import pymupdf
import numpy as np

# default RapidOCR engine, can be configured by calling set_engine()
ENGINE = RapidOCR()

# pass any keyword arguments to RapidOCR when calling exec_ocr()
KWARGS = {}


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
        """Compute a matrix to adjust the width of the rendered text."""
        tl = FONT.text_length(text, fontsize)
        mat = pymupdf.Matrix(rect.width / tl, 1)
        return mat

    assert isinstance(ENGINE, RapidOCR)

    FONT = pymupdf.Font("cjk")  # this is the "Droid Sans Fallback" font
    FONTNAME = "myfont"  # its reference name in the page
    # insert the font into the page if not already present
    page.insert_font(fontname=FONTNAME, fontbuffer=FONT.buffer)
    pix = pixmap

    # make pixmap if not provided
    if pix is None:
        pix = page.get_pixmap(dpi=dpi)

    # make numpy array from pixmap
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, pix.width, pix.n)

    # for converting box coordinates to page coordinates
    matrix = pymupdf.Rect(pix.irect).torect(page.rect)

    # Execute RapidOCR, passing any additional keyword arguments.
    result = ENGINE.__call__(img, **KWARGS)
    boxes = result.boxes  # detected line boundary boxes
    texts = result.txts  # detected line text strings for each box
    try:
        words = result.word_results  # word-level results
    except AttributeError:
        words = []  # no word-level results available

    # Create the OCR text layer by inserting detected text into
    # the page using the fallback font.
    for i, box in enumerate(boxes):
        tl, tr, br, bl = box  # top-left, top-right, bottom-right, bottom-left

        # this is the line box
        rect = pymupdf.Rect(tl[0], tl[1], br[0], br[1]) * matrix

        mywords = []  # collect word boxes for this line (may be missing)

        # If words are present, they have their own coordinate system,
        # so we need to find the bounding box of all words in this line to
        # adjust to the line box coordinates.
        start = pymupdf.EMPTY_RECT()  # union rectangle of words in this line

        try:
            for _, _, wbox in words[i]:
                for pnt in wbox:
                    start |= pnt
        except IndexError:
            # no word boxes for this line: render the line as a single text
            pass
        if start.is_empty:
            # no words detected in the line box:
            # render the entire line as a single text
            mat = adjust_width(texts[i], rect.height, rect)
            page.insert_text(
                rect.bl + (0, -0.2 * rect.height),  # insertion point
                texts[i],  # text to render
                fontsize=rect.height,  # take this as font size
                fontname=FONTNAME,  # fallback font
                render_mode=3,  # text is invisible but still extractable
                morph=(rect.bl, mat),  # adjust width to fit the line box
            )
            continue
        # words detected: render each word separately
        # word boxes have their own coordinate system: make a transform matrix
        minimatrix = start.torect(rect)  # maps coordinates to line bbox
        for wtext, _, wbox in words[i]:
            wrect = (
                pymupdf.Rect(wbox[0][0], wbox[0][1], wbox[2][0], wbox[2][1])
                * minimatrix
            )
            if wrect.is_empty:
                continue
            wrect.y0 = rect.y0  # align top with line box
            wrect.y1 = rect.y1  # align bottom with line box

            # DEBUG: draw word boxes for visualization
            # page.draw_rect(wrect, color=(0, 1, 1), width=0.3)

            wtext = " " + wtext  # make sure we separate from previous word
            mat = adjust_width(wtext, wrect.height, wrect)
            page.insert_text(
                wrect.bl + (0, -0.2 * wrect.height),
                wtext,
                fontsize=wrect.height,
                fontname=FONTNAME,
                render_mode=3,
                morph=(wrect.bl, mat),
            )
    # ------------------------------------------------------------------------
    # Finished creting the OCR text layer of this page:
    # We have inserted all detected text into the page, so it is extractable
    # by the Layout plugin.
    # ------------------------------------------------------------------------
