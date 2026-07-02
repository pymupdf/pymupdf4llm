import math
import os

import pymupdf
from pymupdf import mupdf

MAX_PIXELS = 10  # maximum pixel count for OCR, in millions


def max_dpi_for_page(mediabox, max_pixels: int = 0) -> int:
    """
    Compute the maximum integer DPI such that
    page.get_pixmap(dpi=dpi) has < max_pixels pixels.
    """
    w_pt = mediabox[2] - mediabox[0]
    h_pt = mediabox[3] - mediabox[1]

    if w_pt <= 0 or h_pt <= 0:
        return 0

    # a = width_in_inches, b = height_in_inches
    a = w_pt / 72
    b = h_pt / 72

    dpi_est = math.sqrt(max_pixels / (a * b))
    return int(dpi_est)


def pixmap_is_empty(pix, threshold=250):
    colors = pix.color_count(colors=True)  # dict: {b'\xRR\xGG\xBB': count}

    # No colors ? completely empty (rare, but possible)
    if not colors:
        return True

    # Multiple colors ? content present
    if len(colors) > 1:
        return False

    # Exactly one color ? check if it is (almost) white
    ((rgb, count),) = colors.items()  # unpack single item

    # Threshold due to JPEG artifacts
    if min(rgb) >= threshold:
        return True

    return False


def get_pixmap(displaylist, dpi=150, rects=None, empty_threshold=250):
    """Make a pixmap from the page ignoring text in the rects."""
    mediabox = displaylist.rect
    if not rects:
        rects = [mediabox]
    max_pixels = int(os.getenv("PYMUPDF_MAX_OCRSIZE", MAX_PIXELS)) * 10**6
    max_dpi = max_dpi_for_page(mediabox, max_pixels=max_pixels)
    if dpi > max_dpi:
        pymupdf.message(
            f"Page too large for {dpi=}, reducing to dpi={max_dpi}. Results may be impaired."
        )
        dpi = max_dpi

    # Matrix for desired DPI
    ctm = mupdf.fz_make_matrix(dpi / 72, 0, 0, dpi / 72, 0, 0)

    # Returns a RGB fz_pixmap
    pm = mupdf.fz_new_pixmap_from_display_list_culling_text2(
        displaylist,
        ctm,
        mupdf.fz_device_rgb(),
        0,
        [mupdf.ll_fz_make_rect(*r) for r in rects],
    )

    # Convert to a PyMuPDF pixmap
    pix = pymupdf.Pixmap(pm, 0)
    empty = pixmap_is_empty(pix, threshold=empty_threshold)
    return pix, empty
