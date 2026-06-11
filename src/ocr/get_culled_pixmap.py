import pymupdf
from pymupdf import mupdf


def get_pixmap(displaylist, dpi=300, rects=None):
    """Make a pixmap from the page ignoring text in the rects."""
    if not rects:
        rects = []
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
    return pix
