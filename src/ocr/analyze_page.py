import pprint
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pymupdf
from pymupdf import mupdf
from .compute_ocr_features import FEATURE_NAMES, compute_features

FLAGS = (
    0
    | pymupdf.TEXT_ACCURATE_BBOXES
    | pymupdf.TEXT_PRESERVE_IMAGES
    | pymupdf.TEXT_COLLECT_VECTORS
)
GRAY = mupdf.fz_device_gray()  # MuPDF version of standard gray colorspace
TYPE3_FONT_NAME = "Type3"  # MuPDF starts the fontname with this string
TESSERACT_FONT_NAME = "GlyphLessFont"
REPLACEMENT_CHARACTER = chr(0xFFFD)
TEXT_STROKED = mupdf.FZ_STEXT_STROKED
TEXT_FILLED = mupdf.FZ_STEXT_FILLED
BLOCK_TEXT = mupdf.FZ_STEXT_BLOCK_TEXT
BLOCK_IMAGE = mupdf.FZ_STEXT_BLOCK_IMAGE
BLOCK_VECTOR = mupdf.FZ_STEXT_BLOCK_VECTOR
# Thresholds
BAD_CHAR_THRESHOLD = 0.05  # >=5% bad chars suggests OCR

# Return needs_ocr as True if the probability is at least this:
OCR_MODEL_THRESHOLD = 0.93

# The model file is in our folder!
_MODEL_PATH = Path(__file__).parent / "ocr_decision_model.onnx"
_session: ort.InferenceSession | None = None
_input_name: str = ""


def _get_session():
    global _session, _input_name
    if _session is None:
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        _session = ort.InferenceSession(str(_MODEL_PATH), opts)
        _input_name = _session.get_inputs()[0].name
    return _session, _input_name


def predict_ocr_probability(features: dict) -> float:
    """Return the probability that these features require OCR."""
    session, input_name = _get_session()
    x = np.array([[features[f] for f in FEATURE_NAMES]], dtype=np.float32)
    probas = session.run(None, {input_name: x})[1]  # output[1] = probability
    return float(probas[0, 1])


def check_images(image_blocks, prob, threshold=0.93):
    """Stage 2: Separate OCR check for large images.

    Currently not in use.

    Args:
        image_blocks: (list) image blocks in page.get_text("dict") format.
        prob: (float) OCR probability for rendered full page
        threshold: (float) probability threshold. If larger recommend OCR.

    Returns:
        bool (do OCR), probability, check count
    """
    best_prob = prob  # probability for rendered page

    for i, block in enumerate(image_blocks):
        # Iterate image blocks.
        # Make a 128x128 GRAY pixmap of the image.
        pix = pymupdf.Pixmap(block["image"])
        if pix.alpha:
            pix = pymupdf.Pixmap(pix, 0)
        if block["mask"]:  # Add mask if present
            mask = pymupdf.Pixmap(block["mask"])
            try:
                pix = pymupdf.Pixmap(pix, mask)
                pix = pymupdf.Pixmap(pix, 0)
            except:
                mask = None
        pix = pymupdf.Pixmap(pymupdf.csGRAY, pix)
        pix = pymupdf.Pixmap(pix, 128, 128)  # downscale Pixmap

        # compute the features and predict OCR probability
        features = compute_features([block], block["bbox"], pix)
        img_prob = predict_ocr_probability(features)
        best_prob = max(best_prob, img_prob)

        if best_prob >= threshold:
            return True, best_prob, i  # Early Exit if OCR is probable

    return False, best_prob, i


# def get_pixmap(displaylist, dpi=200, rects=[]):
#     """Make a pixmap from the page removing "good" text."""

#     # Matrix for desired DPI
#     ctm = mupdf.fz_make_matrix(dpi / 72, 0, 0, dpi / 72, 0, 0)

#     # Returns a GRAY fz_pixmap
#     pm = mupdf.fz_new_pixmap_from_display_list_culling_text2(
#         displaylist, ctm, GRAY, 0, rects
#     )

#     # Convert to a PyMuPDF pixmap
#     pix = pymupdf.Pixmap(pm, 0)
#     return pix


def is_ocr_span(span):
    """If this is an OCR text span."""
    return span["font"] == TESSERACT_FONT_NAME or (
        True
        and span["char_flags"] & TEXT_STROKED == 0
        and span["char_flags"] & TEXT_FILLED == 0
    )


def intersect_rects(r1, r2, bbox_only=False):
    """Speedy version using indices."""
    bbox = (max(r1[0], r2[0]), max(r1[1], r2[1]), min(r1[2], r2[2]), min(r1[3], r2[3]))
    return bbox if bbox_only else pymupdf.Rect(*bbox)


def join_rects(r1, r2, bbox_only=False):
    """Speedy version using indices."""
    bbox = (min(r1[0], r2[0]), min(r1[1], r2[1]), max(r1[2], r2[2]), max(r1[3], r2[3]))
    return bbox if bbox_only else pymupdf.Rect(*bbox)


def bbox_is_empty(bbox) -> bool:
    """Speedy version using indices."""
    return bbox[0] >= bbox[2] or bbox[1] >= bbox[3]


def analyze_page(page, blocks=None, replace_ocr=False, ocr_dpi=150, stats=None) -> dict:
    """Analyze the page for the OCR decision.

    Args:
        blocks: (dict) output of page.get_text("dict") if already available
        replace_ocr: (bool) if True, we should make a new OCR text layer
        stats: (dict) if given fill in execution information (debugging)
    Returns:
        A dict with analysis results. The area-related float values are
        computed as fractions of the total covered area.

        "covered": Rect, page area covered by content
        "img_joins": float, fraction of area of the joined images
        "img_area": float, fraction of sum of image area sizes
        "txt_joins": float, fraction of area of the joined text spans
        "txt_area": float, fraction of sum of text span bbox area sizes
        "vec_joins": float, fraction of area of the joined vector characters
        "vec_area": float, fraction of sum of vectors
        "chars_total": int, count of visible characters
        "chars_bad": int, count of Replacement Unicode characters
        "bad_areas": float, fraction of text areas having bad characters
        "ocr_spans": int, count: text spans with ignored text (render mode 3)
        "pixmap": Pixmap of page with all "good" text removed - for OCR use
        "needs_ocr": bool, final decision
        "reason": str | None, reason for the OCR decision
        "probability": float | None, OCR probability
    """
    # --------------------------------------------------------------------
    # Main analysis
    # --------------------------------------------------------------------
    if blocks is None:  # make "dict" text extraction if not provided
        # stextpage = displaylist.get_textpage(flags=FLAGS)
        # textpage = pymupdf.TextPage(stextpage)
        # blocks = textpage.extractDICT()["blocks"]
        blocks = page.get_text(
            "dict",
            flags=FLAGS,
            clip=pymupdf.INFINITE_RECT(),
        )["blocks"]

    page_rect = page.rect
    img_rect = pymupdf.EMPTY_RECT()  # joined image bboxes
    txt_rect = +img_rect  # joined text span bboxes
    vec_rect = +img_rect  # joined suspicious vector bboxes
    chars_total = 0  # total character count
    chars_bad = 0  # bad character count
    bad_areas = 0.0  # sum of areas of text spans having bad characters
    img_area = 0.0  # sum of image block areas
    txt_area = 0.0  # sum of all text span bbox areas
    vec_area = 0.0  # sum of suspicious vector block areas
    ocr_spans = 0  # count text spans with OCR flags
    ocr_span_boxes = []
    bad_char_boxes = []
    good_char_boxes = []
    image_blocks = []  # currently not used

    for b in blocks:
        bbox = intersect_rects(page_rect, b["bbox"])
        area = bbox.width * bbox.height
        if not area:
            continue

        # Text block: we analyze text spans for bad characters and OCR flags.
        if b["type"] == BLOCK_TEXT:
            for l in b["lines"]:
                for s in l["spans"]:
                    sr = intersect_rects(bbox, s["bbox"])
                    sr_area = sr.width * sr.height
                    if not sr_area:
                        continue
                    text = s["text"].strip()
                    if not text or text.isspace():
                        continue  # ignore spans having no relevant text
                    chars_total += len(text)  # total character count
                    # OCR layer / invisible text
                    if is_ocr_span(s):
                        ocr_spans += 1
                        ocr_span_boxes.append(s["bbox"])
                        continue

                    # bad character count
                    bad_chars = sum(1 for c in text if c == REPLACEMENT_CHARACTER)
                    chars_bad += bad_chars

                    txt_rect = join_rects(txt_rect, sr)
                    txt_area += sr_area
                    if bad_chars:
                        # add area of span area if it contains bad characters
                        bad_areas += sr_area
                        bad_char_boxes.append(s["bbox"])
                    else:
                        good_char_boxes.append(s["bbox"])
            continue

        # Image block: We only look at its area now.
        # OCR decisions based on image content disabled for now.
        if b["type"] == BLOCK_IMAGE:
            # Image block
            img_rect = join_rects(img_rect, bbox)
            img_area += area
            # if area / page_area > 0.10:  # image large enough for text?
            #     image_blocks.append(b)
            continue

        if b["type"] == BLOCK_VECTOR:
            # Vector block
            vec_rect = join_rects(vec_rect, bbox)
            vec_area += area
            continue

    # the rectangle on page covered by content
    covered = img_rect | txt_rect | vec_rect
    if bbox_is_empty(covered):
        # no content at all → return early with empty covered area
        return {
            "covered": covered,
            "img_joins": 0.0,
            "img_area": 0.0,
            "txt_joins": 0.0,
            "txt_area": 0.0,
            "vec_joins": 0.0,
            "vec_area": 0.0,
            "chars_total": 0,
            "chars_bad": 0,
            "bad_areas": 0.0,
            "ocr_spans": 0,
            "pixmap": None,
            "needs_ocr": False,
            "reason": None,
            "probability": None,
        }

    cover_area = (covered[2] - covered[0]) * (covered[3] - covered[1])

    analysis = {
        "covered": covered,
        "img_joins": (abs(img_rect) / cover_area) if cover_area else 0.0,
        "img_area": img_area / cover_area if cover_area else 0.0,
        "txt_joins": (abs(txt_rect) / cover_area) if cover_area else 0.0,
        "txt_area": txt_area / cover_area if cover_area else 0.0,
        "vec_joins": (abs(vec_rect) / cover_area) if cover_area else 0.0,
        "vec_area": vec_area / cover_area if cover_area else 0.0,
        "chars_total": chars_total,
        "chars_bad": chars_bad,
        "bad_areas": bad_areas / cover_area if cover_area else 0.0,
        "ocr_spans": ocr_spans,
        "pixmap": None,
    }

    # --- final OCR decision ---

    if ocr_spans:
        # This page has previously been OCRed.
        # If replace_ocr is False, we keep the existing OCR layer
        # and accept the page as is.
        if isinstance(stats, dict):
            stats["old_ocr"] = stats.get("old_ocr", 0) + 1
        if not replace_ocr:
            # Accept the page with its current OCR layer
            return {**analysis, "needs_ocr": False, "reason": None, "probability": None}

        # Else remove old OCR text and request OCR
        for r in ocr_span_boxes:
            page.add_redact_annot(r)
        page.apply_redactions(
            images=pymupdf.PDF_REDACT_IMAGE_NONE,  # do not touch images
            graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,  # do not touch vectors
            text=pymupdf.PDF_REDACT_TEXT_REMOVE,  # remove old OCR layer
        )
        return {
            **analysis,
            "needs_ocr": True,
            "reason": "ocr_spans",
            "probability": None,
        }

    # 2. Bad character check
    # Too many bad characters result in early exit with OCR recommended.
    if (
        True
        and chars_total
        and txt_area
        and (
            False
            or chars_bad / chars_total > BAD_CHAR_THRESHOLD
            or bad_areas / txt_area > BAD_CHAR_THRESHOLD
        )
    ):
        return {
            **analysis,
            "needs_ocr": True,
            "reason": "chars_bad",
            "probability": None,
        }

    if isinstance(stats, dict):
        stats["model_check"] = stats.get("model_check", 0) + 1

    features = compute_features(blocks, page_rect, page)

    if isinstance(stats, dict) and stats.get("show_features"):  # for debugging
        pprint.pp(features)

    prob = predict_ocr_probability(features)
    needs_ocr = prob >= OCR_MODEL_THRESHOLD  # True if beyond threshold

    # If text-like page detected → OCR needed
    if needs_ocr:
        return {
            **analysis,
            "needs_ocr": needs_ocr,
            "reason": "img_text",
            "probability": prob,
        }

    # Otherwise, check large images to see if they qualify for OCR
    # This is currently disabled: the list 'image_blocks' is empty.
    if image_blocks:
        needs_ocr, prob, check_count = check_images(
            image_blocks, prob, OCR_MODEL_THRESHOLD
        )
        if isinstance(stats, dict):
            stats["img_checks"] = stats.get("img_checks", 0) + check_count
    return {
        **analysis,
        "needs_ocr": needs_ocr,
        "reason": "img_text" if needs_ocr else None,
        "probability": prob,
    }


if __name__ == "__main__":
    """For edbugging purposes."""
    import pprint
    import sys
    import time

    duration = 0
    ocr_pages = []
    STATISTICS = {"show_features": True}
    doc = pymupdf.open(sys.argv[1])
    print(f"OCR-Analysis for {doc.name=}")
    for page in doc:
        t0 = time.perf_counter()
        analysis = analyze_page(page, stats=STATISTICS)
        duration += time.perf_counter() - t0
        if analysis["needs_ocr"]:
            ocr_pages.append((page.number, analysis["reason"]))
    print(
        f"NEW: duration: {duration:.2f}s, per page: {duration/len(doc):.4f}s OCR: {len(ocr_pages)} pages ({len(ocr_pages)/len(doc)*100:.2f}%)"
    )
    out = open(doc.name + ".txt", "w")
    pprint.pp(ocr_pages, width=200, stream=out)
    out.close()
    pprint.pp(STATISTICS)
