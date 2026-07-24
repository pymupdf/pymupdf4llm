import numpy as np
import pymupdf
from numpy.lib.stride_tricks import as_strided
from pymupdf import mupdf

FEATURE_NAMES = [
    # Text
    "num_spans",
    "text_area",
    "text_density",
    "avg_span_height",
    "avg_span_width",
    # Layout
    "num_blocks",
    "num_images",
    "image_area",
    "image_density",
    "num_small_oblique_vectors",
    "log_vector_density",
    "page_sobel_energy",
    "page_sobel_entropy",
    "page_sobel_var",
    "page_white_ratio",
    # Pixel features of the most relevant image
    "img_fft_energy",
    "img_fft_ratio",
    "img_black_ratio",
    "img_white_ratio",
    "img_sobel_energy",
    "img_sobel_orientation_entropy",
    "img_sobel_local_variance",
]

SOBEL_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)

SOBEL_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)


def conv2d_fast(img, kernel):
    kh, kw = kernel.shape
    ih, iw = img.shape

    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")

    # Sliding window view
    shape = (ih, iw, kh, kw)
    strides = (
        padded.strides[0],
        padded.strides[1],
        padded.strides[0],
        padded.strides[1],
    )
    windows = as_strided(padded, shape=shape, strides=strides)

    # Vektorisierte Faltung
    return np.einsum("ijkl,kl->ij", windows, kernel)


def sobel_features(gray):
    gx = conv2d_fast(gray, SOBEL_X)
    gy = conv2d_fast(gray, SOBEL_Y)

    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx)

    sobel_energy = float(np.mean(mag))

    bins = np.linspace(-np.pi, np.pi, 37)
    hist, _ = np.histogram(ang, bins=bins)
    p = hist / (hist.sum() + 1e-9)
    sobel_entropy = float(-np.sum(p * np.log(p + 1e-9)))

    sobel_var = float(np.var(mag))

    return sobel_energy, sobel_entropy, sobel_var


def sobel_features_page(page):
    # 1. Render page as GRAY pixmap
    pix = page.get_pixmap(dpi=72, colorspace=pymupdf.csGRAY, alpha=False)

    # 2. Downscale to 128x128 using PyMuPDF
    pix_small = pymupdf.Pixmap(pix, 128, 128)
    gray = np.frombuffer(pix_small.samples, dtype=np.uint8).reshape(128, 128)
    page_white_ratio = (gray > 230).mean()
    rc = list(sobel_features(gray))
    rc.append(page_white_ratio)
    return rc


def compute_features(blocks, page_rect, page):
    """Return model-relevant features for the given page.

    Args:
        blocks: (list) blocks created by get_text("dict",flags=FLAGS)["blocks].
            The FLAGS *MUST* include PRESERVE_IMAGES and COLLECT_VECTORS.
        page_rect: (Rect) the rectangle Page.rect.
        pixmap: GRAY 128x128 sized image of the page without all text

    Returns:
        A dictionary of page "features" (see code here) that can be used to
        request a recommendation whether to OCR the page from a compatible
        ML model.
    """
    num_blocks = len(blocks)  # total block count
    page_area = (page_rect[2] - page_rect[0]) * (page_rect[3] - page_rect[1])

    # list of span rectangles
    span_rects = [
        s["bbox"]
        for b in blocks
        if b["type"] == mupdf.FZ_STEXT_BLOCK_TEXT
        for l in b["lines"]
        for s in l["spans"]
        if not (s["text"].isspace())
    ]
    num_spans = len(span_rects)  # count of text spans

    # total area covered by text
    text_area = sum((r[3] - r[1]) * (r[2] - r[0]) for r in span_rects)
    # text density
    text_density = text_area / page_area if page_area else 0

    if num_spans > 0:
        avg_span_height = np.mean([(r[3] - r[1]) for r in span_rects])
        avg_span_width = np.mean([(r[2] - r[0]) for r in span_rects])
    else:
        avg_span_height = 0
        avg_span_width = 0

    # vector block bboxes
    vectors = [b for b in blocks if b["type"] == mupdf.FZ_STEXT_BLOCK_VECTOR]
    vector_area = sum(
        (b["bbox"][3] - b["bbox"][1]) * (b["bbox"][2] - b["bbox"][0]) for b in vectors
    )
    log_vector_density = np.log1p(vector_area / page_area + 1e-9) if page_area else 0
    num_small_oblique_vectors = sum(
        1
        for b in vectors
        if b.get("isrect", False)
        and b["bbox"][2] - b["bbox"][0] < 15
        and b["bbox"][3] - b["bbox"][1] < 15
    )

    # Compute Sobel energy for the entire page as a layout feature
    page_sobel_energy, page_sobel_entropy, page_sobel_var, page_white_ratio = (
        sobel_features_page(page)
    )
    # image block bboxes
    images = [b for b in blocks if b["type"] == mupdf.FZ_STEXT_BLOCK_IMAGE]
    # total area covered by images
    num_images = len(images)  # image block count
    image_area = 0

    relevant_images = []
    for img in images:
        visible_bbox = [
            max(img["bbox"][0], page_rect[0]),
            max(img["bbox"][1], page_rect[1]),
            min(img["bbox"][2], page_rect[2]),
            min(img["bbox"][3], page_rect[3]),
        ]
        this_img_area = (visible_bbox[3] - visible_bbox[1]) * (
            visible_bbox[2] - visible_bbox[0]
        )
        image_area += this_img_area
        if this_img_area <= 0.01 * page_area:
            continue
        total_img_area = (img["bbox"][3] - img["bbox"][1]) * (
            img["bbox"][2] - img["bbox"][0]
        )
        score = this_img_area / total_img_area * this_img_area
        relevant_images.append((score, img))

    # image density
    image_density = image_area / page_area if page_area else 0

    features = {
        "num_spans": num_spans,
        "text_area": text_area,
        "text_density": text_density,
        "avg_span_height": float(avg_span_height),
        "avg_span_width": float(avg_span_width),
        "num_blocks": num_blocks,
        "num_images": num_images,
        "image_area": image_area,
        "image_density": image_density,
        "num_small_oblique_vectors": num_small_oblique_vectors,
        "log_vector_density": log_vector_density,
        "page_sobel_energy": page_sobel_energy,
        "page_sobel_entropy": page_sobel_entropy,
        "page_sobel_var": page_sobel_var,
        "page_white_ratio": page_white_ratio,
        "img_fft_energy": 0.0,
        "img_fft_ratio": 0.0,
        "img_black_ratio": 0.0,
        "img_white_ratio": 0.0,
        "img_sobel_energy": 0.0,
        "img_sobel_orientation_entropy": 0.0,
        "img_sobel_local_variance": 0.0,
    }

    if not relevant_images:
        return features

    # find most relevant image on page
    best_img = max(relevant_images, key=lambda img: img[0])[1]

    pix = pymupdf.Pixmap(best_img["image"])
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)  # remove alpha channel
    try:  # apply any mask if available and compatible
        if best_img["mask"]:
            mask_pix = pymupdf.Pixmap(best_img["mask"])
            pix = pymupdf.Pixmap(pix, mask_pix)  # apply mask to image
            mask_pix = None
            pix = pymupdf.Pixmap(pix, 0)  # remove alpha channel after masking
    except Exception as e:
        pass
    if pix.n > 1:
        pix = pymupdf.Pixmap(pymupdf.csGRAY, pix)  # convert to grayscale

    # resize to 128x128 for consistent feature extraction
    pix = pymupdf.Pixmap(pix, 128, 128)

    # FFT / textmap
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(128, 128)
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)

    fft_energy = magnitude.mean()
    fft_ratio = magnitude[magnitude > magnitude.mean()].mean() / (
        magnitude[magnitude <= magnitude.mean()].mean() + 1e-6
    )

    black_ratio = (img < 128).mean()
    white_ratio = (img > 230).mean()
    # --- Sobel features ---
    sobel_energy, sobel_entropy, sobel_var = sobel_features(img)
    features.update(
        {
            "img_fft_energy": float(fft_energy),
            "img_fft_ratio": float(fft_ratio),
            "img_black_ratio": float(black_ratio),
            "img_white_ratio": float(white_ratio),
            "img_sobel_energy": float(sobel_energy),
            "img_sobel_orientation_entropy": float(sobel_entropy),
            "img_sobel_local_variance": float(sobel_var),
        }
    )

    return features
