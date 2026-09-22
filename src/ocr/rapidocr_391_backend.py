"""
rapidocr_391_backend.py
Backend for RapidOCR 3.9.1 (or compatible versions).
Unified interface for:
  - full_ocr(img)  -> List[(rect, text, score)]
  - det_only(img)  -> List[(rect, score)]
The engine is initialized exactly once.
Note: RapidOCR 3.9.1 always performs recognition.
      We cannot disable recognition, but we can ignore it.
"""

import logging

import numpy as np
from rapidocr import RapidOCR

# Kill RapidOCR's own handler
rapid_logger = logging.getLogger("RapidOCR")
rapid_logger.handlers.clear()
rapid_logger.propagate = False
rapid_logger.setLevel(logging.CRITICAL)

# global Engine instance, initialized only once
ENGINE = None


def init_engine():
    global ENGINE
    if ENGINE is None:
        ENGINE = RapidOCR()
    return ENGINE


# ------------------------------------------------------------
# Full OCR: Detection + Recognition
# ------------------------------------------------------------
def full_ocr(img: np.ndarray):
    """
    Performs Detection + Recognition.

    Parameters:
        img: numpy array (image)

    Returns:
        List of tuples (rect, text, score)
    """
    engine = init_engine()
    result = engine(img)

    boxes = result.boxes  # List of bounding boxes
    texts = result.txts  # Recognized texts
    scores = result.scores  # Confidence scores
    if any(x is None for x in [boxes, texts, scores]):
        return []
    return [(rect, text, score) for rect, text, score in zip(boxes, texts, scores)]


# ------------------------------------------------------------
# Detection only
# ------------------------------------------------------------
def det_only(img: np.ndarray):
    """
    Performs detection only (recognition results are ignored).

    Parameters:
        img: numpy array (image)

    Returns:
        List of tuples (rect, score)
    """
    engine = init_engine()
    result = engine.text_det(img)

    boxes = result.boxes
    scores = result.scores
    if any(x is None for x in [boxes, scores]):
        return []
    return [(rect, score) for rect, score in zip(boxes, scores)]
