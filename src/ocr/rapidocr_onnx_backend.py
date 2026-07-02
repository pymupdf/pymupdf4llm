"""
rapidocr_onnxruntime_backend.py
Backend for RapidOCR-ONNXRuntime
Unified interface for:
  - full_ocr(img)  -> List[(rect, text, score)]
  - det_only(img)  -> List[(rect, score)]
The engine is initialized exactly once.
Note: RapidOCR-ONNXRuntime can disable recognition.
      We therefore can use different ways to perform
      recognition on the detected text regions, for instance Tesseract.
"""

import numpy as np
from rapidocr_onnxruntime import RapidOCR

# global Engine instance, initialized only once
ENGINE = None


def init_engine():
    """
    Initializes the ONNX Runtime Engine exactly once.
    """
    global ENGINE
    if ENGINE is None:
        ENGINE = RapidOCR()  # no parameters!
    return ENGINE


# ------------------------------------------------------------
# Full OCR: Detection + Recognition
# ------------------------------------------------------------
def full_ocr(img: np.ndarray):
    """
    Returns list of (rect, text, score)
    """
    engine = init_engine()
    results, times = engine(img)

    # results is a list of [box, text, score]
    return [(box, text, float(score)) for box, text, score in results]


# ------------------------------------------------------------
# Detection only
# ------------------------------------------------------------
def det_only(img: np.ndarray):
    """
    Returns list of (rect, score)
    """
    engine = init_engine()

    # Only execute Detection
    dt_boxes, det_time = engine.text_detector(img)

    # dt_boxes is an ndarray of shape (N, 4, 2)
    # we assume score 1. because this is a detection-only function.
    results = [(box.tolist(), 1.0) for box in dt_boxes]

    return results
