"""
rapidocr_universal_backend.py

Universal backend for RapidOCR 3.9.x, 3.10.x, and later versions.

Unified interface for:
  - full_ocr(img)  -> List[(rect, text, score)]
  - det_only(img)  -> List[(rect, score)]

The engine instances are initialized exactly once (lazy singleton pattern).

Compatibility Notes:
  - RapidOCR 3.9.x: Accepts 'params' argument but may not fully respect
    use_cls/use_rec flags. Both functions work correctly.
  - RapidOCR 3.10.x+: Respects 'params' argument with lazy loading.
    Models are only loaded when first used.

Key Differences:
  - In 3.9.x: det_only() returns detection boxes with recognition scores
  - In 3.10.x+: det_only() returns detection boxes with detection scores
    (no recognition model is loaded)

Both versions produce compatible output formats.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
from rapidocr import RapidOCR

# Suppress RapidOCR's logger
rapid_logger = logging.getLogger("RapidOCR")
rapid_logger.handlers.clear()
rapid_logger.propagate = False
rapid_logger.setLevel(logging.CRITICAL)

# Global engine instances - lazily initialized
_ENGINE_FULL: Optional[RapidOCR] = None
_ENGINE_DET_ONLY: Optional[RapidOCR] = None


def get_engine_full() -> RapidOCR:
    """
    Get or create the full OCR engine with det + cls + rec.

    In RapidOCR 3.10.x+, uses lazy loading - models are loaded on first use.
    In RapidOCR 3.9.x, all phases are always loaded.

    Returns:
        RapidOCR engine configured for full OCR pipeline
    """
    global _ENGINE_FULL
    if _ENGINE_FULL is None:
        _ENGINE_FULL = RapidOCR(
            params={
                "Global.use_det": True,
                "Global.use_cls": True,
                "Global.use_rec": True,
            }
        )
    return _ENGINE_FULL


def get_engine_det_only() -> RapidOCR:
    """
    Get or create a detection-only OCR engine.

    In RapidOCR 3.10.x+, this respects the use_rec=False flag and only loads
    the detection model (significant memory/speed improvement).
    In RapidOCR 3.9.x, all phases are always loaded.

    Returns:
        RapidOCR engine configured for detection only
    """
    global _ENGINE_DET_ONLY
    if _ENGINE_DET_ONLY is None:
        _ENGINE_DET_ONLY = RapidOCR(
            params={
                "Global.use_det": True,
                "Global.use_cls": False,
                "Global.use_rec": False,
            }
        )
    return _ENGINE_DET_ONLY


def full_ocr(img: np.ndarray) -> List[Tuple[np.ndarray, str, float]]:
    """
    Perform full optical character recognition: detection + recognition.

    This function extracts text from an image by:
    1. Detecting text regions (bounding boxes)
    2. Classifying text orientation (if enabled)
    3. Recognizing text content

    Parameters:
        img (np.ndarray): Input image as numpy array (BGR or RGB format)

    Returns:
        List[Tuple[np.ndarray, str, float]]: List of tuples containing:
          - rect (np.ndarray): Four-point bounding box coordinates
            [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
          - text (str): Recognized text content
          - score (float): Confidence score of recognition (0.0 to 1.0)

    Returns empty list if no text is detected or if results are invalid.
    """
    try:
        engine = get_engine_full()
        result = engine(img)

        boxes = result.boxes
        texts = result.txts
        scores = result.scores

        # Validate result structure
        if any(x is None for x in [boxes, texts, scores]):
            return []

        # Combine boxes, texts, and scores into tuples
        return [(box, text, score) for box, text, score in zip(boxes, texts, scores)]

    except Exception as e:
        logging.error(f"Error in full_ocr: {e}")
        return []


def det_only(img: np.ndarray) -> List[Tuple[np.ndarray, float]]:
    """
    Perform text detection only (without recognition).

    This function extracts text region locations from an image:
    1. Detecting text regions (bounding boxes)
    2. Returns detection confidence scores

    This mode is faster and more memory-efficient than full_ocr when only
    bounding boxes are needed, especially in RapidOCR 3.10.x+ which supports
    lazy loading of detection-only models.

    Parameters:
        img (np.ndarray): Input image as numpy array (BGR or RGB format)

    Returns:
        List[Tuple[np.ndarray, float]]: List of tuples containing:
          - rect (np.ndarray): Four-point bounding box coordinates
            [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
          - score (float): Detection confidence score (0.0 to 1.0)

    Returns empty list if no text is detected or if results are invalid.

    Note:
        In RapidOCR 3.9.x, all phases are always loaded even when use_rec=False
        In RapidOCR 3.10.x+, only the detection model is loaded (lazy loading)
    """
    try:
        engine = get_engine_det_only()
        result = engine(img)

        boxes = result.boxes
        scores = result.scores

        # Validate result structure
        if any(x is None for x in [boxes, scores]):
            return []

        # Combine boxes and scores into tuples
        return [(box, score) for box, score in zip(boxes, scores)]

    except Exception as e:
        logging.error(f"Error in det_only: {e}")
        return []


def reset_engines() -> None:
    """
    Reset and clear engine instances.

    Call this function to free memory or reinitialize with different settings.
    After calling this, the next call to full_ocr() or det_only() will create
    new engine instances.

    Use case: When switching between different RapidOCR configurations or
    during application shutdown for cleanup.
    """
    global _ENGINE_FULL, _ENGINE_DET_ONLY
    _ENGINE_FULL = None
    _ENGINE_DET_ONLY = None
