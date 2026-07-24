from .detect_rapidocr import detect_rapidocr_backend
from .exec_ocr_interface import exec_ocr_detection

# load the right RapidOCR backend
rapidocr_backend = detect_rapidocr_backend()
if rapidocr_backend == "rapidocr":
    from .rapidocr_391_backend import det_only
elif rapidocr_backend == "rapidocr_onnxruntime":
    from .rapidocr_onnx_backend import det_only
else:
    det_only = None


def exec_ocr(page, dpi=150, pixmap=None, language="eng", keep_ocr_text=False):
    """This callback function performs OCR on the given page.

    It uses RapidOCR for text region detection and Tesseract OCR for text
    recognition in each identified region (bounding box).

    The pixmap parameter is deprecated and ignored.
    """
    return exec_ocr_detection(
        page, det_only, dpi=dpi, language=language, keep_ocr_text=keep_ocr_text
    )
