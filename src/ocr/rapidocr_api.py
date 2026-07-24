from .detect_rapidocr import detect_rapidocr_backend
from .exec_ocr_interface import exec_ocr_full

KWARGS = {}  # ignored / deprecated

# load the right RapidOCR backend
rapidocr_backend = detect_rapidocr_backend()
if rapidocr_backend == "rapidocr":
    from .rapidocr_391_backend import full_ocr
elif rapidocr_backend == "rapidocr_onnxruntime":
    from .rapidocr_onnx_backend import full_ocr
else:
    full_ocr = None
print(f"rapidocr_api using backend: {rapidocr_backend}")


def exec_ocr(page, dpi=150, pixmap=None, language=None, keep_ocr_text=False):
    """OCR callback with flexible RapidOCR backend.
    We ensure that legible extractable text is excluded from OCR. We render
    the page without "good" text and perform OCR on the rest.
    """
    return exec_ocr_full(
        page, full_ocr, dpi=dpi, language=language, keep_ocr_text=keep_ocr_text
    )
