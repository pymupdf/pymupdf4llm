from . import DocumentLayoutAnalyzer
from . import onnx
from . import common_util

import pymupdf

def activate():
    """Create a layout analyzer function using an ONNX model."""
    MODEL = DocumentLayoutAnalyzer.get_model()

    def _get_layout(*args, **kwargs):
        page=args[0]
        data_dict= common_util.create_input_data_by_pymupdf(page)
        det_result = MODEL.predict(data_dict)
        page.layout_information = det_result

    pymupdf._get_layout = _get_layout
