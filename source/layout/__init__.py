from . import DocumentLayoutAnalyzer
from . import onnx
from . import pymupdf_util

import pymupdf


def activate():
    """Create a layout analyzer function using an ONNX model."""
    MODEL = DocumentLayoutAnalyzer.get_model()

    def _get_layout(*args, **kwargs):
        page = args[0]
        data_dict = pymupdf_util.create_input_data_from_page(page)
        det_result = MODEL.predict(data_dict)
        return det_result

    pymupdf._get_layout = _get_layout


activate()
