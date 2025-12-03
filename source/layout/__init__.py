from . import DocumentLayoutAnalyzer
from . import onnx
from . import pymupdf_util

from ._build import git_sha
from ._build import platform_python_implementation
from ._build import version
from ._build import version_tuple

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
