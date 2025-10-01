import sys

if "pymupdf" not in sys.modules:
    raise ImportError("pymupdf must be imported before importing this module.")

def activate():
    """Create a layout analyzer function using an ONNX model."""
    import pymupdf
    from . import DocumentLayoutAnalyzer
    from . import onnx
    from . import common_util
    CU = common_util
    MODEL = DocumentLayoutAnalyzer.get_model()

    def _get_layout(*args, **kwargs):
        page=args[0]
        data_dict=CU.create_input_data_by_pymupdf(page)
        det_result = MODEL.predict(data_dict)
        page.layout_information = det_result

    pymupdf._get_layout = _get_layout
