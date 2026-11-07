import importlib.metadata


def detect_rapidocr_backend():
    try:
        rc = importlib.metadata.version("rapidocr")
        if rc is not None:
            return "rapidocr"
    except Exception:
        pass

    try:
        rc = importlib.metadata.version("rapidocr_onnxruntime")
        if rc is not None:
            return "rapidocr_onnxruntime"
    except Exception:
        pass

    return None
