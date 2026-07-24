def detect_rapidocr_backend():
    try:
        import rapidocr

        return "rapidocr"
    except Exception:
        pass

    try:
        import rapidocr_onnxruntime

        return "rapidocr_onnxruntime"
    except Exception:
        pass

    return None
