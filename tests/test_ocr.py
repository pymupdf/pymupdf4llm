import os

import pymupdf
import pymupdf4llm


REPLACEMENT_UNICODE = chr(0xFFFD)
g_root = os.path.normpath(f'{__file__}/../..')

def _ocr_tesseract_available():
    try:
        tesseract = pymupdf.get_tessdata() 
    except Exception:
        tesseract = None
    return bool(tesseract)


def _ocr_rapidocr_onnxruntime_available():
    try:
        import rapidocr_onnxruntime
    except Exception:
        rapidocr_onnxruntime = None
    return bool(rapidocr_onnxruntime)


def test_ocr_1():
    print()
    path = os.path.normpath(f'{g_root}//tests/test_ocr_loremipsum_FFFD.pdf')
    md = pymupdf4llm.to_markdown(path)
    with open(f'{g_root}/tests/out_test_ocr_1.md', 'w', encoding='utf-8') as f:
        f.write(md)
    if _ocr_tesseract_available() or _ocr_rapidocr_onnxruntime_available():
        assert REPLACEMENT_UNICODE not in md
    else:
        assert REPLACEMENT_UNICODE in md

def test_ocr_2():
    print()
    path = os.path.normpath(f'{g_root}/tests/test_ocr_loremipsum_FFFD.pdf')
    md = pymupdf4llm.to_markdown(path, use_ocr=False)
    with open(f'{g_root}/tests/out_test_ocr_2.md', 'w', encoding='utf-8') as f:
        f.write(md)
    assert REPLACEMENT_UNICODE in md

def test_ocr_3():
    print()
    path = os.path.normpath(f'{g_root}/tests/test_ocr_loremipsum_svg.pdf')
    md = pymupdf4llm.to_markdown(path)
    md_no_ocr = pymupdf4llm.to_markdown(path, use_ocr=False)
    with open(f'{g_root}/tests/out_test_ocr_3.md', 'w') as f:
        f.write(md)
    with open(f'{g_root}/tests/out_test_ocr_3_no_ocr.md', 'w', encoding='utf-8') as f:
        f.write(md_no_ocr)
    if _ocr_tesseract_available():
        assert len(md_no_ocr) < len(md)
    else:
        md_no_ocr == md
