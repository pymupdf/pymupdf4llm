import os

import subprocess

import pymupdf4llm


def test_sce_156():
    subprocess.run(f'pip install rapidocr', shell=1, check=1)
    path = os.path.normpath(f'{__file__}/../../tests/test_sce_156.pdf')
    pymupdf4llm.to_markdown(path, page_chunks=True, show_progress=False, use_ocr=True)
