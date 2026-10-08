import os
import platform
import subprocess

import pymupdf4llm


def test_sce_156():
    GITHUB_ACTIONS = os.environ.get('GITHUB_ACTIONS')
    if platform.system() == 'Windows' and GITHUB_ACTIONS == 'true':
        # 2026-10-08: Have today started to see failures like:
        #   Installing collected packages: antlr4-python3-runtime, opencv-python-headless, omegaconf, colorlog, rapidocr
        #   ERROR: Could not install packages due to an OSError: [WinError 5] Access is denied: 'D:\\a\\aptest\\aptest\\venv-aptest-3.13.16-64\\Lib\\site-packages\\cv2\\cv2.pyd'
        #   Check the permissions.
        #   FAILED tests\test_sce_156.py::test_sce_156 - subprocess.CalledProcessError: Command 'pip install rapidocr' returned non-zero exit status 1.
        #
        # Possibly related to release of rapidocr 3.10.0 on 2026-10-08.
        #
        print(f'test_sce_156(): not running on Windows+github because `pip install rapidocr` can fail.')
        return
    subprocess.run(f'pip install rapidocr', shell=1, check=1)
    path = os.path.normpath(f'{__file__}/../../tests/test_sce_156.pdf')
    pymupdf4llm.to_markdown(path, page_chunks=True, show_progress=False, use_ocr=True)
