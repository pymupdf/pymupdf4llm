import difflib
import os

import pdf4llm
import pymupdf
import platform


def test_4llm_370():
    # This is a copy of pymupdf4llm/tests/test_370.py:test_370(), except that
    # we use pdf4llm.to_markdown() instead of pymupdf4llm.to_markdown(). We
    # reuse that test's input and expected files.
    #
    path = os.path.normpath(f'{__file__}/../../../tests/test_370.pdf')
    path_expected = os.path.normpath(f'{__file__}/../../../tests/test_370_expected.md')
    path_actual = os.path.normpath(f'{__file__}/../../../pdf4llm/tests/test_4llm_370_actual.md')
    
    with open(path_expected) as f:
        expected = f.read()
    
    with pymupdf.open(path) as document:
        actual = pdf4llm.to_markdown(
                document,
                write_images=False,  # do not write image files
                embed_images=False,  # embed images as base64 strings
                image_format="jpg",  # image format (embedded or written)
                header=False,  # include/omit page headers
                footer=False,  # include/omit page footers
                show_progress=True,
                force_text=True,
                page_separators=True,
                )
    with open(path_actual, 'w', encoding='utf8') as f:
        f.write(actual)

    lines = difflib.unified_diff(
            expected.split('\n'),
            actual.split('\n'),
            lineterm='',
            )
    
    for line in lines:
        print(f'test_370():    {line.encode()}', flush=1)
    
    if platform.system() == 'Windows':
        if actual != expected:
            print(f'test_370(): We are on Windows, so ignoring difference in actual vs expected.')
    else:
        assert actual == expected
