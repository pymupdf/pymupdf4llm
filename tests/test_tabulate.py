import os

import pymupdf4llm


def test_tablulate_bug():
    # tabulate 0.10.0 made 4llm raise exception with test_tablulate_bug.pdf.
    print()
    print('test_tablulate_bug():')
    path = os.path.normpath(f'{__file__}/../../tests/test_tablulate_bug.pdf')
    page_list = pymupdf4llm.to_text(
            path,
            page_chunks=True,
            use_ocr=False,
            ignore_images=True,
            ignore_graphics=True,
            write_images=False,
            embed_images=False,
            header=False,
            footer=False,
            pages=[0],
            show_progress=True,
            )
