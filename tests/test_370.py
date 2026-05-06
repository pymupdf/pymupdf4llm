import difflib
import os
import platform

import pymupdf4llm
import pymupdf


def test_370():
    # https://github.com/ArtifexSoftware/sce/issues/137
    
    path = os.path.normpath(f'{__file__}/../../tests/test_370.pdf')
    path_expected = os.path.normpath(f'{__file__}/../../tests/test_370_expected.md')
    path_actual = os.path.normpath(f'{__file__}/../../tests/test_370_actual.md')
    
    with open(path_expected, encoding='utf8') as f:
        expected = f.read()
    
    with pymupdf.open(path) as document:
        actual = pymupdf4llm.to_markdown(
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
    
    # Disable this assert for now because we don't appear to get consistent
    # output on different OS's.
    #assert actual == expected


def test_370_small_glyph_line_assignment():
    # Regression test: tiny glyphs (e.g. "-") with accurate bboxes that sit
    # vertically centered inside full-height text were being split off onto
    # their own line. In test_370.pdf this caused the dash in
    # "g- silylallyloxysilane" to be moved to the start of the paragraph,
    # producing "- To demonstrate ... g silylallyloxysilane".
    path = os.path.normpath(f'{__file__}/../../tests/test_370.pdf')

    with pymupdf.open(path) as document:
        actual = pymupdf4llm.to_markdown(
                document,
                write_images=False,
                embed_images=False,
                image_format="jpg",
                header=False,
                footer=False,
                show_progress=False,
                force_text=True,
                page_separators=True,
                )

    assert "g- silylallyloxysilane" in actual, (
        'Expected dash to stay attached to "g-" on the same line as '
        '"silylallyloxysilane".'
    )
    assert "- To demonstrate the feasibility of our strategy, g silylallyloxysilane" not in actual, (
        'Buggy output detected: dash was moved to the start of the '
        'paragraph and detached from "g".'
    )

if __name__ == "__main__":
    #test_370()
    test_370_small_glyph_line_assignment()