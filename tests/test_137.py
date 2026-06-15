import os
import subprocess
import sys
import textwrap

import pymupdf
import pymupdf4llm


def test_137():
    # This doesn't actually detect any exception, but does show the different
    # output with/without layout.
    if pymupdf.mupdf_version_tuple < (1, 28):
        print(f'test_137(): not running because {pymupdf.mupdf_version=} < 1.28.')
        return
    
    path = os.path.normpath(f'{__file__}/../../tests/test_137.pdf')
    
    pymupdf4llm.use_layout(False)
    with pymupdf.open(path) as document:
        md = pymupdf4llm.to_markdown(document, embed_images=True)
    path_md = f'{path}.out_nolayout.md'
    with open(path_md, 'w') as f:
        f.write(md)
    
    # We attempt to use Devuan's `python3-markdown` package if avaialble, to
    # convert to hmtl, to aid debugging.
    subprocess.run(f'markdown_py {path_md} -f {path_md}.html', shell=1, check=0)
    
    pymupdf4llm.use_layout(True)
    with pymupdf.open(path) as document:
        md = pymupdf4llm.to_markdown(document, embed_images=True)
    path_md = f'{path}.out_layout.md'
    with open(path_md, 'w', encoding='utf8') as f:
        f.write(md)
    subprocess.run(f'markdown_py {path_md} -f {path_md}.html', shell=1, check=0)


def test_to_markdown_link_malicious():
    '''
    Check that when running without layout, we don't propagate bogus links into
    markdown. See: https://bugs.ghostscript.com/show_bug.cgi?id=709173
    '''
    path = os.path.normpath(f'{__file__}/../../tests/test_to_markdown_link_malicious.pdf')
    path_md_expected = os.path.normpath(f'{__file__}/../../tests/test_to_markdown_link_malicious.pdf.expected.md')
    path_md_actual = os.path.normpath(f'{__file__}/../../tests/test_to_markdown_link_malicious.pdf.md')
    # Disable use of layout so we attempt to handle links.
    pymupdf4llm.use_layout(False)
    try:
        with pymupdf.open(path) as document:
            md = pymupdf4llm.to_markdown(document, embed_images=True)
        print('md is:')
        print(textwrap.indent(md, '    '), flush=1)
        with open(path_md_actual, 'w') as f:
            f.write(md)
        with open(path_md_expected) as f:
            md_expected = f.read()
        assert md == md_expected
    finally:
        # Restore default use of layout for other tests.
        pymupdf4llm.use_layout(True)
