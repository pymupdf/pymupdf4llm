import os
import subprocess
import sys

import pymupdf
import pymupdf4llm


def test_137():
    # This doesn't actually detect any exception, but does show the different
    # output with/without layout.
    
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
