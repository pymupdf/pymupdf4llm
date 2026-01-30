import os
import sys

def log(text):
    print(f'### {__file__}: {text}', flush=1)
    
log(f'import pymupdf4llm')
import pymupdf
import pymupdf4llm
import pymupdf4llm._layout.features

if sys.argv[1] == '0':
    pass
elif sys.argv[1] == '1':
    log(f'pymupdf._layout.activate()')
    pymupdf4llm._layout.activate()
else:
    assert 0, f'Unrecognised {sys.argv[1:]=}'

path_out = sys.argv[2]

pdf_path = os.path.normpath(f'{__file__}/../../tests/test_activate.pdf')
log(f'doc = pymupdf.open(pdf_path)')
doc = pymupdf.open(pdf_path)
log(f'md = pymupdf4llm.to_markdown(doc)')
md = pymupdf4llm.to_markdown(doc)
log(f'writing md to {path_out=}.')
with open(path_out, 'w', encoding='utf8') as f:
    f.write(md)
