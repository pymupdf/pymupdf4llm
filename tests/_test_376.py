
import os

from pymupdf4llm import LlamaMarkdownReader
import pymupdf

g_root = os.path.abspath(f'{__file__}/../..')
path = f'{g_root}/_test_376_out.pdf'

with pymupdf.open() as document:
    document.new_page()
    document.save(path)

reader = LlamaMarkdownReader()
documents = reader.load_data(path)

print(f"Loaded {len(documents)} document(s)")
