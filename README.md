# PyMuPDF4LLM

PyMuPDF4LLM is a specialized extension of PyMuPDF designed specifically for extracting content from PDFs in a format that's optimized for Large Language Models (LLMs).

## Key Features

1. Markdown Output

- Converts PDFs to clean, structured Markdown format
- Preserves document hierarchy (headers, lists, tables)
- Makes PDF content easily digestible for LLMs like Claude, GPT, etc.

2. Intelligent Structure Detection

- Automatically identifies headers, paragraphs, tables, and images
- Maintains document layout and reading order
- Preserves semantic structure

3. Image Handling

- Extracts images from PDFs
- Can save images separately or encode them inline
- Useful for multimodal LLMs that can process images

## Installation

The Python package on PyPI [pymupdf4llm](https://pypi.org/project/pymupdf4llm/) (there also is an alias [pdf4llm](https://pypi.org/project/pdf4llm/)) is capable of converting PDF pages into **_text strings in Markdown format_** (GitHub compatible). This includes **standard text** as well as **table-based text** in a consistent and integrated view - a feature particularly important in RAG settings.

```bash
$ pip install -U pymupdf4llm
```

> This command will automatically install or upgrade [PyMuPDF](https://github.com/pymupdf/PyMuPDF) if required.

Then in your script do

```python
import pymupdf4llm

md_text = pymupdf4llm.to_markdown("input.pdf")

# now work with the markdown text, e.g. store as a UTF8-encoded file
import pathlib
pathlib.Path("output.md").write_bytes(md_text.encode())
```

Instead of the filename string as above, one can also provide a PyMuPDF `Document`. By default, all pages in the PDF will be processed. If desired, the parameter `pages=[...]` can be used to provide a list of zero-based page numbers to consider.

Markdown text creation now also processes **multi-column pages**.

To create small **chunks of text** - as opposed to generating one large string for the whole document - the new (v0.0.2) option `page_chunks=True` can be used. The result of `.to_markdown("input.pdf", page_chunks=True)` will be a list of Python dictionaries, one for each page.

Also new in version 0.0.2 is the optional **extraction of images** and vector graphics: use of parameter `write_images=True`. The will store PNG images in the document's folder, and the Markdown text will appropriately refer to them. The images are named like `"input.pdf-page_number-index.png"`.

## Documentation and API

[Documentation](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/index.html)

[API](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/api.html#pymupdf4llm-api)

## Document Support

While PDF is by far the most important document format worldwide, it is worthwhile mentioning that all examples and helper scripts work in the same way and **_without change_** for [all supported file types](https://pymupdf.readthedocs.io/en/latest/how-to-open-a-file.html#supported-file-types).

So for an XPS document or an eBook, simply provide the filename for instance as `"input.mobi"` and everything else will work as before.


## About PyMuPDF
**PyMuPDF** adds **Python** bindings and abstractions to [MuPDF](https://mupdf.com/), a lightweight **PDF**, **XPS**, and **eBook** viewer, renderer, and toolkit. Both **PyMuPDF** and **MuPDF** are maintained and developed by [Artifex Software, Inc](https://artifex.com).

PyMuPDF's homepage is located on [GitHub](https://github.com/pymupdf/PyMuPDF).

## Community
Join us on **Discord** here: [#pymupdf](https://discord.gg/TSpYGBW4eq).

## License and Copyright
**PyMuPDF** is available under [open-source AGPL](https://www.gnu.org/licenses/agpl-3.0.html) and commercial license agreements. If you determine you cannot meet the requirements of the **AGPL**, please contact [Artifex](https://artifex.com/contact/pymupdf-inquiry.php) for more information regarding a commercial license.

# PyMuPDF Layout

**PyMuPDF Layout** is a fast and lightweight layout analysis Python package integrated with PyMuPDF for clean, structured data output from PDF. It's fast, accurate and doesn't need GPUs like vision-based models.

While other tools train machine learning models on rendered page images, PyMuPDF Layout trains Graph Neural Networks directly on PDF internals. This gives us accuracy at 10× the speed utilizing CPU-only resources.

[![License PolyForm Noncommercial](https://img.shields.io/badge/license-Polyform_Noncommercial-purple)](https://polyformproject.org/licenses/noncommercial/1.0.0/)
[![Python version](https://img.shields.io/badge/python-3.11+-blue)](https://pypi.org/project/pymupdf-layout/) [![Docs](https://img.shields.io/badge/docs-RTD-green)](https://polyformproject.org/licenses/noncommercial/1.0.0/)
[![Discord](https://img.shields.io/discord/770681584617652264?color=6A7EC2&logo=discord&logoColor=ffffff)](https://discord.gg/ppTFv8uJ46)

## Features

- 📚 Structured data extraction from your documents in Markdown, JSON or TXT format
- 🧐 Advanced document page layout understanding, including semantic markup for titles, headings, headers, footers, tables, images and text styling
- 🔍 Detect and isolate header and footer patterns on each page


## Usage

**PyMuPDF Layout** works alongside PyMuDF4LLM's `to_markdown` method. Once PyMuPDF Layout is activated just use `to_markdown` and PyMuPDF Layout will work behind the scenes to analyse documents and deliver improved results.

You can also get a `JSON` or `TXT` format of the data with `to_json` or `to_text`.

### Extract Structured data

``` python
import pymupdf.layout
import pymupdf4llm
source = "your.pdf"
doc = pymupdf.open(source)
md = pymupdf4llm.to_markdown(doc)
json = pymupdf4llm.to_json(doc)
txt = pymupdf4llm.to_text(doc)
```

## Try It!

Try **PyMuPDF Layout** on [our PyMuPDF website](https://pymupdf.io).

## Documentation

See the [PyMuPDF Layout documentation page](https://pymupdf.readthedocs.io/en/latest/pymupdf-layout/index.html) for more.
