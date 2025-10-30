# PyMuPDF Layout

**PyMuPDF Layout** is a fast and lightweight layout analysis Python package integrated with PyMuPDF for clean, structured data output from PDF. It's fast, accurate and doesn't need GPUs like vision-based models.

While other tools train machine learning models on rendered page images, PyMuPDF Layout trains Graph Neural Networks directly on PDF internals. This gives us accuracy at 10× the speed utilizing CPU-only resources.

[![License PolyForm Noncommercial](https://img.shields.io/badge/license-Polyform_Noncommercial-purple)](https://polyformproject.org/licenses/noncommercial/1.0.0/)
[![Python version](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue)](https://pypi.org/project/pymupdf-layout/)

## Features

- 📚 Structured data extraction from your documents in Markdown, JSON or TXT format
- 🧐 Advanced document page layout understanding, including semantic markup for titles, headings, headers, footers, tables, images and text styling
- 🔍 Detect and isolate header and footer patterns on each page


## Usage

PyMuPDF Layout works alongside PyMuDF4LLM's `to_markdown` method. Once PyMuPDF Layout is activated just use `to_markdown` and PyMuPDF Layout will work behind the scenes to analyse documents and deliver improved results.

You can also get a `JSON` or `TXT` format of the data with `to_json` or `to_text`.

### Extract Structured data

```
import pymupdf.layout
pymupdf.layout.activate()
import pymupdf4llm
doc = pymupdf.open(source)
md = pymupdf4llm.to_markdown(doc)
json = pymupdf4llm.to_json(doc)
txt = pymupdf4llm.to_text(doc)
```

## Try It!

Try **PyMuPDF Layout** on [our PyMuPDF website](https://pymupdf.io).

