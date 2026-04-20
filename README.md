<p align="center">
  <a href="https://github.com/pymupdf/PyMuPDF/">
    <img loading="lazy" alt="PyMuPDF" src="https://pymupdf.pro/images/py-mupdf4llm-github-icon.png" width="100px"/>
  </a>
</p>

# PyMuPDF4LLM

<p align="center">
 <a href="https://trendshift.io/repositories/11536" target="_blank"><img src="https://trendshift.io/api/badge/repositories/11536" alt="pymupdf%2FPyMuPDF | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>


[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm)
[![PyPI Version](https://img.shields.io/pypi/v/pymupdf4llm?color=blue&label=PyPI)](https://pypi.org/project/pymupdf4llm)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pymupdf4llm)](https://pypi.org/project/pymupdf4llm/)
[![License AGPL](https://img.shields.io/github/license/pymupdf/pymupdf4llm)](https://github.com/pymupdf/pymupdf4llm/blob/master/COPYING)
[![PyPI Downloads](https://static.pepy.tech/badge/pymupdf4llm/month)](https://pepy.tech/projects/pymupdf4llm)
[![Github Stars](https://img.shields.io/github/stars/pymupdf/pymupdf4llm?style=social)](https://github.com/pymupdf/pymupdf4llm/stargazers)
[![Discord](https://img.shields.io/discord/1460622234811895872?color=6A7EC2&logo=discord&logoColor=ffffff)](https://pymupdf.io/discord/pdf4llm/)
[![Forum](https://img.shields.io/badge/Forum-ff6600?logo=python&logoColor=ffffff)](https://forum.mupdf.com/c/general/4)
[![Twitter](https://img.shields.io/twitter/follow/pymupdf4llm)](https://x.com/pymupdf4llm)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-007ec6)](https://huggingface.co/artifex-software)
[![Demo](https://img.shields.io/badge/PyMuPDF4LLM-live?badge&label=DEMO&logo=python&logoColor=ffffff)](https://demo.pymupdf.io)

**Turn PDF and other documents into clean, LLM-ready data — in one line of code. No GPU, no Cloud, no Tokens required.**

PyMuPDF4LLM is a lightweight extension for [PyMuPDF](https://github.com/pymupdf/PyMuPDF) that converts documents into structured Markdown, JSON, and plain text optimised for RAG pipelines, vector embeddings, and LLM ingestion. It handles multi-column layouts, tables, images, headers, and scanned pages with automatic OCR — all powered by the MuPDF C engine.

```python
import pymupdf4llm

md = pymupdf4llm.to_markdown("research-paper.pdf")
# Feed directly into your LLM, vector store, or chunker
```

[![Star on GitHub](https://img.shields.io/github/stars/pymupdf/pymupdf4llm.svg?style=for-the-badge&label=Star&logo=github)](https://github.com/pymupdf/pymupdf4llm/)

[![Demo](https://img.shields.io/badge/Pymupdf4llm-live?style=for-the-badge&label=DEMO&logo=python&logoColor=ffffff)](https://demo.pymupdf.io)

---

## Why PyMuPDF4LLM?

- **One import, three output formats** — Markdown, JSON, and plain text out of the box
- **No GPU, no cloud** — runs on any machine that can run Python
- **Layout-aware** — multi-column pages, reading-order reconstruction, table detection
- **Smart OCR** — automatically OCRs only the regions that need it, skipping clean text
- **Framework integrations** — drop-in support for LlamaIndex and LangChain
- **Page chunking** — chunk output by page with full metadata per chunk, ready for vector stores
- **10–250× cheaper** than vision-based LLM extraction approaches

---

## Installation

```bash
pip install pymupdf4llm
```

This automatically installs or upgrades PyMuPDF as a dependency. No other mandatory dependencies.


### Optional: Office document support (PyMuPDF Pro)

Extend support to Word, Excel, PowerPoint, and HWP/HWPX by pairing with **PyMuPDF Pro**:

```bash
pip install pymupdfpro
```

---

## Quick start

### Markdown output

```python
import pymupdf4llm

md = pymupdf4llm.to_markdown("document.pdf")
print(md)
```

### JSON output

```python
import pymupdf4llm

data = pymupdf4llm.to_json("document.pdf")
# Returns bounding box info, layout data, and text per element
print(data)
```

### Plain text output

```python
import pymupdf4llm

text = pymupdf4llm.to_text("document.pdf")
print(text)
```

### Save to file

```python
import pymupdf4llm
from pathlib import Path

md = pymupdf4llm.to_markdown("document.pdf")
Path("output.md").write_bytes(md.encode())
```

---

## Features

### Output formats

| Format | API | Best for |
|---|---|---|
| **Markdown** | `to_markdown(path)` | LLM prompts, RAG pipelines, vector embeddings |
| **JSON** | `to_json(path)` | Custom pipelines needing bbox + layout metadata |
| **Plain text** | `to_text(path)` | Search indexing, simple NLP tasks |
| **LlamaIndex docs** | `LlamaMarkdownReader().load_data(path)` | Direct LlamaIndex integration |

### Extraction capabilities

| Feature | Description |
|---|---|
| **Layout analysis** | Reconstructs natural reading order across single and multi-column pages |
| **Table detection** | Finds and converts tables to GitHub-compatible Markdown |
| **Header detection** | Maps font sizes to `#` heading levels; customisable via `IdentifyHeaders` or `TocHeaders` |
| **Inline formatting** | Detects and preserves **bold**, *italic*, `monospace`, and code blocks |
| **Image extraction** | Extracts embedded images and inlines references in Markdown output |
| **Vector graphics** | Detects and includes references to vector graphic elements |
| **Page chunking** | Returns one dict per page with text, tables, images, and full document metadata |
| **Hybrid OCR** | Automatically OCRs only image-covered or illegible regions; skips clean digital text. |
| **Header / footer removal** | Configurable exclusion of repetitive page headers and footers |
| **Selective pages** | Process a subset of pages via the `pages` parameter |
| **TOC-driven headers** | Use the document's table of contents to derive heading hierarchy |


### Hybrid OCR Strategy

PyMuPDF4LLM applies OCR selectively — only where it is actually needed. Rather than blindly sending every page through an OCR engine (slow and potentially counterproductive on clean text), or naively skipping OCR on mixed documents (leaving scanned regions unreadable), it analyses each page first and makes a targeted decision. This selective approach typically reduces OCR processing time by around 50%.

#### How it works

Before any text extraction, PyMuPDF4LLM inspects every page using a set of heuristics to decide which OCR mode, if any, to apply. There are three possible outcomes:

**No OCR** — the page contains sufficient, legible digital text. Native extraction runs at full speed; OCR is skipped entirely.

**Full-page OCR** — the page has little or no selectable text, and most of its area is covered by images or character-sized vector graphics (the signature of a scanned document). The entire page is rasterised and passed to the OCR engine.

**Span-level repair OCR** — the page contains text, but a significant number of characters are unreadable (rendered as `?` or `\ufffd` replacement characters — a symptom of missing or corrupt font encoding). Only the affected spans are rasterised and OCR'd; the readable portions of the page are left untouched.

The result of all three paths is merged into a single, seamless output. There is no distinction in the Markdown between pages extracted natively and pages recovered via OCR.

#### Why it matters

OCR is roughly 1,000× slower than native text extraction. Applying it indiscriminately to a large document is expensive, and applying full-page OCR on top of already-readable text can actually *degrade* output quality by introducing recognition errors. The hybrid approach avoids both problems:

- Reduces OCR processing time by around **50%** compared to full-document OCR
- Preserves the precision of native digital text extraction where the text layer is clean
- Recovers only what is broken, leaving surrounding content intact

#### OCR triggers

Two situations cause OCR to be invoked automatically:

1. **No text at all** — the page is image-covered with no selectable content. PyMuPDF4LLM also checks image quality heuristics to distinguish a scanned text page from a photograph, avoiding wasted OCR effort on pages that contain no readable text regardless.
2. **Garbled text** — the page has a text layer, but too many characters are unreadable. Only the broken spans are targeted, not the full page.

#### Configuration

The default behaviour requires no configuration — just install Tesseract and it works:

```python
import pymupdf4llm

# OCR is triggered automatically wherever needed
md = pymupdf4llm.to_markdown("mixed-document.pdf")
```

For cases where you need more control:

```python
# Force OCR on every page (e.g. known-corrupt text layer)
md = pymupdf4llm.to_markdown("document.pdf", force_ocr=True)

# Force OCR on specific pages only
md = pymupdf4llm.to_markdown("document.pdf", pages=[2, 3, 4], force_ocr=True)

# Disable OCR entirely (pages with no text will return empty strings)
md = pymupdf4llm.to_markdown("document.pdf", use_ocr=False)

# Set OCR resolution (default 300 dpi; higher values cost quadratically more)
md = pymupdf4llm.to_markdown("document.pdf", ocr_dpi=150)

# Specify OCR language
md = pymupdf4llm.to_markdown("document.pdf", ocr_language="eng+fra")

# Bring your own OCR function
md = pymupdf4llm.to_markdown("document.pdf", ocr_function=my_ocr_fn)
```

> **Note:** `force_ocr=True` on a clean, text-based PDF will slow processing significantly and may reduce output quality. Use it only when you have reason to distrust the native text layer.

#### OCR engine selection

PyMuPDF4LLM automatically selects the best available OCR engine at runtime — no manual configuration needed. It supports Tesseract (via PyMuPDF's built-in integration) and `rapidocr_onnxruntime`, choosing whichever is installed. If neither is available and OCR is required, an exception is raised with installation instructions.


Find out more with the full [PyMuPDF4LLM OCR documentation](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/index.html#ocr)


### Framework integrations

| Framework | Method |
|---|---|
| **LlamaIndex** | `pymupdf4llm.LlamaMarkdownReader().load_data("doc.pdf")` |
| **LangChain** | `from langchain_community.document_loaders import PyMuPDFLoader` |
| **LangChain + chunking** | `MarkdownTextSplitter` on `to_markdown()` output |

---

## Usage examples

### Page chunking for RAG

```python
import pymupdf4llm

chunks = pymupdf4llm.to_markdown("document.pdf", page_chunks=True)

for chunk in chunks:
    print(chunk["metadata"]["page"])   # page number
    print(chunk["metadata"]["title"])  # document title
    print(chunk["text"])               # markdown text for this page
    print(chunk["tables"])             # tables found on this page
    print(chunk["images"])             # images found on this page
```

Each chunk contains full document metadata alongside the page content — ready to insert into a vector store.

### LlamaIndex integration

```python
import pymupdf4llm

reader = pymupdf4llm.LlamaMarkdownReader()
docs = reader.load_data("document.pdf")

# docs is a list of LlamaIndex Document objects
for doc in docs:
    print(doc.text)
```

### LangChain integration

```python
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import MarkdownTextSplitter
import pymupdf4llm

# Option A — via LangChain loader
loader = PyMuPDFLoader("document.pdf")
pages = loader.load()

# Option B — via to_markdown + splitter
md = pymupdf4llm.to_markdown("document.pdf")
splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.create_documents([md])
```

### Extract specific pages

```python
import pymupdf4llm

# Only extract pages 0, 1, and 5
md = pymupdf4llm.to_markdown("document.pdf", pages=[0, 1, 5])
```

### Extract images alongside text

```python
import pymupdf4llm

md = pymupdf4llm.to_markdown(
    "document.pdf",
    write_images=True,        # save extracted images to disk
    image_path="./images",    # directory for saved images
    image_format="png",       # output format
    dpi=150,                  # image resolution
)
```

### Custom header detection

```python
import pymupdf
import pymupdf4llm

doc = pymupdf.open("document.pdf")

# Automatic: scan font sizes to determine heading levels
headers = pymupdf4llm.IdentifyHeaders(doc, max_levels=3)
md = pymupdf4llm.to_markdown(doc, hdr_info=headers)

# TOC-driven: use the document's table of contents
toc_headers = pymupdf4llm.TocHeaders(doc)
md = pymupdf4llm.to_markdown(doc, hdr_info=toc_headers)

# Custom callable: full control over heading logic
def my_headers(span, page=None):
    if span["size"] > 16:
        return "# "
    elif span["size"] > 12:
        return "## "
    return ""

md = pymupdf4llm.to_markdown(doc, hdr_info=my_headers)
```

### Automatic OCR for scanned documents

```python
import pymupdf4llm

# OCR is triggered automatically for pages with no selectable text.
# No configuration needed — just install Tesseract language packs as required.
md = pymupdf4llm.to_markdown("scanned-report.pdf")
```

---

## Output format reference

### Markdown (`to_markdown`)

GitHub-compatible Markdown with:

- `#` – `######` headings derived from font size hierarchy
- `**bold**`, `*italic*`, `` `monospace` `` inline formatting
- Fenced code blocks for detected code spans
- GFM pipe tables for detected table regions
- `![alt](path)` image references for extracted images
- Ordered and unordered lists

### JSON (`to_json`)

Structured output containing bounding box coordinates, layout element types, font metadata, and text content for every detected element on each page — useful for building custom rendering or retrieval pipelines.

### Page chunks (with `page_chunks=True`)

Each page is returned as a dict:

```python
{
    "metadata": {
        "format": "PDF 1.7",
        "title": "...",
        "author": "...",
        "page": 3,
        "page_count": 42,
        "file_path": "document.pdf",
        # ...
    },
    "toc_items": [[2, "Section Title", 3], ...],
    "text": "## Section Title\n\nBody text...",
    "tables": [...],
    "images": [...],
    "graphics": [...],
}
```

---

## Supported document formats

| Format | Notes |
|---|---|
| **PDF** | Full support including scanned pages (via OCR) |
| **XPS / OXPS** | Text and image extraction |
| **EPUB / MOBI / FB2** | Chapter-aware extraction |
| **Images** (PNG, JPG, TIFF…) | Single-page extraction with optional OCR |
| **Office** (DOCX, XLSX, PPTX, HWP) | Requires **PyMuPDF Pro** |

---

## Performance

PyMuPDF4LLM is built on MuPDF — a best-in-class C rendering engine — and requires no GPU. Compared to vision-based LLM extraction:

- **10× faster** on standard cloud instances
- **Up to 250× lower** infrastructure cost
- **Matches or exceeds** vision-LLM accuracy on table detection
- Smart OCR processes only the regions that need it, reducing OCR time by ~50%

---

## Recipes

<details>
<summary>Index a document into a vector store (Chroma)</summary>

```python
import pymupdf4llm
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

chunks = pymupdf4llm.to_markdown("document.pdf", page_chunks=True)

client = chromadb.Client()
collection = client.create_collection(
    "docs",
    embedding_function=SentenceTransformerEmbeddingFunction(),
)

collection.add(
    documents=[c["text"] for c in chunks],
    metadatas=[c["metadata"] for c in chunks],
    ids=[f"page-{c['metadata']['page']}" for c in chunks],
)
```
</details>

<details>
<summary>Process multiple documents in a loop</summary>

```python
import pymupdf4llm
from pathlib import Path

docs_dir = Path("./documents")
all_chunks = []

for pdf in docs_dir.glob("*.pdf"):
    chunks = pymupdf4llm.to_markdown(str(pdf), page_chunks=True)
    all_chunks.extend(chunks)

print(f"Total chunks: {len(all_chunks)}")
```
</details>

<details>
<summary>Pass a PyMuPDF Document object directly</summary>

```python
import pymupdf
import pymupdf4llm

doc = pymupdf.open("document.pdf")

# Pre-process pages however you like, then extract
md = pymupdf4llm.to_markdown(doc)
```
</details>

<details>
<summary>OCR options</summary>

```python
# Force OCR on every page (e.g. known-corrupt text layer)
md = pymupdf4llm.to_markdown("document.pdf", force_ocr=True)

# Force OCR on specific pages only
md = pymupdf4llm.to_markdown("document.pdf", pages=[2, 3, 4], force_ocr=True)

# Disable OCR entirely (pages with no text will return empty strings)
md = pymupdf4llm.to_markdown("document.pdf", use_ocr=False)

# Set OCR resolution (default 300 dpi; higher values cost quadratically more)
md = pymupdf4llm.to_markdown("document.pdf", ocr_dpi=150)

# Specify OCR language
md = pymupdf4llm.to_markdown("document.pdf", ocr_language="eng+fra")

# Bring your own OCR function
md = pymupdf4llm.to_markdown("document.pdf", ocr_function=my_ocr_fn)
```
</details>

---

## Documentation

Full API reference, guides, and examples at **[pymupdf.readthedocs.io/en/latest/pymupdf4llm](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)**.

- [Getting started](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)
- [Full API reference](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/api.html)
- [LLM & RAG guide](https://pymupdf.readthedocs.io/en/latest/rag.html)
- [Examples on GitHub](https://github.com/pymupdf/pymupdf4llm/tree/main/examples)

---

## Related projects

| Project | Description |
|---|---|
| [PyMuPDF](https://github.com/pymupdf/PyMuPDF) | The core library — low-level PDF manipulation, rendering, annotation |
| [PyMuPDF Pro](https://pymupdf.io/pro) | Adds Office and HWP document support |
| [pymupdf-fonts](https://pypi.org/project/pymupdf-fonts/) | Extended font collection for PyMuPDF text output |

---

## Licensing

PyMuPDF4LLM and PyMuPDF are maintained by [Artifex Software, Inc.](https://artifex.com)

- **Open source** — [GNU AGPL v3](https://www.gnu.org/licenses/agpl-3.0.html). Free for open-source projects.
- **Commercial** — separate licences available from [Artifex](https://artifex.com/licensing) for proprietary applications.

---

## Contributing

Contributions are welcome. Please open an issue before submitting large pull requests.

- [Issue tracker](https://github.com/pymupdf/pymupdf4llm/issues)
- [Discord](https://pymupdf.pro/discord/4llm/)


## ⭐ Support this project

If you find this useful, please consider giving it a star — it helps others discover it!

[![Star on GitHub](https://img.shields.io/github/stars/pymupdf/pymupdf4llm.svg?style=for-the-badge&label=Star&logo=github)](https://github.com/pymupdf/pymupdf4llm/)



