# Using PyMuPDF as a Data Feeder in LLM / RAG Applications

This package converts the pages of a PDF to text in Markdown format using [PyMuPDF](https://pypi.org/project/PyMuPDF/).

Standard text and tables are detected, brought in the right reading sequence and then together converted to GitHub-compatible Markdown text.

Header lines are identified via the font size and appropriately prefixed with one or more '#' tags.

Bold, italic, mono-spaced text and code blocks are detected and formatted accordingly. Similar applies to ordered and unordered lists.

By default, all document pages are processed. If desired, a subset of pages can be specified by providing a sequence of 0-based page numbers.

-----

[PyMuPDF-Layout](https://pypi.org/project/pymupdf-layout/) is an optional extension of PyMuPDF. It offers AI-based improved page layout analysis, for instance entailing a much higher table recognition.

Since version 0.2.0, pymupdf4llm fully supports pymupdf-layout. As part of this, output as plain text or a JSON string is also possible. In addition, every page is automatically OCR'd (based on a number of criteria) provided package [opencv-python](https://pypi.org/project/opencv-python/) is installed and Tesseract is available on the platform.

Layout mode is activated with a simple modification of the import statements - for details, please see below.

# Installation

```bash
$ pip install -U pymupdf4llm
```

> This command will automatically install or upgrade [PyMuPDF](https://github.com/pymupdf/PyMuPDF) as required.

To install all Python packages for full support of the layout feature and automatic OCR, you can use the following command version:

```bash
$ pip install -U pymupdf4llm[ocr,layout]
```

This will install opencv-python and pymupdf-layout in addition to pymupdf4llm and pymupdf.

# Execution
## Legacy Mode
For **_standard (legacy) markdown extraction_**, use the following simple script

```python
import pymupdf4llm

md_text = pymupdf4llm.to_markdown("input.pdf")

# now work with the markdown text, e.g. store as a UTF8-encoded file
import pathlib
pathlib.Path("output.md").write_bytes(md_text.encode())
```

Instead of the filename string as above, one can also provide a PyMuPDF `Document`.

By default, all pages in the PDF will be processed. If desired, the parameter `pages=<sequence>` can be used to provide a sequence of zero-based page numbers to consider.

## Layout Mode
To **_activate layout mode_**, use the following

```python
import pymupdf.layout  # activate PyMuPDF-Layout in pymupdf
import pymupdf4llm

# The remainder of the script is unchanged
md_text = pymupdf4llm.to_markdown("input.pdf")

# now work with the markdown text, e.g. store as a UTF8-encoded file
import pathlib
pathlib.Path("output.md").write_bytes(md_text.encode())
```

Here are the JSON and plain text output versions.

### JSON

```python
import pymupdf.layout  # activate PyMuPDF-Layout in pymupdf
import pymupdf4llm

json_text = pymupdf4llm.to_json("input.pdf")

# now work with the markdown text, e.g. store as a UTF8-encoded file
import pathlib
pathlib.Path("output.json").write_text(json_text)
```

### Plain Text

```python
import pymupdf.layout  # activate PyMuPDF-Layout in pymupdf
import pymupdf4llm

plain_text = pymupdf4llm.to_text("input.pdf")

# now work with the markdown text, e.g. store as a UTF8-encoded file
import pathlib
pathlib.Path("output.txt").write_bytes(plain_text.encode())
```


**Feature Overview:**

* Support for pages with **_multiple text columns_**.
* Support for **_image and vector graphics extraction_**:

    1. Specify either `write_images=True` or `embed_images=True`. Default is `False`.
    2. Images and vector graphics on the page will be stored as images named `"input.pdf-pno-index.extension"` in a folder of your choice or be embedded in the markdown text as base64-encoded strings. The image `extension` can be chosen to represent a PyMuPDF-supported image format (for instance "png" or "jpg"),  `pno` is the 0-based page number and `index` is some sequence number.
    3. The image files will have width and height equal to the values on the page. The desired resolution can be chosen via parameter `dpi` (default: `dpi=150`). So this is not an actual **_extraction_** but rather rendering of the respective page area.
    4. Any standard text written in image areas will become a visible part of the generated image and otherwise be ignored. This behavior can be changed via `force_text=True` which causes the text to also become part of the output.

* Support for **page chunks**: Instead of returning one large string for the whole document, a list of dictionaries can be generated: one for each page. Specify `data = pymupdf4llm.to_markdown("input.pdf", page_chunks=True)`. Then, for instance the first item, `data[0]` will contain a dictionary for the first page with its text and some metadata.

* As a first example for directly supporting LLM / RAG consumers, this version can output **LlamaIndex documents**:

    ```python
    import pymupdf4llm
    
    md_read = pymupdf4llm.LlamaMarkdownReader()
    data = md_read.load_data("input.pdf")

    # The result 'data' is of type List[LlamaIndexDocument]
    # Every list item contains metadata and the markdown text of 1 page.
    ```

    * A LlamaIndex document essentially corresponds to Python dictionary, where the markdown text of the page is one of the dictionary values. For instance the text of the first page is the value of `data[0].to_dict().["text"]`.
    * For details, please consult LlamaIndex documentation.
    * Upon creation of the `LlamaMarkdownReader` all necessary LlamaIndex-related imports are executed. Required related package installations must have been done independently and will not be checked during pymupdf4llm installation.
    