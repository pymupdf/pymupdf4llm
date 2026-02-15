# Using PyMuPDF as Data Feeder in LLM / RAG Applications

This package converts the pages of a PDF to text in Markdown format using [PyMuPDF](https://pypi.org/project/PyMuPDF/).

Standard text and tables are detected, brought in the right reading sequence and then together converted to GitHub-compatible Markdown text.

Header lines are identified via the font size and appropriately prefixed with one or more '#' tags.

Bold, italic, mono-spaced text and code blocks are detected and formatted accordingly. Similar applies to ordered and unordered lists.

By default, all document pages are processed. If desired, a subset of pages can be specified by providing a list of 0-based page numbers.

## Installation

```bash
pip install -U pdf4llm
```

> This command will automatically install [PyMuPDF](https://github.com/pymupdf/PyMuPDF) if required.

Then in your script do:

```python
import pdf4llm

md_text = pdf4llm.to_markdown("input.pdf")

# now work with the markdown text, e.g. store as a UTF8-encoded file
import pathlib
pathlib.Path("output.md").write_bytes(md_text.encode())
```

Instead of the filename string as above, one can also provide a PyMuPDF `Document`. By default, all pages in the PDF will be processed. If desired, the parameter `pages=[...]` can be used to provide a list of zero-based page numbers to consider.

**New features as of v0.0.8:**

* Support for pages with **_multiple text columns_**.
* Support for **_image and vector graphics extraction_**:

    1. Specify `pdf4llm.to_markdown("input.pdf", write_images=True)`. Default is `False`.
    2. Each image or vector graphic on the page will be extracted and stored as a PNG image named `"input.pdf-pno-index.png"` in the folder of `"input.pdf"`. Where `pno` is the 0-based page number and `index` is some sequence number.
    3. The image files will have width and height equal to the values on the page.
    4. Any text contained in the images or graphics will not be extracted, but become visible as image parts.

* Support for **page chunks**: Instead of returning one large string for the whole document, a list of dictionaries can be generated: one for each page. Specify `data = pdf4llm.to_markdown("input.pdf", page_chunks=True)`. Then, for instance the first item, `data[0]` will contain a dictionary for the first page with the text and some metadata.

## Command Line Interface

You can also use pdf4llm (the alias for PyMuPDF4LLM) directly from the command line:

```bash
python -m pdf4llm input.pdf -o output.md
```

The CLI supports various options for output format and features. For a full list of options, run:

```bash
python -m pdf4llm --help
```

* As a first example for directly supporting LLM / RAG consumers, this version can output **LlamaIndex documents**:

    ```python
    import pdf4llm
    
    md_read = pdf4llm.LlamaMarkdownReader()
    data = md_read.load_data("input.pdf")

    # The result 'data' is of type List[LlamaIndexDocument]
    # Every list item contains metadata and the markdown text of 1 page.
    ```

  * A LlamaIndex document essentially corresponds to Python dictionary, where the markdown text of the page is one of the dictionary values. For instance the text of the first page is the the value of `data[0].to_dict().["text"]`.
  * For details, please consult LlamaIndex documentation.
  * Upon creation of the `LlamaMarkdownReader` all necessary LlamaIndex-related imports are executed. Required related package installations must have been done independently and will not be checked during pdf4llm installation.
