<p align="center">
  <a href="https://github.com/pymupdf/pymupdf4llm)">
    <img loading="lazy" alt="PyMuPDF logo" src="https://pymupdf.readthedocs.io/en/latest/_static/sidebar-logo-light.svg" width="150px" height='auto' />
  </a>
</p>

# PyMuPDF4LLM

[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)
[![License MIT](https://img.shields.io/badge/license-AGPL-green)](https://artifex.com/licensing/gnu-agpl-v3)
[![PyPI Downloads](https://static.pepy.tech/badge/pymupdf4llm/month)](https://pepy.tech/projects/pymupdf4llm)
[![Discord](https://img.shields.io/discord/1460622234811895872?color=6A7EC2&logo=discord&logoColor=ffffff)](https://discord.gg/7pH3gqcRtg)


**PyMuPDF4LLM** is a lightweight extension for **PyMuPDF** that turns documents into clean, structured data with minimal setup. It includes layout analysis *without* any GPU requirement.


**PyMuPDF4LLM** makes it easy to extract document content in the format you need for **LLM** & **RAG** environments. It supports structured data extraction to **Markdown**, **JSON** and **TXT** , as well as [LlamaIndex](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/index.html#with-llamaindex) and [LangChain](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/index.html#with-langchain) integration.



## Features

- Parsing of [multiple document formats](https://pymupdf.readthedocs.io/en/latest/about.html#feature-matrix).
- Export structured data as Markdown, JSON and plain text output formats.
- Support for multi-column pages.
- Support for image and vector graphics extraction.
- Layout analysis for better semantic understanding of document structure.
- Support for page chunking output.
- Integration with popular AI frameworks.


## Installation

```bash
$ pip install -U pymupdf4llm
```

> This command will automatically install or upgrade [PyMuPDF](https://github.com/pymupdf/PyMuPDF) as required.


## Execution


### Markdown

```python
import pymupdf4llm

# The remainder of the script is unchanged
md_text = pymupdf4llm.to_markdown("input.pdf")

# now work with the output data, e.g. store as a UTF8-encoded file
import pathlib
pathlib.Path("output.md").write_bytes(md_text.encode())
```


### JSON

```python
import pymupdf4llm

json_text = pymupdf4llm.to_json("input.pdf")

# now work with the output data, e.g. store as a UTF8-encoded file
import pathlib
pathlib.Path("output.json").write_bytes(json_text.encode())
```

### Plain Text

```python
import pymupdf4llm

plain_text = pymupdf4llm.to_text("input.pdf")

# now work with the output data, e.g. store as a UTF8-encoded file
import pathlib
pathlib.Path("output.txt").write_bytes(plain_text.encode())
```


## Documentation

Check out the [PyMuPDF4LLM  documentation](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm), for details on installation, features, sample code and the [full API](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/api.html).

## Examples

Find our [examples on Github](https://github.com/pymupdf/pymupdf4llm/tree/main/examples).

## Integrations

For your AI application development, check out our 
[integrations](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/index.html#integrations) with popular frameworks.

## Support

You can get support for PyMuPDF4LLM via a number of options:

- [Github Issue Board](https://github.com/pymupdf/pymupdf4llm/issues)
- [Discord](https://discord.gg/7pH3gqcRtg)
- [MuPDF Forum](https://forum.mupdf.com)


    