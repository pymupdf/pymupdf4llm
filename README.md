This is an empty package that is required by pymupdf-2.x, in order to avoid
a pip bug that can result in a broken installation when upgrading from
pymupdf-1.x to pymupdf-2.x.

[The pymupdf-2.x package now contains the 4llm code that used to be in
pymupdf4llm-1.x.]

## Progress callback in the embedded 4LLM source

The 4LLM source in `src/` is embedded by PyMuPDF builds. Its `to_markdown()`
accepts an optional callback for GUI or batch-job progress. The callback receives
the number of selected pages completed (starting at 1) and the selected page total:

```python
import pymupdf4llm

markdown = pymupdf4llm.to_markdown(
    "document.pdf",
    progress_callback=lambda done, total: print(f"{done}/{total} pages"),
)
```
