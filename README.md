This is an empty package that is required by pymupdf-2.x, in order to avoid
a pip bug that can result in a broken installation when upgrading from
pymupdf-1.x to pymupdf-2.x.

[The pymupdf-2.x package now contains the 4llm code that used to be in
pymupdf4llm-1.x.]

To install pymupdf please do `pip install pymupdf`.

Currently the command `pip install pympdf4llm` will also install pymupdf,
but this is deprecated and will be disabled at some point in the future.
