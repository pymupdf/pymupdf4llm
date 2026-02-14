# Change Log

## Changes in version 0.3.4

### Fixes:

* [356](https://github.com/pymupdf/pymupdf4llm/discussions/356) - Page Chunk Output under to_text() may fail for erroneous layout bboxes

### Other Changes:

* Added support for RapidOCR via a callable plugin.
* Added support for improved OCR via a combination of RapidOCR and Tesseract-OCR.
* Changed default DPI for OCR to 300 (was 400).
* Added new parameter `ocr_function=None`. If not `None` it must be a callable, which is expected to OCR the page giving it a text layer.
* Added new parameter `force_ocr=False` to all extraction functions. Requires to also provide a callable via `ocr_function`. If `True`, `ocr_function` is called for every page, thus skipping the otherwise executed "OCR worthiness" check.

------

## Changes in version 0.2.9

### Fixes:

* [356](https://github.com/pymupdf/pymupdf4llm/discussions/356) - Page Chunk Output under to_text() may fail for erroneous layout bboxes
* [355](https://github.com/pymupdf/pymupdf4llm/issues/355) - Image saving fails if the document filename contains folder specifications

### Other Changes:

* Added new top-level function `get_key_values()` to extract the field name and their values if the document is a "Form PDF". This is always available, whether or not PyMuPDF-Layout is avtive.
* **_Removed_** OpenCV dependency. Previously, this was used to determine whether a page is worthwhile to be OCR'd. We now use numpy for these checks.

------

## Changes in version 0.2.8

### Fixes:

* [349](https://github.com/pymupdf/pymupdf4llm/discussions/349) - Is it possible to change the OCR language when using -layout?
* [352](https://github.com/pymupdf/pymupdf4llm/issues/352) - Does not respect the image_path keyword argument when write_images=True
* [353](https://github.com/pymupdf/pymupdf4llm/issues/353) - How do I filter out pixmap with non-empty size but empty in value

### Other Changes:

* Support new parameter `ocr_language`. This is a string which is passed through to Tesseract-OCR, so the user is responsible for its format.

* Changed the format of the page chunk dictionary: the new dictionary key "page_boxes" in layout mode is now a **_list of dictionaries_** (was a list of lists). The dictionaries have the following keys / values:

    - "index": 0-based integer enumerating the layout boxes in reading order

    - "class": a string denoting the bbox class ("table", "list-item", "section-header", etc.)

    - "bbox": pymupdf.IRect of the layout boundary box

    - "pos": tuple `(start, stop)` of integers denoting the text substring of the bboxes text in this chunk's text ("text" key of the chunk). The values are in slice format and can be used to extract the bbox text like this `bbox_text = chunk["text"][start : stop]`.

* Implemented multiple performance improvements, primarily around rectangle containment checks.
------

## Changes in version 0.2.7

### Fixes:

* [323](https://github.com/pymupdf/pymupdf4llm/issues/323) - `page_chunks=True` parameter was ignored in PyMuPDF-Layout mode

### Other Changes:

* Methods `to_markdown()` / `to_text()` now both support Page chunk output via parameter `page_chunks=True`.

------

## Changes in version 0.2.6

### Fixes:

* [Forum](https://forum.mupdf.com/t/bug-pymupdf4llm-list-index-out-of-range-in-document-layout-py-2/216) - List index out of range ...

### Other Changes:


------

## Changes in version 0.2.5

### Fixes:

* [341](https://github.com/pymupdf/RAG/issues/341) - Broken markdown parsing for new line directly followed by 'o'...

### Other Changes:

* New parameter `table_format` in method `to_text()` (PyMuPDF-Layout only). This allows selecting the appearance of tables in plain text outputs. The possible values are defined in the list `tabulate.tabulate_formats`. Default is "grid".
* Installing PyMuPDF4LLM now supports including all optional dependencies in the `pip` command: `pip install --upgrade pymupdf4llm[ocr,layout]`. This will install pymupdf4llm, pymupdf, and pymupdf-layout. The "ocr" parameter - when needed - installs opencv-python for automatic OCR support in PyMuPDF-Layout mode. Combine this with parameters `--upgrade`, `--force-reinstall` or `--no-cache-dir` as necessary.
* Major rework of the heuristics that determine whether a page should be OCR'd.

------

## Changes in version 0.2.4

### Fixes:

* [335](https://github.com/pymupdf/RAG/issues/335) - KeyError "has_ocr_text"

### Other Changes:


------

## Changes in version 0.2.3

### Fixes:

* [332](https://github.com/pymupdf/RAG/issues/332) - TypeError("to_markdown() got an unexpected keyword argument 'header'")

### Other Changes:

* Output (backend) methods now accept a new parameter `ocr_dpi=400` which sets the OCR resolution for full-page OCR.
* The OCR detection heuristics is more fine-grained and now detects more OCR situations.
* Resolved multiple performance issues, specifically cases with overwhelmingly many images and extremely large `StructTreeRoot` objects in PDF.
* Reflected layout-specific API changes in the legacy code and will now raise `NotImplementedError` exceptions when layout-only features are used.
* Information messages during document parsing are now written to stdout collectively at the end of the phase. This applies to announcing page OCR decisions specifically.
* Support parameter `page_separators` as in the legacy mode.

------

## Changes in version 0.2.1

### Fixes:

* [320](https://github.com/pymupdf/RAG/issues/320) - [Bug] ValueError: min() iterable argument is empty ...
* [319](https://github.com/pymupdf/RAG/issues/319) - [Bug] ValueError: min() arg is an empty sequence 

### Other Changes:

* OCR invocation now differentiates between full-page OCR and text-only OCR: If the page does contain text but the percentage of unreadable characters exceeds a certain threshold (90%), we only OCR text span boundary boxes and replace span text with OCR'ed text where necessary.

------

## Changes in version 0.2.0

This version introduces full support of the [PyMuPDF-Layout](https://pypi.org/project/pymupdf-layout/) package. This entails a radically new approach for detecting the layout of document pages using the AI-based features of the layout package.

Improvements include:

* Greatly improved table detection
* Support of list item hierachy levels
* Detection of page headers and footers
* Improved detection of text paragraphs, titles and section headers
* New output options beyond Markdown: plain text and JSON
* Automatically detect whether a page needs OCR and invoke Tesseract if both, Tesseract is installed and OpenCV (package [opencv-python](https://pypi.org/project/opencv-python/)) is available. Invocation criteria include absence of readable text, full-page coverage with images, presence of many character-sized vector graphics.

The PyMuPDF-Layout package is not open-source and has its own license, which is different from PyMuPDF4LLM. It also is dependent on a number of other, fairly large packages like [onnxruntime](https://pypi.org/project/onnxruntime/), [numpy](https://pypi.org/project/numpy/), [sympy](https://pypi.org/project/sympy/) and [OpenCV](https://pypi.org/project/opencv-python/), which each in turn have their own dependencies.

We therefore keep the use of the layout feature optional. To activate PyMuPDF-Layout support the following import statement **_must be included before_** importing PyMuPDF4LLM itself:

```python
import pymupdf.layout
import pymupdf4llm
```

Thereafter, PyMuPDF's namespace is available. The known method `pymupdf4llm.to_markdown()` automatically works with AI-based empowerment.
In addition, two new methods become available:
* `pymupdf4llm.to_text()` - which works much like markdown output but produces plain text.
* `pymupdf4llm.to_json()` - which outputs the document's metadata and the selected pages in JSON format.

### Fixes:


### Other Changes:

* If `show_progress=True`, Python package [tqdm](https://pypi.org/project/tqdm/) is automatically used when available to display a progress bar. If tqdm is not installed, our own text-based progress bar is used.

------

## Changes in version 0.0.27

### Fixes:

* [296](https://github.com/pymupdf/RAG/issues/296) - [Bug] A specific diagram recognized as significant ...
* [294](https://github.com/pymupdf/RAG/issues/294) - Unable to extract images from Page
* [272](https://github.com/pymupdf/RAG/issues/272) - Disappeared page breaks

### Other Changes:

* Added new parameter to `to_markdown`: `page_separators=False`. If `True` and `page_chunks=False` a line like `--- end of page=nnn ---` is appended to each pages markdown text. The page number is 0-based. Intended for debugging purposes.


## Changes in version 0.0.26

### Fixes:

* [289](https://github.com/pymupdf/RAG/issues/289) - Content Duplication with the latest version
* [275](https://github.com/pymupdf/RAG/issues/275) - Text with background missing from output
* [262](https://github.com/pymupdf/RAG/issues/262) - Markdown error parsing

### Other Changes:

* The table module in package PyMuPDF has been modified: Its method `to_markdown()` will now output markdown-styled cell text. Previously, table cells were extracted as plain text only.

* The class `TocHeaders` is now a top-level import and can now be directly used.

* Method `to_markdown` has a new parameter `detect_bg_color=True` (default) which guesses the page's background color. If a background is detected, fill-only vectors having this color are ignored. `False` will always consider "fill" vectors in vector graphics detection.

* Text written with a `Type 3` font will now always be considered. Previously, this text was always treated as invisible and was hence suppressed.

* The package now contains the license file GNU Affero GPL 3.0 to ease distribution (see LICENSE). It also clarifies that PyMuPDF4LLM is dual licensed under GNU AGPL 3.0 and individual commercial licenses.

* There is a new file `versions_file.py` which contains version information. This is used to ensure the presence of a minimum PyMuPDF version at import time.

## Changes in version 0.0.25

### Fixes:

* [282](https://github.com/pymupdf/RAG/issues/282) - Content Duplication with the latest version
* [281](https://github.com/pymupdf/RAG/issues/281) - Latest version of pymupdf4llm.to_markdown returns empty text for some PDFs.
* [280](https://github.com/pymupdf/RAG/issues/280) - Cannot extract text when ignore_images=False, can extract otherwise.
* [278](https://github.com/pymupdf/RAG/issues/278) - Title words are fragmented
* [249](https://github.com/pymupdf/RAG/issues/249) - Title duplication problem in markdown format
* [202](https://github.com/pymupdf/RAG/issues/202) - BAD RECT ISSUE

### Other Changes:

* The table module in package PyMuDDF has been: Its method `to_markdown()` will now output markdown-styled cell text. Previously, table cells were extracted as plain text only.

* The class `TocHeaders` is now a top-level import and can now be directly used.

* Text written with a `Type 3` font will now always be considered. Previously, this text was always treated as invisible and was hence suppressed.

## Changes in version 0.0.24

### Fixes:

* Fixing "UnboundLocalError"

### Other Changes:



## Changes in version 0.0.23

### Fixes:

* [265](https://github.com/pymupdf/RAG/issues/265) - Code error correction
* [263](https://github.com/pymupdf/RAG/issues/263) - Table Strategy = None raises error
* [261](https://github.com/pymupdf/RAG/issues/261) - wrong markdown in latest pymupdf versions

### Other Changes:

* Highspeed vector graphics count: if `graphics_limit` is specified, drawings are no longer extracted for counting purposes.


## Changes in version 0.0.22

### Fixes:

* [251](https://github.com/pymupdf/RAG/issues/251) - Images a little larger than the page size are being ignored
* [255](https://github.com/pymupdf/RAG/issues/255) - Single-row/column tables are skipped
* [258](https://github.com/pymupdf/RAG/issues/258) - Pymupdf4llm to_markdown crashes on some documents

### Other Changes:

* Added class `TocHeaders` as an alternative way for identifying headers.


## Changes in version 0.0.21

### Fixes:

* [116](https://github.com/pymupdf/RAG/issues/116) - Handling Graphical Images & Superscripts

### Other Changes:


## Changes in version 0.0.20

### Fixes:

* [171](https://github.com/pymupdf/RAG/issues/171) - Text rects overlap with tables and images that should be excluded.
* [189](https://github.com/pymupdf/RAG/issues/189) - The position of the extracted image is incorrect
* [238](https://github.com/pymupdf/RAG/issues/238) - When text is laid out around the picture, text extraction is missing.

### Other Changes:

* Added **_new parameter_** `ignore_images`: (bool) optional. `True` will not consider images in any way. May be useful for pages where a plethora of images prevents meaningful layout analysis. Typical examples are PowerPoint slides and derived / similar pages.

* Added **_new parameter_** `ignore_graphics`: (bool), optional. `True` will not consider graphics except for table detection. May be useful for pages where a plethora of vector graphics prevents meaningful layout analysis. Typical examples are PowerPoint slides and derived / similar pages.

* Added **_new parameter_** to class `IdentifyHeaders`: Use `max_levels` (integer <= 6) to limit the generation of header tag levels. e.g. `headers = pymupdf4llm.IdentifyHeaders(doc, max_level=3)` ensures that only up to 3 header levels will ever be generated. Any text with a font size less than the value of `###` will be body text. In this case, the markdown generation itself would be coded as `md = pymupdf4llm.to_markdown(doc, hdr_info=headers, ...)`.

* Changed parameter `table_strategy`: When specifying `None`, no effort to detecting tables will be made. This can be useful when tables are of no interest or known to not exist in a given file. This will speed up processing significantly. Be prepared to see more changes and extensions here.


## Changes in version 0.0.19

### Fixes:
The following list includes fixes made in version 0.0.18 already.

* [158](https://github.com/pymupdf/RAG/issues/158) - Very long titles when converting to markdown.
* [155](https://github.com/pymupdf/RAG/issues/155) - Inconsistent image extraction from image-only PDFs
* [161](https://github.com/pymupdf/RAG/issues/161) - force_text param ignored.
* [162](https://github.com/pymupdf/RAG/issues/162) - to_markdown isn't outputting all the pages but get_text is.
* [173](https://github.com/pymupdf/RAG/issues/173) - First column of table is repeated before the actual table.
* [187](https://github.com/pymupdf/RAG/issues/187) - Unsolicited Text Particles
* [188](https://github.com/pymupdf/RAG/issues/188) - Takes lot of time to convert into markdown.
* [191](https://github.com/pymupdf/RAG/issues/191) - Extraction of text stops in the middle while working fine with PyMuPDF.
* [212](https://github.com/pymupdf/RAG/issues/212) - In pymupdf4llm, if a page has multiple images, only 1 image per-page is extracted.
* [213](https://github.com/pymupdf/RAG/issues/213) - Many ���� after converting when using pymupdf4llm
* [215](https://github.com/pymupdf/RAG/issues/215) - Spending too much time on identifying text bboxes
* [218](https://github.com/pymupdf/RAG/issues/218) - IndexError in get_raw_lines when processing PDFs with formulas
* [225](https://github.com/pymupdf/RAG/issues/225) - Text with background missing from output.
* [229](https://github.com/pymupdf/RAG/issues/229) - Duplicated Table Content on pymuPDF4LLM.


### Other Changes:

* Added **_new parameter_** `filename`: (str), optional. Overwrites or sets the filename for saved images. Useful when the document is opened from memory.

* Added **_new parameter_** `use_glyphs`: (bool), optional. Request to use the glyph number (if possible) of a character if the font has no back-translation to the original Unicode value. The default is `False` which causes &#xfffd; symbols to be rendered in these cases.

* Added **_strike-out support_**: We now detect and render ~~striked-out text.~~

* Improved **_background color_** detection: We have introduced a simple background color detection mechanism: If a page shows an identical color in all four corners, we assume this to be the background color. Text and vector graphics with this color will be ignored as invisible.

* Improved **_invisible text detection_**: Text with an alpha value of 0 is now ignored.

* Improved **_fake-bold_** detection: Text mimicking bold appearance is now treated like standard bold text in most cases.

* Header handling changes:
    - Detection now happens based on the **_largest font size_** of the line.
    - Uniformly rendered: All spans of a header line will now be rendered with the same appearance.

* Changed handling of parameter `graphics_limit`: We previously ignored a page completely if the vector graphics count exceeded the limit. We now only ignore vector graphics if their count **_outside table boundary boxes_** is too large. This should only suppress vector graphics on the page, while keeping images, text and table content extractable.

* Changed the `margins` default to 0. The previous default `(0, 50, 0, 50)` ignored 50 points at the top and bottom of pages. This has turned out to cause confusion in too many cases.


## Changes in version 0.0.17

### Fixes:


* [147](https://github.com/pymupdf/RAG/issues/147) - Error when page contains nothing but a table.
* [81](https://github.com/pymupdf/RAG/issues/81) - Issues with bullet points in PDFs.
* [78](https://github.com/pymupdf/RAG/issues/78) - multi column pdf file text extraction.


## Changes in version 0.0.15

### Fixes:


* [138](https://github.com/pymupdf/RAG/issues/138) - Table is not extracted and some text order was wrong.
* [135](https://github.com/pymupdf/RAG/issues/135) - Problem with multiple columns in simple text.
* [134](https://github.com/pymupdf/RAG/issues/134) - Exclude images based on size threshold parameter.
* [132](https://github.com/pymupdf/RAG/issues/132) - Optionally embed images as base64 string.
* [128](https://github.com/pymupdf/RAG/issues/128) - Enhanced image embedding format.


### Improvements

* New parameter `embed_images` (bool) **embeds** images and vector graphics in the markdown text as base64-encoded strings. Ignores `write_images` and `image_path` parameters.
* New parameter `image_size_limit` which is a float between 0 and 1, default is 0.05 (5%). Causes images to be ignored if their width or height values are smaller than the corresponding fraction of the page's width or height.
* The algorithm has been improved which determins the sequence of the text rectangles on multi-column pages.
* Change of the header identification algorithm: If more than six header levels are required for a document, then all text with a font size larger than body text is assumed to be a header of level 6 (i.e. HTML "h6" = "###### ").


## Changes in version 0.0.13


### Fixes

* [112](https://github.com/pymupdf/RAG/issues/112) - Invalid bandwriter header dimensions/setup.


### Improvements

* New parameter `ignore_code` suppresses special formatting of text in mono-spaced fonts.
* New parameter `extract_words` enforces `page_chunks=True` and adds a "words" list to each page dictionary.


## Changes in version 0.0.11


### Fixes

* [90](https://github.com/pymupdf/RAG/issues/90) - 'Quad' object has no attribute 'tl'.
* [88](https://github.com/pymupdf/RAG/issues/88) - Bug in `is_significant` function.


### Improvements

* Extended the list of known bullet point characters.


## Changes in version 0.0.10


### Fixes

* [73](https://github.com/pymupdf/RAG/issues/73) - bug in `to_markdown` internal function.
* [74](https://github.com/pymupdf/RAG/issues/74) - minimum area for images & vector graphics.
* [75](https://github.com/pymupdf/RAG/issues/75) - Poor Markdown Generation for Particular PDF.
* [76](https://github.com/pymupdf/RAG/issues/76) - suggestion on useful api parameters.


### Improvements

* Improved recognition of "insignificant" vector graphics. Graphics like text highlights or borders will be ignored.
* The format of saved images can now be controlled via new parameter `image_format`.
* Images can be stored in a specific folder via the new parameter `image_path`.
* Images are **not stored if contained** in another image on same page.
* Images are **not stored if too small:** if width or height are less than 5% of corresponding page dimension.
* All text is always written. If `write_images=True`, text on images / graphics can be suppressed by setting `force_text=False`.


## Changes in version 0.0.9


### Fixes

* [71](https://github.com/pymupdf/RAG/issues/71) - Unexpected results in pymupdf4llm but pymupdf works.
* [68](https://github.com/pymupdf/RAG/issues/68) - Issue with text extraction near footer of page.


### Improvements

* Improved identification of scattered text span particles. This should address most issues with out-of-sequence situations.
* We now correctly process rotated pages (see [issue 68](https://github.com/pymupdf/RAG/issues/68)).


## Changes in version 0.0.8


### Fixes


* [65](https://github.com/pymupdf/RAG/issues/65) - Fix typo in `pymupdf_rag.py`.


## Changes in version 0.0.7


### Fixes


* [54](https://github.com/pymupdf/RAG/issues/54) - Mistakes in orchestrating sentences. Additional fix: text extraction no longer uses the `TEXT_DEHYPHENATE` flag bit.

### Improvements

* Improved the algorithm dealing with vector graphics. Vector graphics are now more reliably classified as irrelevant: We now detect when "strokes" only exist in the neighborhood of the graphics boundary box border itself. This is quite often the case for code snippets.

## Changes in version 0.0.6


### Fixes


* [55](https://github.com/pymupdf/RAG/issues/55) - Bug in helpers/multi_column.py - IndexError: list index out of range.
* [54](https://github.com/pymupdf/RAG/issues/54) - Mistakes in orchestrating sentences.
* [52](https://github.com/pymupdf/RAG/issues/52) - Chunking of text files.
* Partial fix for [41](https://github.com/pymupdf/RAG/issues/41) / [40](https://github.com/pymupdf/RAG/issues/40) - Improved page column detection, but still no silver bullet for overly complex page layouts.

### Improvements

* New parameter `dpi` to specify the resolution of images.
* New parameters `page_width` / `page_height` for easily processing reflowable documents (Text, Office, e-books).
* New parameter `graphics_limit` to avoid spending runtimes for value-less content.
* New parameter `table_strategy` to directly control the table detection strategy.
