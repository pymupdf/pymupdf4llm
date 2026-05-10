import pathlib

import pymupdf

from .versions_file import VERSION, VERSION_TUPLE

import pymupdf4llm.helpers.pymupdf_rag
import pymupdf4llm.helpers.document_layout

_pvt = tuple(map(int, pymupdf.__version__.split(".")))

if _pvt != VERSION_TUPLE:
    raise ImportError(
        f"Requires PyMuPDF {VERSION=} {VERSION_TUPLE=}, but you have {pymupdf.__version__=} {_pvt=}"
    )

__version__ = VERSION
version = VERSION
version_tuple = tuple(map(int, version.split(".")))


def use_layout(yes):
    global _use_layout
    global IdentifyHeaders
    global TocHeaders

    _use_layout = yes

    if _use_layout:
        # IdentifyHeaders and TocHeaders are not available.
        try:
            del IdentifyHeaders
        except Exception:
            pass
        try:
            del TocHeaders
        except Exception:
            pass
        import pymupdf.layout

        pymupdf.layout.activate()
    else:
        IdentifyHeaders = pymupdf4llm.helpers.pymupdf_rag.IdentifyHeaders
        TocHeaders = pymupdf4llm.helpers.pymupdf_rag.TocHeaders
        import pymupdf

        pymupdf._get_layout = None


# Always attempt to use Layout by default.
try:
    import pymupdf.layout
except ImportError as e:
    use_layout(False)
else:
    use_layout(True)


def _layout_to_markdown(
    doc,
    *,
    dpi=150,
    embed_images=False,
    filename="",
    footer=True,
    force_ocr=False,
    force_text=True,
    header=True,
    ignore_code=False,
    image_format="png",
    image_path="",
    ocr_dpi=300,
    ocr_function=None,
    ocr_language="eng",
    page_chunks=False,
    page_height=None,
    page_separators=False,
    pages=None,
    page_width=612,
    show_progress=False,
    use_ocr=True,
    write_images=False,
    # unsupported options for pymupdf layout:
    **kwargs,
):
    if write_images and embed_images:
        raise ValueError("Cannot both write_images and embed_images")
    parsed_doc = pymupdf4llm.helpers.document_layout.parse_document(
        doc,
        filename=filename,
        image_dpi=dpi,
        image_format=image_format,
        image_path=image_path,
        pages=pages,
        ocr_dpi=ocr_dpi,
        write_images=write_images,
        embed_images=embed_images,
        show_progress=show_progress,
        force_text=force_text,
        use_ocr=use_ocr,
        force_ocr=force_ocr,
        ocr_language=ocr_language,
        ocr_function=ocr_function,
    )
    return parsed_doc.to_markdown(
        header=header,
        footer=footer,
        write_images=write_images,
        embed_images=embed_images,
        ignore_code=ignore_code,
        show_progress=show_progress,
        page_separators=page_separators,
        page_chunks=page_chunks,
    )


def _layout_to_json(
    doc,
    image_dpi=150,
    image_format="png",
    image_path="",
    pages=None,
    ocr_dpi=300,
    write_images=False,
    embed_images=False,
    show_progress=False,
    force_text=True,
    use_ocr=True,
    force_ocr=False,
    ocr_language="eng",
    ocr_function=None,
    # unsupported options for pymupdf layout:
    **kwargs,
):
    parsed_doc = pymupdf4llm.helpers.document_layout.parse_document(
        doc,
        image_dpi=image_dpi,
        image_format=image_format,
        image_path=image_path,
        pages=pages,
        embed_images=embed_images,
        write_images=write_images,
        show_progress=show_progress,
        force_text=force_text,
        use_ocr=use_ocr,
        force_ocr=force_ocr,
        ocr_language=ocr_language,
        ocr_function=ocr_function,
    )
    return parsed_doc.to_json()


def _layout_to_text(
    doc,
    filename="",
    header=True,
    footer=True,
    pages=None,
    ignore_code=False,
    show_progress=False,
    force_text=True,
    ocr_dpi=300,
    use_ocr=True,
    force_ocr=False,
    ocr_language="eng",
    ocr_function=None,
    table_format="grid",
    table_max_width=100,
    table_min_col_width=10,
    page_chunks=False,
    # unsupported options for pymupdf layout:
    **kwargs,
):
    parsed_doc = pymupdf4llm.helpers.document_layout.parse_document(
        doc,
        filename=filename,
        pages=pages,
        embed_images=False,
        write_images=False,
        show_progress=show_progress,
        force_text=force_text,
        use_ocr=use_ocr,
        force_ocr=force_ocr,
        ocr_language=ocr_language,
        ocr_function=ocr_function,
    )
    return parsed_doc.to_text(
        header=header,
        footer=footer,
        ignore_code=ignore_code,
        show_progress=show_progress,
        table_format=table_format,
        table_max_width=table_max_width,
        table_min_col_width=table_min_col_width,
        page_chunks=page_chunks,
    )


def to_markdown(*args, **kwargs):
    if _use_layout:
        return _layout_to_markdown(*args, **kwargs)
    else:
        return pymupdf4llm.helpers.pymupdf_rag.to_markdown(*args, **kwargs)


def to_json(*args, **kwargs):
    if _use_layout:
        return _layout_to_json(*args, **kwargs)
    else:
        return pymupdf4llm.helpers.pymupdf_rag.to_json(*args, **kwargs)


def to_text(*args, **kwargs):
    if _use_layout:
        return _layout_to_text(*args, **kwargs)
    else:
        return pymupdf4llm.helpers.pymupdf_rag.to_text(*args, **kwargs)


def get_key_values(doc, xrefs=False, **kwargs):
    """Extract form fields and their values from a PDF document.

    Args:
        doc: A file path to a PDF document or a pymupdf.Document object.
        xrefs: If True, include the xref numbers of the form fields in the output.
            The xrefs can be useful to directly load a widget via Page.load_widget(xref).
        **kwargs: Additional keyword arguments (currently ignored).
    """
    from .helpers import utils

    if kwargs:
        print(f"Warning: keyword arguments ignored: {set(kwargs.keys())}")
    if isinstance(doc, pymupdf.Document):
        mydoc = doc
    else:
        mydoc = pymupdf.open(doc)
    if mydoc.is_form_pdf:
        rc = utils.extract_form_fields_with_pages(mydoc, xrefs=xrefs)
    else:
        rc = {}

    if mydoc != doc:
        mydoc.close()
    return rc


def markdown_to_pdf(
    md_path,
    user_css=None,
    page_rect=None,
    margins=None,
    image_path=None,
    archive=None,
    output_path=None,
):
    """Return a PDF Document for a given Markdown source string.

    The MD string is converted to HTML using the package 'markdown'. The HTML
    string is then rendered to PDF using PyMuPDF's Story feature.
    The PDF is either returned as a Document object or saved to a file if
    output_path is specified - in this case, None is returned.

    Args:
        md_path: Name of a file containing markdown text.
        user_css: Optional CSS string to style the intermediate HTML source.
            Replaces the default CSS string if specified.
        page_rect: The desired PDF page dimensions (rect-like).
            Default is ISO A4.
        margins: A tuple of four floats representing left, top, right, bottom
            borders in points (1/72 inch). Default is (50, 50, 50, 50).
        image_path: Optional path to a folder containing images referenced in
            the markdown file. If not specified, images will be looked for in
            the same folder as the markdown file.
        archive: Optional Archive object containing the images referenced
            in the markdown file. If not specified, an Archive will be created
            from the path of the markdown file or the image_path if specified.
        output_path: Optional path to save the resulting PDF file.
            If specified, the PDF will be saved to this location and None will
            be returned. Before saving, fonts will be subsetted to reduce
            file size.
    Returns:
        A Document object representing the rendered PDF, or None if
        output_path is specified and the PDF is saved to a file.
        Returning a Document allows for further manipulation or in-memory use
        without writing to disk. Saving to a file is more efficient if the PDF
        is the final output and does not require further processing.
        If output_path is not specified, the caller is responsible for saving
        and closing the returned Document. In this case, using subset_fonts()
        before saving is highly recommended to reduce the file size.
    """
    try:
        import markdown, pymdownx
    except ImportError as e:
        raise ImportError(
            "markdown and pymdownx packages are required for this method. "
            "Install them via 'pip install markdown pymdown-extensions'."
        ) from e

    # desired page dimensions
    MEDIABOX = (
        pymupdf.paper_rect("A4") if page_rect is None else pymupdf.Rect(page_rect)
    )
    try:
        borders = [float(m) for m in margins]
        assert len(borders) == 4
    except Exception as e:
        if margins is not None:
            print(f"Warning: Invalid margins specified: {margins}.")
        borders = (50, 50, 50, 50)

    # available area for writing content.
    WHERE = MEDIABOX + (borders[0], borders[1], -borders[2], -borders[3])

    # Default CSS for styling the intermediate HTML.
    # Can be overridden by user_css argument.
    USER_CSS = """
    /* --- basic layout --- */
    body {
        font-family: sans-serif;
        line-height: 1.3;
        font-size: 12pt;
        color: #000;
        margin: 0 auto;
    }

    /* --- Headers --- */
    h1, h2, h3, h4, h5, h6 {
        font-weight: bold;
        margin-top: 0.6em;
        margin-bottom: 0;
    }

    h1 { font-size: 17pt; }
    h2 { font-size: 15pt; }
    h3 { font-size: 13pt; }

    /* --- Paragraphs --- */
    p {
        margin-top: 0.6em;
        margin-bottom: 0.6em;
    }

    /* --- Lists --- */
    ul, ol {
        margin-top: 0.6em;
        margin-bottom: 0.6em;
    }

    /* --- block quotes --- */
    blockquote {
        border-left: 4px solid #ccc;
        padding-left: 12px;
        color: #555;
        margin: 1em 0;
    }

    /* --- Tables --- */
    table {
        border-collapse: collapse;
        width: auto;
        margin-top: 2em;
        margin-bottom: 2em;
        font-size: 10pt;
    }

    table th, table td {
        border: 1px solid #aaa;
        padding: 3px 3px;
    }

    table th {
        font-weight: bold;
    }

    /* --- Images --- */
    img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 1em auto;
    }

    /* --- Code blocks --- */
    pre code {
        font-family: monospace;
        font-size: 9pt;
        color: #00f;
    }
    code {
        font-family: monospace;
        font-size: 12pt;
    }

    pre {
        /* light blue transparent background for code blocks */
        background: #EFF8;
        padding: 3px;
        overflow-x: visible;/* do not cut text overflowing to the right */
    }

    /* --- Page Breaks (PDF) --- */
    table, pre {
        page-break-inside: avoid;
    }
    """

    def md_to_html(md_text: str) -> str:
        """Convert Markdown to HTML using the 'markdown' package."""
        return markdown.markdown(
            md_text,
            extensions=[
                "extra",  # Tables, Footnotes, Definition Lists, Abbreviations
                "tables",  # explicit table support
                "toc",  # optional: Table of Contents
                "pymdownx.tilde",  # support for strikethrough syntax
            ],
        )

    def rectfn(*args, **kwargs):
        """Return page rectangle and available sub-rectangle."""
        return MEDIABOX, WHERE, None

    if isinstance(md_path, str):
        md_path = pathlib.Path(md_path)
    elif not isinstance(md_path, pathlib.Path):
        raise ValueError("md_path must be a file path or a pathlib.Path")

    md_text = md_path.read_bytes().decode("utf-8")
    if archive is None:
        if image_path is None:
            folder = md_path.resolve().parent
            archive = pymupdf.Archive(folder)
        else:
            archive = pymupdf.Archive(image_path)

    # make a few adjustments to ensure correct processing in PyMuPDF
    md_text = (
        md_text.replace("\r\n", "\n")
        .replace("image/jpg;", "image/jpeg;")
        .replace("**==== Text in picture: ====**", "**==== Text in picture: ====**\n")
    )

    # call the MD-HTML converter
    html_text = md_to_html(md_text)

    css_text = user_css if isinstance(user_css, str) else USER_CSS
    story = pymupdf.Story(html_text, user_css=css_text, archive=archive)

    doc = story.write_with_links(rectfn)

    if isinstance(output_path, (str, pathlib.Path)):
        doc.subset_fonts()
        doc.ez_save(output_path)
        return None
    return doc


def LlamaMarkdownReader(*args, **kwargs):
    from .llama import pdf_markdown_reader

    return pdf_markdown_reader.PDFMarkdownReader(*args, **kwargs)
