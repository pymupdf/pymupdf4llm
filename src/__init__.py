import pathlib

import pymupdf
import pymupdf4llm.helpers.document_layout
import pymupdf4llm.helpers.pymupdf_rag

from .batch_converter import convert_batch
from .versions_file import VERSION, VERSION_TUPLE

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
    ocr_dpi=150,
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
    render_html_tables=None,
    edge_threshold=None,
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
        render_html_tables=render_html_tables,
        edge_threshold=edge_threshold,
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
    ocr_dpi=150,
    write_images=False,
    embed_images=False,
    show_progress=False,
    force_text=True,
    use_ocr=True,
    force_ocr=False,
    ocr_language="eng",
    ocr_function=None,
    render_html_tables=None,
    edge_threshold=None,
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
        render_html_tables=render_html_tables,
        edge_threshold=edge_threshold,
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
    ocr_dpi=150,
    use_ocr=True,
    force_ocr=False,
    ocr_language="eng",
    ocr_function=None,
    table_format="grid",
    table_max_width=100,
    table_min_col_width=10,
    page_chunks=False,
    edge_threshold=None,
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
        edge_threshold=edge_threshold,
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
    # `render_html_tables` is an internal flag this wrapper injects for
    # table_output="html"; it is not a public kwarg. Drop any user-supplied value
    # so it cannot silently enable/disable HTML tables via **kwargs.
    kwargs.pop("render_html_tables", None)
    if kwargs.get("table_output") == "html":
        # Render tables as HTML via table_html.
        kwargs = dict(kwargs)
        kwargs.pop("table_output", None)
        if _use_layout:
            # Preferred path: render HTML tables on the layout path, reusing the
            # GNN layout. Keeps the layout path's text, reading order, and OCR --
            # only the table rendering is swapped.
            return _layout_to_markdown(*args, render_html_tables=True, **kwargs)
        # No layout engine available: fall back to the rag path, which has its own
        # table_output="html" wiring. OCR is not available on this path.
        legacy_kwargs = dict(kwargs)
        for name in (
            "footer",
            "header",
            "ocr_dpi",
            "ocr_function",
            "ocr_language",
            "use_ocr",
            "force_ocr",
        ):
            legacy_kwargs.pop(name, None)
        return pymupdf4llm.helpers.pymupdf_rag.to_markdown(
            *args, table_output="html", **legacy_kwargs
        )
    if _use_layout:
        return _layout_to_markdown(*args, **kwargs)
    else:
        return pymupdf4llm.helpers.pymupdf_rag.to_markdown(*args, **kwargs)


def to_json(*args, **kwargs):
    # See to_markdown: `render_html_tables` is internal, not a public kwarg.
    kwargs.pop("render_html_tables", None)
    if kwargs.get("table_output") == "html":
        kwargs = dict(kwargs)
        kwargs.pop("table_output", None)
        kwargs["render_html_tables"] = True
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


def LlamaMarkdownReader(*args, **kwargs):
    from .llama import pdf_markdown_reader

    return pdf_markdown_reader.PDFMarkdownReader(*args, **kwargs)


# Engine-internal parse flags; not part of the chunking surface. The HTML
# table opt-in is exposed as table_output="html" like to_markdown, which
# translates to the internal render_html_tables parse flag below.
_CHUNK_INTERNAL_PARSE_FLAGS = {"render_html_tables"}


def _layout_to_chunks(
        doc,
        **kwargs,
    ):
    import inspect

    parse_fn = pymupdf4llm.helpers.document_layout.parse_document
    # Split kwargs into parse_document args and to_chunks args, following
    # the current parse_document signature (it has no **kwargs).
    parse_keys = set(inspect.signature(parse_fn).parameters) - {"doc"}
    parse_keys -= _CHUNK_INTERNAL_PARSE_FLAGS

    internal = _CHUNK_INTERNAL_PARSE_FLAGS & set(kwargs)
    if internal:
        raise TypeError(
            f"internal parse flags not accepted by to_chunks: {sorted(internal)}"
        )

    # table_output selects the table content representation, mirroring
    # to_markdown: "html" routes to the parse-level HTML table engine.
    table_output = kwargs.pop("table_output", "markdown")
    if table_output not in ("markdown", "html"):
        raise ValueError("table_output must be 'markdown' or 'html'")

    # Map external names to parse_document names
    aliases = {"dpi": "image_dpi"}
    parse_kwargs = {}
    chunk_kwargs = {}
    for k, v in kwargs.items():
        k = aliases.get(k, k)
        if k in parse_keys:
            parse_kwargs[k] = v
        else:
            chunk_kwargs[k] = v

    if table_output == "html":
        parse_kwargs["render_html_tables"] = True

    # extract_images is parse-through sugar: it maps to parse_document's
    # embed_images and is not a chunking parameter.
    if chunk_kwargs.pop("extract_images", False) and "embed_images" not in parse_kwargs:
        parse_kwargs["embed_images"] = True

    parsed_doc = parse_fn(doc, **parse_kwargs)
    return parsed_doc.to_chunks(**chunk_kwargs)


def to_chunks(*args, **kwargs):
    if _use_layout:
        return _layout_to_chunks(*args, **kwargs)
    else:
        return pymupdf4llm.helpers.pymupdf_rag.to_chunks(*args, **kwargs)


def to_chunk(*args, **kwargs):
    """Deprecated alias of :func:`to_chunks`."""
    import warnings

    warnings.warn(
        "to_chunk is deprecated; use to_chunks",
        DeprecationWarning,
        stacklevel=2,
    )
    return to_chunks(*args, **kwargs)
