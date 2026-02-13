import pymupdf

from .versions_file import MINIMUM_PYMUPDF_VERSION, VERSION

if tuple(map(int, pymupdf.__version__.split("."))) < MINIMUM_PYMUPDF_VERSION:
    raise ImportError(
        f"Requires PyMuPDF v. {MINIMUM_PYMUPDF_VERSION}, but you have {pymupdf.__version__}"
    )

__version__ = VERSION
version = VERSION
version_tuple = tuple(map(int, version.split(".")))

if pymupdf._get_layout is None:
    from .helpers.pymupdf_rag import (
        IdentifyHeaders,
        TocHeaders,
        to_markdown,
        to_json,
        to_text,
    )

    pymupdf._warn_layout_once()  # recommend pymupdf_layout

else:
    from .helpers import document_layout

    def parse_document(
        doc,
        filename="",
        image_dpi=150,
        image_format="png",
        image_path="",
        ocr_dpi=300,
        pages=None,
        write_images=False,
        embed_images=False,
        show_progress=False,
        force_text=True,
        use_ocr=True,
        force_ocr=False,
        ocr_function=None,
        ocr_language="eng",
    ):
        return document_layout.parse_document(
            doc,
            filename=filename,
            image_dpi=image_dpi,
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
            ocr_function=ocr_function,
            ocr_language=ocr_language,
        )

    def to_markdown(
        doc,
        *,
        header=True,
        footer=True,
        pages=None,
        write_images=False,
        embed_images=False,
        image_path="",
        image_format="png",
        filename="",
        force_text=True,
        page_chunks=False,
        page_separators=False,
        dpi=150,
        ocr_dpi=300,
        page_width=612,
        page_height=None,
        ignore_code=False,
        show_progress=False,
        use_ocr=True,
        ocr_language="eng",
        force_ocr=False,
        ocr_function=None,
        # unsupported options for pymupdf layout:
        **kwargs,
    ):
        if write_images and embed_images:
            raise ValueError("Cannot both write_images and embed_images")
        parsed_doc = parse_document(
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

    def to_json(
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
        parsed_doc = parse_document(
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

    def to_text(
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
        parsed_doc = parse_document(
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


def get_key_values(doc, xrefs=False, **kwargs):
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
