try:
    import pymupdf.layout
except ImportError:
    import pymupdf

from .versions_file import MINIMUM_PYMUPDF_VERSION, VERSION

if tuple(map(int, pymupdf.__version__.split("."))) < MINIMUM_PYMUPDF_VERSION:
    raise ImportError(
        f"Requires PyMuPDF v. {MINIMUM_PYMUPDF_VERSION}, but you have {pymupdf.__version__}"
    )

__version__ = VERSION
version = VERSION
version_tuple = tuple(map(int, version.split(".")))

if not callable(pymupdf._get_layout):
    from .helpers.pymupdf_rag import IdentifyHeaders, TocHeaders, to_markdown

    pymupdf._warn_layout_once()  # recommend pymupdf_layout

else:
    from .helpers import document_layout as DL

    def parse_document(
        doc, filename="", image_dpi=150, image_format="png", image_path="", pages=None
    ):
        return DL.parse_document(
            doc,
            filename=filename,
            image_dpi=image_dpi,
            image_format=image_format,
            image_path=image_path,
            pages=pages,
        )

    def to_markdown(
        doc,
        *,
        header=True,
        footer=True,
        pages=None,
        hdr_info=None,
        write_images=False,
        embed_images=False,
        ignore_images=False,
        ignore_graphics=False,
        detect_bg_color=True,
        image_path="",
        image_format="png",
        image_size_limit=0.05,
        filename="",
        force_text=True,
        page_chunks=False,
        page_separators=False,
        margins=0,
        dpi=150,
        page_width=612,
        page_height=None,
        table_strategy="lines_strict",
        graphics_limit=None,
        fontsize_limit=3,
        ignore_code=False,
        extract_words=False,
        show_progress=False,
        use_glyphs=False,
        ignore_alpha=False,
    ):
        parsed_doc = parse_document(
            doc,
            filename=filename,
            image_dpi=dpi,
            image_format=image_format,
            image_path=image_path,
            pages=pages,
        )
        return parsed_doc.to_markdown(
            header=header,
            footer=footer,
            write_images=write_images,
            embed_images=embed_images,
            ignore_code=ignore_code,
        )

    def to_json(doc, image_dpi=150, image_format="png", image_path="", pages=None):
        parsed_doc = parse_document(
            doc,
            image_dpi=image_dpi,
            image_format=image_format,
            image_path=image_path,
            pages=pages,
        )
        return parsed_doc.to_json()

    def to_text(
        doc,
        filename="",
        header=True,
        footer=True,
        pages=None,
        ignore_code=False,
    ):
        parsed_doc = parse_document(
            doc,
            filename=filename,
            image_dpi=150,
            image_format="png",
            image_path="",
            pages=pages,
        )
        return parsed_doc.to_text(
            header=header,
            footer=footer,
            ignore_code=ignore_code,
        )


def LlamaMarkdownReader(*args, **kwargs):
    from .llama import pdf_markdown_reader

    return pdf_markdown_reader.PDFMarkdownReader(*args, **kwargs)
