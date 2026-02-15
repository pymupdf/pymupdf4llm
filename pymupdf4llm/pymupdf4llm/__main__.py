import argparse
import os
import sys
import json
import contextlib
import pymupdf4llm


def generate_bash_completion():
    return """
_pymupdf4llm_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    opts="-h --help -o --output -m --markdown -j --json -t --text -p --pages --images --image-path --image-format --dpi --no-header --no-footer --no-ocr --force-ocr --ocr-language --page-chunks --page-separators --table-format -v --version --info --bash-completion"

    if [[ ${cur} == -* ]] ; then
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
        return 0
    fi
}
complete -F _pymupdf4llm_completion pymupdf4llm
complete -F _pymupdf4llm_completion pdf4llm
"""


def main():
    parser = argparse.ArgumentParser(description="PyMuPDF Utilities for LLM/RAG")
    parser.add_argument("input", nargs="?", help="Input PDF file")
    parser.add_argument(
        "-o", "--output", help="Output file. If not specified, prints to stdout."
    )
    parser.add_argument(
        "-m", "--markdown", action="store_true", help="Convert to markdown (default)"
    )
    parser.add_argument("-j", "--json", action="store_true", help="Convert to JSON")
    parser.add_argument("-t", "--text", action="store_true", help="Convert to text")
    parser.add_argument(
        "-p",
        "--pages",
        help="Comma-separated list of pages to process (1-based, e.g. 1,3,5-7)",
    )
    parser.add_argument("--images", action="store_true", help="Extract images")
    parser.add_argument("--image-path", help="Path to store extracted images")
    parser.add_argument("--image-format", help="Image format (default: png)")
    parser.add_argument("--dpi", type=int, help="DPI for images (default: 150)")
    parser.add_argument(
        "--no-header",
        action="store_false",
        dest="header",
        help="Do not include headers",
    )
    parser.add_argument(
        "--no-footer",
        action="store_false",
        dest="footer",
        help="Do not include footers",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_false",
        dest="use_ocr",
        help="Do not use OCR",
    )
    parser.add_argument("--force-ocr", action="store_true", help="Force OCR")
    parser.add_argument("--ocr-language", help="OCR language (default: eng)")
    parser.add_argument(
        "--page-chunks",
        action="store_true",
        help="Segment output by page (returns JSON)",
    )
    parser.add_argument(
        "--page-separators",
        action="store_true",
        help="Include page separators (markdown only)",
    )
    parser.add_argument(
        "--table-format",
        help="Table format for text output (default: grid)",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"pymupdf4llm {pymupdf4llm.__version__}",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print document information as JSON and exit",
    )
    parser.add_argument(
        "--bash-completion", action="store_true", help="Generate bash completion script"
    )

    args = parser.parse_args()

    if args.bash_completion:
        print(generate_bash_completion().strip())
        sys.exit(0)

    if not args.input:
        parser.print_help()
        sys.exit(0)

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    import pymupdf

    try:
        doc = pymupdf.open(args.input)
    except Exception as e:
        print(f"Error opening document: {e}", file=sys.stderr)
        sys.exit(1)

    if args.info:
        info = {
            "metadata": doc.metadata,
            "page_count": doc.page_count,
            "file_path": os.path.abspath(args.input),
            "is_encrypted": doc.is_encrypted,
            "is_pdf": doc.is_pdf,
        }
        print(json.dumps(info, indent=2, ensure_ascii=False))
        doc.close()
        sys.exit(0)

    page_count = doc.page_count

    pages = None
    if args.pages:
        pages = []
        for part in args.pages.split(","):
            try:
                if "-" in part:
                    parts = part.split("-")
                    start = int(parts[0])
                    if parts[1].upper() == "N":
                        end = page_count
                    else:
                        end = int(parts[1])
                    pages.extend(range(start - 1, end))
                else:
                    pages.append(int(part) - 1)
            except ValueError:
                print(f"Error: Invalid page specification '{part}'", file=sys.stderr)
                sys.exit(1)

    output_format = "markdown"
    if args.json:
        output_format = "json"
    elif args.text:
        output_format = "text"

    try:
        with contextlib.redirect_stdout(sys.stderr):
            kwargs = {"pages": pages}
            if args.use_ocr is not None:
                kwargs["use_ocr"] = args.use_ocr
            if args.force_ocr is not None:
                kwargs["force_ocr"] = args.force_ocr
            if args.ocr_language is not None:
                kwargs["ocr_language"] = args.ocr_language
            if args.page_chunks is not None:
                kwargs["page_chunks"] = args.page_chunks

            if output_format == "markdown":
                if args.header is not None:
                    kwargs["header"] = args.header
                if args.footer is not None:
                    kwargs["footer"] = args.footer
                if args.images is not None:
                    kwargs["write_images"] = args.images
                if args.image_path is not None:
                    kwargs["image_path"] = args.image_path
                if args.image_format is not None:
                    kwargs["image_format"] = args.image_format
                if args.dpi is not None:
                    kwargs["dpi"] = args.dpi
                if args.page_separators is not None:
                    kwargs["page_separators"] = args.page_separators
                result = pymupdf4llm.to_markdown(doc, **kwargs)

            elif output_format == "json":
                if args.dpi is not None:
                    kwargs["image_dpi"] = args.dpi
                if args.image_format is not None:
                    kwargs["image_format"] = args.image_format
                if args.image_path is not None:
                    kwargs["image_path"] = args.image_path
                if args.images is not None:
                    kwargs["write_images"] = args.images
                result = pymupdf4llm.to_json(doc, **kwargs)

            elif output_format == "text":
                if args.header is not None:
                    kwargs["header"] = args.header
                if args.footer is not None:
                    kwargs["footer"] = args.footer
                if args.table_format is not None:
                    kwargs["table_format"] = args.table_format
                result = pymupdf4llm.to_text(doc, **kwargs)

        if isinstance(result, (list, dict)):
            result = json.dumps(result, indent=2, ensure_ascii=False)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(result)
        else:
            sys.stdout.write(result)
            if not result.endswith("\n"):
                sys.stdout.write("\n")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        doc.close()


if __name__ == "__main__":
    main()
