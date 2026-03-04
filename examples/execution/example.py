'''import sys
import time
from pathlib import Path


def ensure_local_import():
    """Allow running the example from the repo without installing the package."""
    repo_root = Path(__file__).resolve().parents[2]
    local_pkg_root = repo_root / "pymupdf4llm"
    sys.path.insert(0, str(local_pkg_root))
    sys.path.insert(1, str(repo_root))
    if "pymupdf4llm" in sys.modules:
        del sys.modules["pymupdf4llm"]
    import pymupdf4llm  # noqa: F401


def read_pdf_to_txt(
    pdf_path: Path,
    output_txt: Path,
    page_number: int | None = None,
    show_progress: bool = False,
) -> None:
    """Read a PDF with PyMuPDF4LLM and write the extracted text to a .txt file."""
    import pymupdf
    import pymupdf4llm as llm

    output_parts = []
    doc = pymupdf.open(str(pdf_path))
    try:
        if page_number is None:
            pages = doc
        else:
            if not 1 <= page_number <= doc.page_count:
                raise ValueError(
                    f"Page must be between 1 and {doc.page_count}, got {page_number}"
                )
            pages = [doc[page_number - 1]]

        if show_progress:
            print("Starting processing with progress enabled...")

        for page in pages:
            chunks = llm.to_markdown(
                page.parent,
                pages=[page.number],
                page_chunks=True,
                table_strategy="lines_strict",
                show_progress=show_progress,
            )

            for chunk in chunks:
                page_number = page.number + 1
                page_text = chunk.get("text_ascii") or chunk.get("text", "")
                page_sections = [f"Page {page_number}", page_text]

                
                output_parts.append("\n".join(page_sections))
    finally:
        doc.close()

    output_txt.write_text("\n\n".join(output_parts), encoding="utf-8")


if __name__ == "__main__":
    ensure_local_import()

    default_pdf = Path(__file__).resolve().parent / "Finerenona_Hinye.pdf"
    pdf_file = Path(sys.argv[1]) if len(sys.argv) > 1 else default_pdf
    out_file = Path(sys.argv[2]) if len(sys.argv) > 2 else pdf_file.with_suffix(".txt")

    print("Choose processing mode:")
    print("1) Entire document")
    print("2) Specific page")
    choice = input("Enter 1 or 2: ").strip()
    if choice == "2":
        page_input = input("Enter page number (1-based): ").strip()
        page_number = int(page_input)
    else:
        page_number = None

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")

    start_time = time.perf_counter()
    read_pdf_to_txt(
        pdf_file,
        out_file,
        page_number=page_number,
        show_progress=True,
    )
    elapsed = time.perf_counter() - start_time
    print(f"Done! Results in: {out_file} (elapsed: {elapsed:.2f}s)")

'''

import pymupdf4llm as llm

text = llm.to_markdown("Finerenona_Hinye.pdf", show_progress=True)

with open("documento.txt", "w", encoding="utf-8") as file:
    file.write(text)