import pathlib

import pymupdf4llm


def test_markdown_to_pdf():
    """Use a pre-existing MD file and generate a PDF from it.
    Then convert the PDF back to MD and check that the content is the same.
    """
    # Read pre-existing MD file.
    old_md_path = pathlib.Path(
        pathlib.Path(__file__).parent.parent
        / "tests"
        / "test_markdown_to_pdf-expected.md"
    )
    old_md = old_md_path.read_bytes().decode()

    # Make new PDF from old MD text
    new_pdf = pymupdf4llm.markdown_to_pdf(
        old_md_path, output_path="test_markdown_to_pdf.pdf"
    )
    # Convert the new PDF back to MD.
    new_md = pymupdf4llm.to_markdown("test_markdown_to_pdf.pdf", use_ocr=False)

    # Compare old and new MD content.
    assert old_md.strip() == new_md.strip()
