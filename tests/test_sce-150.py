import pymupdf4llm
from pathlib import Path


def test_sce_150_1():
    """Correct sequence of MD stylings."""
    filename = Path(__file__).parent / "test_sce_150_1.pdf"
    # Original file in sce issue 150
    # https://github.com/user-attachments/files/29166862/SERFF_CA_random_pages.1_page1434.pdf
    expected = (
        (Path(__file__).parent / "test_sce_150_1.expected.md").read_bytes().decode()
    )
    expected = expected.replace('\r', '')   # For github windows.
    md = pymupdf4llm.to_markdown(
        filename,
        write_images=False,
        embed_images=False,
        header=False,
        footer=False,
    )
    assert md == expected


def test_sce_150_2():
    """Table recognition on OCR'd page."""
    filename = Path(__file__).parent / "test_sce_150_2.pdf"
    # Original file in sce issue 150
    # https://github.com/user-attachments/files/29166863/sim_new-york-times_the-new-york-times_2007-11-04_157_contents.pdf
    expected = (
        (Path(__file__).parent / "test_sce_150_2.expected.md").read_bytes().decode()
    )
    expected = expected.replace('\r', '')   # For github windows.
    md = pymupdf4llm.to_markdown(
        filename,
        write_images=False,
        embed_images=False,
        header=False,
        footer=False,
    )
    assert md == expected


def test_sce_150_3():
    """No new OCR if old text layer should be kept."""
    filename = Path(__file__).parent / "test_sce_150_3.pdf"
    # Original file in sce issue 150
    # https://github.com/user-attachments/files/29166852/text_ocr__inr.pdf
    expected = (
        (Path(__file__).parent / "test_sce_150_3.expected.md").read_bytes().decode()
    )
    expected = expected.replace('\r', '')   # For github windows.
    md = pymupdf4llm.to_markdown(
        filename,
        write_images=False,
        embed_images=False,
        header=False,
        footer=False,
    )
    
    actual = Path(__file__).parent / 'test_sce_150_3_actual.md'
    actual.write_bytes(md.encode())
    
    assert md == expected
