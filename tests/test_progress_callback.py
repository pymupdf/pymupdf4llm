import pymupdf
import pymupdf4llm
import pytest


@pytest.mark.parametrize("use_layout", [True, False])
def test_to_markdown_reports_selected_page_progress(use_layout):
    document = pymupdf.open()
    for number in range(1, 4):
        page = document.new_page()
        page.insert_text((72, 72), f"PAGE {number} CONTENT")

    progress = []
    previous_layout = pymupdf4llm._use_layout
    try:
        pymupdf4llm.use_layout(use_layout)
        markdown = pymupdf4llm.to_markdown(
            document,
            pages=[0, 2],
            show_progress=not use_layout,
            progress_callback=lambda done, total: progress.append((done, total)),
        )
    finally:
        pymupdf4llm.use_layout(previous_layout)
        document.close()

    assert progress == [(1, 2), (2, 2)]
    assert "PAGE 1 CONTENT" in markdown
    assert "PAGE 3 CONTENT" in markdown
    assert "PAGE 2 CONTENT" not in markdown
