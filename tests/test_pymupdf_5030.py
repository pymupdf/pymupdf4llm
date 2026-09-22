import pymupdf
import pymupdf4llm


def test_pymupdf_5030():
    # Eight short text fragments scattered like an OCR'd slide. The layout model
    # reads the region as a table, but the grid finder extracts no cells from it.
    PLACEMENTS = [
        (84, 620, "Cost", 10),
        (214, 280, "Net", 12),
        (88, 505, "12%", 9),
        (213, 378, "Margin", 11),
        (130, 245, "Margin", 10),
        (373, 156, "South", 8),
        (67, 222, "North", 11),
        (140, 475, "3.4", 11),
    ]

    doc = pymupdf.open()
    page = doc.new_page()  # default A4
    for x, y, text, size in PLACEMENTS:
        page.insert_text((x, y), text, fontsize=size)
    data = doc.tobytes()
    doc = pymupdf.open("pdf", data)
    try:
        md = pymupdf4llm.to_markdown(doc)
        passed = True
    except Exception as e:
        print(f"test_pymupdf_5030(): Exception: {e}")
        passed = False
    assert passed
