import pymupdf
import pymupdf4llm


def test_pymupdf_5030():
    # Eight short text fragments scattered like an OCR'd slide. The layout model
    # reads the region as a table, but the grid finder extracts no cells from it.
    PLACEMENTS = [
        (84, 620, "Cost", 10),  (214, 280, "Net", 12),
        (88, 505, "12%", 9),    (213, 378, "Margin", 11),
        (130, 245, "Margin", 10), (373, 156, "South", 8),
        (67, 222, "North", 11), (140, 475, "3.4", 11),
    ]

    doc = pymupdf.open()
    page = doc.new_page()  # default A4
    for x, y, text, size in PLACEMENTS:
        page.insert_text((x, y), text, fontsize=size)

    page.get_layout()
    tables = page.find_tables()
    print("tables found:", len(tables.tables))
    for t in tables.tables:
        print("cells:", len(t.cells))
        print("bbox:", t.bbox)   # <-- raises ValueError for the zero-cell table    
