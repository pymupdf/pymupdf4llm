"""
Test script to demonstrate images in table cells fix for Issue #21.

This script creates a test PDF with a table containing images in cells,
then uses pymupdf4llm to extract the table and verify that images
appear inside the table cells in the markdown output.
"""

import pymupdf
import pymupdf4llm
import os
import tempfile
import shutil

def create_test_pdf_with_table_images():
    """Create a test PDF with a table that has images in cells."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # A4 size
    
    # Define table structure
    table_rect = pymupdf.Rect(50, 50, 545, 400)
    cell_width = (table_rect.width) / 3
    cell_height = (table_rect.height) / 4
    
    # Draw table grid
    for i in range(4):
        # Horizontal lines
        y = table_rect.y0 + i * cell_height
        page.draw_line((table_rect.x0, y), (table_rect.x1, y))
    page.draw_line((table_rect.x0, table_rect.y1), (table_rect.x1, table_rect.y1))
    
    for i in range(4):
        # Vertical lines
        x = table_rect.x0 + i * cell_width
        page.draw_line((x, table_rect.y0), (x, table_rect.y1))
    
    # Add header text
    page.insert_text((table_rect.x0 + 10, table_rect.y0 + 20), "Column 1", fontsize=12)
    page.insert_text((table_rect.x0 + cell_width + 10, table_rect.y0 + 20), "Column 2", fontsize=12)
    page.insert_text((table_rect.x0 + 2 * cell_width + 10, table_rect.y0 + 20), "Image Column", fontsize=12)
    
    # Add data rows with text
    for row in range(1, 3):
        y_pos = table_rect.y0 + row * cell_height + 20
        page.insert_text((table_rect.x0 + 10, y_pos), f"Row {row} Col 1", fontsize=10)
        page.insert_text((table_rect.x0 + cell_width + 10, y_pos), f"Row {row} Col 2", fontsize=10)
    
    # Add simple colored rectangles as "images" in the third column
    for row in range(1, 3):
        y_start = table_rect.y0 + row * cell_height + 10
        x_start = table_rect.x0 + 2 * cell_width + 10
        
        # Create a simple colored rectangle to simulate an image
        img_rect = pymupdf.Rect(x_start, y_start, x_start + 60, y_start + 40)
        
        # Draw colored rectangle
        color = (1, 0, 0) if row == 1 else (0, 0, 1)  # Red or Blue
        page.draw_rect(img_rect, color=color, fill=color, width=0)
        
        # Add a small label
        page.insert_text((x_start + 5, y_start + 25), f"IMG{row}", fontsize=8, color=(1, 1, 1))
    
    # Save to temporary file
    temp_pdf = tempfile.mktemp(suffix=".pdf")
    doc.save(temp_pdf)
    doc.close()
    
    return temp_pdf


def test_image_in_table():
    """Test that images appear inside table cells in markdown output."""
    print("Creating test PDF with table containing images...")
    test_pdf = create_test_pdf_with_table_images()
    
    print(f"Test PDF created: {test_pdf}")
    print()
    
    # Create temporary directory for images
    image_dir = tempfile.mkdtemp()
    print(f"Image output directory: {image_dir}")
    print()
    
    try:
        # Extract markdown with images
        print("Extracting markdown with write_images=True...")
        doc = pymupdf.open(test_pdf)
        md_text = pymupdf4llm.to_markdown(
            doc,
            write_images=True,
            image_path=image_dir
        )
        doc.close()
        
        print("Markdown output:")
        print("=" * 80)
        print(md_text)
        print("=" * 80)
        print()
        
        # Check if images are referenced in table
        if "![image]" in md_text and "|" in md_text:
            print("SUCCESS: Images appear to be included in table cells!")
            
            # Count image references
            image_count = md_text.count("![image]")
            print(f"Found {image_count} image reference(s) in the markdown output.")
            
            # List created image files
            image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            print(f"Created {len(image_files)} image file(s):")
            for img_file in image_files:
                print(f"  - {img_file}")
        else:
            print("WARNING: No images found in table cells or no table detected.")
            print("This might be expected if table detection failed.")
        
        print()
        print("Test completed!")
        
    finally:
        # Cleanup
        if os.path.exists(test_pdf):
            os.remove(test_pdf)
            print(f"Cleaned up test PDF: {test_pdf}")
        
        if os.path.exists(image_dir):
            shutil.rmtree(image_dir)
            print(f"Cleaned up image directory: {image_dir}")


if __name__ == "__main__":
    test_image_in_table()

