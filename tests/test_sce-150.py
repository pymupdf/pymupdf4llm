import textwrap

import pymupdf
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
        header=True,
        footer=True,
    )
    actual = Path(__file__).parent / 'test_sce_150_1.actual.md'
    actual.write_bytes(md.encode())
    assert md == expected


def test_sce_150_2():
    """Table recognition on OCR'd page."""
    filename = Path(__file__).parent / "test_sce_150_2.pdf"
    # Original file in sce issue 150
    # https://github.com/user-attachments/files/29166863/sim_new-york-times_the-new-york-times_2007-11-04_157_contents.pdf
    
    if pymupdf.mupdf_version_tuple >= (1, 29):
        expected = textwrap.dedent('''
                |||SUNDAY,<br>|<br>|NO<br>CO|VEM<br>NTE|BER4, 2007<br>NTS|
                |---|---|---|---|---|---|---|
                |SECTIONS<br>|“HEADI<br>|G<br>|||<br>||
                |2|ARTS<br>|& LEIS<br>|RE<br>|.<br>|<br>.|.|
                ||HOLIDA<br>|YMOVI<br>|ES<br>|.<br>|||
                ||BUSIN<br>|ESS ..<br>|. . <br>|<br>|||
                ||WEEK|IN REVI|EW|.|||
                |4a|EDUCAT<br>|IONLI<br> <br>|FE|.|||
                ||TRAVE|L<br>.|||||
                ||MAGAZ<br>|INE:.<br> <br>|..<br>||||
                ||<br>DESIG|<br> & LIV|<br>ING|<br>|WI|TER .|
                |7|BOOK|REVIEW|.||||
                ||REAL|ESTATE|.|<br>.|||
                ||AUTOM|OBILES|=.|..|.||
                ||NEW|JERSEY|WEE|KLY|||
                ||LONG<br>|ISLAND<br>|WE<br>|EK<br>|LY<br>|.<br>|
                ||WESTC|HESTER|WE|EK|LY|.|
                |14|CONNE|CTICUT|WE|EK|LY|.|
                |14|||||||
                
                
                
                ''').lstrip()
    else:
        expected = (
            (Path(__file__).parent / "test_sce_150_2.expected.md").read_bytes().decode()
        )
        expected = expected.replace('\r', '')   # For github windows.
    md = pymupdf4llm.to_markdown(
        filename,
        write_images=False,
        embed_images=False,
        header=True,
        footer=True,
    )
    actual = Path(__file__).parent / 'test_sce_150_2.actual.md'
    actual.write_bytes(md.encode())
    assert md == expected


def test_sce_150_3():
    """No new OCR if old text layer should be kept."""
    filename = Path(__file__).parent / "test_sce_150_3.pdf"
    # Original file in sce issue 150
    # https://github.com/user-attachments/files/29166852/text_ocr__inr.pdf
    if pymupdf.mupdf_version_tuple >= (1, 29):
        expected = textwrap.dedent('''
                Government of lndia - 

                Rules under which a security deposit amount of 25000 INR is levied on a candidate: 

                # Companies (Acceptance of Deposits) Rules, 2014 

                Provided that if such bonds or debenlures are secured by the charge of any asseis referred to in Schedule lll of the Act, excluding intangible assets, the amount of such bonds or debentures shall not exceed the market value of such assets as assessed by a registered valuer: 

                (x) any amount received from an employee of the company not exceeding his annual salary under a contract of employment with the company in the nature of non-interest i:earing security deposit; 

                {xi) any non-interest bearing amount received or held in trust; 

                (xir) any amount received in the course of, or for the purposes of, the business of the company,- 

                Explanation - For the purposes of this clause, any amount.- 

                {e)<sup>"eligible</sup> company" means a public company as referred to in sub-section (1) of section 76, having a net worth of not less than one hundred crore rupees or a turnover of not less than five hundred crore rupees and which has obtained the prror consent of the company in general meeting by means of a special resolution and also filed the said resolution with the Registrar of Companies before making any invitation to the Public for acceptance of deposits- 

                [File<sup>No. 1181201 3-CL-V]</sup> 

                (Renuka Kumar) 

                Joint Secretary to the Government of lndia 

                ''').lstrip()
    else:
        expected = (
            (Path(__file__).parent / "test_sce_150_3.expected.md").read_bytes().decode()
        )
        expected = expected.replace('\r', '')   # For github windows.
    md = pymupdf4llm.to_markdown(
        filename,
        write_images=False,
        embed_images=False,
        header=True,
        footer=True,
    )

    actual = Path(__file__).parent / 'test_sce_150_3_actual.md'
    actual.write_bytes(md.encode())

    assert md == expected
