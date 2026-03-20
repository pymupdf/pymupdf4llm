import pymupdf

TESSDATA = pymupdf.get_tessdata()
if TESSDATA is None:
    pymupdf.message(
        "Warning: Tesseract OCR is not available. No OCR text will be extracted."
    )

REPLACEMENT_UNICODE = chr(0xFFFD)  # Unicode Replacement Character


def ocr_text(span) -> bool:
    if not (span["char_flags"] & 32) and not (span["char_flags"] & 16):
        return True
    return False


def exec_ocr(page, dpi=300, pixmap=None, language="eng", keep_ocr_text=False):
    """This callback function performs OCR on the given page.

    It uses RapidOCR for text region detection and Tesseract OCR for text
    recognition in each identified region (boundary box).

    If a Pixmap is provided, the DPI parameter is ignored. Otherwise, an RGB
    Pixmap is created from the page at the specified DPI.
    The DPI parameter is also used if extractable text is present.

    We ensure that legible extractable text is excluded from OCR. If present
    on page we make a temporary copy without such text and perform OCR
    on that copy.
    """
    if TESSDATA is None:
        return
    text_blocks = page.get_text("dict", flags=pymupdf.TEXT_ACCURATE_BBOXES)["blocks"]
    # get bboxes with significant legible text on page
    spans = []
    fffd_spans = []
    for b in text_blocks:
        for l in b["lines"]:
            for s in l["spans"]:
                if ocr_text(s):
                    if keep_ocr_text:
                        spans.append(s["bbox"])
                    else:
                        fffd_spans.append(s["bbox"])
                    continue
                if REPLACEMENT_UNICODE in s["text"]:
                    fffd_spans.append(s["bbox"])
                else:
                    spans.append(s["bbox"])

    if spans:
        temp_pdf = pymupdf.open()  # create a temporary PDF in memory
        # insert the page
        temp_pdf.insert_pdf(
            page.parent,
            from_page=page.number,
            to_page=page.number,
        )
        temp_page = temp_pdf[0]
        for sbbox in spans:
            # add redaction annotation for each text span
            temp_page.add_redact_annot(sbbox)

        # remove text
        temp_page.apply_redactions(
            images=pymupdf.PDF_REDACT_IMAGE_NONE,
            graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
            text=pymupdf.PDF_REDACT_TEXT_REMOVE,
        )
        # make pixmap from the page with legible text removed
        pixmap = temp_page.get_pixmap(dpi=dpi)

    # make pixmap if not provided
    if pixmap is None:
        pixmap = page.get_pixmap(dpi=dpi)

    if fffd_spans:
        # if there are spans with U+FFFD, we remove them now because their
        # regions will be occupied by OCR text
        for sbbox in fffd_spans:
            page.add_redact_annot(sbbox)
        page.apply_redactions(
            images=pymupdf.PDF_REDACT_IMAGE_NONE,
            graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
            text=pymupdf.PDF_REDACT_TEXT_REMOVE,
        )

    # OCR the (remainder of the) page and remove everything except the text
    # layer from the OCR result
    temp_pdf = pymupdf.open("pdf", pixmap.pdfocr_tobytes(language=language))
    temp_page = temp_pdf[0]
    temp_page.add_redact_annot(temp_page.rect)
    temp_page.apply_redactions(
        images=pymupdf.PDF_REDACT_IMAGE_REMOVE,
        graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
        text=pymupdf.PDF_REDACT_TEXT_NONE,
    )
    # insert the OCR text layer into the original page
    page.show_pdf_page(page.rect, temp_pdf, 0)
