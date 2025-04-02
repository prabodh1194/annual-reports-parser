import re

import pymupdf as fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")  # Extract text blocks with coordinates
        blocks.sort(key=lambda b: (b[1], b[0]))  # Sort by Y, then X coordinates

        for block in blocks:
            _text = (
                block[4]
                .strip()
                .replace("\n", " ")
                .replace("\r", " ")
                .replace("\t", " ")
            )
            text = re.sub(r"\s+", " ", _text)  # Normalize whitespace
            if text:
                full_text.append(text)

    return "\n".join(full_text)
