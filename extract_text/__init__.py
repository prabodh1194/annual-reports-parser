import pymupdf as fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)  # Open PDF
    text = ""

    page: fitz.Page
    for page in doc:
        page_text = page.get_text("text", sort=True)
        text += page_text + "\n"  # Extract text from each page

    return text.strip()
