import pymupdf as fitz

from extract_text.pymupdf_util.multi_column import column_boxes


def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    page = doc[0]
    columns = column_boxes(page, footer_margin=50, no_image_text=True)
    text = [page.get_text(clip=col) for col in columns]
    linearized = "\n".join(text)

    return linearized
