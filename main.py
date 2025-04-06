from embeddings import generate_embeddings
from extract_text import extract_text_from_all_pages

pdf_path = "/Users/pbd/ar/2024/dixon.pdf"

extract_text_from_all_pages(pdf_path)
generate_embeddings(
    pdf_path=pdf_path, report_year="2024", company_name="Dixon Technologies"
)
