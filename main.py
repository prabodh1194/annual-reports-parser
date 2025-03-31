import clean_and_preprocess
import extract_text

pdf_path = "/Users/pbd/ar/2024/dixon.pdf"

pdf_to_text = extract_text.extract_text_from_pdf(pdf_path)
cleaned_text = clean_and_preprocess.clean_text(pdf_to_text)
chunks = clean_and_preprocess.chunk_text(cleaned_text)
