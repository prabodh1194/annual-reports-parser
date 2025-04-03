import os

import PyPDF2
from PyPDF2.errors import PdfReadError

PDF_PATH = "/Users/pbd/ar/2024/dixon.pdf"
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def validate_pdf(file_path: str) -> bool:
    if not os.path.exists(file_path):
        print(f"Error: File not found at path: {file_path}")
        return False
    if not file_path.lower().endswith(".pdf"):
        print("Error: File is not a PDF")
        return False
    return True


def extract_text_from_pdf(file_path: str, max_chars: int = 100000) -> str | None:
    if not validate_pdf(file_path):
        return None

    try:
        with open(file_path, "rb") as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)

            # Get total number of pages
            num_pages = len(pdf_reader.pages)
            print(f"Processing PDF with {num_pages} pages...")

            extracted_text = []
            total_chars = 0

            # Iterate through all pages
            for page_num in range(num_pages):
                # Extract text from page
                page = pdf_reader.pages[page_num]
                text = page.extract_text()

                # Check if adding this page's text would exceed the limit
                if total_chars + len(text) > max_chars:
                    # Only add text up to the limit
                    remaining_chars = max_chars - total_chars
                    extracted_text.append(text[:remaining_chars])
                    print(f"Reached {max_chars} character limit at page {page_num + 1}")
                    break

                extracted_text.append(text)
                total_chars += len(text)
                print(f"Processed page {page_num + 1}/{num_pages}")

            final_text = "\n".join(extracted_text)
            print(f"\nExtraction complete! Total characters: {len(final_text)}")
            return final_text

    except PdfReadError:
        print("Error: Invalid or corrupted PDF file")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None
