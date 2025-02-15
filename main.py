import os

PDF_PATH = "/Users/pbd/ar/2024/dixon.pdf"
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

def validate_pdf(file_path: str) -> bool:
    if not os.path.exists(file_path):
        print(f"Error: File not found at path: {file_path}")
        return False
    if not file_path.lower().endswith('.pdf'):
        print("Error: File is not a PDF")
        return False
    return True

