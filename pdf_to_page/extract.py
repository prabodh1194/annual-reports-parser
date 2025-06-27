import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from docling.document_converter import DocumentConverter
from docling_parse.pdf_parser import DoclingPdfParser
from tqdm import tqdm

c = DocumentConverter()

regex_replacements = {
    r"--+": "-",  # Replace multiple dashes with a single dash
    r"\n+": "\n",  # Replace multiple newlines with a single space
    r"  +": " ",  # Replace multiple spaces with a single space
    r"\t+": " ",  # Replace multiple tabs with a single space
}


def extract_text_from_pdf(
    pdf_path: str, bronze: str, silver: str, page_no: int = -1
) -> None:
    target_txt_file = f"{pdf_path}/{page_no}.txt".replace(bronze, silver)

    if os.path.exists(target_txt_file):
        print(f"File {target_txt_file} already exists. Skipping...")
        return

    print("Extracting text from page", page_no)

    md_text = (
        c.convert(pdf_path, page_range=(page_no, page_no))
        .document.export_to_text()
        .replace("**", "")
        .strip()
    )

    for pattern, replacement in regex_replacements.items():
        md_text = re.sub(pattern, replacement, md_text)

    print("Extracted text from page", page_no)

    with open(target_txt_file, "w") as f:
        f.write(md_text)


def extract_text_from_all_pages(pdf_path: str, bronze: str, silver: str) -> None:
    threads = []
    exc = ThreadPoolExecutor(max_workers=1)
    doc = DoclingPdfParser().load(pdf_path)
    error_pages = []

    for page_no in range(1 + doc.number_of_pages()):
        threads.append(
            exc.submit(extract_text_from_pdf, pdf_path, bronze, silver, page_no)
        )

    for i, thread in tqdm(enumerate(threads)):
        try:
            thread.result()
        except Exception as e:
            print(f"Error in processing page {i}: {thread}")
            print(e)
            error_pages.append(i + 1)
            continue

    with open(f"{pdf_path}_error_pages.json", "w") as f:
        json.dump(error_pages, f)
