import os
from concurrent.futures import ThreadPoolExecutor

import pymupdf as fitz
import pymupdf4llm
from tqdm import tqdm


def extract_text_from_pdf(pdf_path: str, page_no: int = -1) -> None:
    target_txt_file = f"{pdf_path}_{page_no}.txt"

    if os.path.exists(target_txt_file):
        print(f"File {target_txt_file} already exists. Skipping...")
        return

    md_text = (
        pymupdf4llm.to_markdown(pdf_path, pages=[page_no])
        .replace("**", "")
        .replace("##", "")
        .strip()
    )

    with open(target_txt_file, "w") as f:
        f.write(md_text)


def extract_text_from_all_pages(pdf_path: str) -> None:
    threads = []
    exc = ThreadPoolExecutor()
    doc = fitz.open(pdf_path)
    for page_no in range(doc.page_count):
        threads.append(exc.submit(extract_text_from_pdf, pdf_path, page_no))

    for thread in tqdm(threads):
        thread.result()
