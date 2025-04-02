import os
from concurrent.futures.thread import ThreadPoolExecutor

import pymupdf as fitz  # PyMuPDF
from openai import OpenAI

OPENAI_API_BASE = "https://api.deepseek.com/v1"
client = OpenAI(base_url=OPENAI_API_BASE)

# Configure OpenAI API to point to Ollama
MODEL_NAME = "deepseek-chat"

# TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(base_url=OPENAI_API_BASE)'
# openai.api_base = OPENAI_API_BASE

exec = ThreadPoolExecutor(2)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)  # Open PDF

    fs = []

    page: fitz.Page
    for page in doc:
        _page_text = page.get_text("text", sort=True)
        fs.append(exec.submit(format_and_save, pdf_path, page.number, _page_text))

    for f in fs:
        f.result()

    return ""


def format_and_save(pdf_path: str, page_no: int, page_text: str) -> None:
    """Formats and saves the extracted text."""
    f_name = f"{pdf_path}_page_{page_no}.txt"
    # Save the formatted text to a file
    if os.path.exists(f_name):
        print(f"File {f_name} already exists. Skipping.")
        return

    print(f"Formatting text for {f_name}")

    txt = format_text_with_llama(page_text)
    with open(f_name, "w") as f:
        f.write(txt)

    print(f"Formatted text saved to {f_name}")


# Function to reformat text using OpenAI-compatible API (Ollama)
def format_text_with_llama(raw_text: str) -> str:
    prompt = f"""You are an advanced text processing model. I have a PDF document that contains text arranged in multiple columns. Please convert the text into a linear format, maintaining the original meaning and context. Ensure that the output is in plain text without any column formatting.

    {raw_text}"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a text reformatter."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    return str(response.choices[0].message.content.strip())
