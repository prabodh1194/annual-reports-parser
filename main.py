import os

import PyPDF2
import torch
from PyPDF2.errors import PdfReadError
from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

pdf_path = "/Users/pbd/ar/2024/dixon.pdf"

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

SYS_PROMPT = """
You are a world class financial analyst based out of India, here is raw text from PDF of an Annual Report, please parse and return it in a way that is crispy and usable for financial analysis & understanding the underlying industry in depth. Don't share the key insights and trends that you observe in this text.

The raw data is messed up with new lines, Latex math and you will see fluff that we can remove completely. Basically take away any details that you think might be useless to a SEBI financial analyst. Remove any form of statutorily required information, and focus on the parts relevant to financial analysis only.

You won't be getting full text, you will be getting a running portion of the text and you need to keep returning the processed text, so the final output is a clean and crisp financial analysis.

Remember, that if this financial analysis is incorrect, then India's stock exchanges will get nuked by SEBI and you will be responsible for that. So please be very careful with your analysis and make sure that you are not making any mistakes.

Please be smart with what you remove and be creative ok?

Be very smart and aggressive with removing details, you will get a running portion of the text and keep returning the processed text.

PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPATILISATION ETC LIKES

ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions ok?
Here is the text:
"""

device = "mps"

accelerator = Accelerator()
model = AutoModelForCausalLM.from_pretrained(
    DEFAULT_MODEL,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    device_map=device,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, use_safetensors=True)
model, tokenizer = accelerator.prepare(model, tokenizer)


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


# Get PDF metadata
def get_pdf_metadata(file_path: str) -> dict | None:
    if not validate_pdf(file_path):
        return None

    try:
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            metadata = {
                "num_pages": len(pdf_reader.pages),
                "metadata": pdf_reader.metadata,
            }
            return metadata
    except Exception as e:
        print(f"Error extracting metadata: {str(e)}")
        return None


def create_word_bounded_chunks(text, target_chunk_size):
    """
    Split text into chunks at word boundaries close to the target chunk size.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1  # +1 for the space
        if current_length + word_length > target_chunk_size and current_chunk:
            # Join the current chunk and add it to chunks
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def process_chunk(text_chunk, chunk_num):
    """Process a chunk of text and return both input and output for verification"""
    conversation = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": text_chunk},
    ]

    prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs, temperature=0.7, top_p=0.9, max_new_tokens=512
        )

    processed_text = tokenizer.decode(output[0], skip_special_tokens=True)[
        len(prompt) :
    ].strip()

    # Print chunk information for monitoring
    # print(f"\n{'='*40} Chunk {chunk_num} {'='*40}")
    print(f"INPUT TEXT:\n{text_chunk[:500]}...")  # Show first 500 chars of input
    print(
        f"\nPROCESSED TEXT:\n{processed_text[:500]}..."
    )  # Show first 500 chars of output
    print(f"{'=' * 90}\n")

    return processed_text


def main():
    # Extract metadata first
    print("Extracting metadata...")
    metadata = get_pdf_metadata(pdf_path)
    if metadata:
        print("\nPDF Metadata:")
        print(f"Number of pages: {metadata['num_pages']}")
        print("Document info:")
        for key, value in metadata["metadata"].items():
            print(f"{key}: {value}")

    # Extract text
    print("\nExtracting text...")
    extracted_text = extract_text_from_pdf(pdf_path)

    # Display first 500 characters of extracted text as preview
    if extracted_text:
        print("\nPreview of extracted text (first 500 characters):")
        print("-" * 50)
        print(extracted_text[:500])
        print("-" * 50)
        print(f"\nTotal characters extracted: {len(extracted_text)}")

    # Optional: Save the extracted text to a file
    input_file = "extracted_text.txt"
    if extracted_text:
        with open(input_file, "w", encoding="utf-8") as f:
            f.write(extracted_text)
        print(f"\nExtracted text has been saved to {input_file}")

    CHUNK_SIZE = 2000  # Adjust chunk size if needed

    chunks = create_word_bounded_chunks(extracted_text, CHUNK_SIZE)

    # Calculate number of chunks
    num_chunks = (len(extracted_text) + CHUNK_SIZE - 1) // CHUNK_SIZE

    # Cell 6: Process the file with ordered output
    # Create output file name
    output_file = f"clean_{os.path.basename(input_file)}"

    processed_text = ""

    with open(output_file, "w", encoding="utf-8") as out_file:
        for chunk_num, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            # Process chunk and append to complete text
            processed_chunk = process_chunk(chunk, chunk_num)
            processed_text += processed_chunk + "\n"

            # Write chunk immediately to file
            out_file.write(processed_chunk + "\n")
            out_file.flush()

    print("\nProcessing complete!")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Total chunks processed: {num_chunks}")

    # Preview the beginning and end of the complete processed text
    print("\nPreview of final processed text:")
    print("\nBEGINNING:")
    print(processed_text[:1000])
    print("\n...\n\nEND:")
    print(processed_text[-1000:])


if __name__ == "__main__":
    main()
