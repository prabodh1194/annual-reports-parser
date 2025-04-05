"""
Script to compute embeddings for Pile of Law dataset using `Alibaba-NLP/gte-Qwen2-7B-instruct` through vLLM.
"""

import dataclasses
import logging
import os
import pickle
import shutil
import time
from typing import Iterator

import nltk
import numpy as np
import pandas as pd
import pymupdf as fitz
import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize NLTK to chunk documents
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download("punkt")
    nltk.download("punkt_tab")
    logger.info("Download complete")


@dataclasses.dataclass(kw_only=True, eq=True, frozen=True)
class Document:
    company_name: str
    text: str
    report_year: str


def compute_embeddings_batch(
    chunks: list[dict],
    vllm_endpoint: str,
    output_path: str,
    batch_size: int = 32,
    partition_size: int = 1000,
) -> None:
    """Compute embeddings for document chunks using DeepSeek R1 and save in partitions.

    Args:
        chunks: List of document chunks
        vllm_endpoint: Endpoint for vLLM service
        output_path: Path to save embeddings
    """
    current_partition = []
    partition_counter = 0

    # Process in batches
    for i in tqdm(range(0, len(chunks), batch_size), desc="Computing embeddings"):
        batch = chunks[i : i + batch_size]

        # Create prompt for each chunk - simplified prompt
        prompts = [chunk["content"] for chunk in batch]

        try:
            # Print request payload for debugging
            request_payload = {
                "model": "/tmp/model",
                # because this is loaded from the mounted directory
                "input": prompts,
            }

            response = requests.post(
                f"{vllm_endpoint}/v1/embeddings", json=request_payload, timeout=60
            )

            response.raise_for_status()

            # Extract embeddings - updated response parsing
            result = response.json()

            if "data" not in result:
                raise ValueError(f"Unexpected response format: {result}")

            embeddings = [item["embedding"] for item in result["data"]]

            # Combine embeddings with metadata
            for chunk, embedding in zip(batch, embeddings):
                current_partition.append(
                    {
                        "id": chunk["id"],
                        "name": chunk["name"],
                        "content": chunk["content"],
                        "chunk_text": chunk["chunk_text"],
                        "chunk_start": chunk["chunk_start"],
                        "split": chunk["split"],
                        "source": chunk["source"],
                        "embedding": pickle.dumps(np.array(embedding)),
                        # Include document metadata
                        "document_id": chunk["document_id"],
                        "document_url": chunk["document_url"],
                        "document_created_timestamp": chunk[
                            "document_created_timestamp"
                        ],
                        "document_downloaded_timestamp": chunk[
                            "document_downloaded_timestamp"
                        ],
                    }
                )

            # Save partition when it reaches the desired size
            if len(current_partition) >= partition_size:
                save_partition(current_partition, output_path, partition_counter)
                partition_counter += 1
                current_partition = []

        except Exception as e:
            logger.error(f"Error computing embeddings for batch: {str(e)}")
            time.sleep(5)
            continue

    # Save any remaining embeddings in the final partition
    if current_partition:
        save_partition(current_partition, output_path, partition_counter)


def save_partition(
    results: list[dict], output_path: str, partition_counter: int
) -> None:
    """Save a partition of embeddings to a parquet file with atomic write.

    Args:
        results: List of embeddings
        output_path: Path to save embeddings
        partition_counter: Partition counter
    """
    if not results:
        return

    df = pd.DataFrame(results)
    final_path = f"{output_path}_part_{partition_counter}.parquet"
    temp_path = f"/tmp/embeddings_{partition_counter}.tmp"

    # Write to temporary file first
    df.to_parquet(temp_path, engine="pyarrow", index=False)

    # Copy from temp to final destination
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    shutil.copy2(temp_path, final_path)
    os.remove(temp_path)  # Clean up temp file

    logger.info(
        f"Saved partition {partition_counter} to {final_path} with {len(df)} rows"
    )


def generate_embeddings(*, pdf_path: str, report_year: str, company_name: str) -> None:
    # Load documents
    logger.info("Loading documents from Pile of Law dataset...")
    document = load_document(pdf_path)

    # Chunk documents with global counter
    logger.info("Chunking documents...")
    chunks = []
    next_chunk_idx = 0  # Initialize global chunk counter
    for doc in document:
        doc_chunks, next_chunk_idx = chunk_document(
            Document(text=doc, report_year=report_year, company_name=company_name),
            start_chunk_idx=next_chunk_idx,
        )
        chunks.extend(doc_chunks)
    logger.info(f"Created {len(chunks)} chunks")

    # Compute embeddings and save in partitions
    logger.info("Computing embeddings...")
    compute_embeddings_batch(
        chunks,
        args.vllm_endpoint,
        f"{pdf_path}_embeddings",
    )
    logger.info("Finished computing and saving embeddings")


def load_document(pdf_path: str) -> str:
    pdf_doc = fitz.open(pdf_path)
    text = ""

    for i in range(pdf_doc.page_count):
        txt_file_path = f"{pdf_path}_{i}.txt"
        text += open(txt_file_path).read()
        text += "\n"

    return text


def chunk_document(
    document: Document,
    start_chunk_idx=0,
    chunk_size=2048,
    overlap=512,
) -> tuple[list[dict], int]:
    """Split document into overlapping chunks using sentence-aware splitting.

    Args:
        document: The document to chunk
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        start_chunk_idx: Starting index for global chunk counting

    Returns:
        List of chunks and the next available chunk index
    """
    text = document.text
    chunks = []
    chunk_idx = start_chunk_idx

    # Split into sentences first
    sentences = nltk.sent_tokenize(text)

    current_chunk: list[str] = []
    current_length = 0

    def create_chunk(_chunk_idx: int, _chunk_text: str) -> dict:
        return {
            "id": str(_chunk_idx),
            "name": document.company_name,
            "content": document.text,  # Store full document content
            "chunk_text": _chunk_text.strip(),  # Store the specific chunk
            "chunk_start": len(" ".join(current_chunk[: -(2 if overlap > 0 else 0)]))
            if overlap > 0
            else 0,  # Approximate start position
            "report_year": document.report_year,
        }

    for sentence in sentences:
        sentence_len = len(sentence)

        # If adding this sentence would exceed chunk size, save current chunk
        if current_length + sentence_len > chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(create_chunk(chunk_idx, chunk_text))
            chunk_idx += 1

            # Keep last few sentences for overlap
            overlap_text = " ".join(current_chunk[-2:])  # Keep last 2 sentences
            current_chunk = [overlap_text] if overlap > 0 else []
            current_length = len(overlap_text) if overlap > 0 else 0

        current_chunk.append(sentence)
        current_length += sentence_len + 1  # +1 for space

    # Add the last chunk if it's not empty
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(create_chunk(chunk_idx, chunk_text))
        chunk_idx += 1

    return chunks, chunk_idx
