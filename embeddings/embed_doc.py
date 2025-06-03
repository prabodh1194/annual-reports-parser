import os
from typing import Iterable, Deque

import chromadb

from chroma_ef import openai_ef

import tiktoken
from collections import deque


def chunk_token_generator_streaming(
    *,
    folder_path: str,
    model_name: str = "gpt-4",
    chunk_size: int = 4072,
    overlap: int = 512,
) -> Iterable[str]:
    enc = tiktoken.encoding_for_model(model_name)
    files = sorted(os.listdir(folder_path))

    buffer: Deque[int] = deque()  # sliding window buffer
    current_len = 0  # track current length of buffer

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            tokens = enc.encode(text)

            for token in tokens:
                buffer.append(token)
                current_len += 1

                if current_len == chunk_size:
                    # yield current chunk
                    yield enc.decode(buffer, errors="strict")

                    # slide window with overlap
                    for _ in range(chunk_size - overlap):
                        buffer.popleft()
                        current_len -= 1

    # Yield remaining tokens at the end (if any)
    if current_len > 0:
        yield enc.decode(buffer, errors="strict")


def generate_chroma_client(*, company_name: str, year: str) -> chromadb.Collection:
    chroma_client = chromadb.PersistentClient()
    return chroma_client.get_or_create_collection(
        embedding_function=openai_ef,
        name="financial_data_collection",
        metadata={"company_name": company_name, "year": year},
    )


def embed(
    *, doc: list[str], pages: list[str], chroma_collection: chromadb.Collection
) -> None:
    chroma_collection.add(ids=pages, documents=doc)
