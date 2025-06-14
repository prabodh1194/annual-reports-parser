import os
import re
from collections import deque
from typing import Iterable, Deque, Any

import chromadb
from tqdm import tqdm
from transformers import AutoTokenizer

from embeddings.chroma_ef import openai_ef, get_embedding_path


def chunk_token_generator_streaming(
    *,
    folder_path: str,
    model_name: str = "Alibaba-NLP/gte-Qwen2-7B-instruct",
    chunk_size: int = 4072,
    overlap: int = 512,
) -> Iterable[tuple[str, str]]:
    enc = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    files = sorted(os.listdir(folder_path), key=lambda x: int(x.split(".")[0]))

    buffer: Deque[int] = deque()  # sliding window buffer
    current_len = 0  # track current length of buffer
    pages = set()

    for filename in tqdm(files):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            tokens = enc.encode(re.sub("\s+", " ", text))

            for token in tokens:
                buffer.append(token)
                current_len += 1
                pages.add(filename)

                if current_len == chunk_size:
                    # yield current chunk
                    yield (
                        enc.decode(buffer, errors="strict"),
                        ",".join(map(str, sorted(list(pages)))),
                    )
                    pages = set()

                    # slide window with overlap
                    for _ in range(chunk_size - overlap):
                        buffer.popleft()
                        current_len -= 1

    # Yield remaining tokens at the end (if any)
    if current_len > 0:
        yield (
            enc.decode(buffer, errors="strict"),
            ",".join(map(str, sorted(list(pages)))),
        )


def generate_chroma_client(*, company_name: str, year: str) -> chromadb.Collection:
    chroma_client = chromadb.PersistentClient(path=get_embedding_path())

    return chroma_client.get_or_create_collection(
        embedding_function=openai_ef,
        name="financial_data_collection",
        metadata={"company_name": company_name, "year": year},
    )


def embed(*, doc: str, pages: str, chroma_collection: chromadb.Collection) -> None:
    chroma_collection.add(ids=pages, documents=doc)


def query(
    *,
    query: str,
    n_results: int = 5,
) -> Any:
    chroma_collection = chromadb.PersistentClient(
        path=get_embedding_path()
    ).get_collection(
        embedding_function=openai_ef,
        name="financial_data_collection",
    )
    results = chroma_collection.query(query_texts=[query], n_results=n_results)
    return results  # Return the first result set
