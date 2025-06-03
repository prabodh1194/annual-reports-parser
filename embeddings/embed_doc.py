import os
from typing import Iterable
import chromadb

from tqdm import tqdm

from chroma_ef import openai_ef


def get_text_from_parsed_files(directory: str) -> Iterable[tuple[str, tuple[str, str]]]:
    # Get all text files in the directory
    text_files = sorted([f for f in os.listdir(directory) if f.endswith(".txt")])

    # Iterate through the text files in pairs
    for i in tqdm(range(len(text_files) - 1)):
        file1 = text_files[i]
        file2 = text_files[i + 1]

        # Read the content of the first file
        with open(os.path.join(directory, file1), "r", encoding="utf-8") as f:
            text1 = f.read()

        # Read the content of the second file
        with open(os.path.join(directory, file2), "r", encoding="utf-8") as f:
            text2 = f.read()

        # Combine the two texts
        yield text1 + "\n" + text2, (str(i), str(i + 1))


def generate_chroma_client(*, company_name: str, year: str) -> chromadb.Collection:
    chroma_client = chromadb.PersistentClient()
    return chroma_client.get_or_create_collection(
        name="financial_data_collection",
        configuration={"embedding_function": openai_ef},
        metadata={"company_name": company_name, "year": year},
    )


def embed(
    *, doc: str, pages: list[str], chroma_collection: chromadb.Collection
) -> None:
    chroma_collection.add(ids=pages, documents=doc)
