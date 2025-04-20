import chromadb
import numpy as np
import pandas as pd
from langchain.vectorstores import Chroma


def from_parquet(file: str) -> tuple[list, list, list]:
    df = pd.read_parquet(file)

    texts = df["chunk_text"].tolist()
    embeddings = list(map(np.array, df["embedding"].tolist()))

    ids = [f"id_{i}" for i in range(len(texts))]  # Generate unique IDs

    return texts, embeddings, ids


def store_embedding(texts: list, embeddings: list, ids: list):
    chroma_client = chromadb.PersistentClient()
    collection = chroma_client.create_collection(
        name="my_collection",
        get_or_create=True,
    )

    try:
        collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
        )
    except Exception as e:
        print(f"Error adding to collection: {e}")

    db = Chroma(
        client=chroma_client,
        collection_name="my_collection",
    )

    return db
