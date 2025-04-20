import requests
from langchain_community.vectorstores import Chroma

from embeddings import VLLM_ENDPOINT


def query(db: Chroma) -> None:
    query_text = "What is the revenue of Dixon Technologies?"

    response = requests.post(
        f"{VLLM_ENDPOINT}/v1/embeddings",
        json={"input": query_text},
        headers={
            "Accept": "application/json",
        },
    )

    response.raise_for_status()

    # Extract embeddings - updated response parsing
    result = response.json()

    embeddings = result["data"][0]["embedding"]

    s = db.similarity_search_by_vector(embeddings, k=10)

    print(s)
