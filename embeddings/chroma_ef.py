import os

import chromadb.utils.embedding_functions as embedding_functions

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="abc123",
    api_base="http://emb.openn.ai:8000/v1",
    model_name="/home/ubuntu/model",
)


def get_embedding_path() -> str:
    """
    Set as repo_root + 'chroma/'
    """
    return f"{get_repo_root()}/chroma/"


def get_repo_root() -> str:
    """
    Returns the absolute path to the repository root (where .git directory is found).
    """
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while current_dir != os.path.dirname(current_dir):
        if os.path.isdir(os.path.join(current_dir, ".git")):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    raise RuntimeError("Repository root with .git not found.")
