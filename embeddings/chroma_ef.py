import os

import chromadb.utils.embedding_functions as embedding_functions

assert os.getenv("model_url"), "missing url for where the model has been hosted"

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="abc123",
    api_base=f"http://{os.getenv('model_url')}:8000/v1",
    model_name="/home/ubuntu/model",
)
