import re
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")


def clean_text(text):
    """Removes unwanted characters and normalizes text."""
    text = text.replace("\n", " ")  # Remove newlines
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = re.sub(r"\d{2,4}-\d{2,4}", "", text)  # Remove years like "2022-23"
    return text.strip()


def chunk_text(text, chunk_size=500):
    """Splits text into smaller chunks for embedding."""
    sentences = sent_tokenize(text)
    chunks, chunk = [], ""

    for sentence in sentences:
        if len(chunk) + len(sentence) <= chunk_size:
            chunk += " " + sentence
        else:
            chunks.append(chunk.strip())
            chunk = sentence

    if chunk:
        chunks.append(chunk.strip())

    return chunks
