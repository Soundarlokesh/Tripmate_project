"""Embedding model factory.

`huggingface` is the real semantic path (sentence-transformers via langchain-huggingface).
`offline-hash` is a deterministic lexical stand-in that needs no downloads. It exists so tests and
demos run without network access; it is NOT semantic and must not be used to judge retrieval quality.
"""

from __future__ import annotations

import hashlib
import re

import httpx
import numpy as np
from langchain_core.embeddings import Embeddings

from app.config import Settings

class EmbeddingModelError(RuntimeError):
    """The embedding model could not be loaded (for example, the download failed)."""


_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be but by do does for from how i if in is it me my of on or should so "
    "that the there to we what when where which who will with you your can about".split()
)


class HashingEmbeddings(Embeddings):
    """Bag-of-words feature hashing with stop-word removal and light suffix stripping, L2-normalised."""

    def __init__(self, dimensions: int = 2048) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in _TOKEN.findall(text.lower()):
            if token in _STOPWORDS:
                continue
            if len(token) > 4:
                token = re.sub(r"(ing|ed|es|s)$", "", token)
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            slot = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[slot] += sign
        norm = np.linalg.norm(vector)
        return (vector / norm if norm else vector).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_provider == "offline-hash":
        return HashingEmbeddings()
    # Imported lazily: pulls in torch, which offline tests should not pay for.
    from langchain_huggingface import HuggingFaceEmbeddings

    try:
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
    except (OSError, httpx.HTTPError) as exc:
        raise EmbeddingModelError(
            f"Could not load embedding model {settings.embedding_model!r} ({type(exc).__name__}). "
            "The first run downloads it from Hugging Face, so it needs network access; once cached it "
            "works offline. For a no-download demo set EMBEDDING_PROVIDER=offline-hash."
        ) from exc


def embedding_label(settings: Settings) -> str:
    """Identifier stored in the index manifest so an index is only reused with the model that built it."""
    if settings.embedding_provider == "offline-hash":
        return "offline-hash"
    return f"huggingface:{settings.embedding_model}"
