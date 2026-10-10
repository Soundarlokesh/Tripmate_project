"""Destination knowledge tool logic: validation + formatting around the FAISS knowledge base."""

from __future__ import annotations

from app.rag.vector_store import DestinationKnowledgeBase

MAX_QUERY_CHARS = 500


def search_destination_guide(
    query: str, knowledge_base: DestinationKnowledgeBase, top_k: int, threshold: float
) -> list[str]:
    """Return the most relevant guide chunks for `query` as strings, best match first.

    Each string is "<City>, <Country> - <Section>: <text> (source: <file>)". An empty list means
    nothing in the knowledge base was relevant enough; nothing is ever made up.

    Raises:
        ValueError: if the query is empty or too long.
    """
    cleaned = (query or "").strip()
    if not cleaned:
        raise ValueError("query must be a non-empty string")
    if len(cleaned) > MAX_QUERY_CHARS:
        raise ValueError(f"query is too long ({len(cleaned)} chars, max {MAX_QUERY_CHARS})")
    chunks = knowledge_base.search(cleaned, top_k=top_k, threshold=threshold)
    return [f"{chunk.text} (source: {chunk.source})" for chunk in chunks]
