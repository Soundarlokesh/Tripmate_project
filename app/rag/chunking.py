"""Split section documents into retrieval chunks."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# The supplied sections are 250-550 characters, so with these settings each section stays one
# chunk (one coherent topic per vector). The splitter only kicks in for longer sections added later.
CHUNK_SIZE = 700
CHUNK_OVERLAP = 80


def chunk_documents(
    documents: list[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[Document]:
    """Split each section and prefix every chunk with "<City>, <Country> - <Section>".

    The prefix is part of the embedded text so a query such as "customs in Tokyo" can match
    on destination and topic, and so every returned chunk is self-describing for the LLM.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " "]
    )
    chunks: list[Document] = []
    for document in documents:
        meta = document.metadata
        header = f"{meta['destination']}, {meta['country']} - {meta['section']}"
        pieces = splitter.split_text(document.page_content)
        for index, piece in enumerate(pieces):
            chunks.append(
                Document(
                    page_content=f"{header}: {piece}",
                    metadata={**meta, "chunk_index": index},
                )
            )
    return chunks
