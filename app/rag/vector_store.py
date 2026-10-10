"""FAISS index build / persistence / retrieval for destination chunks.

Persistence note: LangChain's FAISS.save_local stores the docstore with pickle, so loading requires
`allow_dangerous_deserialization=True`. We only ever load an index that this project wrote to a
configured local path, and we verify SHA-256 hashes recorded in a manifest first. Never point
FAISS_INDEX_PATH at an index obtained from an untrusted source.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.rag.chunking import chunk_documents
from app.rag.embeddings import embedding_label
from app.rag.ingestion import load_destination_documents

logger = logging.getLogger(__name__)

INDEX_NAME = "index"
MANIFEST_FILE = "manifest.json"
_INDEX_FILES = (f"{INDEX_NAME}.faiss", f"{INDEX_NAME}.pkl")


class IndexNotFoundError(RuntimeError):
    """The FAISS index has not been built (or is incomplete)."""


class IndexIntegrityError(RuntimeError):
    """The index does not match its manifest or the configured embedding model."""


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    destination: str
    section: str
    source: str
    score: float  # cosine similarity, 1.0 = identical direction


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_index(settings: Settings, embeddings: Embeddings) -> dict:
    """Ingest the dataset, embed the chunks, persist FAISS + manifest. Returns the manifest."""
    documents = load_destination_documents(settings.data_dir)
    chunks = chunk_documents(documents)
    # normalize_L2 makes L2 distance equivalent to cosine similarity (see DestinationKnowledgeBase.search).
    store = FAISS.from_documents(chunks, embeddings, normalize_L2=True)
    index_dir = settings.faiss_index_path
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir), index_name=INDEX_NAME)
    manifest = {
        "embedding": embedding_label(settings),
        "chunk_count": len(chunks),
        "destinations": sorted({c.metadata["destination"] for c in chunks}),
        "source_files": sorted({c.metadata["source"] for c in chunks}),
        "file_hashes": {name: _sha256(index_dir / name) for name in _INDEX_FILES},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (index_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("faiss index built", extra={"chunks": len(chunks), "path": str(index_dir)})
    return manifest


class DestinationKnowledgeBase:
    """Read-only wrapper around the persisted FAISS index."""

    def __init__(self, store: FAISS, manifest: dict) -> None:
        self._store = store
        self.manifest = manifest
        self.destinations: list[str] = list(manifest["destinations"])

    @classmethod
    def load(cls, settings: Settings, embeddings: Embeddings) -> DestinationKnowledgeBase:
        index_dir = settings.faiss_index_path
        manifest_path = index_dir / MANIFEST_FILE
        missing = [n for n in (*_INDEX_FILES, MANIFEST_FILE) if not (index_dir / n).is_file()]
        if missing:
            raise IndexNotFoundError(
                f"FAISS index not found or incomplete at {index_dir} (missing: {', '.join(missing)}). "
                "Build it with: python -m scripts.ingest_data"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_hashes = manifest["file_hashes"]
        except (json.JSONDecodeError, KeyError) as exc:
            raise IndexIntegrityError(
                f"Index manifest {manifest_path} is unreadable. Rebuild with: python -m scripts.ingest_data"
            ) from exc
        for name in _INDEX_FILES:
            if _sha256(index_dir / name) != expected_hashes.get(name):
                raise IndexIntegrityError(
                    f"{name} does not match the manifest hash; refusing to load. "
                    "Rebuild with: python -m scripts.ingest_data"
                )
        if manifest.get("embedding") != embedding_label(settings):
            raise IndexIntegrityError(
                f"Index was built with {manifest.get('embedding')!r} but the configuration uses "
                f"{embedding_label(settings)!r}. Rebuild with: python -m scripts.ingest_data"
            )
        store = FAISS.load_local(
            str(index_dir),
            embeddings,
            index_name=INDEX_NAME,
            allow_dangerous_deserialization=True,  # hash-verified, locally built index only
            normalize_L2=True,
        )
        return cls(store, manifest)

    def detect_destinations(self, query: str) -> list[str]:
        """Destinations from the index whose name appears in the query (case-insensitive)."""
        return [
            name
            for name in self.destinations
            if re.search(rf"\b{re.escape(name)}\b", query, flags=re.IGNORECASE)
        ]

    def search(self, query: str, top_k: int, threshold: float) -> list[RetrievedChunk]:
        """Top-k chunks with cosine similarity >= threshold, best first.

        If the query names an indexed destination, results are restricted to it via a metadata
        filter. FAISS L2 distances are squared distances over unit vectors, so cosine = 1 - d / 2.
        """
        named = self.detect_destinations(query)
        search_filter = (lambda meta: meta["destination"] in named) if named else None
        total = self._store.index.ntotal
        hits = self._store.similarity_search_with_score(
            query, k=top_k, filter=search_filter, fetch_k=total
        )
        chunks: list[RetrievedChunk] = []
        for document, distance in hits:
            similarity = 1.0 - float(distance) / 2.0
            if similarity >= threshold:
                chunks.append(_to_chunk(document, similarity))
        return chunks


def _to_chunk(document: Document, similarity: float) -> RetrievedChunk:
    meta = document.metadata
    return RetrievedChunk(
        text=document.page_content,
        destination=meta["destination"],
        section=meta["section"],
        source=meta["source"],
        score=round(similarity, 4),
    )
