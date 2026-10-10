"""Build and persist the FAISS index from the destination dataset.

Usage: python -m scripts.ingest_data
"""

from __future__ import annotations

import sys

from app.config import ConfigError, Settings
from app.logging_config import setup_logging
from app.rag.embeddings import EmbeddingModelError, get_embeddings
from app.rag.ingestion import IngestionError
from app.rag.vector_store import build_index


def main() -> int:
    try:
        settings = Settings.from_env()
        setup_logging(settings.log_level)
        manifest = build_index(settings, get_embeddings(settings))
    except (ConfigError, IngestionError, EmbeddingModelError) as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Indexed {manifest['chunk_count']} chunks from {len(manifest['source_files'])} files "
        f"({', '.join(manifest['destinations'])}) using {manifest['embedding']} "
        f"-> {settings.faiss_index_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
