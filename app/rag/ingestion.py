"""Read destination guide files into section-level LangChain Documents."""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document

TITLE_PATTERN = re.compile(r"^DESTINATION GUIDE:\s*(?P<city>[^,]+),\s*(?P<country>.+)$")
# Section headings in the supplied files are upper-case lines such as "VISA & ENTRY".
HEADING_PATTERN = re.compile(r"^[A-Z][A-Z &/]+$")


class IngestionError(RuntimeError):
    """Raised when source data is missing or does not match the expected format."""


def load_destination_documents(data_dir: Path) -> list[Document]:
    """Return one Document per guide section, with destination/country/section/source metadata."""
    if not data_dir.is_dir():
        raise IngestionError(f"Data directory not found: {data_dir}")
    files = sorted(data_dir.glob("*.txt"))
    # README.txt in the data pack describes the pack itself and has no guide title.
    guide_files = [f for f in files if f.name.lower() != "readme.txt"]
    if not guide_files:
        raise IngestionError(f"No destination .txt files found in {data_dir}")
    documents: list[Document] = []
    for path in guide_files:
        documents.extend(parse_guide(path))
    return documents


def parse_guide(path: Path) -> list[Document]:
    """Parse one guide file. Raises IngestionError for unreadable or malformed files."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise IngestionError(f"Cannot read {path.name}: {exc}") from exc

    lines = [line.strip() for line in text.splitlines()]
    title = next((TITLE_PATTERN.match(line) for line in lines if line), None)
    if title is None:
        raise IngestionError(
            f"{path.name}: expected first line 'DESTINATION GUIDE: <City>, <Country>'"
        )
    city = title.group("city").strip().title()
    country = title.group("country").strip().title()

    sections: list[tuple[str, list[str]]] = []
    for line in lines[1:]:
        if HEADING_PATTERN.match(line):
            sections.append((line.title(), []))
        elif line and sections:
            sections[-1][1].append(line)
        # Lines before the first heading (the disclaimer note) are intentionally skipped.

    documents = [
        Document(
            page_content=" ".join(body),
            metadata={"destination": city, "country": country, "section": name, "source": path.name},
        )
        for name, body in sections
        if body
    ]
    if not documents:
        raise IngestionError(f"{path.name}: no sections with content found")
    return documents
