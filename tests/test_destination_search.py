"""RAG tests: ingestion, chunking, FAISS persistence and retrieval over the real dataset."""

from __future__ import annotations

import pytest

from app.config import PROJECT_ROOT
from app.rag.chunking import chunk_documents
from app.rag.embeddings import get_embeddings
from app.rag.ingestion import IngestionError, load_destination_documents, parse_guide
from app.rag.vector_store import (
    DestinationKnowledgeBase,
    IndexIntegrityError,
    IndexNotFoundError,
    build_index,
)
from app.tools.destination_search import search_destination_guide
from app.tools.registry import build_registry
from langchain_core.documents import Document
from tests.conftest import DATA_DIR, offline_settings
from tests.fakes import FakeWeatherProvider

DESTINATIONS = {"Tokyo", "Reykjavik", "Bangkok", "Barcelona"}
SECTIONS = {"Visa & Entry", "Best Time To Visit", "Local Customs", "Packing Tips", "Safety & Health"}


# --- ingestion -----------------------------------------------------------------------------------


def test_ingests_every_section_of_the_supplied_dataset():
    documents = load_destination_documents(DATA_DIR)
    assert len(documents) == 20
    assert {d.metadata["destination"] for d in documents} == DESTINATIONS
    assert {d.metadata["section"] for d in documents} == SECTIONS
    assert {d.metadata["source"] for d in documents} == {f"{c.lower()}.txt" for c in DESTINATIONS}


def test_extracts_section_text_and_skips_disclaimer():
    documents = parse_guide(DATA_DIR / "tokyo.txt")
    customs = next(d for d in documents if d.metadata["section"] == "Local Customs")
    assert customs.page_content.startswith("Bowing is a common greeting")
    assert customs.metadata["country"] == "Japan"
    assert not any("assessment exercise" in d.page_content for d in documents)


def test_missing_data_directory(tmp_path):
    with pytest.raises(IngestionError, match="not found"):
        load_destination_documents(tmp_path / "nope")


def test_directory_without_guides(tmp_path):
    (tmp_path / "README.txt").write_text("not a guide")
    with pytest.raises(IngestionError, match="No destination"):
        load_destination_documents(tmp_path)


def test_malformed_file_without_title_is_rejected(tmp_path):
    (tmp_path / "bad.txt").write_text("VISA & ENTRY\nSome text")
    with pytest.raises(IngestionError, match="bad.txt"):
        load_destination_documents(tmp_path)


def test_file_without_sections_is_rejected(tmp_path):
    (tmp_path / "empty.txt").write_text("DESTINATION GUIDE: ROME, ITALY\n\nJust a note.")
    with pytest.raises(IngestionError, match="no sections"):
        load_destination_documents(tmp_path)


def test_unreadable_encoding_is_rejected(tmp_path):
    (tmp_path / "bin.txt").write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(IngestionError, match="Cannot read"):
        load_destination_documents(tmp_path)


# --- chunking ------------------------------------------------------------------------------------


def test_each_supplied_section_becomes_one_prefixed_chunk():
    chunks = chunk_documents(load_destination_documents(DATA_DIR))
    assert len(chunks) == 20
    tokyo_customs = next(
        c for c in chunks if c.metadata["destination"] == "Tokyo" and c.metadata["section"] == "Local Customs"
    )
    assert tokyo_customs.page_content.startswith("Tokyo, Japan - Local Customs: Bowing")


def test_long_sections_are_split_with_metadata_preserved():
    long_text = " ".join(f"Sentence number {i} about local customs." for i in range(60))
    doc = Document(page_content=long_text, metadata={
        "destination": "Rome", "country": "Italy", "section": "Local Customs", "source": "rome.txt"})
    chunks = chunk_documents([doc], chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 3
    assert all(c.metadata["source"] == "rome.txt" for c in chunks)
    assert all(c.page_content.startswith("Rome, Italy - Local Customs: ") for c in chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


# --- index build / persistence -------------------------------------------------------------------


def test_index_is_built_and_persisted(settings, knowledge_base):
    index_dir = settings.faiss_index_path
    assert {p.name for p in index_dir.iterdir()} == {"index.faiss", "index.pkl", "manifest.json"}
    assert knowledge_base.manifest["chunk_count"] == 20
    assert set(knowledge_base.destinations) == DESTINATIONS
    assert knowledge_base.manifest["embedding"] == "offline-hash"


def test_loaded_index_gives_same_results_as_fresh_load(settings, knowledge_base):
    reloaded = DestinationKnowledgeBase.load(settings, get_embeddings(settings))
    query = "visa requirements for Tokyo"
    assert reloaded.search(query, 3, 0.0) == knowledge_base.search(query, 3, 0.0)


def test_missing_index_gives_actionable_error(tmp_path):
    cfg = offline_settings(tmp_path / "missing")
    with pytest.raises(IndexNotFoundError, match="python -m scripts.ingest_data"):
        DestinationKnowledgeBase.load(cfg, get_embeddings(cfg))


def test_tampered_index_is_refused(tmp_path):
    cfg = offline_settings(tmp_path / "idx")
    build_index(cfg, get_embeddings(cfg))
    with (cfg.faiss_index_path / "index.pkl").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(IndexIntegrityError, match="manifest hash"):
        DestinationKnowledgeBase.load(cfg, get_embeddings(cfg))


def test_index_built_with_other_embedding_model_is_refused(tmp_path):
    cfg = offline_settings(tmp_path / "idx")
    build_index(cfg, get_embeddings(cfg))
    other = offline_settings(tmp_path / "idx", embedding_provider="huggingface")
    with pytest.raises(IndexIntegrityError, match="Rebuild"):
        DestinationKnowledgeBase.load(other, get_embeddings(cfg))


def test_corrupt_manifest_is_refused(tmp_path):
    cfg = offline_settings(tmp_path / "idx")
    build_index(cfg, get_embeddings(cfg))
    (cfg.faiss_index_path / "manifest.json").write_text("{not json")
    with pytest.raises(IndexIntegrityError, match="unreadable"):
        DestinationKnowledgeBase.load(cfg, get_embeddings(cfg))


def test_ingest_script_reports_failure_for_bad_data(tmp_path, monkeypatch, capsys):
    from scripts import ingest_data

    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline-hash")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "missing"))
    monkeypatch.setenv("FAISS_INDEX_PATH", str(tmp_path / "idx"))
    monkeypatch.setattr("app.config.PROJECT_ROOT", PROJECT_ROOT)
    assert ingest_data.main() == 1
    assert "Ingestion failed" in capsys.readouterr().err


# --- retrieval -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "destination", "section", "expected_text"),
    [
        ("What are the local customs in Tokyo?", "Tokyo", "Local Customs", "Bowing"),
        ("Is tap water safe to drink in Bangkok?", "Bangkok", "Safety & Health", "Tap water"),
        ("Do I need a visa for Iceland?", "Reykjavik", "Visa & Entry", "Schengen"),
        ("pickpockets in Barcelona", "Barcelona", "Safety & Health", "pickpocketing"),
    ],
)
def test_retrieves_the_relevant_chunk_first(knowledge_base, query, destination, section, expected_text):
    top = knowledge_base.search(query, top_k=3, threshold=0.0)[0]
    assert (top.destination, top.section) == (destination, section)
    assert expected_text in top.text
    assert top.source == f"{destination.lower()}.txt"


def test_named_destination_restricts_results(knowledge_base):
    results = knowledge_base.search("what should I pack for Tokyo", top_k=5, threshold=0.0)
    assert results and {r.destination for r in results} == {"Tokyo"}


def test_score_is_cosine_similarity(knowledge_base):
    exact_text = chunk_documents(load_destination_documents(DATA_DIR))[0].page_content
    assert knowledge_base.search(exact_text, top_k=1, threshold=0.0)[0].score == pytest.approx(1.0, abs=1e-3)


def test_top_k_is_respected(knowledge_base):
    assert len(knowledge_base.search("best time to visit", top_k=2, threshold=0.0)) == 2


def test_threshold_removes_weak_matches(knowledge_base):
    assert knowledge_base.search("customs in Tokyo", top_k=3, threshold=0.99) == []


def test_off_topic_query_returns_nothing(knowledge_base, settings):
    for query in ("how do I bake sourdough bread", "what is the capital of France", "write a poem about the sea"):
        assert search_destination_guide(query, knowledge_base, 3, settings.relevance_threshold) == []


@pytest.mark.parametrize("query", ["", "   ", None])
def test_empty_query_is_rejected(knowledge_base, query):
    with pytest.raises(ValueError, match="non-empty"):
        search_destination_guide(query, knowledge_base, 3, 0.1)


def test_overlong_query_is_rejected(knowledge_base):
    with pytest.raises(ValueError, match="too long"):
        search_destination_guide("a" * 501, knowledge_base, 3, 0.1)


def test_tool_returns_labelled_strings_with_source(knowledge_base):
    results = search_destination_guide("local customs in Tokyo", knowledge_base, 3, 0.1)
    assert results[0].startswith("Tokyo, Japan - Local Customs: Bowing")
    assert results[0].endswith("(source: tokyo.txt)")


# --- registry-level behaviour (what the LLM actually receives) -----------------------------------


@pytest.fixture
def destination_tool(settings, knowledge_base):
    return build_registry(settings, knowledge_base, FakeWeatherProvider())["search_destination_guide"].tool


def test_unknown_destination_adds_coverage_note_listing_supported_cities(destination_tool):
    result = destination_tool.invoke({"query": "local customs in Paris"})
    note = result[-1]
    assert note.startswith("[note]")
    for city in DESTINATIONS:
        assert city in note
    assert "not covered" in note


def test_no_match_returns_only_a_note(destination_tool):
    result = destination_tool.invoke({"query": "how do I bake sourdough bread"})
    assert len(result) == 1 and "No relevant information" in result[0]


def test_known_destination_has_no_note(destination_tool):
    result = destination_tool.invoke({"query": "local customs in Tokyo"})
    assert not any(item.startswith("[note]") for item in result)


def test_tool_description_lists_supported_destinations(destination_tool):
    assert all(city in destination_tool.description for city in DESTINATIONS)


def test_embedding_download_failure_gives_clear_error(monkeypatch, tmp_path):
    import langchain_huggingface

    from app.rag.embeddings import EmbeddingModelError

    def fail(**kwargs):
        raise OSError("offline")

    monkeypatch.setattr(langchain_huggingface, "HuggingFaceEmbeddings", fail)
    with pytest.raises(EmbeddingModelError, match="offline-hash"):
        get_embeddings(offline_settings(tmp_path, embedding_provider="huggingface"))
