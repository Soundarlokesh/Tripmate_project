from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from app.agent.graph import TripMateAgent
from app.config import PROJECT_ROOT, Settings
from app.rag.embeddings import get_embeddings
from app.rag.vector_store import DestinationKnowledgeBase, build_index
from app.tools.registry import build_registry
from tests.fakes import FakeWeatherProvider, ScriptedChatModel

DATA_DIR = PROJECT_ROOT / "data" / "destinations"


def offline_settings(index_path: Path, **overrides) -> Settings:
    """Settings that need no API keys or network: offline-hash embeddings, real dataset."""
    base = Settings(
        embedding_provider="offline-hash",
        relevance_threshold=0.10,
        data_dir=DATA_DIR,
        faiss_index_path=index_path,
    )
    return replace(base, **overrides)


@pytest.fixture(scope="session")
def settings(tmp_path_factory) -> Settings:
    return offline_settings(tmp_path_factory.mktemp("faiss") / "destinations")


@pytest.fixture(scope="session")
def knowledge_base(settings) -> DestinationKnowledgeBase:
    """FAISS index built from the real supplied dataset with the offline embedding stand-in."""
    embeddings = get_embeddings(settings)
    build_index(settings, embeddings)
    return DestinationKnowledgeBase.load(settings, embeddings)


@pytest.fixture
def weather_provider() -> FakeWeatherProvider:
    return FakeWeatherProvider()


@pytest.fixture
def make_agent(settings, knowledge_base, weather_provider):
    def factory(script: list, *, provider=None, repeat_last: bool = False, **setting_overrides):
        llm = ScriptedChatModel(script=script, repeat_last=repeat_last)
        cfg = replace(settings, **setting_overrides)
        registry = build_registry(cfg, knowledge_base, provider or weather_provider)
        return TripMateAgent(llm, registry, cfg), llm

    return factory
