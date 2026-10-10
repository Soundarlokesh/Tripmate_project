"""End-to-end packing flow: ingest script -> FAISS -> LangGraph -> both tools -> trace.

Test doubles used (explicitly): ScriptedChatModel for the LLM and FakeWeatherProvider for weather.
Real: the supplied dataset, the ingestion script, HuggingFace-free offline embeddings, FAISS, the
LangChain tool wrappers, LangGraph and tracing. This does not show live Groq tool selection.
"""

from __future__ import annotations

import json
from datetime import date

from langchain_core.messages import AIMessage, ToolMessage

from app.agent.graph import TripMateAgent
from app.rag.embeddings import get_embeddings
from app.rag.vector_store import DestinationKnowledgeBase
from app.tools.registry import build_registry
from scripts import ingest_data
from tests.conftest import DATA_DIR, offline_settings
from tests.fakes import FakeWeatherProvider, ScriptedChatModel, tool_request


def next_month(today: date) -> str:
    year, month = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
    return f"{year}-{month:02d}"


def synthesize(messages) -> AIMessage:
    """Test-double 'LLM' that merges the two tool results into one answer, using only their contents."""
    results = {m.name: json.loads(m.content) for m in messages[-2:] if isinstance(m, ToolMessage)}
    packing = next(c for c in results["search_destination_guide"] if "Packing Tips" in c)
    weather = results["get_weather_forecast"]
    low, high = weather["temp_range_c"]
    caveat = "; ".join(weather["limitations"])
    return AIMessage(content=(
        f"Guide: {packing}\nWeather ({weather['data_kind']}, {weather['source']}): "
        f"{low}-{high} C, {weather['conditions']}. Note: {caveat}"
    ))


def test_packing_query_end_to_end(tmp_path, monkeypatch):
    # 1. Ingest the real dataset through the real command into a fresh FAISS index.
    index_path = tmp_path / "index"
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline-hash")
    monkeypatch.setenv("DATA_DIR", str(DATA_DIR))
    monkeypatch.setenv("FAISS_INDEX_PATH", str(index_path))
    assert ingest_data.main() == 0
    settings = offline_settings(index_path)
    knowledge_base = DestinationKnowledgeBase.load(settings, get_embeddings(settings))
    assert knowledge_base.manifest["chunk_count"] == 20

    # 2. Agent with real tools/graph, scripted LLM, mocked weather.
    month = next_month(date.today())
    provider = FakeWeatherProvider()
    llm = ScriptedChatModel(script=[
        tool_request(
            ("search_destination_guide", {"query": "packing tips for Tokyo"}),
            ("get_weather_forecast", {"city": "Tokyo", "date_or_month": month}),
        ),
        synthesize,
    ])
    agent = TripMateAgent(llm, build_registry(settings, knowledge_base, provider), settings)

    result = agent.ask("I'm visiting Tokyo next month. What should I pack?")

    # 3. Final answer is supported by the real retrieved chunk and the mocked weather.
    assert result.status == "answered"
    assert "Tokyo, Japan - Packing Tips: Layered clothing works well" in result.response
    assert "Comfortable walking shoes are essential" in result.response
    assert "climate_reference, mock-test-data): 4.0-12.0 C" in result.response
    assert "Month-only request" in result.response

    # 4. Trace reflects what actually ran, in order.
    assert [e.node for e in result.trace] == ["input", "agent", "tools", "tools", "agent", "respond"]
    guide, weather = [e for e in result.trace if e.event == "tool_call"]
    assert (guide.tool, guide.status) == ("search_destination_guide", "success")
    assert "Tokyo, Japan - Packing Tips" in guide.summary
    assert (weather.tool, weather.status) == ("get_weather_forecast", "success")
    assert weather.arguments == {"city": "Tokyo", "date_or_month": month}
    assert "live=False" in weather.summary and "climate_reference" in weather.summary
    assert provider.calls and provider.calls[0][3] == "archive"
    assert result.tools_called == ["search_destination_guide", "get_weather_forecast"]
