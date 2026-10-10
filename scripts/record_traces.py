"""Run the example scenarios and write their execution traces to examples/sample_traces.jsonl.

Default (offline): scripted LLM test double + mock weather. Every line is labelled with what was
real and what was faked, so these are traces of the graph, not of Groq's behaviour.
    python -m scripts.record_traces

Live (needs GROQ_API_KEY, network, a built index): real Groq, Open-Meteo and the configured embeddings.
    python -m scripts.record_traces --live --output examples/live_traces.jsonl
"""

from __future__ import annotations

import argparse
import json
import tempfile
import uuid
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

from langchain_core.messages import AIMessage

from app.agent.graph import TripMateAgent
from app.config import PROJECT_ROOT, Settings
from app.rag.embeddings import get_embeddings
from app.rag.vector_store import DestinationKnowledgeBase, build_index
from app.tools.registry import build_registry
from app.tools.weather import WeatherProviderError

GUIDE, WEATHER = "search_destination_guide", "get_weather_forecast"
SOON = (date.today() + timedelta(days=2)).isoformat()
NEXT_MONTH = (date.today().replace(day=1) + timedelta(days=32)).strftime("%Y-%m")


def scenarios():
    """(name, turns, scripted LLM steps, weather failure). Steps are only used in offline mode."""
    from tests.fakes import answer_from_tool_results as done
    from tests.fakes import tool_request as call

    return [
        ("A_destination_only", ["What are the local customs in Tokyo?"],
         [call((GUIDE, {"query": "local customs in Tokyo"})), done], None),
        ("B_weather_only", [f"What will the weather be like in Tokyo on {SOON}?"],
         [call((WEATHER, {"city": "Tokyo", "date_or_month": SOON})), done], None),
        ("C_multi_tool_packing", ["I'm visiting Tokyo next month. What should I pack?"],
         [call((GUIDE, {"query": "packing tips for Tokyo"}), (WEATHER, {"city": "Tokyo", "date_or_month": NEXT_MONTH})), done], None),
        ("D_missing_destination", ["What should I pack for my holiday?"],
         [AIMessage(content="Happy to help! Which destination are you visiting, and when?")], None),
        ("E_contextual_follow_up", ["I'm planning a trip to Tokyo.", f"What about the weather there on {SOON}?"],
         [AIMessage(content="Tokyo is a great choice! What would you like to know?"),
          call((WEATHER, {"city": "Tokyo", "date_or_month": SOON})), done], None),
        ("F_out_of_scope", ["Book a flight to Tokyo."],
         [AIMessage(content="I can't book flights. I can share visa, customs, packing and weather information for Tokyo.")], None),
        ("G_greeting", ["Hi, how are you?"], [AIMessage(content="Hi! I'm doing well. Where are you thinking of travelling?")], None),
        ("H_weather_timeout", ["What should I pack for Tokyo this week?"],
         [call((GUIDE, {"query": "packing tips for Tokyo"}), (WEATHER, {"city": "Tokyo", "date_or_month": SOON})), done],
         WeatherProviderError("weather service timed out after 10s")),
    ]


def build_agent(live: bool, steps, failure) -> tuple[TripMateAgent, dict]:
    if live:
        from app.main import create_agent

        settings = Settings.from_env()
        return create_agent(settings), {"llm": f"groq:{settings.groq_model}", "weather": "open-meteo (live)"}
    from tests.fakes import FakeWeatherProvider, ScriptedChatModel

    settings = replace(Settings.from_env(), embedding_provider="offline-hash", relevance_threshold=0.10,
                       faiss_index_path=Path(tempfile.mkdtemp()) / "idx")
    embeddings = get_embeddings(settings)
    build_index(settings, embeddings)
    kb = DestinationKnowledgeBase.load(settings, embeddings)
    registry = build_registry(settings, kb, FakeWeatherProvider(failure=failure))
    labels = {"llm": "scripted-test-double (NOT a real model)", "weather": "mock-test-data (NOT live)",
              "embeddings": "offline-hash (NOT semantic); index built from the real dataset"}
    return TripMateAgent(ScriptedChatModel(script=list(steps)), registry, settings), labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "examples" / "sample_traces.jsonl"))
    args = parser.parse_args()
    lines = []
    for name, turns, steps, failure in scenarios():
        agent, labels = build_agent(args.live, steps, failure)
        session = uuid.uuid4().hex
        for turn_number, query in enumerate(turns, start=1):
            result = agent.ask(query, session)
            lines.append(json.dumps({
                "scenario": name, "turn": turn_number, **labels, "query": query, "status": result.status,
                "tools_called": result.tools_called, "response": result.response,
                "trace": [e.model_dump(exclude_none=True) for e in result.trace],
            }, ensure_ascii=False))
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} traces to {args.output}")


if __name__ == "__main__":
    main()
