"""Command-line entry point.

    python -m app.main                      # interactive chat
    python -m app.main --query "..."        # single question
    python -m app.main --show-trace         # also print the execution trace for each answer
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid

from app.agent.graph import TripMateAgent, build_llm
from app.config import ConfigError, Settings
from app.logging_config import setup_logging
from app.rag.embeddings import EmbeddingModelError, get_embeddings
from app.rag.vector_store import DestinationKnowledgeBase, IndexIntegrityError, IndexNotFoundError
from app.schemas.responses import AgentResult
from app.tools.registry import build_registry
from app.tools.weather import OpenMeteoProvider


def create_agent(settings: Settings) -> TripMateAgent:
    """Wire the live components: Groq LLM, FAISS knowledge base, Open-Meteo weather."""
    llm = build_llm(settings)
    knowledge_base = DestinationKnowledgeBase.load(settings, get_embeddings(settings))
    provider = OpenMeteoProvider(timeout_seconds=settings.weather_timeout_seconds)
    return TripMateAgent(llm, build_registry(settings, knowledge_base, provider), settings)


def print_result(result: AgentResult, show_trace: bool) -> None:
    print(f"\n{result.response}\n")
    if show_trace:
        print("--- trace ---")
        for event in result.trace:
            print(json.dumps(event.model_dump(exclude_none=True), ensure_ascii=False))
        print("-------------\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TripMate travel assistant")
    parser.add_argument("--query", help="ask a single question and exit")
    parser.add_argument("--show-trace", action="store_true", help="print the execution trace")
    parser.add_argument("--log-level", help="override LOG_LEVEL (structured JSON logs go to stderr)")
    args = parser.parse_args(argv)

    try:
        settings = Settings.from_env()
        setup_logging(args.log_level or settings.log_level)
        agent = create_agent(settings)
    except (ConfigError, EmbeddingModelError, IndexNotFoundError, IndexIntegrityError) as exc:
        print(f"Startup error: {exc}", file=sys.stderr)
        return 1

    session_id = uuid.uuid4().hex
    if args.query:
        print_result(agent.ask(args.query, session_id), args.show_trace)
        return 0
    print("TripMate - ask about visas, customs, packing, safety or weather. Type 'exit' to quit.")
    while True:
        try:
            query = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if query.lower() in {"exit", "quit"}:
            return 0
        print_result(agent.ask(query, session_id), args.show_trace)


if __name__ == "__main__":
    raise SystemExit(main())
