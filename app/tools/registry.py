"""LangChain tool definitions (what the LLM sees) and the registry the graph executes from."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, field_validator

from app.config import Settings
from app.rag.vector_store import DestinationKnowledgeBase
from app.tools.destination_search import search_destination_guide
from app.tools.weather import WeatherProvider, get_weather_forecast


class DestinationSearchArgs(BaseModel):
    query: str = Field(
        description=(
            "Natural-language question or topic, including the city name when known, "
            "e.g. 'local customs in Tokyo' or 'packing tips for Reykjavik in winter'."
        ),
        max_length=500,
    )

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class WeatherArgs(BaseModel):
    city: str = Field(description="City name only, e.g. 'Tokyo' (no country or extra words).", max_length=100)
    date_or_month: str = Field(
        description=(
            "A specific date as YYYY-MM-DD, or a month as YYYY-MM or a month name such as 'November'. "
            "Convert relative phrases like 'next month' or 'this Friday' using today's date."
        ),
        max_length=40,
    )

    @field_validator("city", "date_or_month")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


@dataclass(frozen=True)
class ToolSpec:
    tool: BaseTool
    purpose: str  # short, user-safe explanation recorded in traces
    summarize: Callable[[Any], str]  # sanitized one-line result summary for traces


def _destination_summary(result: list[str]) -> str:
    headings = [item.split(":", 1)[0] for item in result if not item.startswith("[note]")]
    note = " + coverage note" if any(item.startswith("[note]") for item in result) else ""
    return f"{len(headings)} chunk(s): {'; '.join(headings)}{note}" if headings else f"no relevant chunks{note}"


def _weather_summary(result: dict) -> str:
    if result.get("status") != "ok":
        return f"error ({result.get('error_type')}): {result.get('message')}"
    return (
        f"{result.get('data_kind')} from {result.get('source')} (live={result.get('is_live')}), "
        f"{result.get('conditions')}, temp_range_c={result.get('temp_range_c')}"
    )


def build_registry(
    settings: Settings, knowledge_base: DestinationKnowledgeBase, weather_provider: WeatherProvider
) -> dict[str, ToolSpec]:
    """Create the only tools the agent may execute, keyed by tool name."""
    destinations = ", ".join(knowledge_base.destinations)

    def destination_tool(query: str) -> list[str]:
        chunks = search_destination_guide(
            query, knowledge_base, top_k=settings.top_k, threshold=settings.relevance_threshold
        )
        if not chunks:
            note = "No relevant information found in the knowledge base."
        elif not knowledge_base.detect_destinations(query):
            note = "No specific supported destination was named in the query, so results may be from any city."
        else:
            return chunks
        return [*chunks, f"[note] {note} The knowledge base covers only: {destinations}. If the user asked "
                         "about another place, say it is not covered instead of reusing these results."]

    def weather_tool(city: str, date_or_month: str) -> dict:
        return get_weather_forecast(city, date_or_month, weather_provider)

    destination = StructuredTool.from_function(
        func=destination_tool,
        name="search_destination_guide",
        description=(
            "Search the TripMate destination guide knowledge base. Covers ONLY these destinations: "
            f"{destinations}. Topics: visa and entry requirements, best time to visit, local customs, "
            "packing tips, safety and health. Use it for destination-specific facts and advice. "
            "Returns text chunks labelled '<City>, <Country> - <Section>'. It contains no live weather, "
            "prices, flights or bookings."
        ),
        args_schema=DestinationSearchArgs,
    )
    weather = StructuredTool.from_function(
        func=weather_tool,
        name="get_weather_forecast",
        description=(
            "Get weather for a city on a specific date or in a month (Open-Meteo). Real forecasts exist only "
            "for dates from today up to 15 days ahead; past dates return observed weather. Month-only "
            "requests and dates further ahead return a labelled 'climate_reference' built from last year's "
            "observations, which is NOT a forecast. Needs both a city and a date or month; do not call it "
            "if either is unknown. On status 'error', no weather data is available."
        ),
        args_schema=WeatherArgs,
    )
    return {
        "search_destination_guide": ToolSpec(
            destination,
            "Fetch destination-specific guidance from the knowledge base",
            _destination_summary,
        ),
        "get_weather_forecast": ToolSpec(
            weather,
            "Fetch weather data for the requested city and date or month",
            _weather_summary,
        ),
    }


def serialize_tool_result(result: Any) -> str:
    return json.dumps(result, ensure_ascii=False, default=str)
