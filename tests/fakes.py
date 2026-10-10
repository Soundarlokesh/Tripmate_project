"""Test doubles. Everything here is explicitly fake and is labelled as such in results.

ScriptedChatModel replays pre-written AI messages, so tests verify the graph, routing, tool
execution and tracing. They do NOT show that a real Groq model chooses the right tools.
"""

from __future__ import annotations

import uuid
from datetime import date, timedelta
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from app.schemas.responses import DailyWeather, Location
from app.tools.weather import LocationNotFoundError, WeatherProviderError


class ScriptedChatModel(BaseChatModel):
    """Chat model test double: returns scripted responses in order and records what it was shown."""

    script: list[Any]  # AIMessage, or a callable(messages) -> AIMessage
    repeat_last: bool = False
    seen: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-test-double"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(list(messages))
        if not self.script:
            raise AssertionError("ScriptedChatModel script exhausted: the graph made an unexpected LLM call")
        repeating = self.repeat_last and len(self.script) == 1
        item = self.script[0] if repeating else self.script.pop(0)
        message = item(messages) if callable(item) else item
        if repeating and isinstance(message, AIMessage):
            # Fresh ids per call, as a real model would produce; reused ids would overwrite state messages.
            message = message.model_copy(update={
                "id": None,
                "tool_calls": [{**c, "id": f"call_{uuid.uuid4().hex[:8]}"} for c in message.tool_calls],
            })
        return ChatResult(generations=[ChatGeneration(message=message)])


def tool_request(*calls: tuple[str, dict]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": n, "args": a, "id": f"call_{uuid.uuid4().hex[:8]}"} for n, a in calls],
    )


def answer_from_tool_results(messages: list[BaseMessage]) -> AIMessage:
    """Compose a final answer that quotes every tool result of the current turn."""
    last_human = max(i for i, m in enumerate(messages) if m.type == "human")
    results = [m.content for m in messages[last_human:] if isinstance(m, ToolMessage)]
    return AIMessage(content="Here is what I found:\n" + "\n".join(results))


class FakeWeatherProvider:
    """Deterministic, offline stand-in for Open-Meteo. Labelled 'mock-test-data' in results."""

    name = "mock-test-data"
    is_live = False

    def __init__(
        self,
        failure: WeatherProviderError | None = None,
        known_cities: tuple[str, ...] = ("tokyo", "reykjavik", "bangkok", "barcelona"),
        days_available: int | None = None,
    ) -> None:
        self.failure = failure
        self.known_cities = known_cities
        self.days_available = days_available
        self.calls: list[tuple[str, date, date, str]] = []

    def geocode(self, city: str) -> Location:
        if self.failure:
            raise self.failure
        if city.lower() not in self.known_cities:
            raise LocationNotFoundError(f"no location found for {city!r}")
        return Location(name=city.title(), country="Testland", latitude=35.0, longitude=139.0)

    def daily(
        self, location: Location, start: date, end: date, endpoint: Literal["forecast", "archive"]
    ) -> list[DailyWeather]:
        if self.failure:
            raise self.failure
        self.calls.append((location.name, start, end, endpoint))
        count = (end - start).days + 1
        if self.days_available is not None:
            count = min(count, self.days_available)
        return [
            DailyWeather(day=start + timedelta(days=i), temp_max_c=12.0, temp_min_c=4.0,
                         precipitation_mm=2.0 if i % 3 == 0 else 0.0, weather_code=3)
            for i in range(count)
        ]
