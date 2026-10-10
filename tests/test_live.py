"""Live tests: real Groq, Open-Meteo and Hugging Face. Skipped unless RUN_LIVE_TESTS=1 and GROQ_API_KEY is set.

    RUN_LIVE_TESTS=1 pytest tests/test_live.py -v -s
Requires network access, a built index (python -m scripts.ingest_data) and the HF model download.
"""

from __future__ import annotations

import os

import pytest

from app.config import Settings

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("RUN_LIVE_TESTS") != "1" or not os.environ.get("GROQ_API_KEY"),
        reason="set RUN_LIVE_TESTS=1 and GROQ_API_KEY to run live tests",
    ),
]


@pytest.fixture(scope="module")
def live_agent():
    from app.main import create_agent

    return create_agent(Settings.from_env())


def test_destination_question_uses_guide_only(live_agent):
    result = live_agent.ask("What are the local customs in Tokyo?")
    assert result.tools_called == ["search_destination_guide"]


def test_weather_question_uses_weather_only(live_agent):
    from datetime import date, timedelta

    result = live_agent.ask(f"What will the weather be like in Tokyo on {date.today() + timedelta(days=3)}?")
    assert result.tools_called == ["get_weather_forecast"]


def test_packing_question_uses_both(live_agent):
    result = live_agent.ask("I'm visiting Tokyo next month. What should I pack?")
    assert set(result.tools_called) == {"search_destination_guide", "get_weather_forecast"}


def test_greeting_and_booking_use_no_tools(live_agent):
    assert live_agent.ask("Hi, how are you?").tools_called == []
    assert live_agent.ask("Book a flight to Tokyo.").tools_called == []


def test_missing_destination_asks_for_clarification(live_agent):
    result = live_agent.ask("What should I pack for my holiday?")
    assert result.tools_called == [] and "?" in result.response
