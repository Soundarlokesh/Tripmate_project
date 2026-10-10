"""LangGraph workflow tests.

The LLM is a scripted test double (tests/fakes.py): these tests prove the graph wiring, routing, tool
execution, validation, error handling and tracing. They do NOT prove that a live Groq model selects
the right tools; see tests/test_live.py for that.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.nodes import EMPTY_REPLY, LIMIT_REPLY, LLM_ERROR_REPLY
from tests.fakes import FakeWeatherProvider, answer_from_tool_results, tool_request
from app.tools.weather import WeatherProviderError

GUIDE = "search_destination_guide"
WEATHER = "get_weather_forecast"
SOON = (date.today() + timedelta(days=2)).isoformat()


def trace_of(result, event="tool_call"):
    return [e for e in result.trace if e.event == event]


def test_destination_only_uses_only_the_guide_tool(make_agent, weather_provider):
    agent, llm = make_agent([tool_request((GUIDE, {"query": "local customs in Tokyo"})), answer_from_tool_results])
    result = agent.ask("What are the local customs in Tokyo?")
    assert result.status == "answered"
    assert result.tools_called == [GUIDE]
    assert weather_provider.calls == []
    assert "Bowing is a common greeting" in result.response  # grounded in retrieved FAISS content
    call = trace_of(result)[0]
    assert call.arguments == {"query": "local customs in Tokyo"}
    assert call.status == "success" and call.duration_ms >= 0
    assert call.purpose and "Tokyo, Japan - Local Customs" in call.summary


def test_weather_only_passes_city_and_date_and_skips_the_guide(make_agent, weather_provider):
    agent, _ = make_agent([tool_request((WEATHER, {"city": "Tokyo", "date_or_month": SOON})), answer_from_tool_results])
    result = agent.ask(f"What will the weather be like in Tokyo on {SOON}?")
    assert result.tools_called == [WEATHER]
    assert weather_provider.calls[0][0] == "Tokyo" and weather_provider.calls[0][1].isoformat() == SOON
    assert '"temp_range_c": [4.0, 12.0]' in result.response
    assert '"source": "mock-test-data"' in result.response and '"is_live": false' in result.response


def test_packing_question_can_use_both_tools_in_one_step(make_agent):
    agent, llm = make_agent([
        tool_request((GUIDE, {"query": "packing tips for Tokyo"}), (WEATHER, {"city": "Tokyo", "date_or_month": "2027-01"})),
        answer_from_tool_results,
    ])
    result = agent.ask("I'm visiting Tokyo next January. What should I pack?")
    assert result.tools_called == [GUIDE, WEATHER]
    assert "Layered clothing" in result.response
    assert "climate_reference" in result.response and "Month-only" in result.response
    assert len(llm.seen) == 2  # one decision, one synthesis


def test_tools_can_be_chained_across_steps(make_agent):
    agent, llm = make_agent([
        tool_request((GUIDE, {"query": "best time to visit Reykjavik"})),
        tool_request((WEATHER, {"city": "Reykjavik", "date_or_month": SOON})),
        answer_from_tool_results,
    ])
    result = agent.ask("When should I go to Reykjavik and what's it like soon?")
    assert result.tools_called == [GUIDE, WEATHER]
    assert [e.node for e in result.trace] == ["input", "agent", "tools", "agent", "tools", "agent", "respond"]
    assert len(llm.seen) == 3


def test_greeting_needs_no_tools(make_agent, weather_provider):
    agent, llm = make_agent([AIMessage(content="Hi! I'm doing well. Where are you thinking of travelling?")])
    result = agent.ask("Hi, how are you?")
    assert result.status == "answered" and result.tools_called == []
    assert len(llm.seen) == 1 and weather_provider.calls == []
    assert [e.node for e in result.trace] == ["input", "agent", "respond"]


def test_out_of_scope_request_is_declined_without_tools(make_agent):
    agent, _ = make_agent([AIMessage(content="I can't book flights, but I can share visa and packing info for Tokyo.")])
    result = agent.ask("Book a flight to Tokyo.")
    assert result.tools_called == [] and "can't book" in result.response


def test_hallucinated_tool_is_rejected_and_never_executed(make_agent, weather_provider):
    agent, llm = make_agent([
        tool_request(("book_flight", {"destination": "Tokyo"})),
        AIMessage(content="Sorry, I can't book flights."),
    ])
    result = agent.ask("Book a flight to Tokyo.")
    assert result.tools_called == []  # nothing actually ran
    rejected = trace_of(result)[0]
    assert rejected.tool == "book_flight" and rejected.status == "rejected_unknown_tool"
    tool_message = next(m for m in llm.seen[1] if isinstance(m, ToolMessage))
    assert "Unknown tool 'book_flight'" in tool_message.content and tool_message.status == "error"
    assert weather_provider.calls == []


def test_missing_destination_triggers_clarification_not_tools(make_agent):
    agent, llm = make_agent([AIMessage(content="Where are you going, and when?")])
    result = agent.ask("What should I pack for my holiday?")
    assert result.tools_called == [] and result.response.endswith("when?")


def test_follow_up_uses_conversation_context(make_agent, weather_provider):
    agent, llm = make_agent([
        AIMessage(content="Tokyo is a great choice! What would you like to know?"),
        tool_request((WEATHER, {"city": "Tokyo", "date_or_month": SOON})),
        answer_from_tool_results,
    ])
    agent.ask("I'm planning a trip to Tokyo.", session_id="s1")
    result = agent.ask("What about the weather there?", session_id="s1")
    assert result.tools_called == [WEATHER]
    texts = [m.content for m in llm.seen[1] if isinstance(m, HumanMessage)]
    assert texts == ["I'm planning a trip to Tokyo.", "What about the weather there?"]
    assert weather_provider.calls[0][0] == "Tokyo"


def test_follow_up_without_a_date_can_ask_for_one(make_agent, weather_provider):
    agent, _ = make_agent([AIMessage(content="Nice!"), AIMessage(content="Which date or month will you be there?")])
    agent.ask("I'm planning a trip to Tokyo.", session_id="s1")
    result = agent.ask("What about the weather there?", session_id="s1")
    assert result.tools_called == [] and weather_provider.calls == []


def test_sessions_do_not_share_context(make_agent):
    agent, llm = make_agent([AIMessage(content="ok"), AIMessage(content="ok")])
    agent.ask("I'm planning a trip to Tokyo.", session_id="alice")
    agent.ask("What about the weather there?", session_id="bob")
    assert [m.content for m in llm.seen[1] if isinstance(m, HumanMessage)] == ["What about the weather there?"]


@pytest.mark.parametrize(
    ("tool", "args", "problem"),
    [
        (WEATHER, {"city": "Tokyo"}, "date_or_month"),  # missing argument
        (WEATHER, {"city": "  ", "date_or_month": SOON}, "city"),  # blank
        (GUIDE, {"query": "   "}, "query"),
        (GUIDE, {}, "query"),
    ],
)
def test_invalid_tool_arguments_are_rejected_before_execution(make_agent, weather_provider, tool, args, problem):
    agent, llm = make_agent([tool_request((tool, args)), AIMessage(content="Which date do you mean?")])
    result = agent.ask("weather please")
    event = trace_of(result)[0]
    assert event.status == "invalid_arguments" and problem in event.error
    assert result.tools_called == [] and weather_provider.calls == []
    assert result.response == "Which date do you mean?"  # the graph carried on after the rejection
    assert any(isinstance(m, ToolMessage) and "Invalid arguments" in m.content for m in llm.seen[1])


def test_weather_timeout_is_reported_and_guide_still_used(make_agent):
    provider = FakeWeatherProvider(failure=WeatherProviderError("weather service timed out after 10s"))
    agent, _ = make_agent([
        tool_request((GUIDE, {"query": "packing tips for Tokyo"}), (WEATHER, {"city": "Tokyo", "date_or_month": SOON})),
        answer_from_tool_results,
    ], provider=provider)
    result = agent.ask("What should I pack for Tokyo this week?")
    guide_event, weather_event = trace_of(result)
    assert guide_event.status == "success"
    assert weather_event.status == "error" and "timed out" in weather_event.error
    assert "timed out" in result.response and "Layered clothing" in result.response
    assert "temp_range_c" not in result.response  # no weather was invented
    assert result.status == "answered"


def test_unexpected_tool_exception_is_contained_and_traced(make_agent, caplog):
    class Exploding(FakeWeatherProvider):
        def geocode(self, city):
            raise RuntimeError("boom")

    agent, llm = make_agent([tool_request((WEATHER, {"city": "Tokyo", "date_or_month": SOON})),
                             AIMessage(content="I couldn't get the weather.")], provider=Exploding())
    with caplog.at_level(logging.ERROR):
        result = agent.ask("weather in Tokyo?")
    event = trace_of(result)[0]
    assert event.status == "error" and "RuntimeError" in event.error
    assert result.response == "I couldn't get the weather."
    assert any(r.exc_info for r in caplog.records)  # traceback is logged, not swallowed


def test_iteration_limit_stops_runaway_tool_loops(make_agent, weather_provider):
    agent, llm = make_agent([tool_request((WEATHER, {"city": "Tokyo", "date_or_month": SOON}))],
                            repeat_last=True, max_graph_steps=3)
    result = agent.ask("weather in Tokyo?", session_id="loop")
    assert result.status == "step_limit_reached" and result.response == LIMIT_REPLY
    assert len(llm.seen) == 3 and len(weather_provider.calls) == 2  # third request is not executed
    assert result.trace[-1].node == "limit"


def test_conversation_stays_valid_after_step_limit(make_agent):
    agent, llm = make_agent([tool_request((WEATHER, {"city": "Tokyo", "date_or_month": SOON})),
                             tool_request((WEATHER, {"city": "Tokyo", "date_or_month": SOON})),
                             AIMessage(content="Back to normal.")], max_graph_steps=1)
    agent.ask("loop", session_id="s")
    agent.ask("again", session_id="s")
    history = llm.seen[-1]
    requested = {c["id"] for m in history if isinstance(m, AIMessage) for c in m.tool_calls}
    answered = {m.tool_call_id for m in history if isinstance(m, ToolMessage)}
    assert requested == answered  # every tool call has a matching result, as the Groq API requires


def test_llm_failure_returns_friendly_error(make_agent):
    def fail(messages):
        raise httpx.ConnectError("network down")

    agent, _ = make_agent([fail])
    result = agent.ask("Hello")
    assert result.status == "llm_error" and result.response == LLM_ERROR_REPLY
    assert result.trace[-1].status == "error" and "ConnectError" in result.trace[-1].error


def test_empty_model_reply_gets_fallback(make_agent):
    agent, _ = make_agent([AIMessage(content="")])
    result = agent.ask("Hello")
    assert result.status == "llm_error" and result.response == EMPTY_REPLY


@pytest.mark.parametrize("query", ["", "   ", None, "x" * 2001])
def test_invalid_input_never_reaches_the_llm(make_agent, query):
    agent, llm = make_agent([])
    result = agent.ask(query)
    assert result.status == "invalid_input" and llm.seen == []


def test_list_content_blocks_are_flattened(make_agent):
    agent, _ = make_agent([AIMessage(content=[{"type": "text", "text": "Hello there"}])])
    assert agent.ask("hi").response == "Hello there"


def test_trace_is_consistent_logged_and_free_of_prompts(make_agent, caplog):
    agent, _ = make_agent([tool_request((GUIDE, {"query": "customs in Tokyo"})), answer_from_tool_results])
    with caplog.at_level(logging.INFO, logger="tripmate.trace"):
        result = agent.ask("What are the customs in Tokyo?")
    assert {e.request_id for e in result.trace} == {result.request_id}
    logged = [r.trace for r in caplog.records if r.name == "tripmate.trace"]
    assert [e["event"] for e in logged] == [e.event for e in result.trace]
    assert "You are TripMate" not in json.dumps(logged)


def test_long_history_is_trimmed_and_starts_on_a_user_message(make_agent):
    agent, llm = make_agent([AIMessage(content=f"reply {i}") for i in range(10)], max_history_messages=4)
    for i in range(10):
        agent.ask(f"question {i}", session_id="long")
    sent = [m for m in llm.seen[-1] if m.type != "system"]
    assert len(sent) <= 4 and sent[0].type == "human" and sent[-1].content == "question 9"
