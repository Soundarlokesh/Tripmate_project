from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.routing import route_after_agent


def state(last, steps=1, status=""):
    return {"messages": [HumanMessage("hi"), last], "steps": steps, "status": status}


TOOL_REQUEST = AIMessage(content="", tool_calls=[{"name": "get_weather_forecast", "args": {}, "id": "1"}])


def test_tool_calls_route_to_tools():
    assert route_after_agent(state(TOOL_REQUEST), max_steps=3) == "tools"


def test_plain_answer_routes_to_respond():
    assert route_after_agent(state(AIMessage(content="Hello!")), max_steps=3) == "respond"


def test_step_budget_routes_to_limit():
    assert route_after_agent(state(TOOL_REQUEST, steps=3), max_steps=3) == "limit"
    assert route_after_agent(state(TOOL_REQUEST, steps=2), max_steps=3) == "tools"


def test_llm_error_ends_the_run():
    assert route_after_agent(state(AIMessage(content="sorry"), status="llm_error"), max_steps=3) == "end"
