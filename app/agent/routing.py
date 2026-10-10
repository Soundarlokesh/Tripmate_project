"""Conditional routing for the graph."""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage

from app.agent.state import AgentState

Route = Literal["tools", "respond", "limit", "end"]


def route_after_agent(state: AgentState, max_steps: int) -> Route:
    """Decide what follows an LLM call.

    - llm_error  -> end (the agent node already produced the user-facing message)
    - tool calls -> tools, unless the step budget is spent -> limit
    - otherwise  -> respond (the LLM answered, asked a clarifying question, or declined)
    """
    if state.get("status") == "llm_error":
        return "end"
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "limit" if state["steps"] >= max_steps else "tools"
    return "respond"
