"""Graph state. Only what the workflow needs; tool results live in the message history."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # conversation, kept per session thread
    request_id: str  # per-turn fields below are reset on every call to TripMateAgent.ask
    steps: int  # number of LLM calls made this turn
    trace: list[dict]  # TraceEvent dicts for this turn
    status: str  # answered | step_limit_reached | llm_error
    final_response: str
