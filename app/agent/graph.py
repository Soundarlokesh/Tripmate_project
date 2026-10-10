"""LangGraph assembly and the TripMateAgent facade."""

from __future__ import annotations

import logging
import uuid
from functools import partial

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    LIMIT_REPLY,
    limit_node,
    make_agent_node,
    make_tool_node,
    record,
    respond_node,
)
from app.agent.routing import route_after_agent
from app.agent.state import AgentState
from app.config import ConfigError, Settings
from app.schemas.responses import AgentResult, TraceEvent
from app.tools.registry import ToolSpec

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 2000
EXECUTED = {"success", "error"}  # tool trace statuses meaning the tool actually ran


def build_llm(settings: Settings) -> BaseChatModel:
    """Create the Groq chat model. Requires GROQ_API_KEY; checks the model supports tool calling."""
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model=settings.groq_model,
        api_key=settings.require_groq_key(),
        temperature=0,
        timeout=settings.llm_timeout_seconds,
        max_retries=2,
    )
    profile = llm.profile or {}
    if profile.get("tool_calling") is False:
        raise ConfigError(f"GROQ_MODEL {settings.groq_model!r} does not support tool calling")
    if "tool_calling" not in profile:
        logger.warning("model capabilities unknown to langchain-groq; tool calling not verified",
                       extra={"model": settings.groq_model})
    return llm


def build_graph(llm: BaseChatModel, registry: dict[str, ToolSpec], settings: Settings, checkpointer=None):
    """agent -> (tools -> agent)* -> respond, with step-limit and LLM-error exits."""
    llm_with_tools = llm.bind_tools([spec.tool for spec in registry.values()])
    graph = StateGraph(AgentState)
    graph.add_node("agent", make_agent_node(llm_with_tools, settings))
    graph.add_node("tools", make_tool_node(registry))
    graph.add_node("respond", respond_node)
    graph.add_node("limit", limit_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        partial(route_after_agent, max_steps=settings.max_graph_steps),
        {"tools": "tools", "respond": "respond", "limit": "limit", "end": END},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("respond", END)
    graph.add_edge("limit", END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())


class TripMateAgent:
    """Runs one user turn through the graph. Conversation memory is scoped to `session_id`."""

    def __init__(self, llm: BaseChatModel, registry: dict[str, ToolSpec], settings: Settings) -> None:
        self._settings = settings
        self._graph = build_graph(llm, registry, settings)

    def ask(self, query: str, session_id: str | None = None) -> AgentResult:
        session_id = session_id or uuid.uuid4().hex
        request_id = uuid.uuid4().hex[:12]
        cleaned = (query or "").strip()
        if not cleaned or len(cleaned) > MAX_QUERY_CHARS:
            reason = "Please type a question about a destination." if not cleaned else (
                f"That message is too long (max {MAX_QUERY_CHARS} characters)."
            )
            event = TraceEvent(request_id=request_id, node="input", event="validation", status="rejected", error=reason)
            logger.info("trace", extra={"trace": event.model_dump(exclude_none=True)})
            return AgentResult(request_id=request_id, session_id=session_id, status="invalid_input",
                               response=reason, trace=[event])

        trace = record(request_id, [], node="input", event="request_received", summary=cleaned[:200])
        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": self._settings.max_graph_steps * 2 + 5,  # backstop behind route_after_agent
        }
        try:
            final = self._graph.invoke(
                {"messages": [HumanMessage(cleaned)], "request_id": request_id, "steps": 0, "trace": trace,
                 "status": "", "final_response": ""},
                config,
            )
        except GraphRecursionError:
            logger.error("recursion limit hit", extra={"request_id": request_id})
            return AgentResult(request_id=request_id, session_id=session_id, status="step_limit_reached",
                               response=LIMIT_REPLY)
        events = [TraceEvent(**e) for e in final["trace"]]
        return AgentResult(
            request_id=request_id,
            session_id=session_id,
            status=final["status"],
            response=final["final_response"],
            tools_called=[e.tool for e in events if e.event == "tool_call" and e.status in EXECUTED],
            trace=events,
        )
