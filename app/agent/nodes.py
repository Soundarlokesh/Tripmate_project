"""Graph nodes: agent (LLM call), tools (validated execution), respond, limit."""

from __future__ import annotations

import logging
import time
from datetime import date
from typing import Any

import groq
import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, trim_messages
from pydantic import ValidationError

from app.agent.prompts import build_system_prompt
from app.agent.state import AgentState
from app.config import Settings
from app.schemas.responses import TraceEvent
from app.tools.registry import ToolSpec, serialize_tool_result

logger = logging.getLogger("tripmate.trace")

LLM_ERRORS = (groq.GroqError, httpx.HTTPError)
LLM_ERROR_REPLY = "Sorry, I couldn't reach the language model just now. Please try again in a moment."
EMPTY_REPLY = "Sorry, I wasn't able to produce an answer. Could you rephrase your question?"
LIMIT_REPLY = (
    "I couldn't finish answering within the allowed number of steps. "
    "Please try a more specific question, for example one city and one date at a time."
)


def record(request_id: str, trace: list[dict], **fields: Any) -> list[dict]:
    """Return `trace` plus a new event, and emit that event as one structured log line."""
    event = TraceEvent(request_id=request_id, **fields).model_dump(exclude_none=True)
    logger.info("trace", extra={"trace": event})
    return [*trace, event]


def message_text(message: AIMessage) -> str:
    """AIMessage.content may be a string or a list of content blocks."""
    if isinstance(message.content, str):
        return message.content.strip()
    return " ".join(
        block.get("text", "") if isinstance(block, dict) else str(block) for block in message.content
    ).strip()


def make_agent_node(llm_with_tools: BaseChatModel, settings: Settings):
    def agent_node(state: AgentState) -> dict:
        history = trim_messages(
            state["messages"],
            max_tokens=settings.max_history_messages,
            token_counter=len,  # count messages, not tokens
            strategy="last",
            start_on="human",
            include_system=False,
        )
        started = time.perf_counter()
        steps = state["steps"] + 1
        try:
            response = llm_with_tools.invoke([SystemMessage(build_system_prompt(date.today())), *history])
        except LLM_ERRORS as exc:
            logger.error("llm call failed", extra={"request_id": state["request_id"]}, exc_info=True)
            trace = record(
                state["request_id"], state["trace"], node="agent", event="llm_call", status="error",
                duration_ms=_ms(started), error=f"{type(exc).__name__}: {exc}"[:300],
            )
            return {
                "steps": steps, "status": "llm_error", "final_response": LLM_ERROR_REPLY, "trace": trace,
                "messages": [AIMessage(content=LLM_ERROR_REPLY)],
            }
        if response.tool_calls:
            names = ", ".join(call["name"] for call in response.tool_calls)
            summary = f"model requested tools: {names}"
        else:
            summary = "model answered directly (no tool calls)"
        trace = record(
            state["request_id"], state["trace"], node="agent", event="llm_call", status="ok", duration_ms=_ms(started), summary=summary
        )
        return {"messages": [response], "steps": steps, "trace": trace}

    return agent_node


def make_tool_node(registry: dict[str, ToolSpec]):
    def tool_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        trace = state["trace"]
        results: list[ToolMessage] = []
        for call in last.tool_calls:
            started = time.perf_counter()
            name, args = call["name"], call.get("args") or {}
            content, status, summary, error = _execute(registry, name, args)
            results.append(
                ToolMessage(
                    content=content, tool_call_id=call["id"], name=name,
                    status="success" if status == "success" else "error",
                )
            )
            spec = registry.get(name)
            trace = record(
                state["request_id"], trace, node="tools", event="tool_call", tool=name, arguments=args,
                purpose=spec.purpose if spec else None, status=status,
                duration_ms=_ms(started), summary=summary, error=error,
            )
        return {"messages": results, "trace": trace}

    return tool_node


def _execute(registry: dict[str, ToolSpec], name: str, args: dict) -> tuple[str, str, str | None, str | None]:
    """Run one tool call. Returns (tool message content, status, trace summary, trace error)."""
    spec = registry.get(name)
    if spec is None:
        msg = f"Unknown tool {name!r}. Available tools: {', '.join(registry)}."
        return msg, "rejected_unknown_tool", None, msg
    try:
        validated = spec.tool.args_schema.model_validate(args)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        msg = f"Invalid arguments for {name}: {problems}"
        return msg, "invalid_arguments", None, msg
    try:
        result = spec.tool.invoke(validated.model_dump())
    except Exception as exc:  # noqa: BLE001 - tool boundary: log with traceback, report to the LLM
        logger.error("tool raised", extra={"tool": name}, exc_info=True)
        msg = f"Tool {name} failed: {type(exc).__name__}: {exc}"
        return msg, "error", None, msg
    status = "error" if isinstance(result, dict) and result.get("status") == "error" else "success"
    error = result.get("message") if status == "error" else None
    return serialize_tool_result(result), status, spec.summarize(result), error


def respond_node(state: AgentState) -> dict:
    text = message_text(state["messages"][-1])
    if not text:
        trace = record(state["request_id"], state["trace"], node="respond", event="final_response", status="error", error="empty model reply")
        return {"final_response": EMPTY_REPLY, "status": "llm_error", "trace": trace}
    trace = record(state["request_id"], state["trace"], node="respond", event="final_response", status="answered", summary=f"{len(text)} chars")
    return {"final_response": text, "status": "answered", "trace": trace}


def limit_node(state: AgentState) -> dict:
    """Step budget spent while the model still wants tools: close the dangling calls and stop."""
    last = state["messages"][-1]
    skipped = [
        ToolMessage(content="Not executed: step limit reached.", tool_call_id=c["id"], name=c["name"], status="error")
        for c in last.tool_calls
    ]
    trace = record(
        state["request_id"], state["trace"], node="limit", event="final_response", status="step_limit_reached",
        error=f"stopped after {state['steps']} model calls",
    )
    return {
        "messages": [*skipped, AIMessage(content=LIMIT_REPLY)],
        "final_response": LIMIT_REPLY, "status": "step_limit_reached", "trace": trace,
    }


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
