"""API tests with an injected agent (scripted LLM test double; no network, no keys)."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.api import create_app
from tests.fakes import answer_from_tool_results, tool_request


@pytest.fixture
def client_for(make_agent):
    def factory(script):
        agent, llm = make_agent(script)
        return TestClient(create_app(agent)), llm

    return factory


def test_health(client_for):
    with client_for([])[0] as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_swagger_and_openapi_are_served(client_for):
    with client_for([])[0] as client:
        assert client.get("/docs").status_code == 200
        paths = client.get("/openapi.json").json()["paths"]
        assert "/chat" in paths and "post" in paths["/chat"]


def test_chat_runs_the_agent_and_returns_trace(client_for):
    script = [tool_request(("search_destination_guide", {"query": "customs in Tokyo"})), answer_from_tool_results]
    with client_for(script)[0] as client:
        response = client.post("/chat", json={"message": "What are the local customs in Tokyo?"})
    body = response.json()
    assert response.status_code == 200 and body["status"] == "answered"
    assert body["tools_called"] == ["search_destination_guide"] and "Bowing" in body["response"]
    assert any(event["event"] == "tool_call" for event in body["trace"])


def test_include_trace_false_omits_trace(client_for):
    with client_for([AIMessage(content="Hello!")])[0] as client:
        body = client.post("/chat?include_trace=false", json={"message": "hi"}).json()
    assert body["trace"] == [] and body["response"] == "Hello!"


def test_session_id_keeps_context(client_for):
    client, llm = client_for([AIMessage(content="ok"), AIMessage(content="ok")])
    with client as c:
        c.post("/chat", json={"message": "I'm planning a trip to Tokyo.", "session_id": "s-1"})
        c.post("/chat", json={"message": "What about the weather there?", "session_id": "s-1"})
    assert [m.content for m in llm.seen[1] if m.type == "human"] == [
        "I'm planning a trip to Tokyo.", "What about the weather there?"]


@pytest.mark.parametrize(
    "payload",
    [{}, {"message": ""}, {"message": "   "}, {"message": "x" * 2001}, {"message": "hi", "session_id": "bad id!"}],
)
def test_invalid_requests_return_422_without_calling_the_llm(client_for, payload):
    client, llm = client_for([])
    with client as c:
        assert c.post("/chat", json=payload).status_code == 422
    assert llm.seen == []


def test_llm_failure_maps_to_502(client_for):
    def fail(messages):
        raise httpx.ConnectError("down")

    with client_for([fail])[0] as client:
        response = client.post("/chat", json={"message": "hi"})
    assert response.status_code == 502 and response.json()["status"] == "llm_error"