"""HTTP API with Swagger UI.

    uvicorn app.api:app --reload
    open http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.agent.graph import MAX_QUERY_CHARS, TripMateAgent
from app.config import Settings
from app.logging_config import setup_logging
from app.main import create_agent
from app.schemas.responses import AgentResult


class ChatRequest(BaseModel):
    message: str = Field(
        max_length=MAX_QUERY_CHARS,
        description="The user's question.",
        examples=["I'm visiting Tokyo next month. What should I pack?"],
    )
    session_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_-]{1,64}$",
        description="Reuse the same id across calls to keep conversation context. Omit to start a new session.",
        examples=["demo-session-1"],
    )

    @field_validator("message")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value


def create_app(agent: TripMateAgent | None = None) -> FastAPI:
    """Build the app. Pass an agent to inject one (tests); otherwise live components are created at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if agent is not None:
            app.state.agent = agent
        else:
            settings = Settings.from_env()
            setup_logging(settings.log_level)
            app.state.agent = create_agent(settings)  # raises a clear error if key/index is missing
        yield

    app = FastAPI(
        title="TripMate AI Agent",
        version="0.1.0",
        description="Ask travel questions. The agent decides whether to use the destination guide, "
        "the weather tool, both, or neither.",
        lifespan=lifespan,
    )

    @app.get("/health", tags=["system"])
    def health() -> dict:
        return {"status": "ok"}

    @app.post(
        "/chat",
        response_model=AgentResult,
        tags=["agent"],
        responses={502: {"model": AgentResult, "description": "The language model could not be reached"}},
    )
    def chat(
        body: ChatRequest,
        request: Request,
        include_trace: bool = Query(True, description="Include the execution trace in the response"),
    ):
        result: AgentResult = request.app.state.agent.ask(body.message, body.session_id)
        if not include_trace:
            result = result.model_copy(update={"trace": []})
        if result.status == "llm_error":
            return JSONResponse(status_code=502, content=result.model_dump(mode="json"))
        return result

    return app


# Importable as `uvicorn app.api:app`. Nothing heavy runs at import: the agent is built at server startup.
app = create_app()