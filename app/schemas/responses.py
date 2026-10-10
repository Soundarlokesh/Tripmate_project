"""Pydantic models shared across the weather tool, tracing and the agent result."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

# --- Weather -----------------------------------------------------------------------------------

DataKind = Literal["forecast", "observed_historical", "climate_reference"]


class Location(BaseModel):
    name: str
    country: str = ""
    region: str = ""
    latitude: float
    longitude: float


class DailyWeather(BaseModel):
    day: date
    temp_max_c: float
    temp_min_c: float
    precipitation_mm: float
    weather_code: int | None = None


class WeatherResult(BaseModel):
    """Result returned to the LLM. `status == "error"` carries no weather values by design."""

    status: Literal["ok", "error"]
    city: str
    source: str = Field(description="'open-meteo' for live API data, 'mock-test-data' for test doubles")
    is_live: bool
    data_kind: DataKind | None = None
    location: Location | None = None
    period_start: date | None = None
    period_end: date | None = None
    reference_period: str | None = Field(
        default=None, description="For climate_reference: the past period the numbers come from"
    )
    conditions: str | None = None
    temp_range_c: list[float] | None = Field(default=None, description="[lowest min, highest max]")
    avg_high_c: float | None = None
    avg_low_c: float | None = None
    total_precipitation_mm: float | None = None
    rainy_days: int | None = None
    days_with_data: int | None = None
    limitations: list[str] = Field(default_factory=list)
    error_type: str | None = None
    message: str | None = None


# --- Tracing / agent output ---------------------------------------------------------------------


class TraceEvent(BaseModel):
    request_id: str
    node: str
    event: str
    tool: str | None = None
    arguments: dict | None = None
    purpose: str | None = None
    status: str | None = None
    duration_ms: float | None = None
    summary: str | None = None
    error: str | None = None


class AgentResult(BaseModel):
    request_id: str
    session_id: str
    status: Literal["answered", "step_limit_reached", "llm_error", "invalid_input"]
    response: str
    tools_called: list[str] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
