"""Centralised configuration, loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Minimum cosine similarity per embedding provider. See README ("Relevance threshold").
DEFAULT_RELEVANCE_THRESHOLDS = {
    "offline-hash": 0.10,
    "huggingface": 0.25,
}
EMBEDDING_PROVIDERS = tuple(DEFAULT_RELEVANCE_THRESHOLDS)


class ConfigError(ValueError):
    """Raised when configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = field(default="", repr=False)
    groq_model: str = "llama-3.3-70b-versatile"
    llm_timeout_seconds: float = 30.0
    embedding_provider: str = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    data_dir: Path = PROJECT_ROOT / "data" / "destinations"
    faiss_index_path: Path = PROJECT_ROOT / "data" / "index" / "destinations"
    top_k: int = 3
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLDS["huggingface"]
    weather_provider: str = "open-meteo"
    weather_timeout_seconds: float = 10.0
    max_graph_steps: int = 10
    max_history_messages: int = 30
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> Settings:
        """Build settings from the environment. Does not require a Groq key."""
        load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
        provider = _get("EMBEDDING_PROVIDER", "huggingface").lower()
        if provider not in EMBEDDING_PROVIDERS:
            raise ConfigError(
                f"EMBEDDING_PROVIDER must be one of {EMBEDDING_PROVIDERS}, got {provider!r}"
            )
        weather_provider = _get("WEATHER_PROVIDER", "open-meteo").lower()
        if weather_provider != "open-meteo":
            raise ConfigError("WEATHER_PROVIDER must be 'open-meteo'")
        threshold_raw = _get("RELEVANCE_THRESHOLD", "")
        threshold = (
            _number("RELEVANCE_THRESHOLD", threshold_raw, float)
            if threshold_raw
            else DEFAULT_RELEVANCE_THRESHOLDS[provider]
        )
        if not 0.0 <= threshold <= 1.0:
            raise ConfigError("RELEVANCE_THRESHOLD must be between 0 and 1")
        settings = cls(
            groq_api_key=_get("GROQ_API_KEY", ""),
            groq_model=_get("GROQ_MODEL", cls.groq_model),
            llm_timeout_seconds=_number("LLM_TIMEOUT_SECONDS", _get("LLM_TIMEOUT_SECONDS", "30"), float),
            embedding_provider=provider,
            embedding_model=_get("EMBEDDING_MODEL", cls.embedding_model),
            data_dir=_path(_get("DATA_DIR", ""), cls.data_dir),
            faiss_index_path=_path(_get("FAISS_INDEX_PATH", ""), cls.faiss_index_path),
            top_k=_number("TOP_K", _get("TOP_K", "3"), int),
            relevance_threshold=threshold,
            weather_provider=weather_provider,
            weather_timeout_seconds=_number(
                "WEATHER_TIMEOUT_SECONDS", _get("WEATHER_TIMEOUT_SECONDS", "10"), float
            ),
            max_graph_steps=_number("MAX_GRAPH_STEPS", _get("MAX_GRAPH_STEPS", "10"), int),
            max_history_messages=_number(
                "MAX_HISTORY_MESSAGES", _get("MAX_HISTORY_MESSAGES", "30"), int
            ),
            log_level=_get("LOG_LEVEL", "INFO").upper(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.top_k < 1:
            raise ConfigError("TOP_K must be >= 1")
        if self.max_graph_steps < 1:
            raise ConfigError("MAX_GRAPH_STEPS must be >= 1")
        if self.max_history_messages < 2:
            raise ConfigError("MAX_HISTORY_MESSAGES must be >= 2")
        if self.llm_timeout_seconds <= 0 or self.weather_timeout_seconds <= 0:
            raise ConfigError("Timeout settings must be positive")

    def require_groq_key(self) -> str:
        """Return the Groq key or raise a clear error. Called only when a live LLM is built."""
        if not self.groq_api_key:
            raise ConfigError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
                "(https://console.groq.com/keys)."
            )
        return self.groq_api_key


def _get(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _number(name: str, raw: str, cast: type) -> int | float:
    try:
        return cast(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a valid {cast.__name__}, got {raw!r}") from exc


def _path(raw: str, default: Path) -> Path:
    if not raw:
        return default
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path
