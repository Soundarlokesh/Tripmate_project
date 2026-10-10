from __future__ import annotations

import pytest

from app.config import ConfigError, Settings

ENV_KEYS = ["GROQ_API_KEY", "GROQ_MODEL", "EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "TOP_K", "RELEVANCE_THRESHOLD",
            "MAX_GRAPH_STEPS", "WEATHER_PROVIDER", "LOG_LEVEL", "DATA_DIR", "FAISS_INDEX_PATH",
            "LLM_TIMEOUT_SECONDS", "WEATHER_TIMEOUT_SECONDS", "MAX_HISTORY_MESSAGES"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return tmp_path / "no.env"  # nonexistent env file: ignores any developer .env


def test_defaults_need_no_groq_key(clean_env):
    settings = Settings.from_env(clean_env)
    assert settings.groq_model == "llama-3.3-70b-versatile" and settings.top_k == 3
    assert settings.groq_api_key == ""


def test_groq_key_required_only_when_requested(clean_env):
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        Settings.from_env(clean_env).require_groq_key()


def test_api_key_is_not_in_repr(monkeypatch, clean_env):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_secret_value")
    settings = Settings.from_env(clean_env)
    assert settings.require_groq_key() == "gsk_secret_value" and "gsk_secret_value" not in repr(settings)


def test_env_overrides_and_per_provider_threshold(monkeypatch, clean_env):
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline-hash")
    settings = Settings.from_env(clean_env)
    assert settings.groq_model == "openai/gpt-oss-20b" and settings.relevance_threshold == 0.10
    monkeypatch.setenv("RELEVANCE_THRESHOLD", "0.4")
    assert Settings.from_env(clean_env).relevance_threshold == 0.4


@pytest.mark.parametrize(
    ("name", "value"),
    [("TOP_K", "0"), ("TOP_K", "abc"), ("MAX_GRAPH_STEPS", "0"), ("RELEVANCE_THRESHOLD", "1.5"),
     ("EMBEDDING_PROVIDER", "openai"), ("WEATHER_PROVIDER", "mock"), ("WEATHER_TIMEOUT_SECONDS", "0")],
)
def test_invalid_values_are_rejected(monkeypatch, clean_env, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError):
        Settings.from_env(clean_env)
