"""Tests for configuration loading and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from app.core.config import AppEnv, Settings, get_settings


def test_defaults_are_usable_without_any_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean machine with no .env should still produce a runnable configuration."""
    for name in ("APP_ENV", "DATABASE_URL", "LOG_LEVEL", "API_PREFIX"):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.app_env is AppEnv.DEVELOPMENT
    assert settings.database_url.startswith("sqlite+aiosqlite://")
    assert settings.api_prefix == "/api"
    assert settings.is_production is False


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "10")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.app_env is AppEnv.PRODUCTION
    assert settings.is_production is True
    assert settings.port == 9000
    assert settings.max_upload_size_mb == 10


def test_log_level_is_normalised_to_upper_case(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "debug")
    assert Settings(_env_file=None).log_level == "DEBUG"  # type: ignore[call-arg]


def test_invalid_log_level_is_rejected_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Misconfiguration must fail loudly at load time, not at first use."""
    monkeypatch.setenv("LOG_LEVEL", "chatty")

    with pytest.raises(PydanticValidationError, match="log_level must be one of"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_api_prefix_must_start_with_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_PREFIX", "api")

    with pytest.raises(PydanticValidationError, match="must start with"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_api_prefix_trailing_slash_is_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevents routes being registered at '//documents'."""
    monkeypatch.setenv("API_PREFIX", "/api/")
    assert Settings(_env_file=None).api_prefix == "/api"  # type: ignore[call-arg]


def test_upload_size_bounds_are_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "0")
    with pytest.raises(PydanticValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]

    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "10000")
    with pytest.raises(PydanticValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_cors_origins_parsed_into_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "http://a.test , http://b.test,, ")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.cors_origin_list == ["http://a.test", "http://b.test"]


def test_max_upload_size_bytes_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "25")
    assert Settings(_env_file=None).max_upload_size_bytes == 25 * 1024 * 1024  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("sqlite+aiosqlite:///./test.db", "sqlite"),
        ("postgresql+asyncpg://user:pw@localhost:5432/db", "postgresql"),
        ("postgresql://user:pw@localhost/db", "postgresql"),
    ],
)
def test_database_dialect_detection(
    monkeypatch: pytest.MonkeyPatch, url: str, expected: str
) -> None:
    monkeypatch.setenv("DATABASE_URL", url)
    assert Settings(_env_file=None).database_dialect == expected  # type: ignore[call-arg]


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """The environment is read once per process, not on every access."""
    get_settings.cache_clear()
    monkeypatch.setenv("PORT", "8123")
    first = get_settings()

    monkeypatch.setenv("PORT", "9999")
    second = get_settings()

    assert first is second
    assert second.port == 8123

    get_settings.cache_clear()


def test_no_secret_has_a_real_default() -> None:
    """A secret must never carry a usable default value baked into source."""
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.anthropic_api_key is None
