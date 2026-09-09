"""Tests for configuration loading and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from app.core.config import PROJECT_ROOT, AppEnv, Settings, get_settings


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


class TestAnthropicBaseUrl:
    """The endpoint is configurable so the same provider can talk to an
    Anthropic-compatible service without any code change."""

    def test_defaults_to_unset(self) -> None:
        """Unset means the SDK uses Anthropic's own endpoint."""
        assert Settings(_env_file=None).anthropic_base_url is None  # type: ignore[call-arg]

    def test_reads_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.example.dev/v1")

        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.anthropic_base_url == "https://api.example.dev/v1"

    def test_trailing_slash_is_normalised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.example.dev/v1/")

        assert Settings(_env_file=None).anthropic_base_url == "https://api.example.dev/v1"  # type: ignore[call-arg]

    def test_blank_means_use_the_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Blanking a value in .env is how people "unset" it; it must not become a host."""
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "   ")

        assert Settings(_env_file=None).anthropic_base_url is None  # type: ignore[call-arg]

    def test_a_url_without_a_scheme_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fails at startup rather than at the first AI request."""
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "api.example.dev")

        with pytest.raises(PydanticValidationError, match="http"):
            Settings(_env_file=None)  # type: ignore[call-arg]

    def test_the_model_is_configurable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A compatible provider may use its own model names, such as `auto` routing."""
        monkeypatch.setenv("ANTHROPIC_MODEL", "auto")

        assert Settings(_env_file=None).anthropic_model == "auto"  # type: ignore[call-arg]


class TestAIProvider:
    """AI_PROVIDER selects which LLMProvider implementation is built. It is entirely
    new configuration, so these tests are additive and touch nothing existing."""

    def test_defaults_to_claude(self) -> None:
        """Preserves existing behaviour for anyone not setting this explicitly."""
        assert Settings(_env_file=None).ai_provider == "claude"  # type: ignore[call-arg]

    def test_accepts_nvidia(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_PROVIDER", "nvidia")

        assert Settings(_env_file=None).ai_provider == "nvidia"  # type: ignore[call-arg]

    def test_accepts_gemini(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_PROVIDER", "gemini")

        assert Settings(_env_file=None).ai_provider == "gemini"  # type: ignore[call-arg]

    def test_is_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_PROVIDER", "NVIDIA")

        assert Settings(_env_file=None).ai_provider == "nvidia"  # type: ignore[call-arg]

    def test_an_unknown_provider_is_rejected_at_startup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fails at startup rather than at the first AI request."""
        monkeypatch.setenv("AI_PROVIDER", "openai")

        with pytest.raises(PydanticValidationError, match="ai_provider must be one of"):
            Settings(_env_file=None)  # type: ignore[call-arg]


class TestNvidiaSettings:
    """The NVIDIA_* settings, independent of whether AI_PROVIDER selects them."""

    def test_defaults_match_the_requested_nvidia_nim_configuration(self) -> None:
        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.nvidia_base_url == "https://integrate.api.nvidia.com/v1"
        assert settings.nvidia_model == "deepseek-ai/deepseek-v4-flash-0731"

    def test_no_key_has_a_real_default(self) -> None:
        """A secret must never carry a usable default value baked into source."""
        assert Settings(_env_file=None).nvidia_api_key is None  # type: ignore[call-arg]

    def test_key_is_read_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key-not-real")

        assert Settings(_env_file=None).nvidia_api_key == "test-key-not-real"  # type: ignore[call-arg]

    def test_base_url_is_configurable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NVIDIA_BASE_URL", "https://example.dev/v1")

        assert Settings(_env_file=None).nvidia_base_url == "https://example.dev/v1"  # type: ignore[call-arg]

    def test_base_url_trailing_slash_is_normalised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NVIDIA_BASE_URL", "https://example.dev/v1/")

        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.nvidia_base_url == "https://example.dev/v1"

    def test_base_url_without_a_scheme_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fails at startup rather than at the first AI request."""
        monkeypatch.setenv("NVIDIA_BASE_URL", "example.dev")

        with pytest.raises(PydanticValidationError, match="http"):
            Settings(_env_file=None)  # type: ignore[call-arg]

    def test_model_is_configurable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NVIDIA_MODEL", "deepseek-ai/other-model")

        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.nvidia_model == "deepseek-ai/other-model"


class TestGeminiSettings:
    """The GEMINI_* settings, independent of whether AI_PROVIDER selects them."""

    def test_defaults_match_the_requested_gemini_configuration(self) -> None:
        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.gemini_base_url == "https://generativelanguage.googleapis.com/v1beta"
        assert settings.gemini_model == "gemini-3.6-flash"

    def test_embedding_model_defaults_to_a_real_embedding_model_not_the_chat_model(
        self,
    ) -> None:
        """The generation and embedding models must be distinct settings — a chat model
        name is not valid for the embeddings endpoint."""
        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.gemini_embedding_model == "gemini-embedding-2-preview"
        assert settings.gemini_embedding_model != settings.gemini_model

    def test_embedding_model_is_configurable_independently_of_the_chat_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-3-preview")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")

        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.gemini_embedding_model == "gemini-embedding-3-preview"
        assert settings.gemini_model == "gemini-3.6-flash"

    def test_no_key_has_a_real_default(self) -> None:
        """A secret must never carry a usable default value baked into source."""
        assert Settings(_env_file=None).gemini_api_key is None  # type: ignore[call-arg]

    def test_key_is_read_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")

        assert Settings(_env_file=None).gemini_api_key == "test-key-not-real"  # type: ignore[call-arg]

    def test_base_url_is_configurable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_BASE_URL", "https://example.dev/v1beta")

        assert Settings(_env_file=None).gemini_base_url == "https://example.dev/v1beta"  # type: ignore[call-arg]

    def test_base_url_trailing_slash_is_normalised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_BASE_URL", "https://example.dev/v1beta/")

        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.gemini_base_url == "https://example.dev/v1beta"

    def test_base_url_without_a_scheme_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fails at startup rather than at the first AI request."""
        monkeypatch.setenv("GEMINI_BASE_URL", "example.dev")

        with pytest.raises(PydanticValidationError, match="http"):
            Settings(_env_file=None)  # type: ignore[call-arg]

    def test_model_is_configurable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-pro")

        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.gemini_model == "gemini-2.5-pro"


class TestDatabaseUrlNormalization:
    """A bare `postgresql://` URL is given the async driver it needs.

    Everything reaches the database through SQLAlchemy's async engine, which requires the
    driver in the URL. Managed providers hand out the plain form, which selects the
    default *synchronous* driver and dies with `No module named 'psycopg2'` — naming a
    package this project does not use and never will.
    """

    def test_a_plain_postgresql_url_gains_the_async_driver(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reported deployment case."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pw@host:5432/db")

        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.database_url == "postgresql+asyncpg://user:pw@host:5432/db"

    def test_a_url_that_already_names_asyncpg_is_untouched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Byte-for-byte — normalising an already-correct URL must not rewrite it."""
        url = "postgresql+asyncpg://user:pw@host:5432/db"
        monkeypatch.setenv("DATABASE_URL", url)

        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.database_url == url

    def test_a_sqlite_url_is_untouched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The development default must behave exactly as it did before."""
        url = "sqlite+aiosqlite:///./test.db"
        monkeypatch.setenv("DATABASE_URL", url)

        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.database_url == url
        assert settings.database_dialect == "sqlite"

    def test_the_sqlite_default_is_untouched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With nothing set at all, the zero-setup default still points at SQLite."""
        monkeypatch.delenv("DATABASE_URL", raising=False)

        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.database_url.startswith("sqlite+aiosqlite:///")
        assert settings.database_dialect == "sqlite"

    def test_an_explicitly_chosen_driver_is_respected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the *bare* scheme is rewritten. Someone who named a driver meant it, and
        silently swapping it for another would be a surprise rather than a convenience."""
        url = "postgresql+psycopg://user:pw@host:5432/db"
        monkeypatch.setenv("DATABASE_URL", url)

        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.database_url == url

    def test_everything_after_the_scheme_survives_verbatim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Credentials, port, database name and query parameters are the parts that make
        the URL work at all, so the rewrite must touch the scheme and nothing else."""
        monkeypatch.setenv(
            "DATABASE_URL",
            "postgresql://u%40name:p%3Aw@db.internal.example:5432/docstruct?sslmode=require",
        )

        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.database_url == (
            "postgresql+asyncpg://u%40name:p%3Aw@db.internal.example:5432/docstruct?sslmode=require"
        )

    def test_the_normalised_url_is_one_the_async_engine_accepts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The end of the story, and the only assertion that proves the bug is fixed.

        Creating an engine resolves the dialect and imports the driver, which is exactly
        where the plain URL failed. No connection is opened, so this touches no network
        and needs no database.
        """
        from sqlalchemy.ext.asyncio import create_async_engine

        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pw@host:5432/db")

        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        engine = create_async_engine(settings.database_url)

        assert engine.dialect.driver == "asyncpg"

    def test_the_dialect_reported_for_engine_options_is_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`database_dialect` picks the pool settings, so normalisation must not move a
        PostgreSQL URL onto the SQLite branch (or the reverse)."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pw@host:5432/db")

        assert Settings(_env_file=None).database_dialect == "postgresql"  # type: ignore[call-arg]


class TestStorageBackend:
    """STORAGE_BACKEND selects which `FileStorage` implementation is built.

    Entirely new configuration, so these tests are additive and touch nothing existing.
    "supabase" is accepted before that backend exists on purpose: the setting and its
    validation land first so the implementation can arrive without touching services,
    routes, or any test's isolation.
    """

    def test_defaults_to_local(self) -> None:
        """Preserves existing behaviour for anyone not setting this explicitly."""
        assert Settings(_env_file=None).storage_backend == "local"  # type: ignore[call-arg]

    def test_accepts_supabase(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STORAGE_BACKEND", "supabase")

        assert Settings(_env_file=None).storage_backend == "supabase"  # type: ignore[call-arg]

    def test_is_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STORAGE_BACKEND", "  SUPABASE  ")

        assert Settings(_env_file=None).storage_backend == "supabase"  # type: ignore[call-arg]

    def test_an_unknown_backend_is_rejected_at_startup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fails at startup rather than at the first upload, by which point the request
        has already been accepted and the caller told the file was received."""
        monkeypatch.setenv("STORAGE_BACKEND", "s3")

        with pytest.raises(PydanticValidationError, match="storage_backend must be one of"):
            Settings(_env_file=None)  # type: ignore[call-arg]

    def test_setting_a_backend_does_not_disturb_storage_dir(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The two are independent settings; selecting one must not blank the other."""
        monkeypatch.setenv("STORAGE_BACKEND", "supabase")

        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.storage_dir == PROJECT_ROOT / "storage"


class TestSupabaseStorageSettings:
    """Credentials for the Supabase backend. Optional, so the app still starts without
    them — the same rule the AI provider keys follow."""

    def test_url_and_key_default_to_unset(self) -> None:
        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.supabase_url is None
        assert settings.supabase_service_key is None

    def test_bucket_has_a_usable_default(self) -> None:
        assert Settings(_env_file=None).supabase_storage_bucket == "documents"  # type: ignore[call-arg]

    def test_reads_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SUPABASE_URL", "https://abcdefgh.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "test-key-not-real")
        monkeypatch.setenv("SUPABASE_STORAGE_BUCKET", "uploads")

        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.supabase_url == "https://abcdefgh.supabase.co"
        assert settings.supabase_service_key == "test-key-not-real"
        assert settings.supabase_storage_bucket == "uploads"

    def test_a_trailing_slash_on_the_url_is_normalised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Storage paths are appended to this, so a trailing slash would produce `//`."""
        monkeypatch.setenv("SUPABASE_URL", "https://abcdefgh.supabase.co/")

        assert Settings(_env_file=None).supabase_url == "https://abcdefgh.supabase.co"  # type: ignore[call-arg]

    def test_a_blank_url_means_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Blanking a value is how people turn one off; it must not become a host."""
        monkeypatch.setenv("SUPABASE_URL", "   ")

        assert Settings(_env_file=None).supabase_url is None  # type: ignore[call-arg]

    def test_a_url_without_a_scheme_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fails at startup rather than at the first upload."""
        monkeypatch.setenv("SUPABASE_URL", "abcdefgh.supabase.co")

        with pytest.raises(PydanticValidationError, match="supabase_url must start with"):
            Settings(_env_file=None)  # type: ignore[call-arg]


def test_no_secret_has_a_real_default() -> None:
    """A secret must never carry a usable default value baked into source."""
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.anthropic_api_key is None
    assert settings.supabase_service_key is None
