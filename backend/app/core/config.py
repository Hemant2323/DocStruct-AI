"""Application configuration.

All configuration — without exception — is read from environment variables (or a local
`.env` file, which is gitignored). Nothing is hardcoded, and no secret ever appears in
source control.

The settings object is validated at import time by Pydantic. A missing or malformed
required value fails loudly at startup rather than surfacing as a confusing error deep in
a request handler later.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: .../backend/app/core/config.py -> parents[3] == project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]


class AppEnv(StrEnum):
    """Which environment the application is running in."""

    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Typed application settings loaded from the environment.

    Field names map to upper-case environment variables (``app_env`` -> ``APP_ENV``).
    """

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application -------------------------------------------------------
    app_name: str = "DocStruct AI"
    app_env: AppEnv = AppEnv.DEVELOPMENT
    debug: bool = False
    api_prefix: str = "/api"

    # --- Server ------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000

    # --- Logging -----------------------------------------------------------
    log_level: str = "INFO"

    # --- Database ----------------------------------------------------------
    # PostgreSQL is the production target:
    #   postgresql+asyncpg://user:password@localhost:5432/docstruct
    # SQLite is the zero-setup default so the project runs on a clean machine.
    database_url: str = f"sqlite+aiosqlite:///{BACKEND_ROOT / 'docstruct.db'}"
    database_echo: bool = False

    # --- File storage ------------------------------------------------------
    #: Which `FileStorage` implementation `app.api.deps.get_file_storage()` builds.
    #: Defaults to "local" so existing behaviour is unchanged unless this is set.
    #:
    #: "supabase" is accepted here before the implementation exists, deliberately: the
    #: setting and its test isolation land first so that the day the backend is written,
    #: no test can reach a real bucket by accident.
    storage_backend: str = "local"

    #: Root directory for the "local" backend. Ignored by any other backend — the two
    #: are alternatives, not layers.
    storage_dir: Path = PROJECT_ROOT / "storage"
    max_upload_size_mb: int = Field(default=25, ge=1, le=500)

    # --- Supabase Storage ----------------------------------------------------
    #: The project's base URL, e.g. https://abcdefgh.supabase.co — not the storage
    #: endpoint itself, which is a path beneath it. Optional so the application still
    #: starts without it, exactly like the AI provider keys.
    supabase_url: str | None = None

    #: The service-role key. REAL VALUE IN THE ENVIRONMENT ONLY.
    #:
    #: This key bypasses row-level security, so it is the most dangerous secret in the
    #: project: server-side only, never returned by any endpoint, never logged, and never
    #: sent to the browser.
    supabase_service_key: str | None = None

    #: Bucket that uploaded documents live in. Should be **private** — objects are
    #: addressed by document UUID, but an unauthenticated public bucket would still make
    #: every uploaded file readable by anyone who learns an id.
    supabase_storage_bucket: str = "documents"

    # --- Ingestion limits --------------------------------------------------
    # Resource ceilings. A malicious or merely enormous file must fail fast with a
    # clear message rather than exhausting memory.
    max_pdf_pages: int = Field(default=500, ge=1)
    max_sheets: int = Field(default=50, ge=1)
    max_rows: int = Field(default=100_000, ge=1)
    max_columns: int = Field(default=512, ge=1)
    max_blocks: int = Field(default=50_000, ge=1)
    #: Rows batched into one tabular block. A single row is too small to be a useful
    #: retrieval unit; too many and a citation stops being precise.
    tabular_rows_per_block: int = Field(default=25, ge=1, le=500)
    #: Total uncompressed size allowed inside a DOCX/XLSX archive (zip-bomb guard).
    max_archive_uncompressed_mb: int = Field(default=250, ge=1)
    max_archive_entries: int = Field(default=2_000, ge=1)

    # --- Chunking ----------------------------------------------------------
    chunk_target_chars: int = Field(default=1_500, ge=200, le=20_000)
    chunk_overlap_chars: int = Field(default=150, ge=0, le=2_000)

    # --- Background processing ---------------------------------------------
    #: How many documents may be processed at once. Ten simultaneous large uploads
    #: would otherwise all be held in memory together.
    max_concurrent_processing: int = Field(default=2, ge=1, le=32)

    # --- CORS --------------------------------------------------------------
    # Comma-separated list of allowed browser origins for the frontend dev server.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- AI ------------------------------------------------------------------
    # Optional so the application still starts without a key; AI endpoints then
    # report that they are unconfigured rather than the whole app failing.
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-5"

    #: Where to send API requests. Empty means Anthropic's own endpoint.
    #:
    #: Set this to use an Anthropic-compatible provider instead. The SDK also reads an
    #: ANTHROPIC_BASE_URL environment variable on its own, but that does not help here:
    #: pydantic-settings parses `.env` itself and never exports into os.environ, so a
    #: value written there would be invisible to the SDK unless it is passed explicitly.
    anthropic_base_url: str | None = None

    # --- Alternative provider: NVIDIA NIM (OpenAI-compatible) -----------------
    #: Which LLMProvider implementation `app.api.deps.get_llm_provider()` builds.
    #: Defaults to "claude" so existing behaviour is unchanged unless this is set.
    ai_provider: str = "claude"

    #: Optional so the application still starts without a key; NVIDIAProvider then
    #: reports itself as unconfigured, exactly like ClaudeProvider does.
    nvidia_api_key: str | None = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "deepseek-ai/deepseek-v4-flash-0731"

    # --- Alternative provider: Google Gemini -----------------------------------
    #: Optional so the application still starts without a key; GeminiProvider then
    #: reports itself as unconfigured, exactly like ClaudeProvider and NVIDIAProvider do.
    gemini_api_key: str | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_model: str = "gemini-2.5-flash"

    #: Separate from `gemini_model`: that setting is for generation (GeminiProvider —
    #: /summary, /explanation, /structure); this one is for embeddings (GeminiEmbedder —
    #: /ask's retrieval). A chat model name is not a valid embedding model, so the two
    #: must never share one setting. Defaults to GeminiEmbedder's own default model.
    gemini_embedding_model: str = "gemini-embedding-2-preview"

    #: Ceiling on the model's reply. Summaries and explanations are short; this is
    #: sized for a long explanation, not for the document being echoed back.
    ai_max_output_tokens: int = Field(default=8_000, ge=256, le=64_000)

    #: Thinking depth / overall token spend. "high" is the sweet spot for
    #: understanding a document correctly, which is the whole point here.
    ai_effort: str = "high"

    #: Seconds to wait for the model. Generous: a large document with adaptive
    #: thinking legitimately takes a while.
    ai_timeout_seconds: float = Field(default=300.0, ge=10.0, le=1800.0)

    #: SDK-level retries for transient failures (429, 5xx, connection errors).
    ai_max_retries: int = Field(default=2, ge=0, le=5)

    #: Cache the document prefix so the second and third operation on the same
    #: document are much cheaper. Below ~1024 tokens the API will not cache anyway.
    ai_prompt_caching: bool = True

    #: Largest context we will send, in estimated tokens. Opus 5 has a 1M window;
    #: this leaves headroom for the prompt, thinking and the reply.
    ai_max_context_tokens: int = Field(default=600_000, ge=1_000, le=900_000)

    # --- Optional context optimization ---------------------------------------
    #: Below this many estimated tokens a document does not need optimizing, and
    #: the option is not offered. Optimizing a short document only loses detail.
    ai_optimization_min_tokens: int = Field(default=20_000, ge=1_000)

    #: Share of the token budget an optimized context may use.
    ai_optimization_target_ratio: float = Field(default=0.5, gt=0.0, le=1.0)

    # --- Windowed conversion extraction --------------------------------------
    #: Above this many estimated tokens of extracted content, a conversion extracts the
    #: document in windows instead of one call. Below it, nothing changes — the request
    #: takes the single-call path it always took.
    #:
    #: The number that matters is not the input context (Gemini's window is far larger
    #: than this) but the *output* ceiling: one call must emit every record it finds, and
    #: `ai_max_output_tokens` caps that reply regardless of how much was sent in. A
    #: document big enough to hold more records than one reply can carry loses the rest,
    #: which is what windowing exists to prevent.
    #:
    #: This is both the threshold for windowing at all and the *maximum* window size.
    #: Windows below it are sized per document from measured record density and the
    #: schema's field count — see `app.ai.windows.adaptive_window_tokens`, and
    #: `scripts/benchmark_conversion.py` for what the two densities cost.
    conversion_chunk_window_tokens: int = Field(default=40_000, ge=1_000, le=500_000)

    #: Floor on an adaptively chosen window.
    #:
    #: The first window of a long document is planned at exactly this size: it is small
    #: enough to be safe at any density, and the records it returns are what the rest of
    #: the document is sized from. It is also the size every window falls back to when
    #: that measurement is unavailable, so it has to be conservative rather than merely
    #: small.
    conversion_chunk_min_window_tokens: int = Field(default=2_000, ge=500, le=50_000)

    #: How many window extractions may be in flight at once.
    #:
    #: Deliberately low. Gemini's free tier is measured in requests per minute, and the
    #: provider's retry budget is per-request — so raising this multiplies into
    #: `concurrency x retries` requests arriving during the same rate-limit window, which
    #: is how a "faster" setting ends up slower and half-failed.
    conversion_chunk_concurrency: int = Field(default=3, ge=1, le=16)

    #: Kill switch. With this off, every conversion takes the single-call path however
    #: large the document is — the behaviour that shipped before windowing existed.
    conversion_chunking_enabled: bool = True

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        normalized = value.upper()
        if normalized not in valid:
            raise ValueError(f"log_level must be one of {sorted(valid)}, got {value!r}")
        return normalized

    @field_validator("api_prefix")
    @classmethod
    def _validate_api_prefix(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("api_prefix must start with '/'")
        return value.rstrip("/")

    @field_validator("database_url")
    @classmethod
    def _normalize_database_url(cls, value: str) -> str:
        """Name the async driver on a PostgreSQL URL that does not name one.

        Everything here talks to the database through SQLAlchemy's **async** engine, which
        requires the driver to be part of the URL: ``postgresql+asyncpg://``. Managed
        providers hand out the plain ``postgresql://`` form, which selects the default
        synchronous driver and fails with ``No module named 'psycopg2'`` — an error that
        names a package the project neither has nor wants, and says nothing about the
        actual problem.

        Normalising here rather than at the point of use is what makes it one fix instead
        of two: the application engine and Alembic both read ``settings.database_url``
        (``alembic/env.py`` copies it into ``sqlalchemy.url``), so a URL corrected in only
        one of them would leave migrations failing against a database the app could reach.

        Deliberately narrow. Only the bare ``postgresql`` scheme is rewritten:

        * ``postgresql+asyncpg://`` already names a driver and is returned untouched;
        * so is any other explicit choice such as ``postgresql+psycopg://`` — overriding
          a driver someone named on purpose would be a surprise, not a convenience;
        * ``sqlite+aiosqlite://`` and every other scheme pass through unchanged.
        """
        scheme, separator, rest = value.partition("://")
        if separator and scheme == "postgresql":
            return f"postgresql+asyncpg://{rest}"
        return value

    @field_validator("anthropic_base_url")
    @classmethod
    def _validate_base_url(cls, value: str | None) -> str | None:
        """Reject a base URL that is obviously wrong, and normalise an empty one.

        A blank entry in `.env` is a common way to "unset" a value, and it should mean
        "use the default endpoint" rather than "send requests to an empty host".
        """
        if value is None or not value.strip():
            return None

        cleaned = value.strip().rstrip("/")
        if not cleaned.startswith(("http://", "https://")):
            raise ValueError(
                f"anthropic_base_url must start with http:// or https://, got {value!r}"
            )
        return cleaned

    @field_validator("storage_backend")
    @classmethod
    def _validate_storage_backend(cls, value: str) -> str:
        """Reject an unknown storage backend at startup rather than at the first upload.

        The same shape as :meth:`_validate_ai_provider`, and for the same reason:
        ``get_file_storage()`` branches on this value, so a typo caught here is a clear
        startup failure rather than a confusing error the first time someone uploads a
        file — by which point the response has already been accepted.
        """
        valid = {"local", "supabase"}
        normalized = value.strip().lower()
        if normalized not in valid:
            raise ValueError(f"storage_backend must be one of {sorted(valid)}, got {value!r}")
        return normalized

    @field_validator("supabase_url")
    @classmethod
    def _validate_supabase_url(cls, value: str | None) -> str | None:
        """Normalise the project URL, and reject one that is obviously wrong.

        Blank is treated as unset, matching :meth:`_validate_base_url`: emptying a value
        in the environment is how people turn it off, and it should mean "not configured"
        rather than "send requests to an empty host".
        """
        if value is None or not value.strip():
            return None

        cleaned = value.strip().rstrip("/")
        if not cleaned.startswith(("http://", "https://")):
            raise ValueError(f"supabase_url must start with http:// or https://, got {value!r}")
        return cleaned

    @field_validator("ai_provider")
    @classmethod
    def _validate_ai_provider(cls, value: str) -> str:
        """Reject an unknown provider name at startup rather than at the first request.

        `get_llm_provider()` branches on this value; failing here means a typo in
        `.env` is caught immediately instead of surfacing as a confusing runtime error
        the first time someone clicks an AI button.
        """
        valid = {"claude", "nvidia", "gemini"}
        normalized = value.strip().lower()
        if normalized not in valid:
            raise ValueError(f"ai_provider must be one of {sorted(valid)}, got {value!r}")
        return normalized

    @field_validator("nvidia_base_url")
    @classmethod
    def _validate_nvidia_base_url(cls, value: str) -> str:
        cleaned = value.strip().rstrip("/")
        if not cleaned.startswith(("http://", "https://")):
            raise ValueError(f"nvidia_base_url must start with http:// or https://, got {value!r}")
        return cleaned

    @field_validator("gemini_base_url")
    @classmethod
    def _validate_gemini_base_url(cls, value: str) -> str:
        cleaned = value.strip().rstrip("/")
        if not cleaned.startswith(("http://", "https://")):
            raise ValueError(f"gemini_base_url must start with http:// or https://, got {value!r}")
        return cleaned

    @field_validator("ai_effort")
    @classmethod
    def _validate_ai_effort(cls, value: str) -> str:
        valid = {"low", "medium", "high", "xhigh", "max"}
        normalized = value.lower()
        if normalized not in valid:
            raise ValueError(f"ai_effort must be one of {sorted(valid)}, got {value!r}")
        return normalized

    @property
    def ai_configured(self) -> bool:
        """Whether an API key is present.

        Checked before an AI request so the user gets "AI is not configured" rather
        than an authentication failure from a third party.
        """
        return bool(self.anthropic_api_key and self.anthropic_api_key.strip())

    @model_validator(mode="after")
    def _validate_chunking(self) -> Settings:
        """Reject an overlap that would make chunking fail to advance.

        With an overlap at or above the target, each chunk would start no later than the
        previous one and the chunker would loop forever. Catching it here means a typo in
        `.env` fails at startup instead of hanging a background job.
        """
        if self.chunk_overlap_chars >= self.chunk_target_chars:
            raise ValueError(
                "chunk_overlap_chars must be smaller than chunk_target_chars "
                f"(got overlap={self.chunk_overlap_chars}, target={self.chunk_target_chars})"
            )
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins parsed into a list, empty entries discarded."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def max_upload_size_bytes(self) -> int:
        """Upload ceiling in bytes, for comparison against a content length."""
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.app_env is AppEnv.PRODUCTION

    @property
    def database_dialect(self) -> str:
        """The dialect portion of the database URL, e.g. ``postgresql`` or ``sqlite``."""
        return self.database_url.split("+", 1)[0].split(":", 1)[0]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the application settings.

    Cached so the environment is read once per process. Tests clear the cache
    (``get_settings.cache_clear()``) after overriding environment variables.
    """
    return Settings()
