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
    storage_dir: Path = PROJECT_ROOT / "storage"
    max_upload_size_mb: int = Field(default=25, ge=1, le=500)

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

    # --- AI (not used until Milestone 3; optional so the app runs without it) ---
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-5"

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
