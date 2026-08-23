"""Database engine and session management.

Async SQLAlchemy throughout: request handlers wait on the database and, later, on the
Claude API, so blocking the event loop would cap concurrency at one request.

The same code runs on PostgreSQL (production target, ``postgresql+asyncpg://``) and
SQLite (zero-setup development default, ``sqlite+aiosqlite://``). Dialect differences are
absorbed by the custom column types in ``app.models.db.base``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _engine_kwargs(settings: Settings) -> dict[str, Any]:
    """Engine options appropriate to the configured dialect."""
    kwargs: dict[str, Any] = {"echo": settings.database_echo, "future": True}

    if settings.database_dialect == "sqlite":
        # SQLite has no server-side pool to configure, and aiosqlite rejects the
        # pool sizing arguments that PostgreSQL accepts.
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs.update(
            pool_size=5,
            max_overflow=10,
            # Recycle connections before a proxy or PostgreSQL idle timeout can close
            # them underneath us, which otherwise surfaces as a random dropped request.
            pool_recycle=1800,
            pool_pre_ping=True,
        )
    return kwargs


def get_engine() -> AsyncEngine:
    """Return the process-wide async engine, creating it on first use."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(settings.database_url, **_engine_kwargs(settings))
        logger.info("Database engine created dialect=%s", settings.database_dialect)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the process-wide session factory, creating it on first use."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a database session scoped to one request.

    The session is committed if the handler returns normally and rolled back if it
    raises, so a failed request never leaves a partial write behind. The exception is
    always re-raised — failures are never swallowed here.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def dispose_engine() -> None:
    """Close all pooled connections and reset module state.

    Called on application shutdown, and by tests between cases so each one gets a fresh
    engine bound to its own temporary database.
    """
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        logger.info("Database engine disposed")
    _engine = None
    _session_factory = None


async def check_database_connection() -> bool:
    """Return whether the database answers a trivial query.

    Used by the readiness probe. Any failure is logged with its type and returned as
    ``False`` — the readiness endpoint reports "not ready" rather than raising a 500,
    because an unreachable database is an expected operational state, not a bug.
    """
    from sqlalchemy import text

    try:
        engine = get_engine()
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # deliberately broad: always reported, never hidden
        logger.error("Database connectivity check failed: %s: %s", type(exc).__name__, exc)
        return False
