"""Tests that Alembic migrations stay in sync with the ORM models.

The application creates tables from ``Base.metadata`` in tests for speed, but production
schemas are built by migrations. Without this test the two could silently diverge — a
model change without a matching migration would pass every other test and then fail on
deploy.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import Settings

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _alembic_config(database_url: str) -> Config:
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


async def _table_names(database_url: str) -> list[str]:
    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as connection:
            return await connection.run_sync(lambda sync_conn: inspect(sync_conn).get_table_names())
    finally:
        await engine.dispose()


def test_migrations_upgrade_to_head(settings: Settings) -> None:
    """`alembic upgrade head` must build the full schema from empty."""
    command.upgrade(_alembic_config(settings.database_url), "head")

    tables = asyncio.run(_table_names(settings.database_url))

    assert "documents" in tables
    assert "alembic_version" in tables


def test_migrations_downgrade_cleanly(settings: Settings) -> None:
    """A migration that cannot be rolled back is a migration that cannot be deployed
    safely, so the downgrade path is verified too."""
    config = _alembic_config(settings.database_url)

    command.upgrade(config, "head")
    command.downgrade(config, "base")

    tables = asyncio.run(_table_names(settings.database_url))

    assert "documents" not in tables


def test_no_pending_model_changes(settings: Settings) -> None:
    """Fails when a model has changed without a matching migration.

    Uses Alembic's own comparison machinery: after upgrading to head, comparing the live
    database against ``Base.metadata`` must produce an empty diff.
    """
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext
    from sqlalchemy import create_engine

    command.upgrade(_alembic_config(settings.database_url), "head")

    from app.models.db import Base

    # Alembic's comparison API is synchronous, so use the sync driver for this check.
    sync_url = settings.database_url.replace("+aiosqlite", "")
    engine = create_engine(sync_url)
    try:
        with engine.connect() as connection:
            context = MigrationContext.configure(
                connection, opts={"compare_type": True, "render_as_batch": True}
            )
            diff = compare_metadata(context, Base.metadata)
    finally:
        engine.dispose()

    assert diff == [], (
        "Models and migrations have diverged. Run:\n"
        "  alembic revision --autogenerate -m 'describe your change'\n"
        f"Detected differences: {diff}"
    )
