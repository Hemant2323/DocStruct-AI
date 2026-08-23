"""Alembic migration environment.

Two things this file does that a stock ``alembic init`` does not:

1. Reads the database URL from application settings (i.e. the ``DATABASE_URL``
   environment variable) instead of ``alembic.ini``, so no credential is ever committed.
2. Runs migrations through SQLAlchemy's async engine, matching the application, and
   enables ``render_as_batch`` so migrations also work on SQLite — which cannot ALTER a
   column in place and needs the table copied instead.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlalchemy.pool import NullPool

from app.core.config import get_settings

# Importing the package registers every table on Base.metadata. Without this, autogenerate
# would see an empty schema and produce a migration that drops all the tables.
from app.models.db import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)


def _is_sqlite() -> bool:
    return settings.database_dialect == "sqlite"


def render_item(type_: str, obj: object, autogen_context: object) -> str | bool:
    """Render the project's custom column types correctly in generated migrations.

    Autogenerate writes a custom ``TypeDecorator`` as its fully qualified name
    (``app.models.db.base.GUID()``) but does not add the import for it, producing a
    migration that raises ``NameError`` the moment it runs. This hook registers the
    required import so each migration carries exactly the imports it uses.

    Returning ``False`` delegates everything else to the default renderer.
    """
    if type_ == "type" and obj.__class__.__module__.startswith("app.models.db"):
        autogen_context.imports.add("import app.models.db.base")  # type: ignore[attr-defined]
        return f"app.models.db.base.{obj.__class__.__name__}()"
    return False


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting to a database ('--sql' mode)."""
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        render_as_batch=_is_sqlite(),
        render_item=render_item,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations on an already-established connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # Detect column type changes as well as added/removed columns.
        compare_type=True,
        # SQLite cannot ALTER columns; batch mode rebuilds the table instead.
        render_as_batch=_is_sqlite(),
        render_item=render_item,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations through it."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Connect to the database and apply migrations."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
