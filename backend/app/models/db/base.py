"""SQLAlchemy declarative base and shared column types.

Every table inherits from ``Base``. Alembic reads ``Base.metadata`` to autogenerate
migrations, so any new table must be imported in ``app.models.db.__init__`` or Alembic
will not see it.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, String, TypeDecorator
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import TypeEngine


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class GUID(TypeDecorator[uuid.UUID]):
    """A UUID column that works on both PostgreSQL and SQLite.

    PostgreSQL has a native ``UUID`` type; SQLite does not. This stores a native UUID on
    PostgreSQL and a 36-character string elsewhere, while application code always sees a
    ``uuid.UUID``. That is what lets the same models and migrations run against the
    zero-setup SQLite development database and the PostgreSQL production target.
    """

    impl = String(36)
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PostgresUUID(as_uuid=True))
        return dialect.type_descriptor(String(36))

    def process_bind_param(self, value: uuid.UUID | str | None, dialect: Dialect) -> object:
        if value is None:
            return None
        parsed = value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
        if dialect.name == "postgresql":
            return parsed
        return str(parsed)

    def process_result_value(self, value: object, dialect: Dialect) -> uuid.UUID | None:
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(str(value))


class UTCDateTime(TypeDecorator[datetime]):
    """A timezone-aware datetime column that is consistent across backends.

    SQLite has no native timezone support and returns naive datetimes. Storing UTC and
    re-attaching the UTC tzinfo on read means application code always receives an aware
    datetime, so comparisons never raise "can't compare offset-naive and offset-aware".
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("Naive datetime rejected; use timezone-aware UTC datetimes")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


def utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)
