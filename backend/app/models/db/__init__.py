"""SQLAlchemy tables.

Every table must be imported here so that ``Base.metadata`` is complete when Alembic
autogenerates a migration. A table that is not imported is invisible to Alembic and will
be silently dropped from generated migrations.
"""

from app.models.db.base import Base
from app.models.db.document import DocumentRecord

__all__ = ["Base", "DocumentRecord"]
