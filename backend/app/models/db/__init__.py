"""SQLAlchemy tables.

Every table must be imported here so that ``Base.metadata`` is complete when Alembic
autogenerates a migration. A table that is not imported is invisible to Alembic and will
be silently dropped from generated migrations.
"""

from app.models.db.ai_result import AIResultRecord
from app.models.db.base import Base
from app.models.db.chunk import ChunkRecord
from app.models.db.chunk_embedding import ChunkEmbeddingRecord
from app.models.db.content_block import ContentBlockRecord
from app.models.db.document import DocumentRecord
from app.models.db.document_conversion import DocumentConversionRecord

__all__ = [
    "AIResultRecord",
    "Base",
    "ChunkEmbeddingRecord",
    "ChunkRecord",
    "ContentBlockRecord",
    "DocumentConversionRecord",
    "DocumentRecord",
]
