"""The ``chunk_embeddings`` table — vector embeddings for retrieval chunks (Milestone 4).

Kept as its own table rather than columns on :class:`~app.models.db.chunk.ChunkRecord`:
a chunk can outlive several generations of embeddings (a model change, a re-embed), and
today's ``Chunk`` domain type and every existing call site have no reason to know
embeddings exist. One row here is one embedding of one chunk by one model.

Similarity search and retrieval are not part of this step — this table only stores
vectors and the metadata needed to identify and safely recompute them.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.db.base import GUID, Base, UTCDateTime, utcnow


class ChunkEmbeddingRecord(Base):
    """One embedding vector for one chunk, produced by one model."""

    __tablename__ = "chunk_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)

    # CASCADE: a chunk that is replaced or deleted (re-chunking, document deletion) must
    # take its embeddings with it, or a future chunk could be paired with a stale vector.
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Denormalised from the chunk so embeddings can be listed/deleted for a whole document
    # without a join back through chunks — mirrors ai_results.owner_id.
    document_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Which embedding model produced this vector. A model change must not be silently
    # treated as comparable to the old one — see the unique constraint below.
    model: Mapped[str] = mapped_column(String(64), nullable=False)

    # Hash of the exact chunk text this embedding was computed from. The vector itself
    # carries no way to detect that its source text has changed; this does.
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    # The embedding vector. Stored as JSON, not a native vector column: CLAUDE.md forbids
    # PostgreSQL-only SQL without dialect guarding, and no vector extension is enabled on
    # either the SQLite dev database or the PostgreSQL production target. Similarity
    # search is a later step, not this one.
    values: Mapped[list[float]] = mapped_column(JSON, nullable=False, default=list)

    # Vector length, denormalised for validation and display without decoding the JSON.
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Tokens billed for the embedding request, as reported by the provider.
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)

    __table_args__ = (
        # One embedding per (chunk, model). Recomputing with the same model replaces the
        # existing row; a different model coexists, so switching embedding models does
        # not destroy the ability to compare against the old one mid-migration.
        UniqueConstraint("chunk_id", "model", name="uq_chunk_embeddings_chunk_model"),
    )

    def __repr__(self) -> str:
        return (
            f"<ChunkEmbeddingRecord id={self.id} chunk_id={self.chunk_id} "
            f"model={self.model} dimensions={self.dimensions}>"
        )
