"""Data access for chunk embeddings.

No similarity search here — this is storage only. Retrieval is a later step, once this
layer's data exists to retrieve.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast
from uuid import UUID

from sqlalchemy import CursorResult, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db.chunk_embedding import ChunkEmbeddingRecord


@dataclass(frozen=True, slots=True)
class ChunkEmbedding:
    """A stored embedding, as the service layer sees it.

    Attributes:
        id: Identifier, assigned on insert.
        chunk_id: The chunk this embedding was computed from.
        document_id: The document the chunk belongs to, denormalised for document-level
            queries without a join.
        model: Which embedding model produced this vector.
        content_hash: Hash of the exact chunk text this embedding was computed from — the
            way to detect that a stored vector no longer matches its source text.
        values: The embedding vector.
        dimensions: Length of ``values``.
        token_count: Tokens billed for the embedding request.
        created_at: When this embedding was stored.
    """

    id: UUID
    chunk_id: UUID
    document_id: UUID
    model: str
    content_hash: str
    values: list[float]
    dimensions: int
    token_count: int
    created_at: datetime


def _to_domain(record: ChunkEmbeddingRecord) -> ChunkEmbedding:
    return ChunkEmbedding(
        id=record.id,
        chunk_id=record.chunk_id,
        document_id=record.document_id,
        model=record.model,
        content_hash=record.content_hash,
        values=list(record.values or []),
        dimensions=record.dimensions,
        token_count=record.token_count,
        created_at=record.created_at,
    )


class ChunkEmbeddingRepository:
    """Reads and writes chunk embeddings."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(
        self,
        *,
        chunk_id: UUID,
        document_id: UUID,
        model: str,
        values: list[float],
        content_hash: str,
        token_count: int,
    ) -> UUID:
        """Insert a new embedding.

        Fails on the unique (chunk_id, model) constraint if one already exists — use
        :meth:`replace_for_chunk` when recomputing.
        """
        record = ChunkEmbeddingRecord(
            chunk_id=chunk_id,
            document_id=document_id,
            model=model,
            content_hash=content_hash,
            values=values,
            dimensions=len(values),
            token_count=token_count,
        )
        self._session.add(record)
        await self._session.flush()
        return record.id

    async def replace_for_chunk(
        self,
        *,
        chunk_id: UUID,
        document_id: UUID,
        model: str,
        values: list[float],
        content_hash: str,
        token_count: int,
    ) -> UUID:
        """Replace a chunk's embedding for one model with a freshly computed one.

        The recompute path: any existing (chunk_id, model) row is removed first, so a
        model change or a source-text change never leaves two embeddings claiming to be
        the current one.
        """
        await self._session.execute(
            delete(ChunkEmbeddingRecord).where(
                ChunkEmbeddingRecord.chunk_id == chunk_id,
                ChunkEmbeddingRecord.model == model,
            )
        )
        return await self.save(
            chunk_id=chunk_id,
            document_id=document_id,
            model=model,
            values=values,
            content_hash=content_hash,
            token_count=token_count,
        )

    async def get_for_chunk(self, chunk_id: UUID, *, model: str) -> ChunkEmbedding | None:
        """Return a chunk's embedding for one model, or ``None`` if not yet computed."""
        result = await self._session.execute(
            select(ChunkEmbeddingRecord).where(
                ChunkEmbeddingRecord.chunk_id == chunk_id,
                ChunkEmbeddingRecord.model == model,
            )
        )
        record = result.scalar_one_or_none()
        return _to_domain(record) if record is not None else None

    async def list_for_document(
        self, document_id: UUID, *, model: str | None = None
    ) -> list[ChunkEmbedding]:
        """Return a document's embeddings, optionally filtered to one model."""
        query = select(ChunkEmbeddingRecord).where(ChunkEmbeddingRecord.document_id == document_id)
        if model is not None:
            query = query.where(ChunkEmbeddingRecord.model == model)
        result = await self._session.execute(query)
        return [_to_domain(record) for record in result.scalars().all()]

    async def count_for_document(self, document_id: UUID, *, model: str | None = None) -> int:
        """Count a document's embeddings, optionally filtered to one model."""
        query = (
            select(func.count())
            .select_from(ChunkEmbeddingRecord)
            .where(ChunkEmbeddingRecord.document_id == document_id)
        )
        if model is not None:
            query = query.where(ChunkEmbeddingRecord.model == model)
        result = await self._session.execute(query)
        return int(result.scalar_one())

    async def delete_for_chunk(self, chunk_id: UUID) -> int:
        """Delete every embedding belonging to one chunk. Returns the number removed."""
        result = cast(
            "CursorResult[Any]",
            await self._session.execute(
                delete(ChunkEmbeddingRecord).where(ChunkEmbeddingRecord.chunk_id == chunk_id)
            ),
        )
        await self._session.flush()
        return int(result.rowcount or 0)

    async def delete_for_document(self, document_id: UUID) -> int:
        """Delete every embedding belonging to a document. Returns the number removed."""
        result = cast(
            "CursorResult[Any]",
            await self._session.execute(
                delete(ChunkEmbeddingRecord).where(ChunkEmbeddingRecord.document_id == document_id)
            ),
        )
        await self._session.flush()
        return int(result.rowcount or 0)
