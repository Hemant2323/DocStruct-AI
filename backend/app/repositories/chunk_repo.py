"""Data access for retrieval chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
from uuid import UUID

from sqlalchemy import CursorResult, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db.chunk import ChunkRecord


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrieval-sized piece of a document, as the service layer sees it.

    Attributes:
        id: Identifier, assigned on insert.
        order: Position within the document.
        text: The chunk's text.
        block_ids: Which content blocks it was assembled from. Resolving these back to
            provenance is what turns a retrieved chunk into a citation.
        token_estimate: Approximate token count (see the chunker for the constant).
        char_count: Exact character count.
        location_label: Span covered, e.g. ``"pages 3-4"``.
    """

    id: UUID
    order: int
    text: str
    block_ids: list[UUID]
    token_estimate: int
    char_count: int
    location_label: str


def _to_domain(record: ChunkRecord) -> Chunk:
    return Chunk(
        id=record.id,
        order=record.order,
        text=record.text,
        block_ids=[UUID(value) for value in (record.block_ids or [])],
        token_estimate=record.token_estimate,
        char_count=record.char_count,
        location_label=record.location_label,
    )


class ChunkRepository:
    """Reads and writes retrieval chunks."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def replace_for_document(
        self,
        document_id: UUID,
        chunks: list[tuple[str, list[UUID], int, str]],
    ) -> list[UUID]:
        """Replace a document's chunks with a freshly built set.

        Args:
            document_id: The document the chunks belong to.
            chunks: ``(text, block_ids, token_estimate, location_label)`` in order.

        Returns:
            The generated chunk IDs.
        """
        await self.delete_for_document(document_id)

        records = [
            ChunkRecord(
                document_id=document_id,
                order=order,
                text=text,
                # JSON columns hold strings, not UUID objects.
                block_ids=[str(block_id) for block_id in block_ids],
                token_estimate=token_estimate,
                char_count=len(text),
                location_label=location_label[:255],
            )
            for order, (text, block_ids, token_estimate, location_label) in enumerate(chunks)
        ]

        self._session.add_all(records)
        await self._session.flush()
        return [record.id for record in records]

    async def list_for_document(self, document_id: UUID) -> list[Chunk]:
        """Return a document's chunks in order."""
        result = await self._session.execute(
            select(ChunkRecord)
            .where(ChunkRecord.document_id == document_id)
            .order_by(ChunkRecord.order)
        )
        return [_to_domain(record) for record in result.scalars().all()]

    async def count_for_document(self, document_id: UUID) -> int:
        """Count a document's chunks."""
        result = await self._session.execute(
            select(func.count())
            .select_from(ChunkRecord)
            .where(ChunkRecord.document_id == document_id)
        )
        return int(result.scalar_one())

    async def delete_for_document(self, document_id: UUID) -> int:
        """Delete every chunk belonging to a document. Returns the number removed."""
        result = cast(
            "CursorResult[Any]",
            await self._session.execute(
                delete(ChunkRecord).where(ChunkRecord.document_id == document_id)
            ),
        )
        await self._session.flush()
        return int(result.rowcount or 0)
