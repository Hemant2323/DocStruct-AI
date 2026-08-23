"""Data access for extracted content blocks.

Blocks are always addressed through their document, and the document is always scoped by
owner, so there is no way to read another user's content through this repository.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import UUID

from sqlalchemy import CursorResult, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db.content_block import ContentBlockRecord
from app.models.domain.content import BlockType, ContentBlock, Provenance


def _to_domain(record: ContentBlockRecord) -> ContentBlock:
    """Convert a persistence record into the domain object services use."""
    return ContentBlock(
        block_type=BlockType(record.block_type),
        text=record.text,
        order=record.order,
        provenance=Provenance.from_dict(record.provenance),
        level=record.level,
        metadata=dict(record.block_metadata or {}),
    )


class ContentBlockRepository:
    """Reads and writes extracted content blocks.

    Args:
        session: The request- or job-scoped async session. The repository never commits;
            transaction boundaries belong to the caller.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def replace_for_document(
        self, document_id: UUID, blocks: list[ContentBlock]
    ) -> list[UUID]:
        """Replace a document's blocks with a freshly parsed set.

        Existing blocks are deleted first so that re-parsing a document cannot leave two
        generations interleaved — which the ``(document_id, order)`` unique constraint
        would reject anyway, but confusingly.

        Returns:
            The generated block IDs, in the same order as ``blocks``. The chunker needs
            these to record which blocks each chunk came from.
        """
        await self.delete_for_document(document_id)

        records = [
            ContentBlockRecord(
                document_id=document_id,
                order=block.order,
                block_type=block.block_type,
                text=block.text,
                level=block.level,
                provenance=block.provenance.to_dict(),
                location_label=block.provenance.label[:255],
                block_metadata=block.metadata,
                char_count=block.char_count,
            )
            for block in blocks
        ]

        self._session.add_all(records)
        await self._session.flush()
        return [record.id for record in records]

    async def list_for_document(
        self, document_id: UUID, *, limit: int | None = None, offset: int = 0
    ) -> list[ContentBlock]:
        """Return a document's blocks in reading order."""
        query = (
            select(ContentBlockRecord)
            .where(ContentBlockRecord.document_id == document_id)
            .order_by(ContentBlockRecord.order)
            .offset(offset)
        )
        if limit is not None:
            query = query.limit(limit)

        result = await self._session.execute(query)
        return [_to_domain(record) for record in result.scalars().all()]

    async def count_for_document(self, document_id: UUID) -> int:
        """Count a document's blocks."""
        result = await self._session.execute(
            select(func.count())
            .select_from(ContentBlockRecord)
            .where(ContentBlockRecord.document_id == document_id)
        )
        return int(result.scalar_one())

    async def get_by_ids(self, block_ids: list[UUID]) -> dict[UUID, ContentBlock]:
        """Fetch specific blocks by ID, keyed by ID.

        This is how a chunk is turned back into citations: the chunk records which block
        IDs it came from, and this resolves them to text and provenance.
        """
        if not block_ids:
            return {}

        result = await self._session.execute(
            select(ContentBlockRecord).where(ContentBlockRecord.id.in_(block_ids))
        )
        return {record.id: _to_domain(record) for record in result.scalars().all()}

    async def delete_for_document(self, document_id: UUID) -> int:
        """Delete every block belonging to a document. Returns the number removed."""
        result = cast(
            "CursorResult[Any]",
            await self._session.execute(
                delete(ContentBlockRecord).where(ContentBlockRecord.document_id == document_id)
            ),
        )
        await self._session.flush()
        return int(result.rowcount or 0)
