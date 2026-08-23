"""Data access for documents.

This is the only module that issues queries against the ``documents`` table. Keeping
database access in one place is what makes the owner-scoping rule enforceable: every read
method here takes an ``owner_id`` and filters on it, so a forgotten ``WHERE`` clause in a
route handler cannot leak another user's documents.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db.base import utcnow
from app.models.db.document import DocumentRecord
from app.models.domain.document import Document, ProcessingStatus, SupportedFileType


def _to_domain(record: DocumentRecord) -> Document:
    """Convert a persistence record into the domain object services use."""
    return Document(
        id=record.id,
        owner_id=record.owner_id,
        original_filename=record.original_filename,
        file_type=SupportedFileType(record.file_type),
        size_bytes=record.size_bytes,
        status=ProcessingStatus(record.status),
        error_message=record.error_message,
        failure_code=record.failure_code,
        content_hash=record.content_hash,
        page_count=record.page_count,
        processed_at=record.processed_at,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


class DocumentRepository:
    """Reads and writes document records.

    Args:
        session: The request-scoped async session. The repository never commits —
            transaction boundaries belong to the caller (see ``get_db_session``).
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        owner_id: str,
        original_filename: str,
        file_type: SupportedFileType,
        size_bytes: int,
        status: ProcessingStatus = ProcessingStatus.PENDING,
        content_hash: str | None = None,
    ) -> Document:
        """Insert a new document record and return it."""
        record = DocumentRecord(
            owner_id=owner_id,
            original_filename=original_filename,
            file_type=file_type,
            size_bytes=size_bytes,
            status=status,
            content_hash=content_hash,
        )
        self._session.add(record)
        await self._session.flush()
        await self._session.refresh(record)
        return _to_domain(record)

    async def get(self, document_id: UUID, *, owner_id: str) -> Document | None:
        """Fetch one document belonging to this owner.

        Returns ``None`` both when the document does not exist and when it belongs to
        someone else. The two cases are deliberately indistinguishable to the caller so
        the API cannot be used to probe for the existence of other users' documents.
        """
        record = await self._get_record(document_id, owner_id=owner_id)
        return _to_domain(record) if record else None

    async def list_for_owner(
        self,
        *,
        owner_id: str,
        limit: int = 50,
        offset: int = 0,
        status: ProcessingStatus | None = None,
    ) -> list[Document]:
        """List an owner's documents, newest first."""
        query = select(DocumentRecord).where(DocumentRecord.owner_id == owner_id)
        if status is not None:
            query = query.where(DocumentRecord.status == status)

        query = query.order_by(DocumentRecord.created_at.desc()).limit(limit).offset(offset)

        result = await self._session.execute(query)
        return [_to_domain(record) for record in result.scalars().all()]

    async def count_for_owner(
        self, *, owner_id: str, status: ProcessingStatus | None = None
    ) -> int:
        """Count an owner's documents, so a list response can report a total."""
        query = (
            select(func.count())
            .select_from(DocumentRecord)
            .where(DocumentRecord.owner_id == owner_id)
        )
        if status is not None:
            query = query.where(DocumentRecord.status == status)

        result = await self._session.execute(query)
        return int(result.scalar_one())

    async def update_status(
        self,
        document_id: UUID,
        *,
        owner_id: str,
        status: ProcessingStatus,
        error_message: str | None = None,
        failure_code: str | None = None,
        page_count: int | None = None,
    ) -> Document | None:
        """Move a document to a new processing status.

        When moving to ``FAILED`` an ``error_message`` is required: a failed document
        must always be able to tell the user why. Moving to any non-failed status clears
        the previous error and failure code, so stale text cannot linger on a document
        that has since recovered.

        Reaching a terminal status stamps ``processed_at``, which is also what lets the
        startup reaper recognise a document abandoned mid-parse by a crash.
        """
        if status is ProcessingStatus.FAILED and not error_message:
            raise ValueError("error_message is required when setting status to FAILED")

        record = await self._get_record(document_id, owner_id=owner_id)
        if record is None:
            return None

        record.status = status
        if status is ProcessingStatus.FAILED:
            record.error_message = error_message
            record.failure_code = failure_code
        else:
            record.error_message = None
            record.failure_code = None

        if page_count is not None:
            record.page_count = page_count

        if status.is_terminal:
            record.processed_at = utcnow()

        await self._session.flush()
        await self._session.refresh(record)
        return _to_domain(record)

    async def delete(self, document_id: UUID, *, owner_id: str) -> bool:
        """Delete a document record. Returns whether a row was removed."""
        record = await self._get_record(document_id, owner_id=owner_id)
        if record is None:
            return False

        await self._session.delete(record)
        await self._session.flush()
        return True

    async def fail_stranded_processing(self, *, reason: str, failure_code: str) -> int:
        """Mark every document stuck in PROCESSING as failed. Returns how many.

        Called once at startup. A document is only ever in PROCESSING while a background
        task is actively working on it, so if any are in that state when the process
        starts, the previous process died mid-parse and nothing will ever finish them.
        Without this they would sit in a spinner forever.

        Not owner-scoped: this is an operational sweep across all documents, not a user
        request.
        """
        result = await self._session.execute(
            select(DocumentRecord).where(DocumentRecord.status == ProcessingStatus.PROCESSING)
        )
        records = list(result.scalars().all())

        for record in records:
            record.status = ProcessingStatus.FAILED
            record.error_message = reason
            record.failure_code = failure_code
            record.processed_at = utcnow()

        await self._session.flush()
        return len(records)

    async def _get_record(self, document_id: UUID, *, owner_id: str) -> DocumentRecord | None:
        """Fetch the raw ORM record, always scoped by owner."""
        result = await self._session.execute(
            select(DocumentRecord).where(
                DocumentRecord.id == document_id,
                DocumentRecord.owner_id == owner_id,
            )
        )
        return result.scalar_one_or_none()
