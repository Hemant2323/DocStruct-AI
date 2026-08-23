"""The ``documents`` table.

Milestone 1 stores document metadata and processing status only. Extracted content
(``content_blocks``) and retrieval chunks (``chunks``) arrive in Milestone 2 as separate
tables referencing this one.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.db.base import GUID, Base, UTCDateTime, utcnow
from app.models.domain.document import ProcessingStatus, SupportedFileType


class DocumentRecord(Base):
    """Persistence model for an uploaded document."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)

    # Every read is scoped by owner. Milestone 1 has no auth and uses a single default
    # owner, but the column exists now so adding authentication later is a change to the
    # dependency that supplies the value, not a schema migration plus a query audit.
    owner_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    # Display only. The file on disk is named after `id`, never after this value.
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)

    file_type: Mapped[SupportedFileType] = mapped_column(String(16), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)

    status: Mapped[ProcessingStatus] = mapped_column(
        String(16), nullable=False, default=ProcessingStatus.PENDING, index=True
    )

    # Populated whenever status is FAILED, so a failure is never silent.
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        # Listing a user's documents newest-first is the most common query.
        Index("ix_documents_owner_created", "owner_id", "created_at"),
        CheckConstraint("size_bytes >= 0", name="ck_documents_size_non_negative"),
    )

    def __repr__(self) -> str:
        return f"<DocumentRecord id={self.id} status={self.status} type={self.file_type}>"
