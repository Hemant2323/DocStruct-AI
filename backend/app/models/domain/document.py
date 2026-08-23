"""Domain models for documents.

These are the objects the service layer works with. They are deliberately separate from
the SQLAlchemy tables in ``app.models.db``: persistence concerns (columns, indexes,
nullability) should not leak into business logic, and business logic should not be
reshaped every time the schema changes. Repositories translate between the two.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID


class ProcessingStatus(StrEnum):
    """Lifecycle of a document as it moves through the ingestion pipeline.

    The transitions are::

        PENDING -> PROCESSING -> READY
                              -> FAILED

    ``PENDING`` is set the moment an upload is accepted, before any parsing begins, so
    the API can respond immediately and the client can poll for progress.
    """

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        """True when no further status change is expected."""
        return self in {ProcessingStatus.READY, ProcessingStatus.FAILED}


class SupportedFileType(StrEnum):
    """File types DocStruct AI can ingest.

    Milestone 1 declares the set; the parsers that handle them arrive in Milestone 2.
    Values are the canonical extension without the leading dot.
    """

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    CSV = "csv"
    XLSX = "xlsx"

    @property
    def media_type(self) -> str:
        """The MIME type normally associated with this file type."""
        return _MEDIA_TYPES[self]

    @classmethod
    def from_filename(cls, filename: str) -> SupportedFileType | None:
        """Best-effort type guess from a filename's extension.

        This is a *hint only*. It must never be the sole basis for accepting a file —
        content sniffing decides that (see CLAUDE.md §8). A file named ``report.pdf``
        can contain anything at all.

        Returns:
            The matching type, or ``None`` if the extension is not recognised.
        """
        _, _, extension = filename.rpartition(".")
        if not extension or extension == filename:
            return None
        try:
            return cls(extension.lower())
        except ValueError:
            return None


_MEDIA_TYPES: dict[SupportedFileType, str] = {
    SupportedFileType.PDF: "application/pdf",
    SupportedFileType.DOCX: (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ),
    SupportedFileType.TXT: "text/plain",
    SupportedFileType.CSV: "text/csv",
    SupportedFileType.XLSX: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


@dataclass(frozen=True, slots=True)
class Document:
    """A document known to the system, as the service layer sees it.

    Attributes:
        id: Server-generated identifier. Also the name the file is stored under, which
            is why a user-supplied filename never touches the filesystem.
        owner_id: Who the document belongs to. Every read is scoped by this to prevent
            cross-tenant leaks. Milestone 1 has no authentication, so a single default
            owner is used; the seam is here so adding auth later changes one function.
        original_filename: The name the user uploaded, kept for display only.
        file_type: Which parser will handle it.
        size_bytes: Size of the stored file.
        status: Where it is in the processing lifecycle.
        error_message: Why processing failed, when ``status`` is ``FAILED``. Always
            populated on failure — a failed document never gets an empty reason.
        created_at: When the upload was accepted.
        updated_at: When the record last changed.
    """

    id: UUID
    owner_id: str
    original_filename: str
    file_type: SupportedFileType
    size_bytes: int
    status: ProcessingStatus
    error_message: str | None
    created_at: datetime
    updated_at: datetime
