"""Request and response schemas for document endpoints.

These are the API's public contract. They are separate from the domain models so that an
internal refactor does not silently change the shape of the JSON clients depend on.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models.domain.document import Document, ProcessingStatus, SupportedFileType

# Annotated explicitly: the `None` value would otherwise make mypy infer
# dict[str, object], which is not assignable to Pydantic's JsonDict.
_DOCUMENT_EXAMPLE: dict[str, Any] = {
    "id": "0b6c1f3e-6a4d-4c58-9a1f-2b7d8e5c4a90",
    "original_filename": "quarterly-report.pdf",
    "file_type": "pdf",
    "size_bytes": 1048576,
    "status": "ready",
    "error_message": None,
    "created_at": "2026-08-23T11:11:51.884873Z",
    "updated_at": "2026-08-23T11:12:04.201118Z",
}


class DocumentResponse(BaseModel):
    """A single document's metadata and processing state.

    Note that `owner_id` is not part of this response. It is an internal access-control
    detail: every request is already scoped to its owner, so echoing it back would tell
    a client nothing it does not already know.
    """

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={"example": _DOCUMENT_EXAMPLE},
    )

    id: UUID = Field(
        description=(
            "Server-generated identifier. Also the name the file is stored under, which "
            "is why a user-supplied filename never reaches the filesystem."
        )
    )
    original_filename: str = Field(
        description="The filename as uploaded. Display only — never used as a path."
    )
    file_type: SupportedFileType = Field(
        description="Which of the supported formats this document is."
    )
    size_bytes: int = Field(description="Size of the stored file, in bytes.")
    status: ProcessingStatus = Field(
        description=(
            "Where the document is in the processing pipeline. "
            "`pending` — accepted, not yet started. "
            "`processing` — being extracted and structured. "
            "`ready` — extraction finished; the document can be used. "
            "`failed` — extraction did not complete; see `error_message`."
        )
    )
    error_message: str | None = Field(
        default=None,
        description=(
            "Why processing failed, in plain language. Non-null only when `status` is "
            "`failed`, and always populated in that case — a failed document can always "
            "explain itself."
        ),
    )
    created_at: datetime = Field(description="When the upload was accepted (UTC, ISO 8601).")
    updated_at: datetime = Field(
        description=(
            "When the record last changed (UTC, ISO 8601). Advances on every status "
            "transition, so it doubles as 'when processing last progressed'."
        )
    )

    @classmethod
    def from_domain(cls, document: Document) -> DocumentResponse:
        """Build a response from a domain object."""
        return cls(
            id=document.id,
            original_filename=document.original_filename,
            file_type=document.file_type,
            size_bytes=document.size_bytes,
            status=document.status,
            error_message=document.error_message,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )


class DocumentListResponse(BaseModel):
    """A page of documents plus the information needed to fetch the next one."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [_DOCUMENT_EXAMPLE],
                "total": 1,
                "limit": 50,
                "offset": 0,
            }
        }
    )

    items: list[DocumentResponse] = Field(description="This page of documents, newest first.")
    total: int = Field(
        description=(
            "Total documents matching the filter across all pages — not the number in "
            "`items`. Use it to decide whether another page exists."
        )
    )
    limit: int = Field(description="The `limit` that was applied.")
    offset: int = Field(description="The `offset` that was applied.")
