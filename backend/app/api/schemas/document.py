"""Request and response schemas for document endpoints.

These are the API's public contract. They are separate from the domain models so that an
internal refactor does not silently change the shape of the JSON clients depend on.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models.domain.content import BlockType, ContentBlock
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
    "failure_code": None,
    "page_count": 4,
    "processed_at": "2026-08-23T11:12:04.201118Z",
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
    failure_code: str | None = Field(
        default=None,
        description=(
            "Machine-readable failure reason, present only when `status` is `failed`. "
            "Branch on this rather than on the message text. Known values include "
            "`requires_ocr`, `encrypted_file`, `corrupt_file`, `no_content`, "
            "`too_complex` and `interrupted`."
        ),
    )
    page_count: int | None = Field(
        default=None,
        description=(
            "Pages for a document, worksheets for a workbook. Null until processing succeeds."
        ),
    )
    processed_at: datetime | None = Field(
        default=None,
        description="When extraction finished, successfully or not (UTC, ISO 8601).",
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
            failure_code=document.failure_code,
            page_count=document.page_count,
            processed_at=document.processed_at,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )


class ContentBlockResponse(BaseModel):
    """One extracted piece of content, with where it came from."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "order": 3,
                "block_type": "paragraph",
                "text": "Invoices are due within 30 days.",
                "level": None,
                "location_label": "page 2",
                "is_exact": True,
                "char_count": 32,
            }
        }
    )

    order: int = Field(description="Position in the document, starting at 0.")
    block_type: BlockType = Field(description="What kind of content this block holds.")
    text: str
    level: int | None = Field(default=None, description="Heading depth, for headings.")
    location_label: str = Field(
        description=(
            "Where this came from in the original file, e.g. 'page 4' or "
            "'Q3 Budget!A12:F30'. This is what a citation displays."
        )
    )
    is_exact: bool = Field(
        description=(
            "False when the location is approximate — for example a PDF heading inferred "
            "from font size. Uncertainty is recorded rather than hidden."
        )
    )
    char_count: int

    @classmethod
    def from_domain(cls, block: ContentBlock) -> ContentBlockResponse:
        return cls(
            order=block.order,
            block_type=block.block_type,
            text=block.text,
            level=block.level,
            location_label=block.provenance.label,
            is_exact=block.provenance.is_exact,
            char_count=block.char_count,
        )


class ContentResponse(BaseModel):
    """A page of a document's extracted content."""

    items: list[ContentBlockResponse]
    total: int = Field(description="Total blocks in the document, across all pages.")
    limit: int
    offset: int


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
