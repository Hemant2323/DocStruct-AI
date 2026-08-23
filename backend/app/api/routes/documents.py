"""Document endpoints.

Milestone 1 exposes the read side of the document lifecycle. Upload (``POST``) arrives in
Milestone 2 together with the parsing pipeline, because accepting a file that nothing can
process would be a half-built feature rather than a foundation.

Route handlers stay thin on purpose: validate input, call a service, shape the response.
No business rules and no database queries live here.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, File, Path, Query, UploadFile, status

from app.api.deps import (
    DocumentServiceDep,
    IngestionServiceDep,
    JobServiceDep,
    OwnerIdDep,
    SettingsDep,
)
from app.api.responses import error_responses
from app.api.schemas.document import (
    ContentBlockResponse,
    ContentResponse,
    DocumentListResponse,
    DocumentResponse,
)
from app.core.security import read_upload_within_limit
from app.models.domain.document import ProcessingStatus

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document",
    response_description=(
        "The document was accepted and queued. Processing has not finished yet — poll "
        "`GET /documents/{document_id}` until `status` leaves `processing`."
    ),
    responses=error_responses(
        status.HTTP_413_CONTENT_TOO_LARGE,
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        status.HTTP_422_UNPROCESSABLE_CONTENT,
    ),
)
async def upload_document(
    service: DocumentServiceDep,
    ingestion: IngestionServiceDep,
    jobs: JobServiceDep,
    owner_id: OwnerIdDep,
    settings: SettingsDep,
    file: UploadFile = File(description="The document to upload."),
) -> DocumentResponse:
    """Upload a PDF, DOCX, TXT, CSV or XLSX file.

    Returns **202 Accepted** immediately: the file is stored and recorded, but extracting
    its contents happens in the background, because parsing a large document can take far
    longer than an HTTP request should be held open.

    The response carries the new document with `status: "pending"`. Poll
    `GET /documents/{document_id}` until the status becomes `ready` or `failed`.

    **The file's type is determined by its content, not its name.** A `.pdf` extension on
    a Word document is detected and handled correctly; an unsupported format is rejected
    with 415 whatever it claims to be.
    """
    upload = await read_upload_within_limit(file, settings.max_upload_size_bytes)

    document = await service.create_from_upload(
        owner_id=owner_id,
        filename=file.filename,
        content=upload.content,
        size_bytes=upload.size_bytes,
        content_hash=upload.sha256,
    )

    # Scheduled after the record exists, so the background task always finds its row.
    # The task opens its own session; the one used here is closed with this request.
    jobs.schedule(
        f"ingest:{document.id}",
        lambda: ingestion.process_document(document.id, owner_id=owner_id),
    )

    return DocumentResponse.from_domain(document)


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List documents",
    response_description="A page of documents, newest first, plus the total count.",
    responses=error_responses(status.HTTP_422_UNPROCESSABLE_CONTENT),
)
async def list_documents(
    service: DocumentServiceDep,
    owner_id: OwnerIdDep,
    limit: int = Query(
        default=50,
        ge=1,
        le=100,
        description="Maximum number of documents to return.",
        examples=[50],
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Number of documents to skip. Combine with `limit` to page through.",
        examples=[0],
    ),
    status_filter: ProcessingStatus | None = Query(
        default=None,
        alias="status",
        description=(
            "Return only documents in this processing state. Omit for all states. "
            "Useful for polling — `?status=processing` shows what is still in flight."
        ),
    ),
) -> DocumentListResponse:
    """List the documents belonging to the current user.

    Results are ordered newest first. `total` counts every document matching the filter
    across all pages, so a client can tell whether more results exist without fetching
    them.

    Documents belonging to other users are never included.
    """
    documents, total = await service.list_documents(
        owner_id=owner_id, limit=limit, offset=offset, status=status_filter
    )
    return DocumentListResponse(
        items=[DocumentResponse.from_domain(document) for document in documents],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get a document",
    response_description="The document's metadata and current processing status.",
    responses=error_responses(status.HTTP_404_NOT_FOUND, status.HTTP_422_UNPROCESSABLE_CONTENT),
)
async def get_document(
    service: DocumentServiceDep,
    owner_id: OwnerIdDep,
    document_id: UUID = Path(description="Identifier returned when the document was created."),
) -> DocumentResponse:
    """Fetch one document's metadata and processing status.

    This is the endpoint to poll while a document is being processed: repeat the call
    until `status` becomes `ready` or `failed`. When it is `failed`, `error_message`
    explains why in plain language.

    Returns 404 both when the document does not exist and when it belongs to another
    user — the two are indistinguishable on purpose, so the API cannot be used to probe
    for other users' document IDs.
    """
    document = await service.get_document(document_id, owner_id=owner_id)
    return DocumentResponse.from_domain(document)


@router.get(
    "/{document_id}/content",
    response_model=ContentResponse,
    summary="Get a document's extracted content",
    response_description="A page of extracted blocks, in reading order.",
    responses=error_responses(status.HTTP_404_NOT_FOUND, status.HTTP_422_UNPROCESSABLE_CONTENT),
)
async def get_document_content(
    service: DocumentServiceDep,
    owner_id: OwnerIdDep,
    document_id: UUID = Path(description="Identifier of the document."),
    limit: int = Query(default=100, ge=1, le=500, description="Maximum blocks to return."),
    offset: int = Query(default=0, ge=0, description="Blocks to skip, for pagination."),
) -> ContentResponse:
    """Return the structured content extracted from a document.

    Each block carries a `location_label` saying where in the original file it came from
    — "page 4", "lines 40-48", "Q3 Budget!A12:F30". That is what later milestones turn
    into citations, and what makes an answer checkable rather than merely plausible.

    Returns an empty list while the document is still `pending` or `processing`.
    """
    blocks, total = await service.get_content_blocks(
        document_id, owner_id=owner_id, limit=limit, offset=offset
    )
    return ContentResponse(
        items=[ContentBlockResponse.from_domain(block) for block in blocks],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
    response_description="The document was deleted. No response body.",
    responses=error_responses(status.HTTP_404_NOT_FOUND, status.HTTP_422_UNPROCESSABLE_CONTENT),
)
async def delete_document(
    service: DocumentServiceDep,
    owner_id: OwnerIdDep,
    document_id: UUID = Path(description="Identifier of the document to delete."),
) -> None:
    """Permanently delete a document and its stored file.

    This cannot be undone. The stored file is removed first; if that fails the database
    record is kept, so the document stays visible and the delete can be retried rather
    than leaving an orphaned file behind.
    """
    await service.delete_document(document_id, owner_id=owner_id)
