"""Document endpoints.

Milestone 1 exposes the read side of the document lifecycle. Upload (``POST``) arrives in
Milestone 2 together with the parsing pipeline, because accepting a file that nothing can
process would be a half-built feature rather than a foundation.

Route handlers stay thin on purpose: validate input, call a service, shape the response.
No business rules and no database queries live here.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Path, Query, status

from app.api.deps import DocumentServiceDep, OwnerIdDep
from app.api.responses import error_responses
from app.api.schemas.document import DocumentListResponse, DocumentResponse
from app.models.domain.document import ProcessingStatus

router = APIRouter(prefix="/documents", tags=["documents"])


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
