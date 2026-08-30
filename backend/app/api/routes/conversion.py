"""Document conversion endpoints.

Converting costs several model calls and takes seconds, so it is a ``POST`` that creates
a stored result: the caller gets an id back and can fetch the same document again later
without paying for it twice, or a browser reloading the page re-running it.

Retrieval is **database only**. No route here consults a provider on a ``GET``, and none
of them re-runs the conversion — the stored result is the answer, exactly as it was
produced. That is what makes a conversion something a user can rely on rather than a
number that changes each time they look.

The routes are thin by design: validate, authorise, call
:class:`~app.services.conversion_service.ConversionService`, persist, respond. All the
orchestration — extraction, planning, correction validation, assembly, rendering — lives
in the service and its collaborators, and none of it is repeated here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from fastapi import APIRouter, Body, Path, Query, Response, status

from app.api.deps import ConversionRepositoryDep, ConversionServiceDep, OwnerIdDep
from app.api.responses import error_responses
from app.api.schemas.conversion import (
    ConversionCorrectionResponse,
    ConversionListResponse,
    ConversionPlanResponse,
    ConversionPlanSectionResponse,
    ConversionRejectionResponse,
    ConversionRequest,
    ConversionResponse,
    ConversionSummaryResponse,
    ConvertedFieldResponse,
    ConvertedRecordResponse,
    ConvertedSectionResponse,
    ConvertedValueResponse,
)
from app.core.exceptions import NotFoundError
from app.core.security import content_disposition_attachment
from app.models.domain.conversion import ConvertedRecord, ConvertedValue
from app.services.conversion_export import ConversionExportFormat, export_conversion
from app.services.conversion_service import ConversionResult

router = APIRouter(prefix="/documents", tags=["conversion"])

#: Errors every conversion endpoint can produce.
_CONVERSION_ERRORS = (
    status.HTTP_404_NOT_FOUND,
    status.HTTP_422_UNPROCESSABLE_CONTENT,
    status.HTTP_500_INTERNAL_SERVER_ERROR,
)


@router.post(
    "/{document_id}/convert",
    response_model=ConversionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Convert a document into a new, organized document",
    response_description="The converted document, its rendering, and the corrections audit.",
    responses=error_responses(*_CONVERSION_ERRORS),
)
async def convert_document(
    service: ConversionServiceDep,
    conversions: ConversionRepositoryDep,
    owner_id: OwnerIdDep,
    document_id: UUID = Path(description="The document to convert."),
    request: ConversionRequest = Body(...),
) -> ConversionResponse:
    """Build a new, organized document out of an existing one.

    Describe the document you want in plain language — how to group it, what to
    standardise, what to separate out — and a new document is produced from the source's
    own content.

    **Nothing is invented.** Every value shown is either copied from your document or a
    recorded correction of something copied from it, and each correction says what it
    changed, why, and how confident it was. Corrections the system could not prove are
    *not* applied: the original value stays, and the suggestion is listed so you can
    decide. Changes it judged unsafe are refused outright and reported under `rejected`.

    The AI decides only how the document should be organised and which values look
    inconsistent. The document itself is assembled from your data by ordinary code, so
    the same source and the same decisions always produce the same document.

    The result is stored. Use the returned `id` with `GET
    /documents/{document_id}/conversions/{conversion_id}` to fetch it again — that
    retrieval costs nothing and never re-runs the AI.
    """
    result = await service.convert(
        document_id,
        request.instruction,
        owner_id=owner_id,
        refresh=request.refresh,
    )

    # Persisted before responding: a conversion the caller can see but not fetch again
    # would be worse than a clear failure. A write error propagates, the request-scoped
    # session rolls back, and no half-written row survives.
    payload = _to_response(
        result,
        conversion_id=uuid4(),
        document_id=document_id,
        instruction=request.instruction,
        created_at=datetime.now(UTC),
    )
    stored = await conversions.save(payload, document_id=document_id, owner_id=owner_id)
    return stored.to_response()


@router.get(
    "/{document_id}/conversions",
    response_model=ConversionListResponse,
    summary="List a document's conversions",
    response_description="Every conversion produced for this document, newest first.",
    responses=error_responses(status.HTTP_404_NOT_FOUND, status.HTTP_422_UNPROCESSABLE_CONTENT),
)
async def list_conversions(
    conversions: ConversionRepositoryDep,
    owner_id: OwnerIdDep,
    document_id: UUID = Path(description="The document."),
) -> ConversionListResponse:
    """Return the conversions already produced for a document.

    Read-only and free: this never calls the AI. Summaries only — fetch one by id to get
    its content.
    """
    stored = await conversions.list_for_document(document_id, owner_id=owner_id)
    return ConversionListResponse(
        items=[
            ConversionSummaryResponse(
                id=item.id,
                document_id=item.document_id,
                title=item.title,
                instruction=item.instruction,
                source_record_count=item.source_record_count,
                applied_correction_count=len(item.corrections),
                suggestion_count=len(item.suggestions),
                rejected_correction_count=len(item.rejected),
                created_at=item.created_at,
            )
            for item in stored
        ]
    )


@router.get(
    "/{document_id}/conversions/{conversion_id}",
    response_model=ConversionResponse,
    summary="Get a previously converted document",
    response_description="The stored conversion, exactly as it was produced.",
    responses=error_responses(status.HTTP_404_NOT_FOUND, status.HTTP_422_UNPROCESSABLE_CONTENT),
)
async def get_conversion(
    conversions: ConversionRepositoryDep,
    owner_id: OwnerIdDep,
    document_id: UUID = Path(description="The document."),
    conversion_id: UUID = Path(description="The conversion to fetch."),
) -> ConversionResponse:
    """Return a stored conversion.

    Read-only and free: the stored result is returned as it was produced, with nothing
    regenerated and no AI call made. A conversion belonging to another owner is reported
    as not found rather than forbidden — knowing an id must not confirm that it exists.
    """
    stored = await conversions.get(conversion_id, document_id=document_id, owner_id=owner_id)
    if stored is None:
        raise NotFoundError(
            "Conversion not found",
            details={"conversion_id": str(conversion_id)},
        )
    return stored.to_response()


@router.get(
    "/{document_id}/conversions/{conversion_id}/export",
    summary="Download a converted document",
    response_description="The converted document as a downloadable file.",
    responses=error_responses(status.HTTP_404_NOT_FOUND, status.HTTP_422_UNPROCESSABLE_CONTENT),
)
async def export_conversion_file(
    conversions: ConversionRepositoryDep,
    owner_id: OwnerIdDep,
    document_id: UUID = Path(description="The document."),
    conversion_id: UUID = Path(description="The conversion to download."),
    export_format: ConversionExportFormat = Query(
        alias="format",
        description="The file format: 'md', 'docx', 'xlsx', or 'json'.",
    ),
) -> Response:
    """Download a stored conversion as a file.

    Reads the stored result and reformats it — no AI call, no re-extraction, and no
    re-running of the conversion, so a download is free and returns the same bytes every
    time.

    Markdown and DOCX are the readable report; XLSX gives the same records as a
    worksheet per section; JSON carries the whole result including the corrections that
    were applied, suggested and refused. Every format that can carry the audit does,
    because a tidied report that does not say what was tidied is the thing this feature
    exists to avoid.
    """
    stored = await conversions.get(conversion_id, document_id=document_id, owner_id=owner_id)
    if stored is None:
        raise NotFoundError(
            "Conversion not found",
            details={"conversion_id": str(conversion_id)},
        )

    exported = export_conversion(stored.to_response(), export_format)
    return Response(
        content=exported.content,
        media_type=exported.media_type,
        headers={"Content-Disposition": content_disposition_attachment(exported.filename)},
    )


def _to_response(
    result: ConversionResult,
    *,
    conversion_id: UUID,
    document_id: UUID,
    instruction: str,
    created_at: datetime,
) -> ConversionResponse:
    """Shape a service result for the wire.

    A pure mapping. The two things it deliberately drops are the chunk and block
    identifiers behind each citation: they are how *we* find a location, not something a
    reader can use, and they have no place in a document a user downloads.
    """
    return ConversionResponse(
        id=conversion_id,
        document_id=document_id,
        title=result.document.title,
        instruction=instruction,
        markdown=result.markdown,
        sections=[
            ConvertedSectionResponse(
                heading=section.heading,
                kind=section.kind,
                columns=list(section.columns),
                records=[_record_to_response(record) for record in section.records],
                note=section.note,
            )
            for section in result.document.sections
        ],
        plan=ConversionPlanResponse(
            title=result.plan.title,
            sections=[
                ConversionPlanSectionResponse(
                    heading=section.heading,
                    kind=section.kind,
                    columns=list(section.columns),
                    group_by=section.group_by,
                    note=section.note,
                )
                for section in result.plan.sections
            ],
            labels=dict(result.plan.labels),
        ),
        corrections=[_correction_to_response(item) for item in result.review.applied],
        suggestions=[_correction_to_response(item) for item in result.review.suggestions],
        rejected=[
            ConversionRejectionResponse(
                kind=item.proposal.kind.value,
                record_index=item.proposal.record_index,
                field=item.proposal.field,
                original=item.proposal.original,
                corrected=item.proposal.corrected,
                confidence=item.proposal.confidence.value,
                rejection_reason=item.reason,
            )
            for item in result.review.rejected
        ],
        schema_used=dict(result.schema_used),
        vocabularies={name: list(values) for name, values in result.vocabularies.items()},
        source_record_count=result.source_record_count,
        warnings=list(result.warnings),
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        records_from_cache=result.records_from_cache,
        created_at=created_at,
    )


def _record_to_response(record: ConvertedRecord) -> ConvertedRecordResponse:
    return ConvertedRecordResponse(
        source_record_index=record.source_record_index,
        fields=[
            ConvertedFieldResponse(
                name=item.name,
                label=item.label,
                value=_value_to_response(item.value),
            )
            for item in record.fields
        ],
    )


def _value_to_response(value: ConvertedValue) -> ConvertedValueResponse:
    citation = value.citation

    # Only a confirmed location is reported. An unverified citation contributes nothing
    # rather than a plausible-looking label — the same rule the renderer follows.
    if citation is None or not citation.is_verified:
        return ConvertedValueResponse(
            text=value.text,
            origin=value.origin,
            original_text=value.original_text,
            correction_id=value.correction_id,
            is_verified=False,
        )

    return ConvertedValueResponse(
        text=value.text,
        origin=value.origin,
        original_text=value.original_text,
        correction_id=value.correction_id,
        source_label=citation.location_label,
        quoted_text=citation.quoted_text,
        is_verified=True,
    )


def _correction_to_response(item: object) -> ConversionCorrectionResponse:
    return ConversionCorrectionResponse.model_validate(item, from_attributes=True)
