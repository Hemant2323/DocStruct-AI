"""AI analysis endpoints.

Each of these can cost money and take a while, so they are ``POST`` rather than ``GET``:
they are not safely repeatable, and a browser must not fire one by prefetching a link.

Results are cached, so calling the same endpoint twice returns the first result rather
than paying twice. Pass ``refresh`` to override that deliberately.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Body, Path, Query, Response, status

from app.ai.optimization import should_offer_optimization
from app.api.deps import AnalysisServiceDep, ExportServiceDep, OwnerIdDep, QAServiceDep, SettingsDep
from app.api.responses import error_responses
from app.api.schemas.analysis import (
    AIStatusResponse,
    AnalysisRequest,
    ContextPreviewResponse,
    ExtractionRequest,
    InsightListResponse,
    InsightResponse,
)
from app.api.schemas.export import StructureExportRequest
from app.api.schemas.qa import AskRequest, AskResponse
from app.api.schemas.structure import StructureRequest, StructureResponse
from app.core.security import content_disposition_attachment
from app.services.export_service import ExportFormat

router = APIRouter(prefix="/documents", tags=["analysis"])

#: Errors every analysis endpoint can produce.
_ANALYSIS_ERRORS = (
    status.HTTP_404_NOT_FOUND,
    status.HTTP_422_UNPROCESSABLE_CONTENT,
    status.HTTP_500_INTERNAL_SERVER_ERROR,
)


@router.post(
    "/{document_id}/summary",
    response_model=InsightResponse,
    summary="Summarise a document",
    response_description="A concise summary, with the sources it is based on.",
    responses=error_responses(*_ANALYSIS_ERRORS),
)
async def summarize_document(
    service: AnalysisServiceDep,
    owner_id: OwnerIdDep,
    document_id: UUID = Path(description="The document to summarise."),
    request: AnalysisRequest | None = None,
) -> InsightResponse:
    """Produce a short summary of what a document says.

    The summary is grounded in the document only — the model is instructed not to add
    outside knowledge, and to say so rather than guess when something is unclear.

    Each part of the summary carries citations pointing at the exact places in your file
    it came from. A location is only shown when it was resolved from our own stored
    content; anything we cannot confirm is marked "source not verified" rather than
    given a plausible-looking page number.

    The result is cached. Calling again returns the same answer without paying for
    another AI request unless you pass `refresh`.
    """
    options = request or AnalysisRequest()
    insight = await service.summarize(
        document_id,
        owner_id=owner_id,
        optimize_context=options.optimize_context,
        refresh=options.refresh,
    )
    return InsightResponse.from_domain(insight)


@router.post(
    "/{document_id}/explanation",
    response_model=InsightResponse,
    summary="Explain a document in simple language",
    response_description="A plain-language explanation, with its sources.",
    responses=error_responses(*_ANALYSIS_ERRORS),
)
async def explain_document(
    service: AnalysisServiceDep,
    owner_id: OwnerIdDep,
    document_id: UUID = Path(description="The document to explain."),
    request: AnalysisRequest | None = None,
) -> InsightResponse:
    """Explain a document for someone unfamiliar with its subject.

    Specialist terms are explained in ordinary words, but conditions, exceptions,
    deadlines and figures are kept exactly as written — those are usually the parts that
    matter most, and dropping them is how a "simple" explanation becomes misleading.

    Like the summary, this is grounded in the document and cited back to it.
    """
    options = request or AnalysisRequest()
    insight = await service.explain(
        document_id,
        owner_id=owner_id,
        optimize_context=options.optimize_context,
        refresh=options.refresh,
    )
    return InsightResponse.from_domain(insight)


@router.post(
    "/{document_id}/extraction",
    response_model=InsightResponse,
    summary="Extract structured information",
    response_description="The facts found in the document, each with its source.",
    responses=error_responses(*_ANALYSIS_ERRORS),
)
async def extract_from_document(
    service: AnalysisServiceDep,
    owner_id: OwnerIdDep,
    document_id: UUID = Path(description="The document to extract from."),
    request: ExtractionRequest | None = None,
) -> InsightResponse:
    """Pull specific facts out of a document as structured data.

    The model must return a fixed shape, which is validated before you see it — a
    malformed reply fails loudly instead of becoming half-correct data.

    For each fact the model also supplies the exact wording it relied on. That wording is
    then **searched for in the stored document text**. A fact whose quote is found gets a
    confirmed source; one whose quote cannot be found is still returned, but explicitly
    without a location. The model's own claim about where something appears is never
    trusted.
    """
    options = request or ExtractionRequest()
    insight = await service.extract(
        document_id,
        owner_id=owner_id,
        schema_name=options.extraction_type,
        optimize_context=options.optimize_context,
        refresh=options.refresh,
    )
    return InsightResponse.from_domain(insight)


@router.post(
    "/{document_id}/structure",
    response_model=StructureResponse,
    summary="Convert a document into a structured, multi-record table",
    response_description="One record per matching entity in the document, with sources.",
    responses=error_responses(*_ANALYSIS_ERRORS),
)
async def structure_document(
    service: AnalysisServiceDep,
    owner_id: OwnerIdDep,
    document_id: UUID = Path(description="The document to extract structured data from."),
    request: StructureRequest = Body(...),
) -> StructureResponse:
    """Turn a document into a structured table — one record per matching entity.

    A payment report listing twenty transactions returns twenty records, not just the
    first: each record holds the requested fields for one transaction. Say what you want
    in either of two ways, but not both:

    - **`schema`** — the exact fields, as a flat mapping of field name to type (`string`,
      `number`, `integer`, `boolean`, or `date`). Field names must be lowercase and
      contain only letters, numbers and underscores; at most 40 fields at once.
    - **`instruction`** — a plain-language description of the table you want ("a payment
      report with one row per transaction, including account holder, bank and amount").
      The fields are worked out from it; the schema arrived at is returned as
      `schema_used`, so you can see exactly what was extracted and reuse it.

    Every field is grounded exactly like `/extraction`: the model's quote is verified
    against the stored document text before any source location is shown. A field a given
    record does not state is left out of that record rather than guessed at.

    The result is cached per exact schema — the same fields, in any order, return the
    cached result unless `refresh` is set. An `instruction` request still costs one AI
    call to work out the schema, even when the result for that schema is already cached.
    """
    result = await service.structure(
        document_id,
        owner_id=owner_id,
        schema=request.requested_schema,
        instruction=request.instruction,
        refresh=request.refresh,
    )
    return StructureResponse.from_domain(result)


@router.post(
    "/{document_id}/structure/export",
    summary="Export a structure result as JSON, CSV, or XLSX",
    response_description="A downloadable file in the requested format.",
    responses=error_responses(*_ANALYSIS_ERRORS),
)
async def export_structure(
    service: ExportServiceDep,
    owner_id: OwnerIdDep,
    document_id: UUID = Path(description="The document whose structured result to export."),
    export_format: ExportFormat = Query(
        alias="format", description="The file format to export: 'json', 'csv', or 'xlsx'."
    ),
    request: StructureExportRequest = Body(...),
) -> Response:
    """Export an already-generated `/structure` result as a downloadable file.

    This never calls the AI model — it looks up the exact cached result for the schema
    you supply (the same one you used with `/structure`) and reports 404 if no such
    result has been generated yet, rather than generating one on the spot. Field order in
    the schema does not matter; the field names and types must match exactly.

    JSON includes the extracted fields with their values and verified sources. CSV and
    XLSX contain the same data as one row per field, with any value that could be
    misread as a spreadsheet formula (starting with `=`, `+`, `-`, or `@`) safely escaped.
    No internal identifiers (chunk IDs, block IDs, database IDs) are included in any format.
    """
    exported = await service.export_structure(
        document_id,
        owner_id=owner_id,
        schema=request.requested_schema,
        export_format=export_format,
    )
    return Response(
        content=exported.content,
        media_type=exported.media_type,
        headers={"Content-Disposition": content_disposition_attachment(exported.filename)},
    )


@router.get(
    "/{document_id}/insights",
    response_model=InsightListResponse,
    summary="Get previously generated AI results",
    response_description="Every AI result already produced for this document.",
    responses=error_responses(status.HTTP_404_NOT_FOUND, status.HTTP_422_UNPROCESSABLE_CONTENT),
)
async def list_insights(
    service: AnalysisServiceDep,
    owner_id: OwnerIdDep,
    document_id: UUID = Path(description="The document."),
) -> InsightListResponse:
    """Return AI results that have already been generated for a document.

    Read-only and free: this never triggers a new AI request. Use it to show what already
    exists when a document is opened.
    """
    insights = await service.list_insights(document_id, owner_id=owner_id)
    return InsightListResponse(items=[InsightResponse.from_domain(insight) for insight in insights])


@router.get(
    "/{document_id}/context-preview",
    response_model=ContextPreviewResponse,
    summary="Check how large a document is for the AI",
    response_description="The document's size for AI purposes, and whether it fits whole.",
    responses=error_responses(status.HTTP_404_NOT_FOUND, status.HTTP_422_UNPROCESSABLE_CONTENT),
)
async def context_preview(
    service: AnalysisServiceDep,
    settings: SettingsDep,
    owner_id: OwnerIdDep,
    document_id: UUID = Path(description="The document."),
) -> ContextPreviewResponse:
    """Report how much content would be sent to the AI, without calling it.

    Free and instant — this is ordinary arithmetic over stored content, not an AI request.
    It exists so the interface only offers "Optimize context for AI" on documents where
    it would actually help; on a small document, reducing context only loses detail.
    """
    stats = await service.context_preview(document_id, owner_id=owner_id)
    return ContextPreviewResponse.from_stats(
        stats,
        optimization_available=should_offer_optimization(
            stats.estimated_tokens, settings.ai_optimization_min_tokens
        ),
    )


@router.post(
    "/{document_id}/ask",
    response_model=AskResponse,
    summary="Ask a question about a document",
    response_description="An answer grounded in the document, with its sources.",
    responses=error_responses(*_ANALYSIS_ERRORS),
)
async def ask_document(
    service: QAServiceDep,
    owner_id: OwnerIdDep,
    document_id: UUID = Path(description="The document to ask about."),
    request: AskRequest = Body(...),
) -> AskResponse:
    """Ask a question about a document and get a grounded answer.

    The answer is based only on the parts of the document that retrieval judged most
    relevant to the question — never the model's own outside knowledge. If the retrieved
    content does not contain enough to answer, the model says so rather than guessing.

    Each answer is generated fresh. Unlike summaries and explanations, questions are not
    cached: two different questions about the same document have nothing in common to
    cache against, and this endpoint does not yet support multi-turn conversation.
    """
    result = await service.ask(
        document_id, request.question, owner_id=owner_id, top_k=request.top_k
    )
    return AskResponse.from_domain(result)


# Separate router: this one is about the server, not about a document.
status_router = APIRouter(tags=["analysis"])


@status_router.get(
    "/ai/status",
    response_model=AIStatusResponse,
    summary="Check whether AI features are available",
    response_description="Whether the server can make AI requests, and with which model.",
)
async def ai_status(settings: SettingsDep) -> AIStatusResponse:
    """Report whether AI features are configured on this server.

    Lets the interface disable the AI buttons with a clear explanation instead of letting
    a user click one and receive a failure.

    Only *whether* a key is present is reported. The key itself never leaves the server —
    it is not returned here in any form, not even partially.
    """
    return AIStatusResponse(configured=settings.ai_configured, model=settings.anthropic_model)
