"""AI analysis endpoints.

Each of these can cost money and take a while, so they are ``POST`` rather than ``GET``:
they are not safely repeatable, and a browser must not fire one by prefetching a link.

Results are cached, so calling the same endpoint twice returns the first result rather
than paying twice. Pass ``refresh`` to override that deliberately.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Path, status

from app.ai.optimization import should_offer_optimization
from app.api.deps import AnalysisServiceDep, OwnerIdDep, SettingsDep
from app.api.responses import error_responses
from app.api.schemas.analysis import (
    AIStatusResponse,
    AnalysisRequest,
    ContextPreviewResponse,
    ExtractionRequest,
    InsightListResponse,
    InsightResponse,
)

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
