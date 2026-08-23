"""Request and response schemas for AI analysis endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.domain.insight import (
    Citation,
    ContextMode,
    ContextStats,
    Insight,
    InsightKind,
)


class AnalysisRequest(BaseModel):
    """Options for an analysis request."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"optimize_context": False, "refresh": False}}
    )

    optimize_context: bool = Field(
        default=False,
        description=(
            "Reduce how much of the document is sent to the AI. Off by default. This "
            "affects only what the AI is given for this request — your uploaded file, "
            "its extracted content and its source locations are never changed. Only "
            "worth using on large documents."
        ),
    )
    refresh: bool = Field(
        default=False,
        description=(
            "Regenerate instead of returning the cached result. Costs another AI call, "
            "so leave it off unless you want a fresh answer."
        ),
    )


class ExtractionRequest(AnalysisRequest):
    """Options for a structured extraction request."""

    extraction_type: str = Field(
        default="key_facts",
        description="Which extraction schema to use. Currently only 'key_facts'.",
    )


class CitationResponse(BaseModel):
    """Where a piece of a result came from."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "quoted_text": "Invoices are due within 30 days.",
                "location_label": "page 2",
                "is_verified": True,
                "block_ids": ["0b6c1f3e-6a4d-4c58-9a1f-2b7d8e5c4a90"],
            }
        }
    )

    quoted_text: str = Field(description="The exact wording this is based on.")
    location_label: str = Field(
        description=(
            "Where that wording appears in the original file — 'page 4', 'lines 40-48', "
            "'Q3 Budget!A12:F30'. Reads 'source not verified' when the location could "
            "not be confirmed."
        )
    )
    is_verified: bool = Field(
        description=(
            "True when the location was resolved from our own stored content. False "
            "means we could not confirm it — no location is ever guessed or invented."
        )
    )
    block_ids: list[str] = Field(
        default_factory=list, description="Content blocks this citation resolved to."
    )

    @classmethod
    def from_domain(cls, citation: Citation) -> CitationResponse:
        return cls(
            quoted_text=citation.quoted_text,
            location_label=citation.location_label,
            is_verified=citation.is_verified,
            block_ids=[str(block_id) for block_id in citation.block_ids],
        )


class ExtractedFieldResponse(BaseModel):
    """One fact extracted from a document."""

    name: str = Field(description="What was extracted, e.g. 'total_amount'.")
    value: str = Field(description="The value exactly as written in the document.")
    citation: CitationResponse | None = Field(
        default=None,
        description=(
            "Where this came from. Null when the quote could not be found in the "
            "document, which is reported rather than guessed at."
        ),
    )


class InsightResponse(BaseModel):
    """An AI result about a document."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "kind": "summary",
                "content": "This is a service agreement between two parties…",
                "fields": [],
                "citations": [
                    {
                        "quoted_text": "Invoices are due within 30 days.",
                        "location_label": "page 2",
                        "is_verified": True,
                        "block_ids": [],
                    }
                ],
                "context_mode": "full",
                "model": "claude-opus-5",
                "warnings": [],
                "from_cache": False,
            }
        }
    )

    kind: InsightKind
    content: str = Field(default="", description="The prose result. Empty for extractions.")
    fields: list[ExtractedFieldResponse] = Field(
        default_factory=list, description="Extracted facts. Empty for prose results."
    )
    citations: list[CitationResponse] = Field(default_factory=list)
    context_mode: ContextMode = Field(
        description=(
            "'full' when the whole document was sent, 'optimized' when a reduced selection was."
        )
    )
    model: str = Field(description="Which model produced this.")
    input_tokens: int = 0
    output_tokens: int = 0
    warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Anything the user should know — content left out, sources that could not "
            "be confirmed. Surfaced rather than hidden."
        ),
    )
    verified_citation_count: int = Field(
        description="How many citations resolved to a confirmed location."
    )
    from_cache: bool = Field(
        description="True when this was a previously generated result rather than a new one."
    )
    created_at: datetime | None = None

    @classmethod
    def from_domain(cls, insight: Insight) -> InsightResponse:
        return cls(
            kind=insight.kind,
            content=insight.content,
            fields=[
                ExtractedFieldResponse(
                    name=item.name,
                    value=item.value,
                    citation=(
                        CitationResponse.from_domain(item.citation) if item.citation else None
                    ),
                )
                for item in insight.fields
            ],
            citations=[CitationResponse.from_domain(item) for item in insight.citations],
            context_mode=insight.context_mode,
            model=insight.model,
            input_tokens=insight.input_tokens,
            output_tokens=insight.output_tokens,
            warnings=list(insight.warnings),
            verified_citation_count=insight.verified_citation_count,
            from_cache=insight.from_cache,
            created_at=insight.created_at,
        )


class InsightListResponse(BaseModel):
    """Every AI result already produced for a document."""

    items: list[InsightResponse]


class ContextPreviewResponse(BaseModel):
    """How big this document is for the AI, without calling it.

    Lets the interface decide whether to offer the optimization option, and lets a user
    see the size of a document before spending anything on analysing it.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunk_count": 42,
                "estimated_tokens": 31500,
                "fits_in_full": True,
                "optimization_available": True,
            }
        }
    )

    chunk_count: int = Field(description="Sections the document has.")
    estimated_tokens: int = Field(
        description="Approximate size of the document for the AI. An estimate, not a bill."
    )
    fits_in_full: bool = Field(
        description="True when the whole document can be sent without leaving anything out."
    )
    optimization_available: bool = Field(
        description=(
            "True when this document is large enough that reducing the AI context is "
            "worth offering. Small documents should be sent whole."
        )
    )

    @classmethod
    def from_stats(
        cls, stats: ContextStats, *, optimization_available: bool
    ) -> ContextPreviewResponse:
        return cls(
            chunk_count=stats.total_chunks,
            estimated_tokens=stats.estimated_tokens,
            fits_in_full=stats.is_complete,
            optimization_available=optimization_available,
        )


class AIStatusResponse(BaseModel):
    """Whether AI features are usable on this server.

    Deliberately reports only *whether* a key is configured. The key itself, its length,
    and its prefix are all withheld — nothing about it leaves the server.
    """

    model_config = ConfigDict(
        json_schema_extra={"example": {"configured": True, "model": "claude-opus-5"}}
    )

    configured: bool = Field(description="True when the server can make AI requests.")
    model: str = Field(description="Which model will be used.")

    def model_post_init(self, __context: Any) -> None:  # pragma: no cover - trivial
        return None
