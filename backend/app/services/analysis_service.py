"""Orchestration for AI analysis of a document.

Decides what happens in what order; does not itself talk to HTTP, to the database driver,
or to the model's wire format.

The division of labour is the point of this milestone: **the model is used only where
language understanding is genuinely required.** Loading chunks, choosing what to send,
measuring size, resolving citations, validating output and caching are all ordinary code
here — deterministic, free, fast, and testable without a network.
"""

from __future__ import annotations

import hashlib
from uuid import UUID

from app.ai.citations import resolve_citations, verify_quote
from app.ai.context_builder import build_context, describe_context
from app.ai.prompts import explain, extract, summarize
from app.ai.providers.base import LLMProvider
from app.ai.schemas import DEFAULT_EXTRACTION_SCHEMA, get_extraction_schema
from app.core.config import Settings
from app.core.exceptions import (
    AIConfigurationError,
    InsufficientContentError,
    InvalidModelResponseError,
    NotFoundError,
)
from app.core.logging import get_logger
from app.models.domain.content import ContentBlock
from app.models.domain.document import Document, ProcessingStatus
from app.models.domain.insight import (
    Citation,
    ContextBlock,
    ContextStats,
    ExtractedField,
    Insight,
    InsightKind,
)
from app.repositories.ai_result_repo import AIResultRepository
from app.repositories.block_repo import ContentBlockRepository
from app.repositories.chunk_repo import ChunkRepository
from app.repositories.document_repo import DocumentRepository

logger = get_logger(__name__)


class AnalysisService:
    """Produces summaries, plain-language explanations and extracted facts."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        settings: Settings,
        documents: DocumentRepository,
        chunks: ChunkRepository,
        blocks: ContentBlockRepository,
        results: AIResultRepository,
    ) -> None:
        self._provider = provider
        self._settings = settings
        self._documents = documents
        self._chunks = chunks
        self._blocks = blocks
        self._results = results

    # ------------------------------------------------------------------ public
    async def summarize(
        self,
        document_id: UUID,
        *,
        owner_id: str,
        optimize_context: bool = False,
        refresh: bool = False,
    ) -> Insight:
        """Produce a concise summary of a document."""
        return await self._generate_prose(
            document_id,
            owner_id=owner_id,
            kind=InsightKind.SUMMARY,
            system=summarize.SYSTEM,
            instruction=summarize.INSTRUCTION,
            prompt_version=summarize.VERSION,
            optimize_context=optimize_context,
            refresh=refresh,
        )

    async def explain(
        self,
        document_id: UUID,
        *,
        owner_id: str,
        optimize_context: bool = False,
        refresh: bool = False,
    ) -> Insight:
        """Produce a plain-language explanation of a document."""
        return await self._generate_prose(
            document_id,
            owner_id=owner_id,
            kind=InsightKind.EXPLANATION,
            system=explain.SYSTEM,
            instruction=explain.INSTRUCTION,
            prompt_version=explain.VERSION,
            optimize_context=optimize_context,
            refresh=refresh,
        )

    async def extract(
        self,
        document_id: UUID,
        *,
        owner_id: str,
        schema_name: str = DEFAULT_EXTRACTION_SCHEMA,
        optimize_context: bool = False,
        refresh: bool = False,
    ) -> Insight:
        """Pull structured facts out of a document.

        Citations cannot be requested alongside a schema — the API rejects the
        combination — so each fact's source is established afterwards by checking the
        model's quote against our own stored content.
        """
        output_model = get_extraction_schema(schema_name)
        fingerprint = self._fingerprint(
            InsightKind.EXTRACTION, extract.VERSION, optimize_context, schema_name
        )

        if not refresh:
            cached = await self._results.get(
                document_id, owner_id=owner_id, fingerprint=fingerprint
            )
            if cached is not None:
                return _mark_cached(cached)

        document = await self._ready_document(document_id, owner_id=owner_id)
        context_blocks, stats, blocks_by_id = await self._prepare_context(
            document_id, owner_id=owner_id, optimize=optimize_context
        )

        self._require_provider()
        result = await self._provider.generate_structured(
            system=extract.SYSTEM,
            instruction=extract.INSTRUCTION,
            context_blocks=context_blocks,
            document_title=document.original_filename,
            output_model=output_model,
        )

        fields, citations, unverified = _fields_from_parsed(
            result.parsed, context_blocks, blocks_by_id
        )

        if not fields:
            raise InsufficientContentError("No clear facts could be extracted from this document.")

        warnings = list(describe_context(stats))
        if unverified:
            # Stated plainly rather than hidden: a value without a confirmed source is
            # still useful, but the user must know which ones those are.
            warnings.append(
                f"{unverified} of {len(fields)} facts could not be traced to a confirmed "
                "location in the document and are marked accordingly."
            )

        insight = Insight(
            document_id=document_id,
            kind=InsightKind.EXTRACTION,
            fields=tuple(fields),
            citations=tuple(citations),
            context_mode=stats.mode,
            model=self._provider.model_name,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            warnings=tuple(warnings),
        )
        return await self._results.save(insight, owner_id=owner_id, fingerprint=fingerprint)

    async def list_insights(self, document_id: UUID, *, owner_id: str) -> list[Insight]:
        """Return every AI result already produced for a document."""
        await self._ready_document(document_id, owner_id=owner_id, require_ready=False)
        return [
            _mark_cached(insight)
            for insight in await self._results.list_for_document(document_id, owner_id=owner_id)
        ]

    async def context_preview(self, document_id: UUID, *, owner_id: str) -> ContextStats:
        """Describe what would be sent to the model, without calling it.

        Lets the interface decide whether to offer the optimization option, and lets a
        user see the size of a document before spending anything on it.
        """
        await self._ready_document(document_id, owner_id=owner_id)
        _, stats, _ = await self._prepare_context(document_id, owner_id=owner_id, optimize=False)
        return stats

    # ----------------------------------------------------------------- private
    async def _generate_prose(
        self,
        document_id: UUID,
        *,
        owner_id: str,
        kind: InsightKind,
        system: str,
        instruction: str,
        prompt_version: str,
        optimize_context: bool,
        refresh: bool,
    ) -> Insight:
        """Shared path for summaries and explanations.

        Both send the same content with citations enabled and differ only in their
        prompts, so the pipeline lives here once.
        """
        fingerprint = self._fingerprint(kind, prompt_version, optimize_context)

        if not refresh:
            cached = await self._results.get(
                document_id, owner_id=owner_id, fingerprint=fingerprint
            )
            if cached is not None:
                return _mark_cached(cached)

        document = await self._ready_document(document_id, owner_id=owner_id)
        context_blocks, stats, blocks_by_id = await self._prepare_context(
            document_id, owner_id=owner_id, optimize=optimize_context
        )

        self._require_provider()
        result = await self._provider.generate(
            system=system,
            instruction=instruction,
            context_blocks=context_blocks,
            document_title=document.original_filename,
            enable_citations=True,
        )

        if not result.text.strip():
            raise InvalidModelResponseError("The AI returned an empty response.")

        citations = resolve_citations(result.raw_citations, context_blocks, blocks_by_id)

        warnings = list(describe_context(stats))
        unverified = sum(1 for citation in citations if not citation.is_verified)
        if unverified:
            warnings.append(
                f"{unverified} of {len(citations)} sources could not be traced to a "
                "confirmed location in the document."
            )

        insight = Insight(
            document_id=document_id,
            kind=kind,
            content=result.text,
            citations=tuple(citations),
            context_mode=stats.mode,
            model=self._provider.model_name,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            warnings=tuple(warnings),
        )

        logger.info(
            "Generated %s document_id=%s chunks=%d citations=%d verified=%d",
            kind.value,
            document_id,
            stats.chunk_count,
            len(citations),
            len(citations) - unverified,
        )
        return await self._results.save(insight, owner_id=owner_id, fingerprint=fingerprint)

    async def _ready_document(
        self, document_id: UUID, *, owner_id: str, require_ready: bool = True
    ) -> Document:
        """Fetch a document and confirm it can be analysed.

        Checked before anything is spent: analysing a document that failed to process
        would mean asking the model to interpret nothing.
        """
        document = await self._documents.get(document_id, owner_id=owner_id)
        if document is None:
            raise NotFoundError("Document not found", details={"document_id": str(document_id)})

        if require_ready and document.status is not ProcessingStatus.READY:
            if document.status is ProcessingStatus.FAILED:
                raise InsufficientContentError(
                    "This document could not be processed, so there is nothing to analyse. "
                    + (document.error_message or "")
                )
            raise InsufficientContentError(
                "This document is still being processed. Try again once it is ready."
            )

        return document

    async def _prepare_context(
        self, document_id: UUID, *, owner_id: str, optimize: bool
    ) -> tuple[list[ContextBlock], ContextStats, dict[UUID, ContentBlock]]:
        """Load the document's content and assemble the context. No model involved."""
        chunks = await self._chunks.list_for_document(document_id)

        context_blocks, stats = build_context(
            chunks,
            max_context_tokens=self._settings.ai_max_context_tokens,
            optimize=optimize,
            optimization_min_tokens=self._settings.ai_optimization_min_tokens,
            optimization_target_ratio=self._settings.ai_optimization_target_ratio,
        )

        # Only the blocks behind the chunks actually being sent are needed, since those
        # are the only ones a citation can point at.
        needed = [block_id for block in context_blocks for block_id in block.block_ids]
        blocks_by_id = await self._blocks.get_by_ids(list(dict.fromkeys(needed)))

        return context_blocks, stats, blocks_by_id

    def _require_provider(self) -> None:
        if not self._provider.is_configured:
            raise AIConfigurationError(
                "AI features are not configured on this server. An Anthropic API key is required."
            )

    def _fingerprint(
        self,
        kind: InsightKind,
        prompt_version: str,
        optimize_context: bool,
        schema_name: str = "",
    ) -> str:
        """Identify one exact analysis, for caching.

        Covers everything that could change the answer. Editing a prompt or switching
        models therefore produces a different fingerprint, and old results stop being
        served automatically instead of going stale in silence.
        """
        parts = [
            kind.value,
            self._provider.model_name,
            prompt_version,
            "optimized" if optimize_context else "full",
            schema_name,
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _mark_cached(insight: Insight) -> Insight:
    """Flag a result as served from cache rather than freshly generated."""
    return Insight(
        document_id=insight.document_id,
        kind=insight.kind,
        content=insight.content,
        fields=insight.fields,
        citations=insight.citations,
        context_mode=insight.context_mode,
        model=insight.model,
        input_tokens=insight.input_tokens,
        output_tokens=insight.output_tokens,
        warnings=insight.warnings,
        created_at=insight.created_at,
        from_cache=True,
    )


def _fields_from_parsed(
    parsed: object,
    context_blocks: list[ContextBlock],
    blocks_by_id: dict[UUID, ContentBlock],
) -> tuple[list[ExtractedField], list[Citation], int]:
    """Turn validated extraction output into fields with verified sources.

    Each fact's quote is searched for in the content we sent. A quote that is found gives
    the fact a real location; one that is not leaves it without a source. **The model's
    own belief about where something appears is never used** — only whether its quote
    actually occurs in our text.
    """
    facts = getattr(parsed, "facts", None) or []

    fields: list[ExtractedField] = []
    citations: list[Citation] = []
    unverified = 0

    for fact in facts:
        name = (getattr(fact, "name", "") or "").strip()
        value = (getattr(fact, "value", "") or "").strip()
        if not name or not value:
            continue

        quote = (getattr(fact, "source_quote", "") or "").strip()
        citation = verify_quote(quote, context_blocks, blocks_by_id) if quote else None

        if citation is None:
            unverified += 1
        else:
            citations.append(citation)

        fields.append(ExtractedField(name=name, value=value, citation=citation))

    return fields, citations, unverified
