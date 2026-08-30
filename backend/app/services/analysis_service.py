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
from dataclasses import dataclass
from uuid import UUID

from app.ai.citations import resolve_citations, verify_quote
from app.ai.context_builder import build_context, describe_context
from app.ai.dynamic_schema import (
    InferredStructureSchema,
    build_dynamic_record_list_model,
    canonicalize_schema,
    schema_from_inference,
)
from app.ai.prompts import explain, extract, summarize
from app.ai.prompts import structure as structure_prompt
from app.ai.prompts import structure_schema as structure_schema_prompt
from app.ai.providers.base import LLMProvider
from app.ai.schemas import DEFAULT_EXTRACTION_SCHEMA, get_extraction_schema
from app.core.config import Settings
from app.core.exceptions import (
    AIConfigurationError,
    InsufficientContentError,
    InvalidModelResponseError,
    NotFoundError,
    ValidationError,
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


@dataclass(frozen=True, slots=True)
class StructureResult:
    """A structure result together with the schema that produced it.

    ``schema`` is returned alongside the insight rather than derived from it because it
    genuinely cannot be reconstructed afterwards: in instruction mode the schema is
    inferred inside :meth:`AnalysisService.structure`, and the records alone do not carry
    it — a field the document never stated appears in no record, and no record carries a
    field's *type*. The API layer needs the exact resolved schema to report
    ``schema_used`` (and for the caller to hand back to `/structure/export`, which looks
    a cached result up by it), so it is handed out explicitly.
    """

    insight: Insight
    schema: dict[str, str]


def compute_fingerprint(
    kind: InsightKind,
    model_name: str,
    prompt_version: str,
    optimize_context: bool,
    *,
    schema_name: str = "",
    schema_fields: dict[str, str] | None = None,
) -> str:
    """The exact caching fingerprint for one analysis — module-level and stateless so it
    can be recomputed identically outside :class:`AnalysisService`.

    :meth:`AnalysisService._fingerprint` is the only caller inside this class (it just
    supplies ``self._provider.model_name``); the export feature (Milestone 4.6) is the
    reason this is a free function rather than a private method — export must look up an
    already-cached structure result without ever constructing a full ``AnalysisService``
    or calling a provider, so it needs this exact formula available on its own. Keeping
    one implementation, called from both places, is what guarantees the two can never
    silently drift apart — a copy-pasted second formula would be a live bug waiting for
    the day someone edits one copy and not the other.
    """
    parts = [
        kind.value,
        model_name,
        prompt_version,
        "optimized" if optimize_context else "full",
        schema_name,
    ]
    if schema_fields is not None:
        parts.append(canonicalize_schema(schema_fields))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


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

    async def structure(
        self,
        document_id: UUID,
        *,
        owner_id: str,
        schema: dict[str, str] | None = None,
        instruction: str | None = None,
        refresh: bool = False,
    ) -> StructureResult:
        """Extract a caller-defined or AI-inferred set of fields from a document, as a
        list of records — one per matching transaction or entity the document contains,
        not only the first (Milestone 4).

        Exactly one of ``schema`` and ``instruction`` must be given:

        - ``schema``: the caller supplies the exact fields wanted, as a flat mapping of
          field name to type — unchanged from the original single-record ``/structure``.
          The cache is checked before anything else is loaded, so a repeat request with
          the same schema never touches the document or the model.
        - ``instruction``: a free-text description ("one row per transaction, with
          account holder, bank and amount") is turned into a schema first, by a
          dedicated inference call (see ``app.ai.prompts.structure_schema``). That call
          needs the document's own content to ground its field choices, so — unlike the
          schema path — an instruction-based request always costs at least the
          inference call, even when the schema it resolves to turns out to already be
          cached.

        Both paths converge once a schema is resolved: a record-list model is built
        (``build_dynamic_record_list_model``), the model is asked to extract every
        matching record, and each field's quote is verified against stored content
        exactly like a single-record structure result — nothing about how provenance is
        established changes because there are many records instead of one. Results are
        stored in ``Insight.records``; ``Insight.fields`` is left empty, matching how it
        is already left empty for prose results.

        Returns:
            The insight **and** the schema that produced it — see :class:`StructureResult`
            for why the schema cannot simply be read back off the insight.
        """
        if (schema is None) == (instruction is None):
            raise ValidationError("Provide exactly one of 'schema' or 'instruction'.")

        input_tokens = 0
        output_tokens = 0

        if schema is not None:
            # Validated (and, on success, turned into the model class the provider
            # needs) before anything else — the same fail-fast ordering `extract()` uses
            # for `get_extraction_schema`, and cheaper than touching the database first.
            build_dynamic_record_list_model(schema)

            fingerprint = self._structure_fingerprint(schema)
            if not refresh:
                cached = await self._results.get(
                    document_id, owner_id=owner_id, fingerprint=fingerprint
                )
                if cached is not None:
                    return StructureResult(insight=_mark_cached(cached), schema=schema)

            resolved_schema = schema
            document = await self._ready_document(document_id, owner_id=owner_id)
            context_blocks, stats, blocks_by_id = await self._prepare_context(
                document_id, owner_id=owner_id, optimize=False
            )
            self._require_provider()
        else:
            # Guaranteed non-None by the exactly-one-of check above (`schema is None`
            # here, so the XOR means `instruction` cannot also be None) — spelled out as
            # an explicit raise, not a bare `assert` (stripped under `-O`, and flagged by
            # ruff's S101), so mypy can narrow it too, since it does not trace that
            # equality check itself.
            if instruction is None:  # pragma: no cover - unreachable, see above
                raise AssertionError("instruction must be set when schema is None")

            # The resolved schema — and therefore the cache fingerprint — cannot be
            # known before inference runs, and inference itself needs the document's
            # content. There is no way to check the cache here without first loading the
            # document and spending one AI call, unlike the schema-supplied path above.
            document = await self._ready_document(document_id, owner_id=owner_id)
            context_blocks, stats, blocks_by_id = await self._prepare_context(
                document_id, owner_id=owner_id, optimize=False
            )
            self._require_provider()

            inference = await self._provider.generate_structured(
                system=structure_schema_prompt.SYSTEM,
                instruction=structure_schema_prompt.build_instruction(instruction),
                context_blocks=context_blocks,
                document_title=document.original_filename,
                output_model=InferredStructureSchema,
            )
            if inference.parsed is None:
                raise InvalidModelResponseError(
                    "The AI could not turn this instruction into a field schema."
                )
            # Validated exactly like a caller-supplied schema — an inferred schema is
            # not given any special trust just because it came from the model rather
            # than from a request body.
            resolved_schema = schema_from_inference(inference.parsed)
            input_tokens += inference.input_tokens
            output_tokens += inference.output_tokens

            fingerprint = self._structure_fingerprint(resolved_schema)
            if not refresh:
                cached = await self._results.get(
                    document_id, owner_id=owner_id, fingerprint=fingerprint
                )
                if cached is not None:
                    return StructureResult(insight=_mark_cached(cached), schema=resolved_schema)

        output_model = build_dynamic_record_list_model(resolved_schema)
        result = await self._provider.generate_structured(
            system=structure_prompt.SYSTEM,
            instruction=structure_prompt.INSTRUCTION,
            context_blocks=context_blocks,
            document_title=document.original_filename,
            output_model=output_model,
        )
        input_tokens += result.input_tokens
        output_tokens += result.output_tokens

        records, citations, unverified, field_count = _records_from_parsed(
            result.parsed, resolved_schema, context_blocks, blocks_by_id
        )

        if not records:
            raise InsufficientContentError(
                "None of the requested fields could be found in this document."
            )

        warnings = list(describe_context(stats))
        if unverified:
            warnings.append(
                f"{unverified} of {field_count} fields across {len(records)} records "
                "could not be traced to a confirmed location in the document and are "
                "marked accordingly."
            )

        insight = Insight(
            document_id=document_id,
            kind=InsightKind.EXTRACTION,
            records=tuple(records),
            citations=tuple(citations),
            context_mode=stats.mode,
            model=self._provider.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            warnings=tuple(warnings),
        )
        saved = await self._results.save(insight, owner_id=owner_id, fingerprint=fingerprint)
        return StructureResult(insight=saved, schema=resolved_schema)

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
        schema_fields: dict[str, str] | None = None,
    ) -> str:
        """Identify one exact analysis, for caching.

        Covers everything that could change the answer. Editing a prompt or switching
        models therefore produces a different fingerprint, and old results stop being
        served automatically instead of going stale in silence.

        Args:
            schema_fields: A structure request's requested fields (Milestone 4.5), when
                the schema itself — not just a registry name — determines the answer.
                Canonicalised before hashing so field order never affects the fingerprint.
                ``None`` for every other call site: omitting it here keeps their
                fingerprints byte-for-byte identical to before this parameter existed.
        """
        return compute_fingerprint(
            kind,
            self._provider.model_name,
            prompt_version,
            optimize_context,
            schema_name=schema_name,
            schema_fields=schema_fields,
        )

    def _structure_fingerprint(self, schema: dict[str, str]) -> str:
        """The cache fingerprint for a resolved structure schema.

        Deliberately keyed on the *resolved* schema, never on the instruction text that
        may have produced it: two different instructions that happen to resolve to the
        same fields are the same cached analysis, and a schema supplied directly must
        produce the identical fingerprint an equivalent inferred one would — the cache
        does not know or care which input mode was used.
        """
        return self._fingerprint(
            InsightKind.EXTRACTION,
            structure_prompt.VERSION,
            optimize_context=False,
            schema_name="__structure__",
            schema_fields=schema,
        )


def _mark_cached(insight: Insight) -> Insight:
    """Flag a result as served from cache rather than freshly generated."""
    return Insight(
        document_id=insight.document_id,
        kind=insight.kind,
        content=insight.content,
        fields=insight.fields,
        records=insight.records,
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


def _structured_fields_from_parsed(
    parsed: object,
    schema: dict[str, str],
    context_blocks: list[ContextBlock],
    blocks_by_id: dict[UUID, ContentBlock],
) -> tuple[list[ExtractedField], list[Citation], int]:
    """Turn one dynamic structure result — a single record's worth — into fields with
    verified sources.

    Mirrors :func:`_fields_from_parsed`, but walks the *requested schema's* field names
    rather than a ``facts`` list: both ``build_dynamic_model``'s single-object model and
    each entry of ``build_dynamic_record_list_model``'s ``records`` list have one
    optional attribute per requested field, not a single list attribute, so this same
    function serves a single-record result directly (``parsed`` itself) and a
    multi-record result once per record (``parsed.records[i]``, via
    :func:`_records_from_parsed`). Each present field is still verified against stored
    content exactly the same way; nothing about how provenance is established changes
    because the schema came from the caller, or because there are many records instead
    of one.
    """
    fields: list[ExtractedField] = []
    citations: list[Citation] = []
    unverified = 0

    for name in schema:
        entry = getattr(parsed, name, None)
        if entry is None:
            continue

        # `entry.value` can be a bool/int/float, not only a string (unlike named
        # extraction's ExtractedFact.value, which is always text) — an `or ""` fallback
        # would treat a legitimate `False` or `0` as "missing", so `None` is checked for
        # explicitly before stringifying.
        raw_value = getattr(entry, "value", None)
        if raw_value is None:
            continue
        value = str(raw_value).strip()
        if not value:
            continue

        quote = (getattr(entry, "source_quote", "") or "").strip()
        citation = verify_quote(quote, context_blocks, blocks_by_id) if quote else None

        if citation is None:
            unverified += 1
        else:
            citations.append(citation)

        fields.append(ExtractedField(name=name, value=value, citation=citation))

    return fields, citations, unverified


def _records_from_parsed(
    parsed: object,
    schema: dict[str, str],
    context_blocks: list[ContextBlock],
    blocks_by_id: dict[UUID, ContentBlock],
) -> tuple[list[tuple[ExtractedField, ...]], list[Citation], int, int]:
    """Turn a multi-record structure result into records of fields with verified
    sources.

    ``build_dynamic_record_list_model`` wraps exactly the same per-field
    ``{value, source_quote}`` shape ``build_dynamic_model`` uses for a single row, one
    layer deeper — a ``records`` list of such objects instead of one. So each entry is
    walked and verified via :func:`_structured_fields_from_parsed`, unchanged: nothing
    about how a field's provenance is established differs because it belongs to record 3
    of 20 rather than being the only record. A record that ends up with no verifiable
    fields at all is dropped rather than kept as an empty row — the model was asked not
    to invent records, and an empty one is not a record, it is noise.

    Returns:
        ``(records, citations, unverified_field_count, total_field_count)`` — the last
        two are summed across every record, for the same "N of M could not be traced"
        warning a single-record result already gives.
    """
    raw_records = getattr(parsed, "records", None) or []

    records: list[tuple[ExtractedField, ...]] = []
    citations: list[Citation] = []
    unverified = 0
    field_count = 0

    for raw_record in raw_records:
        fields, record_citations, record_unverified = _structured_fields_from_parsed(
            raw_record, schema, context_blocks, blocks_by_id
        )
        if not fields:
            continue

        records.append(tuple(fields))
        citations.extend(record_citations)
        unverified += record_unverified
        field_count += len(fields)

    return records, citations, unverified, field_count
