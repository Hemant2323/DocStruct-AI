"""Orchestration for converting a document into a new, organized document.

This is where the pieces built in the previous phases become one feature. It owns the
*order* of things and nothing else: extraction, planning, correction proposal, correction
validation, assembly and rendering all live elsewhere and are called, never reimplemented.

The order is the design, so it is worth stating plainly:

1. Extract the source records — via the existing :class:`AnalysisService`, not a second
   extractor.
2. Ask a model how the document should be organised.
3. Ask a model which values look inconsistently written — **against the original
   records**, never against records something has already altered.
4. Validate those proposals in Python. The model's confidence carries no authority here.
5. Assemble, using only the corrections that survived step 4.
6. Render.

Steps 2 and 3 are the only places a model is involved, and neither of them produces
document content: one returns a layout, the other returns a list of proposed edits. The
finished document is built entirely by steps 4-6, in ordinary Python. That is what makes
the output reproducible, and it is also what keeps it possible at all — asking a model to
emit a whole converted report would make the response grow with the document, which is
how an earlier milestone ran into the provider's output-token ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from app.ai.assembly import DocumentPlan, assemble_document
from app.ai.conversion_planner import (
    PlanResult,
    ProposalResult,
    generate_correction_proposals,
    generate_document_plan,
)
from app.ai.conversion_schema import vocabulary_arguments
from app.ai.corrections import CorrectionReview, review_corrections
from app.ai.providers.base import LLMProvider
from app.core.exceptions import (
    InsufficientContentError,
    InvalidModelResponseError,
    ValidationError,
)
from app.core.logging import get_logger
from app.models.domain.conversion import ConvertedDocument
from app.models.domain.insight import ExtractedField
from app.services.analysis_service import AnalysisService, StructureResult
from app.services.conversion_render import render_markdown

logger = get_logger(__name__)

#: Long enough to describe a document in detail, short enough to keep a request bounded.
#: Mirrors the ceilings on the ask and structure instructions.
MAX_INSTRUCTION_LENGTH = 2000


@dataclass(frozen=True, slots=True)
class ConversionResult:
    """Everything a caller needs to show a converted document and justify it.

    Attributes:
        document: The assembled document — the structured form, for a UI to render
            however it likes.
        markdown: The same document rendered. Both are returned because the API layer
            will want the structure and the preview will want the text, and rendering
            twice from the same tree is free.
        plan: The layout that produced it, after invalid parts were stripped.
        review: The full correction audit — what was applied, what is merely suggested,
            and what was rejected with the reason. Rejections are kept deliberately:
            "the model tried to change this and was refused" is information a reviewer
            of a financial document should be able to see.
        vocabularies: The canonical value sets the plan declared, if any.
        records_from_cache: Whether the *extraction* was served from cache. The
            conversion itself is not cached — there is no persistence for it yet — so
            this describes the source records only, and is named to say so.
    """

    document: ConvertedDocument
    markdown: str
    plan: DocumentPlan
    review: CorrectionReview
    schema_used: dict[str, str]
    vocabularies: dict[str, tuple[str, ...]]
    source_record_count: int
    warnings: tuple[str, ...]
    model: str
    input_tokens: int
    output_tokens: int
    records_from_cache: bool

    @property
    def applied_correction_count(self) -> int:
        return len(self.review.applied)

    @property
    def suggestion_count(self) -> int:
        return len(self.review.suggestions)

    @property
    def rejected_correction_count(self) -> int:
        return len(self.review.rejected)


class ConversionService:
    """Turns a document plus an instruction into a new, organized document."""

    def __init__(self, *, analysis: AnalysisService, provider: LLMProvider) -> None:
        self._analysis = analysis
        self._provider = provider

    async def convert(
        self,
        document_id: UUID,
        instruction: str,
        *,
        owner_id: str,
        refresh: bool = False,
    ) -> ConversionResult:
        """Build a new document from an existing one, following an instruction.

        Args:
            document_id: The source document.
            instruction: What the user wants built, in their own words. Also drives the
                extraction, since it is what says which fields matter.
            owner_id: Whose document this is. Every read below is owner-scoped.
            refresh: Re-extract instead of using a cached extraction. Planning and
                correction proposal always run fresh; only the extraction is cached, and
                only by the machinery that already existed.

        Returns:
            The converted document, its rendering, and the full correction audit.

        Raises:
            ValidationError: The instruction is empty.
            NotFoundError: No such document for this owner.
            InsufficientContentError: The document yielded no records to organize.
            InvalidModelResponseError: The model produced no usable plan.
            AIConfigurationError: The provider is not configured.
        """
        cleaned = instruction.strip()
        if not cleaned:
            raise ValidationError("An instruction is required to convert a document.")
        if len(cleaned) > MAX_INSTRUCTION_LENGTH:
            raise ValidationError(
                f"An instruction may be at most {MAX_INSTRUCTION_LENGTH} characters."
            )

        # 1. Source records, from the extraction pipeline that already exists. The same
        #    instruction drives it, since it is what says which fields matter.
        structure = await self._analysis.structure(
            document_id, owner_id=owner_id, instruction=cleaned, refresh=refresh
        )
        records: tuple[tuple[ExtractedField, ...], ...] = structure.insight.records
        schema = structure.schema

        if not records:
            # Defensive: `structure()` already refuses to cache an empty result, so this
            # is unreachable through it today. Stated anyway rather than letting an empty
            # document be planned and rendered as though it meant something.
            raise InsufficientContentError(
                "No information could be extracted from this document to organize."
            )

        # 2. Layout. Untrusted: `generate_document_plan` has already stripped anything
        #    naming a field that was never extracted.
        plan_result = await generate_document_plan(
            self._provider,
            instruction=cleaned,
            schema=schema,
            records=records,
        )

        if not plan_result.plan.sections:
            # A plan with nothing in it cannot be rendered into anything meaningful, and
            # inventing a fallback layout would be deciding for the user what their
            # document is. Reported instead.
            raise InvalidModelResponseError(
                "The AI could not work out how to organize this document. Try describing "
                "the document you want in more detail."
            )

        # 3. Proposals, generated against the ORIGINAL records. Nothing has altered them
        #    at this point and nothing will — every type involved is frozen.
        proposal_result = await generate_correction_proposals(
            self._provider,
            instruction=cleaned,
            records=records,
            vocabularies=vocabulary_arguments(plan_result.vocabularies),
        )

        # 4. Validation. This is the authority on what may change — not the model, and
        #    not this service, which passes the proposals straight through without
        #    inspecting or filtering them itself.
        review = review_corrections(
            proposal_result.proposals,
            records,
            vocabularies=vocabulary_arguments(plan_result.vocabularies),
        )

        warnings = self._collect_warnings(structure, plan_result, proposal_result, review)

        # 5. Assembly applies `review.applied` only; suggestions travel with the document
        #    so a reader can see them, leaving the original values in place.
        document = assemble_document(
            records,
            schema=schema,
            plan=plan_result.plan,
            review=review,
            warnings=warnings,
        )

        if not document.sections:
            # Distinct from an unusable plan: the plan was fine, the data simply matched
            # none of it. That is a legitimate answer ("no overdue items"), so it is
            # surfaced as a warning rather than raised.
            warnings = (*warnings, "No records matched any section of the planned document.")
            document = assemble_document(
                records,
                schema=schema,
                plan=plan_result.plan,
                review=review,
                warnings=warnings,
            )

        # 6. Rendering. The renderer owns how a document becomes text.
        markdown = render_markdown(document)

        logger.info(
            "Converted document_id=%s records=%d sections=%d applied=%d suggested=%d rejected=%d",
            document_id,
            len(records),
            len(document.sections),
            len(review.applied),
            len(review.suggestions),
            len(review.rejected),
        )

        return ConversionResult(
            document=document,
            markdown=markdown,
            plan=plan_result.plan,
            review=review,
            schema_used=dict(schema),
            vocabularies=plan_result.vocabularies,
            source_record_count=len(records),
            warnings=warnings,
            model=self._provider.model_name,
            input_tokens=(
                structure.insight.input_tokens
                + plan_result.input_tokens
                + proposal_result.input_tokens
            ),
            output_tokens=(
                structure.insight.output_tokens
                + plan_result.output_tokens
                + proposal_result.output_tokens
            ),
            records_from_cache=structure.insight.from_cache,
        )

    @staticmethod
    def _collect_warnings(
        structure: StructureResult,
        plan_result: PlanResult,
        proposal_result: ProposalResult,
        review: CorrectionReview,
    ) -> tuple[str, ...]:
        """Gather everything the user should know, from every stage.

        Rejected corrections are summarised rather than listed one by one: the full list
        with reasons is on ``review`` for anyone who wants it, and a warnings banner
        naming twenty refusals would bury the ones that matter.
        """
        warnings: list[str] = []
        warnings.extend(structure.insight.warnings)
        warnings.extend(plan_result.warnings)
        warnings.extend(proposal_result.warnings)

        if review.rejected:
            count = len(review.rejected)
            warnings.append(
                f"{count} proposed correction{'' if count == 1 else 's'} "
                f"{'was' if count == 1 else 'were'} rejected as unsafe and not applied."
            )

        # Preserve order while removing repeats — the same context warning can arrive
        # from more than one stage.
        return tuple(dict.fromkeys(warnings))
