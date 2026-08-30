"""The two AI calls document conversion makes.

Both are deliberately thin: prepare the input, call the provider's existing
``generate_structured``, map the reply onto the types the deterministic core already
defines, return. No assembly happens here, no correction is applied, and no decision is
made about whether a correction is safe — those belong to ``app.ai.assembly`` and
``app.ai.corrections`` respectively, which is what lets the whole Phase 1-4 pipeline stay
usable and testable with no model involved at all.

Two calls rather than one, on purpose. Asking for a layout and a list of corrections in a
single reply would make the response grow with the size of the document, which is exactly
how an earlier milestone hit the provider's output-token ceiling. A plan is small however
big the data is, and a correction list is small whenever the data is tidy.

The provider is passed in rather than constructed. Every provider implements the same
``generate_structured``, so nothing here is specific to one of them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.ai.assembly import DocumentPlan
from app.ai.conversion_schema import (
    PlannedDocument,
    ProposedCorrectionList,
    to_document_plan,
    to_proposed_corrections,
)
from app.ai.prompts import conversion_corrections as corrections_prompt
from app.ai.prompts import conversion_plan as plan_prompt
from app.ai.providers.base import LLMProvider
from app.core.exceptions import AIConfigurationError, InvalidModelResponseError
from app.models.domain.conversion import ProposedCorrection
from app.models.domain.insight import ContextBlock, ExtractedField


@dataclass(frozen=True, slots=True)
class PlanResult:
    """A generated document plan, ready for assembly.

    Attributes:
        vocabularies: Canonical value sets the planner declared, to be handed to
            ``review_corrections``. Empty when it declared none.
        warnings: What was discarded while mapping the reply — a field the planner
            named that nobody extracted, an unknown section kind. Surfaced so the
            caller can show them rather than silently losing part of the plan.
    """

    plan: DocumentPlan
    vocabularies: dict[str, tuple[str, ...]]
    warnings: tuple[str, ...]
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class ProposalResult:
    """Corrections a model proposed. **Not** validated, and not safe to apply.

    Pass ``proposals`` to ``app.ai.corrections.review_corrections`` before anything is
    allowed to change a value.
    """

    proposals: tuple[ProposedCorrection, ...]
    warnings: tuple[str, ...]
    input_tokens: int = 0
    output_tokens: int = 0


async def generate_document_plan(
    provider: LLMProvider,
    *,
    instruction: str,
    schema: Mapping[str, str],
    records: Sequence[Sequence[ExtractedField]],
    context_blocks: Sequence[ContextBlock] = (),
    document_title: str = "",
) -> PlanResult:
    """Ask the model how the converted document should be organised.

    The model sees the field names, a short sample of the data, and what the user asked
    for. It returns a layout — never content. Anything in that layout naming a field
    which was not extracted is removed while mapping, so a plan cannot introduce a column
    the document has no values for.

    Args:
        provider: Any configured provider; the call is not specific to one.
        instruction: What the user asked for.
        schema: Field name to type — the closed set the plan may reference.
        records: Extracted records. Only a small sample reaches the prompt.
        context_blocks: Document context to send alongside. Callers should pass little
            or nothing: planning needs the shape of the data, not its full text, and the
            sample in the instruction already supplies that.
        document_title: Shown to the model as the source document's name.

    Returns:
        The typed plan, any declared vocabularies, and mapping warnings.

    Raises:
        AIConfigurationError: The provider is not configured.
        InvalidModelResponseError: The reply was missing or unusable.
    """
    _require_provider(provider)

    result = await provider.generate_structured(
        system=plan_prompt.SYSTEM,
        instruction=plan_prompt.build_instruction(instruction, schema, records),
        context_blocks=list(context_blocks),
        document_title=document_title,
        output_model=PlannedDocument,
    )

    if not isinstance(result.parsed, PlannedDocument):
        raise InvalidModelResponseError("The AI did not return a usable plan for this document.")

    mapping = to_document_plan(result.parsed, schema=schema)
    return PlanResult(
        plan=mapping.plan,
        vocabularies=mapping.vocabularies,
        warnings=mapping.warnings,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


async def generate_correction_proposals(
    provider: LLMProvider,
    *,
    instruction: str,
    records: Sequence[Sequence[ExtractedField]],
    vocabularies: Mapping[str, Sequence[str]] | None = None,
    context_blocks: Sequence[ContextBlock] = (),
    document_title: str = "",
) -> ProposalResult:
    """Ask the model which values look inconsistently written.

    Everything it returns is a **proposal**. This function does not check whether a
    correction is right, does not apply anything, and gives the model's own confidence
    no authority — ``review_corrections`` decides all of that, in Python, afterwards.

    Args:
        provider: Any configured provider.
        instruction: What the user asked for. Shapes what is worth tidying.
        records: The extracted records, shown to the model with their indices.
        vocabularies: Canonical values per field, normally from :func:`generate_document_plan`.
            Without them the prompt tells the model not to propose vocabulary fixes,
            since there would be nothing to check such a fix against.
        context_blocks: Document context to send alongside. Usually unnecessary: the
            records are the thing being corrected.
        document_title: Shown to the model as the source document's name.

    Returns:
        The proposals and any mapping warnings. **Validate before use.**

    Raises:
        AIConfigurationError: The provider is not configured.
        InvalidModelResponseError: The reply was missing or unusable.
    """
    _require_provider(provider)

    result = await provider.generate_structured(
        system=corrections_prompt.SYSTEM,
        instruction=corrections_prompt.build_instruction(instruction, records, vocabularies),
        context_blocks=list(context_blocks),
        document_title=document_title,
        output_model=ProposedCorrectionList,
    )

    if not isinstance(result.parsed, ProposedCorrectionList):
        raise InvalidModelResponseError("The AI did not return a usable list of corrections.")

    mapping = to_proposed_corrections(result.parsed)
    return ProposalResult(
        proposals=mapping.proposals,
        warnings=mapping.warnings,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


def _require_provider(provider: LLMProvider) -> None:
    """Fail with the configuration error the rest of the app uses, not a call failure."""
    if not provider.is_configured:
        raise AIConfigurationError(
            "AI features are not configured on this server. An API key is required."
        )
