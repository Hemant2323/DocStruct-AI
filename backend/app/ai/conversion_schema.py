"""Structured-output schemas for conversion planning, and their mapping to typed plans.

Two layers live here, deliberately separated.

The ``Planned*`` and ``Proposed*`` models are what a *model* returns. They are loose on
purpose: ``kind``, ``mode`` and ``confidence`` are plain strings, and field names are
plain strings too. This mirrors ``InferredStructureSchema`` — a value we cannot accept
should come back as a named rejection we can explain, not as a raw Pydantic error that
fails the whole request because one section named its layout "grid".

The ``to_*`` functions are the boundary. They turn that loose output into the strict
types ``app.ai.assembly`` and ``app.ai.corrections`` already define, and every value that
cannot be mapped is dropped with a warning rather than guessed at. Two rules decide what
survives:

**A field must exist.** Columns, groupings, filters, orderings and labels are checked
against the schema that was actually extracted. A planner that asks to group by
``credit_score`` when no such field was extracted gets that grouping removed — it cannot
conjure a column by naming one.

**A correction must be representable.** A proposal whose ``kind`` is unrecognised is
dropped, because each kind carries its own proof obligation and there is no safe default.
An unrecognised ``confidence`` is downgraded to ``LOW`` instead of dropped: ``LOW`` can
never be applied, so the proposal survives as something the user can look at while being
unable to change anything.

Nothing here decides whether a correction is *correct*. That remains
``app.ai.corrections.review_corrections``, the single authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.ai.assembly import DocumentPlan, MatchMode, OrderKey, RecordFilter, SectionPlan
from app.models.domain.conversion import (
    Confidence,
    CorrectionKind,
    ProposedCorrection,
    SectionKind,
)

#: A document with dozens of sections has not been designed, it has been fragmented.
#: Bounds the planning reply too, which is the other reason limits like this exist.
MAX_PLAN_SECTIONS = 20

#: Ceiling on corrections in one reply. Generous for a tidy-up pass, and low enough that
#: the response cannot grow into the output-token ceiling an earlier milestone hit.
MAX_CORRECTION_PROPOSALS = 200

#: Ceiling on canonical values for one field. A "closed" set of eighty values is not a
#: vocabulary, it is the data.
MAX_VOCABULARY_VALUES = 40

#: Fallback when a planner returns no usable title.
DEFAULT_TITLE = "Converted Document"


# --------------------------------------------------------------------- plan


class PlannedFilter(BaseModel):
    """A rule for which records belong in a section."""

    field: str = Field(description="Field name to test. Must be one of the available fields.")
    mode: str = Field(
        default=MatchMode.EQUALS.value,
        description=(
            "One of: 'equals' (keep records whose value is listed), 'not_equals' (keep "
            "records whose value is not listed), 'present' (keep records that state "
            "this field), 'absent' (keep records that do not)."
        ),
        max_length=32,
    )
    values: list[str] = Field(
        default_factory=list,
        description="Values to compare against. Ignored for 'present' and 'absent'.",
        max_length=MAX_VOCABULARY_VALUES,
    )


class PlannedOrder(BaseModel):
    """How to sort records within a section."""

    field: str = Field(description="Field name to sort by. Must be one of the available fields.")
    descending: bool = Field(default=False, description="Sort from highest/last to lowest/first.")
    numeric: bool = Field(
        default=False,
        description="Compare as numbers rather than text. Use for amounts and counts.",
    )


class PlannedSection(BaseModel):
    """One section of the planned document."""

    heading: str = Field(description="The section's title.", max_length=200)
    kind: str = Field(
        default=SectionKind.TABLE.value,
        description="One of: 'table', 'detail', 'note'.",
        max_length=32,
    )
    columns: list[str] = Field(
        default_factory=list,
        description=(
            "Field names to show, in order. Must all be available fields. Leave empty "
            "to show every field."
        ),
        max_length=64,
    )
    filters: list[PlannedFilter] = Field(
        default_factory=list,
        description="Which records belong here. Empty means all of them.",
        max_length=8,
    )
    group_by: str | None = Field(
        default=None,
        description=(
            "Split this section into one sub-section per distinct value of this field. "
            "Must be an available field."
        ),
    )
    order_by: list[PlannedOrder] = Field(
        default_factory=list, description="Sort order, most significant first.", max_length=4
    )
    note: str = Field(
        default="",
        description="Explanatory prose. This is the body of a 'note' section.",
        max_length=2_000,
    )


class PlannedLabel(BaseModel):
    """A display name for a field. The field's own name is never changed."""

    field: str = Field(description="Field name. Must be one of the available fields.")
    label: str = Field(description="How to show it, e.g. 'Site Name'.", max_length=120)


class PlannedVocabulary(BaseModel):
    """The canonical values a field's messy values should be normalised to."""

    field: str = Field(description="Field name. Must be one of the available fields.")
    values: list[str] = Field(
        description="The complete set of canonical values for this field.",
        max_length=MAX_VOCABULARY_VALUES,
    )


class PlannedDocument(BaseModel):
    """A model's proposal for how the converted document should be organised.

    Deliberately loose — see the module docstring. :func:`to_document_plan` converts this
    into the strict :class:`app.ai.assembly.DocumentPlan`, discarding anything that
    refers to a field which was never extracted.
    """

    title: str = Field(default="", description="The document's title.", max_length=200)
    sections: list[PlannedSection] = Field(
        default_factory=list,
        description="The sections, in the order they should appear.",
        max_length=MAX_PLAN_SECTIONS,
    )
    labels: list[PlannedLabel] = Field(
        default_factory=list, description="Display names for fields.", max_length=64
    )
    vocabularies: list[PlannedVocabulary] = Field(
        default_factory=list,
        description=(
            "Canonical value sets for fields whose values are written inconsistently. "
            "Only include a field when the data really does show one concept written "
            "several ways."
        ),
        max_length=32,
    )


@dataclass(frozen=True, slots=True)
class PlanMapping:
    """A planned document translated into types the assembler accepts.

    Attributes:
        vocabularies: Canonical values by field, ready to hand to
            ``review_corrections``. Carried separately from the plan because they
            constrain *corrections*, not layout.
        warnings: What was discarded and why. Surfaced rather than swallowed, so a
            planner that asked for a field nobody extracted leaves a trace.
    """

    plan: DocumentPlan
    vocabularies: dict[str, tuple[str, ...]]
    warnings: tuple[str, ...]


def to_document_plan(planned: PlannedDocument, *, schema: Mapping[str, str]) -> PlanMapping:
    """Convert a model's plan into a typed :class:`DocumentPlan`.

    Anything referring to a field that is not in ``schema`` is removed. A section whose
    ``kind`` is unrecognised is dropped whole: the three kinds render differently and
    there is no way to know which was meant, so guessing would be inventing layout.

    Args:
        planned: What the model returned.
        schema: The fields that were actually extracted — the closed set.

    Returns:
        The typed plan, any declared vocabularies, and warnings for what was dropped.
    """
    known = set(schema)
    warnings: list[str] = []

    sections: list[SectionPlan] = []
    for position, item in enumerate(planned.sections):
        section = _to_section(item, position, known=known, warnings=warnings)
        if section is not None:
            sections.append(section)

    labels: dict[str, str] = {}
    for label in planned.labels:
        if label.field not in known:
            warnings.append(f"Ignored a label for unknown field {label.field!r}.")
            continue
        labels[label.field] = label.label

    vocabularies: dict[str, tuple[str, ...]] = {}
    for vocabulary in planned.vocabularies:
        if vocabulary.field not in known:
            warnings.append(f"Ignored a vocabulary for unknown field {vocabulary.field!r}.")
            continue
        values = tuple(value for value in vocabulary.values if value.strip())
        if values:
            vocabularies[vocabulary.field] = values

    title = planned.title.strip() or DEFAULT_TITLE
    if not planned.title.strip():
        warnings.append("The plan had no title; a default was used.")

    return PlanMapping(
        plan=DocumentPlan(title=title, sections=tuple(sections), labels=labels),
        vocabularies=vocabularies,
        warnings=tuple(warnings),
    )


def _to_section(
    item: PlannedSection, position: int, *, known: set[str], warnings: list[str]
) -> SectionPlan | None:
    """Convert one planned section, or return ``None`` when it cannot be represented."""
    kind = _to_section_kind(item.kind)
    if kind is None:
        warnings.append(
            f"Dropped section {item.heading!r}: {item.kind!r} is not a known section kind."
        )
        return None

    heading = item.heading.strip() or f"Section {position + 1}"

    columns = tuple(name for name in item.columns if name in known)
    for name in item.columns:
        if name not in known:
            warnings.append(f"Removed unknown field {name!r} from section {heading!r}.")

    group_by = item.group_by
    if group_by is not None and group_by not in known:
        warnings.append(f"Removed grouping by unknown field {group_by!r} in section {heading!r}.")
        group_by = None

    filters: list[RecordFilter] = []
    for planned_filter in item.filters:
        record_filter = _to_filter(planned_filter, heading, known=known, warnings=warnings)
        if record_filter is not None:
            filters.append(record_filter)

    order_by: list[OrderKey] = []
    for planned_order in item.order_by:
        if planned_order.field not in known:
            warnings.append(
                f"Removed ordering by unknown field {planned_order.field!r} in section {heading!r}."
            )
            continue
        order_by.append(
            OrderKey(
                field=planned_order.field,
                descending=planned_order.descending,
                numeric=planned_order.numeric,
            )
        )

    return SectionPlan(
        heading=heading,
        kind=kind,
        columns=columns,
        filters=tuple(filters),
        group_by=group_by,
        order_by=tuple(order_by),
        note=item.note.strip(),
    )


def _to_filter(
    planned: PlannedFilter, heading: str, *, known: set[str], warnings: list[str]
) -> RecordFilter | None:
    if planned.field not in known:
        warnings.append(
            f"Removed a filter on unknown field {planned.field!r} in section {heading!r}."
        )
        return None

    mode = _to_match_mode(planned.mode)
    if mode is None:
        warnings.append(
            f"Removed a filter in section {heading!r}: {planned.mode!r} is not a known mode."
        )
        return None

    return RecordFilter(
        field=planned.field,
        mode=mode,
        values=tuple(planned.values),
    )


def _to_section_kind(value: str) -> SectionKind | None:
    try:
        return SectionKind(value.strip().lower())
    except ValueError:
        return None


def _to_match_mode(value: str) -> MatchMode | None:
    try:
        return MatchMode(value.strip().lower())
    except ValueError:
        return None


# --------------------------------------------------------------- corrections


class ProposedCorrectionModel(BaseModel):
    """One correction a model thinks should be made. Untrusted until validated."""

    record_index: int = Field(ge=0, description="Which record this applies to, counting from 0.")
    field: str = Field(description="The field within that record.", max_length=64)
    original: str = Field(
        description="The current value, quoted EXACTLY as it appears.", max_length=2_000
    )
    corrected: str = Field(description="The replacement value.", max_length=2_000)
    kind: str = Field(
        description=(
            "One of: 'case_normalization', 'whitespace', 'vocabulary', 'date_format', 'spelling'."
        ),
        max_length=32,
    )
    confidence: str = Field(
        default=Confidence.LOW.value,
        description="One of: 'high', 'medium', 'low'.",
        # Bounded, but wider than the three valid words: a model that answers "extremely
        # certain" should be downgraded to low by the mapping below, not blow up the
        # whole correction list with a Pydantic length error before it gets there.
        max_length=32,
    )
    reason: str = Field(description="Why this is a correction, in plain language.", max_length=300)


class ProposedCorrectionList(BaseModel):
    """Every correction a model proposes for one document.

    Carries corrections only — never the data. That is what keeps the reply small on a
    document whose values are already consistent, where the right answer is an empty
    list.
    """

    corrections: list[ProposedCorrectionModel] = Field(
        default_factory=list,
        description="The proposed corrections. Empty is a good answer for clean data.",
        max_length=MAX_CORRECTION_PROPOSALS,
    )


@dataclass(frozen=True, slots=True)
class ProposalMapping:
    """Model-proposed corrections translated into the type the validator accepts.

    These are still **proposals**. Nothing here has been checked for correctness — that
    is ``review_corrections``' job, and it rejects most of what a careless model would
    send regardless of what this mapping let through.
    """

    proposals: tuple[ProposedCorrection, ...]
    warnings: tuple[str, ...]


def to_proposed_corrections(payload: ProposedCorrectionList) -> ProposalMapping:
    """Convert model output into :class:`ProposedCorrection` values.

    Only what cannot be *represented* is dropped here — an unrecognised ``kind``, since
    each kind carries its own proof obligation and defaulting would mean choosing a proof
    the model never asked for. An unrecognised ``confidence`` is downgraded to ``LOW``
    rather than dropped: ``LOW`` is never applied, so the proposal survives for the user
    to see while being unable to change anything.

    Whether a proposal is *valid* is deliberately not decided here. Field names,
    anchoring, dates, similarity and vocabulary membership are all checked by
    ``review_corrections``, which stays the single authority.
    """
    proposals: list[ProposedCorrection] = []
    warnings: list[str] = []

    for item in payload.corrections:
        kind = _to_correction_kind(item.kind)
        if kind is None:
            warnings.append(
                f"Dropped a proposed correction for {item.field!r}: "
                f"{item.kind!r} is not a known correction kind."
            )
            continue

        confidence = _to_confidence(item.confidence)
        if confidence is None:
            warnings.append(
                f"A proposed correction for {item.field!r} had unrecognised confidence "
                f"{item.confidence!r}; treated as low."
            )
            confidence = Confidence.LOW

        proposals.append(
            ProposedCorrection(
                kind=kind,
                record_index=item.record_index,
                field=item.field,
                original=item.original,
                corrected=item.corrected,
                reason=item.reason,
                confidence=confidence,
            )
        )

    return ProposalMapping(proposals=tuple(proposals), warnings=tuple(warnings))


def _to_correction_kind(value: str) -> CorrectionKind | None:
    try:
        return CorrectionKind(value.strip().lower())
    except ValueError:
        return None


def _to_confidence(value: str) -> Confidence | None:
    try:
        return Confidence(value.strip().lower())
    except ValueError:
        return None


def vocabulary_arguments(
    vocabularies: Mapping[str, Sequence[str]],
) -> dict[str, Sequence[str]]:
    """Shape declared vocabularies the way ``review_corrections`` expects them.

    A one-line adapter, present so the call site reads as intent rather than as a dict
    comprehension, and so the two sides can diverge later without a hunt.
    """
    return {name: list(values) for name, values in vocabularies.items()}
