"""Assembling a converted document from records, a plan, and validated corrections.

Pure, deterministic, and offline. Given the same records, plan and review, this module
produces the same :class:`ConvertedDocument` every time — no clock, no randomness, no
network. That is what makes the output reproducible and what lets the interesting
behaviour (grouping, filtering, ordering, how a correction reaches a value) be tested
without a model anywhere in sight.

The division of labour this file depends on: a model may eventually decide *what the
document should look like* (the plan) and *what looks wrong* (the proposals), but it
never builds the document. Code does, from data that has already been validated. So the
plan below is an ordinary typed structure — assembly does not care whether a human wrote
it or a model did, and it cannot be talked into doing something the types do not allow.

There is no template here for any kind of document. A section is whatever the plan
says it is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from app.ai.corrections import CorrectionReview
from app.models.domain.conversion import (
    ConvertedDocument,
    ConvertedField,
    ConvertedRecord,
    ConvertedSection,
    ConvertedValue,
    Correction,
    SectionKind,
    ValueOrigin,
)
from app.models.domain.insight import ExtractedField


class MatchMode(StrEnum):
    """How a filter compares a record's value against the values it is given."""

    #: Keep the record when its value is one of ``values``.
    EQUALS = "equals"
    #: Keep the record when its value is not one of ``values``.
    NOT_EQUALS = "not_equals"
    #: Keep the record when the field is missing or empty. ``values`` is ignored.
    ABSENT = "absent"
    #: Keep the record when the field is present and non-empty. ``values`` is ignored.
    PRESENT = "present"


@dataclass(frozen=True, slots=True)
class RecordFilter:
    """Which records belong in a section.

    Comparison is case-insensitive and whitespace-trimmed, because the whole point of
    this feature is that source data is messy — a filter that only matched exact casing
    would drop the very rows the user is trying to organize.

    Filtering reads the value the document will *show*: an applied correction is visible
    to the filter, so normalizing ``"gd"`` to ``"Good"`` and then grouping by
    condition behaves the way a reader would expect.
    """

    field: str
    mode: MatchMode = MatchMode.EQUALS
    values: tuple[str, ...] = ()

    def matches(self, record: ConvertedRecord) -> bool:
        item = record.field(self.field)
        text = "" if item is None else item.value.text.strip()
        has_value = bool(text) and (
            item is not None and item.value.origin is not ValueOrigin.ABSENT
        )

        if self.mode is MatchMode.ABSENT:
            return not has_value
        if self.mode is MatchMode.PRESENT:
            return has_value

        wanted = {value.strip().casefold() for value in self.values}
        if self.mode is MatchMode.EQUALS:
            return text.casefold() in wanted
        return text.casefold() not in wanted


@dataclass(frozen=True, slots=True)
class OrderKey:
    """How to sort records within a section.

    Attributes:
        numeric: Compare as numbers when both values parse as numbers. Without this,
            "145320" sorts before "45320" as text, which is not what anyone means by
            sorting a quantity. Values that do not parse fall back to text and sort after
            the numeric ones, so a stray "n/a" cannot break the ordering.
    """

    field: str
    descending: bool = False
    numeric: bool = False


@dataclass(frozen=True, slots=True)
class SectionPlan:
    """One section the document should contain.

    Attributes:
        group_by: When set, this plan expands into one section per distinct value of
            that field, with the value appended to the heading. Groups are emitted in a
            stable order (text order, case-insensitive), with records whose value is
            missing collected last.
        include_empty: Whether to emit a section that ends up with no records. Off by
            default, so an "Overdue" section simply does not appear when nothing is
            pending — an empty section reads like a system failure.
    """

    heading: str
    kind: SectionKind = SectionKind.TABLE
    columns: tuple[str, ...] = ()
    filters: tuple[RecordFilter, ...] = ()
    group_by: str | None = None
    order_by: tuple[OrderKey, ...] = ()
    note: str = ""
    include_empty: bool = False


@dataclass(frozen=True, slots=True)
class DocumentPlan:
    """The shape of the document the user asked for.

    Attributes:
        labels: Display names for schema fields, e.g. ``{"site_name": "Site
            Holder"}``. A field with no entry gets a humanized form of its key. The
            key itself is never rewritten — it is what the value was extracted under.
    """

    title: str
    sections: tuple[SectionPlan, ...] = ()
    labels: dict[str, str] = field(default_factory=dict)


def humanize_field_name(name: str) -> str:
    """``site_name`` -> ``Site Name``. Display only."""
    return " ".join(part.capitalize() for part in name.split("_") if part)


def assemble_document(
    records: Sequence[Sequence[ExtractedField]],
    *,
    schema: dict[str, str],
    plan: DocumentPlan,
    review: CorrectionReview,
    warnings: Sequence[str] = (),
) -> ConvertedDocument:
    """Build a converted document. Pure: nothing here reads a clock or a network.

    Args:
        records: Extracted records in extraction order; ``records[i]`` is record ``i``.
        schema: Field name to type. Supplies the default column set for a section that
            does not name its own.
        plan: What the document should look like.
        review: The result of :func:`app.ai.corrections.review_corrections`. Only
            ``review.applied`` changes any value; ``review.suggestions`` travel with the
            document so a reader can see them, leaving the originals in place.
        warnings: Anything the user should know — surfaced, never swallowed.

    Returns:
        The assembled document.

    Raises:
        ValueError: If an applied correction has no id. A changed value that cannot be
            traced to an audit entry is exactly the silent edit this feature exists to
            prevent, so it fails loudly rather than being rendered.
    """
    applied = review.applied_by_target
    for target, correction in applied.items():
        if not correction.id:
            raise ValueError(f"applied correction for {target} has no id")

    converted = tuple(
        _build_record(index, fields, schema=schema, plan=plan, applied=applied)
        for index, fields in enumerate(records)
    )

    sections: list[ConvertedSection] = []
    for section_plan in plan.sections:
        sections.extend(_build_sections(section_plan, converted, schema=schema))

    return ConvertedDocument(
        title=plan.title,
        sections=tuple(sections),
        corrections=review.applied,
        suggestions=review.suggestions,
        schema_used=dict(schema),
        source_record_count=len(records),
        warnings=tuple(warnings),
    )


def _build_record(
    index: int,
    fields: Sequence[ExtractedField],
    *,
    schema: dict[str, str],
    plan: DocumentPlan,
    applied: dict[tuple[int, str], Correction],
) -> ConvertedRecord:
    """Turn one extracted record into a converted one, applying its corrections."""
    by_name = {item.name: item for item in fields}

    converted_fields: list[ConvertedField] = []
    for name in schema:
        source = by_name.get(name)
        correction = applied.get((index, name))

        if source is None:
            # The document does not state this field for this record. Represented as
            # absent rather than blank-but-present, so a renderer can tell the
            # difference between "not stated" and "stated as empty".
            value = ConvertedValue(text="", origin=ValueOrigin.ABSENT)
        elif correction is not None:
            # The citation stays with the *original* text: it describes what the source
            # says, while the correction describes how the shown value differs from it.
            value = ConvertedValue(
                text=correction.corrected,
                origin=ValueOrigin.CORRECTED,
                original_text=source.value,
                citation=source.citation,
                correction_id=correction.id,
            )
        else:
            value = ConvertedValue(
                text=source.value,
                origin=ValueOrigin.SOURCE,
                citation=source.citation,
            )

        converted_fields.append(
            ConvertedField(
                name=name,
                label=plan.labels.get(name) or humanize_field_name(name),
                value=value,
            )
        )

    # merged_from stays empty: duplicate merging is deferred out of v1, and this module
    # never combines records.
    return ConvertedRecord(
        fields=tuple(converted_fields), source_record_index=index, merged_from=()
    )


def _build_sections(
    plan: SectionPlan,
    records: Sequence[ConvertedRecord],
    *,
    schema: dict[str, str],
) -> list[ConvertedSection]:
    """Expand one section plan into the sections it describes."""
    columns = plan.columns or tuple(schema)

    if plan.kind is SectionKind.NOTE:
        # A note carries prose the user asked for and holds no records, so filtering,
        # grouping and ordering do not apply to it.
        return [ConvertedSection(heading=plan.heading, kind=SectionKind.NOTE, note=plan.note)]

    kept = [record for record in records if all(f.matches(record) for f in plan.filters)]

    if plan.group_by is None:
        ordered = _order(kept, plan.order_by)
        if not ordered and not plan.include_empty:
            return []
        return [
            ConvertedSection(
                heading=plan.heading,
                kind=plan.kind,
                columns=columns,
                records=tuple(ordered),
                note=plan.note,
            )
        ]

    return _build_groups(plan, kept, columns=columns)


def _build_groups(
    plan: SectionPlan, records: Sequence[ConvertedRecord], *, columns: tuple[str, ...]
) -> list[ConvertedSection]:
    """One section per distinct value of ``plan.group_by``."""
    assert plan.group_by is not None  # noqa: S101 - guarded by the caller

    groups: dict[str, list[ConvertedRecord]] = {}
    for record in records:
        item = record.field(plan.group_by)
        key = "" if item is None else item.value.text.strip()
        groups.setdefault(key, []).append(record)

    # Deterministic group order: named groups by text (case-insensitive), then the
    # unnamed group last so "not stated" never leads the document.
    ordered_keys = sorted((key for key in groups if key), key=str.casefold)
    if "" in groups:
        ordered_keys.append("")

    sections: list[ConvertedSection] = []
    for key in ordered_keys:
        members = _order(groups[key], plan.order_by)
        if not members and not plan.include_empty:
            continue
        label = key if key else "Not stated"
        sections.append(
            ConvertedSection(
                heading=f"{plan.heading} — {label}",
                kind=plan.kind,
                columns=columns,
                records=tuple(members),
                note=plan.note,
            )
        )
    return sections


def _order(records: Sequence[ConvertedRecord], keys: Sequence[OrderKey]) -> list[ConvertedRecord]:
    """Sort records by the plan's keys.

    Applied last key first, relying on Python's stable sort so earlier keys win. Ties
    fall back to ``source_record_index``, so records that compare equal keep the order
    the document stated them in rather than an arbitrary one — which is what makes the
    output reproducible.
    """
    ordered = sorted(records, key=lambda record: record.source_record_index)
    for key in reversed(list(keys)):
        ordered.sort(key=lambda record: _sort_value(record, key), reverse=key.descending)
    return ordered


def _sort_value(record: ConvertedRecord, key: OrderKey) -> tuple[int, float, str]:
    """A sortable tuple for one record under one key.

    The leading rank keeps types from being compared against each other: numeric values
    first, then text, then missing. Without it a section containing both "1200" and
    "n/a" would raise rather than sort.
    """
    item = record.field(key.field)
    text = "" if item is None else item.value.text.strip()

    if not text:
        return (2, 0.0, "")

    if key.numeric:
        number = _as_number(text)
        if number is not None:
            return (0, number, "")

    return (1, 0.0, text.casefold())


def _as_number(text: str) -> float | None:
    """Parse a number written the way documents write them: ``1,200.50``, ``₹450``."""
    cleaned = "".join(ch for ch in text if ch.isdigit() or ch in ".-")
    if not cleaned or cleaned in {"-", ".", "-."}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None
