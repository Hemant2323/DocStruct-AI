"""Domain models for document conversion — producing a new, organized document.

This is deliberately **not** the same thing as structured extraction. Extraction answers
"what values does this document contain?" and yields ``Insight.records``. Conversion
answers "what document does the user want built out of those values?" and yields a
:class:`ConvertedDocument`: a tree of sections, each holding records whose every value
knows where it came from and whether it was changed on the way.

Two rules shape every type here.

**Nothing is invented.** A value is either copied from the source, a *recorded* correction
of something copied from the source, a structural label the user asked for, or explicitly
absent. :class:`ValueOrigin` makes that a closed set, so there is no representation for
"text that appeared from nowhere" — a value carrying facts must carry an origin that
explains it.

**Nothing changes silently.** A value whose origin is ``CORRECTED`` keeps the text it
replaced (``original_text``), the citation of that original text, and the id of the audit
entry that justifies the change. The chain a reader can always walk is:

    displayed value -> correction -> original value -> citation -> provenance -> page/row

Citations are the existing :class:`app.models.domain.insight.Citation`. There is no second
citation system, and — exactly as in extraction — a location can only ever come from our
own records, never from something a model asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from app.models.domain.insight import Citation


class ValueOrigin(StrEnum):
    """Where a value in the converted document came from.

    Persisted and sent over the API, so these values are part of the contract.
    """

    #: Copied from the source document unchanged.
    SOURCE = "source"
    #: A correction of source text. ``original_text`` and ``correction_id`` are both set.
    CORRECTED = "corrected"
    #: A heading, label or note that organizes the document. Carries no factual claim
    #: about the source, which is why it needs no citation.
    STRUCTURAL = "structural"
    #: The source does not state this field for this record. Rendered as blank rather
    #: than guessed at.
    ABSENT = "absent"


class Confidence(StrEnum):
    """How sure a proposed correction is.

    The gate that matters: only :attr:`HIGH` is ever applied. Anything less is surfaced
    as a suggestion with the original value left standing, because a wrong "fix" to
    somebody's record is worse than an unfixed typo.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CorrectionKind(StrEnum):
    """What sort of change a correction makes.

    The kind is not decoration: each one has its own deterministic proof obligation in
    ``app.ai.corrections``. A correction that cannot prove itself under its own kind is
    rejected, whatever confidence was claimed for it.

    ``duplicate_merge`` is deliberately absent — merging records is the correction kind
    most able to destroy data, and it is deferred out of v1.
    """

    CASE_NORMALIZATION = "case_normalization"
    VOCABULARY = "vocabulary"
    DATE_FORMAT = "date_format"
    WHITESPACE = "whitespace"
    SPELLING = "spelling"


@dataclass(frozen=True, slots=True)
class ProposedCorrection:
    """A correction someone (eventually a model) suggests making.

    Input to validation, never trusted. It becomes a :class:`Correction` only after
    ``app.ai.corrections.review_corrections`` has proved it under its own kind's rules.

    Attributes:
        kind: Which proof obligation this correction must satisfy.
        record_index: Position in the extracted record list this applies to.
        field: The schema field name within that record.
        original: What the proposer claims the current value is. Checked against the
            actual extracted value — a mismatch means the proposer is describing data
            that does not exist, and the correction is discarded.
        corrected: The replacement text.
        reason: Plain-language justification, shown to the user in the audit.
        confidence: How sure the proposer is. Only ``HIGH`` can be applied.
    """

    kind: CorrectionKind
    record_index: int
    field: str
    original: str
    corrected: str
    reason: str
    confidence: Confidence


@dataclass(frozen=True, slots=True)
class Correction:
    """A correction that survived validation — either applied, or offered as a suggestion.

    Attributes:
        id: Stable identifier, derived from the proposal's position so that the same
            input always produces the same ids. Referenced by
            :attr:`ConvertedValue.correction_id`.
        applied: True when the document actually shows ``corrected``. False means the
            document still shows ``original`` and this is a suggestion.
    """

    id: str
    kind: CorrectionKind
    record_index: int
    field: str
    original: str
    corrected: str
    reason: str
    confidence: Confidence
    applied: bool


@dataclass(frozen=True, slots=True)
class RejectedCorrection:
    """A proposal that failed validation, kept with the reason it failed.

    Retained rather than dropped so a rejection is debuggable and testable: "the model
    tried to change something" is information, even when the answer was no.
    """

    proposal: ProposedCorrection
    reason: str


@dataclass(frozen=True, slots=True)
class ConvertedValue:
    """One value in the converted document, with its origin.

    Attributes:
        text: What the document shows.
        origin: Why this text is here — see :class:`ValueOrigin`.
        original_text: The text this replaced. Set only when ``origin`` is
            ``CORRECTED``; ``None`` otherwise.
        citation: Where the *original* text came from. A corrected value keeps the
            original's citation deliberately: the citation describes what the document
            says, and the correction describes how the displayed value differs from it.
            ``None`` when the quote could not be verified.
        correction_id: The audit entry justifying the change. Set only when ``origin``
            is ``CORRECTED``; a corrected value without one is invalid and assembly
            refuses to build it.
    """

    text: str
    origin: ValueOrigin
    original_text: str | None = None
    citation: Citation | None = None
    correction_id: str | None = None

    @property
    def is_verified(self) -> bool:
        """True when this value's source location was confirmed from stored content."""
        return self.citation is not None and self.citation.is_verified


@dataclass(frozen=True, slots=True)
class ConvertedField:
    """One named field of a record.

    Attributes:
        name: The schema key, e.g. ``"site_name"``. Never rewritten — it is what
            the value was extracted under.
        label: Human-readable form, e.g. ``"Site Name"``. Display only.
    """

    name: str
    label: str
    value: ConvertedValue


@dataclass(frozen=True, slots=True)
class ConvertedRecord:
    """One entity in the converted document — whatever kind of thing it describes.

    Attributes:
        source_record_index: Which extracted record this came from. This is the lineage
            link that lets any row in the output be traced back to the extraction that
            produced it.
        merged_from: Source indices folded into this record. Always empty in v1;
            duplicate merging is deferred, and the field exists so adding it later does
            not change this shape.
    """

    fields: tuple[ConvertedField, ...]
    source_record_index: int
    merged_from: tuple[int, ...] = ()

    def field(self, name: str) -> ConvertedField | None:
        """Return one field by name, or ``None`` when this record has no such field."""
        for item in self.fields:
            if item.name == name:
                return item
        return None


class SectionKind(StrEnum):
    """How a section presents its records."""

    #: A grid: one row per record, one column per field.
    TABLE = "table"
    #: One block per record, each field on its own line. Suits few records with many
    #: fields, where a table would be unreadably wide.
    DETAIL = "detail"
    #: Prose the user asked for. Holds no records.
    NOTE = "note"


@dataclass(frozen=True, slots=True)
class ConvertedSection:
    """One section of the converted document.

    Attributes:
        heading: The section title.
        columns: Field names to show, in order. Empty for ``NOTE``.
        records: The records in this section, already filtered and ordered by assembly.
        note: Prose body. Structural text the user asked for; used by ``NOTE``.
    """

    heading: str
    kind: SectionKind
    columns: tuple[str, ...] = ()
    records: tuple[ConvertedRecord, ...] = ()
    note: str = ""

    @property
    def is_empty(self) -> bool:
        """True when this section shows nothing — no records and no note."""
        return not self.records and not self.note.strip()


@dataclass(frozen=True, slots=True)
class ConvertedDocument:
    """A complete converted document, ready to render.

    Attributes:
        corrections: Changes actually applied. Every ``CORRECTED`` value in the tree
            references one of these by id.
        suggestions: Changes proposed but **not** applied, because confidence was below
            ``HIGH``. The document still shows the original values; these are shown to
            the user as possible corrections so nothing is decided behind their back.
        schema_used: The field set this document was built from.
        source_record_count: How many extracted records went in, before any filtering.
            A section showing fewer rows than this is doing so because the plan asked
            it to, not because records were lost.
    """

    title: str
    sections: tuple[ConvertedSection, ...] = ()
    corrections: tuple[Correction, ...] = ()
    suggestions: tuple[Correction, ...] = ()
    schema_used: dict[str, str] = field(default_factory=dict)
    source_record_count: int = 0
    warnings: tuple[str, ...] = ()

    @property
    def applied_correction_count(self) -> int:
        return len(self.corrections)

    def suggestions_for(self, record_index: int, field_name: str) -> tuple[Correction, ...]:
        """Suggestions attached to one field of one record.

        Used by renderers to mark a value the user may want to look at, without the
        value itself having been touched.
        """
        return tuple(
            item
            for item in self.suggestions
            if item.record_index == record_index and item.field == field_name
        )
