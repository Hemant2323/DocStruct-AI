"""Request and response schemas for document conversion.

A conversion result is a *document* — a tree of sections whose every value knows where it
came from and whether it was changed — plus the audit that justifies it. That is a
different thing from an insight (prose plus facts) and from a structure result (one flat
table), so it gets its own schemas rather than widening either of theirs. Overloading
``InsightResponse`` would force `/summary` and `/extraction` to carry a section tree and
a correction audit they can never populate.

Two things are deliberately **not** in the response. Chunk and block identifiers exist on
the citations behind these values, but they are database bookkeeping with no meaning to a
reader, so a value exposes only what a person can act on: the text, where it came from,
and whether that location was confirmed. And nothing here carries a prompt, a provider
name beyond the model identifier, or any part of a credential.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.domain.conversion import Confidence, CorrectionKind, SectionKind, ValueOrigin
from app.services.conversion_service import MAX_INSTRUCTION_LENGTH


class ConversionRequest(BaseModel):
    """What kind of document to build out of an existing one."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "instruction": (
                    "Build a maintenance report grouped by site. Standardize the "
                    "condition ratings, list overdue inspections separately, and do not "
                    "invent missing information."
                ),
                "refresh": False,
            }
        }
    )

    instruction: str = Field(
        min_length=1,
        max_length=MAX_INSTRUCTION_LENGTH,
        description=(
            "A plain-language description of the document you want built — how it should "
            "be organised, what to standardise, what to separate out. The fields "
            "themselves are worked out from the source document."
        ),
    )
    refresh: bool = Field(
        default=False,
        description=(
            "Re-read the source document instead of reusing its cached extraction. Each "
            "request produces a new conversion either way; this only controls whether the "
            "extraction behind it is reused."
        ),
    )

    @field_validator("instruction")
    @classmethod
    def _validate_instruction(cls, value: str) -> str:
        """Reject an instruction that is empty once whitespace is stripped.

        ``min_length=1`` alone would accept a string of spaces, which would then be sent
        to the model as an instruction saying nothing.
        """
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("instruction must not be empty or whitespace-only")
        return cleaned


class ConvertedValueResponse(BaseModel):
    """One value in the converted document, with its origin."""

    text: str = Field(description="What the document shows.")
    origin: ValueOrigin = Field(
        description=(
            "'source' when copied unchanged, 'corrected' when changed (with the original "
            "kept below), 'absent' when the document does not state this field for this "
            "record, 'structural' for a label the document itself supplies."
        )
    )
    original_text: str | None = Field(
        default=None, description="The text this replaced. Present only when corrected."
    )
    correction_id: str | None = Field(
        default=None,
        description="The audit entry justifying the change. Present only when corrected.",
    )
    source_label: str | None = Field(
        default=None,
        description=(
            "Where the original text appears in the source — 'page 4', 'rows 12-18'. "
            "Null when the location could not be confirmed; never guessed."
        ),
    )
    quoted_text: str | None = Field(
        default=None, description="The exact source wording this value came from, when confirmed."
    )
    is_verified: bool = Field(
        default=False,
        description="True when the source location was resolved from our own stored content.",
    )


class ConvertedFieldResponse(BaseModel):
    """One named field of a record."""

    name: str = Field(description="The field's key, e.g. 'site_name'. Never rewritten.")
    label: str = Field(description="How to show it, e.g. 'Site Name'. Display only.")
    value: ConvertedValueResponse


class ConvertedRecordResponse(BaseModel):
    """One entity in the converted document."""

    source_record_index: int = Field(
        description="Which extracted record this came from — the lineage back to extraction."
    )
    fields: list[ConvertedFieldResponse] = Field(default_factory=list)


class ConvertedSectionResponse(BaseModel):
    """One section of the converted document."""

    heading: str
    kind: SectionKind = Field(description="'table', 'detail' or 'note'.")
    columns: list[str] = Field(
        default_factory=list, description="Field names to show, in order. Empty for a note."
    )
    records: list[ConvertedRecordResponse] = Field(default_factory=list)
    note: str = Field(default="", description="Prose body, for a 'note' section.")


class ConversionCorrectionResponse(BaseModel):
    """One change that was applied, or one that was proposed and not applied."""

    id: str
    kind: CorrectionKind
    record_index: int
    field: str
    original: str
    corrected: str
    reason: str
    confidence: Confidence
    applied: bool = Field(
        description="True when the document shows the correction; false when it shows the original."
    )


class ConversionRejectionResponse(BaseModel):
    """A proposed change that was refused, and why.

    Surfaced rather than discarded: that the system was asked to alter a value and
    declined is something a reviewer of a financial document should be able to see.
    """

    kind: str
    record_index: int
    field: str
    original: str
    corrected: str
    confidence: str
    rejection_reason: str = Field(description="Why the change was refused.")


class ConversionPlanSectionResponse(BaseModel):
    """One section of the layout that produced the document, for transparency."""

    heading: str
    kind: SectionKind
    columns: list[str] = Field(default_factory=list)
    group_by: str | None = None
    note: str = ""


class ConversionPlanResponse(BaseModel):
    """The layout the document was built from, after invalid parts were stripped."""

    title: str
    sections: list[ConversionPlanSectionResponse] = Field(default_factory=list)
    labels: dict[str, str] = Field(default_factory=dict)


class ConversionSummaryResponse(BaseModel):
    """One past conversion, without its content — for listing."""

    id: UUID
    document_id: UUID
    title: str
    instruction: str
    source_record_count: int
    applied_correction_count: int
    suggestion_count: int
    rejected_correction_count: int
    created_at: datetime


class ConversionListResponse(BaseModel):
    """Every conversion produced for a document, newest first."""

    items: list[ConversionSummaryResponse] = Field(default_factory=list)


class ConversionResponse(BaseModel):
    """A completed conversion: the new document, and everything that justifies it."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "0b6c1f3e-6a4d-4c58-9a1f-2b7d8e5c4a90",
                "document_id": "1f4a7c22-9b3e-4d51-8c7a-5e2b9d0a6f31",
                "title": "Site Maintenance Report",
                "instruction": "Build a maintenance report grouped by site.",
                "markdown": "# Site Maintenance Report\n\n## North Ridge\n…",
                "source_record_count": 3,
                "warnings": [],
                "model": "claude-opus-5",
                "records_from_cache": False,
            }
        }
    )

    id: UUID = Field(description="This conversion. Use it to fetch the result again.")
    document_id: UUID = Field(description="The source document.")
    title: str
    instruction: str = Field(description="What was asked for, kept so the result explains itself.")
    markdown: str = Field(description="The rendered document, ready to display or download.")
    sections: list[ConvertedSectionResponse] = Field(
        default_factory=list, description="The same document in structured form."
    )
    plan: ConversionPlanResponse = Field(description="How it was organised, and why it looks so.")
    corrections: list[ConversionCorrectionResponse] = Field(
        default_factory=list, description="Changes that were applied."
    )
    suggestions: list[ConversionCorrectionResponse] = Field(
        default_factory=list,
        description=(
            "Changes that were proposed but not applied — the document still shows the "
            "original values."
        ),
    )
    rejected: list[ConversionRejectionResponse] = Field(
        default_factory=list, description="Changes that were refused as unsafe, with reasons."
    )
    schema_used: dict[str, str] = Field(
        default_factory=dict, description="The fields the source was read into."
    )
    vocabularies: dict[str, list[str]] = Field(
        default_factory=dict, description="Canonical value sets used to standardise fields."
    )
    source_record_count: int = Field(description="How many records the source yielded.")
    warnings: list[str] = Field(
        default_factory=list,
        description="Anything the user should know. Surfaced rather than hidden.",
    )
    model: str = Field(description="Which model planned and proposed.")
    input_tokens: int = 0
    output_tokens: int = 0
    records_from_cache: bool = Field(
        description=(
            "Whether the extraction behind this conversion was reused. The conversion "
            "itself is always produced fresh."
        )
    )
    created_at: datetime
