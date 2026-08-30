"""Request and response schemas for document structuring (Milestone 4).

A structure result is a *table*: one record per matching entity a document describes (one
per transaction in a payment report), each record holding the requested fields. That is a
different shape from every other AI result — summaries and explanations are prose,
`/extraction` is a single flat set of facts — so it gets its own response model here
rather than widening the shared ``InsightResponse``. Widening that one would force
`/summary`, `/explanation` and `/extraction` to carry a ``records`` field they can never
populate, which is exactly the kind of "one model, several meanings" contract this
project avoids.

The per-field and per-citation shapes are *not* redefined: ``ExtractedFieldResponse`` and
``CitationResponse`` are imported from ``analysis.py`` and reused, so a field inside a
record serialises identically to a field anywhere else in the API.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.ai.dynamic_schema import validate_structure_schema
from app.api.schemas.analysis import CitationResponse, ExtractedFieldResponse
from app.core.exceptions import InvalidStructureSchemaError
from app.models.domain.insight import ContextMode, Insight, InsightKind
from app.services.analysis_service import StructureResult

#: Long enough for a real description of a wanted table, short enough to keep a request
#: bounded — mirrors ``AskRequest.MAX_QUESTION_LENGTH``'s reasoning.
MAX_INSTRUCTION_LENGTH = 2000


class StructureRequest(BaseModel):
    """What to extract from a document, given either as an explicit schema or as a
    plain-language description of the table wanted."""

    # `populate_by_name` lets tests/internal code construct this via `requested_schema=`
    # too, not only the wire alias — the JSON body key is still "schema" either way.
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "schema": {
                    "invoice_number": "string",
                    "vendor": "string",
                    "date": "date",
                    "total": "number",
                },
                "refresh": False,
            }
        },
    )

    # Named `requested_schema`, not `schema`: Pydantic's BaseModel reserves `schema` as a
    # (deprecated) attribute name and warns on every import if it is shadowed. The public
    # API contract is unaffected — `alias="schema"` keeps the JSON body key exactly as
    # specified (`{"schema": {...}}`).
    requested_schema: dict[str, str] | None = Field(
        default=None,
        alias="schema",
        description=(
            "The fields to extract, as a flat mapping of field name to type. Supported "
            "types: 'string', 'number', 'integer', 'boolean', 'date'. Field names must be "
            "lowercase, start with a letter, and contain only letters, numbers and "
            "underscores (max 64 characters, max 40 fields). Provide this or "
            "'instruction', not both."
        ),
    )
    instruction: str | None = Field(
        default=None,
        max_length=MAX_INSTRUCTION_LENGTH,
        description=(
            "A plain-language description of the structured document wanted — for "
            "example 'a payment report with one row per transaction, including account "
            "holder, bank, amount and status'. The fields are worked out from this "
            "description and returned as 'schema_used'. Provide this or 'schema', not both."
        ),
    )
    refresh: bool = Field(
        default=False,
        description=(
            "Regenerate instead of returning the cached result for this exact schema. "
            "Costs another AI call, so leave it off unless you want a fresh answer."
        ),
    )

    @field_validator("requested_schema")
    @classmethod
    def _validate_schema(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        # Reused, not reimplemented: the same check the service itself runs, so a request
        # can never reach the service with a schema the service would reject — it fails
        # here, at the request boundary, with a plain 422 instead.
        if value is None:
            return None
        try:
            validate_structure_schema(value)
        except InvalidStructureSchemaError as exc:
            raise ValueError(exc.message) from exc
        return value

    @field_validator("instruction")
    @classmethod
    def _validate_instruction(cls, value: str | None) -> str | None:
        """Reject an instruction that is empty once whitespace is stripped.

        ``max_length`` alone would happily accept a string of only spaces, which would
        then be sent to the model as an instruction saying nothing.
        """
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("instruction must not be empty or whitespace-only")
        return cleaned

    @model_validator(mode="after")
    def _exactly_one_input(self) -> StructureRequest:
        """Require exactly one of ``schema`` and ``instruction``.

        Neither is unanswerable. Both is ambiguous, and silently preferring one would
        mean a caller who supplied a schema *and* a description could get a result that
        ignores half of what they asked for without being told.
        """
        if self.requested_schema is None and self.instruction is None:
            raise ValueError("Provide either 'schema' or 'instruction'.")
        if self.requested_schema is not None and self.instruction is not None:
            raise ValueError("Provide only one of 'schema' or 'instruction', not both.")
        return self


class StructureResponse(BaseModel):
    """A structured, multi-record result: one record per matching entity in a document."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "kind": "extraction",
                "document_id": "0b6c1f3e-6a4d-4c58-9a1f-2b7d8e5c4a90",
                "records": [
                    [
                        {
                            "name": "account_holder",
                            "value": "Arjun Malhotra",
                            "citation": {
                                "quoted_text": "Account holder Arjun Malhotra",
                                "location_label": "page 1",
                                "is_verified": True,
                                "block_ids": [],
                            },
                        },
                        {"name": "amount", "value": "2100", "citation": None},
                    ],
                    [
                        {"name": "account_holder", "value": "Person 2", "citation": None},
                        {"name": "amount", "value": "4500", "citation": None},
                    ],
                ],
                "schema_used": {"account_holder": "string", "amount": "number"},
                "record_count": 2,
                "context_mode": "full",
                "model": "claude-opus-5",
                "warnings": [],
                "from_cache": False,
            }
        }
    )

    kind: InsightKind
    document_id: UUID = Field(description="The document this was extracted from.")
    records: list[list[ExtractedFieldResponse]] = Field(
        default_factory=list,
        description=(
            "One entry per matching record found in the document — a payment report with "
            "twenty transactions returns twenty records. Each record holds the requested "
            "fields that are actually stated for that entity; a field the document does "
            "not state for a given record is left out of that record rather than guessed at."
        ),
    )
    schema_used: dict[str, str] = Field(
        description=(
            "The exact schema this result was produced with — as supplied when 'schema' "
            "was given, or as worked out from the description when 'instruction' was. "
            "Pass this back to POST /structure/export to export this result."
        )
    )
    record_count: int = Field(description="How many records were found.")
    citations: list[CitationResponse] = Field(
        default_factory=list,
        description="Every confirmed source across all records.",
    )
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
    def from_domain(cls, result: StructureResult) -> StructureResponse:
        insight: Insight = result.insight
        return cls(
            kind=insight.kind,
            document_id=insight.document_id,
            records=[
                [
                    ExtractedFieldResponse(
                        name=item.name,
                        value=item.value,
                        citation=(
                            CitationResponse.from_domain(item.citation) if item.citation else None
                        ),
                    )
                    for item in record
                ]
                for record in insight.records
            ],
            schema_used=result.schema,
            record_count=len(insight.records),
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
