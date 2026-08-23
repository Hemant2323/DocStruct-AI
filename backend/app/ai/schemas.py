"""Schemas for structured information extraction.

Asking the model for a shape rather than prose means the reply can be **validated**
instead of parsed. A malformed answer fails loudly rather than becoming half-correct
data, which is the behaviour that matters when the output is somebody's contract terms.

Milestone 3 ships one general-purpose schema. The registry below is what makes adding a
domain-specific one later a matter of writing a model and adding a line — the same
extension pattern used for file-format parsers.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

#: Guards against a runaway reply. A document with hundreds of "key facts" has not been
#: understood, it has been transcribed.
MAX_EXTRACTED_FACTS = 40


class ExtractedFact(BaseModel):
    """One fact the model found, with the wording that supports it."""

    name: str = Field(
        description=(
            "Short snake_case identifier for what this is, e.g. 'total_amount', "
            "'due_date', 'counterparty'."
        ),
        max_length=64,
    )
    value: str = Field(
        description="The value exactly as written in the document, not reformatted.",
        max_length=2_000,
    )
    source_quote: str = Field(
        description=(
            "The exact sentence or phrase from the document that states this fact, "
            "copied verbatim. This is verified against the document; a quote that does "
            "not appear in the text causes the source to be discarded."
        ),
        max_length=2_000,
    )


class KeyFactsExtraction(BaseModel):
    """The general-purpose extraction result.

    Deliberately generic. A taxonomy of invoice fields, contract clauses and so on would
    be guessing at what users need before anyone has used the feature; this demonstrates
    the mechanism and gets replaced or joined by specific schemas once there is evidence
    of what to build.
    """

    facts: list[ExtractedFact] = Field(
        default_factory=list,
        description="The key facts stated in the document.",
        max_length=MAX_EXTRACTED_FACTS,
    )
    document_type: str | None = Field(
        default=None,
        description=(
            "What kind of document this appears to be, in a few words, based only on "
            "its content. Omit if genuinely unclear."
        ),
        max_length=120,
    )


#: Available extraction schemas, by name. Adding one is a new model plus a line here.
EXTRACTION_SCHEMAS: dict[str, type[BaseModel]] = {
    "key_facts": KeyFactsExtraction,
}

#: Used when a request does not name a schema.
DEFAULT_EXTRACTION_SCHEMA = "key_facts"


def get_extraction_schema(name: str) -> type[BaseModel]:
    """Return an extraction schema by name.

    Raises:
        ValidationError: If the name is unknown. Failing loudly beats silently
            substituting a different schema and returning the wrong shape.
    """
    from app.core.exceptions import ValidationError

    schema = EXTRACTION_SCHEMAS.get(name)
    if schema is None:
        raise ValidationError(
            f"Unknown extraction type {name!r}.",
            details={"available": sorted(EXTRACTION_SCHEMAS)},
        )
    return schema
