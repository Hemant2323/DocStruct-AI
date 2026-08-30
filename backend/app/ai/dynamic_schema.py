"""Runtime-built extraction schemas for user-defined document structuring (Milestone 4.5).

The user-provided schema is untrusted input. It is deliberately restricted to a flat
mapping of field name to a small, fixed set of type keywords — never arbitrary JSON
Schema, never nesting, never a `$ref` or `oneOf`. This is both a safety measure (a schema
field name ends up embedded in the prompt sent to the model, exactly like document
content, so it gets the same "untrusted input" treatment CLAUDE.md requires) and what
keeps this file simple: five known types, one regex, one call to
``pydantic.create_model()``.

Each requested field is represented internally the same way a named extraction's facts
already are — a ``{value, source_quote}`` pair — so the existing ``verify_quote()``
provenance mechanism applies unchanged. See ``app/ai/schemas.py``'s ``ExtractedFact`` for
the pattern this mirrors.

Two model builders share that per-field shape: :func:`build_dynamic_model` for a single
row, and :func:`build_dynamic_record_list_model` (Milestone 4, banking/payment documents)
for a list of rows — one per transaction or person a document contains, rather than only
the first match.

:class:`InferredStructureSchema` is a third, unrelated structured-output shape: not a
schema for *document* content, but the output of asking the model to turn a user's
plain-language instruction (see ``app/ai/prompts/structure_schema.py``) into a schema in
the first place. :func:`schema_from_inference` converts its result into the same flat
``{name: type}`` dict every function above accepts, so nothing downstream needs to know
whether a schema was typed in directly or inferred from an instruction.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, create_model

from app.core.exceptions import InvalidStructureSchemaError

#: Mirrors MAX_EXTRACTED_FACTS's reasoning: a schema with hundreds of fields has not been
#: designed, it has been dumped. Each field roughly doubles prompt size (value +
#: source_quote), so this also bounds how large the generated prompt can get.
MAX_STRUCTURE_FIELDS = 40

#: Upper bound on how many records a single structure result may contain (Milestone 4).
#: Generous enough for a real transaction log or payment report; a reply claiming
#: thousands of records has not been read carefully, it has been dumped — the same
#: reasoning as MAX_STRUCTURE_FIELDS, applied to rows instead of columns.
MAX_STRUCTURE_RECORDS = 200

#: Lowercase, starts with a letter, letters/digits/underscores only, max 64 characters.
#: This single pattern is what rejects uppercase names, special characters, and
#: overly long names all at once — there is nothing else those checks could be split into.
_FIELD_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

#: Pydantic v2 reserves this prefix on BaseModel (`model_config`, `model_fields`, ...). A
#: user-requested field with this prefix would collide with Pydantic's own attributes
#: when building the dynamic model below, not just look confusing.
_RESERVED_PREFIX = "model_"

#: The only field types a requested schema may use, mapped to the Python type
#: `create_model()` builds each field's `value` from. "date" is deliberately `str`, not
#: `datetime.date` — see the module-level note on why.
SUPPORTED_FIELD_TYPES: dict[str, type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    # A strict `datetime.date` type would make the *entire* request fail whenever the
    # model's date formatting is imperfect (InvalidModelResponseError), rather than
    # returning a slightly-loose value. Kept as a string until there is a concrete reason
    # to tighten it.
    "date": str,
}


def validate_structure_schema(schema: dict[str, str]) -> None:
    """Reject a requested schema that is empty, too large, or contains an unsafe field.

    Raises:
        InvalidStructureSchemaError: The schema fails any of the checks below. Every
            failure uses this one error type; ``details`` says which field and why.
    """
    if not schema:
        raise InvalidStructureSchemaError("The requested schema must have at least one field.")

    if len(schema) > MAX_STRUCTURE_FIELDS:
        raise InvalidStructureSchemaError(
            f"A schema may request at most {MAX_STRUCTURE_FIELDS} fields.",
            details={"field_count": len(schema), "max": MAX_STRUCTURE_FIELDS},
        )

    for name, field_type in schema.items():
        if not _FIELD_NAME_RE.fullmatch(name):
            raise InvalidStructureSchemaError(
                f"{name!r} is not a valid field name. Field names must be lowercase, "
                "start with a letter, and contain only letters, numbers and underscores "
                "(max 64 characters).",
                details={"field": name},
            )

        if name.startswith(_RESERVED_PREFIX):
            raise InvalidStructureSchemaError(
                f"{name!r} is a reserved field name and cannot be used.",
                details={"field": name},
            )

        if field_type not in SUPPORTED_FIELD_TYPES:
            raise InvalidStructureSchemaError(
                f"{field_type!r} is not a supported type for field {name!r}.",
                details={
                    "field": name,
                    "type": field_type,
                    "supported": sorted(SUPPORTED_FIELD_TYPES),
                },
            )


def _build_field_definitions(schema: dict[str, str]) -> dict[str, Any]:
    """Build one `{value, source_quote}` field definition per requested schema field.

    The reusable half of both model builders below — :func:`build_dynamic_model` wraps
    these directly into a single row, :func:`build_dynamic_record_list_model` wraps them
    into a record repeated once per list entry. Sharing this loop is what guarantees a
    field means exactly the same thing (same type, same description, same optionality)
    whether it appears once or once per record; two separately written copies would be a
    live bug waiting for the day one is edited and not the other.

    Does not validate the schema itself — every caller validates first, since the error
    should name the schema-level problem, not surface as a confusing model-construction
    failure.
    """
    field_definitions: dict[str, Any] = {}
    for name, field_type in schema.items():
        python_type = SUPPORTED_FIELD_TYPES[field_type]
        value_description = (
            "ISO-8601 date in the form YYYY-MM-DD, exactly as it should be reported."
            if field_type == "date"
            else f"The value for '{name}', as a {field_type}, exactly as written in the document."
        )

        field_model = create_model(
            f"_StructureField_{name}",
            value=(python_type, Field(description=value_description)),
            source_quote=(
                str,
                Field(
                    description=(
                        "The exact sentence or phrase from the document that states this "
                        "value, copied verbatim."
                    ),
                    max_length=2_000,
                ),
            ),
        )

        field_definitions[name] = (
            field_model | None,
            Field(
                default=None,
                description=(
                    f"The requested field '{name}'. Omitted (null) if the document does "
                    "not state it."
                ),
            ),
        )

    return field_definitions


def build_dynamic_model(schema: dict[str, str]) -> type[BaseModel]:
    """Build a one-off Pydantic model matching a requested structure schema.

    Validates the schema itself first — this function is safe to call directly, not only
    after a request has already passed through API-layer validation.

    Each requested field becomes an *optional* nested `{value, source_quote}` object,
    never a required scalar: a required field would pressure the model into inventing a
    value when the document genuinely does not contain one, which is exactly the harm
    CLAUDE.md's grounding rule exists to prevent. A field the document does not mention is
    correctly represented as absent, not guessed.

    Args:
        schema: Field name to type keyword (``string``, ``number``, ``integer``,
            ``boolean``, or ``date``).

    Returns:
        A ``BaseModel`` subclass suitable for ``LLMProvider.generate_structured()``.
    """
    validate_structure_schema(schema)
    return create_model("DynamicStructureResult", **_build_field_definitions(schema))


def build_dynamic_record_list_model(
    schema: dict[str, str], *, max_records: int = MAX_STRUCTURE_RECORDS
) -> type[BaseModel]:
    """Build a one-off Pydantic model for a *list* of records matching a requested schema.

    Where :func:`build_dynamic_model` produces a single flat row, this produces
    ``{"records": [...]}`` — the same per-field ``{value, source_quote}`` shape, repeated
    once per record. For a document that legitimately contains many matching entries (a
    payment report with twenty transactions, a list of account holders), the model should
    return one record per entry rather than only the first.

    Each record's fields are optional for the same reason as ``build_dynamic_model``'s: a
    given record legitimately might not state every requested field, and an absent field
    must be represented as absent rather than guessed at.

    Validates the schema itself first, exactly like ``build_dynamic_model`` — safe to call
    directly, not only after request-layer validation has already run.

    Args:
        schema: Field name to type keyword (``string``, ``number``, ``integer``,
            ``boolean``, or ``date``).
        max_records: Upper bound on how many records the model will accept, so a reply
            cannot claim an unbounded number of records. Defaults to
            ``MAX_STRUCTURE_RECORDS``. Must be at least 1.

    Returns:
        A ``BaseModel`` subclass, with a single ``records`` field, suitable for
        ``LLMProvider.generate_structured()``.
    """
    validate_structure_schema(schema)
    if max_records < 1:
        raise ValueError("max_records must be at least 1")

    record_model = create_model("DynamicStructureRecord", **_build_field_definitions(schema))

    return create_model(
        "DynamicStructureRecordList",
        records=(
            list[record_model],
            Field(
                default_factory=list,
                description=(
                    "One entry per matching record found in the document — e.g. one per "
                    "transaction or person. Include every record that matches; do not "
                    "stop after the first."
                ),
                max_length=max_records,
            ),
        ),
    )


class InferredStructureField(BaseModel):
    """One field the model proposed while inferring a schema from a user's instruction."""

    name: str = Field(
        description=(
            "A concise, lowercase snake_case field name — letters, digits and "
            "underscores only, starting with a letter (max 64 characters)."
        ),
        max_length=64,
    )
    type: str = Field(
        description=f"One of: {', '.join(sorted(SUPPORTED_FIELD_TYPES))}.",
        max_length=32,
    )


class InferredStructureSchema(BaseModel):
    """The result of inferring a schema from a user's natural-language instruction.

    Deliberately loose at the Pydantic layer: ``type`` is a bare ``str``, not a
    ``Literal`` restricted to ``SUPPORTED_FIELD_TYPES``. An unsupported type name is
    instead reported through :func:`schema_from_inference`, which converts this result
    into the same ``{name: type}`` shape every other structure function accepts and
    validates it with the one function every schema — typed or inferred — is checked by:
    ``validate_structure_schema``. That gives a bad type name the same
    ``InvalidStructureSchemaError`` (naming the field, not just "invalid response") that
    a caller-supplied schema would get, rather than a raw Pydantic error from this class.
    """

    fields: list[InferredStructureField] = Field(
        default_factory=list,
        description="The proposed fields, in the order the user described them.",
        max_length=MAX_STRUCTURE_FIELDS,
    )


def schema_from_inference(inferred: InferredStructureSchema) -> dict[str, str]:
    """Turn a schema-inference result into the flat ``{name: type}`` shape every other
    structure function accepts.

    Validates the result exactly the same way a caller-supplied schema is validated — an
    inferred schema is not given any special trust just because it came from the model
    rather than from a request body. Raises ``InvalidStructureSchemaError`` (via
    :func:`validate_structure_schema`) for an empty field list, an unsupported type, an
    invalid field name, or too many fields.
    """
    schema = {field.name: field.type for field in inferred.fields}
    validate_structure_schema(schema)
    return schema


def canonicalize_schema(schema: dict[str, str]) -> str:
    """Canonical string form of a requested schema, for fingerprinting.

    Field order must never affect the result: two requests naming the same fields in a
    different order are the same schema and must produce the same cache fingerprint.
    """
    return json.dumps(dict(sorted(schema.items())), separators=(",", ":"))
