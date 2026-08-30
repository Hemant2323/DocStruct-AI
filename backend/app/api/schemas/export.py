"""Request schema for exporting a structure result (Milestone 4.6).

A new, small schema rather than reusing ``StructureRequest``: the two request bodies
share a field, but ``StructureRequest.refresh`` has no meaning for export — export never
regenerates anything, only reads a result that already exists. Reusing it as-is would
expose a caller-visible field that silently does nothing, which is worse than one extra
small file.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.ai.dynamic_schema import validate_structure_schema
from app.core.exceptions import InvalidStructureSchemaError


class StructureExportRequest(BaseModel):
    """The exact schema a prior `/structure` result was generated with."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "schema": {
                    "invoice_number": "string",
                    "vendor": "string",
                    "date": "date",
                    "total": "number",
                }
            }
        },
    )

    requested_schema: dict[str, str] = Field(
        alias="schema",
        description=(
            "The exact schema previously used with POST /structure. Field order does "
            "not matter, but the field names and types must match exactly, or the "
            "cached result for that schema will not be found."
        ),
    )

    @field_validator("requested_schema")
    @classmethod
    def _validate_schema(cls, value: dict[str, str]) -> dict[str, str]:
        # Reused, not reimplemented: the same check /structure itself runs, so an
        # export request can never claim validity for a schema /structure would reject.
        try:
            validate_structure_schema(value)
        except InvalidStructureSchemaError as exc:
            raise ValueError(exc.message) from exc
        return value
