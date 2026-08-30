"""Export a previously-generated structure result as a downloadable file (Milestone 4.6).

Strictly read-only over data :meth:`AnalysisService.structure` already produced and
cached: this service never calls a provider and never calls ``AnalysisService.structure``
itself (which would fall through to a real AI call on a cache miss — exactly what export
must never risk). It recomputes the same cache fingerprint and does one direct
``AIResultRepository.get()`` lookup; a miss is reported as "not found," never silently
generated.

**Two shapes, one endpoint.** Since Milestone 4 a structure result is a *table* —
``Insight.records``, one record per matching entity — and that is what the formatters
below export: one row per record, one column per requested field. A result carrying only
``Insight.fields`` is a single-record result cached before that change (or written
directly, as one test does); those are still exported in the original field-per-row
layout rather than being dropped or silently reshaped. Which layout applies is decided
once, in :func:`_uses_records`, not re-derived by each formatter.

The requested schema — not the union of names actually present — supplies the column set
and its order, so every record lines up under the same headers and a field missing from
one record leaves an empty cell instead of shifting its neighbours left.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from openpyxl import Workbook

from app.ai.dynamic_schema import validate_structure_schema
from app.ai.prompts import structure as structure_prompt
from app.ai.providers.base import LLMProvider
from app.core.exceptions import NotFoundError
from app.core.security import sanitize_filename
from app.models.domain.insight import Citation, ExtractedField, Insight, InsightKind
from app.repositories.ai_result_repo import AIResultRepository
from app.repositories.document_repo import DocumentRepository
from app.services.analysis_service import compute_fingerprint

ExportFormat = Literal["json", "csv", "xlsx"]

_MEDIA_TYPES: dict[ExportFormat, str] = {
    "json": "application/json",
    "csv": "text/csv",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

#: A cell beginning with any of these is executed as a formula by Excel/Sheets/LibreOffice
#: when the exported file is opened (CLAUDE.md §8). Values come from a user's uploaded
#: document, so they are untrusted exactly like every other piece of document content.
_FORMULA_TRIGGER_CHARS = ("=", "+", "-", "@")

_UNVERIFIED_LOCATION_LABEL = "source not verified"

#: CSV/XLSX column order for a **legacy single-record** result. Kept identical between the
#: two formats deliberately — the same logical table, just two different container formats.
_TABLE_COLUMNS = ("field_name", "value", "location_label", "is_verified", "quoted_text")

#: Key holding a record's per-field sources in the JSON export. Cannot collide with a real
#: field name: ``validate_structure_schema`` requires names to match
#: ``^[a-z][a-z0-9_]{0,63}$``, which no leading-underscore name satisfies. That guarantee
#: is what makes it safe to put values and their sources in one object rather than in two
#: parallel arrays a consumer would have to keep in step by index.
_JSON_SOURCES_KEY = "_sources"


@dataclass(frozen=True, slots=True)
class ExportedFile:
    """A finished export, ready to hand to an HTTP response.

    Attributes:
        content: The file's bytes, already fully built in memory — small enough (one
            document's structured fields) that streaming would add complexity for no
            benefit.
        media_type: The MIME type for the ``Content-Type`` header.
        filename: Already sanitised — safe to place directly in a ``Content-Disposition``
            header.
    """

    content: bytes
    media_type: str
    filename: str


class ExportService:
    """Turns an already-persisted structure result into a downloadable file."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        documents: DocumentRepository,
        results: AIResultRepository,
    ) -> None:
        self._provider = provider
        self._documents = documents
        self._results = results

    async def export_structure(
        self,
        document_id: UUID,
        *,
        owner_id: str,
        schema: dict[str, str],
        export_format: ExportFormat,
    ) -> ExportedFile:
        """Export the cached `/structure` result for one document and exact schema.

        Raises:
            NotFoundError: The document does not exist (or is not visible to this
                owner), or no cached result exists for this exact schema. The two are
                reported with different messages but the same status, matching how
                ``AnalysisService`` reports "not found" elsewhere.
        """
        # Validated the same way `/structure` validates it — a malformed schema can
        # never accidentally compute a fingerprint that happens to collide with a real
        # one, and this makes the check self-contained for direct/unit-level callers.
        validate_structure_schema(schema)

        document = await self._documents.get(document_id, owner_id=owner_id)
        if document is None:
            raise NotFoundError("Document not found", details={"document_id": str(document_id)})

        fingerprint = compute_fingerprint(
            InsightKind.EXTRACTION,
            self._provider.model_name,
            structure_prompt.VERSION,
            optimize_context=False,
            schema_name="__structure__",
            schema_fields=schema,
        )

        # A direct cache lookup, never `AnalysisService.structure()` — that method calls
        # the provider on a cache miss, which export must never do.
        insight = await self._results.get(document_id, owner_id=owner_id, fingerprint=fingerprint)
        if insight is None:
            raise NotFoundError(
                "No structured result exists yet for this document and schema. Run "
                "POST /structure with this exact schema first, then export it.",
                details={"document_id": str(document_id)},
            )

        content = _FORMATTERS[export_format](insight, schema)
        filename = sanitize_filename(f"{document.original_filename}-structure.{export_format}")
        return ExportedFile(
            content=content, media_type=_MEDIA_TYPES[export_format], filename=filename
        )


def _escape_cell(value: str) -> str:
    """Neutralise a value a spreadsheet program would otherwise read as a formula.

    Prefixing with a single quote is the standard mitigation: Excel, Sheets and
    LibreOffice all display the value with the leading quote stripped and never evaluate
    it as a formula, so the visible data is unchanged — only its interpretation is.
    """
    if value and value[0] in _FORMULA_TRIGGER_CHARS:
        return "'" + value
    return value


def _citation_row(citation: Citation | None) -> tuple[str, bool, str]:
    """The three citation-derived columns shared by CSV and XLSX, escaped for cells."""
    if citation is None:
        return _UNVERIFIED_LOCATION_LABEL, False, ""
    return citation.location_label, citation.is_verified, _escape_cell(citation.quoted_text)


def _uses_records(insight: Insight) -> bool:
    """Whether to export this result as a multi-record table.

    Records win when present. A result with no records but some ``fields`` is a
    single-record result cached before Milestone 4 made `/structure` multi-record — it is
    exported in its original layout rather than dropped. A result with neither (only
    reachable by writing the cache row directly; `/structure` itself refuses to cache a
    result with nothing in it) exports as an empty *table*, since that is the shape
    `/structure` produces now and the requested schema still names the columns.
    """
    return bool(insight.records) or not insight.fields


def _record_values(record: tuple[ExtractedField, ...]) -> dict[str, ExtractedField]:
    """Index one record's fields by name, so a column lookup is a dict hit rather than a
    scan — and so a field absent from this record is simply a missing key."""
    return {field.name: field for field in record}


def _to_json(insight: Insight, schema: dict[str, str]) -> bytes:
    """JSON export.

    Never escaped for formula injection — that mitigation exists only for spreadsheet
    programs that interpret a leading `=`/`+`/`-`/`@` as a formula; a JSON parser does no
    such thing, and escaping here would corrupt the actual extracted value for no
    security benefit. No chunk/block/database IDs are included — only what a caller
    could already see in a `/structure` response.

    A multi-record result exports as ``records``: one object per record, keyed by field
    name, so the file reads as the table it represents. Each record's sources travel with
    it under :data:`_JSON_SOURCES_KEY`, in the same ``{quoted_text, location_label,
    is_verified}`` shape a single-record export already used — provenance is the one thing
    JSON can carry that the flat CSV/XLSX table cannot, and dropping it would make the
    export the only place in the product where a value appears without its source.
    """
    payload: dict[str, object] = {
        "document_id": str(insight.document_id),
        "model": insight.model,
        "created_at": insight.created_at.isoformat() if insight.created_at else None,
    }

    if _uses_records(insight):
        payload["records"] = [_record_to_json(record, schema) for record in insight.records]
    else:
        payload["fields"] = [_field_to_json(field) for field in insight.fields]

    return json.dumps(payload, indent=2).encode("utf-8")


def _record_to_json(
    record: tuple[ExtractedField, ...], schema: dict[str, str]
) -> dict[str, object]:
    """One record as a flat ``{field_name: value}`` object, plus its sources.

    Only fields this record actually states are included: a field the document does not
    give for this entity is absent rather than present-and-null, matching how the record
    itself was built.
    """
    by_name = _record_values(record)

    values: dict[str, object] = {}
    sources: dict[str, object] = {}
    for name in schema:
        field = by_name.get(name)
        if field is None:
            continue
        values[name] = field.value
        sources[name] = _citation_to_json(field.citation)

    values[_JSON_SOURCES_KEY] = sources
    return values


def _citation_to_json(citation: Citation | None) -> dict[str, object] | None:
    if citation is None:
        return None
    return {
        "quoted_text": citation.quoted_text,
        "location_label": citation.location_label,
        "is_verified": citation.is_verified,
    }


def _field_to_json(field: ExtractedField) -> dict[str, object]:
    return {
        "name": field.name,
        "value": field.value,
        "citation": _citation_to_json(field.citation),
    }


def _table_rows(insight: Insight, schema: dict[str, str]) -> tuple[list[str], list[list[object]]]:
    """The header and data rows shared by the CSV and XLSX exports.

    Built once here rather than twice, so the two formats cannot drift into disagreeing
    about column order or about how a missing value is rendered — the same reason the
    single-record layout already shared ``_TABLE_COLUMNS`` between them.

    Values stay text, exactly as ``ExtractedField.value`` stores them: a document's amount
    may be "$1,200" or "1200 USD", and converting it to a number here would be
    interpretation, which is not export's job.
    """
    if _uses_records(insight):
        # The requested schema is the column set, so every row lines up under the same
        # headers even when a record omits a field.
        header = list(schema)
        rows: list[list[object]] = []
        for record in insight.records:
            by_name = _record_values(record)
            rows.append(
                [_escape_cell(by_name[name].value) if name in by_name else "" for name in header]
            )
        return header, rows

    header = list(_TABLE_COLUMNS)
    legacy_rows: list[list[object]] = []
    for field in insight.fields:
        location_label, is_verified, quoted_text = _citation_row(field.citation)
        legacy_rows.append(
            [field.name, _escape_cell(field.value), location_label, is_verified, quoted_text]
        )
    return header, legacy_rows


def _to_csv(insight: Insight, schema: dict[str, str]) -> bytes:
    """CSV export: one row per record, one column per requested field.

    ``csv.writer`` handles quoting, so a value containing a comma, quote or newline
    cannot break the row structure; formula-injection escaping is applied on top of that.
    """
    header, rows = _table_rows(insight, schema)

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)

    return buffer.getvalue().encode("utf-8")


def _to_xlsx(insight: Insight, schema: dict[str, str]) -> bytes:
    """XLSX export: one worksheet holding every record, same columns as the CSV export."""
    header, rows = _table_rows(insight, schema)

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Structured Data"
    sheet.append(header)
    for row in rows:
        sheet.append(row)

    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


_FORMATTERS: dict[ExportFormat, Callable[[Insight, dict[str, str]], bytes]] = {
    "json": _to_json,
    "csv": _to_csv,
    "xlsx": _to_xlsx,
}
