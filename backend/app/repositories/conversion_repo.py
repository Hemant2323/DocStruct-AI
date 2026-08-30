"""Data access for completed document conversions.

Serialisation between the stored JSON and the API shape lives here, so the route works
with typed objects and never with dictionaries — the same division ``AIResultRepository``
uses.

**What is stored is the finished response shape**, not the internal domain tree. Two
reasons. Retrieval must return exactly what the conversion produced, with nothing
recomputed and no model consulted, and storing the answer directly makes that true by
construction rather than by careful re-derivation. And the response shape is already the
sanitised one: it carries a value's text, where it came from and whether that was
confirmed, but not the chunk and block identifiers behind it, so those never reach the
database at all.

The trade-off is that a future change to the response schema would need a migration or a
tolerant reader for rows written under the old one. That is noted rather than solved: the
alternative — persisting the domain tree and rebuilding the response on every read —
costs a second full serialisation layer for provenance we deliberately do not store.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast
from uuid import UUID

from sqlalchemy import CursorResult, delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.conversion import (
    ConversionCorrectionResponse,
    ConversionPlanResponse,
    ConversionRejectionResponse,
    ConversionResponse,
    ConvertedSectionResponse,
)
from app.models.db.document_conversion import DocumentConversionRecord


@dataclass(frozen=True, slots=True)
class StoredConversion:
    """A conversion as it exists in the database.

    Holds the response payload plus the row's own identity and timestamp, which are
    assigned on insert and so cannot be part of what the service produced.
    """

    id: UUID
    document_id: UUID
    owner_id: str
    instruction: str
    title: str
    markdown: str
    sections: list[dict[str, Any]]
    plan: dict[str, Any]
    corrections: list[dict[str, Any]]
    suggestions: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    schema_used: dict[str, str]
    vocabularies: dict[str, list[str]]
    warnings: list[str]
    source_record_count: int
    model: str
    input_tokens: int
    output_tokens: int
    records_from_cache: bool
    created_at: datetime

    def to_response(self) -> ConversionResponse:
        """Rebuild the API response from stored data. Reads only — nothing is regenerated."""
        return ConversionResponse(
            id=self.id,
            document_id=self.document_id,
            title=self.title,
            instruction=self.instruction,
            markdown=self.markdown,
            sections=[ConvertedSectionResponse.model_validate(item) for item in self.sections],
            plan=ConversionPlanResponse.model_validate(self.plan),
            corrections=[
                ConversionCorrectionResponse.model_validate(item) for item in self.corrections
            ],
            suggestions=[
                ConversionCorrectionResponse.model_validate(item) for item in self.suggestions
            ],
            rejected=[ConversionRejectionResponse.model_validate(item) for item in self.rejected],
            schema_used=dict(self.schema_used),
            vocabularies={name: list(values) for name, values in self.vocabularies.items()},
            source_record_count=self.source_record_count,
            warnings=list(self.warnings),
            model=self.model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            records_from_cache=self.records_from_cache,
            created_at=self.created_at,
        )


def _to_domain(record: DocumentConversionRecord) -> StoredConversion:
    return StoredConversion(
        id=record.id,
        document_id=record.document_id,
        owner_id=record.owner_id,
        instruction=record.instruction,
        title=record.title,
        markdown=record.markdown,
        sections=list(record.sections or []),
        plan=dict(record.plan or {}),
        corrections=list(record.corrections or []),
        suggestions=list(record.suggestions or []),
        rejected=list(record.rejected or []),
        schema_used=dict(record.schema_used or {}),
        vocabularies=dict(record.vocabularies or {}),
        warnings=list(record.warnings or []),
        source_record_count=record.source_record_count,
        model=record.model,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        records_from_cache=record.records_from_cache,
        created_at=record.created_at,
    )


class ConversionRepository:
    """Reads and writes completed conversions.

    Every read is owner-scoped, matching every other repository in the project: a
    conversion id is guessable, ownership is not, and a single forgotten ``WHERE`` is how
    multi-tenant leaks happen.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(
        self,
        response: ConversionResponse,
        *,
        document_id: UUID,
        owner_id: str,
    ) -> StoredConversion:
        """Store a completed conversion as a new row.

        Always an insert. Conversions are never replaced or de-duplicated: two runs of
        the same instruction are two results, and the second must not overwrite the first
        just because it was asked for in the same words.

        The response's own ``id`` and ``created_at`` are used for the row rather than
        letting the database generate its own, so what is stored is exactly what the
        caller was handed — no second identity appears between the two.

        Flushes before returning so an integrity failure surfaces here, inside the
        request, rather than at commit time after a response has been shaped.
        """
        record = DocumentConversionRecord(
            id=response.id,
            created_at=response.created_at,
            document_id=document_id,
            owner_id=owner_id,
            instruction=response.instruction,
            title=response.title,
            markdown=response.markdown,
            sections=[item.model_dump(mode="json") for item in response.sections],
            plan=response.plan.model_dump(mode="json"),
            corrections=[item.model_dump(mode="json") for item in response.corrections],
            suggestions=[item.model_dump(mode="json") for item in response.suggestions],
            rejected=[item.model_dump(mode="json") for item in response.rejected],
            schema_used=dict(response.schema_used),
            vocabularies={name: list(values) for name, values in response.vocabularies.items()},
            warnings=list(response.warnings),
            source_record_count=response.source_record_count,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            records_from_cache=response.records_from_cache,
        )
        self._session.add(record)
        await self._session.flush()
        await self._session.refresh(record)
        return _to_domain(record)

    async def get(
        self, conversion_id: UUID, *, document_id: UUID, owner_id: str
    ) -> StoredConversion | None:
        """Return one conversion, or ``None`` when it does not exist for this owner.

        Scoped by document *and* owner: knowing an id is not enough to read a result.
        """
        result = await self._session.execute(
            select(DocumentConversionRecord).where(
                DocumentConversionRecord.id == conversion_id,
                DocumentConversionRecord.document_id == document_id,
                DocumentConversionRecord.owner_id == owner_id,
            )
        )
        record = result.scalar_one_or_none()
        return _to_domain(record) if record else None

    async def list_for_document(
        self, document_id: UUID, *, owner_id: str
    ) -> list[StoredConversion]:
        """Return a document's conversions, newest first."""
        result = await self._session.execute(
            select(DocumentConversionRecord)
            .where(
                DocumentConversionRecord.document_id == document_id,
                DocumentConversionRecord.owner_id == owner_id,
            )
            .order_by(DocumentConversionRecord.created_at.desc())
        )
        return [_to_domain(record) for record in result.scalars().all()]

    async def delete_for_document(self, document_id: UUID) -> int:
        """Delete every conversion for a document. Returns how many were removed."""
        result = cast(
            "CursorResult[Any]",
            await self._session.execute(
                delete(DocumentConversionRecord).where(
                    DocumentConversionRecord.document_id == document_id
                )
            ),
        )
        await self._session.flush()
        return int(result.rowcount or 0)
