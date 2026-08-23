"""Data access for cached AI results.

Serialisation between the stored JSON and the domain objects lives here, so the service
layer works with :class:`Insight` and never with dictionaries.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import UUID

from sqlalchemy import CursorResult, delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db.ai_result import AIResultRecord
from app.models.domain.content import Provenance
from app.models.domain.insight import (
    Citation,
    ContextMode,
    ExtractedField,
    Insight,
    InsightKind,
)


def _citation_to_dict(citation: Citation) -> dict[str, Any]:
    return {
        "quoted_text": citation.quoted_text,
        "provenance": citation.provenance.to_dict() if citation.provenance else None,
        "chunk_id": str(citation.chunk_id) if citation.chunk_id else None,
        "block_ids": [str(block_id) for block_id in citation.block_ids],
    }


def _citation_from_dict(data: dict[str, Any]) -> Citation:
    provenance_data = data.get("provenance")
    return Citation(
        quoted_text=data.get("quoted_text", ""),
        # A stored null stays null. An unverified citation must never gain a location
        # on the way back out of the database.
        provenance=Provenance.from_dict(provenance_data) if provenance_data else None,
        chunk_id=UUID(data["chunk_id"]) if data.get("chunk_id") else None,
        block_ids=tuple(UUID(value) for value in data.get("block_ids", [])),
    )


def _field_to_dict(extracted: ExtractedField) -> dict[str, Any]:
    return {
        "name": extracted.name,
        "value": extracted.value,
        "citation": _citation_to_dict(extracted.citation) if extracted.citation else None,
    }


def _field_from_dict(data: dict[str, Any]) -> ExtractedField:
    citation_data = data.get("citation")
    return ExtractedField(
        name=data.get("name", ""),
        value=data.get("value", ""),
        citation=_citation_from_dict(citation_data) if citation_data else None,
    )


def _to_domain(record: AIResultRecord) -> Insight:
    return Insight(
        document_id=record.document_id,
        kind=InsightKind(record.kind),
        content=record.content,
        fields=tuple(_field_from_dict(item) for item in (record.fields or [])),
        citations=tuple(_citation_from_dict(item) for item in (record.citations or [])),
        context_mode=ContextMode(record.context_mode),
        model=record.model,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        warnings=tuple(record.warnings or []),
        created_at=record.created_at,
    )


class AIResultRepository:
    """Reads and writes cached AI results.

    Every read is scoped by owner, matching the rule applied to every other repository:
    an analysis of someone else's document must be unreachable.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, document_id: UUID, *, owner_id: str, fingerprint: str) -> Insight | None:
        """Return a cached result, or ``None`` if this exact analysis has not been run."""
        result = await self._session.execute(
            select(AIResultRecord).where(
                AIResultRecord.document_id == document_id,
                AIResultRecord.owner_id == owner_id,
                AIResultRecord.fingerprint == fingerprint,
            )
        )
        record = result.scalar_one_or_none()
        return _to_domain(record) if record else None

    async def list_for_document(self, document_id: UUID, *, owner_id: str) -> list[Insight]:
        """Return every cached result for a document, newest first."""
        result = await self._session.execute(
            select(AIResultRecord)
            .where(
                AIResultRecord.document_id == document_id,
                AIResultRecord.owner_id == owner_id,
            )
            .order_by(AIResultRecord.created_at.desc())
        )
        return [_to_domain(record) for record in result.scalars().all()]

    async def save(self, insight: Insight, *, owner_id: str, fingerprint: str) -> Insight:
        """Store a result, replacing any previous one with the same fingerprint.

        Replacing rather than appending keeps exactly one cached answer per distinct
        analysis, which is what the unique constraint expresses.
        """
        existing = await self._session.execute(
            select(AIResultRecord).where(
                AIResultRecord.document_id == insight.document_id,
                AIResultRecord.owner_id == owner_id,
                AIResultRecord.fingerprint == fingerprint,
            )
        )
        record = existing.scalar_one_or_none()

        if record is None:
            record = AIResultRecord(
                document_id=insight.document_id,
                owner_id=owner_id,
                kind=insight.kind,
                fingerprint=fingerprint,
            )
            self._session.add(record)

        record.content = insight.content
        record.fields = [_field_to_dict(item) for item in insight.fields]
        record.citations = [_citation_to_dict(item) for item in insight.citations]
        record.warnings = list(insight.warnings)
        record.context_mode = insight.context_mode
        record.model = insight.model
        record.input_tokens = insight.input_tokens
        record.output_tokens = insight.output_tokens

        await self._session.flush()
        await self._session.refresh(record)
        return _to_domain(record)

    async def delete_for_document(self, document_id: UUID) -> int:
        """Delete every cached result for a document. Returns how many were removed."""
        result = cast(
            "CursorResult[Any]",
            await self._session.execute(
                delete(AIResultRecord).where(AIResultRecord.document_id == document_id)
            ),
        )
        await self._session.flush()
        return int(result.rowcount or 0)
