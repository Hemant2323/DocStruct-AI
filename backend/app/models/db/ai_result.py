"""The ``ai_results`` table — cached AI output for a document.

A model call costs money and takes seconds, so re-running the same analysis every time
someone opens a document would be pure waste. This table stores each result so it is
produced once and read many times.

The important column is ``fingerprint``. It covers everything that could change the
answer — the kind of analysis, the model, the prompt version, and the context mode — so a
prompt edit or a model change invalidates old results automatically instead of quietly
serving something stale.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.db.base import GUID, Base, UTCDateTime, utcnow
from app.models.domain.insight import ContextMode, InsightKind


class AIResultRecord(Base):
    """One cached AI result."""

    __tablename__ = "ai_results"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)

    # CASCADE: deleting a document must take its AI results with it. Enforced on SQLite
    # by the foreign-key pragma set in app/core/database.py.
    document_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Denormalised from the document so a result can be owner-scoped without a join.
    owner_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    kind: Mapped[InsightKind] = mapped_column(String(16), nullable=False)

    # Hash of (kind, model, prompt version, context mode, schema). Changing any input
    # that could change the answer produces a different fingerprint, so stale results
    # are never returned.
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)

    # Prose for summaries and explanations; empty for extractions.
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # Extracted fields, for a single-record extraction result.
    fields: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)

    # Multi-record structured results (Milestone 4): one list of field-dicts per matching
    # record a document contains (e.g. one per transaction), rather than the single row
    # `fields` holds. A sibling of `fields`, not a replacement — populated only for
    # multi-record structure results; every other kind leaves this an empty list, exactly
    # as it leaves `fields` empty today.
    records: Mapped[list[list[dict[str, Any]]]] = mapped_column(JSON, nullable=False, default=list)

    # Resolved citations: quoted text plus the provenance we looked up ourselves.
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)

    # Anything the user should know about how this result was produced.
    warnings: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    context_mode: Mapped[ContextMode] = mapped_column(
        String(16), nullable=False, default=ContextMode.FULL
    )

    model: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)

    __table_args__ = (
        # The cache lookup: "do we already have this exact analysis for this document?"
        Index("ix_ai_results_document_fingerprint", "document_id", "fingerprint"),
        # One cached result per (document, fingerprint). A second identical run replaces
        # the first rather than accumulating duplicates.
        UniqueConstraint("document_id", "fingerprint", name="uq_ai_results_document_fingerprint"),
    )

    def __repr__(self) -> str:
        return (
            f"<AIResultRecord id={self.id} document_id={self.document_id} "
            f"kind={self.kind} model={self.model}>"
        )
