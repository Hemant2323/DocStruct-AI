"""The ``document_conversions`` table — completed document conversions.

A conversion is expensive: it costs three or four model calls and takes seconds, and the
user will want to look at the result again, share it, and eventually download it in
several formats. So the finished result is stored rather than recomputed.

**Its own table, not a row in ``ai_results``.** A conversion is a different artefact from
an insight: it has a section tree, a layout plan, and a three-way correction audit, none
of which ``ai_results`` has a column for. Adding them there would leave six columns null
on every summary, explanation and extraction row — the same shape-fork that already had
to be worked around once when ``records`` joined ``fields``, repeated at larger scale.

**No fingerprint column, deliberately.** ``ai_results`` is keyed by one so an identical
analysis is served from cache. Conversion has no such cache: each run is its own record,
because the same instruction legitimately produces a different document as the model's
plan or corrections change, and silently returning yesterday's report would be worse than
paying for a new one. Rows are therefore distinguished by ``id`` and ``created_at``, and
listing is newest-first.

What is stored is the finished, already-sanitised result — the shape the API returns. No
provider response, no prompt, no credential, and no chunk or block identifier goes in
here.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.db.base import GUID, Base, UTCDateTime, utcnow


class DocumentConversionRecord(Base):
    """One completed conversion of one document."""

    __tablename__ = "document_conversions"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)

    # CASCADE: deleting a document must take its conversions with it. Enforced on SQLite
    # by the foreign-key pragma set in app/core/database.py.
    document_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Denormalised from the document so a conversion can be owner-scoped without a join.
    # Every read filters on it: a conversion id is guessable, ownership is not.
    owner_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    # What the user asked for. Kept so a stored conversion can explain itself later —
    # without it, a report's organisation would be unaccountable.
    instruction: Mapped[str] = mapped_column(Text, nullable=False, default="")

    title: Mapped[str] = mapped_column(String(255), nullable=False, default="")

    # The rendered document. Stored rather than re-rendered so retrieval is a read and
    # nothing about the result can drift after the fact.
    markdown: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # The section tree, the layout that produced it, and the correction audit, each in
    # the already-sanitised shape the API returns.
    sections: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    plan: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    corrections: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    suggestions: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    #: Proposals the validator refused, with the reason. Kept because "the AI tried to
    #: change this and was not allowed to" is exactly what a reviewer of a financial
    #: document needs to be able to see.
    rejected: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)

    schema_used: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False, default=dict)
    vocabularies: Mapped[dict[str, list[str]]] = mapped_column(JSON, nullable=False, default=dict)
    warnings: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    source_record_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    model: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    #: Whether the *extraction* behind this conversion was served from cache. Named to
    #: say what it covers: the conversion itself is never cached.
    records_from_cache: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)

    __table_args__ = (
        # The listing query: "this document's conversions, newest first."
        Index("ix_document_conversions_document_created", "document_id", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<DocumentConversionRecord id={self.id} document_id={self.document_id} "
            f"title={self.title!r}>"
        )
