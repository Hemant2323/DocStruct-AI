"""The ``chunks`` table — retrieval-sized pieces assembled from content blocks.

A chunk is what will later be searched and handed to the model. The important column is
``block_ids``: it is the middle link in the citation chain

    file -> block(provenance) -> chunk(block_ids) -> context -> citation

Without it a chunk could be retrieved but never attributed, and citations would become
plausible-sounding fiction.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.db.base import GUID, Base, UTCDateTime, utcnow


class ChunkRecord(Base):
    """One retrieval-sized piece of a document."""

    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)

    document_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )

    order: Mapped[int] = mapped_column(Integer, nullable=False)

    text: Mapped[str] = mapped_column(Text, nullable=False)

    # Which content blocks this chunk was built from, as a JSON array of block id
    # strings. This is the citation link: retrieved chunk -> blocks -> provenance.
    block_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    # Rough token count. Deliberately an estimate: Anthropic ships no local tokenizer,
    # and count_tokens is a network call, which would be wrong inside an ingest loop.
    # See app/ingestion/chunker.py for the character-per-token constant.
    token_estimate: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Span covered by this chunk, e.g. "pages 3-4". Built from the blocks' labels so a
    # retrieved chunk can be attributed without loading every block it came from.
    location_label: Mapped[str] = mapped_column(String(255), nullable=False, default="")

    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)

    __table_args__ = (
        Index("ix_chunks_document_order", "document_id", "order"),
        UniqueConstraint("document_id", "order", name="uq_chunks_document_order"),
    )

    def __repr__(self) -> str:
        return (
            f"<ChunkRecord id={self.id} document_id={self.document_id} "
            f"order={self.order} tokens~{self.token_estimate}>"
        )
