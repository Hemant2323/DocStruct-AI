"""The ``content_blocks`` table — extracted content, one row per block.

Blocks are stored individually rather than as one JSON blob on ``documents`` because a
citation points at exactly one of them. Individually addressable rows are what make
"contract.pdf, page 4" resolvable later.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.db.base import GUID, Base, UTCDateTime, utcnow
from app.models.domain.content import BlockType


class ContentBlockRecord(Base):
    """One extracted block of content belonging to a document."""

    __tablename__ = "content_blocks"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)

    # CASCADE so deleting a document can never orphan its content.
    # NOTE: SQLite only enforces this when `PRAGMA foreign_keys=ON` is set per
    # connection — see the listener in app/core/database.py. Without it, cascade
    # silently does nothing on SQLite while working correctly on PostgreSQL.
    document_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Reading order within the document, starting at 0.
    order: Mapped[int] = mapped_column(Integer, nullable=False)

    block_type: Mapped[BlockType] = mapped_column(String(16), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    # Heading depth (1 = top level); null for non-headings.
    level: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Where this came from in the original file. Stored as JSON because the shape
    # genuinely differs per format (page vs. sheet+cell vs. line range), and in the MVP
    # provenance is rendered rather than queried. Nine mostly-null columns would buy
    # queryability nothing needs yet.
    provenance: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    # The rendered location ("page 4"), denormalized from provenance so displaying a
    # citation needs no per-format formatting logic in the API or the frontend.
    location_label: Mapped[str] = mapped_column(String(255), nullable=False)

    # Format-specific extras: column types for a schema block, row counts for a table.
    block_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)

    __table_args__ = (
        # Reading a document's content in order is the only access pattern.
        Index("ix_content_blocks_document_order", "document_id", "order"),
        # Two blocks cannot occupy the same position; catches a buggy re-parse that
        # writes a second generation of blocks over the first.
        UniqueConstraint("document_id", "order", name="uq_content_blocks_document_order"),
    )

    def __repr__(self) -> str:
        return (
            f"<ContentBlockRecord id={self.id} document_id={self.document_id} "
            f"order={self.order} type={self.block_type}>"
        )
