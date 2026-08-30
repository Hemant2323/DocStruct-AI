"""add records column to ai results

Revision ID: 2a78b7a9d62e
Revises: 8eff50a40d0b
Create Date: 2026-08-25 17:48:22.638176
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "2a78b7a9d62e"
down_revision: str | None = "8eff50a40d0b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # server_default backfills every existing row (empty JSON array) so a NOT NULL
    # column can be added to a table that may already hold rows — the app layer's own
    # default (`default=list` on the mapped column) only applies to new inserts and
    # would otherwise leave existing rows with no value to satisfy NOT NULL.
    with op.batch_alter_table("ai_results", schema=None) as batch_op:
        batch_op.add_column(sa.Column("records", sa.JSON(), nullable=False, server_default="[]"))


def downgrade() -> None:
    with op.batch_alter_table("ai_results", schema=None) as batch_op:
        batch_op.drop_column("records")
