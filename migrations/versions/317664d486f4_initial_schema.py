"""initial_schema

Revision ID: 317664d486f4
Revises:
Create Date: 2026-10-23 16:39:38.377359

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "317664d486f4"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

json_type = sa.JSON().with_variant(JSONB, "postgresql")
pk_bigint = sa.BigInteger().with_variant(sa.Integer(), "sqlite")


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "leads",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("phone", sa.String(length=32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.create_table(
        "calls",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "lead_id",
            sa.Uuid(),
            sa.ForeignKey("leads.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("room_name", sa.String(length=128), nullable=False),
        sa.Column(
            "mode", sa.String(length=16), nullable=False, server_default="browser"
        ),
        sa.Column(
            "status", sa.String(length=32), nullable=False, server_default="dialing"
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_seconds", sa.Integer(), nullable=True),
        sa.Column("extracted_answers", json_type, nullable=True),
        sa.Column("extracted_missing", json_type, nullable=True),
        sa.Column("extraction_status", sa.String(length=32), nullable=True),
        sa.Column("latency_summary", json_type, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_calls_room_name", "calls", ["room_name"], unique=True)
    op.create_index("ix_calls_status", "calls", ["status"])
    op.create_index("ix_calls_created_at", "calls", ["created_at"])

    op.create_table(
        "call_turns",
        sa.Column("id", pk_bigint, primary_key=True, autoincrement=True),
        sa.Column(
            "call_id",
            sa.Uuid(),
            sa.ForeignKey("calls.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("call_id", "turn_index", name="uq_call_turns_call_turn"),
    )
    op.create_index("ix_call_turns_call_id", "call_turns", ["call_id"])

    op.create_table(
        "call_events",
        sa.Column("id", pk_bigint, primary_key=True, autoincrement=True),
        sa.Column(
            "call_id",
            sa.Uuid(),
            sa.ForeignKey("calls.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("payload", json_type, nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_call_events_call_id", "call_events", ["call_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("call_events")
    op.drop_table("call_turns")
    op.drop_table("calls")
    op.drop_table("leads")
