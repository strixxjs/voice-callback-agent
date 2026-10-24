"""call_config_and_turn_metrics

Revision ID: c4b1a8e2d3f5
Revises: 317664d486f4
Create Date: 2026-10-24 15:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "c4b1a8e2d3f5"
down_revision: str | Sequence[str] | None = "317664d486f4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

json_type = sa.JSON().with_variant(JSONB, "postgresql")


def upgrade() -> None:
    """Upgrade schema: add call_config to calls and metrics to call_turns."""
    op.add_column("calls", sa.Column("call_config", json_type, nullable=True))
    op.add_column("call_turns", sa.Column("metrics", json_type, nullable=True))


def downgrade() -> None:
    """Downgrade schema: drop call_config from calls and metrics from call_turns."""
    op.drop_column("call_turns", "metrics")
    op.drop_column("calls", "call_config")
