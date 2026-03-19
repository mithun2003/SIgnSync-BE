"""Add emergency_contacts json column.

Revision ID: emergency_contacts_001
Revises: b88374be620f
Create Date: 2026-03-19
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "emergency_contacts_001"
down_revision: str | Sequence[str] | None = "b88374be620f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add emergency_contacts JSON column to user table."""
    op.add_column("user", sa.Column("emergency_contacts", sa.JSON(), nullable=True))


def downgrade() -> None:
    """Remove emergency_contacts column from user table."""
    op.drop_column("user", "emergency_contacts")
