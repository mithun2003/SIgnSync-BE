"""Add version support.

Revision ID: b88374be620f
Revises: 42076eab43ab
Create Date: 2026-03-08 11:08:46.494368
"""

from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "b88374be620f"
down_revision: str | None = "42076eab43ab"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    return


def downgrade() -> None:
    return
