"""Expand sign_detection.detected_sign length.

Revision ID: a2df0b1e9f21
Revises: emergency_contacts_001
Create Date: 2026-03-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a2df0b1e9f21"
down_revision: str | Sequence[str] | None = "emergency_contacts_001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("sign_detection"):
        return

    op.alter_column(
        "sign_detection",
        "detected_sign",
        existing_type=sa.String(length=10),
        type_=sa.String(length=32),
        existing_nullable=False,
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("sign_detection"):
        return

    op.alter_column(
        "sign_detection",
        "detected_sign",
        existing_type=sa.String(length=32),
        type_=sa.String(length=10),
        existing_nullable=False,
    )
