"""Update signs table.

Revision ID: fdcfd32542bb
Revises: 30c13cb53dea
Create Date: 2026-03-05 14:40:41.065093
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "fdcfd32542bb"
down_revision: str | None = "30c13cb53dea"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not inspector.has_table("signs"):
        op.create_table(
            "signs",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("character", sa.String(length=10), nullable=False),
            sa.Column("cloudinary_url", sa.String(length=1000), nullable=False),
            sa.Column("updated_by", sa.Integer(), nullable=True),
            sa.Column("version", sa.Integer(), nullable=False),
            sa.Column("notes", sa.String(length=500), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )

    if inspector.has_table("signs"):
        existing_indexes = {idx["name"] for idx in inspector.get_indexes("signs")}
        if "ix_signs_character" not in existing_indexes:
            op.create_index("ix_signs_character", "signs", ["character"], unique=True)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not inspector.has_table("signs"):
        return

    existing_indexes = {idx["name"] for idx in inspector.get_indexes("signs")}
    if "ix_signs_character" in existing_indexes:
        op.drop_index("ix_signs_character", table_name="signs")
    op.drop_table("signs")
