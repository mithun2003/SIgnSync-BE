"""Add version support.

Revision ID: 42076eab43ab
Revises: 7d5bfd91c20b
Create Date: 2026-03-08 09:37:19.649626
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "42076eab43ab"
down_revision: str | None = "7d5bfd91c20b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if inspector.has_table("signs"):
        existing_columns = {column["name"] for column in inspector.get_columns("signs")}
        for column in [
            sa.Column("cloudinary_public_id", sa.String(length=500), nullable=True),
            sa.Column("file_size", sa.Integer(), nullable=True),
            sa.Column("width", sa.Integer(), nullable=True),
            sa.Column("height", sa.Integer(), nullable=True),
            sa.Column("mime_type", sa.String(length=100), nullable=True),
        ]:
            if column.name not in existing_columns:
                op.add_column("signs", column)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("signs"):
        return

    existing_columns = {column["name"] for column in inspector.get_columns("signs")}
    for column_name in ["mime_type", "height", "width", "file_size", "cloudinary_public_id"]:
        if column_name in existing_columns:
            op.drop_column("signs", column_name)
