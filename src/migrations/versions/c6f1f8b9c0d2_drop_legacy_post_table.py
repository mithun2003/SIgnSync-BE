"""Drop legacy post table.

Revision ID: c6f1f8b9c0d2
Revises: a2df0b1e9f21
Create Date: 2026-03-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c6f1f8b9c0d2"
down_revision: str | Sequence[str] | None = "a2df0b1e9f21"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("post"):
        return

    existing_indexes = {idx["name"] for idx in inspector.get_indexes("post")}
    for index_name in ("ix_post_created_by_user_id", "ix_post_is_deleted"):
        if index_name in existing_indexes:
            op.drop_index(index_name, table_name="post")

    op.drop_table("post")


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("post"):
        return

    op.create_table(
        "post",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("created_by_user_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=30), nullable=False),
        sa.Column("text", sa.String(length=63206), nullable=False),
        sa.Column("uuid", sa.UUID(), nullable=False),
        sa.Column("media_url", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(["created_by_user_id"], ["user.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("uuid"),
    )
    op.create_index("ix_post_created_by_user_id", "post", ["created_by_user_id"], unique=False)
    op.create_index("ix_post_is_deleted", "post", ["is_deleted"], unique=False)
