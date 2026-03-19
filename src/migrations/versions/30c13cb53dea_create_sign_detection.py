"""Create sign_detection.

Revision ID: 30c13cb53dea
Revises:
Create Date: 2026-03-03 13:34:50.030687
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "30c13cb53dea"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not inspector.has_table("sign_detection"):
        op.create_table(
            "sign_detection",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("detected_sign", sa.String(length=10), nullable=False),
            sa.Column("confidence", sa.Float(), nullable=False),
            sa.Column("is_correct", sa.Boolean(), nullable=False),
            sa.Column("session_id", sa.String(length=50), nullable=True),
            sa.Column("duration_seconds", sa.Float(), nullable=False),
            sa.Column("is_deleted", sa.Boolean(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(["user_id"], ["user.id"]),
            sa.PrimaryKeyConstraint("id"),
        )

    if inspector.has_table("sign_detection"):
        existing_indexes = {idx["name"] for idx in inspector.get_indexes("sign_detection")}
        index_specs = [
            ("ix_sign_detection_detected_sign", ["detected_sign"]),
            ("ix_sign_detection_is_deleted", ["is_deleted"]),
            ("ix_sign_detection_session_id", ["session_id"]),
            ("ix_sign_detection_user_id", ["user_id"]),
        ]
        for index_name, columns in index_specs:
            if index_name not in existing_indexes:
                op.create_index(index_name, "sign_detection", columns, unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not inspector.has_table("sign_detection"):
        return

    existing_indexes = {idx["name"] for idx in inspector.get_indexes("sign_detection")}
    for index_name in [
        "ix_sign_detection_user_id",
        "ix_sign_detection_session_id",
        "ix_sign_detection_is_deleted",
        "ix_sign_detection_detected_sign",
    ]:
        if index_name in existing_indexes:
            op.drop_index(index_name, table_name="sign_detection")

    op.drop_table("sign_detection")
