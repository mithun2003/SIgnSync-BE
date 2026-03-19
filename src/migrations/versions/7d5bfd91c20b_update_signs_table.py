"""Update signs table.

Revision ID: 7d5bfd91c20b
Revises: fdcfd32542bb
Create Date: 2026-03-05 15:57:14.289162
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7d5bfd91c20b"
down_revision: str | None = "fdcfd32542bb"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("signs"):
        return


def downgrade() -> None:
    return
