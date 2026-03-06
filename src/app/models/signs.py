# backend/src/app/models/asl_sign.py
# Minimal ASL Sign model - backend only stores URL and version

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from ..core.db.database import Base


class Signs(Base):
    """
    Minimal ASL Sign model

    Backend only stores:
    - Which Cloudinary URL is currently active
    - Version number

    All image management happens in frontend!
    """

    __tablename__ = "signs"

    id: Mapped[int] = mapped_column(
        autoincrement=True,
        nullable=False,
        unique=True,
        primary_key=True,
        init=False,
    )

    # Character (A-Z, SPACE) - ONE record per character
    character: Mapped[str] = mapped_column(String(10), nullable=False, unique=True, index=True)

    # Current Cloudinary URL (selected by user)
    cloudinary_url: Mapped[str] = mapped_column(String(1000), nullable=False)

    # Last updated by user ID
    updated_by: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Version number (increments each time URL changes)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    # Optional notes
    notes: Mapped[str | None] = mapped_column(String(500), default=None)

    # Timestamps
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=False, default=func.now())

    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None, onupdate=func.now())

    def __repr__(self):
        return f"<Signs(character='{self.character}', version={self.version})>"
