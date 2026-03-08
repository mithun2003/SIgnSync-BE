from datetime import datetime

from sqlalchemy import DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from ..core.db.database import Base


class Signs(Base):
    """One record per ASL character (A-Z, SPACE).

    Stores the currently-active Cloudinary image URL and its metadata. All historical versions live in Cloudinary under
    asl-signs/{CHARACTER}/. The admin can list those versions via the Cloudinary API and switch the active image at any
    time.
    """

    __tablename__ = "signs"

    id: Mapped[int] = mapped_column(
        autoincrement=True,
        nullable=False,
        unique=True,
        primary_key=True,
        init=False,
    )

    # Character (A-Z, SPACE) – ONE record per character
    character: Mapped[str] = mapped_column(String(10), nullable=False, unique=True, index=True)

    # Active Cloudinary image
    cloudinary_url: Mapped[str] = mapped_column(String(1000), nullable=False)
    # Who last updated
    updated_by: Mapped[int | None] = mapped_column(Integer, nullable=True)

    cloudinary_public_id: Mapped[str | None] = mapped_column(String(500), nullable=True, default=None)

    # Image metadata for the active image
    file_size: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)  # bytes
    width: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)  # pixels
    height: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)  # pixels
    mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True, default=None)

    # Increments every time the active image changes
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    notes: Mapped[str | None] = mapped_column(String(500), default=None)

    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None, onupdate=func.now())

    def __repr__(self) -> str:
        return f"<Signs(character='{self.character}', version={self.version})>"
