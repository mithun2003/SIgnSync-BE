from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column

from ..core.db.database import Base


class SignDetection(Base):
    __tablename__ = "sign_detection"

    id: Mapped[int] = mapped_column(
        autoincrement=True,
        nullable=False,
        unique=True,
        primary_key=True,
        init=False,
    )

    # ── Who ──
    user_id: Mapped[int] = mapped_column(
        ForeignKey("user.id"),
        index=True,
        nullable=False,
    )

    # ── What was detected ──
    detected_sign: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        index=True,
    )
    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )
    is_correct: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
    )

    # ── Session tracking ──
    session_id: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        default=None,
        index=True,
    )
    duration_seconds: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )

    # ── Soft delete ──
    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        index=True,
    )

    # ── Timestamps ──
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=func.now(),
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=None,
    )

    # user = relationship(
    #     "User",
    #     back_populates="sign_detections"
    # )