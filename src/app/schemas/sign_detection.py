from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────────────────────────────────────────
#  BASE / CREATE / READ / UPDATE / DELETE — Standard FastCRUD set
# ─────────────────────────────────────────────────────────────
class SignDetectionBase(BaseModel):
    detected_sign: str = Field(..., max_length=32, examples=["A", "thumbs_down"])
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, examples=[0.95])
    is_correct: bool = Field(default=True)
    session_id: str | None = Field(default=None, max_length=50)
    duration_seconds: float = Field(default=0.0, ge=0.0)


class SignDetectionCreate(SignDetectionBase):
    """What the frontend sends to POST /detection/log."""

    model_config = ConfigDict(extra="forbid")


class SignDetectionCreateInternal(SignDetectionBase):
    """Internal — adds user_id before inserting."""

    user_id: int


class SignDetectionRead(BaseModel):
    """What the API returns."""

    id: int
    user_id: int
    detected_sign: str
    confidence: float
    is_correct: bool
    session_id: str | None
    duration_seconds: float
    created_at: datetime


class SignDetectionUpdate(BaseModel):
    """For PATCH (rarely used)"""

    model_config = ConfigDict(extra="forbid")

    is_correct: bool | None = None
    confidence: float | None = None


class SignDetectionUpdateInternal(SignDetectionUpdate):
    pass


class SignDetectionDelete(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_deleted: bool
    deleted_at: datetime


# ─────────────────────────────────────────────────────────────
#  BATCH CREATE
# ─────────────────────────────────────────────────────────────
class SignDetectionBatchCreate(BaseModel):
    """For POST /detection/log/batch."""

    model_config = ConfigDict(extra="forbid")

    detections: list[SignDetectionCreate] = Field(
        ...,
        min_length=1,
        max_length=100,
    )


# ─────────────────────────────────────────────────────────────
#  DASHBOARD RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────
class DashboardStats(BaseModel):
    total_signs_detected: int = 0
    today_signs_count: int = 0
    total_practice_hours: float = 0.0
    today_minutes: float = 0.0
    average_accuracy: float = 0.0
    accuracy_change: float = 0.0
    current_streak: int = 0


class FrequentSign(BaseModel):
    sign: str
    count: int


class DailyActivity(BaseModel):
    date: str
    count: int


class SignAccuracy(BaseModel):
    sign: str
    count: int
    accuracy: float


class AccuracyDistribution(BaseModel):
    high: int = 0  # 90%+
    medium: int = 0  # 70-90%
    low: int = 0  # <70%


class RecentActivity(BaseModel):
    id: int
    emoji: str
    description: str
    time_ago: str
    badge: str | None = None
    badge_text: str | None = None


class RecommendedLetter(BaseModel):
    char: str
    count: int
    accuracy: float


class DashboardResponse(BaseModel):
    """Single response for GET /dashboard — returns everything."""

    stats: DashboardStats
    frequent_signs: list[FrequentSign]
    daily_activity: list[DailyActivity]
    accuracy_distribution: AccuracyDistribution
    mastered_letters: int
    recent_activities: list[RecentActivity]
    recommended_letters: list[RecommendedLetter]
