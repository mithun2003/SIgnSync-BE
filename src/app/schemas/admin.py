# app/schemas/admin.py
from pydantic import BaseModel

# ─── Dashboard ───────────────────────────────────────────────────────────────


class AdminStats(BaseModel):
    total_users: int
    active_users_today: int
    total_detections: int
    system_health: int
    user_growth_percent: float
    detection_growth_percent: float


class RecentActivity(BaseModel):
    id: int
    type: str
    emoji: str
    title: str
    description: str
    time_ago: str


class SystemService(BaseModel):
    name: str
    status: str  # "online" | "offline" | "warning"


class TopUser(BaseModel):
    username: str
    detections: int
    accuracy: float


class AdminDashboardData(BaseModel):
    stats: AdminStats
    recent_activities: list[RecentActivity]
    system_services: list[SystemService]
    top_users: list[TopUser]


class AdminDashboardResponse(BaseModel):
    data: AdminDashboardData


# ─── System Health ────────────────────────────────────────────────────────────


class ServiceHealth(BaseModel):
    name: str
    status: str  # "online" | "offline" | "warning"
    latency_ms: float | None = None


class SystemHealthDetail(BaseModel):
    status: str
    timestamp: str
    services: list[ServiceHealth]


class SystemHealthResponse(BaseModel):
    data: SystemHealthDetail


# ─── Active Users ─────────────────────────────────────────────────────────────


class ActiveUserItem(BaseModel):
    username: str
    email: str
    last_login_at: str | None
    created_at: str


class ActiveUsersResponse(BaseModel):
    data: list[ActiveUserItem]
    total: int
    period_hours: int


# ─── Cache ────────────────────────────────────────────────────────────────────


class CacheClearResponse(BaseModel):
    message: str
    cleared_at: str


# ─── Backup ───────────────────────────────────────────────────────────────────


class BackupRecord(BaseModel):
    backup_id: str
    created_at: str
    size_bytes: int
    record_counts: dict[str, int]
    file_path: str


class BackupInfoResponse(BaseModel):
    last_backup: BackupRecord | None
    message: str


# ─── Analytics ────────────────────────────────────────────────────────────────


class DailyCount(BaseModel):
    date: str
    count: int


class CountryCount(BaseModel):
    country: str
    count: int


class UserAnalyticsData(BaseModel):
    period_days: int
    total_users: int
    new_users_in_period: int
    active_users_in_period: int
    inactive_users: int
    growth_percent: float
    daily_registrations: list[DailyCount]
    top_countries: list[CountryCount]


class UserAnalyticsResponse(BaseModel):
    data: UserAnalyticsData


# ─── Settings ─────────────────────────────────────────────────────────────────


class GeneralSettingsUpdate(BaseModel):
    app_name: str | None = None
    app_description: str | None = None
    timezone: str | None = None
    language: str | None = None


class SecuritySettingsUpdate(BaseModel):
    min_password_length: int | None = None
    require_uppercase: bool | None = None
    require_numbers: bool | None = None
    require_special_chars: bool | None = None
    session_timeout: int | None = None
    enable_2fa: bool | None = None


class EmailNotificationsUpdate(BaseModel):
    new_users: bool | None = None
    system_alerts: bool | None = None
    backup_completion: bool | None = None


class InAppNotificationsUpdate(BaseModel):
    show_badges: bool | None = None
    auto_dismiss: bool | None = None


class NotificationsSettingsUpdate(BaseModel):
    email: EmailNotificationsUpdate | None = None
    in_app: InAppNotificationsUpdate | None = None


class SystemSettingsUpdate(BaseModel):
    cache_duration: int | None = None
    debug_mode: bool | None = None
    maintenance_mode: bool | None = None
    auto_backup: bool | None = None


class SettingsUpdate(BaseModel):
    general: GeneralSettingsUpdate | None = None
    security: SecuritySettingsUpdate | None = None
    notifications: NotificationsSettingsUpdate | None = None
    system: SystemSettingsUpdate | None = None
