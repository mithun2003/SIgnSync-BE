# app/schemas/admin.py
from pydantic import BaseModel

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