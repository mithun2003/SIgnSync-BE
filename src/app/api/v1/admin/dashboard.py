import csv
import io
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from fastcrud import JoinConfig
from sqlalchemy import Integer, cast, select, text
from sqlalchemy import func as sa_func
from sqlalchemy.ext.asyncio import AsyncSession

from ....api.dependencies import get_current_superuser
from ....core.db.database import async_get_db
from ....core.ml.predict import get_health_status
from ....crud.crud_admin import detection_crud, user_crud
from ....models.sign_detection import SignDetection
from ....models.user import User
from ....schemas.admin import AdminDashboardResponse

router = APIRouter(prefix="/dashboard", tags=["Admin Dashboard"])


# ─── HELPERS ─────────────────────────────────────────────


def _calc_growth(current: int, previous: int) -> float:
    if previous == 0:
        return 100.0 if current > 0 else 0.0
    return round(((current - previous) / previous) * 100, 1)


async def _check_db(db: AsyncSession) -> str:
    try:
        await db.execute(text("SELECT 1"))
        return "online"
    except Exception:
        return "offline"


def _check_ml_model() -> str:
    try:
        health = get_health_status()
        return "online" if health.get("status") == "healthy" else "warning"
    except Exception:
        return "offline"


def _format_time_ago(elapsed: timedelta) -> str:
    seconds = int(elapsed.total_seconds())
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{elapsed.days}d ago"


def _format_confidence(conf: float) -> str:
    """Handle both 0.95 and 95.0 formats."""
    if conf > 1:
        return f"{conf:.0f}%"
    return f"{conf:.0%}"


# ─── MAIN ENDPOINT ───────────────────────────────────────


@router.get(
    "",
    response_model=AdminDashboardResponse,
    dependencies=[Depends(get_current_superuser)],
)
async def get_admin_dashboard(
    db: Annotated[AsyncSession, Depends(async_get_db)],
):
    # ✅ FIX: Use timezone-aware datetime
    now = datetime.now(UTC)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    month_ago = now - timedelta(days=30)
    prev_month_start = now - timedelta(days=60)

    # ─── USER STATS (FastCRUD — simple counts) ───
    total_users = await user_crud.count(db, is_deleted=False)
    active_today = await user_crud.count(db, is_deleted=False, last_login_at__gte=today_start)
    users_this_month = await user_crud.count(db, is_deleted=False, created_at__gte=month_ago)
    users_last_month = await user_crud.count(
        db,
        is_deleted=False,
        created_at__gte=prev_month_start,
        created_at__lt=month_ago,
    )
    user_growth = _calc_growth(users_this_month, users_last_month)

    # ─── DETECTION STATS (FastCRUD — simple counts) ───
    total_detections = await detection_crud.count(db)
    detections_this_month = await detection_crud.count(db, created_at__gte=month_ago)
    detections_last_month = await detection_crud.count(
        db,
        created_at__gte=prev_month_start,
        created_at__lt=month_ago,
    )
    detection_growth = _calc_growth(detections_this_month, detections_last_month)

    # ─── TOP USERS (Raw SQLAlchemy — needs GROUP BY + ORDER BY alias) ───
    top_users_stmt = (
        select(
            User.username,
            sa_func.count(SignDetection.id).label("detections_count"),
            sa_func.avg(cast(SignDetection.is_correct, Integer)).label("avg_accuracy"),
        )
        .outerjoin(SignDetection, SignDetection.user_id == User.id)
        .where(~User.is_deleted)
        .group_by(User.id, User.username)
        .order_by(sa_func.count(SignDetection.id).desc())
        .limit(5)
    )
    result = await db.execute(top_users_stmt)
    rows = result.all()

    top_users = [
        {
            "username": row.username,
            "detections": row.detections_count,
            "accuracy": round(row.avg_accuracy, 4) if row.avg_accuracy is not None else 0.0,
        }
        for row in rows
    ]

    # ─── RECENT ACTIVITIES (FastCRUD — sort by real column created_at) ───
    join_config = JoinConfig(
        model=User,
        join_on=SignDetection.user_id == User.id,
        join_prefix="user_",
        join_type="inner",
    )
    recent_result = await detection_crud.get_multi_joined(
        db=db,
        joins_config=[join_config],
        sort_columns="created_at",
        sort_orders="desc",
        limit=10,
    )

    recent_activities = []
    for i, det in enumerate(recent_result.get("data", [])):
        created = det.get("created_at")

        # ✅ FIX: Handle both naive and aware datetimes
        if created:
            # Make created timezone-aware if it's naive
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            elapsed = now - created
        else:
            elapsed = timedelta(hours=99)

        username = det.get("user_username", det.get("username", "User"))
        sign = det.get("detected_sign", "?")
        conf = det.get("confidence", 0)

        recent_activities.append(
            {
                "id": i + 1,
                "type": "detection",
                "emoji": "🤟",
                "title": f"{username} detected '{sign}'",
                "description": f"Confidence: {_format_confidence(conf)}",
                "time_ago": _format_time_ago(elapsed),
            }
        )

    # ─── SYSTEM SERVICES ───
    system_services = [
        {"name": "Web Server (Uvicorn)", "status": "online"},
        {"name": "PostgreSQL Database", "status": await _check_db(db)},
        {"name": "ML Model", "status": _check_ml_model()},
        {"name": "WebSocket Server", "status": "online"},
        {"name": "File Storage", "status": "online"},
    ]
    online_count = sum(1 for s in system_services if s["status"] == "online")
    system_health = round((online_count / len(system_services)) * 100)

    return {
        "data": {
            "stats": {
                "total_users": total_users,
                "active_users_today": active_today,
                "total_detections": total_detections,
                "system_health": system_health,
                "user_growth_percent": user_growth,
                "detection_growth_percent": detection_growth,
            },
            "recent_activities": recent_activities,
            "system_services": system_services,
            "top_users": top_users,
        }
    }


@router.get("/export", dependencies=[Depends(get_current_superuser)])
async def export_admin_report(
    db: Annotated[AsyncSession, Depends(async_get_db)],
):
    export_stmt = (
        select(
            User.id,
            User.username,
            User.email,
            User.created_at,
            User.last_login_at,
            User.is_active,
            sa_func.count(SignDetection.id).label("total_detections"),
        )
        .outerjoin(SignDetection, SignDetection.user_id == User.id)
        .where(~User.is_deleted)  # ✅ FIX: Use ~ instead of not
        .group_by(User.id)
        .order_by(User.created_at.desc())
    )

    result = await db.execute(export_stmt)
    rows = result.all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "ID",
            "Username",
            "Email",
            "Joined",
            "Last Login",
            "Active",
            "Detections",
        ]
    )

    for row in rows:
        writer.writerow(
            [
                row.id,
                row.username,
                row.email,
                row.created_at.strftime("%Y-%m-%d") if row.created_at else "",
                row.last_login_at.strftime("%Y-%m-%d %H:%M") if row.last_login_at else "Never",
                "Yes" if row.is_active else "No",
                row.total_detections,
            ]
        )

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=admin-report.csv"},
    )
