import json
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends
from redis.asyncio import Redis
from sqlalchemy import func as sa_func
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ....api.dependencies import get_current_superuser
from ....core.db.database import async_get_db
from ....core.health import check_database_health, check_redis_health
from ....core.ml.predict import get_health_status
from ....core.utils.cache import async_get_redis
from ....models.sign_detection import SignDetection
from ....models.user import User
from ....schemas.admin import (
    ActiveUserItem,
    ActiveUsersResponse,
    BackupInfoResponse,
    BackupRecord,
    CacheClearResponse,
    ServiceHealth,
    SystemHealthDetail,
    SystemHealthResponse,
)

router = APIRouter(prefix="/system", tags=["Admin System"])

BACKUP_DIR = Path("/app/backups")
BACKUP_META_FILE = BACKUP_DIR / ".meta.json"


def _load_backup_meta() -> dict | None:
    """Read last backup metadata from disk."""
    if BACKUP_META_FILE.exists():
        try:
            return json.loads(BACKUP_META_FILE.read_text())
        except Exception:
            return None
    return None


def _save_backup_meta(meta: dict) -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_META_FILE.write_text(json.dumps(meta, indent=2))


# ─── System Health ────────────────────────────────────────────────────────────


@router.get(
    "/health",
    response_model=SystemHealthResponse,
    dependencies=[Depends(get_current_superuser)],
)
async def get_system_health(
    db: Annotated[AsyncSession, Depends(async_get_db)],
    redis: Annotated[Redis | None, Depends(async_get_redis)],
) -> dict:
    """Detailed health status for every infrastructure service.

    Measures response latency for DB and Redis, reports ML model state,
    and returns an aggregated overall status.
    Redis is shown as "disabled" when ``REDIS_CACHE_ENABLED=false``.
    """
    services: list[ServiceHealth] = []

    # PostgreSQL
    t0 = time.monotonic()
    db_ok = await check_database_health(db)
    db_latency = round((time.monotonic() - t0) * 1000, 2)
    services.append(
        ServiceHealth(
            name="PostgreSQL Database",
            status="online" if db_ok else "offline",
            latency_ms=db_latency,
        )
    )

    # Redis — skip check entirely when the pool is not initialised
    t0 = time.monotonic()
    redis_result = await check_redis_health(redis)
    redis_latency = round((time.monotonic() - t0) * 1000, 2)
    if redis_result is None:
        redis_status = "disabled"
        redis_latency = None
    else:
        redis_status = "online" if redis_result else "offline"
    services.append(
        ServiceHealth(
            name="Redis Cache",
            status=redis_status,
            latency_ms=redis_latency,
        )
    )

    # ML Model
    try:
        health = get_health_status()
        ml_status = "online" if health.get("status") == "healthy" else "warning"
    except Exception:
        ml_status = "offline"
    services.append(ServiceHealth(name="ML Model (MobileNet)", status=ml_status))

    # WebSocket / Uvicorn (always online if this endpoint responds)
    services.append(ServiceHealth(name="WebSocket Server", status="online"))

    # "disabled" services are neutral — don't count them as offline
    active_services = [s for s in services if s.status != "disabled"]
    overall = "online" if all(s.status == "online" for s in active_services) else "degraded"
    if any(s.status == "offline" for s in active_services):
        overall = "offline"

    return {
        "data": SystemHealthDetail(
            status=overall,
            timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
            services=services,
        )
    }


# ─── Active Users ─────────────────────────────────────────────────────────────


@router.get(
    "/active-users",
    response_model=ActiveUsersResponse,
    dependencies=[Depends(get_current_superuser)],
)
async def get_active_users(
    db: Annotated[AsyncSession, Depends(async_get_db)],
    hours: int = 24,
    limit: int = 50,
) -> dict:
    """List users who have logged in within the last N hours.

    Query parameters:
    - **hours**: look-back window (default 24, max 720)
    - **limit**: max users returned (default 50, max 200)
    """
    hours = min(hours, 720)
    limit = min(limit, 200)
    cutoff = datetime.now(UTC) - timedelta(hours=hours)

    stmt = (
        select(User.username, User.email, User.last_login_at, User.created_at)
        .where(~User.is_deleted, User.last_login_at >= cutoff)
        .order_by(User.last_login_at.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    rows = result.all()

    users = [
        ActiveUserItem(
            username=row.username,
            email=row.email,
            last_login_at=row.last_login_at.isoformat() if row.last_login_at else None,
            created_at=row.created_at.isoformat() if row.created_at else "",
        )
        for row in rows
    ]

    return ActiveUsersResponse(data=users, total=len(users), period_hours=hours)


# ─── Cache ────────────────────────────────────────────────────────────────────


@router.post(
    "/cache/clear",
    response_model=CacheClearResponse,
    dependencies=[Depends(get_current_superuser)],
)
async def clear_cache(
    redis: Annotated[Redis | None, Depends(async_get_redis)],
) -> CacheClearResponse:
    """Flush all keys from the Redis application cache.

    Uses SCAN to iterate and delete in batches — safe for production use. Does not affect the rate-limit or queue Redis
    instances. Returns zero deleted keys when Redis is disabled.
    """
    if redis is None:
        return CacheClearResponse(
            message="Redis cache is disabled — nothing to clear",
            cleared_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
    cursor = 0
    deleted = 0
    while True:
        cursor, keys = await redis.scan(cursor, match="*", count=200)
        if keys:
            await redis.delete(*keys)
            deleted += len(keys)
        if cursor == 0:
            break

    return CacheClearResponse(
        message=f"Cache cleared — {deleted} key(s) removed",
        cleared_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


# ─── Backup ───────────────────────────────────────────────────────────────────


@router.post(
    "/backup",
    response_model=BackupInfoResponse,
    dependencies=[Depends(get_current_superuser)],
)
async def create_backup(
    db: Annotated[AsyncSession, Depends(async_get_db)],
) -> dict:
    """Create a JSON snapshot of all active users and their detection counts.

    The snapshot is written to ``/app/backups/`` and metadata is persisted
    in ``.meta.json`` for retrieval by the backup info endpoint.
    """

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    # Query users with their detection counts
    stmt = (
        select(
            User.id,
            User.username,
            User.email,
            User.first_name,
            User.last_name,
            User.is_active,
            User.is_superuser,
            User.created_at,
            User.last_login_at,
            sa_func.count(SignDetection.id).label("detection_count"),
        )
        .outerjoin(SignDetection, SignDetection.user_id == User.id)
        .where(~User.is_deleted)
        .group_by(User.id)
        .order_by(User.created_at)
    )
    result = await db.execute(stmt)
    rows = result.all()

    user_records = [
        {
            "id": row.id,
            "username": row.username,
            "email": row.email,
            "first_name": row.first_name,
            "last_name": row.last_name,
            "is_active": row.is_active,
            "is_superuser": row.is_superuser,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "last_login_at": row.last_login_at.isoformat() if row.last_login_at else None,
            "detection_count": row.detection_count,
        }
        for row in rows
    ]

    backup_id = f"backup_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    payload = {
        "backup_id": backup_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "users": user_records,
    }

    file_path = BACKUP_DIR / f"{backup_id}.json"
    file_path.write_text(json.dumps(payload, indent=2))
    size_bytes = file_path.stat().st_size

    meta = {
        "backup_id": backup_id,
        "created_at": payload["created_at"],
        "size_bytes": size_bytes,
        "record_counts": {"users": len(user_records)},
        "file_path": str(file_path),
    }
    _save_backup_meta(meta)

    return {
        "last_backup": BackupRecord(**meta),
        "message": f"Backup created — {len(user_records)} user(s) exported",
    }


@router.get(
    "/backup/info",
    response_model=BackupInfoResponse,
    dependencies=[Depends(get_current_superuser)],
)
async def get_backup_info() -> dict:
    """Return metadata about the most recent backup.

    The backup file itself is stored on disk at the path shown in ``file_path``.
    Returns ``null`` for ``last_backup`` if no backup has been created yet.
    """
    meta = _load_backup_meta()
    if meta is None:
        return {"last_backup": None, "message": "No backups have been created yet"}

    return {
        "last_backup": BackupRecord(**meta),
        "message": "Last backup info retrieved successfully",
    }
