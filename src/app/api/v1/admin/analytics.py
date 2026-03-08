from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import Date, cast, select
from sqlalchemy import func as sa_func
from sqlalchemy.ext.asyncio import AsyncSession

from ....api.dependencies import get_current_superuser
from ....core.db.database import async_get_db
from ....models.user import User
from ....schemas.admin import (
    CountryCount,
    DailyCount,
    UserAnalyticsData,
    UserAnalyticsResponse,
)

router = APIRouter(prefix="/analytics", tags=["Admin Analytics"])


@router.get(
    "/users",
    response_model=UserAnalyticsResponse,
    dependencies=[Depends(get_current_superuser)],
)
async def get_user_analytics(
    db: Annotated[AsyncSession, Depends(async_get_db)],
    period: int = 30,
) -> dict:
    """User registration and activity analytics.

    Query parameters:
    - **period**: number of past days to analyse (default 30, max 365)

    Returns:
    - Total user count
    - New registrations in the period, broken down by day
    - Active vs inactive user counts
    - Growth percentage vs the previous equal-length period
    - Top 10 countries by user count
    """
    period = min(period, 365)
    now = datetime.now(UTC)
    period_start = now - timedelta(days=period)
    prev_period_start = now - timedelta(days=period * 2)

    # ── Totals ──────────────────────────────────────────────
    total_users: int = (await db.scalar(select(sa_func.count(User.id)).where(~User.is_deleted))) or 0

    active_users: int = (
        await db.scalar(select(sa_func.count(User.id)).where(~User.is_deleted, User.is_active.is_(True)))
    ) or 0

    # ── New users this period vs previous period ─────────────
    new_this_period: int = (
        await db.scalar(select(sa_func.count(User.id)).where(~User.is_deleted, User.created_at >= period_start))
    ) or 0

    new_prev_period: int = (
        await db.scalar(
            select(sa_func.count(User.id)).where(
                ~User.is_deleted,
                User.created_at >= prev_period_start,
                User.created_at < period_start,
            )
        )
    ) or 0

    if new_prev_period == 0:
        growth_percent = 100.0 if new_this_period > 0 else 0.0
    else:
        growth_percent = round(((new_this_period - new_prev_period) / new_prev_period) * 100, 1)

    # ── Daily registrations breakdown ────────────────────────
    daily_stmt = (
        select(
            cast(User.created_at, Date).label("reg_date"),
            sa_func.count(User.id).label("count"),
        )
        .where(~User.is_deleted, User.created_at >= period_start)
        .group_by(cast(User.created_at, Date))
        .order_by(cast(User.created_at, Date))
    )
    daily_result = await db.execute(daily_stmt)
    daily_registrations = [DailyCount(date=str(row.reg_date), count=row.count) for row in daily_result.all()]

    # ── Top countries ─────────────────────────────────────────
    country_stmt = (
        select(
            User.country.label("country"),
            sa_func.count(User.id).label("count"),
        )
        .where(~User.is_deleted, User.country.isnot(None))
        .group_by(User.country)
        .order_by(sa_func.count(User.id).desc())
        .limit(10)
    )
    country_result = await db.execute(country_stmt)
    top_countries = [CountryCount(country=row.country or "Unknown", count=row.count) for row in country_result.all()]

    return {
        "data": UserAnalyticsData(
            period_days=period,
            total_users=total_users,
            new_users_in_period=new_this_period,
            active_users_in_period=active_users,
            inactive_users=total_users - active_users,
            growth_percent=growth_percent,
            daily_registrations=daily_registrations,
            top_countries=top_countries,
        )
    }
