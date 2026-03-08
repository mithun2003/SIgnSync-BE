import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.db.database import async_get_db
from ...core.schemas import CommonResponse
from ...services.dashboard_service import DashboardService
from ..dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/", response_model=CommonResponse)
async def get_dashboard(
    days: Annotated[int, Query(ge=1, le=90, description="Days of history to analyze")] = 7,
    db: Annotated[AsyncSession, Depends(async_get_db)] = None,
    current_user: Annotated[dict, Depends(get_current_user)] = None,
):
    """Get complete dashboard analytics for the logged-in user."""
    try:
        service = DashboardService(db)
        dashboard_data = await service.get_dashboard(
            user_id=current_user["id"],
            days=days,
        )
        return CommonResponse(
            status_code=200,
            message="Dashboard fetched successfully",
            data=dashboard_data,
        )
    except Exception as e:
        logger.exception("Dashboard fetch failed for user_id=%s: %s", current_user["id"], e)
        raise HTTPException(status_code=500, detail="Failed to fetch dashboard data")
