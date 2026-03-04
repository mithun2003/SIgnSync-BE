from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.api.dependencies import get_current_user
from src.app.core.db.database import async_get_db
from src.app.core.schemas import CommonResponse
from src.app.models.user import User  # your user model
from src.app.services.dashboard_service import DashboardService

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/", response_model=CommonResponse)
async def get_dashboard(
    days: int = Query(7, ge=1, le=90),
    db: AsyncSession = Depends(async_get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get complete dashboard analytics for logged-in user.
    """

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
