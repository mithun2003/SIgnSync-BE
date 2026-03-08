# app/api/v1/admin/__init__.py
from fastapi import APIRouter

from .analytics import router as analytics_router
from .dashboard import router as dashboard_router
from .settings import router as settings_router
from .signs import router as signs_router
from .system import router as system_router
from .users import router as admin_users_router

router = APIRouter(prefix="/v1/admin", tags=["Admin"])

router.include_router(admin_users_router)
router.include_router(signs_router)
router.include_router(dashboard_router)
router.include_router(system_router)
router.include_router(analytics_router)
router.include_router(settings_router, prefix="/settings")
