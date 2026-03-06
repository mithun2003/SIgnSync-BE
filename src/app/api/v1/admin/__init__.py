# app/api/v1/admin/__init__.py
from fastapi import APIRouter

from .dashboard import router as dashboard_router
from .signs import router as signs_router
from .users import router as admin_users_router

router = APIRouter(prefix="/v1/admin", tags=["Admin"])

router.include_router(admin_users_router)
router.include_router(signs_router)
router.include_router(dashboard_router)
