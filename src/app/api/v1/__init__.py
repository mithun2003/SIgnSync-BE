from fastapi import APIRouter

from .admin.users import router as admin_users_router
from .auth import router as auth_router
from .health import router as health_router
from .login import router as login_router
from .logout import router as logout_router
from .posts import router as posts_router
from .predict import router as predict_router
from .rate_limits import router as rate_limits_router
from .tasks import router as tasks_router
from .tiers import router as tiers_router
from .users import router as users_router

router = APIRouter(prefix="/v1")
router.include_router(health_router)
router.include_router(login_router)
router.include_router(logout_router)
router.include_router(users_router)
router.include_router(posts_router)
router.include_router(predict_router)
router.include_router(tasks_router)
router.include_router(tiers_router)
router.include_router(rate_limits_router)

router.include_router(auth_router)


router.include_router(admin_users_router)
