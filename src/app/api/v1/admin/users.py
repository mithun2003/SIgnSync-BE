from typing import Annotated, Any

from fastapi import APIRouter, Depends
from sqlalchemy import func as sa_func
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ....api.dependencies import get_current_superuser, get_current_user
from ....core.db.database import async_get_db
from ....core.exceptions.http_exceptions import (
    DuplicateValueException,
    ForbiddenException,
    NotFoundException,
)
from ....core.security import blacklist_token, get_password_hash, oauth2_scheme
from ....crud.crud_rate_limit import crud_rate_limits
from ....crud.crud_tier import crud_tiers
from ....crud.crud_users import crud_users
from ....models.user import User
from ....schemas.tier import TierRead
from ....schemas.user import (
    UserCreate,
    UserCreateInternal,
    UserRead,
    UserResponse,
    UserTierUpdate,
    UserUpdate,
)

router = APIRouter(tags=["users"])


@router.post("/auth/user", response_model=UserRead, status_code=201, dependencies=[Depends(get_current_superuser)])
async def write_user(
    user: UserCreate,
    db: Annotated[AsyncSession, Depends(async_get_db)],
) -> dict[str, Any]:
    email_row = await crud_users.exists(db=db, email=user.email)
    if email_row:
        raise DuplicateValueException("Email is already registered")

    username_row = await crud_users.exists(db=db, username=user.username)
    if username_row:
        raise DuplicateValueException("Username not available")

    user_internal_dict = user.model_dump()
    user_internal_dict["hashed_password"] = get_password_hash(password=user_internal_dict["password"])
    del user_internal_dict["password"]

    user_internal = UserCreateInternal(**user_internal_dict)
    created_user = await crud_users.create(db=db, object=user_internal, schema_to_select=UserRead)

    if created_user is None:
        raise NotFoundException("Failed to create user")

    return created_user


@router.get("/users", dependencies=[Depends(get_current_superuser)])
async def read_users(
    db: Annotated[AsyncSession, Depends(async_get_db)],
    search: str = "",
    status: str = "all",
    page: int = 1,
    limit: int = 10,
) -> dict[str, Any]:
    """List users with optional server-side search and status filtering."""
    # ── Build filter conditions ────────────────────────────────────────────────
    conditions: list = [User.is_deleted == False]  # noqa: E712

    if status == "active":
        conditions.append(User.is_active == True)  # noqa: E712
    elif status in ("inactive", "suspended"):
        conditions.append(User.is_active == False)  # noqa: E712

    if search:
        term = f"%{search.lower()}%"
        conditions.append(
            or_(
                sa_func.lower(User.username).like(term),
                sa_func.lower(User.email).like(term),
            )
        )

    # ── Summary counts (always unfiltered, for stats cards) ───────────────────
    total_all = (await db.execute(select(sa_func.count()).where(User.is_deleted == False))).scalar() or 0  # noqa: E712
    total_active = (await db.execute(select(sa_func.count()).where(not User.is_deleted, User.is_active))).scalar() or 0
    total_inactive = (
        await db.execute(select(sa_func.count()).where(not User.is_deleted, not User.is_active))
    ).scalar() or 0

    # ── Filtered count ─────────────────────────────────────────────────────────
    total_filtered = (await db.execute(select(sa_func.count()).where(*conditions))).scalar() or 0

    # ── Paginated data ─────────────────────────────────────────────────────────
    offset = (page - 1) * limit
    result = await db.execute(
        select(User).where(*conditions).order_by(User.created_at.desc()).offset(offset).limit(limit)
    )
    rows = result.scalars().all()

    user_list = [
        {
            "id": u.id,
            "username": u.username,
            "email": u.email,
            "role": "admin" if u.is_superuser else "user",
            "status": "active" if u.is_active else "inactive",
            "joined_at": u.created_at.isoformat() if u.created_at else None,
            "last_seen": u.updated_at.isoformat() if u.updated_at else None,
            "total_signs": 0,
        }
        for u in rows
    ]

    return {
        "data": user_list,
        "total_count": total_filtered,
        "page": page,
        "items_per_page": limit,
        "has_more": (page * limit) < total_filtered,
        "summary": {
            "total": total_all,
            "active": total_active,
            "inactive": total_inactive,
            "suspended": 0,
        },
    }


@router.get("/user/me/", response_model=UserResponse)
async def read_users_me(current_user: Annotated[dict, Depends(get_current_user)]) -> dict:
    return UserResponse(data=current_user)


@router.get("/user/{username}", response_model=UserRead)
async def read_user(username: str, db: Annotated[AsyncSession, Depends(async_get_db)]) -> dict[str, Any]:
    db_user = await crud_users.get(db=db, username=username, is_deleted=False, schema_to_select=UserRead)
    if db_user is None:
        raise NotFoundException("User not found")

    return db_user


@router.patch("/user/{username}")
async def patch_user(
    values: UserUpdate,
    username: str,
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(async_get_db)],
) -> dict[str, str]:
    db_user = await crud_users.get(db=db, username=username)
    if db_user is None:
        raise NotFoundException("User not found")

    db_username = db_user["username"]
    db_email = db_user["email"]

    if db_username != current_user["username"]:
        raise ForbiddenException()

    if values.email is not None and values.email != db_email:
        if await crud_users.exists(db=db, email=values.email):
            raise DuplicateValueException("Email is already registered")

    if values.username is not None and values.username != db_username:
        if await crud_users.exists(db=db, username=values.username):
            raise DuplicateValueException("Username not available")

    await crud_users.update(db=db, object=values, username=username)
    return {"message": "User updated"}


@router.delete("/user/{username}")
async def erase_user(
    username: str,
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(async_get_db)],
    token: str = Depends(oauth2_scheme),
) -> dict[str, str]:
    db_user = await crud_users.get(db=db, username=username, schema_to_select=UserRead)
    if not db_user:
        raise NotFoundException("User not found")

    if username != current_user["username"]:
        raise ForbiddenException()

    await crud_users.delete(db=db, username=username)
    await blacklist_token(token=token, db=db)
    return {"message": "User deleted"}


@router.delete("/db_user/{username}", dependencies=[Depends(get_current_superuser)])
async def erase_db_user(
    username: str,
    db: Annotated[AsyncSession, Depends(async_get_db)],
    token: str = Depends(oauth2_scheme),
) -> dict[str, str]:
    db_user = await crud_users.exists(db=db, username=username)
    if not db_user:
        raise NotFoundException("User not found")

    await crud_users.db_delete(db=db, username=username)
    await blacklist_token(token=token, db=db)
    return {"message": "User deleted from the database"}


@router.get("/user/{username}/rate_limits", dependencies=[Depends(get_current_superuser)])
async def read_user_rate_limits(username: str, db: Annotated[AsyncSession, Depends(async_get_db)]) -> dict[str, Any]:
    db_user = await crud_users.get(db=db, username=username, schema_to_select=UserRead)
    if db_user is None:
        raise NotFoundException("User not found")

    user_dict = dict(db_user)
    if db_user["tier_id"] is None:
        user_dict["tier_rate_limits"] = []
        return user_dict

    db_tier = await crud_tiers.get(db=db, id=db_user["tier_id"], schema_to_select=TierRead)
    if db_tier is None:
        raise NotFoundException("Tier not found")

    db_rate_limits = await crud_rate_limits.get_multi(db=db, tier_id=db_tier["id"])

    user_dict["tier_rate_limits"] = db_rate_limits["data"]

    return user_dict


@router.get("/user/{username}/tier")
async def read_user_tier(username: str, db: Annotated[AsyncSession, Depends(async_get_db)]) -> dict | None:
    db_user = await crud_users.get(db=db, username=username, schema_to_select=UserRead)
    if db_user is None:
        raise NotFoundException("User not found")

    if db_user["tier_id"] is None:
        return None

    db_tier = await crud_tiers.get(db=db, id=db_user["tier_id"], schema_to_select=TierRead)
    if not db_tier:
        raise NotFoundException("Tier not found")

    user_dict = dict(db_user)
    tier_dict = dict(db_tier)

    for key, value in tier_dict.items():
        user_dict[f"tier_{key}"] = value

    return user_dict


@router.patch("/user/{username}/tier", dependencies=[Depends(get_current_superuser)])
async def patch_user_tier(
    username: str,
    values: UserTierUpdate,
    db: Annotated[AsyncSession, Depends(async_get_db)],
) -> dict[str, str]:
    db_user = await crud_users.get(db=db, username=username, schema_to_select=UserRead)
    if db_user is None:
        raise NotFoundException("User not found")

    db_tier = await crud_tiers.get(db=db, id=values.tier_id, schema_to_select=TierRead)
    if db_tier is None:
        raise NotFoundException("Tier not found")

    await crud_users.update(db=db, object=values.model_dump(), username=username)
    return {"message": f"User {db_user['name']} Tier updated"}
