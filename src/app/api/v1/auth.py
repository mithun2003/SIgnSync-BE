# api/v1/auth.py

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.db.database import async_get_db
from ...core.exceptions.http_exceptions import DuplicateValueException, NotFoundException
from ...core.security import get_password_hash
from ...crud.crud_users import crud_users
from ...schemas.user import UserCreate, UserCreateInternal, UserRead

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserRead, status_code=201)
async def register_user(
    user: UserCreate,
    db: Annotated[AsyncSession, Depends(async_get_db)],
) -> dict[str, Any]:
    if await crud_users.exists(db=db, email=user.email):
        raise DuplicateValueException("Email already registered")

    if await crud_users.exists(db=db, username=user.username):
        raise DuplicateValueException("Username not available")

    user_dict = user.model_dump()
    user_dict["hashed_password"] = get_password_hash(user_dict["password"])
    del user_dict["password"]

    user_internal = UserCreateInternal(**user_dict)

    created_user = await crud_users.create(
        db=db,
        object=user_internal,
        schema_to_select=UserRead,
    )

    if not created_user:
        raise NotFoundException("User creation failed")

    return created_user
