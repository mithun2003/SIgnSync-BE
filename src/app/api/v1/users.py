from typing import Annotated

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.db.database import async_get_db
from ...core.exceptions.http_exceptions import DuplicateValueException, NotFoundException
from ...crud.crud_users import crud_users
from ...schemas.user import UserRead, UserResponse, UserUpdate
from ..dependencies import get_current_user

router = APIRouter(prefix="/user", tags=["users"])


@router.get("/me", response_model=UserResponse)
async def read_users_me(
    request: Request,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    return UserResponse(data=current_user)


@router.patch("/me", response_model=UserResponse)
async def update_current_user(
    values: UserUpdate,
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(async_get_db)],
):
    db_user = await crud_users.get(db=db, id=current_user["id"])
    if not db_user:
        raise NotFoundException("User not found")

    # Check duplicate email
    if values.email and values.email != db_user["email"]:
        if await crud_users.exists(db=db, email=values.email):
            raise DuplicateValueException("Email already registered")

    # Check duplicate username
    if values.username and values.username != db_user["username"]:
        if await crud_users.exists(db=db, username=values.username):
            raise DuplicateValueException("Username not available")

    await crud_users.update(db=db, object=values, id=current_user["id"])

    updated_user = await crud_users.get(db=db, id=current_user["id"], schema_to_select=UserRead)

    return updated_user


@router.delete("/me")
async def delete_me(
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(async_get_db)],
):
    await crud_users.delete(
        db=db,
        username=current_user["username"],
    )

    return {"message": "Account deleted"}
