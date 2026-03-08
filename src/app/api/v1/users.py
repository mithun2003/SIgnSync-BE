# app/api/v1/users.py
import logging
import os
import shutil
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Request, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.db.database import async_get_db
from ...core.exceptions.http_exceptions import (
    DuplicateValueException,
    ForbiddenException,
    NotFoundException,
    UnauthorizedException,
)
from ...core.security import get_password_hash, verify_password
from ...crud.crud_users import crud_users
from ...schemas.user import PasswordChange, UserRead, UserResponse, UserUpdate
from ..dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/user", tags=["users"])

MEDIA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "media", "profile_images")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB


def _delete_old_image(profile_image_url: str | None) -> None:
    """Helper to delete an old profile image file from disk."""
    if not profile_image_url or not profile_image_url.startswith("/media/"):
        return
    relative_path = profile_image_url.lstrip("/")
    base_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    old_path = os.path.normpath(os.path.join(base_dir, relative_path))
    if os.path.exists(old_path):
        os.remove(old_path)


@router.get("/me", response_model=UserResponse)
async def read_users_me(
    request: Request,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Return the currently authenticated user's profile."""
    logger.debug("User profile requested for user_id=%s", current_user.get("id"))
    return UserResponse(data=current_user)


@router.patch("/me", response_model=UserResponse)
async def update_current_user(
    values: UserUpdate,
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(async_get_db)],
):
    """Update the current user's profile fields."""
    db_user = await crud_users.get(db=db, id=current_user["id"])
    if not db_user:
        raise NotFoundException("User not found")
    if values.email and values.email != db_user["email"]:
        if await crud_users.exists(db=db, email=values.email):
            raise DuplicateValueException("Email already registered")
    if values.username and values.username != db_user["username"]:
        if await crud_users.exists(db=db, username=values.username):
            raise DuplicateValueException("Username not available")
    await crud_users.update(db=db, object=values, id=current_user["id"])
    updated_user = await crud_users.get(db=db, id=current_user["id"], schema_to_select=UserRead)
    return UserResponse(data=updated_user)


# ───────────────────────────────────────────────
# Profile image upload
# ───────────────────────────────────────────────
@router.post("/me/profile-image", response_model=UserResponse)
async def upload_profile_image(
    file: Annotated[UploadFile, File(description="Profile image file")],
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(async_get_db)],
):
    """Upload a profile image for the current user.

    - Accepts: .jpg, .jpeg, .png, .gif, .webp
    - Max size: 5MB
    - Stores in: media/profile_images/
    - Auto-deletes previous image
    """
    if not file.filename:
        raise ForbiddenException("No filename provided")
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise ForbiddenException(f"File type '{file_ext}' not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise ForbiddenException(f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)}MB")
    await file.seek(0)

    os.makedirs(MEDIA_DIR, exist_ok=True)

    unique_filename = f"{uuid.uuid4().hex[:12]}{file_ext}"
    file_path = os.path.join(MEDIA_DIR, unique_filename)

    db_user = await crud_users.get(db=db, id=current_user["id"])
    if db_user:
        _delete_old_image(db_user.get("profile_image_url"))

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image_url = f"/media/profile_images/{unique_filename}"
    update_data = UserUpdate(profile_image_url=image_url)
    await crud_users.update(db=db, object=update_data, id=current_user["id"])

    updated_user = await crud_users.get(db=db, id=current_user["id"], schema_to_select=UserRead)
    logger.info("Profile image updated for user_id=%s", current_user["id"])
    return UserResponse(data=updated_user)


@router.delete("/me/profile-image", response_model=UserResponse)
async def delete_profile_image(
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(async_get_db)],
):
    """Remove the current user's profile image."""
    db_user = await crud_users.get(db=db, id=current_user["id"])
    if not db_user:
        raise NotFoundException("User not found")
    _delete_old_image(db_user.get("profile_image_url"))
    update_data = UserUpdate(profile_image_url=None)
    await crud_users.update(db=db, object=update_data, id=current_user["id"])
    updated_user = await crud_users.get(db=db, id=current_user["id"], schema_to_select=UserRead)
    return UserResponse(data=updated_user)


# ───────────────────────────────────────────────
# Delete account
# ───────────────────────────────────────────────
@router.delete("/me")
async def delete_me(
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(async_get_db)],
):
    """Soft-delete the current user's account and clean up their profile image."""
    db_user = await crud_users.get(db=db, id=current_user["id"])
    if db_user:
        _delete_old_image(db_user.get("profile_image_url"))
    await crud_users.delete(db=db, username=current_user["username"])
    return {"message": "Account deleted"}


# ───────────────────────────────────────────────
# Change password
# ───────────────────────────────────────────────
@router.post("/me/change-password")
async def change_password(
    payload: PasswordChange,
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(async_get_db)],
) -> dict[str, str]:
    """Change the authenticated user's password.

    Verifies the current password before applying the new one.
    """
    if not await verify_password(payload.current_password, current_user["hashed_password"]):
        raise UnauthorizedException("Current password is incorrect")

    new_hash = get_password_hash(payload.new_password)
    await crud_users.update(
        db=db,
        object={"hashed_password": new_hash},
        id=current_user["id"],
    )
    return {"message": "Password changed successfully"}
