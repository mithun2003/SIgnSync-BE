# app/api/v1/users.py — UPDATED with profile image upload
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
)
from ...crud.crud_users import crud_users
from ...schemas.user import UserRead, UserResponse, UserUpdate
from ..dependencies import get_current_user

router = APIRouter(prefix="/user", tags=["users"])

# ✅ Directory where profile images will be stored
MEDIA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "media", "profile_images")

# ✅ Allowed image extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB


def _delete_old_image(profile_image_url: str | None) -> None:
    """Helper to delete an old profile image file from disk."""
    if not profile_image_url or not profile_image_url.startswith("/media/"):
        return
    # /media/profile_images/xxx.jpg → media/profile_images/xxx.jpg
    relative_path = profile_image_url.lstrip("/")
    # Resolve from /app/src/ (3 levels up from this file's dir)
    base_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    old_path = os.path.normpath(os.path.join(base_dir, relative_path))
    if os.path.exists(old_path):
        os.remove(old_path)


@router.get("/me", response_model=UserResponse)
async def read_users_me(
    request: Request,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    print(current_user)
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
    """
    Upload a profile image for the current user.
    - Accepts: .jpg, .jpeg, .png, .gif, .webp
    - Max size: 5MB
    - Stores in: media/profile_images/
    - Auto-deletes previous image
    """
    # 1. Validate filename
    if not file.filename:
        raise ForbiddenException("No filename provided")
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise ForbiddenException(f"File type '{file_ext}' not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    # 2. Validate file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise ForbiddenException(f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)}MB")
    await file.seek(0)
    # 3. Create directory if needed
    os.makedirs(MEDIA_DIR, exist_ok=True)
    # 4. Generate unique filename
    unique_filename = f"{uuid.uuid4().hex[:12]}{file_ext}"
    file_path = os.path.join(MEDIA_DIR, unique_filename)
    # 5. Delete old image
    db_user = await crud_users.get(db=db, id=current_user["id"])
    if db_user:
        _delete_old_image(db_user.get("profile_image_url"))
    # 6. Save new file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # 7. Update DB with new URL
    image_url = f"/media/profile_images/{unique_filename}"
    update_data = UserUpdate(profile_image_url=image_url)
    await crud_users.update(db=db, object=update_data, id=current_user["id"])
    # 8. Return updated user
    updated_user = await crud_users.get(db=db, id=current_user["id"], schema_to_select=UserRead)
    return UserResponse(data=updated_user)


# Delete profile image
@router.delete("/me/profile-image", response_model=UserResponse)
async def delete_profile_image(
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(async_get_db)],
):
    """Remove the current user's profile image."""
    db_user = await crud_users.get(db=db, id=current_user["id"])
    if not db_user:
        raise NotFoundException("User not found")
    # Delete file from disk
    _delete_old_image(db_user.get("profile_image_url"))
    # Set to None in DB
    update_data = UserUpdate(profile_image_url=None)
    await crud_users.update(db=db, object=update_data, id=current_user["id"])
    updated_user = await crud_users.get(db=db, id=current_user["id"], schema_to_select=UserRead)
    return UserResponse(data=updated_user)


# ───────────────────────────────────────────────
# Delete account (updated to clean up image)
# ───────────────────────────────────────────────
@router.delete("/me")
async def delete_me(
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(async_get_db)],
):
    # Clean up profile image file before deleting account
    db_user = await crud_users.get(db=db, id=current_user["id"])
    if db_user:
        _delete_old_image(db_user.get("profile_image_url"))
    await crud_users.delete(db=db, username=current_user["username"])
    return {"message": "Account deleted"}
