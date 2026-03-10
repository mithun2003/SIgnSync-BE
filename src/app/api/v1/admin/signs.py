import logging
import os
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from ....api.dependencies import get_current_superuser
from ....core.db.database import async_get_db
from ....core.utils.cloudinary import delete_image, get_image_details, list_sign_images, upload_sign_image
from ....crud.crud_signs import crud_signs
from ....schemas.signs import (
    BulkUploadResult,
    CloudinaryImageInfo,
    SetActiveImageRequest,
    SignCreate,
    SignRead,
    SignUpdate,
    UpdateURLRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/signs", tags=["Admin ASL Signs"])

# A-Z, SPACE, DEL, plus the emergency/custom signs supported by the ML model.
# Characters are stored and looked up in uppercase.
ALLOWED_CHARACTERS: list[str] = (
    [chr(i) for i in range(65, 91)] + ["SPACE", "DEL"] + ["HELP", "DANGER", "EMERGENCY", "THUMBS_DOWN", "OK_SIGN"]
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _char_from_filename(filename: str) -> str | None:
    """Derive the ASL character from an uploaded filename.

    A.jpg  → "A"   b.PNG  → "B"   space.jpg → "SPACE"   SPACE.png → "SPACE" Returns None if the stem cannot be mapped to
    a valid character.
    """
    stem = os.path.splitext(filename)[0].upper().strip()
    return stem if stem in ALLOWED_CHARACTERS else None


async def _upsert_sign(
    *,
    db: AsyncSession,
    character: str,
    url: str,
    public_id: str,
    file_size: int,
    width: int,
    height: int,
    mime_type: str,
    user_id: int,
    notes: str | None,
    version: int,
) -> SignRead:
    """Create or update the Signs record for *character* and return SignRead.

    The caller is responsible for supplying the correct *version* number:
    - For a new upload it should be ``len(existing_cloudinary_images) + 1``.
    - For set-active it should be the version parsed from the public_id.
    The version is stored as-is; this function never auto-increments it.
    """
    existing = await crud_signs.get(
        db=db,
        character=character,
        schema_to_select=SignRead,
        return_as_model=True,
    )
    if existing:
        updated = await crud_signs.update(
            db=db,
            object=SignUpdate(
                cloudinary_url=url,
                cloudinary_public_id=public_id,
                file_size=file_size,
                width=width,
                height=height,
                mime_type=mime_type,
                version=version,
                updated_by=user_id,
                notes=notes,
            ),
            id=existing.id,
            schema_to_select=SignRead,
            return_as_model=True,
        )
        return updated  # type: ignore[return-value]
    else:
        created = await crud_signs.create(
            db=db,
            object=SignCreate(
                character=character,
                cloudinary_url=url,
                cloudinary_public_id=public_id,
                file_size=file_size,
                width=width,
                height=height,
                mime_type=mime_type,
                version=version,
                updated_by=user_id,
                notes=notes,
            ),
            schema_to_select=SignRead,
            return_as_model=True,
        )
        return created  # type: ignore[return-value]


# ── GET /signs ────────────────────────────────────────────────────────────────


@router.get("", summary="List all signs (admin)")
async def get_all_signs(
    db: Annotated[AsyncSession, Depends(async_get_db)],
    current_user: Annotated[dict, Depends(get_current_superuser)],
):
    signs = await crud_signs.get_multi(db=db, schema_to_select=SignRead, return_as_model=True)
    data = signs.get("data", [])
    existing = {s.character for s in data}
    missing = [c for c in ALLOWED_CHARACTERS if c not in existing]
    return {
        "data": data,
        "total": signs.get("total_count", 0),
        "missing_characters": missing,
    }


# ── GET /signs/stats ──────────────────────────────────────────────────────────


@router.get("/stats", summary="Admin sign statistics")
async def get_statistics(
    db: Annotated[AsyncSession, Depends(async_get_db)],
    current_user: Annotated[dict, Depends(get_current_superuser)],
):
    signs = await crud_signs.get_multi(db=db, schema_to_select=SignRead, return_as_model=True)
    existing = {s.character for s in signs.get("data", [])}
    missing = [c for c in ALLOWED_CHARACTERS if c not in existing]
    total = signs.get("total_count", 0)
    return {
        "total_signs": total,
        "missing_signs": len(missing),
        "missing_characters": missing,
        "completion_percentage": round((total / len(ALLOWED_CHARACTERS)) * 100, 2),
    }


# ── GET /signs/{character}/images ─────────────────────────────────────────────


@router.get(
    "/{character}/images",
    response_model=list[CloudinaryImageInfo],
    summary="List all uploaded images (versions) for a character",
)
async def list_images_for_character(
    character: str,
    current_user: Annotated[dict, Depends(get_current_superuser)],
):
    """Returns every image stored in Cloudinary under  asl-signs/{CHARACTER}/.

    The admin can use these public_ids to switch the active image.
    """
    char = character.upper()
    if char not in ALLOWED_CHARACTERS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid character")

    images = list_sign_images(char)
    return images


# ── DELETE /signs/{character}/images ─────────────────────────────────────────


@router.delete(
    "/{character}/images",
    summary="Delete a specific image version for a character",
)
async def delete_sign_image(
    character: str,
    public_id: str,
    db: Annotated[AsyncSession, Depends(async_get_db)],
    current_user: Annotated[dict, Depends(get_current_superuser)],
):
    """Delete one image from Cloudinary by its ``public_id``.

    Pass the full Cloudinary public_id as a query parameter, e.g.
    ``?public_id=asl-signs/A/3``.

    - If the deleted image was the **active** image for the character, the
      latest remaining version is automatically promoted to active.
    - If no images remain after deletion, the database record for the
      character is also removed.
    """
    char = character.upper()
    if char not in ALLOWED_CHARACTERS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid character")

    try:
        found = delete_image(public_id)
    except Exception as exc:
        logger.error("Cloudinary delete failed for '%s': %s", public_id, exc)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Image deletion failed") from exc

    if not found:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found in Cloudinary")

    logger.info("Deleted Cloudinary image '%s' for character '%s'", public_id, char)

    # Check whether the deleted image was the currently active one
    current_sign = await crud_signs.get(db=db, character=char, schema_to_select=SignRead, return_as_model=True)
    if current_sign and current_sign.cloudinary_public_id == public_id:
        remaining = list_sign_images(char)
        if remaining:
            # Promote the latest remaining image
            latest = remaining[-1]
            await _upsert_sign(
                db=db,
                character=char,
                url=latest["url"],
                public_id=latest["public_id"],
                file_size=latest["file_size"],
                width=latest["width"],
                height=latest["height"],
                mime_type=latest["mime_type"],
                user_id=current_user["id"],
                notes=current_sign.notes,
                version=latest["version"] or current_sign.version,
            )
            logger.info("Auto-promoted '%s' as active for character '%s'", latest["public_id"], char)
        else:
            # No images left — remove the DB record entirely
            await crud_signs.delete(db=db, id=current_sign.id)
            logger.info("No images remaining for '%s'; database record removed", char)

    return {"detail": f"Image '{public_id}' deleted successfully"}


# ── POST /signs/{character}/upload ────────────────────────────────────────────


@router.post(
    "/{character}/upload",
    response_model=SignRead,
    status_code=status.HTTP_200_OK,
    summary="Upload a single image for a specific character",
)
async def upload_sign(
    character: str,
    db: Annotated[AsyncSession, Depends(async_get_db)],
    current_user: Annotated[dict, Depends(get_current_superuser)],
    file: UploadFile = File(...),
):
    """Upload one image for *character*.

    The image is stored in Cloudinary under asl-signs/{CHARACTER}/ and immediately becomes the active image for that
    character.
    """
    char = character.upper()
    if char not in ALLOWED_CHARACTERS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid character")

    # Next version = how many images already exist in Cloudinary for this character + 1
    existing_images = list_sign_images(char)
    next_version = len(existing_images) + 1

    try:
        result = upload_sign_image(await file.read(), char, next_version)
    except Exception as exc:
        logger.error("Cloudinary upload failed for %s: %s", char, exc)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Image upload failed") from exc

    sign = await _upsert_sign(
        db=db,
        character=char,
        url=result["url"],
        public_id=result["public_id"],
        file_size=result["file_size"],
        width=result["width"],
        height=result["height"],
        mime_type=result["mime_type"],
        user_id=current_user["id"],
        notes=None,
        version=next_version,
    )
    logger.info("Uploaded image for '%s' as version %d: %s", char, next_version, result["public_id"])
    return sign


# ── POST /signs/bulk-upload ───────────────────────────────────────────────────


@router.post(
    "/bulk-upload",
    response_model=BulkUploadResult,
    status_code=status.HTTP_200_OK,
    summary="Bulk upload: character is inferred from each filename (A.jpg → A)",
)
async def bulk_upload_signs(
    db: Annotated[AsyncSession, Depends(async_get_db)],
    current_user: Annotated[dict, Depends(get_current_superuser)],
    files: list[UploadFile] = File(...),
):
    """
    Upload many images at once.  The ASL character for each image is determined
    by its filename stem:

    - ``A.jpg``     → character **A**
    - ``space.png`` → character **SPACE**
    - ``b.webp``    → character **B**

    Files whose names cannot be mapped to a valid character (A-Z / SPACE) are
    returned in the ``skipped`` list so the admin knows what to rename and
    re-upload.

    Each successfully processed file immediately becomes the active image for
    its character.
    """
    uploaded: list[dict] = []
    skipped: list[dict] = []
    # Track how many images already exist per character so that multiple files
    # for the same character in one batch each get a unique, sequential version.
    version_offsets: dict[str, int] = {}

    for file in files:
        char = _char_from_filename(file.filename or "")
        if char is None:
            skipped.append({"filename": file.filename, "reason": "Cannot determine character from filename"})
            continue

        # Fetch the current Cloudinary count once per character per batch
        if char not in version_offsets:
            version_offsets[char] = len(list_sign_images(char))

        version_offsets[char] += 1
        next_version = version_offsets[char]

        try:
            result = upload_sign_image(await file.read(), char, next_version)
        except Exception as exc:
            logger.error("Bulk upload failed for file '%s': %s", file.filename, exc)
            # Roll back the offset so the next attempt reuses this version slot
            version_offsets[char] -= 1
            skipped.append({"filename": file.filename, "reason": "Upload to Cloudinary failed"})
            continue

        sign = await _upsert_sign(
            db=db,
            character=char,
            url=result["url"],
            public_id=result["public_id"],
            file_size=result["file_size"],
            width=result["width"],
            height=result["height"],
            mime_type=result["mime_type"],
            user_id=current_user["id"],
            notes=None,
            version=next_version,
        )
        uploaded.append(
            {
                "filename": file.filename,
                "character": char,
                "sign": SignRead.model_validate(sign).model_dump(),
            }
        )
        logger.info("Bulk upload: '%s' → character '%s' version %d", file.filename, char, next_version)

    return BulkUploadResult(uploaded=uploaded, skipped=skipped)


# ── PUT /signs/{character}/set-active ─────────────────────────────────────────


@router.put(
    "/{character}/set-active",
    response_model=SignRead,
    summary="Choose an existing Cloudinary image as the active image for a character",
)
async def set_active_image(
    character: str,
    request: SetActiveImageRequest,
    db: Annotated[AsyncSession, Depends(async_get_db)],
    current_user: Annotated[dict, Depends(get_current_superuser)],
):
    """Point a character's active image at any image that already exists in Cloudinary (i.e. previously uploaded to
    asl-signs/{CHARACTER}/).

    Image metadata is fetched from Cloudinary automatically.
    """
    char = character.upper()
    if char not in ALLOWED_CHARACTERS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid character")

    details = get_image_details(request.cloudinary_public_id)
    if details is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found in Cloudinary",
        )

    # Extract the version number from the public_id's last path segment.
    # e.g. "asl-signs/A/2" → version 2.  Legacy random-id images get version 0.
    last_segment = request.cloudinary_public_id.split("/")[-1]
    version = int(last_segment) if last_segment.isdigit() else 0

    sign = await _upsert_sign(
        db=db,
        character=char,
        url=details["url"],
        public_id=details["public_id"],
        file_size=details["file_size"],
        width=details["width"],
        height=details["height"],
        mime_type=details["mime_type"],
        user_id=current_user["id"],
        notes=request.notes,
        version=version,
    )
    logger.info("Active image for '%s' set to version %d (%s)", char, version, request.cloudinary_public_id)
    return sign


# ── POST /signs/character/{character}/update-url (legacy) ────────────────────


@router.post(
    "/character/{character}/update-url",
    response_model=SignRead,
    summary="Manually set the active Cloudinary URL for a character (no file upload)",
)
async def update_sign_url(
    character: str,
    request: UpdateURLRequest,
    db: Annotated[AsyncSession, Depends(async_get_db)],
    current_user: Annotated[dict, Depends(get_current_superuser)],
):
    char = character.upper()
    if char not in ALLOWED_CHARACTERS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid character")

    existing = await crud_signs.get(db=db, character=char, schema_to_select=SignRead, return_as_model=True)

    if existing:
        sign = await crud_signs.update(
            db=db,
            object=SignUpdate(
                cloudinary_url=request.cloudinary_url,
                version=existing.version,  # URL change does not bump the version
                updated_by=current_user["id"],
                notes=request.notes,
            ),
            id=existing.id,
            schema_to_select=SignRead,
            return_as_model=True,
        )
    else:
        sign = await crud_signs.create(
            db=db,
            object=SignCreate(
                character=char,
                cloudinary_url=request.cloudinary_url,
                version=1,
                updated_by=current_user["id"],
                notes=request.notes,
            ),
            schema_to_select=SignRead,
            return_as_model=True,
        )

    return sign
