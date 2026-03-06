import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

from app.api.dependencies import get_current_superuser
from app.core.db.database import async_get_db
from app.core.utils import cloudinary
from app.crud.crud_signs import crud_signs
from app.models.user import User
from app.schemas.signs import SignCreate, SignRead, SignUpdate, UpdateURLRequest

router = APIRouter(
    prefix="/signs",
    tags=["Admin ASL Signs"]
)

ALLOWED_CHARACTERS = [chr(i) for i in range(65, 91)] + ["SPACE"]


# ─────────────────────────────────
# GET ALL SIGNS (ADMIN)
# ─────────────────────────────────

@router.get("")
async def get_all_signs(
    db: Annotated[AsyncSession, Depends(async_get_db)],
    current_user: User = Depends(get_current_superuser)
):

    signs = await crud_signs.get_multi(
        db=db,
        schema_to_select=SignRead,
        return_as_model=True
    )

    data = signs.get("data", [])
    total = signs.get("total_count", 0)

    existing = {sign.character for sign in data}
    missing = [c for c in ALLOWED_CHARACTERS if c not in existing]

    return {
        "data": data,
        "total": total,
        "missing_characters": missing,
    }


# ─────────────────────────────────
# UPDATE OR CREATE SIGN
# ─────────────────────────────────

@router.post("/character/{character}/update-url")
async def update_sign_url(
    character: str,
    request: UpdateURLRequest,
    db: Annotated[AsyncSession, Depends(async_get_db)],
    current_user: User = Depends(get_current_superuser)
):

    character = character.upper()

    if character not in ALLOWED_CHARACTERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid character"
        )

    existing = await crud_signs.get(db=db, character=character, schema_to_select=SignRead, return_as_model=True)

    if existing:

        update_data = SignUpdate(
            cloudinary_url=request.cloudinary_url,
            version=existing.version + 1,
            updated_by=current_user["id"],
            notes=request.notes
        )

        sign = await crud_signs.update(
            db=db,
            object=update_data,
            id=existing.id,
            schema_to_select=SignRead,
            return_as_model=True
        )

    else:

        create_data = SignCreate(
            character=character,
            cloudinary_url=request.cloudinary_url,
            version=1,
            updated_by=current_user["id"],
            notes=request.notes
        )

        sign = await crud_signs.create(
            db=db,
            object=create_data,
            schema_to_select=SignRead,
            return_as_model=True
        )

    return sign


# ─────────────────────────────────
# GET ADMIN STATISTICS
# ─────────────────────────────────

@router.get("/stats")
async def get_statistics(
    db: Annotated[AsyncSession, Depends(async_get_db)],
    current_user: User = Depends(get_current_superuser)
):

    signs = await crud_signs.get_multi(db=db, schema_to_select=SignRead, return_as_model=True)
    logger.debug("Admin signs stats retrieved: total=%s", signs.get("total_count", 0))

    existing = {sign.character for sign in signs.get("data", [])}
    missing = [c for c in ALLOWED_CHARACTERS if c not in existing]

    total = signs.get("total_count", 0)

    return {
        "total_signs": total,
        "missing_signs": len(missing),
        "missing_characters": missing,
        "completion_percentage": round((total / len(ALLOWED_CHARACTERS)) * 100, 2),
    }

@router.get("/cloudinary-images/{character}")
async def list_sign_images(character: str):

    result = cloudinary.search.Search() \
        .expression(f"tags=character_{character}") \
        .sort_by("created_at", "desc") \
        .max_results(30) \
        .execute()

    return result["resources"]
