from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db.database import async_get_db
from app.crud.crud_signs import crud_signs
from app.schemas.signs import SignRead

router = APIRouter(prefix="/signs", tags=["Public Signs"])

ALLOWED_CHARACTERS: list[str] = [chr(i) for i in range(65, 91)] + ["SPACE"]


@router.get("", response_model=list[SignRead], summary="Get all active sign images")
async def get_all_signs(
    db: Annotated[AsyncSession, Depends(async_get_db)],
) -> list:
    """Return all available sign records (Cloudinary URLs) — no auth required."""
    result = await crud_signs.get_multi(
        db=db,
        schema_to_select=SignRead,
        return_as_model=True,
        limit=100,
    )
    return result.get("data", [])


@router.get("/{character}", response_model=SignRead, summary="Get the active sign image for a character")
async def get_sign_by_character(
    character: str,
    db: Annotated[AsyncSession, Depends(async_get_db)],
):
    char = character.upper()

    if char not in ALLOWED_CHARACTERS:
        raise HTTPException(status_code=400, detail="Invalid character")

    sign = await crud_signs.get(
        db=db,
        character=char,
        schema_to_select=SignRead,
        return_as_model=True,
    )

    if not sign:
        raise HTTPException(status_code=404, detail="Sign not available")

    return sign
