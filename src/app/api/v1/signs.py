from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db.database import async_get_db
from app.crud.crud_signs import crud_signs
from app.schemas.signs import SignRead

router = APIRouter(prefix="/signs", tags=["Public Signs"])


# ─────────────────────────────────
# GET SIGN BY CHARACTER (PUBLIC)
# ─────────────────────────────────


@router.get("/{character}", response_model=SignRead)
async def get_sign_by_character(character: str, db: Annotated[AsyncSession, Depends(async_get_db)]):

    character = character.upper()

    sign = await crud_signs.get(db=db, character=character)

    if not sign:
        raise HTTPException(status_code=404, detail="Sign not available")

    return SignRead.model_validate(sign)
