"""Sign Detection endpoints — follows the same pattern as your posts.py router.

Uses crud_sign_detections for ALL database operations.
Router only handles: validation → CRUD call → response formatting.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastcrud import PaginatedListResponse, compute_offset, paginated_response
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.db.database import async_get_db
from ...core.exceptions.http_exceptions import NotFoundException
from ...crud.crud_sign_detections import crud_sign_detections
from ...schemas.sign_detection import (
    SignDetectionBatchCreate,
    SignDetectionCreate,
    SignDetectionCreateInternal,
    SignDetectionRead,
    SignDetectionUpdate,
)
from ..dependencies import get_current_user

router = APIRouter(prefix="/detection", tags=["sign-detection"])


# ─────────────────────────────────────────────────────────────
#  CREATE — Log a single detection
# ─────────────────────────────────────────────────────────────
@router.post("/log", response_model=SignDetectionRead, status_code=201)
async def log_detection(
    detection: SignDetectionCreate,
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(async_get_db)],
) -> dict[str, Any]:
    # Add user_id and create internal schema
    detection_dict = detection.model_dump()
    detection_dict["user_id"] = current_user["id"]

    detection_internal = SignDetectionCreateInternal(**detection_dict)
    created = await crud_sign_detections.create(
        db=db,
        object=detection_internal,
        schema_to_select=SignDetectionRead,
    )

    if created is None:
        raise NotFoundException("Failed to log detection")

    return created


# ─────────────────────────────────────────────────────────────
#  CREATE BATCH — Log multiple detections at once
# ─────────────────────────────────────────────────────────────
@router.post("/log/batch", status_code=201)
async def log_batch_detections(
    batch: SignDetectionBatchCreate,
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(async_get_db)],
) -> dict[str, Any]:
    created_count = 0

    for detection in batch.detections:
        detection_dict = detection.model_dump()
        detection_dict["user_id"] = current_user["id"]

        detection_internal = SignDetectionCreateInternal(**detection_dict)
        await crud_sign_detections.create(db=db, object=detection_internal)
        created_count += 1

    return {"message": f"{created_count} detections logged", "count": created_count}


# ─────────────────────────────────────────────────────────────
#  READ — Get user's detections (paginated)
# ─────────────────────────────────────────────────────────────
@router.get("/list", response_model=PaginatedListResponse[SignDetectionRead])
async def read_detections(
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(async_get_db)],
    page: int = 1,
    items_per_page: int = 20,
) -> dict:
    detections_data = await crud_sign_detections.get_multi(
        db=db,
        offset=compute_offset(page, items_per_page),
        limit=items_per_page,
        user_id=current_user["id"],
        is_deleted=False,
    )

    response: dict[str, Any] = paginated_response(
        crud_data=detections_data,
        page=page,
        items_per_page=items_per_page,
    )
    return response


# ─────────────────────────────────────────────────────────────
#  READ ONE — Get a specific detection
# ─────────────────────────────────────────────────────────────
@router.get("/{id}", response_model=SignDetectionRead)
async def read_detection(
    id: int,
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(async_get_db)],
) -> dict[str, Any]:
    detection = await crud_sign_detections.get(
        db=db,
        id=id,
        user_id=current_user["id"],
        is_deleted=False,
        schema_to_select=SignDetectionRead,
    )

    if detection is None:
        raise NotFoundException("Detection not found")

    return detection


# ─────────────────────────────────────────────────────────────
#  UPDATE — Correct a detection
# ─────────────────────────────────────────────────────────────
@router.patch("/{id}")
async def update_detection(
    id: int,
    values: SignDetectionUpdate,
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(async_get_db)],
) -> dict[str, str]:
    detection = await crud_sign_detections.get(
        db=db,
        id=id,
        user_id=current_user["id"],
        is_deleted=False,
    )

    if detection is None:
        raise NotFoundException("Detection not found")

    await crud_sign_detections.update(db=db, object=values, id=id)
    return {"message": "Detection updated"}


# ─────────────────────────────────────────────────────────────
#  DELETE — Soft-delete a detection
# ─────────────────────────────────────────────────────────────
@router.delete("/{id}")
async def delete_detection(
    id: int,
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(async_get_db)],
) -> dict[str, str]:
    detection = await crud_sign_detections.get(
        db=db,
        id=id,
        user_id=current_user["id"],
        is_deleted=False,
    )

    if detection is None:
        raise NotFoundException("Detection not found")

    await crud_sign_detections.delete(db=db, id=id)
    return {"message": "Detection deleted"}
