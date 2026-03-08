"""Prediction API Router — HTTP POST and WebSocket endpoints for sign language prediction."""

import asyncio
import logging
import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.db.database import async_get_db
from ...core.ml.predict import get_health_status, get_public_info, predict_sign
from ...core.ml.schema import HealthResponse, PredictionData, PredictResponse, ServiceInfoResponse
from ...core.security import TokenType, verify_token
from ...crud.crud_sign_detections import crud_sign_detections
from ...crud.crud_users import crud_users
from ...schemas.sign_detection import SignDetectionCreateInternal
from ..dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Prediction"])


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════


@router.post("/", response_model=PredictResponse)
async def predict_sign_image(
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(async_get_db)],
    file: UploadFile = File(...),
):
    """Predict sign language gesture from skeleton image.

    **Expected input:** Skeleton image (white hand landmarks on black background)

    **Returns:**
    - label: Predicted gesture (A-Z, del, space, nothing) or status (error, no_hand, uncertain)
    - confidence: Confidence percentage (0-100)
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image file (jpg, png, etc.)")

    start_time = time.time()
    image_bytes = await file.read()

    if len(image_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 5MB.")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, predict_sign, image_bytes)
    duration = time.time() - start_time

    label = result.get("label", "error")
    confidence = float(result.get("confidence", 0.0)) / 100.0  # normalize 0-100 → 0.0-1.0
    is_success = label not in ["error", "no_hand", "uncertain"]

    # Log successful detections to the database
    if is_success:
        try:
            detection_internal = SignDetectionCreateInternal(
                user_id=current_user["id"],
                detected_sign=label,
                confidence=confidence,
                duration_seconds=round(duration, 4),
            )
            await crud_sign_detections.create(db=db, object=detection_internal)
        except Exception:
            logger.exception("Failed to log detection for user_id=%s label=%s", current_user["id"], label)

    message_map = {
        "error": "Prediction failed. Please try again.",
        "no_hand": "No hand detected in the image.",
        "uncertain": f"Low confidence prediction ({result.get('confidence', 0)}%).",
    }
    message = message_map.get(label, f"Detected gesture: {label} ({result.get('confidence', 0)}%)")

    return PredictResponse(
        success=is_success,
        message=message,
        data=PredictionData(**result),
        query_generated_time=round(duration, 4),
    )


@router.get("/info", response_model=ServiceInfoResponse)
async def get_service_info(current_user: Annotated[dict, Depends(get_current_user)]):
    """Get information about the prediction service.

    **Returns:**
    - available: Is the service ready?
    - supported_gestures: List of gesture labels (A-Z, del, space, nothing)
    - total_gestures: Number of supported gestures
    """
    return get_public_info()


@router.get("/health", response_model=HealthResponse)
async def health_check(current_user: Annotated[dict, Depends(get_current_user)]):
    """Health check endpoint for monitoring/load balancers."""
    return get_health_status()


# ═══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET ENDPOINT (Real-time prediction)
# ═══════════════════════════════════════════════════════════════════════════════


@router.websocket("/ws")
async def websocket_prediction(
    websocket: WebSocket,
    token: Annotated[str | None, Query(description="Bearer access token for authentication")] = None,
    db: AsyncSession = Depends(async_get_db),
):
    """WebSocket endpoint for real-time sign language prediction.

    **Authentication:** Pass your JWT access token as a query parameter:
    `ws://your-server/api/v1/predict/ws?token=<your_access_token>`

    **Protocol:**
    1. Connect with a valid token
    2. Send skeleton image as binary bytes
    3. Receive JSON: `{ success, data: { label, confidence }, time, frame }`
    4. Repeat for real-time prediction
    """
    # Authenticate before accepting the connection
    if not token:
        await websocket.close(code=4001, reason="Authentication required")
        return

    token_data = await verify_token(token, TokenType.ACCESS, db)
    if token_data is None:
        await websocket.close(code=4001, reason="Invalid or expired token")
        return
    if "@" in token_data.username_or_email:
        user = await crud_users.get(db=db, email=token_data.username_or_email, is_deleted=False)
    else:
        user = await crud_users.get(db=db, username=token_data.username_or_email, is_deleted=False)

    if not user:
        await websocket.close(code=4001, reason="User not found")
        return

    await websocket.accept()
    user_id = user.get("id")
    logger.info("WebSocket connected: user_id=%s", user_id)

    session_id = str(uuid.uuid4())
    frame_count = 0

    # Deduplication state: only log when sign changes or after 1-second gap
    last_logged_label: str | None = None
    last_logged_time: float = 0.0

    try:
        while True:
            data = await websocket.receive_bytes()

            if len(data) > 5 * 1024 * 1024:
                await websocket.send_json({"success": False, "error": "Frame too large (max 5MB)"})
                continue

            start_time = time.time()
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, predict_sign, data)
            duration = time.time() - start_time

            frame_count += 1

            label = result.get("label", "error")
            confidence = float(result.get("confidence", 0.0)) / 100.0  # normalize to 0-1
            is_success = label not in ["error", "no_hand", "uncertain"]

            # Log to sign_detection: only on successful detections, and only when
            # the sign changed or at least 1 second has passed since the last log.
            if is_success:
                now = time.time()
                if label != last_logged_label or (now - last_logged_time) >= 1.0:
                    try:
                        detection_internal = SignDetectionCreateInternal(
                            user_id=user_id,
                            detected_sign=label,
                            confidence=confidence,
                            session_id=session_id,
                            duration_seconds=round(duration, 4),
                        )
                        await crud_sign_detections.create(db=db, object=detection_internal)
                        last_logged_label = label
                        last_logged_time = now
                    except Exception:
                        logger.exception("Failed to log detection for user_id=%s label=%s", user_id, label)

            await websocket.send_json(
                {"success": is_success, "data": result, "time": f"{duration:.4f}s", "frame": frame_count}
            )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: user_id=%s after %d frames", user_id, frame_count)
    except Exception as e:
        logger.exception("WebSocket error for user_id=%s: %s", user_id, e)
        try:
            await websocket.send_json(
                {"success": False, "data": {"label": "error", "confidence": 0.0}, "time": "0s", "error": str(e)}
            )
            await websocket.close()
        except Exception:
            pass
