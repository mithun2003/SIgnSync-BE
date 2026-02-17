"""Prediction API Router Handles HTTP POST and WebSocket endpoints for sign language prediction."""

import time

from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from ...core.ml.predict import get_health_status, get_public_info, predict_sign
from ...core.ml.schema import HealthResponse, PredictionData, PredictResponse, ServiceInfoResponse

router = APIRouter(prefix="/predict", tags=["Prediction"])


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════


@router.post("/", response_model=PredictResponse)
async def predict_sign_image(file: UploadFile = File(...)):
    """Predict sign language gesture from skeleton image.

    **Expected input:** Skeleton image (white hand landmarks on black background)

    **Returns:**
    - label: Predicted gesture (A-Z, del, space, nothing) or status (error, no_hand, uncertain)
    - confidence: Confidence percentage (0-100)
    """
    # 1. Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image file (jpg, png, etc.)")

    # 2. Read image bytes
    start_time = time.time()
    image_bytes = await file.read()

    # 3. Validate file size (max 5MB)
    if len(image_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 5MB.")

    # 4. Run prediction
    result = predict_sign(image_bytes)

    # 5. Calculate duration
    duration = time.time() - start_time

    # 6. Determine success based on label
    label = result.get("label", "error")
    is_success = label not in ["error", "no_hand", "uncertain"]

    # 7. Build appropriate message
    if label == "error":
        message = "Prediction failed. Please try again."
    elif label == "no_hand":
        message = "No hand detected in the image."
    elif label == "uncertain":
        message = f"Low confidence prediction ({result.get('confidence', 0)}%)."
    else:
        message = f"Detected gesture: {label} ({result.get('confidence', 0)}%)"

    return PredictResponse(
        success=is_success,
        message=message,
        data=PredictionData(**result),
        query_generated_time=round(duration, 4),
    )


@router.get("/info", response_model=ServiceInfoResponse)
async def get_service_info():
    """Get information about the prediction service.

    **Returns:**
    - available: Is the service ready?
    - supported_gestures: List of gesture labels (A-Z, del, space, nothing)
    - total_gestures: Number of supported gestures
    """
    return get_public_info()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring/load balancers."""
    return get_health_status()


# ═══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET ENDPOINT (Real-time prediction)
# ═══════════════════════════════════════════════════════════════════════════════


@router.websocket("/ws")
async def websocket_prediction(websocket: WebSocket):
    """WebSocket endpoint for real-time sign language prediction.

    **Protocol:**
    1. Client connects to ws://your-server/api/v1/predict/ws
    2. Client sends skeleton image as binary bytes
    3. Server responds with JSON: { success, data: { label, confidence }, time }
    4. Repeat for real-time prediction
    """
    await websocket.accept()
    print("✅ Client connected to WebSocket")

    frame_count = 0

    try:
        while True:
            # 1. Receive image bytes from frontend
            data = await websocket.receive_bytes()

            # 2. Run prediction
            start_time = time.time()
            result = predict_sign(data)
            duration = time.time() - start_time

            frame_count += 1

            # 3. Determine success
            label = result.get("label", "error")
            is_success = label not in ["error", "no_hand", "uncertain"]

            # 4. Send response
            await websocket.send_json(
                {"success": is_success, "data": result, "time": f"{duration:.4f}s", "frame": frame_count}
            )

    except WebSocketDisconnect:
        print(f"🔌 Client disconnected after {frame_count} frames")
    except Exception as e:
        print(f"❌ WebSocket Error: {e}")
        try:
            await websocket.send_json(
                {"success": False, "data": {"label": "error", "confidence": 0.0}, "time": "0s", "error": str(e)}
            )
            await websocket.close()
        except Exception:
            pass
