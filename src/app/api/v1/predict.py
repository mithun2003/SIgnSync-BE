import time

from fastapi import APIRouter, File, UploadFile, WebSocket, WebSocketDisconnect

# Import your existing ML logic
from ...core.ml.predict import predict_sign
from ...core.ml.schema import PredictResponse

router = APIRouter(prefix="/predict", tags=["Prediction"])


# --- EXISTING HTTP POST ENDPOINT ---
@router.post("/", response_model=PredictResponse)
async def predict_sign_image(file: UploadFile = File(...)):
    start_time = time.time()
    image_bytes = await file.read()

    # Run Prediction
    result = predict_sign(image_bytes)

    end_time = time.time()
    duration = end_time - start_time

    return PredictResponse(
        success=True,
        message=f"The program took {duration:.4f} seconds to complete.",
        data=result,
        query_generated_time=duration,
    )


# --- NEW WEBSOCKET ENDPOINT ---
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ Client connected to WebSocket")

    try:
        while True:
            # 1. Receive Image Bytes from Frontend
            # (The frontend sends the cropped hand blob directly)
            data = await websocket.receive_bytes()

            start_time = time.time()

            # 2. Predict (Reusing your existing logic)
            # Ensure predict_image returns a Dict (e.g. {'label': 'A', 'confidence': 90.5})
            result = predict_sign(data)

            end_time = time.time()
            duration = end_time - start_time

            # 3. Construct Response (JSON)
            response = {"success": True, "data": result, "time": f"{duration:.4f}s"}

            # 4. Send back to Frontend
            await websocket.send_json(response)

    except WebSocketDisconnect:
        print("❌ Client disconnected")
    except Exception as e:
        print(f"⚠️ WebSocket Error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
