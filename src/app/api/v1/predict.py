from fastapi import APIRouter, File, UploadFile

from ...core.ml.predict import predict_image
from ...core.ml.schema import PredictResponse

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("/", response_model=PredictResponse)
async def predict_sign(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict_image(image_bytes)
    return result
