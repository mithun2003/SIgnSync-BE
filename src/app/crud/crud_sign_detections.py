from src.app.crud.base_analytics import AnalyticsCRUD
from src.app.models.sign_detection import SignDetection
from src.app.schemas.sign_detection import (
    SignDetectionCreateInternal,
    SignDetectionDelete,
    SignDetectionRead,
    SignDetectionUpdate,
    SignDetectionUpdateInternal,
)

CRUDSignDetection = AnalyticsCRUD[
    SignDetection,
    SignDetectionCreateInternal,
    SignDetectionUpdate,
    SignDetectionUpdateInternal,
    SignDetectionDelete,
    SignDetectionRead,
]
crud_sign_detections = CRUDSignDetection(SignDetection)
