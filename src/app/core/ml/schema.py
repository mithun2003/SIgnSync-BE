from pydantic import BaseModel

from ...core.schemas import CommonResponse


class PredictData(BaseModel):
    labels: str
    confidences: float


class PredictResponse(CommonResponse):
    data: PredictData
