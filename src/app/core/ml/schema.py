from pydantic import BaseModel

from ...core.schemas import CommonResponse


class PredictData(BaseModel):
    label: str
    confidence: float


class PredictResponse(CommonResponse):
    data: PredictData
