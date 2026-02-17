"""Pydantic schemas for ML prediction endpoints."""

from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════


class PredictionData(BaseModel):
    """Clean prediction result — only label and confidence."""

    label: str = Field(..., description="Predicted gesture or status (error, no_hand, uncertain)")
    confidence: float = Field(..., description="Confidence percentage (0-100)")


class PredictResponse(BaseModel):
    """Response for prediction endpoint."""

    success: bool = Field(..., description="True if valid gesture detected")
    message: str = Field(..., description="Human-readable message")
    data: PredictionData = Field(..., description="Prediction result")
    query_generated_time: float = Field(..., description="Time taken in seconds")


# ═══════════════════════════════════════════════════════════════════════════════
# INFO SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════


class ServiceInfoResponse(BaseModel):
    """Public service information."""

    available: bool = Field(..., description="Is the service ready?")
    supported_gestures: list[str] = Field(..., description="List of supported gestures")
    total_gestures: int = Field(..., description="Number of supported gestures")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="healthy or unhealthy")
    service: str = Field(..., description="Service name")
