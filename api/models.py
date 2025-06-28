from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING" 
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class EmotionType(str, Enum):
    contempt = "contempt"
    sad = "sad"
    happy = "happy"
    surprised = "surprised"
    angry = "angry"
    disgusted = "disgusted"
    fear = "fear"
    neutral = "neutral"


class SynthesisRequest(BaseModel):
    emotion: EmotionType = Field(..., description="Emotion type for the synthesis")
    ref_scale: float = Field(3.0, ge=0.0, le=10.0, description="Reference scale (0.0-10.0)")
    emo_scale: float = Field(6.0, ge=0.0, le=10.0, description="Emotion scale (0.0-10.0)")
    crop: bool = Field(False, description="Whether to crop the image")


class TaskResponse(BaseModel):
    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Current task status")
    message: Optional[str] = Field(None, description="Status message or error description")


class TaskStatusResponse(BaseModel):
    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Current task status")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Task progress (0.0-1.0)")
    message: Optional[str] = Field(None, description="Status message or error description")
    created_at: str = Field(..., description="Task creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")


class HealthResponse(BaseModel):
    status: Literal["healthy", "unhealthy"] = Field(..., description="Service health status")
    gpu_available: bool = Field(..., description="GPU availability")
    model_loaded: bool = Field(..., description="Model loading status")
    version: str = Field("1.0.0", description="API version")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information") 