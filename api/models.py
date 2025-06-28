from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING" 
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class EmotionType(str, Enum):
    CONTEMPT = "contempt"
    SAD = "sad"
    HAPPY = "happy"
    SURPRISED = "surprised"
    ANGRY = "angry"
    DISGUSTED = "disgusted"
    FEAR = "fear"
    NEUTRAL = "neutral"


class SynthesisRequest(BaseModel):
    emotion: EmotionType = Field(..., description="Emotion type for the synthesis")
    ref_scale: float = Field(3.0, ge=0.0, le=10.0, description="Reference scale (0.0-10.0)")
    emo_scale: float = Field(6.0, ge=0.0, le=10.0, description="Emotion scale (0.0-10.0)")
    crop: bool = Field(False, description="Whether to crop the image")
    inference_steps: int = Field(20, ge=8, le=50, description="Number of inference steps (8-50)")
    duration: Optional[float] = Field(None, ge=0.1, le=300.0, description="Video duration in seconds (0.1-300.0)")
    fps: int = Field(24, ge=8, le=60, description="Frames per second (8-60)")
    seed: Optional[int] = Field(None, ge=0, le=2147483647, description="Random seed for generation")
    dynamic_scale: float = Field(1.0, ge=0.1, le=3.0, description="Dynamic scale factor (0.1-3.0)")
    min_resolution: int = Field(256, ge=128, le=2048, description="Minimum resolution (128-2048)")
    expand_ratio: float = Field(0.5, ge=0.1, le=2.0, description="Expand ratio (0.1-2.0)")


class TaskResponse(BaseModel):
    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Current task status")
    message: Optional[str] = Field(None, description="Status message or error description")
    original_task_id: Optional[str] = Field(None, description="Original task ID for cached results")


class TaskStatusResponse(BaseModel):
    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Current task status")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Task progress (0.0-1.0)")
    message: Optional[str] = Field(None, description="Status message or error description")
    created_at: str = Field(..., description="Task creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    original_task_id: Optional[str] = Field(None, description="Original task ID for cached results")


class HealthResponse(BaseModel):
    status: Literal["healthy", "unhealthy"] = Field(..., description="Service health status")
    gpu_available: bool = Field(..., description="GPU availability")
    model_loaded: bool = Field(..., description="Model loading status")
    version: str = Field("1.0.0", description="API version")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information") 