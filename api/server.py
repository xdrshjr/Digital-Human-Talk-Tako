import os
import sys
import shutil
from pathlib import Path
from typing import Dict, Any
import logging
import torch

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from dice_talk import DICE_Talk
from .models import (
    TaskResponse, TaskStatusResponse, HealthResponse, ErrorResponse,
    EmotionType, TaskStatus
)
from .utils import (
    validate_image_file, validate_audio_file, save_upload_file,
    get_file_hash, ensure_safe_path, cleanup_file
)
from .task_manager import task_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="DICE-Talk API",
    description="API for DICE-Talk: Disentangle Identity, Cooperate Emotion - Emotional Talking Portrait Generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
dice_talk_pipe = None
BASE_DIR = project_root
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
EMOTION_DIR = os.path.join(BASE_DIR, "examples", "emo")

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """Initialize DICE-Talk pipeline on startup."""
    global dice_talk_pipe
    try:
        logger.info("Initializing DICE-Talk pipeline...")
        dice_talk_pipe = DICE_Talk(device_id=0)
        task_manager.set_dice_talk_pipe(dice_talk_pipe)
        logger.info("DICE-Talk pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize DICE-Talk pipeline: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    task_manager.shutdown()
    logger.info("DICE-Talk API server shutdown")


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    gpu_available = torch.cuda.is_available()
    model_loaded = dice_talk_pipe is not None
    
    status = "healthy" if (gpu_available and model_loaded) else "unhealthy"
    
    return HealthResponse(
        status=status,
        gpu_available=gpu_available,
        model_loaded=model_loaded,
        version="1.0.0"
    )


@app.post("/api/v1/synthesis/start", response_model=TaskResponse)
async def start_synthesis(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="Portrait image file"),
    audio: UploadFile = File(..., description="Audio file for speech"),
    emotion: EmotionType = Form(..., description="Emotion type"),
    ref_scale: float = Form(3.0, description="Reference scale (0.0-10.0)"),
    emo_scale: float = Form(6.0, description="Emotion scale (0.0-10.0)"),
    crop: bool = Form(False, description="Whether to crop the image")
):
    """Start a new synthesis task."""
    try:
        # Validate input files
        validate_image_file(image)
        validate_audio_file(audio)
        
        # Validate parameters
        if not (0.0 <= ref_scale <= 10.0):
            raise HTTPException(status_code=400, detail="ref_scale must be between 0.0 and 10.0")
        if not (0.0 <= emo_scale <= 10.0):
            raise HTTPException(status_code=400, detail="emo_scale must be between 0.0 and 10.0")
        
        # Generate file paths
        image_content = await image.read()
        audio_content = await audio.read()
        
        image_hash = get_file_hash(image_content)
        audio_hash = get_file_hash(audio_content)
        
        # Create unique filenames
        image_ext = os.path.splitext(image.filename)[1]
        audio_ext = os.path.splitext(audio.filename)[1]
        
        image_filename = f"{image_hash}{image_ext}"
        audio_filename = f"{audio_hash}{audio_ext}"
        
        image_path = os.path.join(UPLOAD_DIR, image_filename)
        audio_path = os.path.join(UPLOAD_DIR, audio_filename)
        emotion_path = os.path.join(EMOTION_DIR, f"{emotion.value}.npy")
        
        # Check if emotion file exists
        if not os.path.exists(emotion_path):
            raise HTTPException(status_code=400, detail=f"Emotion file not found: {emotion.value}")
        
        # Save uploaded files
        with open(image_path, 'wb') as f:
            f.write(image_content)
        with open(audio_path, 'wb') as f:
            f.write(audio_content)
        
        # Generate output path
        output_filename = f"{image_hash}_{audio_hash}_{emotion.value}_{ref_scale}_{emo_scale}_{int(crop)}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Check if result already exists
        if os.path.exists(output_path):
            # Return existing result as completed task
            task_id = f"cached_{image_hash[:8]}_{audio_hash[:8]}"
            return TaskResponse(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                message="Result already exists"
            )
        
        # Create synthesis task
        task_id = task_manager.create_task(
            image_path=image_path,
            audio_path=audio_path,
            emotion_path=emotion_path,
            output_path=output_path,
            ref_scale=ref_scale,
            emo_scale=emo_scale,
            crop=crop
        )
        
        # Note: File cleanup is now handled by the task manager after processing
        # to prevent premature deletion of files still needed by the synthesis process
        
        return TaskResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            message="Task created successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting synthesis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/synthesis/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get task status by ID."""
    # Handle cached results
    if task_id.startswith("cached_"):
        return TaskStatusResponse(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            progress=1.0,
            message="Cached result",
            created_at="",
            updated_at=""
        )
    
    status = task_manager.get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return status


@app.get("/api/v1/synthesis/{task_id}/download")
async def download_result(task_id: str):
    """Download synthesis result."""
    try:
        # Handle cached results
        if task_id.startswith("cached_"):
            # Extract parameters from cached task ID and reconstruct filename
            # This is a simplified approach for demo purposes
            raise HTTPException(status_code=400, detail="Please use the original task ID for download")
        
        # Get task status first
        status = task_manager.get_task_status(task_id)
        if not status:
            raise HTTPException(status_code=404, detail="Task not found")
        
        if status.status != TaskStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="Task not completed yet")
        
        # Get output file path
        output_path = task_manager.get_task_output_path(task_id)
        if not output_path or not os.path.exists(output_path):
            raise HTTPException(status_code=404, detail="Result file not found")
        
        # Ensure file path is safe
        safe_path = ensure_safe_path(OUTPUT_DIR, output_path)
        
        return FileResponse(
            path=safe_path,
            filename=os.path.basename(safe_path),
            media_type="video/mp4"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading result: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.detail).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="Internal server error").dict()
    )


if __name__ == "__main__":
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    ) 