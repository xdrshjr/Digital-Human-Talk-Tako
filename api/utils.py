import os
import hashlib
import mimetypes
from typing import Optional, Tuple
from fastapi import HTTPException, UploadFile
import aiofiles


# Allowed file types
ALLOWED_IMAGE_TYPES = {
    "image/jpeg": [".jpg", ".jpeg"],
    "image/png": [".png"],
}

ALLOWED_AUDIO_TYPES = {
    "audio/wav": [".wav"],
    "audio/mpeg": [".mp3"],
    "audio/mp3": [".mp3"],
}

# File size limits (in bytes)
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_AUDIO_SIZE = 50 * 1024 * 1024  # 50MB


def validate_file_type(file: UploadFile, allowed_types: dict) -> bool:
    """Validate file type based on content type and extension."""
    if file.content_type not in allowed_types:
        return False
    
    # Also check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_types[file.content_type]:
        return False
    
    return True


def validate_file_size(file: UploadFile, max_size: int) -> bool:
    """Validate file size."""
    if hasattr(file, 'size') and file.size and file.size > max_size:
        return False
    return True


async def save_upload_file(file: UploadFile, file_path: str) -> str:
    """Save uploaded file to specified path."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    return file_path


def get_file_hash(content: bytes) -> str:
    """Generate MD5 hash for file content."""
    return hashlib.md5(content).hexdigest()


def validate_image_file(file: UploadFile) -> None:
    """Validate image file."""
    if not validate_file_type(file, ALLOWED_IMAGE_TYPES):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file type. Allowed types: {list(ALLOWED_IMAGE_TYPES.keys())}"
        )
    
    if not validate_file_size(file, MAX_IMAGE_SIZE):
        raise HTTPException(
            status_code=400,
            detail=f"Image file too large. Maximum size: {MAX_IMAGE_SIZE // (1024*1024)}MB"
        )


def validate_audio_file(file: UploadFile) -> None:
    """Validate audio file."""
    if not validate_file_type(file, ALLOWED_AUDIO_TYPES):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio file type. Allowed types: {list(ALLOWED_AUDIO_TYPES.keys())}"
        )
    
    if not validate_file_size(file, MAX_AUDIO_SIZE):
        raise HTTPException(
            status_code=400,
            detail=f"Audio file too large. Maximum size: {MAX_AUDIO_SIZE // (1024*1024)}MB"
        )


def ensure_safe_path(base_path: str, file_path: str) -> str:
    """Ensure file path is safe and within base directory."""
    # Resolve absolute paths
    base_path = os.path.abspath(base_path)
    file_path = os.path.abspath(file_path)
    
    # Check if file path is within base path
    if not file_path.startswith(base_path):
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    return file_path


def cleanup_file(file_path: str) -> None:
    """Safely delete a file if it exists."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        # Log error but don't raise exception
        pass 