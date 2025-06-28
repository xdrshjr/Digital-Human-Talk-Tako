import os
import uuid
import asyncio
import threading
from datetime import datetime
from typing import Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import logging
import shutil
import time

from .models import TaskStatus, TaskStatusResponse
from .utils import cleanup_file

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskInfo:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.status = TaskStatus.PENDING
        self.progress = 0.0
        self.message = ""
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        self.image_path: Optional[str] = None
        self.audio_path: Optional[str] = None
        self.emotion_path: Optional[str] = None
        self.output_path: Optional[str] = None
        self.error: Optional[str] = None
        self.original_task_id: Optional[str] = None

    def update_status(self, status: TaskStatus, progress: float = None, message: str = ""):
        self.status = status
        if progress is not None:
            self.progress = progress
        self.message = message
        self.updated_at = datetime.now().isoformat()

    def to_response(self) -> TaskStatusResponse:
        return TaskStatusResponse(
            task_id=self.task_id,
            status=self.status,
            progress=self.progress,
            message=self.message,
            created_at=self.created_at,
            updated_at=self.updated_at,
            original_task_id=self.original_task_id
        )


class TaskManager:
    def __init__(self, max_workers: int = 2):
        self.tasks: Dict[str, TaskInfo] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.dice_talk_pipe = None
        self._lock = threading.Lock()

    def set_dice_talk_pipe(self, pipe):
        """Set the DICE-Talk pipeline instance."""
        self.dice_talk_pipe = pipe

    def create_task(self, image_path: str, audio_path: str, emotion_path: str, 
                   output_path: str, ref_scale: float, emo_scale: float, 
                   crop: bool, inference_steps: int = 20, duration: Optional[float] = None,
                   fps: int = 24, seed: Optional[int] = None, dynamic_scale: float = 1.0,
                   min_resolution: int = 256, expand_ratio: float = 0.5) -> str:
        """Create a new synthesis task."""
        task_id = str(uuid.uuid4())
        
        with self._lock:
            task_info = TaskInfo(task_id)
            task_info.image_path = image_path
            task_info.audio_path = audio_path
            task_info.emotion_path = emotion_path
            task_info.output_path = output_path
            self.tasks[task_id] = task_info

        # Submit task to thread pool with all parameters
        future = self.executor.submit(
            self._process_task, task_id, image_path, audio_path, emotion_path,
            output_path, ref_scale, emo_scale, crop, inference_steps, duration,
            fps, seed, dynamic_scale, min_resolution, expand_ratio
        )
        
        logger.info(f"Created task {task_id}")
        return task_id

    def get_task_status(self, task_id: str) -> Optional[TaskStatusResponse]:
        """Get task status by ID."""
        with self._lock:
            task_info = self.tasks.get(task_id)
            if task_info:
                return task_info.to_response()
        return None

    def get_task_output_path(self, task_id: str) -> Optional[str]:
        """Get task output file path."""
        with self._lock:
            task_info = self.tasks.get(task_id)
            if task_info and task_info.status == TaskStatus.COMPLETED:
                return task_info.output_path
        return None

    def cleanup_task_files(self, task_id: str) -> None:
        """Clean up temporary files for a task."""
        with self._lock:
            task_info = self.tasks.get(task_id)
            if task_info:
                # Clean up uploaded files
                if task_info.image_path:
                    cleanup_file(task_info.image_path)
                if task_info.audio_path:
                    cleanup_file(task_info.audio_path)
                if task_info.emotion_path:
                    cleanup_file(task_info.emotion_path)

    def _process_task(self, task_id: str, image_path: str, audio_path: str, 
                     emotion_path: str, output_path: str, ref_scale: float, 
                     emo_scale: float, crop: bool, inference_steps: int = 20,
                     duration: Optional[float] = None, fps: int = 24,
                     seed: Optional[int] = None, dynamic_scale: float = 1.0,
                     min_resolution: int = 256, expand_ratio: float = 0.5):
        """Process a synthesis task."""
        try:
            with self._lock:
                if task_id not in self.tasks:
                    logger.error(f"Task {task_id} not found")
                    return
                self.tasks[task_id].status = TaskStatus.PROCESSING
                self.tasks[task_id].message = "Starting synthesis..."
                self.tasks[task_id].updated_at = datetime.now().isoformat()

            logger.info(f"Starting synthesis for task {task_id} with parameters: inference_steps={inference_steps}, duration={duration}, fps={fps}, seed={seed}")

            if self.dice_talk_pipe is None:
                raise Exception("DICE-Talk pipeline not initialized")

            # Log the parameters being used
            logger.info(f"Task {task_id} synthesis parameters:")
            logger.info(f"  - ref_scale: {ref_scale}")
            logger.info(f"  - emo_scale: {emo_scale}")
            logger.info(f"  - crop: {crop}")
            logger.info(f"  - inference_steps: {inference_steps}")
            logger.info(f"  - duration: {duration}")
            logger.info(f"  - fps: {fps}")
            logger.info(f"  - seed: {seed}")
            logger.info(f"  - dynamic_scale: {dynamic_scale}")
            logger.info(f"  - min_resolution: {min_resolution}")
            logger.info(f"  - expand_ratio: {expand_ratio}")

            # Update progress
            with self._lock:
                self.tasks[task_id].progress = 0.1
                self.tasks[task_id].message = "Loading models..."
                self.tasks[task_id].updated_at = datetime.now().isoformat()

            # Preprocess image if needed
            with self._lock:
                self.tasks[task_id].progress = 0.2
                self.tasks[task_id].message = "Preprocessing image..."
                self.tasks[task_id].updated_at = datetime.now().isoformat()

            face_info = self.dice_talk_pipe.preprocess(image_path, expand_ratio=expand_ratio)
            
            if face_info['face_num'] <= 0:
                raise Exception("No face detected in the image")

            # Crop image if needed
            work_image_path = image_path
            if crop:
                with self._lock:
                    self.tasks[task_id].progress = 0.3
                    self.tasks[task_id].message = "Cropping image..."
                    self.tasks[task_id].updated_at = datetime.now().isoformat()

                crop_image_path = image_path + '.crop.png'
                self.dice_talk_pipe.crop_image(image_path, crop_image_path, face_info['crop_bbox'])
                work_image_path = crop_image_path

            # Update progress before synthesis
            with self._lock:
                self.tasks[task_id].progress = 0.4
                self.tasks[task_id].message = "Generating video..."
                self.tasks[task_id].updated_at = datetime.now().isoformat()

            # Process audio trimming if duration is specified
            work_audio_path = audio_path
            if duration is not None:
                with self._lock:
                    self.tasks[task_id].progress = 0.35
                    self.tasks[task_id].message = "Processing audio duration..."
                    self.tasks[task_id].updated_at = datetime.now().isoformat()
                logger.info(f"Task {task_id}: Audio will be trimmed to {duration} seconds")

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Call DICE-Talk pipeline process method with supported parameters
            # Now includes duration parameter for audio trimming
            result = self.dice_talk_pipe.process(
                image_path=work_image_path,
                audio_path=work_audio_path,
                emotion_path=emotion_path,
                output_path=output_path,
                min_resolution=min_resolution,
                inference_steps=inference_steps,
                ref_scale=ref_scale,
                emo_scale=emo_scale,
                seed=seed,
                duration=duration
            )

            logger.info(f"Task {task_id}: DICE-Talk processing completed with result: {result}")
            
            if result != 0:
                raise Exception(f"DICE-Talk processing failed with return code: {result}")

            # Check if output file was created
            if not os.path.exists(output_path):
                # Check for alternative output file (without audio)
                noaudio_output_path = output_path.replace('.mp4', '_noaudio.mp4')
                if os.path.exists(noaudio_output_path):
                    # Move the file to the expected location
                    os.rename(noaudio_output_path, output_path)
                    logger.warning(f"Task {task_id}: Used video without audio")
                else:
                    raise Exception("No output video file was created")

            with self._lock:
                if task_id in self.tasks:
                    self.tasks[task_id].status = TaskStatus.COMPLETED
                    self.tasks[task_id].progress = 1.0
                    self.tasks[task_id].message = "Synthesis completed successfully"
                    self.tasks[task_id].updated_at = datetime.now().isoformat()

            logger.info(f"Task {task_id} completed successfully")

            # Clean up crop file if created
            if crop and work_image_path.endswith('.crop.png'):
                try:
                    if os.path.exists(work_image_path):
                        os.remove(work_image_path)
                        logger.info(f"Cleaned up crop file for task {task_id}")
                except Exception as e:
                    logger.warning(f"Failed to clean up crop file for task {task_id}: {e}")

        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            with self._lock:
                if task_id in self.tasks:
                    self.tasks[task_id].status = TaskStatus.FAILED
                    self.tasks[task_id].message = f"Synthesis failed: {str(e)}"
                    self.tasks[task_id].updated_at = datetime.now().isoformat()

    def shutdown(self):
        """Shutdown the task manager."""
        self.executor.shutdown(wait=True)


# Global task manager instance
task_manager = TaskManager() 