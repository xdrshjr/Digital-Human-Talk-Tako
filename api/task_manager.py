import os
import uuid
import asyncio
import threading
from datetime import datetime
from typing import Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import logging
import shutil

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
            updated_at=self.updated_at
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
                   crop: bool) -> str:
        """Create a new synthesis task."""
        task_id = str(uuid.uuid4())
        
        with self._lock:
            task_info = TaskInfo(task_id)
            task_info.image_path = image_path
            task_info.audio_path = audio_path
            task_info.emotion_path = emotion_path
            task_info.output_path = output_path
            self.tasks[task_id] = task_info

        # Submit task to thread pool
        future = self.executor.submit(
            self._process_task, task_id, image_path, audio_path, emotion_path,
            output_path, ref_scale, emo_scale, crop
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
                     emo_scale: float, crop: bool) -> None:
        """Process synthesis task in background thread."""
        
        try:
            # Update status to processing
            with self._lock:
                if task_id in self.tasks:
                    self.tasks[task_id].update_status(TaskStatus.PROCESSING, 0.1, "Starting synthesis...")

            logger.info(f"Processing task {task_id}")

            if not self.dice_talk_pipe:
                raise Exception("DICE-Talk pipeline not initialized")

            # Use original files directly since user handles cleanup with scheduled tasks
            work_image_path = image_path
            work_audio_path = audio_path
            
            logger.info(f"Task {task_id}: Using original uploaded files:")
            logger.info(f"  - image_path: {work_image_path}")
            logger.info(f"  - audio_path: {work_audio_path}")

            # Step 1: Preprocess image
            with self._lock:
                if task_id in self.tasks:
                    self.tasks[task_id].update_status(TaskStatus.PROCESSING, 0.2, "Preprocessing image...")

            face_info = self.dice_talk_pipe.preprocess(work_image_path, expand_ratio=0.5)
            
            if face_info['face_num'] <= 0:
                raise Exception("No face detected in the image")

            # Step 2: Crop image if needed
            if crop:
                with self._lock:
                    if task_id in self.tasks:
                        self.tasks[task_id].update_status(TaskStatus.PROCESSING, 0.3, "Cropping image...")

                crop_image_path = work_image_path + '.crop.png'
                self.dice_talk_pipe.crop_image(work_image_path, crop_image_path, face_info['crop_bbox'])
                work_image_path = crop_image_path

            # Step 3: Process synthesis
            with self._lock:
                if task_id in self.tasks:
                    self.tasks[task_id].update_status(TaskStatus.PROCESSING, 0.4, "Generating video...")

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Run the main synthesis process
            logger.info(f"Task {task_id}: Starting DICE-Talk processing with:")
            logger.info(f"  - work_image_path: {work_image_path}")
            logger.info(f"  - work_audio_path: {work_audio_path}")
            logger.info(f"  - emotion_path: {emotion_path}")
            logger.info(f"  - output_path: {output_path}")
            
            result = self.dice_talk_pipe.process(
                work_image_path, 
                work_audio_path,  # Use original audio file
                emotion_path, 
                output_path,
                min_resolution=256,
                inference_steps=20,
                ref_scale=ref_scale,
                emo_scale=emo_scale,
                seed=None
            )
            
            logger.info(f"Task {task_id}: DICE-Talk processing completed with result: {result}")
            
            if result != 0:
                raise Exception(f"DICE-Talk processing failed with return code: {result}")

            # Check which output file was actually created
            # The DICE-Talk pipeline might create filename_noaudio.mp4 if ffmpeg fails
            actual_output_path = output_path
            noaudio_output_path = output_path.replace('.mp4', '_noaudio.mp4')
            
            if os.path.exists(output_path):
                # Final video with audio was created successfully
                actual_output_path = output_path
                logger.info(f"Task {task_id}: Final video with audio created successfully")
            elif os.path.exists(noaudio_output_path):
                # Only video without audio was created (ffmpeg failed)
                actual_output_path = noaudio_output_path
                logger.warning(f"Task {task_id}: Only video without audio was created. Using {noaudio_output_path}")
            else:
                raise Exception("No output video file was created")

            # Update the task with the actual output path
            with self._lock:
                if task_id in self.tasks:
                    self.tasks[task_id].output_path = actual_output_path
                    self.tasks[task_id].update_status(TaskStatus.COMPLETED, 1.0, "Synthesis completed successfully")

            logger.info(f"Task {task_id} completed successfully. Output: {actual_output_path}")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Task {task_id} failed: {error_msg}")
            
            with self._lock:
                if task_id in self.tasks:
                    self.tasks[task_id].update_status(TaskStatus.FAILED, message=f"Task failed: {error_msg}")
                    self.tasks[task_id].error = error_msg
        
        finally:
            # Clean up any temporary crop files if created
            try:
                if crop and work_image_path and work_image_path.endswith('.crop.png'):
                    if os.path.exists(work_image_path):
                        os.remove(work_image_path)
                        logger.info(f"Cleaned up crop file for task {task_id}")
            except Exception as e:
                logger.warning(f"Failed to clean up crop file for task {task_id}: {e}")
            
            # Note: Upload file cleanup is handled by scheduled tasks, not here

    def shutdown(self):
        """Shutdown the task manager."""
        self.executor.shutdown(wait=True)


# Global task manager instance
task_manager = TaskManager() 