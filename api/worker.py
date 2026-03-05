"""
Celery worker for async/distributed batch video processing.
Usage:
    celery -A api.worker worker --loglevel=info --concurrency=1
"""
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from celery import Celery
from pipeline import PipelineConfig, VideoPipeline

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "video_pipeline",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # Process one job at a time (GPU exclusive)
)

# Lazy pipeline — initialized once per worker process
_pipeline = None


def get_pipeline() -> VideoPipeline:
    global _pipeline
    if _pipeline is None:
        config = PipelineConfig(
            whisper_model=os.getenv("WHISPER_MODEL", "large-v3"),
            whisper_device=os.getenv("WHISPER_DEVICE", "cuda"),
            whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "float16"),
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:32b"),
            llm_language=os.getenv("LLM_LANGUAGE", "fr"),
            ocr_enabled=os.getenv("OCR_ENABLED", "true").lower() == "true",
            output_dir=os.getenv("OUTPUT_DIR", "./outputs"),
        )
        _pipeline = VideoPipeline(config)
        _pipeline.initialize()
    return _pipeline


@celery_app.task(bind=True, name="pipeline.process_video")
def process_video_task(self, video_path: str, language: str = None):
    """
    Celery task to process a single video.
    
    Args:
        video_path: Path to the video file
        language: Optional ISO language code for Whisper
    """
    self.update_state(state="STARTED", meta={"step": "init", "progress": 0})

    def progress_callback(step: str, pct: int):
        self.update_state(
            state="PROGRESS",
            meta={"step": step, "progress": pct},
        )
        logger.info(f"[{self.request.id}] {step}: {pct}%")

    pipeline = get_pipeline()
    if language:
        pipeline.config.whisper_language = language

    result = pipeline.process_video(video_path, progress_callback=progress_callback)

    if result.status == "failed":
        raise Exception(result.error)

    return {
        "status": "done",
        "video_name": result.video_name,
        "duration": result.duration_seconds,
        "processing_time": result.processing_time_seconds,
        "title": result.summary.title if result.summary else None,
        "category": result.summary.category if result.summary else None,
    }


@celery_app.task(bind=True, name="pipeline.process_batch")
def process_batch_task(self, video_paths: list[str], language: str = None):
    """
    Celery task to process a batch of videos sequentially.
    """
    results = []
    total = len(video_paths)

    for i, path in enumerate(video_paths):
        self.update_state(
            state="PROGRESS",
            meta={"current": i + 1, "total": total, "current_file": path},
        )
        # Dispatch individual tasks
        task_result = process_video_task.delay(path, language)
        results.append({"video": path, "task_id": task_result.id})

    return {"batch_size": total, "tasks": results}
