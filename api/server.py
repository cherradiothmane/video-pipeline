"""
FastAPI REST API for the video processing pipeline.
Supports single video, batch processing, and real-time WebSocket progress.
"""
import asyncio
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Adjust path so we can import pipeline
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import PipelineConfig, PipelineResult, VideoPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Video Pipeline API",
    description="Offline video transcription, summarization and OCR pipeline",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("./uploads")
OUTPUT_DIR = Path("./outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Global pipeline instance (lazy-initialized)
_pipeline: Optional[VideoPipeline] = None

# In-memory job store: job_id -> status dict
_jobs: dict[str, dict] = {}

# WebSocket connections: job_id -> list of websockets
_ws_connections: dict[str, list[WebSocket]] = {}


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
            output_dir=str(OUTPUT_DIR),
        )
        _pipeline = VideoPipeline(config)
    return _pipeline


# ─── Models ───────────────────────────────────────────────────────────────────

class JobStatus(BaseModel):
    job_id: str
    status: str           # pending | processing | done | failed
    video_name: str
    step: Optional[str] = None
    step_progress: Optional[int] = None
    error: Optional[str] = None
    result_path: Optional[str] = None


class BatchRequest(BaseModel):
    video_paths: list[str]
    whisper_language: Optional[str] = None


# ─── Helper functions ─────────────────────────────────────────────────────────

async def _broadcast(job_id: str, message: dict):
    """Send progress update to all WebSocket clients watching this job."""
    conns = _ws_connections.get(job_id, [])
    dead = []
    for ws in conns:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        conns.remove(ws)


def _make_progress_callback(job_id: str, loop: asyncio.AbstractEventLoop):
    def callback(step: str, pct: int):
        _jobs[job_id]["step"] = step
        _jobs[job_id]["step_progress"] = pct
        try:
            asyncio.run_coroutine_threadsafe(
                _broadcast(job_id, {"job_id": job_id, "step": step, "progress": pct}),
                loop,
            )
        except Exception:
            pass
    return callback


def _run_pipeline_job(job_id: str, video_path: str, loop: asyncio.AbstractEventLoop):
    """Run in a thread pool (blocking)."""
    _jobs[job_id]["status"] = "processing"
    pipeline = get_pipeline()
    cb = _make_progress_callback(job_id, loop)

    try:
        result: PipelineResult = pipeline.process_video(video_path, progress_callback=cb)
        _jobs[job_id]["status"] = result.status
        _jobs[job_id]["error"] = result.error
        if result.status == "done":
            result_json_path = OUTPUT_DIR / result.video_name / "result.json"
            _jobs[job_id]["result_path"] = str(result_json_path)
        asyncio.run_coroutine_threadsafe(
            _broadcast(job_id, {"job_id": job_id, "status": result.status, "done": True}),
            loop,
        )
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)
        asyncio.run_coroutine_threadsafe(
            _broadcast(job_id, {"job_id": job_id, "status": "failed", "error": str(e)}),
            loop,
        )


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/models/status")
async def models_status():
    """Check if Ollama and models are available."""
    pipeline = get_pipeline()
    ollama_ok = pipeline.summarizer.check_connection() if pipeline.summarizer else False
    return {
        "ollama": ollama_ok,
        "ollama_model": os.getenv("OLLAMA_MODEL", "qwen2.5:32b"),
        "whisper_model": os.getenv("WHISPER_MODEL", "large-v3"),
    }


@app.post("/jobs/upload", response_model=JobStatus)
async def upload_and_process(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
):
    """Upload a video file and start processing."""
    # Validate extension
    allowed = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".ts", ".m4v"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    # Save upload
    job_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{job_id}{suffix}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Register job
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "video_name": Path(file.filename).stem,
        "step": None,
        "step_progress": None,
        "error": None,
        "result_path": None,
    }
    _ws_connections[job_id] = []

    loop = asyncio.get_event_loop()
    background_tasks.add_task(
        asyncio.get_event_loop().run_in_executor,
        None,
        _run_pipeline_job,
        job_id,
        str(dest),
        loop,
    )

    return JobStatus(**_jobs[job_id])


@app.post("/jobs/path", response_model=JobStatus)
async def process_by_path(
    background_tasks: BackgroundTasks,
    video_path: str = Form(...),
    language: Optional[str] = Form(None),
):
    """Start processing a video already on the server by path."""
    if not Path(video_path).exists():
        raise HTTPException(404, f"Video not found: {video_path}")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "video_name": Path(video_path).stem,
        "step": None,
        "step_progress": None,
        "error": None,
        "result_path": None,
    }
    _ws_connections[job_id] = []

    loop = asyncio.get_event_loop()
    background_tasks.add_task(
        asyncio.get_event_loop().run_in_executor,
        None,
        _run_pipeline_job,
        job_id,
        video_path,
        loop,
    )
    return JobStatus(**_jobs[job_id])


@app.get("/jobs", response_model=list[JobStatus])
async def list_jobs():
    return [JobStatus(**j) for j in _jobs.values()]


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    return JobStatus(**_jobs[job_id])


@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    job = _jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(400, f"Job is not done (status={job['status']})")
    result_path = job.get("result_path")
    if not result_path or not Path(result_path).exists():
        raise HTTPException(404, "Result file not found")
    with open(result_path) as f:
        return JSONResponse(json.load(f))


@app.get("/jobs/{job_id}/download/{filename}")
async def download_output(job_id: str, filename: str):
    """Download a specific output file (result.json, .srt, summary.md, etc.)"""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    video_name = _jobs[job_id]["video_name"]
    file_path = OUTPUT_DIR / video_name / filename
    if not file_path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(file_path), filename=filename)


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    _jobs.pop(job_id, None)
    _ws_connections.pop(job_id, None)
    return {"deleted": job_id}


# ─── WebSocket for real-time progress ─────────────────────────────────────────

@app.websocket("/ws/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    if job_id not in _jobs:
        await websocket.send_json({"error": "Job not found"})
        await websocket.close()
        return

    if job_id not in _ws_connections:
        _ws_connections[job_id] = []
    _ws_connections[job_id].append(websocket)

    # Send current state immediately
    await websocket.send_json(_jobs[job_id])

    try:
        while True:
            # Keep connection alive; updates are pushed via _broadcast
            await asyncio.sleep(1)
            job = _jobs.get(job_id, {})
            if job.get("status") in ("done", "failed"):
                await websocket.send_json(job)
                break
    except WebSocketDisconnect:
        pass
    finally:
        if job_id in _ws_connections:
            try:
                _ws_connections[job_id].remove(websocket)
            except ValueError:
                pass


# ─── Serve frontend ───────────────────────────────────────────────────────────

UI_DIR = Path(__file__).parent.parent / "ui" / "dist"
if UI_DIR.exists():
    app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,  # Single worker since we use global state
    )
