"""VoidEnhancer web API.

Jobs run in a small background worker so long videos do not block the HTTP
request. The AI pipeline stays in voidenhancer.video.
"""
from __future__ import annotations

import shutil
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from voidenhancer.enhancement import EnhancementOptions
from voidenhancer.video import enhance_video

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
CHECKPOINT = ROOT / "checkpoints" / "voidenhancer.pt"
WORK_DIR = ROOT / "runtime" / "jobs"
WORK_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024 * 1024
MAX_URL_BYTES = 8 * 1024 * 1024 * 1024
executor = ThreadPoolExecutor(max_workers=1)
jobs: dict[str, dict] = {}
jobs_lock = Lock()

app = FastAPI(title="VoidEnhancer")
app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")


class UrlRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2048)
    scale: int = Field(default=4, ge=2, le=4)


def set_job(job_id: str, **values) -> None:
    with jobs_lock:
        jobs.setdefault(job_id, {}).update(values)


def run_job(job_id: str, input_path: Path, output_path: Path, scale: int) -> None:
    try:
        set_job(job_id, status="processing", progress=0, message="Enhancing video")
        options = EnhancementOptions(scale=scale)
        total = 0

        def progress(done: int, count: int) -> None:
            nonlocal total
            total = count
            value = round(done * 100 / count) if count else 0
            set_job(job_id, progress=min(100, value), message=f"Processing frame {done}" if count else "Processing video")

        enhance_video(str(CHECKPOINT), str(input_path), str(output_path), scale=scale,
                      device_name="auto", options=options, progress=progress)
        set_job(job_id, status="completed", progress=100, message="Complete", total_frames=total)
    except Exception as exc:
        set_job(job_id, status="failed", progress=0, message=str(exc))
    finally:
        input_path.unlink(missing_ok=True)


def submit_job(input_path: Path, scale: int) -> str:
    job_id = uuid.uuid4().hex
    job_dir = input_path.parent
    output_path = job_dir / f"voidenhancer_{scale}x.mp4"
    set_job(job_id, status="queued", progress=0, message="Waiting for worker", output=str(output_path), scale=scale)
    executor.submit(run_job, job_id, input_path, output_path, scale)
    return job_id


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return (WEB_DIR / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "model_ready": CHECKPOINT.exists(), "worker_slots": 1}


@app.post("/api/enhance")
async def enhance(file: UploadFile = File(...), scale: int = 4) -> dict:
    if not CHECKPOINT.exists():
        raise HTTPException(status_code=503, detail="AI checkpoint is not available on this server yet.")
    if scale not in (2, 4):
        raise HTTPException(status_code=400, detail="Scale must be 2 or 4.")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(status_code=400, detail="Unsupported video format.")
    job_dir = WORK_DIR / uuid.uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / f"input{suffix}"
    size = 0
    try:
        with input_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Video is larger than 8 GB.")
                destination.write(chunk)
        job_id = submit_job(input_path, scale)
        return {"job_id": job_id, "status": "queued"}
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Could not create job: {exc}") from exc


@app.post("/api/enhance/url")
def enhance_url(request: UrlRequest) -> dict:
    if not CHECKPOINT.exists():
        raise HTTPException(status_code=503, detail="AI checkpoint is not available on this server yet.")
    if not request.url.lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Only HTTP(S) video URLs are supported.")
    job_dir = WORK_DIR / uuid.uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=True)
    source = job_dir / "source.%(ext)s"
    try:
        command = ["yt-dlp", "--no-playlist", "--max-filesize", str(MAX_URL_BYTES),
                   "-f", "bv*+ba/b", "--merge-output-format", "mp4", "-o", str(source), request.url]
        result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="Could not download the supplied video URL.")
        candidates = [p for p in job_dir.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED]
        if not candidates:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="The URL did not produce a supported video file.")
        job_id = submit_job(candidates[0], request.scale)
        return {"job_id": job_id, "status": "queued"}
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=408, detail="Video download timed out.") from exc


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    data = {k: v for k, v in job.items() if k != "output"}
    if job.get("status") == "completed":
        data["download_url"] = f"/api/jobs/{job_id}/download"
    return data


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str) -> FileResponse:
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Enhanced video is not ready.")
    output = Path(job["output"])
    if not output.exists():
        raise HTTPException(status_code=404, detail="Output file no longer exists.")
    return FileResponse(output, media_type="video/mp4", filename=f"voidenhancer_{job['scale']}x.mp4")
