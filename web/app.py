"""Minimal production-shaped web entrypoint for VoidEnhancer.

The UI is intentionally small until the trained model is validated. Processing
runs through the same inference pipeline used by the CLI, so the website does
not contain a second copy of the AI logic.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from voidenhancer.video import enhance_video

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
CHECKPOINT = ROOT / "checkpoints" / "voidenhancer.pt"
WORK_DIR = ROOT / "runtime" / "jobs"
WORK_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
MAX_UPLOAD_BYTES = 250 * 1024 * 1024

app = FastAPI(title="VoidEnhancer", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return (WEB_DIR / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "model_ready": CHECKPOINT.exists()}


@app.post("/api/enhance")
async def enhance(file: UploadFile = File(...)) -> FileResponse:
    if not CHECKPOINT.exists():
        raise HTTPException(status_code=503, detail="The model is not trained yet.")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(status_code=400, detail="Unsupported video format.")

    job_id = uuid.uuid4().hex
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / f"input{suffix}"
    output_path = job_dir / "voidenhancer_4x.mp4"

    size = 0
    try:
        with input_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Video is larger than 250 MB.")
                destination.write(chunk)

        enhance_video(str(CHECKPOINT), str(input_path), str(output_path), scale=4, device_name="auto")
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename="voidenhancer_4x.mp4",
            background=None,
        )
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {exc}") from exc
