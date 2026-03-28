from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import hashlib
import uvicorn
import json
import asyncio
import os

# Import your agent orchestrator
from manager_agent import task_manager_agent
from config import DATA_DIR, GENERATED_DIR
from moe_layer.video_agent.ppt_converter import PPTConverter

app = FastAPI()

# Setup paths
current_dir = Path(__file__).resolve().parent
static_dir = current_dir / "static"
templates_dir = current_dir / "templates"
uploads_dir = current_dir / "static" / "data" / "uploads"
ALLOWED_ARTIFACT_ROOTS = [
    GENERATED_DIR.resolve(),
    DATA_DIR.resolve(),
    uploads_dir.resolve(),
]

# Mount static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.mount("/generated", StaticFiles(directory=GENERATED_DIR), name="generated")

templates = Jinja2Templates(directory=templates_dir)


def _resolve_artifact_path(raw_path: str) -> Path:
    if not raw_path:
        raise HTTPException(status_code=400, detail="Artifact path is required.")

    normalized = raw_path.strip()
    if normalized.startswith("/generated/"):
        relative = normalized[len("/generated/"):].lstrip("/")
        candidate = GENERATED_DIR / relative
    elif normalized.startswith("/static/"):
        candidate = current_dir / normalized.lstrip("/")
    else:
        candidate = Path(normalized)
        if not candidate.is_absolute():
            candidate = current_dir / candidate

    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Artifact not found.") from exc

    if not any(resolved == root or root in resolved.parents for root in ALLOWED_ARTIFACT_ROOTS):
        raise HTTPException(status_code=403, detail="Artifact path is not allowed.")

    return resolved

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.get("/artifact_file")
async def artifact_file(path: str = Query(...), download: bool = False):
    artifact_path = _resolve_artifact_path(path)
    suffix = artifact_path.suffix.lower()
    media_type = None
    if not download and suffix in {".py", ".json", ".txt", ".md"}:
        media_type = "text/plain; charset=utf-8"
    return FileResponse(
        path=artifact_path,
        filename=artifact_path.name,
        media_type=media_type,
        content_disposition_type="attachment" if download else "inline",
    )


@app.get("/presentation_preview")
async def presentation_preview(path: str = Query(...)):
    ppt_path = _resolve_artifact_path(path)
    if ppt_path.suffix.lower() != ".pptx":
        raise HTTPException(status_code=400, detail="Presentation preview only supports .pptx files.")

    preview_key = hashlib.sha1(
        f"{ppt_path.resolve()}::{ppt_path.stat().st_mtime_ns}".encode("utf-8")
    ).hexdigest()[:16]
    preview_dir = GENERATED_DIR / "_preview_cache" / preview_key
    preview_dir.mkdir(parents=True, exist_ok=True)

    slide_images = sorted(preview_dir.glob("slide_*.png"))
    if not slide_images:
        try:
            converter = PPTConverter()
            slide_images = converter.convert_to_images(ppt_path, preview_dir)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to build presentation preview: {exc}") from exc

    slide_urls = [
        f"/generated/_preview_cache/{preview_key}/{image_path.name}"
        for image_path in slide_images
    ]
    return JSONResponse(
        {
            "name": ppt_path.name,
            "slide_count": len(slide_urls),
            "slides": slide_urls,
        }
    )

@app.post("/upload")
async def upload_files(request: Request):
    form = await request.form()
    files = form.getlist("files")  # List of UploadFile objects
    saved_paths = []
    
    upload_dir = uploads_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    for file in files:
        if file.filename:
            # Secure filename: strip paths, keep basename
            safe_filename = Path(file.filename).name
            path = upload_dir / safe_filename
            with open(path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            saved_paths.append(str(path))
            
    return {"paths": saved_paths}

@app.get("/stream_generate")
async def stream_generate(topic: str, files: str = "[]", video: bool = True, slides: bool = True, code: bool = True, quizzes: bool = True):
    config = {
        "video": video,
        "slides": slides,
        "code": code,
        "quizzes": quizzes
    }
    
    # Use the orchestrator's generator
    return StreamingResponse(
        task_manager_agent.stream_workflow(topic, config, files_json=files),
        media_type="text/event-stream"
    )

@app.get("/refine_stream")
async def refine_generate(run_id: str, task_id: str, feedback: str):
    return StreamingResponse(
        task_manager_agent.refine_stream(run_id, task_id, feedback),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    is_dev = os.getenv("DEV_MODE", "false").lower() == "true"
    uvicorn.run(
        "demo_front_end:app",
        host="0.0.0.0",
        port=8000,
        reload=is_dev,
        reload_excludes=["data/*", "*.pptx", "*.mp4", "*.pdf"] if is_dev else [],
    )

