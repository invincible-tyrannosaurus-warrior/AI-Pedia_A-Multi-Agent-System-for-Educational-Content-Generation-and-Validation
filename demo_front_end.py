from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uvicorn
import json
import asyncio
import os

# Import your agent orchestrator
from manager_agent import task_manager_agent
from config import GENERATED_DIR

app = FastAPI()

# Setup paths
current_dir = Path(__file__).resolve().parent
static_dir = current_dir / "static"
templates_dir = current_dir / "templates"

# Mount static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.mount("/generated", StaticFiles(directory=GENERATED_DIR), name="generated")

templates = Jinja2Templates(directory=templates_dir)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )

@app.post("/upload")
async def upload_files(request: Request):
    form = await request.form()
    files = form.getlist("files")  # List of UploadFile objects
    saved_paths = []
    
    upload_dir = current_dir / "static" / "data" / "uploads"
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

