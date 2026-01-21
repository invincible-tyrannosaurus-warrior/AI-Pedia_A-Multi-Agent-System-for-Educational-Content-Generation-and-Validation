
from __future__ import annotations
import sys
import os
import shutil
import json
from pathlib import Path
from typing import List

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Ensure we can find the manager_agent
# Assuming this script is in archive/
sys.path.append(str(Path(__file__).parent))

try:
    from manager_agent import task_manager_agent
except ImportError:
    # Try importing from root if run as module
    pass

app = FastAPI()

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=templates_dir)

DATA_DIR = Path(__file__).parent / "data" / "uploads"
DATA_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    uploaded_paths = []
    
    for file in files:
        # Secure filename? For demo, simple is fine.
        file_path = DATA_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # We return absolute path for the agent to use
        uploaded_paths.append(str(file_path.absolute()))
        
    return {"paths": uploaded_paths}

@app.get("/stream_generate")
async def stream_generate(topic: str, files: str = "[]", video: bool = True, slides: bool = True, code: bool = True, quizzes: bool = True):
    import json
    
    # Parse files JSON from query param
    # handling potentially malformed json or empty string
    if not files:
        files = "[]"
        
    config = {
        "video": video,
        "slides": slides,
        "code": code,
        "quizzes": quizzes
    }
    
    # We need to make sure we import the agent correctly if the top-level import failed
    from manager_agent import task_manager_agent
    
    return StreamingResponse(
        task_manager_agent.stream_workflow(topic, config, files_json=files),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
