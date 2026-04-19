# AI Pedia Local Stream

AI Pedia Local Stream is a local-first, multi-agent educational content generation system.
It transforms a user topic (and optional uploaded files) into coordinated learning assets:

- Python code demos
- PowerPoint slides
- narrated lecture videos
- quizzes

The system combines an orchestrator, specialized agent pipelines, deterministic validation,
and a web UI with real-time streaming progress.

## Table of Contents

- [1. Core Features](#1-core-features)
- [2. System Architecture](#2-system-architecture)
- [3. Repository Structure](#3-repository-structure)
- [4. Execution Flow](#4-execution-flow)
- [5. Agent Responsibilities](#5-agent-responsibilities)
- [6. API Endpoints](#6-api-endpoints)
- [7. Streaming Event Protocol](#7-streaming-event-protocol)
- [8. Configuration](#8-configuration)
- [9. Installation](#9-installation)
- [10. Running the System](#10-running-the-system)
- [11. Docker Deployment](#11-docker-deployment)
- [12. Output and Logs](#12-output-and-logs)
- [13. RAG and MCP Tools](#13-rag-and-mcp-tools)
- [14. Troubleshooting](#14-troubleshooting)
- [15. License](#15-license)

## 1. Core Features

- Multi-agent orchestration with task dependency handling
- Real-time UI updates via Server-Sent Events (SSE)
- Automated code generation and runtime validation
- Deterministic slide building from storyboard JSON
- PPTX-to-video generation with slide narration
- Quiz generation with structure validation and retry logic
- Judger loop with deterministic and semantic acceptance checks
- Local artifact persistence and run-level log tracking
- Optional retrieval-augmented context using local vector store

## 2. System Architecture

The architecture has five layers:

1. Web layer (FastAPI + HTML/CSS/JS)
2. Task orchestration layer (`manager_agent`)
3. Specialized agent pipelines (`moe_layer/*`)
4. Validation layer (`judger_agent`)
5. Tooling layer (`ai_pedia_mcp_server`, local RAG, python checker)

High-level flow:

1. User submits topic and optional files in the web UI.
2. Files are persisted, text is extracted, and optional RAG ingestion is performed.
3. Task Manager produces task steps and executes agent pipelines.
4. Agent outputs are judged against acceptance criteria.
5. Failed tasks can be retried with fix instructions.
6. Artifacts are streamed back to the UI and stored under run directories.

## 3. Repository Structure

```text
AI_Pedia_Local_stream-main/
  demo_front_end.py                     # FastAPI app and streaming endpoints
  config.py                             # Paths and artifact reference helpers
  manager_agent/
    task_manager_agent.py               # Main orchestration logic
  judger_agent/
    judger_pipeline.py                  # Acceptance criteria evaluation
  moe_layer/
    orchestrator/agent_registry.py      # Agent registry
    coder_agent/                        # Code generation + validation
    presentation_agent/                 # Storyboard + deterministic PPT builder
    video_agent/                        # PPT conversion + narration + video composition
    quizzer_agent/                      # Quiz generation and validation
    text_generator_agent/               # Placeholder text pipeline
  ai_pedia_mcp_server/
    mcp_tools/python_checker.py         # Python compile/run checker
    mcp_tools/rag_search.py             # Local vector search
  templates/index.html                  # UI
  static/css/style.css                  # UI styling
  data/
    uploads/                            # Uploaded files
    generated_code/                     # Run outputs
    vector_store/                       # Chroma persistent store
  logs/
    task_manager/                       # Per-run orchestration logs
    judger/                             # Per-run judging rounds
```

## 4. Execution Flow

### 4.1 Request Entry

- Endpoint: `GET /stream_generate`
- Parameters include:
  - `topic`
  - `files` (JSON list of uploaded file refs)
  - feature toggles (`video`, `slides`, `code`, `quizzes`)

### 4.2 File Handling

- Upload endpoint: `POST /upload`
- Files are saved with sanitized unique names.
- Text extraction:
  - PDF: `pypdf` fallback to `pdfplumber`
  - image: OCR with `pytesseract` if available
- Extracted text can be ingested to Chroma via `RAGEngine`.

### 4.3 Planning and Task Build

`stream_workflow` creates normalized task steps per selected output type:

- `presentation`
- `video` (depends on presentation when slides are enabled)
- `coder`
- `quizzer`

Each task carries acceptance criteria used by the Judger.

### 4.4 Agent Execution

- Dependency-aware scheduling
- Retry loop per task
- Async handling for coroutine pipelines
- Video pipeline emits detailed stage progress

### 4.5 Judging and Retry

Judger evaluates each task using:

- deterministic criteria (`output_shape`, `file_exists`, `mcp_tool`)
- optional semantic criteria through LLM

If failed, fix instructions are injected and the task is retried.

### 4.6 Finalization

- Artifacts are emitted to UI as SSE events.
- `workflow_complete` returns final status.
- Run data is persisted to `logs/task_manager/<run_id>.json`.

## 5. Agent Responsibilities

### Task Manager (`manager_agent/task_manager_agent.py`)

- registers available agents
- builds and executes task graph
- handles dependencies and retries
- streams structured events
- supports artifact refinement (`refine_stream`)

### Coder Agent (`moe_layer/coder_agent`)

- model: OpenRouter (`qwen/qwen3-coder-flash` by default)
- strict filesystem rules injected into prompts
- output file persisted under run `scripts/`
- validation uses MCP `python_check`

### Presentation Agent (`moe_layer/presentation_agent`)

- uses LLM to generate a storyboard JSON
- deterministic PPT builder consumes storyboard
- supports title/content/two-column layouts
- can render code/chart/formula visuals to images

### Video Agent (`moe_layer/video_agent`)

- resolves source PPTX from dependency outputs or assets
- converts PPTX to images (`libreoffice` + `PyMuPDF`)
- writes narration script per slide (`gpt-4o` vision)
- composes final MP4 (`edge-tts` + `ffmpeg`)
- reports stage progress:
  - `prepare`
  - `convert_slides`
  - `generate_scripts`
  - `build_clips`
  - `encode_video`

### Quizzer Agent (`moe_layer/quizzer_agent`)

- generates quiz with strict schema constraints:
  - exactly 10 questions
  - 4 options per question
  - answer in `A/B/C/D`
- validates and auto-normalizes answers
- retries generation when validation fails

### Text Agent (`moe_layer/text_generator_agent`)

- currently a placeholder pipeline
- not actively registered in default workflow

## 6. API Endpoints

Defined in `demo_front_end.py`:

- `GET /`
  - returns main web UI
- `POST /upload`
  - uploads one or more files
  - returns persisted public paths
- `GET /stream_generate`
  - starts streaming generation workflow
- `GET /refine_stream`
  - reruns a specific task with user feedback
- `GET /artifact_file?path=...`
  - serves generated/uploaded artifact (inline or attachment)
- `GET /presentation_preview?path=...`
  - converts PPTX into preview images and returns slide URLs

## 7. Streaming Event Protocol

Main generation stream emits:

- `plan`
- `step_start`
- `log`
- `artifact`
- `video_progress`
- `quiz`
- `step_complete`
- `workflow_complete`
- `error`

Refinement stream emits:

- `log`
- `artifact`
- `complete`
- `error`

## 8. Configuration

Core paths are defined in `config.py`:

- `DATA_DIR`
- `GENERATED_DIR`
- `SLIDES_DIR`
- `ASSETS_DIR`
- `UPLOADS_DIR`
- `LOGS_DIR`
- `TEMPLATE_PATH`

Environment variables used by the system:

- `OPENAI_API_KEY` (required for planner, judger, presentation, quiz, video script writing)
- `OPENROUTER_API_KEY` or `CODER_API_KEY` (required for coder model)
- `CODER_MODEL` (default: `qwen/qwen3-coder-flash`)
- `CODER_BASE_URL` (default: `https://openrouter.ai/api/v1`)
- `TASK_MANAGER_MODEL` (default: `gpt-5.2`)
- `JUDGER_MODEL` (default: `gpt-5.2`)
- `TASK_PLAN_MAX_RETRIES` (default: `3`)
- `TASK_MAX_RETRIES` (default: `3`)
- `DEV_MODE` (`true` enables uvicorn reload mode)

## 9. Installation

### 9.1 Prerequisites

- Python 3.11+
- `ffmpeg` on `PATH`
- `libreoffice` on `PATH` (for PPTX to PDF conversion)

Optional:

- Tesseract OCR binary (for image text extraction)

### 9.2 Python Environment

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
# Linux/macOS
source .venv/bin/activate
```

### 9.3 Install Dependencies

```bash
pip install -r requirements.txt
```

Optional RAG extras:

```bash
pip install -r requirements-rag.txt
```

## 10. Running the System

### 10.1 Local Run

Set keys first:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your_openai_key"
$env:OPENROUTER_API_KEY="your_openrouter_key"
```

Start server:

```bash
python demo_front_end.py
```

Open:

`http://localhost:8000`

### 10.2 Uvicorn Direct Run

```bash
uvicorn demo_front_end:app --host 0.0.0.0 --port 8000
```

## 11. Docker Deployment

Build:

```bash
docker build -t ai-pedia-local-stream .
```

Run:

```bash
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY="your_openai_key" \
  -e OPENROUTER_API_KEY="your_openrouter_key" \
  ai-pedia-local-stream
```

## 12. Output and Logs

### 12.1 Generated Artifacts

Artifacts are grouped per run:

`data/generated_code/run_<run_id>/`

Typical structure:

- `scripts/` (generated Python scripts)
- `assets/` (plots, images, intermediate assets)
- `output/` (video, quiz JSON, temp clips)
- `storyboard.json` (presentation intermediate)

### 12.2 Logs

- `logs/task_manager/<run_id>.json`
  - run metadata
  - final plan
  - per-task results
  - overall status
- `logs/judger/<run_id>.json`
  - judging rounds
  - per-round criteria evidence and verdicts

## 13. RAG and MCP Tools

### 13.1 `python_check`

Location:

`ai_pedia_mcp_server/mcp_tools/python_checker.py`

Behavior:

- compile check via `py_compile`
- runtime execution with timeout
- structured result with stdout/stderr
- error log append to JSONL on failure

### 13.2 `rag_query`

Location:

`ai_pedia_mcp_server/mcp_tools/rag_search.py`

Behavior:

- persistent Chroma collection (`data/vector_store`)
- sentence-transformer embeddings (`all-MiniLM-L6-v2`)
- chunked retrieval for contextual augmentation

## 14. Troubleshooting

### Missing API Key

Symptoms:

- planner/judger/presentation/quiz/video script calls fail
- coder model call fails

Fix:

- set `OPENAI_API_KEY`
- set `OPENROUTER_API_KEY` or `CODER_API_KEY`

### Video Pipeline Runtime Errors

Symptoms:

- no slide images generated
- ffmpeg or libreoffice failures

Fix:

- install `libreoffice` and ensure it is on `PATH`
- install `ffmpeg` and ensure it is on `PATH`
- verify source PPTX exists and is valid

### OCR Unavailable

Symptoms:

- image upload yields no extracted text

Fix:

- install Tesseract OCR runtime
- install Python packages from requirements

### RAG Search Empty or Failing

Fix:

- install `requirements-rag.txt`
- verify `data/vector_store` write permissions
- ensure uploaded text extraction succeeded

## 15. License

This project is licensed under Apache License 2.0.
See `LICENSE` for full text.
