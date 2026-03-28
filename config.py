from pathlib import Path
import os

# ========== API Keys ==========
# API keys are read from environment variables (not hardcoded).
# Set them before running:
#   Windows:  $env:OPENAI_API_KEY = "sk-proj-..."
#             $env:OPENROUTER_API_KEY = "sk-or-..."
#   Docker:   docker run -e OPENAI_API_KEY="..." -e OPENROUTER_API_KEY="..." ...



# Base Directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Data Directories
DATA_DIR = BASE_DIR / "data"
GENERATED_DIR = DATA_DIR / "generated_code"
SLIDES_DIR = DATA_DIR / "slides"
ASSETS_DIR = DATA_DIR / "assets"
UPLOADS_DIR = DATA_DIR / "uploads"

# Logs Directory
LOGS_DIR = BASE_DIR / "logs"

# Specific File Paths
TEMPLATE_PATH = ASSETS_DIR / "master_template.pptx"

# Ensure crucial directories exist
# We do not force creation of ASSETS_DIR or TEMPLATE_PATH as they should exist, but we ensure output dirs.
DATA_DIR.mkdir(exist_ok=True)
GENERATED_DIR.mkdir(exist_ok=True)
SLIDES_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


def artifact_to_public_ref(path: Path | str) -> str:
    candidate = Path(path).resolve()
    if candidate == GENERATED_DIR.resolve() or GENERATED_DIR.resolve() in candidate.parents:
        relative = candidate.relative_to(GENERATED_DIR.resolve()).as_posix()
        return f"/generated/{relative}"
    if candidate == UPLOADS_DIR.resolve() or UPLOADS_DIR.resolve() in candidate.parents:
        relative = candidate.relative_to(UPLOADS_DIR.resolve()).as_posix()
        return f"/uploads/{relative}"
    raise ValueError(f"Unsupported artifact path: {candidate}")


def resolve_artifact_ref(ref: str) -> Path:
    normalized = (ref or "").strip()
    if not normalized:
        raise ValueError("Artifact reference is required.")

    if normalized.startswith("/generated/"):
        return (GENERATED_DIR / normalized[len("/generated/") :].lstrip("/")).resolve()
    if normalized.startswith("generated/"):
        return (GENERATED_DIR / normalized[len("generated/") :].lstrip("/")).resolve()
    if normalized.startswith("/uploads/"):
        return (UPLOADS_DIR / normalized[len("/uploads/") :].lstrip("/")).resolve()
    if normalized.startswith("uploads/"):
        return (UPLOADS_DIR / normalized[len("uploads/") :].lstrip("/")).resolve()

    return Path(normalized).expanduser().resolve()
