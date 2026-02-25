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

# Logs Directory
LOGS_DIR = BASE_DIR / "logs"

# Specific File Paths
TEMPLATE_PATH = ASSETS_DIR / "master_template.pptx"

# Ensure crucial directories exist
# We do not force creation of ASSETS_DIR or TEMPLATE_PATH as they should exist, but we ensure output dirs.
DATA_DIR.mkdir(exist_ok=True)
GENERATED_DIR.mkdir(exist_ok=True)
SLIDES_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
