from pathlib import Path
import os

# ========== API Keys ==========
# These are injected into environment variables so that all OpenAI() / OpenRouter
# clients across the project pick them up automatically.
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENROUTER_API_KEY"] = "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"

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
