"""Tools for prompting Qwen3 Coder Flash through OpenRouter and validating outputs."""
# qwen3 coder flash: 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 (OpenRouter currently using This) 

# in terminal: $env:OPENROUTER_API_KEY = "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"

from __future__ import annotations 

import logging 
import os
import re
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI

# bridge between coder agent and mcp server, so that we can use python_check tool to validate generated code 
from ai_pedia_mcp_server.client import MCPClientSync

logger = logging.getLogger(__name__)


# alerting qwen3 coder flash to return code only. 
SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a helpful coding assistant that generates runnable Python scripts.
    Produce complete, well-structured code that doubles as a teaching reference.
    Include meaningful inline comments where they clarify intent.
    IMPORTANT: Return code only—no narrative explanation or markdown fencing.
    """
).strip()

# set model and base url
DEFAULT_MODEL = os.getenv("CODER_MODEL", "qwen/qwen3-coder-flash")
DEFAULT_BASE_URL = os.getenv("CODER_BASE_URL", "https://openrouter.ai/api/v1")


# api key should be has been set in environment variable
# use this function to get it

def _get_openrouter_key() -> str:
    key = os.getenv("CODER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not key: # alert if cant find it
        raise EnvironmentError(
            "Missing OpenRouter API key. Set CODER_API_KEY or OPENROUTER_API_KEY."
        )
    return key


# Instantiate the OpenRouter-configured OpenAI client for coder calls.
def _default_client() -> OpenAI:
    key = _get_openrouter_key()
    return OpenAI(base_url=DEFAULT_BASE_URL, api_key=key)



# Generate code using Qwen3 Coder Flash
def _strip_code_fences(code: str) -> str:
    # Remove Markdown code fences from the model output if present.
    trimmed = code.strip()
    if trimmed.startswith("```"):
        # Split once on the opening fence; expected form ```python\n...```
        parts = trimmed.split("```", 2)
        if len(parts) >= 3:
            trimmed = parts[1]
        else:
            trimmed = trimmed.lstrip("`")
    if trimmed.startswith("python\n"):
        trimmed = trimmed[len("python\n") :]
    return trimmed.strip("\n")


def generate_code(guidance: str, client: Optional[OpenAI] = None) -> str:
    if not guidance.strip():
        raise ValueError("Guidance must be a non-empty string.")

    client = client or _default_client()  # use OpenRouter-configured client or default

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        # feed in the prompt (system & user)
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT}, 
            {"role": "user", "content": guidance},
        ],
        temperature=0.2,  # aim for deterministic output
    )

    try:
        code = response.choices[0].message.content # extract the code from response
    except (AttributeError, IndexError) as exc:  # pragma: no cover
        logger.error("Unexpected coder response: %s", exc)
        raise RuntimeError("Unable to parse Qwen3 Coder Flash output") from exc

    if not code:
        raise RuntimeError("Qwen3 Coder Flash returned empty content.")

    cleaned = _strip_code_fences(code)
    return cleaned + ("\n" if cleaned and not cleaned.endswith("\n") else "")


SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9_.-]+")  # safe filename pattern

# Save generated code as a file to `data/generated_code/`
def save_generated_code(
    code: str,
    directory: Path,
    stem: str = "lesson",
    filename: Optional[str] = None,
) -> Path:
    # Persist the generated script with a safe filename.
    code = _strip_code_fences(code)
    directory.mkdir(parents=True, exist_ok=True)
    if filename:
        safe_name = SAFE_FILENAME_RE.sub("_", filename).strip("_") or "lesson_example.py"
        if not safe_name.endswith(".py"):
            safe_name += ".py"
        file_name = safe_name
    else:
        safe_stem = SAFE_FILENAME_RE.sub("_", stem).strip("_") or "lesson"
        file_name = f"{safe_stem}_example.py"
    filepath = directory / file_name
    filepath.write_text(code, encoding="utf-8")
    logger.info("Saved generated code to %s", filepath)
    return filepath


# Workflow step 1: Run code directly (legacy, for backward compatibility)
def run_code(filepath: Path, timeout: int = 120) -> tuple[bool, str, str]:
    """Execute the generated Python file and capture stdout/stderr."""
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        completed = subprocess.run(
            [sys.executable, str(filepath)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False, "", f"Execution timed out after {timeout}s."

    success = completed.returncode == 0
    return success, completed.stdout, completed.stderr


# Workflow step 2: Validate code using MCP python_checker tool
def validate_code_with_mcp(filepath: Path, timeout: int = 30) -> dict[str, object]:
    """
    Validate code using the python_checker MCP tool.
    This checks both syntax errors and runtime errors.
    
    Args:
        filepath: Path to the generated Python file
        timeout: Timeout for code execution
        
    Returns:
        Dictionary with validation results:
        {
            "success": bool,
            "path": Path,
            "stdout": str,
            "stderr": str,
            "compile_success": bool,
            "run_success": bool,
        }
    """
    if not filepath.exists():
        return {
            "success": False,
            "path": filepath,
            "stdout": "",
            "stderr": f"File not found: {filepath}",
            "compile_success": False,
            "run_success": False,
        }
    
    try:
        with MCPClientSync() as client:
            result = client.call_tool("python_check", filepath=str(filepath), timeout=timeout)
            
            # Extract relevant info from MCP result
            compile_result = result.get("compile", {})
            run_result = result.get("run", {})
            
            return {
                "success": result.get("success", False),
                "path": filepath,
                "stdout": run_result.get("stdout", "") if run_result else "",
                "stderr": run_result.get("stderr", "") if run_result else compile_result.get("stderr", ""),
                "compile_success": compile_result.get("success", False),
                "run_success": run_result.get("success", False) if run_result else False,
                "compile_details": compile_result,
                "run_details": run_result,
            }
    except Exception as exc:
        logger.error("MCP validation failed: %s", exc)
        return {
            "success": False,
            "path": filepath,
            "stdout": "",
            "stderr": f"Validation error: {str(exc)}",
            "compile_success": False,
            "run_success": False,
        }


# Workflow step 3: Full pipeline - generate → save → validate
def validate_code(
    code: str,
    output_dir: Path,
    stem: str = "lesson",
    filename: Optional[str] = None,
) -> dict[str, object]:
    """
    Full code generation workflow:
    1. Save the generated code to a file
    2. Use MCP python_checker to validate it
    
    Args:
        code: Generated Python code string
        output_dir: Directory to save the generated file
        stem: Base name for the generated file
        
    Returns:
        Validation result dictionary
    """
    # Step 1: Save code to file
    output_path = save_generated_code(code, output_dir, stem=stem, filename=filename)
    logger.info(f"Code saved to: {output_path}")
    
    # Step 2: Validate using MCP tool
    validation_result = validate_code_with_mcp(output_path, timeout=120)
    logger.info(f"Code validation result: success={validation_result['success']}")
    
    return validation_result


# make the functions exportable
__all__ = [
    "generate_code",
    "validate_code",
    "save_generated_code",
    "run_code",
]
