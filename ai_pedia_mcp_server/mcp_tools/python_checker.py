# this file contains the tool that checks python files for errors (from coder_agent)
# only contains the functions and logic, the tool will be activated in main.py
# main purpose:
#   is to check python files for syntax and runtime errors
#   logs any errors to a jsonl file for later review

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

ERROR_LOG_NAME = "python_check_errors.jsonl"


def _resolve_path(filepath: str) -> Path:
    path = Path(filepath).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _run_python(args: list[str], timeout: int, cwd: Optional[Path] = None) -> dict[str, object]:
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        completed = subprocess.run(
            [sys.executable, *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
            cwd=str(cwd) if cwd else None,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "success": False,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or f"Execution timed out after {timeout}s.",
            "timed_out": True,
        }

    return {
        "success": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "timed_out": False,
    }


def _record_error(
    path: Path, compile_result: dict[str, object], run_result: Optional[dict[str, object]]
) -> Optional[Path]:
    log_dir = Path(__file__).resolve().parents[1] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / ERROR_LOG_NAME

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "path": str(path),
        "compile": compile_result,
        "run": run_result,
    }

    try:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to write python_check error log: %s", exc)
        return None

    return log_path


def python_check(filepath: str, timeout: int = 120) -> dict[str, object]:
    """Check a Python file for syntax/runtime errors and log failures."""
    path = _resolve_path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Expected a file, got directory: {path}")
    if path.suffix != ".py":
        raise ValueError("python_check expects a .py file path.")

    compile_result = _run_python(["-m", "py_compile", str(path)], timeout)
    run_result: Optional[dict[str, object]] = None
    success = bool(compile_result["success"])

    if success:
        # Run execution with CWD set to the script's directory
        run_result = _run_python([str(path)], timeout, cwd=path.parent)
        success = bool(run_result["success"])

    log_path = None
    if not success:
        log_path = _record_error(path, compile_result, run_result)

    return {
        "success": success,
        "path": str(path),
        "compile": compile_result,
        "run": run_result,
        "error_log": str(log_path) if log_path else None,
    }


# Register the python_check tool with the given MCP server.
def register(mcp: FastMCP) -> None:
    mcp.tool()(python_check)


__all__ = ["python_check", "register"]
