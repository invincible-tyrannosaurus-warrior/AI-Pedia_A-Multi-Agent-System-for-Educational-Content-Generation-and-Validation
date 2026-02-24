"""
Coder Agent Pipeline

This module implements the specific pipeline for the Coder Agent.
It receives instructions (guidance) and generates/validates Python code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from moe_layer.coder_agent.coder import generate_code, validate_code
from moe_layer.coder_agent.storage.local import StoredAsset

logger = logging.getLogger(__name__)


def run_coder_pipeline(
    instruction: str,
    output_dir: Path,
    assets: Optional[List[StoredAsset]] = None,
    client: Optional[OpenAI] = None,
    output_filename: Optional[str] = None,
    output_subdir: Optional[str] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Execute the Coder Agent pipeline.

    Args:
        instruction: The detailed coding guidance/instruction.
        output_dir: Directory to save generated artifacts.
        assets: Optional list of assets (not primarily used here as instruction contains context).
        client: Optional OpenAI client for the coder model.
        **kwargs: Additional agent-specific arguments.

    Returns:
        Dictionary containing success status, code, and validation results.
    """
    # 1. Prepare Directories
    # Handle 'output_subdir' to isolate runs (e.g. data/generated_code/run_<id>)
    # NOTE: output_subdir comes in as a named parameter from task_manager inputs.
    # We also check kwargs as a fallback for backward compatibility.
    if not output_subdir:
        output_subdir = kwargs.get("output_subdir")
    run_dir = output_dir / output_subdir if output_subdir else output_dir
    
    scripts_dir = run_dir / "scripts"
    assets_dir = run_dir / "assets"
    
    scripts_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Inject Strict Path Instructions
    # We provide absolute paths (forward slashes) to be unambiguous
    assets_path_str = str(assets_dir.resolve()).replace("\\", "/")
    
    strict_instruction = (
        f"{instruction}\n\n"
        f"*** STRICT FILE SYSTEM RULES ***\n"
        f"1. You MUST save all generated artifacts (plots, images, csvs, etc.) to: `{assets_path_str}`\n"
        f"2. Use this path exactly. Example: `plt.savefig('{assets_path_str}/my_plot.png')`\n"
        f"3. Do NOT create any new directories.\n"
        f"4. Do NOT save files to the current directory.\n"
        f"5. **CRITICAL**: For plots, you MUST use the non-interactive backend:\n"
        f"   ```python\n"
        f"   import matplotlib\n"
        f"   matplotlib.use('Agg')\n"
        f"   import matplotlib.pyplot as plt\n"
        f"   ```\n"
        f"6. **NEVER** use `plt.show()`. It blocks execution. ALWAYS use `plt.savefig()`.\n"
    )

    logger.info("Coder Agent started. Outputting scripts to: %s", scripts_dir)

    # Generate code based on the instruction
    code = generate_code(strict_instruction)
    logger.info("Generated code (%d chars)", len(code))

    # Validate the generated code
    # We save the script to the 'scripts' subdirectory
    validation = validate_code(code, output_dir=scripts_dir, filename=output_filename)

    return {
        "success": validation.get("success", False),
        "output": {
            "code": code,
            "validation": validation,
            "paths": {
                "scripts": str(scripts_dir),
                "assets": str(assets_dir)
            }
        },
        "artifacts": [str(validation.get("path"))] if validation.get("path") else [],
        "metadata": {
            "model": "qwen3-coder-flash"
        }
    }

__all__ = ["run_coder_pipeline"]
