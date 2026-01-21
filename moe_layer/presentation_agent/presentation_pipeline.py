"""
Presentation Agent Pipeline

This module implements the Presentation Agent, responsible for creating PowerPoint slides.
Architecture:
1. Instructional Designer (Planner): Generates a JSON "Storyboard" of slides.
2. PPT Engineer (Builder): Generates a Python script using python-pptx.
3. Execution: Runs the script to produce the .pptx file.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

from openai import OpenAI
from config import TEMPLATE_PATH
from moe_layer.coder_agent.storage.local import StoredAsset
from moe_layer.coder_agent.coder import generate_code

logger = logging.getLogger(__name__)

PPT_BUILDER_PROMPT = """
You are a Python Expert specializing in `python-pptx`.
Your goal is to write a COMPLETE script to generate a `.pptx` from a JSON Storyboard file.

*** INPUT DATA ***
1. Storyboard: `{{STORYBOARD_PATH}}`
   ```python
   import json
   with open(r"{{STORYBOARD_PATH}}", "r", encoding="utf-8") as f:
       storyboard = json.load(f)
   slides_data = storyboard.get("slides", [])
   ```
2. Assets Directory: `{{ASSETS_PATH}}`
   - You MUST save all generated images (code snippets, charts) here.
   - Example: `img_path = r"{{ASSETS_PATH}}/code_snippet_1.png"`
   - **DO NOT** create a "temp" folder. **DO NOT** use `os.makedirs`. Use existing `{{ASSETS_PATH}}`.

*** TEMPLATE SCHEMA (CRITICAL) ***
You are using a custom template `{{TEMPLATE_PATH}}` with specific Layout IDs and Placeholders.
You MUST use these exact **Placeholder IDX** keys (python-pptx accesses placeholders by IDX, e.g. `slide.placeholders[11]`).

1.  **Title Slide** (Layout Index: 0)
    - Title: placeholders[0]
    - Subtitle: placeholders[1]

2.  **Content Slide (1 Column)** (Layout Index: 5)
    - Title: placeholders[0]
    - Body Text: placeholders[11]

3.  **Content Slide (2 Columns)** (Layout Index: 6)
    - Title: placeholders[0]
    - Left Column: placeholders[12]
    - Right Column: placeholders[13]

*** SCRIPT REQUIREMENTS ***
1.  **Imports**: 
    - `from pptx import Presentation`
    - `from pptx.util import Inches, Pt`
    - `from pptx.dml.color import RGBColor` (DO NOT import from `pptx.shared`)
    - `from PIL import Image, ImageDraw`
    - `import json`
    - `import os`
2.  **Load & Clean Template**: 
    - `prs = Presentation(r"{{TEMPLATE_PATH}}")`
    - **CRITICAL**: The template has existing slides. You MUST remove them to start fresh.
    - Use this exact code to safely clear slides:
      ```python
      # Robustly clear existing slides
      xml_slides = prs.slides._sldIdLst
      slides = list(xml_slides)
      # Iterate in reverse to avoid index shifting issues
      for i in range(len(slides) - 1, -1, -1):
          rId = xml_slides[i].rId
          # Drop relationship to the slide part
          prs.part.drop_rel(rId)
          # Remove the slide element
          del xml_slides[i]
      ```
3.  **Iterate & Build**: Loop through `slides_data` and create slides.
    - Check `layout_idx` from data.
    - **CRITICAL**: Do NOT use `if idx in slide.placeholders`. This check FAILS because placeholders is a sequence, not a dict.
    - Instead, simple TRY/EXCEPT or direct access:
      ```python
      try:
          slide.placeholders[0].text = title
      except KeyError:
          pass
      ```
4.  **Content Mapping (Strict)**:
    - **General Rule**: ALWAYS set text color to **Black** (`RGBColor(0, 0, 0)`) for **EVERY PARAGRAPH**.
      - *Incorrect*: `shape.text_frame.paragraphs[0].font.color.rgb = ...` (Only fixes first line)
      - *Correct*:
        ```python
        shape.text_frame.text = content
        for p in shape.text_frame.paragraphs:
            p.font.color.rgb = RGBColor(0,0,0)
        ```
    - **Layout 5 (1-Column)**: 
      - Title -> `placeholders[0]`
      - `slide_data['content']` -> `placeholders[11]` (Body Text). Set color for ALL paragraphs.
    - **Layout 6 (2-Column)**:
      - Title -> `placeholders[0]`
      - **Left Column**: Map `slide_data['content']` -> `placeholders[12]` (Body Text). Set color for ALL paragraphs.
      - **Right Column**: Map `slide_data['visual_assets']` -> `placeholders[13]` (Visual)
        - **Asset Handling logic**:
        - If `visual_assets` is a list, take the first item.
        - **Case A: String Path** (e.g. `assets/plot.png`): 
            - Verify file exists.
            - **CRITICAL**: Use Robust Image Insertion (Placeholder object may not have insert_picture):
              ```python
              ph = slide.placeholders[13]
              slide.shapes.add_picture(asset_path, ph.left, ph.top, ph.width, ph.height)
              ```
        - **Case B: Dictionary (Code Snippet)** (e.g. `{'type': 'code_snippet', 'code': '...'}`):
            - You MUST locally generate a syntax-highlighted (or matching) image using `PIL`.
            - Create a new white image (`Image.new('RGB', (800, 600), 'white')`).
            - Draw the `code` text onto it (black text).
            - Save to `{{ASSETS_PATH}}/generated_code_{slide_idx}.png`.
            - Then insert using the Robust method above.
5.  **Error Handling**: Wrap `build_presentation` in try-except.


Return ONLY valid Python code.
"""

INSTRUCTIONAL_DESIGNER_PROMPT = """
You are an expert Instructional Designer.
Create a "Storyboard" JSON for a presentation.

*** CREATIVE FREEDOM ***
- Do NOT just write bullet points. Use Questions, Quotes, and Code Comparisons.
- Choose layouts that fit the content density.

*** AVAILABLE LAYOUTS ***
- `layout_idx: 0`: **Title Slide**. Use for the very first slide.
- `layout_idx: 5`: **1-Column Text**. Use for definitions, main concepts, or list-heavy slides.
- `layout_idx: 6`: **2-Col (Text + Visual)**. BEST for "Code vs Explanation", "Image + Text", or "Comparison".
  - Put main text in `content`.
  - Put visual description in `visual_assets`.

*** OUTPUT JSON ***
{
  "slides": [
    {
      "layout_idx": 6,
      "title": "Why Python?",
      "content": "- Easy to read\\n- Huge ecosystem",
      "visual_assets": [
        { "type": "code_snippet", "code": "print('Hello')", "language": "python", "caption": "Simplicity" }
      ]
    }
  ]
}
Constraint: Max 8 slides.
"""

def _generate_storyboard(instruction: str, assets: List[StoredAsset], client: OpenAI) -> Dict[str, Any]:
    """Step 1: Instructional Designer"""
    
    # 1. Context Injection from Assets
    context_text = ""
    if assets:
        extracted = []
        for asset in assets:
            if asset.extracted_text:
                extracted.append(f"--- Document: {asset.original_filename} ---\n{asset.extracted_text[:5000]}...") # Truncate for token limit
        if extracted:
            context_text = "\n\nReference Materials:\n" + "\n".join(extracted)
    
    # RAG Context Enhancement
    try:
        from ai_pedia_mcp_server.mcp_tools.rag_search import rag_query
        rag_context = rag_query(instruction, n_results=3)
        if "No relevant information found" not in rag_context:
             context_text += f"\n\n[Knowledge Base Context]:\n{rag_context}\n"
    except Exception as e:
        logger.warning(f"RAG Enhancement failed: {e}")

    # 4. Generate Storyboard
    gpt_input = f"""
    User Request: "{instruction}"
    
    {context_text}
    """
    messages = [
        {"role": "system", "content": INSTRUCTIONAL_DESIGNER_PROMPT},
        {"role": "user", "content": gpt_input}
    ]
    response = client.chat.completions.create(
        model="gpt-4o",  # or prompt's preference
        messages=messages,
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except Exception as e:
        logger.error("Failed to parse storyboard JSON: %s", e)
        raise

def _generate_builder_script(storyboard: Dict[str, Any], output_path: str, output_dir: Path) -> str:
    """Step 2: PPT Engineer (Delegated to Coder Agent)"""
    
    # 1. Save Storyboard to JSON to avoid token limits in prompt
    storyboard_path = output_dir / "storyboard.json"
    with open(storyboard_path, "w", encoding="utf-8") as f:
        json.dump(storyboard, f, indent=2)
    
    # Resolve paths to absolute strings with forward slashes
    template_path_str = str(TEMPLATE_PATH).replace("\\", "/")
    storyboard_path_str = str(storyboard_path).replace("\\", "/")
    # Use the 'assets' subdirectory within the output_dir
    assets_path_str = str(output_dir / "assets").replace("\\", "/")
    
    # Inject paths into prompt
    prompt = PPT_BUILDER_PROMPT.replace("{{TEMPLATE_PATH}}", template_path_str)
    prompt = prompt.replace("{{STORYBOARD_PATH}}", storyboard_path_str)
    prompt = prompt.replace("{{ASSETS_PATH}}", assets_path_str)
    
    # Delegate to Coder Agent
    logger.info("Delegating script generation to Coder Agent (Data-Driven)...")
    
    final_prompt = f"{prompt}\n\nTarget Output Path: {output_path}"
    code = generate_code(final_prompt)
    
    return code

def _fix_builder_script(previous_code: str, error_msg: str, output_dir: Path) -> str:
    """Step 3: Auto-Fixer"""
    repaired_prompt = f"""
The previous Python script failed with the following error:
{error_msg}

Please FIX the code. 
- Do not remove imports.
- Ensure 'master_template.pptx' is loaded correctly.
- Return the FULL corrected script.
"""
    template_path_str = str(TEMPLATE_PATH).replace("\\", "/")
    assets_path_str = str(output_dir / "assets").replace("\\", "/")
    
    prompt = PPT_BUILDER_PROMPT.replace("{{TEMPLATE_PATH}}", template_path_str)
    prompt = prompt.replace("{{ASSETS_PATH}}", assets_path_str)
    
    full_prompt = f"{prompt}\n\nPREVIOUS CODE:\n{previous_code}\n\nERROR:\n{repaired_prompt}"
    
    logger.info("Delegating script repair to Coder Agent...")
    code = generate_code(full_prompt)
    
    return code

def run_presentation_pipeline(
    instruction: str,
    output_dir: Path,
    assets: Optional[List[StoredAsset]] = None,
    client: Optional[OpenAI] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Execute the Presentation Agent pipeline.
    """
    logger.info("Presentation Agent started: %s", instruction[:50])
    
    # 0. Ensure Client
    if not client:
        # Fallback or error, usually passed by task manager
        raise ValueError("OpenAI client not provided to pipeline")

    # Handle 'output_subdir' to ensure we work in the isolated run directory
    output_subdir = kwargs.get("output_subdir")
    run_dir = output_dir / output_subdir if output_subdir else output_dir

    # 1. Plan
    try:
        storyboard = _generate_storyboard(instruction, assets or [], client)
    except Exception as e:
        return {"success": False, "error": f"Planning failed: {e}"}

    # 2. Build Script
    timestamp = int(time.time())
    
    # Define dedicated slides directory: data/slides
    # root is parent.parent.parent relative to this file
    project_root = Path(__file__).resolve().parent.parent.parent
    slides_dir = project_root / "data" / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)
    
    output_pptx_filename = f"lesson_{timestamp}.pptx"
    output_pptx_path = slides_dir / output_pptx_filename
    
    # Ensure assets dir exists for the script to use (in the isolated run_dir)
    (run_dir / "assets").mkdir(parents=True, exist_ok=True)

    try:
        # Pass run_dir instead of output_dir to ensure proper asset injection
        script_code = _generate_builder_script(storyboard, str(output_pptx_path).replace("\\", "/"), run_dir)
    except Exception as e:
        return {"success": False, "error": f"Script generation failed: {e}"}

    # 3. Execute with Self-Healing
    max_retries = 3
    last_error = ""
    
    # Define script path in standardized 'scripts' directory
    scripts_dir = run_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_path = scripts_dir / f"builder_{timestamp}.py"
    
    for attempt in range(max_retries):
        # Save script (overwriting previous attempt)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_code)

        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            result = subprocess.run(
                [sys.executable, str(script_path)],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(run_dir), # Execute in run_dir to keep assets local
                env=env
            )
            logger.info("Builder output: %s", result.stdout)
            break # Success!
        except subprocess.CalledProcessError as e:
            last_error = e.stderr
            logger.warning("Builder failed (Attempt %d/%d): %s", attempt+1, max_retries, last_error)
            
            if attempt < max_retries - 1:
                # Try to fix it
                try:
                    logger.info("Attempting to auto-fix builder script...")
                    script_code = _fix_builder_script(script_code, last_error, output_dir)
                except Exception as fix_err:
                    logger.error("Auto-fix failed: %s", fix_err)
                    break # Stop if fixer fails
            else:
                # Out of retries
                return {
                    "success": False, 
                    "error": f"Builder execution failed after {max_retries} attempts. Last error: {last_error}",
                    "artifacts": [str(script_path)]
                }

    # 4. Verify Output
    if not output_pptx_path.exists():
        return {"success": False, "error": "PPTX file was not created by the script."}

    # Aggregate text for Judger Semantic Checks
    all_text = []
    slides = storyboard.get("slides", [])
    for idx, slide in enumerate(slides):
        all_text.append(f"Slide {idx+1}: {slide.get('title', '')}")
        all_text.append(slide.get('content', ''))
        if slide.get('speaker_notes'):
            all_text.append(f"Notes: {slide.get('speaker_notes')}")
    
    full_text_content = "\n\n".join(all_text)
    
    # Calculate Metrics for Judger
    metrics = {
        "slide_count": len(slides),
        "code_snippets_count": 0,
        "charts_count": 0,
        "formulas_count": 0
    }
    
    for slide in slides:
        assets = slide.get("visual_assets", [])
        for asset in assets:
            atype = asset.get("type")
            if atype == "code_snippet":
                metrics["code_snippets_count"] += 1
            elif atype == "chart_data":
                metrics["charts_count"] += 1
            elif atype == "formula_latex":
                metrics["formulas_count"] += 1

    return {
        "success": True,
        "output": {
            "text": full_text_content,
            "format": "pptx",
            "storyboard": storyboard,
            "metrics": metrics
        },
        "artifacts": [str(output_pptx_path)]
    }
