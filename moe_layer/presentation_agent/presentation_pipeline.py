"""
Presentation Agent Pipeline (Rebuilt)

Architecture:
1. Instructional Designer (LLM): Generates a JSON Storyboard
2. PPT Builder (Deterministic): Builds PPTX from the Storyboard

NO LLM code generation - the builder is pure Python.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

from openai import OpenAI
from config import TEMPLATE_PATH
from moe_layer.coder_agent.storage.local import StoredAsset
from moe_layer.presentation_agent.ppt_builder import build_presentation

logger = logging.getLogger(__name__)

# =============================================================================
# INSTRUCTIONAL DESIGNER PROMPT (Simplified & Clear)
# =============================================================================

INSTRUCTIONAL_DESIGNER_PROMPT = """
You are an expert Instructional Designer. Create a storyboard JSON for an educational presentation.

## SLIDE LAYOUTS

1. **Title Slide** (`layout_idx: 0`)
   - Use for the FIRST slide only
   - Fields: `title`, `subtitle`

2. **Content Slide** (`layout_idx: 5`) 
   - Single column of text
   - Fields: `title`, `content`

3. **Two-Column Slide** (`layout_idx: 6`)
   - Text on left, visual on right
   - Fields: `title`, `content`, `visual_assets`

## VISUAL ASSETS (for layout_idx: 6)

Include ONE of these in `visual_assets` array:

1. **Code Snippet**:
   ```json
   {"type": "code_snippet", "code": "def example():\\n    return 'Hello'", "language": "python"}
   ```

2. **Chart**:
   ```json
   {"type": "chart_data", "title": "Comparison", "chart_type": "bar", "data": {"labels": ["A", "B", "C"], "values": [10, 20, 15]}}
   ```
   (chart_type can be: bar, line, pie)

3. **Formula**:
   ```json
   {"type": "formula_latex", "content": "E = mc^2"}
   ```

## REQUIREMENTS

1. First slide MUST be a title slide (layout_idx: 0)
2. Include 5-8 slides total
3. Use layout_idx: 6 for slides that need visuals
4. Write clear, educational content
5. Include at least 2 slides with visual_assets

## OUTPUT FORMAT

Return ONLY valid JSON (no markdown):

{
  "slides": [
    {
      "layout_idx": 0,
      "title": "Main Title Here",
      "subtitle": "Subtitle or description"
    },
    {
      "layout_idx": 6,
      "title": "Key Concept",
      "content": "Explanation of the concept in 2-3 sentences.",
      "visual_assets": [
        {"type": "formula_latex", "content": "y = mx + b"}
      ]
    },
    {
      "layout_idx": 5,
      "title": "Summary",
      "content": "Key takeaways from this presentation."
    }
  ]
}
"""


def run_presentation_pipeline(
    instruction: str,
    output_dir: Path,
    assets: Optional[List[StoredAsset]] = None,
    client: Optional[OpenAI] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Execute the Presentation Agent pipeline.
    
    1. Generate storyboard JSON via LLM
    2. Build PPTX using deterministic builder
    """
    logger.info("Presentation Agent started: %s", instruction[:50])
    
    if not client:
        raise ValueError("OpenAI client not provided to pipeline")

    # Get run directory
    output_subdir = kwargs.get("output_subdir")
    run_dir = output_dir / output_subdir if output_subdir else output_dir
    
    # Ensure directories exist
    assets_dir = run_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # STEP 1: Generate Storyboard via LLM
    # ==========================================================================
    try:
        storyboard = _generate_storyboard(instruction, assets or [], client)
        
        # Save storyboard for debugging/reference
        storyboard_path = run_dir / "storyboard.json"
        with open(storyboard_path, "w", encoding="utf-8") as f:
            json.dump(storyboard, f, indent=2, ensure_ascii=False)
        logger.info(f"Storyboard saved to: {storyboard_path}")
        
    except Exception as e:
        logger.exception("Storyboard generation failed")
        return {"success": False, "error": f"Storyboard generation failed: {e}"}

    # ==========================================================================
    # STEP 2: Build PPTX using deterministic builder
    # ==========================================================================
    timestamp = int(time.time())
    output_filename = f"lesson_{timestamp}.pptx"
    output_path = run_dir / output_filename
    
    try:
        build_presentation(
            storyboard=storyboard,
            template_path=TEMPLATE_PATH,
            output_path=output_path,
            assets_dir=assets_dir
        )
    except Exception as e:
        logger.exception("PPTX build failed")
        return {"success": False, "error": f"PPTX build failed: {e}"}

    # ==========================================================================
    # STEP 3: Verify and return
    # ==========================================================================
    if not output_path.exists():
        return {"success": False, "error": "PPTX file was not created."}

    # Aggregate text for downstream agents
    all_text = []
    slides = storyboard.get("slides", [])
    for idx, slide in enumerate(slides):
        all_text.append(f"Slide {idx+1}: {slide.get('title', '')}")
        if slide.get('content'):
            all_text.append(slide.get('content'))

    # Calculate metrics
    metrics = {
        "slide_count": len(slides),
        "code_snippets_count": 0,
        "charts_count": 0,
        "formulas_count": 0
    }
    
    for slide in slides:
        for asset in slide.get("visual_assets", []):
            atype = asset.get("type", "")
            if atype == "code_snippet":
                metrics["code_snippets_count"] += 1
            elif atype == "chart_data":
                metrics["charts_count"] += 1
            elif atype == "formula_latex":
                metrics["formulas_count"] += 1

    return {
        "success": True,
        "output": {
            "text": "\n\n".join(all_text),
            "format": "pptx",
            "storyboard": storyboard,
            "metrics": metrics
        },
        "artifacts": [str(output_path)]
    }


def _generate_storyboard(
    instruction: str, 
    assets: List[StoredAsset], 
    client: OpenAI
) -> Dict[str, Any]:
    """Generate storyboard JSON via LLM."""
    
    # Build context from assets
    context_text = ""
    if assets:
        extracted = []
        for asset in assets:
            if asset.extracted_text:
                extracted.append(f"--- {asset.original_filename} ---\n{asset.extracted_text[:3000]}")
        if extracted:
            context_text = "\n\nReference Materials:\n" + "\n".join(extracted)
    
    # Try RAG enhancement
    try:
        from ai_pedia_mcp_server.mcp_tools.rag_search import rag_query
        rag_context = rag_query(instruction, n_results=3)
        if "No relevant information found" not in rag_context:
            context_text += f"\n\n[Knowledge Base]:\n{rag_context}\n"
    except Exception as e:
        logger.warning(f"RAG enhancement failed: {e}")

    # Build user message
    user_message = f"Create a presentation about: {instruction}"
    if context_text:
        user_message += f"\n{context_text}"

    # Call LLM
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": INSTRUCTIONAL_DESIGNER_PROMPT},
            {"role": "user", "content": user_message}
        ],
        response_format={"type": "json_object"},
        temperature=0.7
    )
    
    content = response.choices[0].message.content
    
    # Parse JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse storyboard JSON: {e}")
        logger.error(f"Raw content: {content}")
        raise

