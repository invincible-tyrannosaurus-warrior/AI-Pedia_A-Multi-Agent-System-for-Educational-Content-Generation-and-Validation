
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from openai import OpenAI

from moe_layer.video_agent.ppt_converter import PPTConverter
from moe_layer.video_agent.script_writer import ScriptWriter
from moe_layer.video_agent.video_composer import VideoComposer
from moe_layer.coder_agent.storage.local import StoredAsset

logger = logging.getLogger(__name__)

async def run_video_pipeline(
    instruction: str,
    output_dir: Path,
    assets: List[StoredAsset] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Video Agent Entry Point.
    
    Workflow:
    1. Identify PPTX asset.
    2. Convert PPTX -> Images.
    3. Generate Scripts (Vision LLM).
    4. Compose Video (EdgeTTS + MoviePy).
    """
    logger.info("Starting Video Agent Pipeline...")
    
    # 0. Setup
    client = kwargs.get("client") or OpenAI(
        base_url=kwargs.get("base_url"),
        api_key=kwargs.get("api_key")
    )
    
    # Handle 'output_subdir' to ensure we work in the isolated run directory
    output_subdir_kw = kwargs.get("output_subdir")
    run_dir = output_dir / output_subdir_kw if output_subdir_kw else output_dir
    
    ppt_path = _resolve_ppt_path(assets, kwargs.get("dependency_results"))
    if not ppt_path:
        return {
            "success": False,
            "error": "No .pptx file found in assets or dependency results."
        }
        
    try:
        # 1. Convert PPT -> Images
        converter = PPTConverter()
        
        # STRICT PATH: run_dir/assets/slides_images
        assets_dir = run_dir / "assets"
        images_dir = assets_dir / "slides_images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        image_paths = converter.convert_to_images(ppt_path, images_dir)
        
        if not image_paths:
            return {"success": False, "error": "Failed to extract images from PPTX."}

        # 2. Generate Scripts
        writer = ScriptWriter(client)
        # Use instruction as the 'Topic' or 'Lecture Goal'
        scripts = writer.generate_scripts(image_paths, topic=instruction)
        
        # 3. Compose Video
        composer = VideoComposer()
        
        # STRICT PATH: run_dir/output/
        video_out_dir = run_dir / "output"
        video_out_dir.mkdir(parents=True, exist_ok=True)
        
        video_filename = f"{ppt_path.stem}_lecture.mp4"
        video_path = video_out_dir / video_filename
        
        await composer.compose_video(image_paths, scripts, video_path)
        
        return {
            "success": True,
            "artifacts": [str(video_path)],
            "message": f"Successfully generated video: {video_filename}",
            "scripts": scripts # Return scripts for debugging/review
        }

    except Exception as e:
        logger.exception("Video Pipeline Failed")
        return {
            "success": False, 
            "error": str(e)
        }

def _resolve_ppt_path(assets: List[StoredAsset] = None, dependency_results: Dict[str, Any] = None) -> Optional[Path]:
    """Finds a PPTX file in assets or dependency outputs."""
    # 1. Check generated artifacts from dependencies
    if dependency_results:
        for task_id, result in dependency_results.items():
            artifacts = result.get("artifacts", [])
            for artifact in artifacts:
                if str(artifact).lower().endswith(".pptx"):
                    return Path(artifact)
    
    # 2. Check uploaded assets
    if assets:
        for asset in assets:
            if asset.original_filename.lower().endswith(".pptx"):
                return asset.path
                
    return None
