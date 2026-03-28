
from __future__ import annotations
	
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from openai import OpenAI
	
from moe_layer.video_agent.ppt_converter import PPTConverter
from moe_layer.video_agent.script_writer import ScriptWriter
from moe_layer.video_agent.video_composer import VideoComposer
from moe_layer.coder_agent.storage.local import StoredAsset
	
logger = logging.getLogger(__name__)

_VIDEO_STAGE_SPECS = [
    ("prepare", "Prepare", 5),
    ("convert_slides", "Convert Slides", 15),
    ("generate_scripts", "Generate Scripts", 30),
    ("build_clips", "Build Clips", 30),
    ("encode_video", "Encode Video", 20),
]
_VIDEO_STAGE_LOOKUP = {
    name: {
        "index": index + 1,
        "label": label,
        "weight": weight,
    }
    for index, (name, label, weight) in enumerate(_VIDEO_STAGE_SPECS)
}


class VideoProgressReporter:
    def __init__(
        self,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        task_id: Optional[str] = None,
    ) -> None:
        self.callback = callback
        self.task_id = task_id
        self.current_stage = "prepare"

    def bind(self, stage: str) -> Callable[..., None]:
        def _bound(
            *,
            stage_progress: float,
            current: Optional[int] = None,
            total: Optional[int] = None,
            message: Optional[str] = None,
            status: str = "running",
            indeterminate: bool = False,
        ) -> None:
            self.emit(
                stage=stage,
                stage_progress=stage_progress,
                current=current,
                total=total,
                message=message,
                status=status,
                indeterminate=indeterminate,
            )

        return _bound

    def start(self, stage: str, message: Optional[str] = None) -> None:
        self.emit(stage=stage, stage_progress=0.0, message=message, status="running")

    def complete(
        self,
        stage: str,
        message: Optional[str] = None,
        current: Optional[int] = None,
        total: Optional[int] = None,
    ) -> None:
        self.emit(
            stage=stage,
            stage_progress=1.0,
            current=current,
            total=total,
            message=message,
            status="running",
        )

    def fail(self, message: str, stage: Optional[str] = None) -> None:
        self.emit(
            stage=stage or self.current_stage,
            stage_progress=0.0,
            message=message,
            status="error",
        )

    def emit(
        self,
        *,
        stage: str,
        stage_progress: float,
        current: Optional[int] = None,
        total: Optional[int] = None,
        message: Optional[str] = None,
        status: str = "running",
        indeterminate: bool = False,
    ) -> None:
        if self.callback is None:
            return

        stage_meta = _VIDEO_STAGE_LOOKUP[stage]
        bounded_progress = max(0.0, min(1.0, stage_progress))
        completed_weight = sum(
            item[2]
            for item in _VIDEO_STAGE_SPECS
            if _VIDEO_STAGE_LOOKUP[item[0]]["index"] < stage_meta["index"]
        )
        overall_progress = min(
            100.0,
            completed_weight + stage_meta["weight"] * bounded_progress,
        )

        self.current_stage = stage
        payload = {
            "agent": "video",
            "stage": stage,
            "stage_label": stage_meta["label"],
            "stage_index": stage_meta["index"],
            "stage_count": len(_VIDEO_STAGE_SPECS),
            "stage_weight": stage_meta["weight"],
            "stage_progress": round(bounded_progress, 4),
            "overall_progress": round(overall_progress, 2),
            "status": status,
            "indeterminate": indeterminate,
        }
        if self.task_id:
            payload["id"] = self.task_id
        if current is not None:
            payload["current"] = current
        if total is not None:
            payload["total"] = total
        if message:
            payload["message"] = message

        self.callback(payload)
	
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

    reporter = VideoProgressReporter(
        callback=kwargs.get("progress_callback"),
        task_id=kwargs.get("task_id"),
    )
    reporter.start("prepare", "Preparing video pipeline.")

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
        reporter.fail("No .pptx file found in assets or dependency results.", stage="prepare")
        return {
            "success": False,
            "error": "No .pptx file found in assets or dependency results."
        }
        
    try:
        # Fail fast on runtime dependencies before spending time on narration.
        converter = PPTConverter()
        composer = VideoComposer()
        reporter.complete("prepare", f"Prepared input deck: {ppt_path.name}")

        # 1. Convert PPT -> Images
        # STRICT PATH: run_dir/assets/slides_images
        assets_dir = run_dir / "assets"
        images_dir = assets_dir / "slides_images"
        images_dir.mkdir(parents=True, exist_ok=True)

        reporter.start("convert_slides", "Converting slides to images.")
        image_paths = converter.convert_to_images(
            ppt_path,
            images_dir,
            progress_callback=reporter.bind("convert_slides"),
        )
        
        if not image_paths:
            reporter.fail("Failed to extract images from PPTX.", stage="convert_slides")
            return {"success": False, "error": "Failed to extract images from PPTX."}
        reporter.complete(
            "convert_slides",
            f"Converted {len(image_paths)} slides to images.",
            current=len(image_paths),
            total=len(image_paths),
        )

        # 2. Generate Scripts
        reporter.start("generate_scripts", "Generating narration scripts.")
        writer = ScriptWriter(client)
        # Use instruction as the 'Topic' or 'Lecture Goal'
        scripts = writer.generate_scripts(
            image_paths,
            topic=instruction,
            progress_callback=reporter.bind("generate_scripts"),
        )
        reporter.complete(
            "generate_scripts",
            f"Generated narration for {len(scripts)} slides.",
            current=len(scripts),
            total=len(image_paths),
        )
        
        # STRICT PATH: run_dir/output/
        video_out_dir = run_dir / "output"
        video_out_dir.mkdir(parents=True, exist_ok=True)
        
        video_filename = f"{ppt_path.stem}_lecture.mp4"
        video_path = video_out_dir / video_filename
        
        await composer.compose_video(
            image_paths,
            scripts,
            video_path,
            build_progress_callback=reporter.bind("build_clips"),
            encode_progress_callback=reporter.bind("encode_video"),
        )
        
        return {
            "success": True,
            "artifacts": [str(video_path)],
            "message": f"Successfully generated video: {video_filename}",
            "scripts": scripts # Return scripts for debugging/review
        }

    except Exception as e:
        logger.exception("Video Pipeline Failed")
        reporter.fail(str(e))
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
