
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Callable, List, Optional
import asyncio

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
    EDGE_TTS_IMPORT_ERROR = None
except ImportError as exc:
    edge_tts = None
    EDGE_TTS_AVAILABLE = False
    EDGE_TTS_IMPORT_ERROR = exc

try:
    from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
    MOVIEPY_AVAILABLE = True
    MOVIEPY_IMPORT_ERROR = None
except ImportError as exc:
    MOVIEPY_AVAILABLE = False
    MOVIEPY_IMPORT_ERROR = exc
    logging.warning("MoviePy import failed (%s). Video generation is unavailable.", exc)

logger = logging.getLogger(__name__)

class VideoComposer:
    """
    Video Agent Sub-module:
    1. TTS: Convert Script -> Audio (mp3)
    2. Stitching: Image + Audio -> Video Clip (mp4)
    3. Concatenation: Clips -> Final Video
    """
    
    def __init__(self, voice: str = "en-US-ChristopherNeural"):
        self.voice = voice
        self._validate_runtime()

    def _validate_runtime(self) -> None:
        if not EDGE_TTS_AVAILABLE:
            raise RuntimeError(
                "edge-tts is not installed. Install edge-tts before running the video agent. "
                f"Original import error: {EDGE_TTS_IMPORT_ERROR}"
            )
        if not MOVIEPY_AVAILABLE:
            raise RuntimeError(
                "MoviePy is unavailable. Install moviepy<2.0.0 before running the video agent. "
                f"Original import error: {MOVIEPY_IMPORT_ERROR}"
            )
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is not available on PATH. Install ffmpeg before running the video agent.")

    async def compose_video(
        self,
        image_paths: List[Path],
        scripts: List[str],
        output_path: Path,
        build_progress_callback: Optional[Callable[..., None]] = None,
        encode_progress_callback: Optional[Callable[..., None]] = None,
    ) -> Path:
        """
        Main entry point: Compose a full video from images and scripts.
        """
        logger.info(f"Composing video to {output_path}...")
        temp_dir = output_path.parent / "temp_clips"
        temp_dir.mkdir(exist_ok=True)
        
        clips = []
        total_items = min(len(image_paths), len(scripts))
        if build_progress_callback is not None:
            build_progress_callback(
                stage_progress=0.0,
                current=0,
                total=total_items,
                message="Building slide clips.",
            )
        
        for i, (img_path, script) in enumerate(zip(image_paths, scripts)):
            try:
                # 1. Generate Audio
                audio_path = temp_dir / f"audio_{i}.mp3"
                await self._generate_tts(script, audio_path)
                
                # 2. Create Video Clip (Image + Audio)
                # MoviePy is synchronous, so it might block lightly, but it's CPU bound.
                # In a real heavy app, we'd run this in a thread pool.
                clip = self._create_clip(img_path, audio_path)
                clips.append(clip)
                if build_progress_callback is not None:
                    build_progress_callback(
                        stage_progress=(i + 1) / max(total_items, 1),
                        current=i + 1,
                        total=total_items,
                        message=f"Built clip {i + 1} of {total_items}.",
                    )
                
            except Exception as e:
                logger.error(f"Failed to create clip for slide {i}: {e}")
                if build_progress_callback is not None:
                    build_progress_callback(
                        stage_progress=(i + 1) / max(total_items, 1),
                        current=i + 1,
                        total=total_items,
                        message=f"Skipped slide {i + 1} after clip error: {e}",
                    )
                continue
        
        if not clips:
            raise RuntimeError("No clips were generated successfully.")
            
        # 3. Concatenate
        final_video = concatenate_videoclips(clips, method="compose")
        
        # 4. Write File
        # We run this in a thread because video encoding is heavy and blocking
        logger.info("Rendering final video...")
        if encode_progress_callback is not None:
            encode_progress_callback(
                stage_progress=0.0,
                message="Encoding final video.",
                indeterminate=True,
            )
        await asyncio.to_thread(
            final_video.write_videofile,
            str(output_path),
            fps=24,
            codec="libx264",
            audio_codec="aac",
            logger=None # Silence moviepy logger to keep stdout clean
        )
        if encode_progress_callback is not None:
            encode_progress_callback(
                stage_progress=1.0,
                message="Final video encoded.",
            )
        
        # Cleanup clips memory (MoviePy caveat)
        for clip in clips:
            clip.close()
            
        return output_path

    async def _generate_tts(self, text: str, output_path: Path):
        """Use EdgeTTS to save mp3."""
        if not text.strip():
            # Handle empty script
            text = "..."
            
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(str(output_path))

    def _create_clip(self, img_path: Path, audio_path: Path):
        """Create a MoviePy ImageClip with Audio set."""
        # Load Audio
        # AudioFileClip needs a string path
        audio_clip = AudioFileClip(str(audio_path))
        
        # Load Image, set duration to audio duration
        # Adding a small buffer (0.5s) for pacing
        video_clip = ImageClip(str(img_path)).set_duration(audio_clip.duration + 0.5)
        
        # Check image resizing (optional, assuming images are already 16:9 from PPT)
        # video_clip = video_clip.resize(width=1920) 
        
        video_clip = video_clip.set_audio(audio_clip)
        video_clip.fps = 24
        
        return video_clip
