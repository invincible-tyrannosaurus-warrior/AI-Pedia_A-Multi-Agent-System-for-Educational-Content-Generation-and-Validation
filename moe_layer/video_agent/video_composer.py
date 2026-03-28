
from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Callable, List, Optional

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
    EDGE_TTS_IMPORT_ERROR = None
except ImportError as exc:
    edge_tts = None
    EDGE_TTS_AVAILABLE = False
    EDGE_TTS_IMPORT_ERROR = exc

logger = logging.getLogger(__name__)

class VideoComposer:
    """
    Video Agent Sub-module:
    1. TTS: Convert Script -> Audio (mp3)
    2. Rendering: Image + Audio -> Per-slide Video Segment (mp4)
    3. Concatenation: Segments -> Final Video (mp4)
    """
    
    def __init__(
        self,
        voice: str = "en-US-ChristopherNeural",
        video_width: int = 1280,
        video_height: int = 720,
        fps: int = 12,
        ffmpeg_preset: str = "veryfast",
    ):
        self.voice = voice
        self.video_width = video_width
        self.video_height = video_height
        self.fps = fps
        self.ffmpeg_preset = ffmpeg_preset
        self._validate_runtime()

    def _validate_runtime(self) -> None:
        if not EDGE_TTS_AVAILABLE:
            raise RuntimeError(
                "edge-tts is not installed. Install edge-tts before running the video agent. "
                f"Original import error: {EDGE_TTS_IMPORT_ERROR}"
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
        temp_root = output_path.parent / "temp_video"
        audio_dir = temp_root / "audio"
        segments_dir = temp_root / "segments"
        audio_dir.mkdir(parents=True, exist_ok=True)
        segments_dir.mkdir(parents=True, exist_ok=True)
        
        segment_paths: List[Path] = []
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
                audio_path = audio_dir / f"audio_{i + 1:03d}.mp3"
                await self._generate_tts(script, audio_path)

                # 2. Render a per-slide MP4 segment with ffmpeg.
                segment_path = segments_dir / f"segment_{i + 1:03d}.mp4"
                await self._render_segment(img_path, audio_path, segment_path)
                segment_paths.append(segment_path)
                if build_progress_callback is not None:
                    build_progress_callback(
                        stage_progress=(i + 1) / max(total_items, 1),
                        current=i + 1,
                        total=total_items,
                        message=f"Rendered segment {i + 1} of {total_items}.",
                    )
                
            except Exception as e:
                logger.error(f"Failed to create segment for slide {i + 1}: {e}")
                if build_progress_callback is not None:
                    build_progress_callback(
                        stage_progress=(i + 1) / max(total_items, 1),
                        current=i + 1,
                        total=total_items,
                        message=f"Skipped slide {i + 1} after segment error: {e}",
                    )
                continue
        
        if not segment_paths:
            raise RuntimeError("No slide segments were generated successfully.")
        
        # 3. Concatenate slide segments into the final MP4.
        logger.info("Concatenating %d slide segments into %s", len(segment_paths), output_path)
        if encode_progress_callback is not None:
            encode_progress_callback(
                stage_progress=0.0,
                message="Encoding final video.",
                indeterminate=True,
            )
        concat_list_path = segments_dir / "segments.txt"
        self._write_concat_list(concat_list_path, segment_paths)
        await self._concat_segments(concat_list_path, output_path)
        if encode_progress_callback is not None:
            encode_progress_callback(
                stage_progress=1.0,
                message="Final video encoded.",
            )
            
        return output_path

    async def _generate_tts(self, text: str, output_path: Path):
        """Use EdgeTTS to save mp3."""
        if not text.strip():
            # Handle empty script
            text = "..."
            
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(str(output_path))

    async def _render_segment(self, img_path: Path, audio_path: Path, output_path: Path) -> None:
        """Render a single slide segment as MP4 using ffmpeg."""
        video_filter = (
            f"scale={self.video_width}:{self.video_height}:force_original_aspect_ratio=decrease,"
            f"pad={self.video_width}:{self.video_height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"fps={self.fps}"
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            str(img_path),
            "-i",
            str(audio_path),
            "-vf",
            video_filter,
            "-af",
            "apad=pad_dur=0.5",
            "-c:v",
            "libx264",
            "-preset",
            self.ffmpeg_preset,
            "-tune",
            "stillimage",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        await self._run_command(cmd, step_name=f"segment render for {img_path.name}")

    def _write_concat_list(self, concat_list_path: Path, segment_paths: List[Path]) -> None:
        """Write the ffmpeg concat manifest using relative segment filenames."""
        lines = [f"file '{path.name}'" for path in segment_paths]
        concat_list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    async def _concat_segments(self, concat_list_path: Path, output_path: Path) -> None:
        """Concatenate per-slide MP4 segments into the final output."""
        copy_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list_path.name,
            "-c",
            "copy",
            str(output_path),
        ]
        try:
            await self._run_command(
                copy_cmd,
                step_name="fast segment concat",
                cwd=concat_list_path.parent,
            )
        except RuntimeError as exc:
            logger.warning("Fast concat failed, retrying with re-encode: %s", exc)
            reencode_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_list_path.name,
                "-c:v",
                "libx264",
                "-preset",
                self.ffmpeg_preset,
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
            await self._run_command(
                reencode_cmd,
                step_name="segment concat re-encode",
                cwd=concat_list_path.parent,
            )

    async def _run_command(
        self,
        cmd: List[str],
        *,
        step_name: str,
        cwd: Optional[Path] = None,
    ) -> None:
        """Execute a subprocess asynchronously and raise a useful error on failure."""
        logger.info("Running %s: %s", step_name, " ".join(cmd))
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd) if cwd else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            stdout_text = stdout.decode("utf-8", errors="replace").strip()
            detail = stderr_text or stdout_text or "No subprocess output captured."
            raise RuntimeError(f"{step_name} failed (exit {process.returncode}): {detail}")
