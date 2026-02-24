"""
evaluation/metrics.py

Pure metric-computation functions for evaluating multi-agent system artefacts.
No system orchestration happens here — this module is imported by
evaluation_runner.py.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_text_from_pptx(pptx_path: str) -> str:
    """Extract all text from a .pptx file using python-pptx."""
    try:
        from pptx import Presentation
    except ImportError:
        logger.warning("python-pptx not installed; cannot extract slide text.")
        return ""

    path = Path(pptx_path)
    if not pptx_path or not path.is_file():
        return ""

    try:
        prs = Presentation(str(path))
        chunks: List[str] = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    chunks.append(shape.text)
        return "\n".join(chunks)
    except Exception as exc:
        logger.error("Error extracting text from %s: %s", pptx_path, exc)
        return ""


def read_text_file(path: str) -> str:
    """Read a plain-text / code file and return its content."""
    p = Path(path)
    if not path or not p.is_file():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.error("Error reading %s: %s", path, exc)
        return ""


# ---------------------------------------------------------------------------
# 1. Slides
# ---------------------------------------------------------------------------

_SUMMARY_KEYWORDS = {"summary", "conclusion", "recap", "review", "wrap-up", "key takeaways"}
_DENSITY_WORD_LIMIT = 300  # words per slide considered "dense"
_DENSITY_BULLET_LIMIT = 12  # bullet points per slide


def evaluate_slides(
    pptx_path: str,
    topic_keywords: List[str],
    min_slides: int = 5,
    max_slides: int = 12,
) -> Dict[str, Any]:
    """Evaluate a .pptx file for structural and content quality.

    Returns a dict with at least the fields specified in the task spec.
    """
    result: Dict[str, Any] = {
        "slide_count": 0,
        "has_title_slide": False,
        "has_summary_slide": False,
        "slide_density_violations": 0,
        "slide_keyword_coverage": 0.0,
        "slides_pass": False,
    }

    path = Path(pptx_path)
    if not pptx_path or not path.is_file():
        logger.warning("PPTX not found: %s", pptx_path)
        return result

    try:
        from pptx import Presentation
    except ImportError:
        logger.warning("python-pptx not installed; skipping slide evaluation.")
        return result

    try:
        prs = Presentation(str(path))
        slides = list(prs.slides)
        result["slide_count"] = len(slides)

        # --- Title slide --------------------------------------------------
        if slides:
            first = slides[0]
            if first.shapes.title and first.shapes.title.text.strip():
                result["has_title_slide"] = True
            else:
                # Heuristic: if the first slide has *any* text, treat it as a
                # title slide when there are <= 3 shapes (minimal layout).
                texts = [s.text.strip() for s in first.shapes if hasattr(s, "text") and s.text.strip()]
                if texts and len(first.shapes) <= 4:
                    result["has_title_slide"] = True

        # --- Summary slide ------------------------------------------------
        for slide in slides:
            title_text = ""
            if slide.shapes.title and slide.shapes.title.text:
                title_text = slide.shapes.title.text.lower()
            if any(kw in title_text for kw in _SUMMARY_KEYWORDS):
                result["has_summary_slide"] = True
                break

        # --- Density violations -------------------------------------------
        violations = 0
        for slide in slides:
            slide_text = " ".join(
                s.text for s in slide.shapes if hasattr(s, "text")
            )
            word_count = len(slide_text.split())
            bullet_count = slide_text.count("\n")
            if word_count > _DENSITY_WORD_LIMIT or bullet_count > _DENSITY_BULLET_LIMIT:
                violations += 1
        result["slide_density_violations"] = violations

        # --- Keyword coverage ---------------------------------------------
        full_text = extract_text_from_pptx(pptx_path).lower()
        if topic_keywords:
            found = sum(1 for kw in topic_keywords if kw.lower() in full_text)
            result["slide_keyword_coverage"] = round(found / len(topic_keywords), 4)

        # --- Pass / Fail --------------------------------------------------
        sc = result["slide_count"]
        # Pass criteria: page count in range + has title slide
        # (summary slide is optional — not enforced by current system)
        result["slides_pass"] = (
            min_slides <= sc <= max_slides
            and result["has_title_slide"]
        )

    except Exception as exc:
        logger.error("Error evaluating slides %s: %s", pptx_path, exc)

    return result


# ---------------------------------------------------------------------------
# 2. Code
# ---------------------------------------------------------------------------

def evaluate_code(
    code_path: str,
    timeout_sec: int = 30,
    run_dir: str = "",
) -> Dict[str, Any]:
    """Run a Python script via subprocess and report execution metrics.

    Parameters
    ----------
    code_path : str
        Path to the Python script.
    timeout_sec : int
        Maximum execution time.
    run_dir : str
        Working directory for execution.  Falls back to the script's parent
        directory when empty, but using the *run* directory is preferred so
        that relative paths  (e.g. ``assets/``) resolve correctly.
    """
    result: Dict[str, Any] = {
        "code_exec_pass": False,
        "code_runtime_sec": 0.0,
        "code_error_type": None,
        "code_error_msg": None,
    }

    path = Path(code_path)
    if not code_path or not path.is_file():
        result["code_error_type"] = "FileNotFound"
        result["code_error_msg"] = f"{code_path} does not exist"
        return result

    # Prefer run_dir so that assets/ and other siblings are reachable
    cwd = run_dir if run_dir and Path(run_dir).is_dir() else str(path.parent)

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=cwd,
        )
        elapsed = time.perf_counter() - start
        result["code_runtime_sec"] = round(elapsed, 3)

        if proc.returncode == 0:
            result["code_exec_pass"] = True
        else:
            result["code_error_type"] = "RuntimeError"
            result["code_error_msg"] = (proc.stderr or proc.stdout)[:500]

    except subprocess.TimeoutExpired:
        result["code_runtime_sec"] = float(timeout_sec)
        result["code_error_type"] = "Timeout"
        result["code_error_msg"] = f"Exceeded {timeout_sec}s"
    except Exception as exc:
        result["code_error_type"] = type(exc).__name__
        result["code_error_msg"] = str(exc)[:500]

    return result


# ---------------------------------------------------------------------------
# 3. Quiz
# ---------------------------------------------------------------------------

_VALID_ANSWERS = {"A", "B", "C", "D", "0", "1", "2", "3"}
_ANSWER_KEYS = ("answer", "correct_answer")  # support both field names


def evaluate_quiz(
    quiz_json_path: str,
    expected_n_questions: int = 10,
) -> Dict[str, Any]:
    """Validate a quiz.json file for structure and answer legality."""
    result: Dict[str, Any] = {
        "quiz_count": 0,
        "quiz_format_pass": False,
        "quiz_answer_valid_rate": 0.0,
    }

    path = Path(quiz_json_path)
    if not quiz_json_path or not path.is_file():
        return result

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)

        # Accept both a raw list or {"questions": [...]}
        if isinstance(data, dict):
            questions = data.get("questions", data.get("quiz", []))
        elif isinstance(data, list):
            questions = data
        else:
            return result

        result["quiz_count"] = len(questions)

        valid_answers = 0
        format_ok = True

        for q in questions:
            if not isinstance(q, dict):
                format_ok = False
                continue

            # Must have: question text, options (len 4), answer
            opts = q.get("options", [])
            # Support both "answer" and "correct_answer" field names
            raw_ans = ""
            for k in _ANSWER_KEYS:
                if k in q and q[k]:
                    raw_ans = str(q[k]).strip()
                    break
            ans = raw_ans.upper()
            q_text = q.get("question", "")

            if not q_text or len(opts) != 4:
                format_ok = False

            if ans in _VALID_ANSWERS:
                valid_answers += 1

        if result["quiz_count"] > 0:
            result["quiz_answer_valid_rate"] = round(
                valid_answers / result["quiz_count"], 4
            )

        result["quiz_format_pass"] = (
            format_ok
            and result["quiz_count"] == expected_n_questions
            and result["quiz_answer_valid_rate"] == 1.0
        )

    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s", quiz_json_path, exc)
    except Exception as exc:
        logger.error("Error evaluating quiz %s: %s", quiz_json_path, exc)

    return result


# ---------------------------------------------------------------------------
# 4. Video
# ---------------------------------------------------------------------------

def evaluate_video(
    video_path: str,
    min_size_mb: float = 1.0,
) -> Dict[str, Any]:
    """Check video file existence, size, and optionally duration."""
    result: Dict[str, Any] = {
        "video_exists": False,
        "video_size_mb": None,
        "video_duration_sec": None,
        "video_pass": False,
    }

    path = Path(video_path)
    if not video_path or not path.is_file():
        return result

    result["video_exists"] = True
    size_bytes = path.stat().st_size
    result["video_size_mb"] = round(size_bytes / (1024 * 1024), 3)

    # Attempt to get duration via ffprobe (graceful fallback)
    result["video_duration_sec"] = _probe_duration(str(path))

    result["video_pass"] = (
        result["video_exists"]
        and result["video_size_mb"] is not None
        and result["video_size_mb"] >= min_size_mb
    )
    return result


def _probe_duration(video_path: str) -> Optional[float]:
    """Try ffprobe first, then moviepy, then return None."""
    # 1. ffprobe
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return round(float(proc.stdout.strip()), 2)
    except Exception:
        pass

    # 2. moviepy
    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(video_path)
        dur = clip.duration
        clip.close()
        return round(dur, 2)
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# 5. Consistency
# ---------------------------------------------------------------------------

def evaluate_consistency(
    slides_text: str,
    code_text: str,
    quiz_text: str,
    topic_keywords: List[str],
) -> Dict[str, Any]:
    """Cross-artefact consistency via keyword overlap.

    * slide_code_consistency  – fraction of topic_keywords present in BOTH
      slides and code.
    * slide_quiz_coverage_rate – fraction of topic_keywords found in quiz
      text that are also in slides.
    * out_of_scope_rate = 1 - slide_quiz_coverage_rate.
    """
    result: Dict[str, Any] = {
        "slide_code_consistency": 0.0,
        "slide_quiz_coverage_rate": 0.0,
        "out_of_scope_rate": 1.0,
    }

    if not topic_keywords:
        return result

    s_lower = slides_text.lower()
    c_lower = code_text.lower()
    q_lower = quiz_text.lower()

    # slide_code_consistency
    both = sum(
        1 for kw in topic_keywords
        if kw.lower() in s_lower and kw.lower() in c_lower
    )
    result["slide_code_consistency"] = round(both / len(topic_keywords), 4)

    # slide_quiz_coverage_rate
    quiz_kws_in_slides = sum(
        1 for kw in topic_keywords
        if kw.lower() in q_lower and kw.lower() in s_lower
    )
    quiz_kws_total = sum(1 for kw in topic_keywords if kw.lower() in q_lower)
    if quiz_kws_total > 0:
        cov = quiz_kws_in_slides / quiz_kws_total
    else:
        # If none of the keywords appear in quiz, fall back to overall
        cov = sum(1 for kw in topic_keywords if kw.lower() in s_lower) / len(topic_keywords)

    result["slide_quiz_coverage_rate"] = round(cov, 4)
    result["out_of_scope_rate"] = round(1.0 - result["slide_quiz_coverage_rate"], 4)

    return result
