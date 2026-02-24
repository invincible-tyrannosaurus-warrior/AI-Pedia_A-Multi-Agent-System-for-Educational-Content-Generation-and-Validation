"""
evaluation/evaluation_runner.py

Main entry-point for the engineering-grade evaluation tool-chain.
Reads a topic config, runs the multi-agent system N times per topic,
collects quality / consistency / system metrics, and writes:
  • results/run_level_results.csv   (one row per run, 6 × N rows)
  • results/topic_best_results.csv  (one row per topic, best run)
  • results/aggregate_summary.json  (mean / std across all runs)

Usage
-----
    python evaluation/evaluation_runner.py \
        --config configs/eval_topics.json  \
        --runs 3                           \
        --out results/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import (
    evaluate_slides,
    evaluate_code,
    evaluate_quiz,
    evaluate_video,
    evaluate_consistency,
    extract_text_from_pptx,
    read_text_file,
)
from evaluation.token_logging import load_run_summary

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("evaluation_run.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Artefact discovery — the real system uses dynamic filenames, so we
# scan by extension instead of expecting fixed names.
# ---------------------------------------------------------------------------

_EXT_MAP = {
    "slides": ".pptx",
    "code":   ".py",
    "quiz":   ".json",
    "video":  ".mp4",
}


def discover_artifacts(run_dir: str) -> Dict[str, Optional[str]]:
    """Discover artefact paths by extension under *run_dir*.

    Search order for each type:
      • slides (.pptx) — run_dir, then run_dir/output
      • code   (.py)   — run_dir/scripts, then run_dir, then run_dir/output
      • quiz   (.json) — run_dir/output, then run_dir
      • video  (.mp4)  — run_dir/output, then run_dir

    Returns a dict ``{"slides": path_or_None, "code": ..., "quiz": ..., "video": ...}``.
    """
    rd = Path(run_dir)
    search_dirs: Dict[str, List[Path]] = {
        "slides": [rd, rd / "output"],
        "code":   [rd / "scripts", rd, rd / "output"],
        "quiz":   [rd / "output", rd],
        "video":  [rd / "output", rd],
    }

    found: Dict[str, Optional[str]] = {}
    for label, ext in _EXT_MAP.items():
        found[label] = None
        for d in search_dirs[label]:
            if not d.exists():
                continue
            # Pick the first file matching the extension (newest mtime)
            candidates = sorted(d.glob(f"*{ext}"), key=lambda p: p.stat().st_mtime, reverse=True)
            # For quiz, skip non-quiz json files (e.g. storyboard.json)
            if label == "quiz":
                candidates = [c for c in candidates if "quiz" in c.stem.lower() or c.stem.lower() == "quiz"]
                if not candidates:
                    # Fallback: any .json that has "questions" key inside
                    all_json = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                    for jf in all_json:
                        try:
                            data = json.loads(jf.read_text(encoding="utf-8"))
                            if isinstance(data, list) or (isinstance(data, dict) and ("questions" in data or "quiz" in data)):
                                candidates = [jf]
                                break
                        except Exception:
                            pass
            if candidates:
                found[label] = str(candidates[0])
                break
    return found

# ---------------------------------------------------------------------------
# Scoring weights (configurable)
# ---------------------------------------------------------------------------
SCORE_WEIGHTS = {
    "slides_pass":              0.20,
    "code_exec_pass":           0.20,
    "quiz_format_pass":         0.20,
    "video_pass":               0.20,
    "slide_code_consistency":   0.10,
    "slide_quiz_coverage_rate": 0.10,
}


# ===================================================================
# 1. run_system — stub / hook for real system execution
# ===================================================================

def run_system(topic_cfg: Dict[str, Any], run_id: int, output_dir: str) -> Dict[str, Any]:
    """Execute the multi-agent system for one (topic, run_id) pair.

    Calls ``stream_workflow()`` from the task-manager, consumes all SSE
    events, and copies generated artefacts into *output_dir*.

    Returns
    -------
    dict
        ``{"success": bool, "error": str | None, "source_run_dir": str | None}``
    """
    od = Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)

    try:
        from manager_agent.task_manager_agent import stream_workflow
        from config import GENERATED_DIR
    except ImportError as exc:
        return {"success": False, "error": f"Import error: {exc}"}

    prompt = topic_cfg.get("prompt", topic_cfg.get("topic_name", ""))
    config = {"slides": True, "video": True, "code": True, "quizzes": True}

    # --- consume the SSE generator ----------------------------------------
    artifact_paths: List[str] = []
    source_run_dir: Optional[str] = None

    try:
        for event_str in stream_workflow(prompt, config=config, files_json="[]"):
            # Parse SSE: "event: <type>\ndata: <json>\n\n"
            if not event_str or not event_str.strip():
                continue
            lines = event_str.strip().split("\n")
            etype = ""
            edata = {}
            for ln in lines:
                if ln.startswith("event: "):
                    etype = ln[7:].strip()
                elif ln.startswith("data: "):
                    try:
                        edata = json.loads(ln[6:])
                    except json.JSONDecodeError:
                        pass

            if etype == "artifact":
                url = edata.get("url", "")
                if url:
                    artifact_paths.append(url)
            elif etype == "error":
                logger.error("System error: %s", edata.get("content", ""))

    except Exception as exc:
        logger.error("stream_workflow crashed: %s", exc)
        return {"success": False, "error": str(exc)}

    # --- locate the source run directory ----------------------------------
    # The system writes to GENERATED_DIR/run_<uuid>/. We find it via the
    # first artifact path.
    if artifact_paths:
        first = Path(artifact_paths[0])
        # Walk up to find the run_* directory
        for parent in [first.parent, first.parent.parent, first.parent.parent.parent]:
            if parent.name.startswith("run_") and parent.parent == GENERATED_DIR:
                source_run_dir = str(parent)
                break

    if not source_run_dir:
        # Try finding the most recent run_* dir in GENERATED_DIR
        run_dirs = sorted(
            [d for d in GENERATED_DIR.iterdir() if d.is_dir() and d.name.startswith("run_")],
            key=lambda p: p.stat().st_mtime, reverse=True
        )
        if run_dirs:
            source_run_dir = str(run_dirs[0])

    if not source_run_dir:
        return {"success": False, "error": "Could not locate system output directory"}

    # --- copy artefacts to evaluation output dir --------------------------
    src = Path(source_run_dir)
    try:
        # Copy everything from source run dir to evaluation output dir
        for item in src.iterdir():
            dest = od / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        logger.info("Copied artefacts: %s → %s", src, od)
    except Exception as exc:
        logger.error("Failed to copy artefacts: %s", exc)
        return {"success": False, "error": f"Copy failed: {exc}"}

    # --- verify -----------------------------------------------------------
    arts = discover_artifacts(output_dir)
    missing = [label for label, path in arts.items() if path is None]

    return {
        "success": len(missing) == 0,
        "error": f"Missing after copy: {', '.join(missing)}" if missing else None,
        "source_run_dir": source_run_dir,
    }


# ===================================================================
# 2. evaluate_run — orchestrates all metric calls for one run
# ===================================================================

def evaluate_run(
    output_dir: str,
    topic_cfg: Dict[str, Any],
    latency_sec: float,
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score every artefact produced by a single run.

    Returns a flat dict ready to be written as one CSV row.
    """
    od = Path(output_dir)
    kw = topic_cfg.get("keywords", [])
    stg = settings or {}

    # --- settings with defaults -------------------------------------------
    min_slides       = stg.get("min_slides", 5)
    max_slides       = stg.get("max_slides", 12)
    expected_quiz_n  = stg.get("expected_quiz_count", 10)
    code_timeout     = stg.get("code_timeout_sec", 30)
    min_video_mb     = stg.get("min_video_size_mb", 1.0)

    # --- discover actual file paths ----------------------------------------
    arts = discover_artifacts(output_dir)
    slides_path = arts["slides"] or ""
    code_path   = arts["code"]   or ""
    quiz_path   = arts["quiz"]   or ""
    video_path  = arts["video"]  or ""

    # --- individual metrics -----------------------------------------------
    s_metrics = evaluate_slides(slides_path, kw,
                                min_slides=min_slides, max_slides=max_slides)
    c_metrics = evaluate_code(code_path, timeout_sec=code_timeout,
                              run_dir=str(od))
    q_metrics = evaluate_quiz(quiz_path, expected_n_questions=expected_quiz_n)
    v_metrics = evaluate_video(video_path, min_size_mb=min_video_mb)

    # --- consistency -------------------------------------------------------
    slides_text = extract_text_from_pptx(slides_path) if slides_path else ""
    code_text   = read_text_file(code_path) if code_path else ""
    quiz_text   = read_text_file(quiz_path) if quiz_path else ""
    cons = evaluate_consistency(slides_text, code_text, quiz_text, kw)

    # --- token / system ----------------------------------------------------
    tok = load_run_summary(output_dir)

    # --- success flag ------------------------------------------------------
    missing = [label for label, path in arts.items() if path is None]
    success_flag = len(missing) == 0

    # --- assemble row ------------------------------------------------------
    row: Dict[str, Any] = {
        # identifiers
        "topic_id":   topic_cfg["topic_id"],
        "run_id":     0,  # filled by caller

        # slides
        "slide_count":            s_metrics.get("slide_count", 0),
        "has_title_slide":        s_metrics.get("has_title_slide", False),
        "has_summary_slide":      s_metrics.get("has_summary_slide", False),
        "slide_density_violations": s_metrics.get("slide_density_violations", 0),
        "slide_keyword_coverage": s_metrics.get("slide_keyword_coverage", 0.0),
        "slides_pass":            s_metrics.get("slides_pass", False),

        # code
        "code_exec_pass":   c_metrics.get("code_exec_pass", False),
        "code_runtime_sec": c_metrics.get("code_runtime_sec", 0.0),
        "code_error_type":  c_metrics.get("code_error_type"),
        "code_error_msg":   c_metrics.get("code_error_msg"),

        # quiz
        "quiz_count":             q_metrics.get("quiz_count", 0),
        "quiz_format_pass":       q_metrics.get("quiz_format_pass", False),
        "quiz_answer_valid_rate": q_metrics.get("quiz_answer_valid_rate", 0.0),

        # video
        "video_exists":       v_metrics.get("video_exists", False),
        "video_size_mb":      v_metrics.get("video_size_mb"),
        "video_duration_sec": v_metrics.get("video_duration_sec"),
        "video_pass":         v_metrics.get("video_pass", False),

        # consistency
        "slide_code_consistency":   cons.get("slide_code_consistency", 0.0),
        "slide_quiz_coverage_rate": cons.get("slide_quiz_coverage_rate", 0.0),
        "out_of_scope_rate":        cons.get("out_of_scope_rate", 1.0),

        # system
        "latency_sec":   round(latency_sec, 2),
        "total_tokens":  tok["total_tokens"] if tok else 0,
        "tokens_by_agent": json.dumps(tok["tokens_by_agent"]) if tok else "{}",
        "llm_call_count": tok["llm_call_count"] if tok else 0,
        "success_flag":  success_flag,
    }

    row["overall_score"] = compute_overall_score(row)
    return row


# ===================================================================
# 3. compute_overall_score
# ===================================================================

def compute_overall_score(row: Dict[str, Any]) -> float:
    """Weighted score used for best-run selection.

    Components (default weights in SCORE_WEIGHTS):
        slides_pass              20 %
        code_exec_pass           20 %
        quiz_format_pass         20 %
        video_pass               20 %
        slide_code_consistency   10 %
        slide_quiz_coverage_rate 10 %

    All boolean values are cast to 0/1.  Float values are kept as-is (0–1).
    """
    score = 0.0
    for key, weight in SCORE_WEIGHTS.items():
        val = row.get(key, 0)
        if isinstance(val, bool):
            val = 1.0 if val else 0.0
        try:
            score += float(val) * weight
        except (TypeError, ValueError):
            pass
    return round(score, 4)


# ===================================================================
# 4. select_best_run
# ===================================================================

def select_best_run(run_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pick the best run from multiple runs of the same topic.

    Selection logic
    ~~~~~~~~~~~~~~~
    1. Only rows where ``success_flag is True`` are eligible.
    2. Among eligible rows, choose the one with the **highest** ``overall_score``.
    3. If no run succeeded, return the first row with ``success_flag=False``.
    """
    successful = [r for r in run_rows if r.get("success_flag")]
    if successful:
        return max(successful, key=lambda r: r.get("overall_score", 0))
    # No success → return first (so it still appears in best results)
    return run_rows[0] if run_rows else {}


# ===================================================================
# 5. Aggregate summary
# ===================================================================

_NUMERIC_KEYS_FOR_AGG = [
    "overall_score",
    "latency_sec",
    "total_tokens",
    "slide_count",
    "slide_keyword_coverage",
    "slide_code_consistency",
    "slide_quiz_coverage_rate",
    "out_of_scope_rate",
    "code_runtime_sec",
    "quiz_answer_valid_rate",
    "video_size_mb",
    "video_duration_sec",
]


def _compute_aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean / std for selected numeric columns."""
    import math

    agg: Dict[str, Any] = {}
    for key in _NUMERIC_KEYS_FOR_AGG:
        vals = []
        for r in rows:
            v = r.get(key)
            if v is not None:
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
        if vals:
            mean = sum(vals) / len(vals)
            variance = sum((x - mean) ** 2 for x in vals) / len(vals) if len(vals) > 1 else 0.0
            agg[key] = {
                "mean": round(mean, 4),
                "std": round(math.sqrt(variance), 4),
                "n": len(vals),
            }
        else:
            agg[key] = {"mean": None, "std": None, "n": 0}

    # success rate
    total = len(rows)
    successes = sum(1 for r in rows if r.get("success_flag"))
    agg["success_rate"] = round(successes / total, 4) if total else 0.0

    # pass rates
    for flag in ("slides_pass", "code_exec_pass", "quiz_format_pass", "video_pass"):
        passed = sum(1 for r in rows if r.get(flag))
        agg[f"{flag}_rate"] = round(passed / total, 4) if total else 0.0

    return agg


# ===================================================================
# 6. CSV writing
# ===================================================================

def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        logger.warning("No rows to write to %s", path)
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows → %s", len(rows), path)


# ===================================================================
# 7. Main driver
# ===================================================================

def main(config_path: str, n_runs: int, out_dir: str) -> None:
    """End-to-end evaluation driver."""
    # --- load config ------------------------------------------------------
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    topics: List[Dict[str, Any]] = cfg["topics"]
    settings = cfg.get("settings", {})

    logger.info("Loaded %d topics from %s  (runs_per_topic=%d)", len(topics), config_path, n_runs)

    all_rows: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []

    for t_cfg in topics:
        tid = t_cfg["topic_id"]
        topic_runs: List[Dict[str, Any]] = []
        logger.info("━━━ Topic: %s (%s) ━━━", tid, t_cfg["topic_name"])

        for run_id in range(1, n_runs + 1):
            run_out = str(Path(out_dir) / tid / f"run_{run_id}")
            Path(run_out).mkdir(parents=True, exist_ok=True)

            logger.info("  ▸ run %d/%d  →  %s", run_id, n_runs, run_out)

            # ---- execute system ------------------------------------------
            t0 = time.time()
            try:
                sys_result = run_system(t_cfg, run_id, run_out)
            except Exception as exc:
                logger.error("  ✗ system crashed: %s", exc)
                sys_result = {"success": False, "error": str(exc)}
            latency = time.time() - t0

            # ---- evaluate ------------------------------------------------
            try:
                row = evaluate_run(run_out, t_cfg, latency, settings=settings)
            except Exception as exc:
                logger.error("  ✗ evaluation crashed: %s", exc)
                row = {
                    "topic_id": tid,
                    "run_id": run_id,
                    "success_flag": False,
                    "overall_score": 0.0,
                    "latency_sec": round(latency, 2),
                }

            row["run_id"] = run_id

            # If run_system itself reported failure, override
            if not sys_result.get("success", True):
                row["success_flag"] = False
                if sys_result.get("error"):
                    row.setdefault("code_error_msg", sys_result["error"])

            topic_runs.append(row)
            all_rows.append(row)
            logger.info("    score=%.3f  success=%s", row.get("overall_score", 0), row.get("success_flag"))

        # best-run for this topic
        best = select_best_run(topic_runs)
        best_rows.append(best)

    # ---- write outputs ---------------------------------------------------
    run_csv  = str(Path(out_dir) / "run_level_results.csv")
    best_csv = str(Path(out_dir) / "topic_best_results.csv")
    agg_json = str(Path(out_dir) / "aggregate_summary.json")

    _write_csv(run_csv, all_rows)
    _write_csv(best_csv, best_rows)

    agg = _compute_aggregate(all_rows)
    agg["generated_at"] = datetime.now().isoformat()
    agg["config_path"] = config_path
    agg["n_topics"] = len(topics)
    agg["runs_per_topic"] = n_runs

    Path(agg_json).parent.mkdir(parents=True, exist_ok=True)
    with open(agg_json, "w", encoding="utf-8") as fh:
        json.dump(agg, fh, indent=2, ensure_ascii=False)
    logger.info("Wrote aggregate summary → %s", agg_json)

    logger.info("✔  Evaluation finished.  %d runs, %d topics.", len(all_rows), len(best_rows))


# ===================================================================
# CLI
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the evaluation tool-chain for the multi-agent system.",
    )
    parser.add_argument(
        "--config", default="configs/eval_topics.json",
        help="Path to the topic configuration JSON (default: configs/eval_topics.json)",
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of runs per topic (default: 3)",
    )
    parser.add_argument(
        "--out", default="results/",
        help="Output directory for CSV / JSON results (default: results/)",
    )
    args = parser.parse_args()
    main(args.config, args.runs, args.out)
