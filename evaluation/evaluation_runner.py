"""Evaluation runner implementing the paper-aligned workflow."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing as mp
import queue
import re
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import GENERATED_DIR
from evaluation.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONCURRENT_BATCHES,
    DEFAULT_SINGLE_UNITS,
    EXPECTED_QUIZ_QUESTION_COUNT,
    LATENCY_BUDGET_SEC,
    REQUIRED_ARTIFACTS,
    SILENT_FAILURE_INACTIVITY_SEC,
)
from evaluation.metrics import (
    compute_concept_metrics,
    evaluate_code,
    evaluate_quiz,
    evaluate_slides,
    evaluate_video,
    read_text_file,
)
from evaluation.rookie_eval import run_rookie_evaluation
from evaluation.sanity import evaluate_sanity
from evaluation.scoring import compute_overall_score
from evaluation.token_logging import load_run_summary

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
# Worker + SSE parsing
# ---------------------------------------------------------------------------

def _stream_workflow_worker(prompt: str, config: Dict[str, Any], files_json: str, out_q: mp.Queue) -> None:
    """Process worker that streams SSE events from the system."""
    try:
        from manager_agent.task_manager_agent import stream_workflow

        for event_str in stream_workflow(prompt, config=config, files_json=files_json):
            out_q.put({"type": "event", "payload": event_str})
        out_q.put({"type": "done"})
    except Exception as exc:  # noqa: BLE001
        out_q.put(
            {
                "type": "worker_error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def _parse_sse_event(event_str: str) -> Tuple[str, Dict[str, Any]]:
    event_type = ""
    data: Dict[str, Any] = {}

    if not event_str:
        return event_type, data

    for line in event_str.strip().splitlines():
        if line.startswith("event: "):
            event_type = line[len("event: ") :].strip()
        elif line.startswith("data: "):
            raw = line[len("data: ") :].strip()
            try:
                data = json.loads(raw)
            except Exception:
                data = {"raw": raw}
    return event_type, data


# ---------------------------------------------------------------------------
# Artifact utilities
# ---------------------------------------------------------------------------

_EXT_MAP = {
    "slides": ".pptx",
    "code": ".py",
    "quiz": ".json",
    "video": ".mp4",
}


def discover_artifacts(run_dir: str) -> Dict[str, Optional[str]]:
    rd = Path(run_dir)
    search_dirs: Dict[str, List[Path]] = {
        "slides": [rd, rd / "output"],
        "code": [rd / "scripts", rd, rd / "output"],
        "quiz": [rd / "output", rd],
        "video": [rd / "output", rd],
    }

    found: Dict[str, Optional[str]] = {k: None for k in _EXT_MAP}

    for label, ext in _EXT_MAP.items():
        for d in search_dirs[label]:
            if not d.exists() or not d.is_dir():
                continue

            candidates = sorted(d.glob(f"*{ext}"), key=lambda p: p.stat().st_mtime, reverse=True)

            if label == "quiz":
                filtered: List[Path] = []
                for c in candidates:
                    if "quiz" in c.stem.lower() or c.stem.lower() == "quiz":
                        filtered.append(c)
                        continue
                    try:
                        payload = json.loads(c.read_text(encoding="utf-8"))
                        if isinstance(payload, list):
                            filtered.append(c)
                        elif isinstance(payload, dict) and ("questions" in payload or "quiz" in payload):
                            filtered.append(c)
                    except Exception:
                        continue
                candidates = filtered

            if candidates:
                found[label] = str(candidates[0].resolve())
                break

    return found


def _copy_run_dir(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        target = dst_dir / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _find_source_run_dir(artifact_urls: List[str], run_start_ts: float) -> Optional[Path]:
    # 1) Try explicit artifact paths first (most reliable, including concurrent runs)
    for url in artifact_urls:
        p = Path(url)
        candidates = [p]
        if not p.is_absolute():
            candidates.append((PROJECT_ROOT / p).resolve())

        for cp in candidates:
            for parent in [cp] + list(cp.parents):
                if parent.name.startswith("run_") and parent.parent == GENERATED_DIR:
                    return parent

    # 2) Fallback to recent run_* dirs created after this run started
    recent = [
        d
        for d in GENERATED_DIR.iterdir()
        if d.is_dir() and d.name.startswith("run_") and d.stat().st_mtime >= (run_start_ts - 1.0)
    ]
    recent.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    if recent:
        return recent[0]

    return None


def _split_script_segments(script_blob: str) -> List[str]:
    if not script_blob:
        return []
    parts = [p.strip() for p in re.split(r"\n{2,}", script_blob) if p.strip()]
    return parts


# ---------------------------------------------------------------------------
# System execution for one request
# ---------------------------------------------------------------------------

def run_system_once(
    prompt: str,
    output_dir: str,
    latency_budget_sec: int,
    inactivity_sec: int,
) -> Dict[str, Any]:
    run_start = time.time()

    generation_config = {"slides": True, "video": True, "code": True, "quizzes": True}

    event_count = 0
    workflow_completed = False
    artifact_urls: List[str] = []
    error_messages: List[str] = []
    script_blobs: List[str] = []

    task_to_agent: Dict[str, str] = {}
    started_agents: set[str] = set()
    failed_agents: set[str] = set()

    q: mp.Queue = mp.Queue()
    p = mp.Process(
        target=_stream_workflow_worker,
        args=(prompt, generation_config, "[]", q),
        daemon=True,
    )
    p.start()

    last_event_ts = time.time()
    silent_failure = False
    latency_exceeded = False
    worker_error: Optional[str] = None

    while True:
        now = time.time()
        elapsed = now - run_start

        if elapsed > latency_budget_sec:
            latency_exceeded = True
            break

        try:
            msg = q.get(timeout=1.0)
        except queue.Empty:
            if (time.time() - last_event_ts) > inactivity_sec and not workflow_completed:
                silent_failure = True
                break
            if not p.is_alive() and q.empty():
                break
            continue

        mtype = msg.get("type")
        if mtype == "event":
            event_count += 1
            last_event_ts = time.time()
            event_str = msg.get("payload", "")
            event_type, data = _parse_sse_event(event_str)

            if event_type == "workflow_complete":
                workflow_completed = True
            elif event_type == "error":
                err = str(data.get("content", "Unknown system error"))
                error_messages.append(err)
            elif event_type == "artifact":
                url = str(data.get("url", ""))
                if url:
                    artifact_urls.append(url)
            elif event_type == "script":
                content = str(data.get("content", ""))
                if content.strip():
                    script_blobs.append(content)
            elif event_type == "step_start":
                t_id = str(data.get("id", ""))
                agent = str(data.get("agent", "")).strip().lower()
                if t_id and agent:
                    task_to_agent[t_id] = agent
                    started_agents.add(agent)
            elif event_type == "step_complete":
                t_id = str(data.get("id", ""))
                status = str(data.get("status", "")).strip().lower()
                agent = task_to_agent.get(t_id)
                if status == "fail" and agent:
                    failed_agents.add(agent)

        elif mtype == "worker_error":
            worker_error = str(msg.get("error", "worker_error"))
            tb = msg.get("traceback")
            if tb:
                logger.error("stream worker traceback:\n%s", tb)
            break
        elif mtype == "done":
            break

    # Cleanup process
    if p.is_alive():
        p.terminate()
        p.join(timeout=3)
        if p.is_alive():
            p.kill()
            p.join(timeout=1)
    else:
        p.join(timeout=1)

    generation_latency_sec = round(time.time() - run_start, 3)

    if worker_error:
        error_messages.append(worker_error)

    source_run_dir = _find_source_run_dir(artifact_urls, run_start_ts=run_start)

    # Copy artifacts to evaluation output directory for reproducibility
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if source_run_dir and source_run_dir.exists():
        try:
            _copy_run_dir(source_run_dir, output_path)
        except Exception as exc:  # noqa: BLE001
            error_messages.append(f"copy_failed:{exc}")

    script_blob = script_blobs[-1] if script_blobs else ""
    script_segments = _split_script_segments(script_blob)

    return {
        "workflow_completed": workflow_completed,
        "silent_failure": silent_failure,
        "latency_exceeded": latency_exceeded,
        "generation_latency_sec": generation_latency_sec,
        "artifact_urls": artifact_urls,
        "source_run_dir": str(source_run_dir) if source_run_dir else None,
        "error_messages": error_messages,
        "script_text": script_blob,
        "script_segments": script_segments,
        "event_count": event_count,
        "started_agents": sorted(started_agents),
        "failed_agents": sorted(failed_agents),
    }


# ---------------------------------------------------------------------------
# Per-request evaluation
# ---------------------------------------------------------------------------

def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return str(value)


def evaluate_request(
    spec: Dict[str, Any],
    run_out_dir: str,
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    topic_cfg = spec["topic_cfg"]
    prompt = topic_cfg.get("prompt", topic_cfg.get("topic_name", ""))

    sys_result = run_system_once(
        prompt=prompt,
        output_dir=run_out_dir,
        latency_budget_sec=int(settings.get("latency_budget_sec", LATENCY_BUDGET_SEC)),
        inactivity_sec=int(settings.get("silent_inactivity_sec", SILENT_FAILURE_INACTIVITY_SEC)),
    )

    arts = discover_artifacts(run_out_dir)
    missing_artifacts = [k for k in REQUIRED_ARTIFACTS if not arts.get(k)]

    slides_metrics = evaluate_slides(arts.get("slides") or "")
    code_metrics = evaluate_code(
        arts.get("code") or "",
        timeout_sec=int(settings.get("code_timeout_sec", 30 * 60)),
        run_dir=run_out_dir,
        allow_auto_install=True,
        pip_install_timeout_sec=int(settings.get("pip_install_timeout_sec", 20 * 60)),
    )
    quiz_metrics = evaluate_quiz(
        arts.get("quiz") or "",
        expected_n_questions=EXPECTED_QUIZ_QUESTION_COUNT,
    )
    video_metrics = evaluate_video(arts.get("video") or "")

    slides_text = slides_metrics.get("slides_text", "")
    code_text = read_text_file(arts.get("code") or "") if arts.get("code") else ""
    quiz_text = quiz_metrics.get("quiz_text", "")

    script_segments = sys_result.get("script_segments") or []
    video_text = "\n\n".join(script_segments)

    concept = compute_concept_metrics(
        slides_text=slides_text,
        code_text=code_text,
        quiz_text=quiz_text,
        video_text=video_text,
        slide_segments=slides_metrics.get("slide_texts", []),
        video_segments=script_segments,
    )

    rookie = run_rookie_evaluation(
        original_questions=quiz_metrics.get("quiz_questions", []),
        materials_text=(slides_text + "\n\n" + code_text + "\n\n" + video_text).strip(),
        run_seed=int(spec["request_index"]),
    )

    per_model_ok = [x for x in rookie.get("per_model_results", []) if x.get("ok")]
    sanity = evaluate_sanity(per_model_ok)

    # Token logs: prefer source_run_dir, fallback to copied run dir
    token_summary = None
    if sys_result.get("source_run_dir"):
        token_summary = load_run_summary(sys_result["source_run_dir"])
    if token_summary is None:
        token_summary = load_run_summary(run_out_dir)

    tokens_total = int((token_summary or {}).get("tokens_total", 0))
    tokens_by_agent = (token_summary or {}).get(
        "tokens_by_agent",
        {"coder": 0, "presentation": 0, "quiz": 0, "video": 0},
    )
    llm_call_count = int((token_summary or {}).get("llm_call_count", 0))

    # Hard gates
    fail_reasons: List[str] = []

    if missing_artifacts:
        fail_reasons.append("missing_artifact")

    if sys_result.get("generation_latency_sec", 0.0) > float(settings.get("latency_budget_sec", LATENCY_BUDGET_SEC)):
        fail_reasons.append("latency_exceeded")

    if sys_result.get("latency_exceeded"):
        if "latency_exceeded" not in fail_reasons:
            fail_reasons.append("latency_exceeded")

    if sys_result.get("silent_failure"):
        fail_reasons.append("silent_failure")

    if not sys_result.get("workflow_completed"):
        fail_reasons.append("workflow_incomplete")

    if rookie.get("rookie_eval_failed"):
        fail_reasons.append("rookie_eval_failed")

    if sanity.get("sanity_fail"):
        fail_reasons.append("sanity_fail")

    # Partial failure (separate system-robustness axis)
    artifact_present_n = sum(1 for k in REQUIRED_ARTIFACTS if arts.get(k))
    partial_failure = False
    if 0 < artifact_present_n < len(REQUIRED_ARTIFACTS):
        partial_failure = True
    if sys_result.get("failed_agents"):
        partial_failure = True
    if sys_result.get("error_messages"):
        partial_failure = True
    if not code_metrics.get("code_exec_pass", False):
        partial_failure = True
    if not quiz_metrics.get("quiz_format_validity", False):
        partial_failure = True
    if not slides_metrics.get("slides_pass", False):
        partial_failure = True
    if not video_metrics.get("video_pass", False):
        partial_failure = True

    success_flag = len(fail_reasons) == 0

    row: Dict[str, Any] = {
        "topic_id": topic_cfg.get("topic_id", ""),
        "topic_name": topic_cfg.get("topic_name", ""),
        "request_index": spec["request_index"],
        "unit_type": spec["unit_type"],
        "unit_id": spec["unit_id"],
        "batch_slot": spec["batch_slot"],

        "workflow_completed": bool(sys_result.get("workflow_completed")),
        "silent_failure": bool(sys_result.get("silent_failure")),
        "partial_failure": bool(partial_failure),
        "event_count": int(sys_result.get("event_count", 0)),
        "failed_agents": _json_dumps(sys_result.get("failed_agents", [])),
        "system_errors": _json_dumps(sys_result.get("error_messages", [])),

        "latency_sec": round(float(sys_result.get("generation_latency_sec", 0.0)), 3),

        "slides_path": arts.get("slides") or "",
        "code_path": arts.get("code") or "",
        "quiz_path": arts.get("quiz") or "",
        "video_path": arts.get("video") or "",
        "missing_artifacts": _json_dumps(missing_artifacts),

        "slide_count": int(slides_metrics.get("slide_count", 0)),
        "title_slide_count": int(slides_metrics.get("title_slide_count", 0)),
        "content_slide_count": int(slides_metrics.get("content_slide_count", 0)),
        "summary_slide_count": int(slides_metrics.get("summary_slide_count", 0)),
        "slide_structural_compliance": float(slides_metrics.get("slide_structural_compliance", 0.0)),
        "slides_pass": bool(slides_metrics.get("slides_pass", False)),

        "code_exec_pass": bool(code_metrics.get("code_exec_pass", False)),
        "code_runtime_sec": float(code_metrics.get("code_runtime_sec", 0.0)),
        "code_error_type": code_metrics.get("code_error_type"),
        "code_error_msg": code_metrics.get("code_error_msg"),
        "auto_install_attempted": bool(code_metrics.get("auto_install_attempted", False)),
        "auto_install_succeeded": bool(code_metrics.get("auto_install_succeeded", False)),
        "auto_installed_packages": _json_dumps(code_metrics.get("auto_installed_packages", [])),

        "quiz_count": int(quiz_metrics.get("quiz_count", 0)),
        "quiz_format_validity": bool(quiz_metrics.get("quiz_format_validity", False)),
        "quiz_answer_valid_rate": float(quiz_metrics.get("quiz_answer_valid_rate", 0.0)),
        "quiz_explanation_rate": float(quiz_metrics.get("quiz_explanation_rate", 0.0)),
        "quiz_parse_error": quiz_metrics.get("quiz_parse_error"),

        "video_exists": bool(video_metrics.get("video_exists", False)),
        "video_size_mb": float(video_metrics.get("video_size_mb", 0.0)),
        "video_duration_sec": video_metrics.get("video_duration_sec"),
        "video_pass": bool(video_metrics.get("video_pass", False)),

        "AlignSC": float(concept.get("AlignSC", 0.0)),
        "Coverage": float(concept.get("Coverage", 0.0)),
        "OutOfScope": float(concept.get("OutOfScope", 1.0)),
        "Consistency": float(concept.get("Consistency", 0.0)),
        "Sync": float(concept.get("Sync", 0.0)),
        "KS": _json_dumps(concept.get("KS", [])),
        "KC": _json_dumps(concept.get("KC", [])),
        "KQ": _json_dumps(concept.get("KQ", [])),
        "KV": _json_dumps(concept.get("KV", [])),

        "Acc_ori_pre": float(rookie.get("Acc_ori_pre", 0.0)),
        "Acc_ori_post": float(rookie.get("Acc_ori_post", 0.0)),
        "Acc_eq_pre": float(rookie.get("Acc_eq_pre", 0.0)),
        "Acc_eq_post": float(rookie.get("Acc_eq_post", 0.0)),
        "Accpre": float(rookie.get("Accpre", 0.0)),
        "Accpost": float(rookie.get("Accpost", 0.0)),
        "Acccontrol": float(rookie.get("Acccontrol", 0.0)),
        "Δori": float(rookie.get("Δori", 0.0)),
        "Δeq": float(rookie.get("Δeq", 0.0)),
        "ΔAcc": float(rookie.get("ΔAcc", 0.0)),
        "ΔAccnet": float(rookie.get("ΔAccnet", 0.0)),
        "rookie_eval_failed": bool(rookie.get("rookie_eval_failed", False)),
        "rookie_eval_error": rookie.get("rookie_eval_error"),
        "rookie_models_success_n": int(rookie.get("rookie_models_success_n", 0)),
        "rookie_failed_models": _json_dumps(rookie.get("rookie_failed_models", [])),
        "rookie_per_model_results": _json_dumps(rookie.get("per_model_results", [])),

        "sanity_warnings": _json_dumps(sanity.get("sanity_warnings", [])),
        "sanity_fail_reasons": _json_dumps(sanity.get("sanity_fail_reasons", [])),
        "sanity_fail": bool(sanity.get("sanity_fail", False)),

        "tokens_total": int(tokens_total),
        "tokens_by_agent": _json_dumps(tokens_by_agent),
        "llm_call_count": int(llm_call_count),

        "success_flag": bool(success_flag),
        "hard_fail_reasons": _json_dumps(fail_reasons),
    }

    if success_flag:
        row["overall_score"] = compute_overall_score(row)
    else:
        row["overall_score"] = 0.0

    return row


# ---------------------------------------------------------------------------
# Scheduling and output
# ---------------------------------------------------------------------------

def _build_request_plan(
    topics: List[Dict[str, Any]],
    single_units: int,
    concurrent_batches: int,
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    if not topics:
        return [], []

    topic_iter = cycle(topics)
    req_idx = 1

    singles: List[Dict[str, Any]] = []
    for u in range(1, single_units + 1):
        topic_cfg = next(topic_iter)
        singles.append(
            {
                "request_index": req_idx,
                "unit_type": "single",
                "unit_id": u,
                "batch_slot": 1,
                "topic_cfg": topic_cfg,
            }
        )
        req_idx += 1

    batches: List[List[Dict[str, Any]]] = []
    for b in range(1, concurrent_batches + 1):
        group: List[Dict[str, Any]] = []
        for slot in range(1, batch_size + 1):
            topic_cfg = next(topic_iter)
            group.append(
                {
                    "request_index": req_idx,
                    "unit_type": "batch",
                    "unit_id": b,
                    "batch_slot": slot,
                    "topic_cfg": topic_cfg,
                }
            )
            req_idx += 1
        batches.append(group)

    return singles, batches


def _row_run_dir(out_dir: Path, spec: Dict[str, Any]) -> Path:
    topic_id = spec["topic_cfg"].get("topic_id", "topic")
    req = spec["request_index"]
    if spec["unit_type"] == "single":
        label = f"single_u{spec['unit_id']:03d}_req{req:04d}_{topic_id}"
    else:
        label = f"batch_u{spec['unit_id']:03d}_slot{spec['batch_slot']}_req{req:04d}_{topic_id}"
    return out_dir / "runs" / label


def _select_best_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_topic: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_topic.setdefault(str(r.get("topic_id", "")), []).append(r)

    best_rows: List[Dict[str, Any]] = []
    for topic_id, topic_rows in by_topic.items():
        successful = [x for x in topic_rows if x.get("success_flag")]
        if successful:
            best = max(successful, key=lambda x: float(x.get("overall_score", 0.0)))
        else:
            best = max(topic_rows, key=lambda x: float(x.get("overall_score", 0.0)))
        best_rows.append(best)

    # Stable order by topic_id
    best_rows.sort(key=lambda x: str(x.get("topic_id", "")))
    return best_rows


def _collect_headers(rows: List[Dict[str, Any]]) -> List[str]:
    headers: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                headers.append(k)
    return headers


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    headers = _collect_headers(rows)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in headers})


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def _std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if len(vals) <= 1:
        return 0.0
    m = _mean(vals)
    var = sum((x - m) ** 2 for x in vals) / len(vals)
    return var ** 0.5


def _aggregate_summary(rows: List[Dict[str, Any]], settings: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    total = len(rows)

    completion_rate = _mean([1.0 if r.get("workflow_completed") else 0.0 for r in rows]) if rows else 0.0
    partial_failure_rate = _mean([1.0 if r.get("partial_failure") else 0.0 for r in rows]) if rows else 0.0
    silent_failure_rate = _mean([1.0 if r.get("silent_failure") else 0.0 for r in rows]) if rows else 0.0
    success_rate = _mean([1.0 if r.get("success_flag") else 0.0 for r in rows]) if rows else 0.0

    latency_mean = _mean([float(r.get("latency_sec", 0.0)) for r in rows]) if rows else 0.0
    tokens_total_sum = int(sum(int(r.get("tokens_total", 0)) for r in rows))

    tokens_by_agent_total = {"coder": 0, "presentation": 0, "quiz": 0, "video": 0}
    for r in rows:
        raw = r.get("tokens_by_agent", "{}")
        try:
            payload = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except Exception:
            payload = {}
        for k in tokens_by_agent_total:
            tokens_by_agent_total[k] += int(payload.get(k, 0))

    success_rows = [r for r in rows if r.get("success_flag")]

    paper_fields = [
        "Accpre",
        "Accpost",
        "Acccontrol",
        "ΔAcc",
        "ΔAccnet",
        "AlignSC",
        "Coverage",
        "OutOfScope",
        "Consistency",
        "Sync",
    ]

    paper_all_runs_mean = {
        f: round(_mean([float(r.get(f, 0.0)) for r in rows]), 4) if rows else 0.0 for f in paper_fields
    }
    paper_success_runs_mean = {
        f: round(_mean([float(r.get(f, 0.0)) for r in success_rows]), 4) if success_rows else 0.0
        for f in paper_fields
    }

    numeric_keys = [
        "overall_score",
        "latency_sec",
        "tokens_total",
        "slide_count",
        "code_runtime_sec",
        "quiz_count",
        "video_size_mb",
        "Accpre",
        "Accpost",
        "Acccontrol",
        "ΔAcc",
        "ΔAccnet",
        "AlignSC",
        "Coverage",
        "OutOfScope",
        "Consistency",
        "Sync",
    ]

    stats: Dict[str, Dict[str, float]] = {}
    for key in numeric_keys:
        vals = [float(r.get(key, 0.0)) for r in rows]
        if vals:
            stats[key] = {
                "mean": round(_mean(vals), 4),
                "std": round(_std(vals), 4),
                "n": len(vals),
            }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path,
        "settings": settings,
        "total_requests": total,
        "completion_rate": round(completion_rate, 4),
        "partial_failure_rate": round(partial_failure_rate, 4),
        "silent_failure_rate": round(silent_failure_rate, 4),
        "success_rate": round(success_rate, 4),
        "latency_mean": round(latency_mean, 4),
        "tokens_total": tokens_total_sum,
        "tokens_by_agent": tokens_by_agent_total,
        "paper_all_runs_mean": paper_all_runs_mean,
        "paper_success_runs_mean": paper_success_runs_mean,
        "metrics": stats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_evaluation(config_path: str, out_dir: str, single_units: int, concurrent_batches: int, batch_size: int) -> None:
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    topics = cfg.get("topics", [])
    settings = cfg.get("settings", {})

    settings.setdefault("latency_budget_sec", LATENCY_BUDGET_SEC)
    settings.setdefault("silent_inactivity_sec", SILENT_FAILURE_INACTIVITY_SEC)
    settings.setdefault("code_timeout_sec", 30 * 60)
    settings.setdefault("pip_install_timeout_sec", 20 * 60)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    singles, batches = _build_request_plan(
        topics=topics,
        single_units=single_units,
        concurrent_batches=concurrent_batches,
        batch_size=batch_size,
    )

    logger.info(
        "Loaded %d topics. Plan: %d singles + %d batches x %d (total requests=%d)",
        len(topics),
        len(singles),
        len(batches),
        batch_size,
        len(singles) + len(batches) * batch_size,
    )

    rows: List[Dict[str, Any]] = []

    # Phase 1: single requests
    for spec in singles:
        run_dir = _row_run_dir(out_path, spec)
        logger.info(
            "[single u=%03d req=%04d] topic=%s",
            spec["unit_id"],
            spec["request_index"],
            spec["topic_cfg"].get("topic_id", ""),
        )
        row = evaluate_request(spec=spec, run_out_dir=str(run_dir), settings=settings)
        rows.append(row)
        logger.info(
            "  -> success=%s score=%.4f latency=%.2fs",
            row.get("success_flag"),
            float(row.get("overall_score", 0.0)),
            float(row.get("latency_sec", 0.0)),
        )

    # Phase 2: concurrent batches (target concurrency=2)
    for batch_specs in batches:
        b_id = batch_specs[0]["unit_id"] if batch_specs else -1
        logger.info("[batch u=%03d] launching %d concurrent requests", b_id, len(batch_specs))

        with ThreadPoolExecutor(max_workers=batch_size) as ex:
            future_map = {}
            for spec in batch_specs:
                run_dir = _row_run_dir(out_path, spec)
                fut = ex.submit(evaluate_request, spec, str(run_dir), settings)
                future_map[fut] = spec

            batch_rows: List[Dict[str, Any]] = []
            for fut in as_completed(future_map):
                spec = future_map[fut]
                try:
                    row = fut.result()
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Batch request failed req=%04d topic=%s error=%s",
                        spec.get("request_index"),
                        spec["topic_cfg"].get("topic_id", ""),
                        exc,
                    )
                    row = {
                        "topic_id": spec["topic_cfg"].get("topic_id", ""),
                        "topic_name": spec["topic_cfg"].get("topic_name", ""),
                        "request_index": spec["request_index"],
                        "unit_type": spec["unit_type"],
                        "unit_id": spec["unit_id"],
                        "batch_slot": spec["batch_slot"],
                        "workflow_completed": False,
                        "silent_failure": False,
                        "partial_failure": True,
                        "latency_sec": 0.0,
                        "tokens_total": 0,
                        "tokens_by_agent": _json_dumps({"coder": 0, "presentation": 0, "quiz": 0, "video": 0}),
                        "success_flag": False,
                        "overall_score": 0.0,
                        "hard_fail_reasons": _json_dumps(["runner_exception"]),
                    }
                batch_rows.append(row)

            batch_rows.sort(key=lambda x: int(x.get("request_index", 0)))
            rows.extend(batch_rows)

        logger.info("[batch u=%03d] done", b_id)

    best_rows = _select_best_rows(rows)

    run_csv = out_path / "run_level_results.csv"
    best_csv = out_path / "topic_best_results.csv"
    agg_json = out_path / "aggregate_summary.json"

    _write_csv(run_csv, rows)
    _write_csv(best_csv, best_rows)

    agg = _aggregate_summary(rows, settings=settings, config_path=config_path)
    agg_json.write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Wrote %s", run_csv)
    logger.info("Wrote %s", best_csv)
    logger.info("Wrote %s", agg_json)
    logger.info("Evaluation complete. total_rows=%d best_rows=%d", len(rows), len(best_rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full evaluation workflow (paper protocol).")
    parser.add_argument("--config", default="configs/eval_topics.json", help="Path to topic config JSON")
    parser.add_argument("--out", default="results", help="Output directory")
    parser.add_argument("--single-units", type=int, default=DEFAULT_SINGLE_UNITS, help="Number of single-request units")
    parser.add_argument(
        "--concurrent-batches",
        type=int,
        default=DEFAULT_CONCURRENT_BATCHES,
        help="Number of concurrent batches",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Requests per concurrent batch")

    args = parser.parse_args()

    run_evaluation(
        config_path=args.config,
        out_dir=args.out,
        single_units=args.single_units,
        concurrent_batches=args.concurrent_batches,
        batch_size=args.batch_size,
    )
