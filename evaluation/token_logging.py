"""
evaluation/token_logging.py

Reads per-run token-usage summaries produced by the observability module
(trace_log.jsonl / summary.json).  Provides a uniform interface for the
evaluation runner.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def load_run_summary(run_dir: str) -> Optional[Dict[str, Any]]:
    """Load token / LLM-call statistics from a run directory.

    The function looks for the following files (in order):
      1. ``<run_dir>/summary.json``  – written by the observability module
      2. ``<run_dir>/trace_log.jsonl`` – raw per-call log; aggregated here

    Returns
    -------
    dict or None
        ``{"total_tokens": int, "tokens_by_agent": dict, "llm_call_count": int}``
        or *None* if no log files are found.
    """
    rd = Path(run_dir)

    # --- Try summary.json first -------------------------------------------
    summary_path = rd / "summary.json"
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            return _normalise(data)
        except Exception as exc:
            logger.warning("Could not parse %s: %s", summary_path, exc)

    # --- Fallback: aggregate trace_log.jsonl ------------------------------
    trace_path = rd / "trace_log.jsonl"
    if trace_path.exists():
        try:
            return _aggregate_trace(trace_path)
        except Exception as exc:
            logger.warning("Could not aggregate %s: %s", trace_path, exc)

    # Nothing found
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure the summary dict contains the expected keys."""
    total = data.get("total_tokens", 0)
    by_agent: Dict[str, int] = {}
    call_count = 0

    # tokens_by_agent might be nested inside "agents" or directly  
    agents_raw = data.get("tokens_by_agent", data.get("agents", {}))
    if isinstance(agents_raw, dict):
        for agent, info in agents_raw.items():
            if isinstance(info, int):
                by_agent[agent] = info
            elif isinstance(info, dict):
                by_agent[agent] = info.get("total_tokens", 0)
                call_count += info.get("call_count", 0)
            else:
                by_agent[agent] = 0

    if total == 0 and by_agent:
        total = sum(by_agent.values())

    if call_count == 0:
        call_count = data.get("llm_call_count", 0)

    return {
        "total_tokens": total,
        "tokens_by_agent": by_agent,
        "llm_call_count": call_count,
    }


def _aggregate_trace(trace_path: Path) -> Dict[str, Any]:
    """Build a summary from per-call JSONL entries."""
    by_agent: Dict[str, int] = {}
    total = 0
    call_count = 0

    with open(trace_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            tokens = entry.get("total_tokens", entry.get("tokens", 0))
            agent = entry.get("agent_name", entry.get("agent", "unknown"))
            total += tokens
            by_agent[agent] = by_agent.get(agent, 0) + tokens
            call_count += 1

    return {
        "total_tokens": total,
        "tokens_by_agent": by_agent,
        "llm_call_count": call_count,
    }
