"""Token usage loader with schema and filename compatibility."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


_AGENT_KEYS = {
    "coder": "coder",
    "code": "coder",
    "presentation": "presentation",
    "slides": "presentation",
    "slide": "presentation",
    "quiz": "quiz",
    "quizzer": "quiz",
    "quizzes": "quiz",
    "video": "video",
}

_TRACKED_AGENTS = ("coder", "presentation", "quiz", "video")


def _safe_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:  # noqa: BLE001
        return 0


def _map_agent_name(name: str) -> Optional[str]:
    key = (name or "").strip().lower()
    return _AGENT_KEYS.get(key)


def _extract_total_tokens(node: Any) -> int:
    if node is None:
        return 0

    if isinstance(node, int):
        return node

    if isinstance(node, float):
        return int(node)

    if isinstance(node, dict):
        if "total_tokens" in node:
            return _safe_int(node.get("total_tokens"))
        if "total" in node and isinstance(node["total"], dict):
            return _safe_int(node["total"].get("total_tokens"))

        # Could be nested model map: {"gpt-4o": {"total_tokens": 123}}
        subtotal = 0
        has_nested = False
        for v in node.values():
            if isinstance(v, dict):
                has_nested = True
                subtotal += _extract_total_tokens(v)
        if has_nested:
            return subtotal

    return 0


def _normalize_agent_tokens(raw: Any) -> Dict[str, int]:
    out: Dict[str, int] = {k: 0 for k in _TRACKED_AGENTS}

    if not isinstance(raw, dict):
        return out

    for agent, value in raw.items():
        mapped = _map_agent_name(str(agent))
        if not mapped:
            continue
        out[mapped] += _extract_total_tokens(value)

    return out


def _merge_agent_tokens(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    out = {k: 0 for k in _TRACKED_AGENTS}
    for k in _TRACKED_AGENTS:
        out[k] = _safe_int(a.get(k, 0)) + _safe_int(b.get(k, 0))
    return out


def _parse_summary(summary_path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not parse %s: %s", summary_path, exc)
        return None

    total = 0
    by_agent = {k: 0 for k in _TRACKED_AGENTS}
    llm_call_count = _safe_int(data.get("llm_call_count", 0))

    # Schema 1: top-level total_tokens / tokens_by_agent
    if "total_tokens" in data:
        total = _safe_int(data.get("total_tokens", 0))
    if "tokens_by_agent" in data:
        by_agent = _merge_agent_tokens(by_agent, _normalize_agent_tokens(data.get("tokens_by_agent")))

    # Schema 2: nested tokens.total.total_tokens / tokens.by_agent
    tokens_node = data.get("tokens", {})
    if isinstance(tokens_node, dict):
        if total == 0:
            total = _extract_total_tokens(tokens_node)

        nested_by_agent = tokens_node.get("by_agent")
        if nested_by_agent is not None:
            by_agent = _merge_agent_tokens(by_agent, _normalize_agent_tokens(nested_by_agent))

    # Schema 3: agents map
    agents_node = data.get("agents")
    if isinstance(agents_node, dict):
        by_agent = _merge_agent_tokens(by_agent, _normalize_agent_tokens(agents_node))

    if total == 0:
        total = sum(by_agent.values())

    return {
        "tokens_total": total,
        "tokens_by_agent": by_agent,
        "llm_call_count": llm_call_count,
        "source": str(summary_path.name),
    }


def _parse_trace(trace_path: Path) -> Optional[Dict[str, Any]]:
    total = 0
    by_agent = {k: 0 for k in _TRACKED_AGENTS}
    call_count = 0

    try:
        with trace_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue

                usage_node = entry.get("usage") if isinstance(entry.get("usage"), dict) else {}
                tok = (
                    _safe_int(entry.get("total_tokens", 0))
                    or _safe_int(entry.get("tokens", 0))
                    or _safe_int(usage_node.get("total_tokens", 0))
                )

                agent_raw = entry.get("agent_name", entry.get("agent", ""))
                mapped = _map_agent_name(str(agent_raw))
                if mapped:
                    by_agent[mapped] += tok

                total += tok

                event = str(entry.get("event", "")).upper()
                if event == "LLM_CALL" or tok > 0:
                    call_count += 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not aggregate %s: %s", trace_path, exc)
        return None

    return {
        "tokens_total": total,
        "tokens_by_agent": by_agent,
        "llm_call_count": call_count,
        "source": str(trace_path.name),
    }


def load_run_summary(run_dir: str) -> Optional[Dict[str, Any]]:
    """Load token summary from a run directory with compatibility fallbacks.

    Supported summary files:
    - summary.json

    Supported trace files:
    - trace_log.jsonl
    - trace.jsonl
    """
    rd = Path(run_dir)
    if not rd.exists() or not rd.is_dir():
        return None

    summary_path = rd / "summary.json"
    if summary_path.exists():
        parsed = _parse_summary(summary_path)
        if parsed is not None:
            return parsed

    for trace_name in ("trace_log.jsonl", "trace.jsonl"):
        trace_path = rd / trace_name
        if trace_path.exists():
            parsed = _parse_trace(trace_path)
            if parsed is not None:
                return parsed

    return None
