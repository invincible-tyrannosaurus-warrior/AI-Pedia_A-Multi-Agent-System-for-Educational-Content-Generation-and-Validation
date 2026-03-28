"""
Judger Agent Pipeline

This module validates agent outputs against structured acceptance criteria.
It uses MCP tools for deterministic checks and an LLM for semantic evaluation
and corrective guidance.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from ai_pedia_mcp_server.client import MCPClientSync

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("JUDGER_MODEL", "gpt-5.2")
MAX_TEXT_LEN = int(os.getenv("JUDGER_MAX_TEXT_LEN", "2000"))


def _default_client() -> OpenAI:
    try:
        return OpenAI()
    except Exception as exc:
        logger.error("Failed to instantiate OpenAI client: %s", exc)
        raise


def _truncate_strings(value: Any, max_len: int = MAX_TEXT_LEN) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        if len(value) <= max_len:
            return value
        return value[: max_len - 3] + "..."
    if isinstance(value, list):
        return [_truncate_strings(item, max_len=max_len) for item in value]
    if isinstance(value, dict):
        return {k: _truncate_strings(v, max_len=max_len) for k, v in value.items()}
    return value


_INDEX_RE = re.compile(r"(.+)\[(\d+)\]$")
_PLACEHOLDER_RE = re.compile(r"\$([a-zA-Z0-9_.\[\]]+)")


def _resolve_path(data: Any, path: str) -> Any:
    if not path:
        return None

    current = data
    for part in path.split("."):
        if current is None:
            return None

        match = _INDEX_RE.match(part)
        if match:
            key = match.group(1)
            idx = int(match.group(2))
            if not isinstance(current, dict):
                return None
            current = current.get(key)
            if not isinstance(current, list) or idx >= len(current):
                return None
            current = current[idx]
            continue

        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None

    return current


def _normalize_target(target: str) -> str:
    if target.startswith("result.") or target.startswith("tools."):
        return target
    return f"result.{target}"


def _resolve_target(target: str, result: Dict[str, Any], tools: Dict[str, Any]) -> Any:
    normalized = _normalize_target(target)
    root = {"result": result, "tools": tools}
    return _resolve_path(root, normalized)


def _resolve_placeholders(value: Any, result: Dict[str, Any]) -> Any:
    if not isinstance(value, str):
        return value

    def _replace(match: re.Match) -> str:
        path = match.group(1)
        resolved = _resolve_path(result, path)
        return str(resolved) if resolved is not None else match.group(0)

    return _PLACEHOLDER_RE.sub(_replace, value)


def _apply_operator(value: Any, operator: str, expected: Any) -> bool:
    if operator == "exists":
        return value is not None
    if operator == "equals":
        return value == expected
    if operator == "contains":
        if isinstance(value, Path):
            return expected in str(value)
        if isinstance(value, str):
            return expected in value
        if isinstance(value, list):
            return expected in value
        if isinstance(value, dict):
            return expected in value.keys()
        return False
    if operator == "gte":
        try:
            return value >= expected
        except Exception:
            return False
    if operator == "lte":
        try:
            return value <= expected
        except Exception:
            return False
    return False


def _call_tool(
    tool_client: MCPClientSync,
    tool_name: str,
    tool_args: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        return tool_client.call_tool(tool_name, **tool_args)
    except Exception as exc:
        logger.error("Tool call failed (%s): %s", tool_name, exc)
        return {"success": False, "error": str(exc)}


def _evaluate_nonsemantic_criterion(
    criterion: Dict[str, Any],
    result: Dict[str, Any],
    tools_context: Dict[str, Any],
    tool_client: MCPClientSync,
    tool_cache: Dict[str, Dict[str, Any]],
) -> tuple[bool, Dict[str, Any]]:
    c_type = criterion.get("criterion_type") or criterion.get("type")
    target = criterion.get("target", "")
    operator = criterion.get("operator", "exists")
    expected = criterion.get("expected")

    if c_type == "file_exists":
        value = _resolve_target(target, result, tools_context)
        path = Path(str(value)) if value else None
        exists = bool(path and path.exists())
        return exists, {"value": str(path) if path else None, "exists": exists}

    if c_type == "output_shape":
        value = _resolve_target(target, result, tools_context)
        passed = _apply_operator(value, operator, expected)
        return passed, {"value": value, "expected": expected, "operator": operator}

    if c_type == "mcp_tool":
        tool_def = criterion.get("tool") or {}
        tool_name = tool_def.get("name")
        tool_args = tool_def.get("args", {})
        tool_args = {k: _resolve_placeholders(v, result) for k, v in tool_args.items()}

        if tool_name == "python_check" and "filepath" not in tool_args:
            fallback = _resolve_path(result, "artifacts[0]") or _resolve_path(
                result, "output.validation.path"
            )
            if fallback:
                tool_args["filepath"] = str(fallback)

        cache_key = json.dumps([tool_name, tool_args], sort_keys=True)
        if cache_key in tool_cache:
            tool_result = tool_cache[cache_key]
        else:
            tool_result = _call_tool(tool_client, tool_name, tool_args) if tool_name else {"success": False}
            tool_cache[cache_key] = tool_result

        if tool_name:
            tools_context[tool_name] = tool_result

        value = _resolve_target(target, result, tools_context)
        passed = _apply_operator(value, operator, expected)
        return passed, {
            "tool": tool_name,
            "args": tool_args,
            "result": tool_result,
            "value": value,
            "expected": expected,
            "operator": operator,
        }

    return False, {"error": f"Unknown criterion type: {c_type}"}


def _judge_with_llm(
    task_id: str,
    agent: str,
    instruction: str,
    output: Dict[str, Any],
    semantic_criteria: List[Dict[str, Any]],
    deterministic_failures: List[str],
    deterministic_evidence: Dict[str, Any],
    client: OpenAI,
    log_capture: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    system_prompt = (
        "You are a strict but fair evaluator. Assessment Guide:\n"
        "1. Deterministic Failures: If 'deterministic_failures' includes missing files (file_exists) or empty output, verdict MUST be 'fail'. "
        "However, if the failure is minor (e.g., filename mismatch but valid file exists), you may exercise discretion and PASS with a warning in 'fix_instructions'.\n"
        "2. Semantic Criteria: Evaluate each semantic criterion thoroughly.\n"
        "3. Verdict: Return 'pass' if the core requirements are met. Return 'fail' only if the output is unusable or critically flawed.\n"
        "4. Fix Instructions: If verdict is 'fail', provide clear instructions on how to fix the issue."
    )

    payload = {
        "task_id": task_id,
        "agent": agent,
        "instruction": instruction,
        "output": _truncate_strings(output),
        "semantic_criteria": semantic_criteria,
        "deterministic_failures": deterministic_failures,
        "deterministic_evidence": _truncate_strings(deterministic_evidence),
    }

    tool_schema = {
        "type": "function",
        "function": {
            "name": "judge_task",
            "description": "Return verdict and fixes for a single task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "verdict": {"type": "string", "enum": ["pass", "fail"]},
                    "failed_criteria": {"type": "array", "items": {"type": "string"}},
                    "evidence": {"type": "object"},
                    "fix_instructions": {"type": "string"},
                },
                "required": ["verdict", "failed_criteria", "evidence", "fix_instructions"],
            },
        },
    }

    if log_capture is not None:
        log_capture["llm_request"] = {
            "model": MODEL_NAME,
            "system_prompt": system_prompt,
            "payload": payload,
            "tool_schema": tool_schema,
        }

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True, default=str)},
        ],
        tools=[tool_schema],
        tool_choice={"type": "function", "function": {"name": "judge_task"}},
    )

    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        raise RuntimeError("Judger failed to return a tool call.")

    args_json = tool_calls[0].function.arguments
    if log_capture is not None:
        log_capture["llm_response"] = {
            "tool_calls": [
                {
                    "name": tool_calls[0].function.name,
                    "arguments": args_json,
                }
            ]
        }
    return json.loads(args_json)


def run_judger_pipeline(
    *,
    plan: Dict[str, Any],
    agent_results: Dict[str, Dict[str, Any]],
    assets: Optional[List[Any]] = None,
    client: Optional[OpenAI] = None,
    log_capture: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Validate agent outputs against acceptance criteria and return judger verdicts.
    """

    del assets  # not used for now, reserved for future enhancements

    tasks = plan.get("subtasks", [])

    verdicts: List[Dict[str, Any]] = []

    if log_capture is not None:
        log_capture["request"] = {
            "plan": plan,
            "agent_results": agent_results,
        }
        log_capture["trace"] = {"tasks": []}

    with MCPClientSync() as tool_client:
        for task in tasks:
            task_id = task.get("task_id") or task.get("id")
            agent = task.get("agent")
            instruction = task.get("instruction", "")
            criteria = task.get("acceptance_criteria") or task.get("criteria") or []

            result = agent_results.get(task_id, {})
            output = result.get("output", {}) if isinstance(result, dict) else {}
            task_trace: Dict[str, Any] = {
                "task_id": task_id,
                "agent": agent,
                "instruction": instruction,
                "criteria": criteria,
            }

            criteria_by_id = {}
            required_ids = set()
            for criterion in criteria:
                cid = criterion.get("criterion_id") or criterion.get("id")
                if not cid:
                    continue
                criteria_by_id[cid] = criterion
                if (criterion.get("severity") or "required") == "required":
                    required_ids.add(cid)

            tools_context: Dict[str, Any] = {}
            tool_cache: Dict[str, Dict[str, Any]] = {}
            deterministic_failures: List[str] = []
            deterministic_evidence: Dict[str, Any] = {}
            semantic_criteria: List[Dict[str, Any]] = []

            for criterion in criteria:
                cid = criterion.get("criterion_id") or criterion.get("id")
                c_type = criterion.get("criterion_type") or criterion.get("type")
                severity = criterion.get("severity") or "required"

                if c_type == "semantic":
                    semantic_criteria.append(criterion)
                    continue

                passed, evidence = _evaluate_nonsemantic_criterion(
                    criterion, result, tools_context, tool_client, tool_cache
                )
                deterministic_evidence[cid or "unknown"] = evidence

                if not passed and severity == "required" and cid:
                    deterministic_failures.append(cid)

            llm_failed: List[str] = []
            llm_evidence: Dict[str, Any] = {}
            fix_instructions = ""
            llm_trace: Dict[str, Any] = {}

            needs_llm = bool(semantic_criteria)
            if needs_llm:
                try:
                    eval_client = client or _default_client()
                    llm_result = _judge_with_llm(
                        task_id=task_id,
                        agent=agent,
                        instruction=instruction,
                        output=output,
                        semantic_criteria=semantic_criteria,
                        deterministic_failures=deterministic_failures,
                        deterministic_evidence=deterministic_evidence,
                        client=eval_client,
                        log_capture=llm_trace,
                    )
                    llm_failed = llm_result.get("failed_criteria", [])
                    llm_evidence = llm_result.get("evidence", {})
                    fix_instructions = llm_result.get("fix_instructions", "")
                except Exception as exc:
                    logger.error("Judger LLM call failed for %s: %s", task_id, exc)
                    llm_failed = []
                    llm_evidence = {"error": str(exc)}

            all_failed = set(deterministic_failures)
            for cid in llm_failed:
                if cid in criteria_by_id and cid in required_ids:
                    all_failed.add(cid)

            verdict = "fail" if all_failed else "pass"
            if verdict == "pass":
                fix_instructions = ""
            elif verdict == "fail" and not fix_instructions:
                fix_instructions = instruction

            evidence = {"deterministic": deterministic_evidence, "semantic": llm_evidence}

            task_trace["deterministic"] = deterministic_evidence
            task_trace["llm"] = llm_trace
            task_trace["verdict"] = verdict

            verdicts.append(
                {
                    "task_id": task_id,
                    "verdict": verdict,
                    "failed_criteria": sorted(all_failed),
                    "evidence": evidence,
                    "fix_instructions": fix_instructions,
                }
            )

            if log_capture is not None:
                log_capture["trace"]["tasks"].append(task_trace)

    passes = [v for v in verdicts if v["verdict"] == "pass"]
    fails = [v for v in verdicts if v["verdict"] == "fail"]

    if passes and not fails:
        overall_status = "pass"
    elif passes and fails:
        overall_status = "partial"
    else:
        overall_status = "fail"

    return {
        "overall_status": overall_status,
        "tasks": verdicts,
    }


__all__ = ["run_judger_pipeline"]
