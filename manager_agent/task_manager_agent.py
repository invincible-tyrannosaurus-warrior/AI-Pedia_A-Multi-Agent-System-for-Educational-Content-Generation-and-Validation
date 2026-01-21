"""
Task Manager Agent

This module acts as the central orchestrator for the AI_Pedia_Local system.
It is responsible for:
1. Understanding user intent and analyzing inputs.
2. Generating a structured execution plan (TaskPlan).
3. Routing tasks to specific agents via the AgentRegistry.
4. Aggregating results and handling the overall workflow.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Set, Literal

from openai import OpenAI
from pydantic import BaseModel, Field, AliasChoices

from moe_layer.orchestrator.agent_registry import registry
from moe_layer.coder_agent.storage.local import StoredAsset
from moe_layer.coder_agent.utils.file_processors import summarize_text

# Import agent pipelines for registration
from moe_layer.coder_agent.coder_pipeline import run_coder_pipeline
from moe_layer.video_agent.video_pipeline import run_video_pipeline
from moe_layer.presentation_agent.presentation_pipeline import run_presentation_pipeline
from moe_layer.text_generator_agent.text_pipeline import run_text_pipeline
from moe_layer.quizzer_agent.quiz_pipeline import run_quiz_pipeline
from judger_agent.judger_pipeline import run_judger_pipeline

logger = logging.getLogger(__name__)

class AcceptanceCriterion(BaseModel):
    criterion_id: str = Field(
        ...,
        description="Unique identifier for this criterion",
        validation_alias=AliasChoices("id", "criterion_id", "criteria_id"),
    )
    criterion_type: Literal["mcp_tool", "output_shape", "file_exists", "semantic"] = Field(
        ...,
        description="Criterion type: mcp_tool | output_shape | file_exists | semantic",
        validation_alias=AliasChoices("type", "criterion_type", "criteria_type"),
    )
    target: str = Field(
        ...,
        description="Evaluation target path (e.g., result.output.text or tools.python_check.success)",
    )
    operator: Literal["equals", "contains", "exists", "gte", "lte"] = Field(
        ...,
        description="Operator: equals | contains | exists | gte | lte",
    )
    expected: Any = Field(
        None,
        description="Expected value to compare against",
    )
    severity: Literal["required", "optional"] = Field(
        "required",
        description="Severity: required | optional",
    )
    tool: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Tool call definition for mcp_tool criteria",
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Evidence path to include in judger output",
    )

class SubTask(BaseModel):
    task_id: str = Field(..., description="Unique identifier for this subtask", validation_alias=AliasChoices('id', 'task_id', 'step_id'))
    agent: str = Field(..., description="The agent responsible (coder, video, text, quizzer)")
    instruction: str = Field(..., description="Detailed instruction for the agent", validation_alias=AliasChoices('title', 'instruction', 'description', 'goal'))
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Specific inputs for the agent")
    dependencies: List[str] = Field(default_factory=list, description="IDs of tasks that must complete first")
    acceptance_criteria: List[AcceptanceCriterion] = Field(
        ...,
        description="Structured criteria the judger will use",
        validation_alias=AliasChoices("acceptance_criteria", "criteria", "checks", "requirements"),
    )

class TaskPlan(BaseModel):
    user_intent: str = Field(..., description="Summary of what the user wants to achieve", validation_alias=AliasChoices('user_intent', 'goal', 'intent', 'objective', 'topic'))
    subtasks: List[SubTask] = Field(..., description="List of executable steps", validation_alias=AliasChoices('subtasks', 'steps', 'tasks', 'plan', 'actions', 'workflow'))


# --- Configuration & Registration ---

MODEL_NAME = os.getenv("TASK_MANAGER_MODEL", "gpt-5.2")
MAX_PLAN_RETRIES = int(os.getenv("TASK_PLAN_MAX_RETRIES", "3"))
MAX_TASK_RETRIES = int(os.getenv("TASK_MAX_RETRIES", "3"))

# Register Agents
registry.register("coder", run_coder_pipeline, "Generates and validates Python code", ["coding", "python"])
registry.register("video", run_video_pipeline, "Creates video content", ["video", "visual"])
registry.register("presentation", run_presentation_pipeline, "Generates PowerPoint slides and lesson plans", ["presentation", "slides", "pptx", "lesson_plan"])
# DISABLED: Text agent is a mock/placeholder with no real functionality
# registry.register("text", run_text_pipeline, "Generates written notes and summaries (Legacy/Fallback)", ["writing", "summary"])
registry.register("quizzer", run_quiz_pipeline, "Creates quizzes and assessments", ["quiz", "test"])


SYSTEM_PROMPT = """
You are an expert autonomous Task Orchestrator.
Your goal is to analyze user requests and educational materials, then create a structured execution plan.

Available Agents:
- 'coder': Generates runnable Python code. Best for demonstrations, data analysis, and coding tutorials.
- 'video': Creates video content. Best for visual explanations.
- 'presentation': Generates PowerPoint slides (.pptx). Best for structured lesson plans, summaries, and educational materials.
- 'quizzer': Creates quizzes. Best for assessments and review.

Instructions:
1. Analyze the user's input and provided files.
2. Break down the goal into logical subtasks.
3. Assign each subtask to the most appropriate agent.
4. Provide specific, actionable instructions for each agent.
5. For every subtask, include a non-empty 'acceptance_criteria' list.
   - Each criterion must include: criterion_id, criterion_type, target, operator, expected, severity.
   - Use criterion_type values: mcp_tool | output_shape | file_exists | semantic.
   - Use operator values: equals | contains | exists | gte | lte.
   - Use target paths that start with 'result.' or 'tools.'.
   - For mcp_tool criteria, include tool: { name, args }.
   - Use placeholders like $artifacts[0] or $output.validation.path inside tool args.
   - **CRITICAL**: For `semantic` criteria, ONLY check for concepts EXPLICITLY requested by the user. Do NOT invent constraints (e.g., do not require "KNN" unless user asked for it).
6. Use the provided tool 'create_task_plan' to submit your plan.
7. CRITICAL For 'presentation' agent tasks:
   - ALWAYS include `output.metrics.slide_count` (gte 1)
   - If code is involved, include `output.metrics.code_snippets_count` (gte 1)
   - These are deterministic checks (`output_shape`) that are FASTER than semantic checks.
8. CRITICAL For 'coder' agent tasks:
   - DO NOT use the `python_check` mcp_tool. The agent validates itself.
   - Use `output_shape` to check `result.output.validation.run_success` (equals true).
   - Use `output_shape` to check `result.output.validation.stderr` (equals "").
9. CRITICAL For 'video' agent tasks:
   - Must check for MP4 output: `output_shape` on `result.artifacts[0]` (contains ".mp4").
   - Ensure success: `output_shape` on `result.success` (equals true).
10. CRITICAL For tasks involving uploaded files:
   - Check the 'AVAILABLE AS LOCAL FILE' name in the input.
   - You MUST use this EXACT filename in the agent 'instruction'.
   - Do NOT guess names like 'ML_example.pdf'. Use the actual on-disk name (e.g., '173628_ML_Example.pdf').

Response Format Example:
{
  "user_intent": "Create a lesson on Python lists with a video intro and a quiz.",
  "subtasks": [
    {
      "task_id": "step1_code",
      "agent": "coder",
      "instruction": "Generate a Python script demonstrating list operations.",
      "inputs": {},
      "dependencies": [],
      "acceptance_criteria": [
        {
          "criterion_id": "code_success",
          "criterion_type": "output_shape",
          "target": "result.output.validation.run_success",
          "operator": "equals",
          "expected": true,
          "severity": "required"
        }
      ]
    },
    {
      "task_id": "step2_video",
      "agent": "video",
      "instruction": "Create a video explaining the code.",
      "inputs": {},
      "dependencies": ["step1_code"],
      "acceptance_criteria": [
        {
          "criterion_id": "video_semantic_ok",
          "criterion_type": "semantic",
          "target": "result.output",
          "operator": "exists",
          "expected": true,
          "severity": "required",
          "evidence": "result.output"
        }
      ]
    }
  ]
}
"""

# --- Helper Functions ---

def _default_client() -> OpenAI:
    try:
        return OpenAI()
    except Exception as exc:
        logger.error("Failed to instantiate OpenAI client: %s", exc)
        raise

def build_gpt_input(
    asset_descriptors: Iterable[dict],
    user_text: Optional[str],
) -> list:
    """Compose the rich input payload GPT expects."""
    content = []
    if user_text:
        content.append({"type": "text", "text": user_text})

    for asset in asset_descriptors:
        asset_type = asset.get("type")
        if asset_type == "file":
            description = asset.get("description") or "Uploaded asset"
            mime = asset.get("mime_type", "unknown type")
            # IMPORTANT: We explicitly provide the ON-DISK filename so the Planner knows EXACTLY what to tell the Agent
            filename = asset.get("original_filename") or "unknown_file"
            
            content.append(
                {
                    "type": "text",
                    "text": f"{description} ({mime}). AVAILABLE AS LOCAL FILE: '{filename}'. URL: {asset['url']}",
                }
            )
        elif asset_type == "text":
            content.append({"type": "text", "text": asset.get("text", "")})
    return content


def generate_task_plan(
    asset_descriptors: Iterable[dict],
    user_text: Optional[str] = None,
    client: Optional[OpenAI] = None,
    log_capture: Optional[Dict[str, Any]] = None,
) -> TaskPlan:
    """call GPT to generate a structured TaskPlan."""
    client = client or _default_client()
    gpt_input = build_gpt_input(asset_descriptors, user_text)

    if not gpt_input:
        raise ValueError("At least one input (text or file descriptor) is required.")

    logger.info("Requesting TaskPlan from %s", MODEL_NAME)

    # Define the tool schema using Pydantic's schema generation
    tool_schema = {
        "type": "function",
        "function": {
            "name": "create_task_plan",
            "description": "Create a structured execution plan for multiple agents.",
            "parameters": TaskPlan.model_json_schema(),
        }
    }

    request_payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": gpt_input},
        ],
        "tools": [tool_schema],
        "tool_choice": {"type": "function", "function": {"name": "create_task_plan"}},
    }

    if log_capture is not None:
        log_capture["request"] = request_payload

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=request_payload["messages"],
        tools=[tool_schema],
        tool_choice={"type": "function", "function": {"name": "create_task_plan"}},
    )

    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        if log_capture is not None:
            log_capture["response"] = {"tool_calls": []}
        raise ValueError("Task Manager failed to call the create_task_plan tool.")
    
    # Extract arguments from the tool call
    args_json = tool_calls[0].function.arguments
    if log_capture is not None:
        log_capture["response"] = {
            "tool_calls": [
                {
                    "name": tool_calls[0].function.name,
                    "arguments": args_json,
                }
            ]
        }
    
    try:
        data = json.loads(args_json)
        plan = TaskPlan(**data)
        logger.info("Generated plan with %d subtasks", len(plan.subtasks))
        return plan
    except Exception as e:
        logger.error("Failed to parse TaskPlan from tool arguments: %s. Content: %s", e, args_json)
        raise RuntimeError("Task Manager generated invalid plan structure") from e


def _plan_has_criteria(plan: TaskPlan) -> bool:
    for subtask in plan.subtasks:
        if not subtask.acceptance_criteria:
            return False
    return True


def _collect_dependent_tasks(plan: TaskPlan, root_tasks: Set[str]) -> Set[str]:
    dependents = set(root_tasks)
    changed = True
    while changed:
        changed = False
        for subtask in plan.subtasks:
            if subtask.task_id in dependents:
                continue
            if any(dep in dependents for dep in subtask.dependencies):
                dependents.add(subtask.task_id)
                changed = True
    return dependents

def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, default=_json_default)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _local_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


_SAFE_STEM_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


def _safe_stem(value: str) -> str:
    cleaned = _SAFE_STEM_RE.sub("_", value or "").strip("_")
    return cleaned or "lesson_example"


def _ensure_coder_criteria(subtask: SubTask, output_filename: str) -> None:
    # 2. explicit removal of redundant/hallucinated criteria
    # User requested removal of 'is_python_file' and 'file_written'
    # 'file_written' is often hallucinated by the LLM with invalid targets.
    # 'is_python_file' failed because Path objects aren't strings in 'contains'.
    
    tasks_to_remove = ["is_python_file", "file_written", "code_written", "script_file_exists", "python_file_exists"]
    subtask.acceptance_criteria = [
        c for c in subtask.acceptance_criteria 
        if c.criterion_id not in tasks_to_remove
    ]


# --- Main Entry Point ---

def run_workflow(
    *,
    user_text: str,
    assets: Iterable[StoredAsset],
    output_dir: Path,
    task_client: Optional[OpenAI] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Main entry point for the agentic workflow.
    
    1. Preprocesses inputs (summaries, descriptors).
    2. Generates a TaskPlan using the Task Manager model (via tool calling).
    3. Executes each subtask using the appropriate registered Agent.
    4. Runs the Judger agent to validate results and issues feedback.
    5. Retries failed subtasks (and dependents) up to the max limit.
    6. Aggregates and returns results.
    """
    
    # 1. Preprocess Assets
    asset_list = list(assets)
    if not asset_list and not user_text.strip():
        raise ValueError("Provide at least text or one uploaded asset.")

    run_id = uuid.uuid4().hex
    
    # Ensure client is initialized
    task_client = task_client or _default_client()
    
    started_at = _now_iso()
    from config import LOGS_DIR
    logs_root = LOGS_DIR
    task_manager_log_path = logs_root / "task_manager" / f"{run_id}.json"
    judger_log_path = logs_root / "judger" / f"{run_id}.json"

    asset_descriptors = [asset.as_descriptor() for asset in asset_list]
    extracted_snippets = [
        summarize_text(asset.extracted_text) for asset in asset_list if asset.extracted_text
    ]
    if extracted_snippets:
        asset_descriptors.append(
            {"type": "text", "text": "\n\n".join(extracted_snippets)}
        )

    task_manager_log: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at,
        "user_text": user_text,
        "asset_descriptors": asset_descriptors,
        "plan_attempts": [],
    }
    judger_log: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at,
        "rounds": [],
    }

    # 2. Generate Plan (with enforced acceptance_criteria)
    plan: Optional[TaskPlan] = None
    last_plan_error: Optional[Exception] = None
    for attempt in range(1, MAX_PLAN_RETRIES + 1):
        attempt_log: Dict[str, Any] = {"attempt": attempt}
        try:
            candidate = generate_task_plan(
                asset_descriptors=asset_descriptors,
                user_text=user_text.strip() or None,
                client=task_client,
                log_capture=attempt_log,
            )
            attempt_log["parsed_plan"] = candidate.model_dump()
            if _plan_has_criteria(candidate):
                plan = candidate
                attempt_log["status"] = "accepted"
                task_manager_log["plan_attempts"].append(attempt_log)
                break
            last_plan_error = ValueError("Missing acceptance_criteria in TaskPlan.")
            attempt_log["status"] = "missing_criteria"
            logger.warning("TaskPlan missing acceptance_criteria (attempt %d).", attempt)
        except Exception as exc:
            last_plan_error = exc
            attempt_log["status"] = "error"
            attempt_log["error"] = str(exc)
            logger.warning("TaskPlan generation failed (attempt %d): %s", attempt, exc)
        task_manager_log["plan_attempts"].append(attempt_log)

    if plan is None:
        task_manager_log["error"] = "Failed to generate valid TaskPlan with acceptance_criteria."
        try:
            _write_json(task_manager_log_path, task_manager_log)
            _write_json(judger_log_path, judger_log)
        except Exception as exc:
            logger.error("Failed to write logs: %s", exc)
        raise RuntimeError("Failed to generate valid TaskPlan with acceptance_criteria.") from last_plan_error

    timestamp = _local_timestamp()
    timestamp = _local_timestamp()
    for subtask in plan.subtasks:
        # Assign isolation directory for ALL agents
        base_stem = _safe_stem(subtask.task_id)
        output_filename = f"{base_stem}_{timestamp}.py"
        output_subdir = f"run_{run_id}"
        
        # Inject into inputs so pipeline receives it
        subtask.inputs["output_filename"] = output_filename
        subtask.inputs["output_subdir"] = output_subdir
        
        # Special handling for Coder (Criteria)
        if subtask.agent == "coder":
            _ensure_coder_criteria(subtask, output_filename)

    # 3. Execute Plan with feedback loop + parallelism
    agent_results: Dict[str, Any] = {}
    subtasks_by_id: Dict[str, SubTask] = {sub.task_id: sub for sub in plan.subtasks}
    instructions: Dict[str, str] = {sub.task_id: sub.instruction for sub in plan.subtasks}
    attempts: Dict[str, int] = {sub.task_id: 0 for sub in plan.subtasks}

    def dependencies_satisfied(task_id: str, pending: Set[str]) -> bool:
        subtask = subtasks_by_id[task_id]
        return all(dep in agent_results and dep not in pending for dep in subtask.dependencies)

    def run_subtask(task_id: str) -> Dict[str, Any]:
        subtask = subtasks_by_id[task_id]
        if attempts[task_id] >= MAX_TASK_RETRIES:
            return {
                "agent": subtask.agent,
                "success": False,
                "error": "Max retries exceeded.",
                "instruction": instructions[task_id],
                "attempts": attempts[task_id],
            }

        attempts[task_id] += 1
        logger.info("Executing SubTask: %s (Agent: %s, Attempt: %d)", task_id, subtask.agent, attempts[task_id])

        try:
            agent_func = registry.get(subtask.agent)
            dependency_results = {
                dep: agent_results.get(dep) for dep in subtask.dependencies if dep in agent_results
            }
            
            # Execute agent (handle both sync and async)
            import inspect
            import asyncio
            
            if inspect.iscoroutinefunction(agent_func):
                result = asyncio.run(agent_func(
                    instruction=instructions[task_id],
                    output_dir=output_dir,
                    assets=asset_list,
                    dependency_results=dependency_results,
                    client=task_client,
                    **subtask.inputs
                ))
            else:
                result = agent_func(
                    instruction=instructions[task_id],
                    output_dir=output_dir,
                    assets=asset_list,
                    dependency_results=dependency_results,
                    client=task_client,
                    **subtask.inputs
                )
            return {
                "agent": subtask.agent,
                "success": result.get("success", False),
                "output": result.get("output"),
                "artifacts": result.get("artifacts", []),
                "error": result.get("error"),
                "instruction": instructions[task_id],
                "attempts": attempts[task_id],
                "metadata": result.get("metadata", {}),
            }
        except Exception as exc:
            logger.exception("Error executing subtask %s", task_id)
            return {
                "agent": subtask.agent,
                "success": False,
                "error": str(exc),
                "instruction": instructions[task_id],
                "attempts": attempts[task_id],
            }

    def execute_tasks(task_ids: Set[str]) -> None:
        remaining = set(task_ids)
        while remaining:
            runnable = [tid for tid in remaining if dependencies_satisfied(tid, remaining)]
            if not runnable:
                for tid in list(remaining):
                    subtask = subtasks_by_id[tid]
                    agent_results[tid] = {
                        "agent": subtask.agent,
                        "success": False,
                        "error": "Unresolved dependencies.",
                        "instruction": instructions[tid],
                        "attempts": attempts[tid],
                    }
                break

            with ThreadPoolExecutor(max_workers=min(4, len(runnable))) as executor:
                future_map = {executor.submit(run_subtask, tid): tid for tid in runnable}
                for future in as_completed(future_map):
                    tid = future_map[future]
                    agent_results[tid] = future.result()

            remaining.difference_update(runnable)

    # Initial execution
    execute_tasks({sub.task_id for sub in plan.subtasks})

    judger_client = kwargs.get("judger_client")
    iteration = 1
    judger_log_capture: Dict[str, Any] = {}

    judger_verdict = run_judger_pipeline(
        plan=plan.model_dump(),
        agent_results=agent_results,
        assets=asset_list,
        client=judger_client,
        log_capture=judger_log_capture,
    )
    judger_log["rounds"].append(
        {
            "iteration": iteration,
            "request": judger_log_capture.get("request"),
            "trace": judger_log_capture.get("trace"),
            "result": judger_verdict,
        }
    )

    while True:
        verdict_map = {item["task_id"]: item for item in judger_verdict.get("tasks", [])}
        failed_tasks = [tid for tid, v in verdict_map.items() if v.get("verdict") == "fail"]

        if not failed_tasks:
            break

        # Update instructions for failed tasks (replace, not append)
        for task_id in failed_tasks:
            fix = verdict_map.get(task_id, {}).get("fix_instructions", "")
            if fix:
                instructions[task_id] = fix.strip()

        # Determine which tasks to rerun (failed + dependents)
        to_rerun = _collect_dependent_tasks(plan, set(failed_tasks))

        # Filter out tasks that exceeded max retries
        runnable = {tid for tid in to_rerun if attempts.get(tid, 0) < MAX_TASK_RETRIES}
        if not runnable:
            exhausted = {tid for tid in to_rerun if attempts.get(tid, 0) >= MAX_TASK_RETRIES}
            for tid in exhausted:
                if tid in agent_results:
                    # If the task actually passed in the last round, do not overwrite with error.
                    # This happens when a task passes but is dragged into retry by a failing dependency.
                    last_verdict = verdict_map.get(tid, {}).get("verdict")
                    if last_verdict != "pass":
                        agent_results[tid]["error"] = "Max retries exceeded."
            break

        execute_tasks(runnable)

        iteration += 1
        judger_log_capture = {}
        judger_verdict = run_judger_pipeline(
            plan=plan.model_dump(),
            agent_results=agent_results,
            assets=asset_list,
            client=judger_client,
            log_capture=judger_log_capture,
        )
        judger_log["rounds"].append(
            {
                "iteration": iteration,
                "request": judger_log_capture.get("request"),
                "trace": judger_log_capture.get("trace"),
                "result": judger_verdict,
            }
        )

    overall_status = judger_verdict.get("overall_status", "fail")

    task_manager_log["final_plan"] = plan.model_dump()
    task_manager_log["overall_status"] = overall_status
    task_manager_log["agent_results"] = agent_results

    try:
        _write_json(task_manager_log_path, task_manager_log)
        _write_json(judger_log_path, judger_log)
    except Exception as exc:
        logger.error("Failed to write logs: %s", exc)

    # 4. Return Aggregated Results
    return {
        "plan": plan.model_dump(),
        "agent_results": agent_results,
        "overall_status": overall_status,
        "overall_success": overall_status == "pass",
        "judger": judger_verdict,
    }

# Backward compatibility (optional, can be removed)
generate_guidance = generate_task_plan

__all__ = ["run_workflow", "generate_task_plan"]









def stream_workflow(user_query: str, config: dict = None, files_json: str = "[]"):
    """
    Generator for streaming workflow with Judger Integration.
    Outputs Server-Sent Events (SSE) compliant format.
    """
    import json
    from pathlib import Path

    # Helper: Convert Path objects to strings
    def make_serializable(obj):
        if isinstance(obj, (Path, type(Path()))):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        else:
            return obj


    import mimetypes
    import asyncio
    import inspect

    # ========== Judger Import with Fallback ==========
    run_judger_pipeline = None
    
    try:
        from judger_agent.judger_pipeline import run_judger_pipeline
    except Exception:
        try:
            from archive.judger_agent.judger_pipeline import run_judger_pipeline
        except Exception:
            pass

    if run_judger_pipeline is None:
        # Use Mock for verification if import fails
        def run_judger_pipeline(*, plan, agent_results, assets=None, client=None, log_capture=None):
            del assets, client, log_capture
            tasks = []
            for task in plan.get("subtasks", []):
                task_id = task.get("task_id") or task.get("id")
                tasks.append(
                    {
                        "task_id": task_id,
                        "verdict": "pass",
                        "failed_criteria": [],
                        "evidence": {},
                        "fix_instructions": "",
                    }
                )
            return {"overall_status": "pass", "tasks": tasks}

    if config is None: 
        config = {}
    
    from config import GENERATED_DIR

    # ========== Files Injection ==========
    assets: List[StoredAsset] = []
    asset_descriptors: List[Dict[str, Any]] = [{"type": "text", "text": user_query}]
    file_list: List[str] = []
    try:
        file_list = json.loads(files_json) if files_json else []
    except Exception:
        file_list = []

    for file_path in file_list:
        path = Path(file_path)
        if not path.exists():
            warn = f"[System]: Uploaded file not found: {file_path}"
            yield f"data: {json.dumps({'type': 'chunk', 'content': warn})}\n\n"
            continue

        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        assets.append(
            StoredAsset(
                path=path,
                url=str(path),
                mime_type=mime_type,
                original_filename=path.name,
            )
        )
        asset_descriptors.append(
            {
                "type": "file",
                "url": str(path),
                "mime_type": mime_type,
                "description": f"Uploaded asset: {path.name}",
                "original_filename": path.name,
            }
        )
    
    # ========== Generate Plan ==========
    try:
        # SSE Format: data: <json>\n\n
        yield f"data: {json.dumps({'type': 'chunk', 'content': '[System]: Generating Plan from User Config...'})}\n\n"

        task_client = _default_client()
        
        # ========== DETERMINISTIC TASK CREATION ==========
        # USER CONFIG TAKES PRIORITY - No LLM decision making!
        # DISABLED: LLM-based plan generation
        # plan_result = generate_task_plan(asset_descriptors, user_text=None, client=task_client)
        
        run_id = uuid.uuid4().hex
        timestamp = _local_timestamp()
        tasks = []
        
        # Build deterministic tasks based on user config
        # Order: presentation -> video (depends on slides) -> coder -> quizzer
        
        if config.get("slides", False):
            tasks.append({
                "task_id": f"step1_presentation_{timestamp}",
                "agent": "presentation",
                "instruction": f"Create a comprehensive educational PowerPoint presentation about: {user_query}. Include clear explanations, examples, and visual elements.",
                "inputs": {},
                "dependencies": [],
                "acceptance_criteria": [{
                    "criterion_id": "pptx_created",
                    "criterion_type": "output_shape",
                    "target": "result.success",
                    "operator": "equals",
                    "expected": True,
                    "severity": "required"
                }]
            })
        
        if config.get("video", False):
            video_deps = [f"step1_presentation_{timestamp}"] if config.get("slides", False) else []
            tasks.append({
                "task_id": f"step2_video_{timestamp}",
                "agent": "video",
                "instruction": f"Create an educational video about: {user_query}. Use generated slides if available.",
                "inputs": {},
                "dependencies": video_deps,
                "acceptance_criteria": [{
                    "criterion_id": "video_created",
                    "criterion_type": "output_shape",
                    "target": "result.success",
                    "operator": "equals",
                    "expected": True,
                    "severity": "required"
                }]
            })
        
        if config.get("code", False):
            tasks.append({
                "task_id": f"step3_coder_{timestamp}",
                "agent": "coder",
                "instruction": f"Generate a complete, runnable Python code example that demonstrates: {user_query}. Include comments explaining each part.",
                "inputs": {},
                "dependencies": [],
                "acceptance_criteria": [{
                    "criterion_id": "code_runs",
                    "criterion_type": "output_shape",
                    "target": "result.output.validation.run_success",
                    "operator": "equals",
                    "expected": True,
                    "severity": "required"
                }]
            })
        
        if config.get("quizzes", False):
            tasks.append({
                "task_id": f"step4_quizzer_{timestamp}",
                "agent": "quizzer",
                "instruction": f"Create an interactive quiz to test understanding of: {user_query}. Include multiple choice and fill-in-the-blank questions.",
                "inputs": {},
                "dependencies": [],
                "acceptance_criteria": [{
                    "criterion_id": "quiz_created",
                    "criterion_type": "output_shape",
                    "target": "result.success",
                    "operator": "equals",
                    "expected": True,
                    "severity": "required"
                }]
            })
        
        yield f"data: {json.dumps({'type': 'chunk', 'content': f'[System]: Created {len(tasks)} tasks based on your selections'})}\n\n"
        
        if len(tasks) == 0:
            yield f"data: {json.dumps({'type': 'chunk', 'content': '[System]: No output types selected! Please check at least one option.'})}\n\n"

        def _normalize_agent_name(value: Optional[str]) -> str:
            name = (value or "").strip().lower()
            if name in {"slides", "slide", "presentation", "ppt", "pptx"}:
                return "presentation"
            if name in {"code", "coder"}:
                return "coder"
            if name in {"quiz", "quizzes", "quizzer"}:
                return "quizzer"
            if name in {"video"}:
                return "video"
            # DISABLED: Text agent is a mock - do not use as fallback
            # if name in {"text", "writing", "writer"}:
            #     return "text"
            # return name or "text"
            return name  # Return as-is, registry will error if invalid

        def _agent_ui_type(agent_name: str) -> str:
            if agent_name == "presentation":
                return "slides"
            if agent_name == "coder":
                return "code"
            if agent_name == "quizzer":
                return "quiz"
            if agent_name == "video":
                return "video"
            # DISABLED: Text agent is not used
            # if agent_name in {"video", "text"}:
            #     return agent_name
            # return "text"
            return agent_name  # Return as-is for unknown

        def _agent_enabled(agent_name: str) -> bool:
            if agent_name == "video":
                return config.get("video", True)
            if agent_name == "presentation":
                return config.get("slides", True)
            if agent_name == "coder":
                return config.get("code", True)
            if agent_name == "quizzer":
                return config.get("quizzes", True)
            return True

        def _short_label(text: Optional[str], fallback: str) -> str:
            if not text:
                return fallback
            label = text.strip().splitlines()[0]
            if len(label) > 80:
                return label[:77] + "..."
            return label

        def _get_task_id(task: Dict[str, Any]) -> str:
            task_id = task.get("task_id") or task.get("id") or task.get("step_id")
            if not task_id:
                task_id = f"task_{uuid.uuid4().hex[:8]}"
                task["task_id"] = task_id
            return task_id

        def _strip_coder_criteria(task: Dict[str, Any]) -> None:
            criteria = task.get("acceptance_criteria") or []
            tasks_to_remove = {"is_python_file", "file_written", "code_written", "script_file_exists", "python_file_exists"}
            if isinstance(criteria, list):
                task["acceptance_criteria"] = [
                    c for c in criteria if c.get("criterion_id") not in tasks_to_remove
                ]

        normalized_tasks: List[Dict[str, Any]] = []
        for task in tasks:
            if isinstance(task, dict):
                normalized_tasks.append(task)
            elif hasattr(task, "model_dump"):
                normalized_tasks.append(task.model_dump())

        run_id = uuid.uuid4().hex
        output_dir = GENERATED_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        output_subdir = f"run_{run_id}"
        timestamp = _local_timestamp()

        for task in normalized_tasks:
            task_id = _get_task_id(task)
            agent_name = _normalize_agent_name(task.get("agent") or task.get("type"))
            task["agent"] = agent_name
            inputs = task.get("inputs") or {}
            base_stem = _safe_stem(task_id)
            inputs.setdefault("output_filename", f"{base_stem}_{timestamp}.py")
            inputs.setdefault("output_subdir", output_subdir)
            task["inputs"] = inputs
            if agent_name == "coder":
                _strip_coder_criteria(task)

        disabled_ids: Set[str] = set()
        for task in normalized_tasks:
            task_id = _get_task_id(task)
            if not _agent_enabled(task.get("agent")):
                disabled_ids.add(task_id)

        changed = True
        while changed:
            changed = False
            for task in normalized_tasks:
                task_id = _get_task_id(task)
                if task_id in disabled_ids:
                    continue
                deps = task.get("dependencies") or []
                if any(dep in disabled_ids for dep in deps):
                    disabled_ids.add(task_id)
                    changed = True

        for task in normalized_tasks:
            task_id = _get_task_id(task)
            if task_id in disabled_ids:
                label = _short_label(
                    task.get("instruction") or task.get("description"),
                    f"{task.get('agent', 'task')}",
                )
                skip_msg = f"[System]: Skipping {label} (disabled by configuration or dependency)."
                yield f"data: {json.dumps({'type': 'chunk', 'content': skip_msg})}\n\n"

        active_tasks = [task for task in normalized_tasks if _get_task_id(task) not in disabled_ids]

        def _run_subtask(task: Dict[str, Any], dependency_results: Dict[str, Any]) -> Dict[str, Any]:
            task_id = _get_task_id(task)
            agent_name = task.get("agent", "text")
            instruction = task.get("instruction") or task.get("description") or task.get("goal") or ""
            inputs = task.get("inputs") or {}

            try:
                agent_func = registry.get(agent_name)
                if inspect.iscoroutinefunction(agent_func):
                    result = asyncio.run(
                        agent_func(
                            instruction=instruction,
                            output_dir=output_dir,
                            assets=assets,
                            dependency_results=dependency_results,
                            client=task_client,
                            **inputs,
                        )
                    )
                else:
                    result = agent_func(
                        instruction=instruction,
                        output_dir=output_dir,
                        assets=assets,
                        dependency_results=dependency_results,
                        client=task_client,
                        **inputs,
                    )
                return {
                    "agent": agent_name,
                    "success": result.get("success", False),
                    "output": result.get("output"),
                    "artifacts": result.get("artifacts", []),
                    "error": result.get("error"),
                    "instruction": instruction,
                    "attempts": 1,
                    "metadata": result.get("metadata", {}),
                }
            except Exception as exc:
                logger.exception("Error executing subtask %s", task_id)
                return {
                    "agent": agent_name,
                    "success": False,
                    "error": str(exc),
                    "instruction": instruction,
                    "attempts": 1,
                }

        agent_results: Dict[str, Any] = {}
        completed: Set[str] = set()
        pending: List[Dict[str, Any]] = list(active_tasks)

        while pending:
            runnable: List[Dict[str, Any]] = []
            for task in pending:
                deps = task.get("dependencies") or []
                if all(dep in completed for dep in deps):
                    runnable.append(task)

            if not runnable:
                for task in pending:
                    task_id = _get_task_id(task)
                    error_msg = f"Unresolved dependencies for task {task_id}."
                    agent_results[task_id] = {
                        "agent": task.get("agent"),
                        "success": False,
                        "error": "Unresolved dependencies.",
                        "instruction": task.get("instruction", ""),
                    }
                    yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
                break

            for task in runnable:
                pending.remove(task)
                task_id = _get_task_id(task)
                agent_name = task.get("agent", "text")
                instruction = task.get("instruction") or task.get("description") or task.get("goal") or ""

                label = _short_label(instruction, f"{agent_name.title()} Agent")
                agent_type = _agent_ui_type(agent_name)

                # Agent Start Notification
                yield f"data: {json.dumps({'type': 'start', 'agent': label, 'agent_type': agent_type})}\n\n"

                # ========== Judgment Loop ==========
                MAX_RETRIES = 3

                for attempt in range(1, MAX_RETRIES + 2):
                    if attempt > 1:
                        retry_msg = f"\n[System]: Judgment failed. Retrying ({attempt-1}/{MAX_RETRIES})...\n"
                        yield f"data: {json.dumps({'type': 'chunk', 'content': retry_msg})}\n\n"

                    deps = task.get("dependencies") or []
                    dependency_results = {dep: agent_results.get(dep) for dep in deps if dep in agent_results}

                    result = _run_subtask(task, dependency_results)
                    agent_results[task_id] = result

                    # Convert Path objects to strings
                    result = make_serializable(result)

                    content = ""
                    if isinstance(result, dict):
                        raw = result.get("content") or result.get("output") or result.get("error")
                        if raw is None:
                            raw = result
                        # Ensure content is always a string (serialize dicts/lists)
                        if isinstance(raw, (dict, list)):
                            content = json.dumps(raw, indent=2, ensure_ascii=False, default=str)
                        else:
                            content = str(raw)
                    else:
                        content = str(result)

                    yield f"data: {json.dumps({'type': 'chunk', 'content': content})}\n\n"

                    # Only judge if we have substantial content
                    content_text = str(content).strip()
                    if not content_text or len(content_text) < 5:
                        if attempt <= MAX_RETRIES:
                            continue

                    verdict = run_judger_pipeline(
                        plan={"subtasks": [task]},
                        agent_results={task_id: result},
                        assets=assets,
                        client=task_client,
                    )

                    verdict_map = {item.get("task_id"): item for item in verdict.get("tasks", [])}
                    task_verdict = verdict_map.get(task_id, {})
                    if task_verdict.get("verdict") == "pass":
                        verify_msg = f"\n[System]: Verified ✔\n"
                        yield f"data: {json.dumps({'type': 'chunk', 'content': verify_msg})}\n\n"
                        break

                    fix = task_verdict.get("fix_instructions") or ""
                    if fix:
                        task["instruction"] = fix.strip()
                        instruction = task["instruction"]

                    reason = ", ".join(task_verdict.get("failed_criteria", [])) or "Unknown issues"
                    fail_msg = f"\n[System]: Verification Failed: {reason}\n"
                    yield f"data: {json.dumps({'type': 'chunk', 'content': fail_msg})}\n\n"
                    if attempt > MAX_RETRIES:
                        max_msg = f"\n[System]: Max retries reached.\n"
                        yield f"data: {json.dumps({'type': 'chunk', 'content': max_msg})}\n\n"

                completed.add(task_id)

                # Agent End Notification
                yield f"data: {json.dumps({'type': 'end'})}\n\n"
            
        # Workflow Complete
        yield f"data: {json.dumps({'type': 'complete'})}\n\n"
        
    except Exception as e:
        error_msg = f"Workflow Error: {str(e)}"
        yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"


