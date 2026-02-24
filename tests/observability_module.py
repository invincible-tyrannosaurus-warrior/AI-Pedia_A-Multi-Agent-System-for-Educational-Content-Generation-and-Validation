"""
Observability Module for AI_Pedia_Local_stream.

This module provides a unified mechanism for:
1. Logging detailed LLM execution traces (trace.jsonl).
2. Tracking token usage across agents and models.
3. Generating task-level summaries (summary.json).

Classes:
    - TraceLogger: Handles safe writing of event logs.
    - TokenTracker: Aggregates token usage statistics.
    - LLMWrapper: Wraps model calls to automatically log and track usage.
    - TaskContext: Manages the lifecycle and context of a specific task execution.

Usage:
    with TaskContext(task_id="task_123", run_dir="./runs/task_123") as ctx:
        wrapper = LLMWrapper(ctx)
        response = wrapper.chat("agent_a", "gpt-4", messages)
"""

import json
import time
import os
import uuid
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Configure logging for the module itself (not the trace log)
logger = logging.getLogger("observability")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class TraceLogger:
    """
    Logs events to a JSONL file.
    Ensures thread-safe (in simple use cases) and exception-safe writing.
    """
    def __init__(self, trace_file: Path):
        self.trace_file = Path(trace_file)
        # Ensure directory exists
        self.trace_file.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event_type: str, task_id: str, run_id: str, payload: Dict[str, Any]):
        """
        Writes a single event to the trace file.
        
        Args:
            event_type: e.g., "LLM_CALL", "LLM_ERROR"
            task_id: Unique identifier for the task
            run_id: Unique identifier for the specific run
            payload: Dictionary containing event data (agent, model, latency, usage, etc.)
        """
        entry = {
            "ts": datetime.now().isoformat(),
            "task_id": task_id,
            "run_id": run_id,
            "event": event_type,
            **payload
        }
        
        try:
            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            # Fallback logging to stderr so we don't lose the error, but don't crash main loop
            logger.error(f"Failed to write to trace log: {e}")

class TokenTracker:
    """
    Aggregates token usage statistics.
    Supports slicing by agent and by model.
    """
    def __init__(self):
        # Structure: self.by_agent[agent_name][model_name] = {usage_dict}
        self.by_agent: Dict[str, Dict[str, Dict[str, int]]] = {}
        # Structure: self.by_model[model_name] = {usage_dict}
        self.by_model: Dict[str, Dict[str, int]] = {}
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def add(self, agent: str, model: str, usage: Dict[str, int]):
        """
        Adds token usage to the tracker.
        
        Args:
            agent: Name of the agent (e.g., "coder", "video")
            model: Name of the model (e.g., "gpt-4o")
            usage: Dict with prompt_tokens, completion_tokens, total_tokens
        """
        p = usage.get("prompt_tokens", 0)
        c = usage.get("completion_tokens", 0)
        t = usage.get("total_tokens", 0)

        # Update totals
        self.total_usage["prompt_tokens"] += p
        self.total_usage["completion_tokens"] += c
        self.total_usage["total_tokens"] += t

        # Update by agent
        if agent not in self.by_agent:
            self.by_agent[agent] = {}
        if model not in self.by_agent[agent]:
            self.by_agent[agent][model] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        self.by_agent[agent][model]["prompt_tokens"] += p
        self.by_agent[agent][model]["completion_tokens"] += c
        self.by_agent[agent][model]["total_tokens"] += t

        # Update by model
        if model not in self.by_model:
            self.by_model[model] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            
        self.by_model[model]["prompt_tokens"] += p
        self.by_model[model]["completion_tokens"] += c
        self.by_model[model]["total_tokens"] += t

    def get_totals(self) -> Dict[str, int]:
        return self.total_usage

    def get_by_agent(self) -> Dict[str, Any]:
        return self.by_agent

    def get_by_model(self) -> Dict[str, Any]:
        return self.by_model

class TaskContext:
    """
    Context manager for a single task execution.
    Manages the run directory, TraceLogger, and TokenTracker.
    Automatically generates summary.json on exit.
    """
    def __init__(self, task_id: str, run_dir: Union[str, Path]):
        self.task_id = task_id
        self.run_id = str(uuid.uuid4())
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.trace_logger = TraceLogger(self.run_dir / "trace.jsonl")
        self.token_tracker = TokenTracker()
        
        self.start_time = None
        self.end_time = None
        self.success = False

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Starting Task {self.task_id} (Run ID: {self.run_id})")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.success = (exc_type is None)
        
        if exc_type:
            logger.error(f"Task failed with error: {exc_val}")
            # Log the fatal error to trace as well
            self.trace_logger.log_event(
                "TASK_ERROR", 
                self.task_id, 
                self.run_id, 
                {"error": str(exc_val), "traceback": traceback.format_exc()}
            )

        self._save_summary()
        logger.info(f"Task complete. Summary saved to {self.run_dir / 'summary.json'}")

    def _save_summary(self):
        latency = self.end_time - self.start_time
        summary = {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat(),
            "latency_seconds": round(latency, 2),
            "success": self.success,
            "tokens": {
                "total": self.token_tracker.get_totals(),
                "by_agent": self.token_tracker.get_by_agent(),
                "by_model": self.token_tracker.get_by_model()
            }
        }
        
        try:
            with open(self.run_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")

class LLMWrapper:
    """
    Wrapper for LLM calls.
    Handles timing, usage extraction, tracking, and logging transparently.
    """
    def __init__(self, ctx: TaskContext):
        self.ctx = ctx

    def chat(self, agent_name: str, model_name: str, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Simulates or executes a chat completion call.
        
        Args:
            agent_name: Name of the calling agent.
            model_name: Model identifier.
            messages: List of message dicts (role, content).
            **kwargs: Additional args for the model API.

        Returns:
            Full response dictionary (simulating OpenAI format).
        """
        start_ts = time.time()
        
        try:
            # --- External Call Simulation ---
            # In a real system, this would be: client.chat.completions.create(...)
            response = mock_external_call(model_name, messages, **kwargs)
            # --------------------------------
            
            latency_ms = int((time.time() - start_ts) * 1000)
            
            usage = self._extract_usage(response)
            
            # Update Tracker
            self.ctx.token_tracker.add(agent_name, model_name, usage)
            
            # Log Event
            self.ctx.trace_logger.log_event(
                "LLM_CALL",
                self.ctx.task_id,
                self.ctx.run_id,
                {
                    "agent": agent_name,
                    "model": model_name,
                    "provider": "openai-compatible", # Could be dynamic
                    "latency_ms": latency_ms,
                    "usage": usage,
                    "request_messages_count": len(messages),
                    "response_len": len(response["choices"][0]["message"]["content"])
                }
            )
            
            return response

        except Exception as e:
            latency_ms = int((time.time() - start_ts) * 1000)
            logger.error(f"LLM Call failed: {e}")
            
            # Log Error
            self.ctx.trace_logger.log_event(
                "LLM_ERROR",
                self.ctx.task_id,
                self.ctx.run_id,
                {
                    "agent": agent_name,
                    "model": model_name,
                    "latency_ms": latency_ms,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )
            raise # Re-raise to let caller handle flow control

    def _extract_usage(self, response: Dict[str, Any]) -> Dict[str, int]:
        """
        Extracts usage from response or estimates fallback.
        """
        if "usage" in response:
            return {
                "prompt_tokens": response["usage"].get("prompt_tokens", 0),
                "completion_tokens": response["usage"].get("completion_tokens", 0),
                "total_tokens": response["usage"].get("total_tokens", 0)
            }
        
        # Fallback estimation
        content = response["choices"][0]["message"]["content"]
        est_tokens = len(content) // 4
        return {
            "prompt_tokens": 0, # Cannot easily est prompt without tokenizer
            "completion_tokens": est_tokens,
            "total_tokens": est_tokens
        }

# --- Mock Infrastructure for Demo ---

def mock_external_call(model: str, messages: List[Dict], **kwargs) -> Dict[str, Any]:
    """
    Simulates an OpenAI-compatible API response.
    """
    # Simulate network latency
    time.sleep(0.1) 
    
    # Simple logic to generate "content"
    last_msg = messages[-1]['content']
    response_content = f"Simulated response to: {last_msg[:20]}..."
    
    # Calculate fake token usage
    prompt_len = sum(len(m['content']) for m in messages)
    completion_len = len(response_content)
    
    prompt_tokens = prompt_len // 4
    completion_tokens = completion_len // 4
    
    # Fail randomly for demo if requested (omitted for stability)
    
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }

# --- Main Demo ---

def main():
    """
    Demonstrates the observability module usage.
    """
    # Create a unique run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(__file__).parent / f"test_run_{timestamp}"
    
    print(f"Starting Demo Run in {run_dir}...")
    
    # 1. Initialize TaskContext
    with TaskContext(task_id="demo_task_001", run_dir=run_dir) as ctx:
        
        # 2. Initialize Wrapper
        llm = LLMWrapper(ctx)
        
        # 3. Simulate Agent Calls
        
        # Agent 1: Coder
        print("Agent: Coder calling gpt-4...")
        llm.chat(
            agent_name="coder",
            model_name="gpt-4",
            messages=[{"role": "user", "content": "Write a python script to sort a list."}]
        )
        
        # Agent 2: Reviewer (using same model)
        print("Agent: Reviewer calling gpt-4...")
        llm.chat(
            agent_name="reviewer",
            model_name="gpt-4",
            messages=[{"role": "user", "content": "Check this code for bugs."}]
        )
        
        # Agent 3: Summarizer (using distinct model)
        print("Agent: Summarizer calling gpt-3.5-turbo...")
        llm.chat(
            agent_name="summarizer",
            model_name="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Summarize the changes."}]
        )
        
        # Agent 1 again
        print("Agent: Coder calling gpt-4 again...")
        llm.chat(
            agent_name="coder",
            model_name="gpt-4",
            messages=[{"role": "user", "content": "Fix the bugs."}]
        )

    print("\nRun Complete.")
    print(f"Check {run_dir / 'trace.jsonl'} for detailed logs.")
    print(f"Check {run_dir / 'summary.json'} for token stats.")

if __name__ == "__main__":
    main()
