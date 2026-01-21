
import sys
import os
sys.path.append(os.getcwd())

from unittest.mock import MagicMock
from judger_agent.judger_pipeline import run_judger_pipeline

def test_judger_passes_with_filename_mismatch():
    """
    Simulates a scenario where:
    1. Task Plan expects 'lesson_123.py' (implied by strict logic, now relaxed).
    2. Agent actually produces 'lesson.py'.
    3. The criterion 'is_python_file' checks for '.py' extension.
    4. Deterministic checks should pass (since we removed strict name match).
    """

    # Mock Plan
    plan = {
        "subtasks": [
            {
                "task_id": "task1",
                "agent": "coder",
                "instruction": "Generate code",
                "acceptance_criteria": [
                    {
                        "criterion_id": "is_python_file",
                        "criterion_type": "output_shape",
                        "target": "result.artifacts[0]",
                        "operator": "contains",
                        "expected": ".py",
                        "severity": "required"
                    }
                ]
            }
        ]
    }

    # Mock Results (Agent produced a file with ANY name ending in .py)
    agent_results = {
        "task1": {
            "success": True,
            "output": {"code": "print('hello')"},
            # Simulate agent returning a path that ends in .py
            "artifacts": ["d:\\temp\\random_name.py"] 
        }
    }

    # Mock Logic to avoid OpenAI Key Error
    mock_client = MagicMock()
    
    result = run_judger_pipeline(
        plan=plan,
        agent_results=agent_results,
        client=mock_client
    )

    print("\nJudger Result:", result)
    
    # Verification
    assert result["overall_status"] == "pass", "Judger should PASS even if filename is random_name.py"
    assert result["tasks"][0]["verdict"] == "pass"

if __name__ == "__main__":
    test_judger_passes_with_filename_mismatch()
