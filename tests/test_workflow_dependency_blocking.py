from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from manager_agent.task_manager_agent import TaskPlan, SubTask, AcceptanceCriterion, run_workflow


def _criterion(criterion_id: str, target: str, expected):
    return AcceptanceCriterion(
        criterion_id=criterion_id,
        criterion_type="output_shape",
        target=target,
        operator="equals",
        expected=expected,
        severity="required",
    )


def test_failed_dependency_blocks_downstream_execution():
    plan = TaskPlan(
        user_intent="test dependency blocking",
        subtasks=[
            SubTask(
                task_id="step1",
                agent="coder",
                instruction="fail upstream",
                inputs={},
                dependencies=[],
                acceptance_criteria=[_criterion("step1_success", "result.success", True)],
            ),
            SubTask(
                task_id="step2",
                agent="quizzer",
                instruction="depends on upstream",
                inputs={},
                dependencies=["step1"],
                acceptance_criteria=[_criterion("step2_success", "result.success", True)],
            ),
        ],
    )

    def fake_registry_get(name):
        if name == "coder":
            return lambda **kwargs: {"success": False, "error": "upstream failed"}
        if name == "quizzer":
            raise AssertionError("Downstream task should not execute when dependency fails")
        raise AssertionError(f"Unexpected agent: {name}")

    def fake_judger(*, plan, agent_results, assets=None, client=None, log_capture=None):
        del assets, client, log_capture
        verdicts = []
        for task in plan.get("subtasks", []):
            task_id = task["task_id"]
            verdicts.append(
                {
                    "task_id": task_id,
                    "verdict": "pass" if agent_results.get(task_id, {}).get("success") else "fail",
                    "failed_criteria": [] if agent_results.get(task_id, {}).get("success") else ["result_success"],
                    "evidence": {},
                    "fix_instructions": "",
                }
            )
        overall = "pass" if all(v["verdict"] == "pass" for v in verdicts) else "fail"
        return {"overall_status": overall, "tasks": verdicts}

    with TemporaryDirectory() as temp_dir:
        with patch("manager_agent.task_manager_agent.generate_task_plan", return_value=plan), patch(
            "manager_agent.task_manager_agent.registry.get", side_effect=fake_registry_get
        ), patch("manager_agent.task_manager_agent.run_judger_pipeline", side_effect=fake_judger), patch(
            "manager_agent.task_manager_agent._default_client", return_value=object()
        ):
            result = run_workflow(user_text="topic", assets=[], output_dir=Path(temp_dir))

    assert result["overall_status"] == "fail"
    assert result["agent_results"]["step1"]["success"] is False
    assert result["agent_results"]["step2"]["success"] is False
    assert result["agent_results"]["step2"]["error"] == "Unresolved dependencies."
