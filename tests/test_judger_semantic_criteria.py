from types import SimpleNamespace

from judger_agent.judger_pipeline import run_judger_pipeline


class _FakeClient:
    def __init__(self, verdict: str, failed_criteria: list[str]):
        self._verdict = verdict
        self._failed_criteria = failed_criteria
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        del kwargs
        tool_call = SimpleNamespace(
            function=SimpleNamespace(
                name="judge_task",
                arguments=(
                    '{"verdict":"%s","failed_criteria":%s,"evidence":{},"fix_instructions":"revise output"}'
                    % (self._verdict, self._failed_criteria)
                ),
            )
        )
        message = SimpleNamespace(tool_calls=[tool_call])
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


def test_semantic_criterion_can_fail_task():
    plan = {
        "subtasks": [
            {
                "task_id": "task_video",
                "agent": "video",
                "instruction": "Create a coherent lesson video",
                "acceptance_criteria": [
                    {
                        "criterion_id": "video_exists",
                        "criterion_type": "output_shape",
                        "target": "result.success",
                        "operator": "equals",
                        "expected": True,
                        "severity": "required",
                    },
                    {
                        "criterion_id": "semantic_quality",
                        "criterion_type": "semantic",
                        "target": "result.output",
                        "operator": "exists",
                        "expected": True,
                        "severity": "required",
                    },
                ],
            }
        ]
    }
    agent_results = {"task_video": {"success": True, "output": {"summary": "bad"}, "artifacts": ["video.mp4"]}}

    result = run_judger_pipeline(
        plan=plan,
        agent_results=agent_results,
        client=_FakeClient("fail", '["semantic_quality"]'),
    )

    assert result["overall_status"] == "fail"
    assert result["tasks"][0]["verdict"] == "fail"
    assert "semantic_quality" in result["tasks"][0]["failed_criteria"]


def test_deterministic_only_task_does_not_need_llm_client():
    plan = {
        "subtasks": [
            {
                "task_id": "task_code",
                "agent": "coder",
                "instruction": "Generate code",
                "acceptance_criteria": [
                    {
                        "criterion_id": "code_runs",
                        "criterion_type": "output_shape",
                        "target": "result.output.validation.run_success",
                        "operator": "equals",
                        "expected": True,
                        "severity": "required",
                    }
                ],
            }
        ]
    }
    agent_results = {"task_code": {"success": True, "output": {"validation": {"run_success": True}}}}

    result = run_judger_pipeline(plan=plan, agent_results=agent_results, client=None)

    assert result["overall_status"] == "pass"
    assert result["tasks"][0]["verdict"] == "pass"
