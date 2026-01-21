from pydantic import BaseModel, Field, AliasChoices, ValidationError
from typing import List

print("=== Verifying Robust Pydantic Validation ===")

try:
    # Defining models exactly as they are in task_manager_agent.py now
    class SubTask(BaseModel):
        task_id: str = Field(..., validation_alias=AliasChoices('id', 'task_id', 'step_id'))
        agent: str
        instruction: str = Field(..., validation_alias=AliasChoices('title', 'instruction', 'description', 'goal'))

    class TaskPlan(BaseModel):
        user_intent: str = Field(..., validation_alias=AliasChoices('user_intent', 'goal', 'intent', 'objective', 'topic'))
        subtasks: List[SubTask] = Field(..., validation_alias=AliasChoices('subtasks', 'steps', 'tasks', 'plan', 'actions', 'workflow'))

    # Mock LLM Output (The "Bad" one that failed before)
    bad_output = {
        "goal": "Learn Python",
        "steps": [
            {"id": "1", "agent": "coder", "title": "Print hello"}
        ]
    }

    print(f"Testing with chaotic input: {bad_output}")

    plan = TaskPlan(**bad_output)
    print("✅ Validation SUCCESS!")
    print(f"Parsed Intent: {plan.user_intent}")
    print(f"Parsed Task ID: {plan.subtasks[0].task_id}")
    print(f"Parsed Instruction: {plan.subtasks[0].instruction}")

except ValidationError as e:
    print("❌ Validation FAILED:")
    print(e)
