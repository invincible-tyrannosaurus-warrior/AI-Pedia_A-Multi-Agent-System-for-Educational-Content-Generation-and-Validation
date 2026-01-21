
import sys
import os
sys.path.append(os.getcwd())

import logging
from pathlib import Path
from moe_layer.coder_agent.coder import save_generated_code
from judger_agent.judger_pipeline import _resolve_target, _resolve_path

def test_path_handoff():
    print("\n--- Testing Path Handoff ---")
    
    # 1. Simulate Coder Agent output path generation
    output_dir = Path("./data/generated_code").resolve()
    code_content = "print('Integration Test')"
    
    # This uses the actual function from coder.py
    saved_path = save_generated_code(
        code=code_content, 
        directory=output_dir, 
        filename="integration_test.py"
    )
    print(f"Coder Generated Path: {saved_path} (Type: {type(saved_path)})")
    
    # 2. Simulate Metadata Payload constructed in task_manager_agent.py
    # Ref: task_manager_agent.py:247
    agent_result = {
        "success": True,
        "output": {"code": code_content},
        "artifacts": [saved_path]  # This lists Path objects
    }
    
    # 3. Simulate Judger Criterion Resolution
    # Ref: task_manager_agent.py:336 target="result.artifacts[0]"
    target_path_str = "result.artifacts[0]"
    
    # Use internal resolution logic from judger_pipeline.py
    resolved_value = _resolve_target(
        target=target_path_str, 
        result=agent_result, 
        tools={}
    )
    
    print(f"Judger Resolved Value: {resolved_value} (Type: {type(resolved_value)})")
    
    # 4. Verification
    if resolved_value == saved_path:
        print("✅ SUCCESS: Judger correctly resolved the exact Path object.")
    else:
        print("❌ FAILURE: Path mismatch.")
        print(f"Expected: {saved_path}")
        print(f"Got:      {resolved_value}")

    # 5. Test String Conversion for JSON serialization
    # In a real workflow, data often gets serialized.
    # Let's see if stringifying breaks resolution.
    agent_result_serialized = {
        "success": True,
        "artifacts": [str(saved_path)] # Serialized to string
    }
    resolved_str = _resolve_target("result.artifacts[0]", agent_result_serialized, tools={})
    print(f"Resolved from String: {resolved_str}")
    
    path_from_str = Path(resolved_str)
    if path_from_str.resolve() == saved_path.resolve():
         print("✅ SUCCESS: Path remains valid after string serialization.")
    else:
         print("❌ FAILURE: Serialization broke the path.")

if __name__ == "__main__":
    test_path_handoff()
