
import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append("d:/L3/Individual_project/AI_Pedia_Local")

from manager_agent.task_manager_agent import run_workflow

def test_presentation_workflow():
    print("Starting Presentation Agent Integration Test...")
    
    # 1. Define Request
    user_text = "Create a lesson for beginners on Python 'class'. Include code examples and a diagram."
    output_dir = Path("d:/L3/Individual_project/AI_Pedia_Local/data/generated_code/test_ppt_run")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Run Workflow
    val = run_workflow(
        user_text=user_text,
        output_dir=output_dir,
        assets=[]  # No uploads
    )
    
    # 3. Analyze Results
    print("\nWorkflow Finished.")
    print(f"Overall Status: {val.get('overall_status')}")
    
    agent_results = val.get("agent_results", {})
    if "presentation" not in agent_results:
        print("FAIL: Presentation agent was not triggered in the plan.")
        print(f"Agents triggered: {agent_results.keys()}")
        # Check plan
        print("Plan:", val.get("plan"))
        return
        
    ppt_result = agent_results["presentation"]
    print(f"Presentation Agent Success: {ppt_result.get('success')}")
    
    if not ppt_result.get("success"):
        print(f"FAIL: Presentation Agent returned error: {ppt_result.get('error')}")
        return

    # 4. Verify Artifacts
    artifacts = ppt_result.get("artifacts", [])
    if not artifacts:
        print("FAIL: No artifacts returned.")
        return
        
    pptx_path = Path(artifacts[0])
    print(f"Generated PPTX: {pptx_path}")
    
    if not pptx_path.exists():
        print(f"FAIL: File does not exist at {pptx_path}")
        return
        
    if pptx_path.suffix != ".pptx":
        print(f"FAIL: Artifact is not a .pptx file: {pptx_path}")
        return
        
    print("PASS: Presentation workflow executed and produced a PPTX file.")

if __name__ == "__main__":
    test_presentation_workflow()
