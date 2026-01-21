import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from openai import OpenAI
import os

from moe_layer.coder_agent.storage.local import StoredAsset

logger = logging.getLogger(__name__)

QUIZ_GENERATOR_PROMPT = """
You are an expert Educational Content Creator.
Your goal is to create a "Check for Understanding" quiz based on the provided learning material.

**Constraints**:
1. **Language**: The quiz MUST be entirely in **ENGLISH**. No other languages allowed.
2. **Format**: Output a valid **JSON** object (no markdown fencing like ```json).
3. **Content**: Generate 3-5 questions. Mix Multiple Choice and Fill-in-the-Blank.
4. **Tone**: Encouraging and educational.

**Input Context**:
{context}

**JSON Schema**:
{{
  "title": "Quiz Title",
  "questions": [
    {{
      "id": 1,
      "type": "multiple_choice",
      "question": "Question text here?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": "Option B",
      "explanation": "Explanation of why B is correct."
    }},
    {{
      "id": 2,
      "type": "fill_in_blank",
      "question": "Python is a dynamic ___ language.",
      "correct_answer": "programming",
      "explanation": "Python is a high-level, general-purpose programming language."
    }}
  ]
}}
"""

def run_quiz_pipeline(
    instruction: str,
    output_dir: Path,
    assets: Optional[List[StoredAsset]] = None,
    client: Optional[OpenAI] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Execute the Quizzer Agent pipeline (Real LLM).
    
    Args:
        instruction: Topic/Instruction.
        output_dir: Base output directory.
        assets: Optional assets (files).
        client: OpenAI client passed from task_manager.
        kwargs: Must contain 'output_subdir'.
    """
    try:
        output_subdir = kwargs.get("output_subdir", "")
        run_dir = output_dir / output_subdir if output_subdir else output_dir
        
        # 1. Resolve Context
        context_text = f"Topic: {instruction}\n"
        
        # Try to load storyboard from the SAME run execution
        storyboard_path = run_dir / "storyboard.json"
        if storyboard_path.exists():
            try:
                with open(storyboard_path, "r", encoding="utf-8") as f:
                    sb = json.load(f)
                    # Extract text content from slides to use as context
                    slides_text = []
                    for s in sb.get("slides", []):
                        title = s.get("title", "")
                        content = s.get("content", "")
                        slides_text.append(f"Slide: {title}\nContent: {content}")
                    context_text += "\n".join(slides_text)
                    logger.info("Loaded context from storyboard.json")
            except Exception as e:
                logger.warning(f"Failed to load storyboard: {e}")
        
        # 2. Call LLM - Use passed client or fallback to default OpenAI
        if client is None:
            client = OpenAI()  # Uses OPENAI_API_KEY env var
        
        prompt = QUIZ_GENERATOR_PROMPT.format(context=context_text)
        
        logger.info("Generating quiz via LLM...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use OpenAI model for reliability
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content.strip()
        # Clean markdown fencing if present
        if content.startswith("```json"):
            content = content.replace("```json", "", 1).replace("```", "", 1)
        elif content.startswith("```"):
            content = content.replace("```", "", 1).replace("```", "", 1)
            
        quiz_data = json.loads(content)
        
        # 3. Save Output
        final_output_dir = run_dir / "output"
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        quiz_file = final_output_dir / "quiz.json"
        with open(quiz_file, "w", encoding="utf-8") as f:
            json.dump(quiz_data, f, indent=2)
            
        logger.info(f"Quiz saved to {quiz_file}")

        return {
            "success": True,
            "output": quiz_data, # Return the whole JSON object so frontend can render it
            "artifacts": [str(quiz_file)],
            "metadata": {
                "model": "gpt-4o-mini"
            }
        }

    except Exception as e:
        logger.error(f"Quizzer Agent failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }
