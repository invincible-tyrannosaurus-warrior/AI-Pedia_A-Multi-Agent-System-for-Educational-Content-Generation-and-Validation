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
3. **Question Count**: Generate exactly **10** questions. No more, no less.
4. **Question Type**: ALL questions MUST be **Multiple Choice**. Do NOT include fill-in-the-blank.
5. **Options**: Each question MUST have exactly **4** options.
6. **Answer Format**: The `"answer"` field MUST be a single letter: `"A"`, `"B"`, `"C"`, or `"D"`.
7. **Tone**: Encouraging and educational.

**Input Context**:
{context}

**JSON Schema** (follow this EXACTLY):
{{
  "title": "Quiz Title",
  "questions": [
    {{
      "id": 1,
      "question": "What is the primary purpose of a list in Python?",
      "options": ["To store a single value", "To store an ordered collection of items", "To define a function", "To create a loop"],
      "answer": "B",
      "explanation": "Lists are used to store ordered collections of items in Python."
    }},
    {{
      "id": 2,
      "question": "Which method adds an element to the end of a list?",
      "options": ["insert()", "append()", "extend()", "add()"],
      "answer": "B",
      "explanation": "The append() method adds a single element to the end of a list."
    }}
  ]
}}

IMPORTANT: You MUST output exactly 10 questions. The "answer" field MUST be "A", "B", "C", or "D" only.
"""

_VALID_ANSWERS = {"A", "B", "C", "D"}
_EXPECTED_Q_COUNT = 10
_MAX_ATTEMPTS = 3


def _parse_llm_content(raw: str) -> dict:
    """Strip markdown fencing, fix bad escapes, parse JSON."""
    import re

    content = raw.strip()
    # Remove markdown fencing
    if content.startswith("```json"):
        content = content.replace("```json", "", 1).replace("```", "", 1)
    elif content.startswith("```"):
        content = content.replace("```", "", 1).replace("```", "", 1)

    # Fix invalid JSON escape sequences (e.g. \e, \a from LaTeX)
    content = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', content)

    return json.loads(content.strip())


def _validate_and_fix_quiz(data: dict) -> tuple:
    """Validate quiz structure and auto-fix minor answer issues.

    Returns (is_valid, issues_list, fixed_data).
    """
    issues: List[str] = []
    questions = []

    if isinstance(data, dict):
        questions = data.get("questions", data.get("quiz", []))
    elif isinstance(data, list):
        questions = data
    else:
        return False, ["Root element is neither dict nor list"], data

    # --- Check question count ---
    if len(questions) != _EXPECTED_Q_COUNT:
        issues.append(f"Expected {_EXPECTED_Q_COUNT} questions, got {len(questions)}")

    # --- Validate & fix each question ---
    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            issues.append(f"Q{i+1}: not a dict")
            continue

        # Ensure question text exists
        if not q.get("question"):
            issues.append(f"Q{i+1}: missing question text")

        # Ensure 4 options
        opts = q.get("options", [])
        if len(opts) != 4:
            issues.append(f"Q{i+1}: has {len(opts)} options instead of 4")

        # Auto-fix answer field: extract leading letter
        raw_ans = str(q.get("answer", q.get("correct_answer", ""))).strip()
        if raw_ans:
            letter = raw_ans[0].upper()
            if letter in _VALID_ANSWERS:
                q["answer"] = letter          # normalise to single letter
                q.pop("correct_answer", None) # unify field name
            else:
                issues.append(f"Q{i+1}: invalid answer '{raw_ans}'")
        else:
            issues.append(f"Q{i+1}: missing answer")

    # Update questions back into data
    if isinstance(data, dict):
        if "questions" in data:
            data["questions"] = questions
        elif "quiz" in data:
            data["quiz"] = questions

    is_valid = len(issues) == 0
    return is_valid, issues, data


def run_quiz_pipeline(
    instruction: str,
    output_dir: Path,
    assets: Optional[List[StoredAsset]] = None,
    client: Optional[OpenAI] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Execute the Quizzer Agent pipeline (Real LLM).

    Generates a 10-question MCQ quiz with A/B/C/D answers.
    Automatically validates the output and retries up to 3 times
    if the LLM produces an invalid structure.
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
                    slides_text = []
                    for s in sb.get("slides", []):
                        title = s.get("title", "")
                        content = s.get("content", "")
                        slides_text.append(f"Slide: {title}\nContent: {content}")
                    context_text += "\n".join(slides_text)
                    logger.info("Loaded context from storyboard.json")
            except Exception as e:
                logger.warning(f"Failed to load storyboard: {e}")

        # 2. Call LLM with retry loop
        if client is None:
            client = OpenAI()

        prompt = QUIZ_GENERATOR_PROMPT.format(context=context_text)

        quiz_data = None
        last_issues: List[str] = []

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            logger.info("Generating quiz via LLM (attempt %d/%d)...", attempt, _MAX_ATTEMPTS)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7 if attempt == 1 else 0.5,  # lower temp on retry
                max_tokens=3000,
            )

            raw = response.choices[0].message.content
            try:
                parsed = _parse_llm_content(raw)
            except json.JSONDecodeError as exc:
                last_issues = [f"JSON parse error: {exc}"]
                logger.warning("Attempt %d: %s", attempt, last_issues[0])
                continue

            is_valid, issues, fixed = _validate_and_fix_quiz(parsed)
            quiz_data = fixed

            if is_valid:
                logger.info("Quiz validation passed on attempt %d", attempt)
                break

            last_issues = issues
            logger.warning("Attempt %d validation issues: %s", attempt, "; ".join(issues))
            # Append a reminder to the prompt for the next attempt
            prompt = (
                QUIZ_GENERATOR_PROMPT.format(context=context_text)
                + f"\n\n*** PREVIOUS ATTEMPT HAD ERRORS: {'; '.join(issues)} ***\n"
                  "Please fix these issues. Remember: EXACTLY 10 questions, "
                  "each with 4 options and answer as a single letter A/B/C/D."
            )

        if quiz_data is None:
            raise ValueError(f"Failed after {_MAX_ATTEMPTS} attempts: {'; '.join(last_issues)}")

        # 3. Save Output
        final_output_dir = run_dir / "output"
        final_output_dir.mkdir(parents=True, exist_ok=True)

        quiz_file = final_output_dir / "quiz.json"
        with open(quiz_file, "w", encoding="utf-8") as f:
            json.dump(quiz_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Quiz saved to {quiz_file}")

        return {
            "success": True,
            "output": quiz_data,
            "artifacts": [str(quiz_file)],
            "metadata": {
                "model": "gpt-4o-mini",
                "attempts": attempt,
                "validation_issues": last_issues if last_issues else None,
            }
        }

    except Exception as e:
        logger.error(f"Quizzer Agent failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }
