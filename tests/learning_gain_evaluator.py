"""
Learning Gain Evaluator for AI_Pedia_Local_stream System.

This script simulates a "student" agent learning from generated materials to evaluate:
1. Educational Effectiveness (Learning Gain: Post-test - Pre-test Accuracy)
2. Content Coverage/Consistency (Do materials cover the quiz concepts?)

It uses the `observability_module` for tracing and token tracking.

Author: AI_Pedia_Team
"""

import json
import csv
import logging
import random
import re
import sys
import uuid
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import Observability Module
try:
    from tests.observability_module import TaskContext, LLMWrapper, TraceLogger
except ImportError:
    print("Error: Could not import observability_module. Ensure tests/observability_module.py exists.")
    sys.exit(1)

# Configure Logging
logger = logging.getLogger("learning_evaluator")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class LearningSimulator:
    """
    Simulates a student taking tests and studying materials.
    """
    def __init__(self, llm: LLMWrapper, model: str = "gpt-4o"):
        self.llm = llm
        self.model = model

    def run_pretest(self, quiz: List[Dict]) -> Dict[str, Any]:
        """
        Runs the pre-test (Zero-shot, guessing allowed).
        """
        logger.info("Starting Pre-test...")
        results = []
        total_questions = len(quiz)
        correct_count = 0
        total_confidence = 0.0

        system_prompt = (
            "You are a novice student who has NOT studied this topic yet. "
            "You must answer the following quiz questions. "
            "If you do not know the answer, you MUST guess. "
            "Do not use any external knowledge beyond common sense. "
            "For each question, provide your answer and a confidence score between 0.0 and 1.0."
        )

        for q in quiz:
            verdict, confidence = self._answer_question(q, system_prompt, context="")
            results.append({
                "question_id": q.get("question_id"),
                "correct": verdict,
                "confidence": confidence
            })
            if verdict:
                correct_count += 1
            total_confidence += confidence

        return {
            "accuracy": correct_count / total_questions if total_questions > 0 else 0,
            "avg_confidence": total_confidence / total_questions if total_questions > 0 else 0,
            "details": results
        }

    def run_study(self, slides_text: str, video_transcript: str, code_text: str):
        """
        Simulates the study phase. The model reads materials and summarizes them.
        This establishes the context for the post-test.
        """
        logger.info("Starting Study Phase...")
        
        study_prompt = (
            "You are a diligent student. You are now studying the following materials to prepare for a test.\n\n"
            f"=== SLIDES CONTENT ===\n{slides_text[:10000]}...\n\n" # Truncate to avoid context limit if huge
            f"=== VIDEO TRANSCRIPT ===\n{video_transcript[:10000]}...\n\n"
            f"=== CODE EXAMPLES ===\n{code_text[:5000]}...\n\n"
            "Task:\n"
            "1. Summarize the core concept in 1 sentence.\n"
            "2. List 3 key takeaways.\n"
            "3. Explain one code example provided."
        )

        # We don't really use the output, just the act of processing it (and potentially caching it if we were using a stateful chat, 
        # but here we rely on the fact that we will Pass this study context into the Post-test prompt).
        # Actually, to simulate "memory", we will include the materials in the Post-test prompt.
        # But we run this call to generate a "Study Event" log and to verify the model can process the text.
        
        self.llm.chat(
            agent_name="student_learning",
            model_name=self.model,
            messages=[{"role": "user", "content": study_prompt}]
        )
        # We don't return anything, the "state" is the materials themselves which passed to post-test.

    def run_posttest(self, quiz: List[Dict], materials_context: str) -> Dict[str, Any]:
        """
        Runs the post-test (With access to study materials in context).
        """
        logger.info("Starting Post-test...")
        results = []
        total_questions = len(quiz)
        correct_count = 0
        total_confidence = 0.0

        system_prompt = (
            "You are a student who has just studied the following materials.\n"
            f"{materials_context}\n\n"
            "Now answer the quiz questions based on what you have learned. "
            "For each question, provide your answer and a confidence score between 0.0 and 1.0."
        )

        for q in quiz:
            verdict, confidence = self._answer_question(q, system_prompt, context="")
            results.append({
                "question_id": q.get("question_id"),
                "correct": verdict,
                "confidence": confidence
            })
            if verdict:
                correct_count += 1
            total_confidence += confidence

        return {
            "accuracy": correct_count / total_questions if total_questions > 0 else 0,
            "avg_confidence": total_confidence / total_questions if total_questions > 0 else 0,
            "details": results
        }

    def _answer_question(self, question: Dict, system_prompt: str, context: str) -> Tuple[bool, float]:
        """
        Helper to answer a single question and grade it.
        Returns (is_correct, confidence).
        """
        q_text = question["question"]
        q_type = question["type"]
        options = question.get("options", [])
        
        user_msg = f"Question: {q_text}\n"
        if q_type == "mcq" and options:
             user_msg += "Options:\n" + "\n".join(options) + "\n"
             user_msg += "Output format: JSON with keys 'answer' (the option letter A/B/C/D) and 'confidence' (float)."
        else:
             user_msg += "Output format: JSON with keys 'answer' (your short answer) and 'confidence' (float)."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ]
        
        try:
            response = self.llm.chat(
                agent_name="student_test",
                model_name=self.model,
                messages=messages,
                response_format={"type": "json_object"} # Force JSON if model supports, else text
            )
            content = response["choices"][0]["message"]["content"]
            
            # Simple JSON parse (assuming model follows instruction)
            try:
                data = json.loads(content)
                answer = data.get("answer", "")
                confidence = float(data.get("confidence", 0.5))
            except json.JSONDecodeError:
                # Fallback heuristic parsing
                answer = content.strip()
                confidence = 0.5
            
            # Grading
            is_correct = False
            correct_answer = question["answer"]
            
            if q_type == "mcq":
                # Check if the letter matches
                # Normalize: remove punctuation, uppercase
                pred = str(answer).upper().strip()
                truth = str(correct_answer).upper().strip()
                # If pred is "A) Text", extract "A"
                if len(pred) > 1 and pred[1] in [")", "."]:
                    pred = pred[0]
                is_correct = (pred == truth)
                
            else: # short_answer
                # Keyword matching (current scheme: 70% match)
                # Split truth into keywords (simple space split as requested)
                truth_keywords = set(correct_answer.lower().split())
                pred_keywords = set(str(answer).lower().split())
                
                if not truth_keywords:
                    is_correct = True # Empty answer?
                else:
                    intersection = truth_keywords.intersection(pred_keywords)
                    match_rate = len(intersection) / len(truth_keywords)
                    is_correct = match_rate >= 0.70
            
            return is_correct, confidence

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return False, 0.0

class KnowledgeTracer:
    """
    Evaluates consistency and coverage using LLM.
    """
    def __init__(self, llm: LLMWrapper, model: str = "gpt-4o"):
        self.llm = llm
        self.model = model

    def evaluate_coverage(self, quiz: List[Dict], materials_text: str) -> Dict[str, float]:
        """
        Checks if quiz concepts are present in materials.
        Returns {coverage_rate, out_of_scope_rate}.
        """
        logger.info("Starting Coverage Check (LLM-based)...")
        covered_count = 0
        total_questions = len(quiz)
        
        if total_questions == 0:
            return {"coverage_rate": 0.0, "out_of_scope_rate": 0.0}

        # Batch check or single check? Single is more precise.
        for q in quiz:
            prompt = (
                "You are a strict Teaching Assistant. Check if the following question's core concept is covered in the provided teaching materials.\n\n"
                f"Question: {q['question']}\n"
                f"Correct Answer: {q['answer']}\n\n"
                "=== TEACHING MATERIALS ===\n"
                f"{materials_text[:15000]}...\n\n" # Truncate 
                "Does the material contain information necessary to answer this question?\n"
                "Output JSON: {\"covered\": true/false, \"reason\": \"...\"}"
            )
            
            try:
                response = self.llm.chat(
                    agent_name="knowledge_tracer",
                    model_name=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                content = json.loads(response["choices"][0]["message"]["content"])
                if content.get("covered", False):
                    covered_count += 1
                    
            except Exception as e:
                logger.error(f"Error checking coverage: {e}")
                # Assume covered if error to be lenient, or not? Let's assume NOT covered to be strict.
        
        coverage_rate = covered_count / total_questions
        return {
            "coverage_rate": coverage_rate,
            "out_of_scope_rate": 1.0 - coverage_rate
        }

def write_results_to_csv(filepath: str, row: Dict[str, Any]):
    file_exists = Path(filepath).exists()
    headers = [
        "topic", 
        "accuracy_pre", "accuracy_post", "delta_accuracy", 
        "confidence_pre", "confidence_post", 
        "coverage_rate", "out_of_scope_rate",
        "timestamp"
    ]
    
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        
        # Add timestamp
        row["timestamp"] = datetime.now().isoformat()
        writer.writerow(row)

# --- Main Demo ---

def main():
    # Setup dummy data for demo
    topic = "Python Lists"
    
    # 1. Dummy Materials (In reality, these come from the system output)
    slides_text = "Python lists are mutable sequences. You can append items using .append(). Lists are defined with square brackets []."
    video_text = "Hello students. Today we talk about lists. Remember, lists can be changed, meaning they are mutable. Use brackets to make one."
    code_text = "my_list = [1, 2, 3]\nmy_list.append(4)\nprint(my_list) # Output: [1, 2, 3, 4]"
    
    combined_materials = f"{slides_text}\n{video_text}\n{code_text}"
    
    # 2. Dummy Quiz
    quiz = [
        {
            "question_id": "q1",
            "type": "mcq",
            "question": "Which method adds an item to a list?",
            "options": ["A. add()", "B. append()", "C. push()", "D. insert()"],
            "answer": "B"
        },
        {
            "question_id": "q2",
            "type": "short_answer",
            "question": "Are Python lists mutable or immutable?",
            "answer": "Mutable"
        },
        { # Out of scope question
            "question_id": "q3",
            "type": "short_answer",
            "question": "What is the time complexity of len()?",
            "answer": "O(1)" 
        }
    ]

    # 3. Setup Context
    run_dir = Path("tests") / "learning_eval_run" 
    with TaskContext(task_id="learning_eval_demo", run_dir=run_dir) as ctx:
        llm = LLMWrapper(ctx)
        
        # --- Mocking LLM for Demo Purposes ---
        # Since we don't have a real API key in this env, we override chat to return
        # valid JSON for the specific agents, so the parsing logic doesn't fail.
        original_chat = llm.chat
        
        def mock_chat_override(agent_name, model_name, messages, **kwargs):
            # 1. Pre-test / Post-test (Student) -> Returns Answer + Confidence
            if agent_name.startswith("student"):
                return {
                    "choices": [{
                        "message": {
                            "content": json.dumps({"answer": "B", "confidence": 0.8}) 
                        }
                    }],
                    "usage": {"total_tokens": 50}
                }
            # 2. Knowledge Tracer -> Returns Coverage JSON
            elif agent_name == "knowledge_tracer":
                return {
                    "choices": [{
                        "message": {
                            "content": json.dumps({"covered": True, "reason": "Concept found in slides."})
                        }
                    }],
                    "usage": {"total_tokens": 50}
                }
            # 3. Study Phase -> Returns Summary (Text)
            elif agent_name == "student_learning":
                 return {
                    "choices": [{
                        "message": {
                            "content": "Summary: Lists are mutable. Key points: 1. Mutable 2. Ordered 3. Versatile."
                        }
                    }],
                    "usage": {"total_tokens": 100}
                }
            return original_chat(agent_name, model_name, messages, **kwargs)
            
        llm.chat = mock_chat_override
        # -------------------------------------

        # 4. Initialize Evaluators

        simulator = LearningSimulator(llm)
        tracer = KnowledgeTracer(llm)
        
        # 5. Run Pre-test
        pre_metrics = simulator.run_pretest(quiz)
        print(f"Pre-test Accuracy: {pre_metrics['accuracy']:.2f}")
        
        # 6. Run Study
        simulator.run_study(slides_text, video_text, code_text)
        
        # 7. Run Post-test
        post_metrics = simulator.run_posttest(quiz, combined_materials)
        print(f"Post-test Accuracy: {post_metrics['accuracy']:.2f}")
        
        # 8. Check Coverage
        cov_metrics = tracer.evaluate_coverage(quiz, combined_materials)
        print(f"Coverage Rate: {cov_metrics['coverage_rate']:.2f}")
        
        # 9. Compile Results
        result_row = {
            "topic": topic,
            "accuracy_pre": round(pre_metrics["accuracy"], 2),
            "accuracy_post": round(post_metrics["accuracy"], 2),
            "delta_accuracy": round(post_metrics["accuracy"] - pre_metrics["accuracy"], 2),
            "confidence_pre": round(pre_metrics["avg_confidence"], 2),
            "confidence_post": round(post_metrics["avg_confidence"], 2),
            "coverage_rate": round(cov_metrics["coverage_rate"], 2),
            "out_of_scope_rate": round(cov_metrics["out_of_scope_rate"], 2)
        }
        
        # 10. Write to CSV
        output_csv = "learning_gain_report.csv"
        write_results_to_csv(output_csv, result_row)
        print(f"Results written to {output_csv}")

if __name__ == "__main__":
    main()
