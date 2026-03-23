"""Rookie-student evaluation (primary educational-effectiveness metric)."""

from __future__ import annotations

import json
import logging
import os
import random
import re
from statistics import mean
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - environment-dependent import
    OpenAI = None  # type: ignore[assignment]

from evaluation.constants import (
    ROOKIE_CALL_TIMEOUT_SEC,
    ROOKIE_MODELS,
    ROOKIE_MODEL_MAX_ATTEMPTS,
    VALID_ANSWER_LETTERS,
)

logger = logging.getLogger(__name__)


class ModelInvocationError(RuntimeError):
    """Raised when a rookie model call fails after retries."""


def _letter_to_index(letter: str) -> Optional[int]:
    if not letter:
        return None
    letter = letter.strip().upper()
    if letter in VALID_ANSWER_LETTERS:
        return ord(letter) - ord("A")
    return None


def _index_to_letter(idx: int) -> str:
    return chr(ord("A") + idx)


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty response")

    # Fast path
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"answers": parsed}
    except Exception:
        pass

    # Fallback: find first JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found")

    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON is not an object")
    return parsed


def _normalize_model_answers(payload: Dict[str, Any], questions: List[Dict[str, Any]]) -> Dict[str, str]:
    """Normalize model answer payload into id -> A/B/C/D mapping."""
    answers = payload.get("answers", payload.get("result", payload))

    pred: Dict[str, str] = {}

    if isinstance(answers, dict):
        # Could be {"1":"A", "2":"B"}
        for k, v in answers.items():
            letter = str(v).strip().upper()[:1]
            if letter in VALID_ANSWER_LETTERS:
                pred[str(k)] = letter
    elif isinstance(answers, list):
        for i, item in enumerate(answers):
            if isinstance(item, dict):
                qid = item.get("id", item.get("question_id", i + 1))
                ans = item.get("answer", item.get("choice", ""))
            else:
                qid = i + 1
                ans = item
            letter = str(ans).strip().upper()[:1]
            if letter in VALID_ANSWER_LETTERS:
                pred[str(qid)] = letter

    # Backfill positional keys if model used 0/1-based positional-only list
    if not pred and isinstance(answers, list):
        for i, item in enumerate(answers):
            letter = str(item).strip().upper()[:1]
            if letter in VALID_ANSWER_LETTERS:
                pred[str(i + 1)] = letter

    # Map to known ids if model replied with sparse/mismatched ids
    if pred:
        normalized: Dict[str, str] = {}
        ordered_ids = [str(q.get("id", i + 1)) for i, q in enumerate(questions)]

        for i, qid in enumerate(ordered_ids):
            if qid in pred:
                normalized[qid] = pred[qid]
                continue

            # fallback key candidates
            one_based = str(i + 1)
            zero_based = str(i)
            if one_based in pred:
                normalized[qid] = pred[one_based]
            elif zero_based in pred:
                normalized[qid] = pred[zero_based]

        return normalized

    return {}


def _grade_quiz(questions: List[Dict[str, Any]], pred: Dict[str, str]) -> float:
    if not questions:
        return 0.0

    correct = 0
    for i, q in enumerate(questions):
        qid = str(q.get("id", i + 1))
        gt = str(q.get("answer_letter", "")).strip().upper()
        pd = pred.get(qid, "")
        if gt and pd == gt:
            correct += 1

    return round(correct / len(questions), 4)


def _build_quiz_prompt(questions: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for i, q in enumerate(questions):
        qid = q.get("id", i + 1)
        lines.append(f"QID: {qid}")
        lines.append(f"Question: {q.get('question', '')}")
        opts = q.get("options", [])
        for j, opt in enumerate(opts[:4]):
            letter = chr(ord("A") + j)
            lines.append(f"{letter}. {opt}")
        lines.append("")
    return "\n".join(lines)


def _build_equivalent_quiz_heuristic(questions: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    """Dynamic per-run equivalent quiz via deterministic rewrite heuristics."""
    rng = random.Random(seed)
    stems = [
        "Equivalent check:",
        "Parallel form:",
        "Same-level variation:",
        "Alternative wording:",
    ]

    eq_questions: List[Dict[str, Any]] = []
    for i, q in enumerate(questions):
        opts = list(q.get("options", []))
        if len(opts) != 4:
            # Keep structure; upstream validation should already catch malformed quiz.
            opts = (opts + ["N/A", "N/A", "N/A", "N/A"])[:4]

        answer_letter = str(q.get("answer_letter", "A")).upper()
        answer_idx = _letter_to_index(answer_letter)
        if answer_idx is None:
            answer_idx = 0

        shift = rng.randint(1, 3)
        rotated = opts[shift:] + opts[:shift]
        new_answer_idx = (answer_idx - shift) % 4

        prefix = rng.choice(stems)
        new_question_text = f"{prefix} {q.get('question', '').strip()}"

        eq_questions.append(
            {
                "id": q.get("id", i + 1),
                "question": new_question_text,
                "options": rotated,
                "answer_letter": _index_to_letter(new_answer_idx),
                "explanation": f"Equivalent variant. {q.get('explanation', '')}".strip(),
            }
        )

    return eq_questions


def _get_rookie_client() -> OpenAI:
    if OpenAI is None:
        raise ModelInvocationError("openai package is not installed in current environment")

    api_key = (
        os.getenv("ROOKIE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
    )
    if not api_key:
        raise ModelInvocationError("Missing ROOKIE_API_KEY/OPENAI_API_KEY/OPENROUTER_API_KEY")

    base_url = os.getenv("ROOKIE_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def _is_placeholder_model(model_id: str) -> bool:
    s = (model_id or "").strip().upper()
    return (not s) or ("PLACEHOLDER" in s)


def _call_model_for_answers(
    client: OpenAI,
    model_id: str,
    questions: List[Dict[str, Any]],
    materials_text: str,
    is_pretest: bool,
) -> Dict[str, str]:
    if _is_placeholder_model(model_id):
        raise ModelInvocationError(f"Model '{model_id}' is placeholder")

    rookie_constraint = (
        "You are a rookie student. "
        "If this is pre-test, assume you have NOT studied this topic and must avoid prior knowledge. "
        "If this is post-test, answer only from provided study notes. "
        "If uncertain, guess one option. "
        "Always output strict JSON only."
    )

    stage_note = "PRETEST" if is_pretest else "POSTTEST"
    study_blob = "" if is_pretest else f"\n\nStudy Notes:\n{materials_text[:18000]}"

    user_prompt = (
        f"Stage: {stage_note}\n"
        "Answer all questions using one option letter (A/B/C/D).\n"
        "Output schema:\n"
        "{\"answers\":[{\"id\":<qid>,\"answer\":\"A\"}, ...]}\n\n"
        "Questions:\n"
        f"{_build_quiz_prompt(questions)}"
        f"{study_blob}"
    )

    last_err: Optional[Exception] = None
    for attempt in range(1, ROOKIE_MODEL_MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": rookie_constraint},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                timeout=ROOKIE_CALL_TIMEOUT_SEC,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            payload = _extract_json_object(content)
            return _normalize_model_answers(payload, questions)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning("Rookie call failed model=%s attempt=%d error=%s", model_id, attempt, exc)

    raise ModelInvocationError(f"Model call failed after retries for {model_id}: {last_err}")


def _run_single_model_eval(
    client: OpenAI,
    model_spec: Dict[str, str],
    original_questions: List[Dict[str, Any]],
    equivalent_questions: List[Dict[str, Any]],
    materials_text: str,
) -> Dict[str, Any]:
    model_name = model_spec.get("name", "unknown")
    model_id = model_spec.get("model", "")

    try:
        ori_pre_pred = _call_model_for_answers(client, model_id, original_questions, materials_text, is_pretest=True)
        ori_post_pred = _call_model_for_answers(client, model_id, original_questions, materials_text, is_pretest=False)
        eq_pre_pred = _call_model_for_answers(client, model_id, equivalent_questions, materials_text, is_pretest=True)
        eq_post_pred = _call_model_for_answers(client, model_id, equivalent_questions, materials_text, is_pretest=False)

        acc_ori_pre = _grade_quiz(original_questions, ori_pre_pred)
        acc_ori_post = _grade_quiz(original_questions, ori_post_pred)
        acc_eq_pre = _grade_quiz(equivalent_questions, eq_pre_pred)
        acc_eq_post = _grade_quiz(equivalent_questions, eq_post_pred)

        return {
            "model": model_name,
            "model_id": model_id,
            "ok": True,
            "Acc_ori_pre": acc_ori_pre,
            "Acc_ori_post": acc_ori_post,
            "Acc_eq_pre": acc_eq_pre,
            "Acc_eq_post": acc_eq_post,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "model": model_name,
            "model_id": model_id,
            "ok": False,
            "error": str(exc),
        }


def run_rookie_evaluation(
    original_questions: List[Dict[str, Any]],
    materials_text: str,
    run_seed: int,
    rookie_models: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Run rookie evaluation with 3 models and average available-model scores."""
    models = rookie_models or ROOKIE_MODELS

    if not original_questions:
        return {
            "rookie_eval_failed": True,
            "rookie_eval_error": "No valid quiz questions provided.",
            "rookie_models_success_n": 0,
            "rookie_failed_models": [m.get("name", "unknown") for m in models],
            "per_model_results": [],
            "Acc_ori_pre": 0.0,
            "Acc_ori_post": 0.0,
            "Acc_eq_pre": 0.0,
            "Acc_eq_post": 0.0,
            "Accpre": 0.0,
            "Accpost": 0.0,
            "Acccontrol": 0.0,
            "Δori": 0.0,
            "Δeq": 0.0,
            "ΔAcc": 0.0,
            "ΔAccnet": 0.0,
            "equivalent_quiz": [],
        }

    equivalent_questions = _build_equivalent_quiz_heuristic(original_questions, seed=run_seed)

    try:
        client = _get_rookie_client()
    except Exception as exc:  # noqa: BLE001
        # Keep this explicit so runs record model-call failure cleanly.
        failed_models = [m.get("name", "unknown") for m in models]
        return {
            "rookie_eval_failed": True,
            "rookie_eval_error": str(exc),
            "rookie_models_success_n": 0,
            "rookie_failed_models": failed_models,
            "per_model_results": [],
            "Acc_ori_pre": 0.0,
            "Acc_ori_post": 0.0,
            "Acc_eq_pre": 0.0,
            "Acc_eq_post": 0.0,
            "Accpre": 0.0,
            "Accpost": 0.0,
            "Acccontrol": 0.0,
            "Δori": 0.0,
            "Δeq": 0.0,
            "ΔAcc": 0.0,
            "ΔAccnet": 0.0,
            "equivalent_quiz": equivalent_questions,
        }

    per_model: List[Dict[str, Any]] = []
    ok_rows: List[Dict[str, Any]] = []

    for model_spec in models:
        row = _run_single_model_eval(
            client=client,
            model_spec=model_spec,
            original_questions=original_questions,
            equivalent_questions=equivalent_questions,
            materials_text=materials_text,
        )
        per_model.append(row)
        if row.get("ok"):
            ok_rows.append(row)

    failed_models = [r.get("model", "unknown") for r in per_model if not r.get("ok")]

    if not ok_rows:
        return {
            "rookie_eval_failed": True,
            "rookie_eval_error": "All rookie model calls failed.",
            "rookie_models_success_n": 0,
            "rookie_failed_models": failed_models,
            "per_model_results": per_model,
            "Acc_ori_pre": 0.0,
            "Acc_ori_post": 0.0,
            "Acc_eq_pre": 0.0,
            "Acc_eq_post": 0.0,
            "Accpre": 0.0,
            "Accpost": 0.0,
            "Acccontrol": 0.0,
            "Δori": 0.0,
            "Δeq": 0.0,
            "ΔAcc": 0.0,
            "ΔAccnet": 0.0,
            "equivalent_quiz": equivalent_questions,
        }

    acc_ori_pre = round(mean(float(r["Acc_ori_pre"]) for r in ok_rows), 4)
    acc_ori_post = round(mean(float(r["Acc_ori_post"]) for r in ok_rows), 4)
    acc_eq_pre = round(mean(float(r["Acc_eq_pre"]) for r in ok_rows), 4)
    acc_eq_post = round(mean(float(r["Acc_eq_post"]) for r in ok_rows), 4)

    delta_ori = round(acc_ori_post - acc_ori_pre, 4)
    delta_eq = round(acc_eq_post - acc_eq_pre, 4)
    delta_acc = delta_ori
    delta_accnet = round(delta_ori - delta_eq, 4)

    return {
        "rookie_eval_failed": False,
        "rookie_eval_error": None,
        "rookie_models_success_n": len(ok_rows),
        "rookie_failed_models": failed_models,
        "per_model_results": per_model,
        "Acc_ori_pre": acc_ori_pre,
        "Acc_ori_post": acc_ori_post,
        "Acc_eq_pre": acc_eq_pre,
        "Acc_eq_post": acc_eq_post,
        "Accpre": acc_ori_pre,
        "Accpost": acc_ori_post,
        "Acccontrol": acc_eq_pre,
        "Δori": delta_ori,
        "Δeq": delta_eq,
        "ΔAcc": delta_acc,
        "ΔAccnet": delta_accnet,
        "equivalent_quiz": equivalent_questions,
    }
