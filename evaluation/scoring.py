"""Overall score computation for successful runs."""

from __future__ import annotations

from typing import Dict

from evaluation.constants import OVERALL_WEIGHTS


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def compute_overall_score(metrics: Dict[str, float]) -> float:
    """Compute overall score for a run that already passed hard gates.

    Formula:
      overall = 0.30*Learn + 0.10*Slide + 0.20*Code + 0.15*Quiz + 0.15*Consistency + 0.10*Sync
    """
    delta_ori = float(metrics.get("Δori", 0.0))
    delta_eq = float(metrics.get("Δeq", 0.0))

    learn = _clip01((max(0.0, delta_ori) + max(0.0, delta_eq)) / (0.30 * 2.0))
    slide = _clip01(float(metrics.get("slide_structural_compliance", 0.0)))

    code_exec_pass = 1.0 if bool(metrics.get("code_exec_pass", False)) else 0.0
    align_sc = _clip01(float(metrics.get("AlignSC", 0.0)))
    code = _clip01(0.70 * code_exec_pass + 0.30 * align_sc)

    quiz_format_validity = 1.0 if bool(metrics.get("quiz_format_validity", False)) else 0.0
    coverage = _clip01(float(metrics.get("Coverage", 0.0)))
    quiz = _clip01(0.50 * quiz_format_validity + 0.50 * coverage)

    consistency = _clip01(float(metrics.get("Consistency", 0.0)))
    sync = _clip01(float(metrics.get("Sync", 0.0)))

    score = (
        OVERALL_WEIGHTS["learn"] * learn
        + OVERALL_WEIGHTS["slide"] * slide
        + OVERALL_WEIGHTS["code"] * code
        + OVERALL_WEIGHTS["quiz"] * quiz
        + OVERALL_WEIGHTS["consistency"] * consistency
        + OVERALL_WEIGHTS["sync"] * sync
    )
    return round(_clip01(score), 4)
