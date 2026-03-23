"""Sanity checks for rookie-student evaluation outputs."""

from __future__ import annotations

from statistics import pstdev
from typing import Any, Dict, List


def evaluate_sanity(per_model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply sanity rules and return warning/fail diagnostics."""
    warnings: List[str] = []
    fails: List[str] = []

    # Per-model checks
    for r in per_model_results:
        model_name = r.get("model", "unknown_model")
        ori_pre = float(r.get("Acc_ori_pre", 0.0))
        ori_post = float(r.get("Acc_ori_post", 0.0))
        eq_pre = float(r.get("Acc_eq_pre", 0.0))
        eq_post = float(r.get("Acc_eq_post", 0.0))

        if ori_pre > 0.45 or eq_pre > 0.45:
            warnings.append(f"knowledge_leak_warning:{model_name}")
        if ori_pre > 0.65 or eq_pre > 0.65:
            fails.append(f"knowledge_leak_fail:{model_name}")

        if ori_post < (ori_pre - 0.05) or eq_post < (eq_pre - 0.05):
            fails.append(f"post_drop_fail:{model_name}")

        delta_ori = ori_post - ori_pre
        delta_eq = eq_post - eq_pre
        if delta_ori > 0.60 or delta_eq > 0.60:
            fails.append(f"abnormal_gain_fail:{model_name}")

    # Cross-model variance warnings
    if len(per_model_results) >= 2:
        keys = ("Acc_ori_pre", "Acc_ori_post", "Acc_eq_pre", "Acc_eq_post")
        for key in keys:
            vals = [float(x.get(key, 0.0)) for x in per_model_results]
            if len(vals) >= 2 and pstdev(vals) > 0.25:
                warnings.append(f"high_variance_warning:{key}")

    # Deduplicate while preserving order
    dedup_warnings = list(dict.fromkeys(warnings).keys())
    dedup_fails = list(dict.fromkeys(fails).keys())

    return {
        "sanity_warnings": dedup_warnings,
        "sanity_fail_reasons": dedup_fails,
        "sanity_fail": bool(dedup_fails),
    }
