"""Shared constants for the evaluation workflow."""

from __future__ import annotations

# Hard gates
LATENCY_BUDGET_SEC = 2400
SILENT_FAILURE_INACTIVITY_SEC = 15 * 60

# Artifact requirements
REQUIRED_ARTIFACTS = ("slides", "code", "quiz", "video")

# Quiz constraints
EXPECTED_QUIZ_QUESTION_COUNT = 10
EXPECTED_QUIZ_OPTION_COUNT = 4
VALID_ANSWER_LETTERS = ("A", "B", "C", "D")

# Code execution constraints
CODE_EXEC_TIMEOUT_SEC = 30 * 60
PIP_INSTALL_TIMEOUT_SEC = 20 * 60

# Concept extraction
TFIDF_TOP_K = 5
TFIDF_NGRAM_RANGE = (1, 2)

# Rookie model placeholders (to be replaced later)
ROOKIE_MODELS = [
    {"name": "phi-3 small", "model": "PHI3_SMALL_PLACEHOLDER"},
    {"name": "qwen2-7b", "model": "QWEN2_7B_PLACEHOLDER"},
    {"name": "gemma 7b", "model": "GEMMA_7B_PLACEHOLDER"},
]

# Rookie call behavior
ROOKIE_MODEL_MAX_ATTEMPTS = 3  # initial + 2 retries
ROOKIE_CALL_TIMEOUT_SEC = 120

# Overall score weights
OVERALL_WEIGHTS = {
    "learn": 0.30,
    "slide": 0.10,
    "code": 0.20,
    "quiz": 0.15,
    "consistency": 0.15,
    "sync": 0.10,
}

# Scheduler defaults (paper protocol)
DEFAULT_SINGLE_UNITS = 25
DEFAULT_CONCURRENT_BATCHES = 25
DEFAULT_BATCH_SIZE = 2

# Output fields (paper-friendly + traceability)
PAPER_FRIENDLY_FIELDS = [
    "Accpre",
    "Accpost",
    "Acccontrol",
    "ΔAcc",
    "ΔAccnet",
    "AlignSC",
    "Coverage",
    "OutOfScope",
    "Consistency",
    "Sync",
    "completion_rate",
    "partial_failure_rate",
    "silent_failure_rate",
    "latency_mean",
    "tokens_total",
    "tokens_by_agent",
]
