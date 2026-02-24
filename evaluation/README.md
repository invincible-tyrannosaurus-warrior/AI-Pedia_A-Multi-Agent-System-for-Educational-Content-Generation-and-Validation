# Evaluation Tool-chain

Engineering-grade evaluation suite for the AI-Pedia multi-agent teaching-content generation system.

## Quick Start

```bash
# From project root
python evaluation/evaluation_runner.py \
    --config configs/eval_topics.json \
    --runs 3 \
    --out results/
```

## Output Files

| File | Description |
|---|---|
| `results/run_level_results.csv` | One row per run (6 topics × 3 runs = 18 rows) |
| `results/topic_best_results.csv` | Best run per topic (6 rows) |
| `results/aggregate_summary.json` | Mean / std of all numeric metrics |

## Directory Convention

Each run produces artefacts in a fixed directory structure:

```
results/
├── <topic_id>/
│   ├── run_1/
│   │   ├── lesson_<ts>.pptx          # presentation agent output
│   │   ├── storyboard.json           # intermediate (ignored)
│   │   ├── scripts/
│   │   │   └── <task_id>_<ts>.py      # coder agent output
│   │   ├── output/
│   │   │   ├── quiz.json              # quizzer agent output
│   │   │   └── <stem>_lecture.mp4     # video agent output
│   │   └── assets/                    # generated images (ignored)
│   ├── run_2/ ...
│   └── run_3/ ...
├── run_level_results.csv
├── topic_best_results.csv
└── aggregate_summary.json
```

File names are **dynamic** (timestamped). The evaluation runner discovers files by extension:
- `.pptx` → slides
- `.py` (in `scripts/`) → code
- `.json` (containing quiz data, in `output/`) → quiz
- `.mp4` → video

## Connecting the Real System

The default `run_system()` in `evaluation_runner.py` is a **stub** that checks whether artefact files have been pre-placed in the output directory.

### Option A – Pre-generate artefacts

Place files manually or via a CI step, then run the evaluation:

```bash
# 1. Generate (your own script / manual)
python -m your_system --topic "Python Lists" --out results/t1_python_lists/run_1/

# 2. Evaluate
python evaluation/evaluation_runner.py --config configs/eval_topics.json --runs 3 --out results/
```

### Option B – Inline integration

Replace the body of `run_system()` with a call to `stream_workflow`:

```python
from manager_agent.task_manager_agent import stream_workflow

def run_system(topic_cfg, run_id, output_dir):
    config = {"video": True, "slides": True, "code": True, "quizzes": True}
    for event in stream_workflow(topic_cfg["prompt"], config):
        pass  # consume the generator; artefacts are written to disk
    return {"success": True, "error": None}
```

## Acceptance Criteria

| Artefact | Pass Condition |
|---|---|
| **Slides** | 5–12 pages, title slide present |
| **Code** | `python <script>.py` exits with return code 0 within 30 s |
| **Quiz** | 10 questions, all multiple-choice with 4 options, valid answers (A/B/C/D) |
| **Video** | File exists and ≥ 1 MB |

### Cross-artefact Consistency

- `slide_code_consistency` – keyword overlap between slides and code
- `slide_quiz_coverage_rate` – quiz keywords found in slides
- `out_of_scope_rate` – 1 − coverage rate

### Scoring Formula

```
overall_score =
    0.20 × slides_pass
  + 0.20 × code_exec_pass
  + 0.20 × quiz_format_pass
  + 0.20 × video_pass
  + 0.10 × slide_code_consistency
  + 0.10 × slide_quiz_coverage_rate
```

Best-run selection: among runs with `success_flag=True`, pick the one with the highest `overall_score`.

## Dependencies

| Package | Required | Purpose |
|---|---|---|
| `python-pptx` | Yes | PPTX text extraction |
| `ffprobe` / `moviepy` | Optional | Video duration probing (graceful fallback to `None`) |
| Standard library | Yes | `subprocess`, `csv`, `json`, `pathlib`, `math` |

## quiz.json Expected Format

```json
[
  {
    "question": "What does list.append() do?",
    "options": ["Adds to end", "Removes last", "Sorts list", "Reverses list"],
    "answer": "A"
  }
]
```

Answers must be `A`/`B`/`C`/`D` or `0`/`1`/`2`/`3`.
