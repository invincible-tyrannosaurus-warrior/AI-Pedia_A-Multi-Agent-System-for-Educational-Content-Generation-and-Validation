# Evaluation Workflow (Paper Protocol)

This folder implements the rewritten evaluation pipeline aligned with `evaluation rules.md`.

## What It Evaluates

Each request runs the full generation workflow and then evaluates all four artifacts:
- Slides
- Code
- Quiz
- Video

Hard gates:
- Missing any artifact -> `success_flag=False`, `overall_score=0`
- Generation latency > 2400s -> fail
- Silent failure (no event for 15 min, no `workflow_complete`) -> fail
- Rookie evaluation all models failed -> fail
- Sanity fail -> fail

## Rookie Student Evaluation

Per run, four groups are executed:
- `Acc_ori_pre`: original quiz, pre-learning
- `Acc_ori_post`: original quiz, post-learning
- `Acc_eq_pre`: equivalent quiz, pre-learning
- `Acc_eq_post`: equivalent quiz, post-learning

Current model IDs are placeholders in `evaluation/constants.py`:
- `PHI3_SMALL_PLACEHOLDER`
- `QWEN2_7B_PLACEHOLDER`
- `GEMMA_7B_PLACEHOLDER`

If model/API is not configured, runs will record model-call failures and fail by design.

## Metrics

Deterministic evaluation includes:
- Slide structure compliance: exactly 1 title, >=1 content, >=1 summary
- Code executability: timeout 30 min, with optional auto pip install (20 min per install)
- Quiz format validity: exactly 10 MCQ, each 4 options, single valid answer
- Concept metrics (TF-IDF, 1-2 gram, top_k=5):
  - `AlignSC`
  - `Coverage`
  - `OutOfScope`
  - `Consistency`
  - `Sync`

## Experiment Protocol

Default scheduler:
- 25 single-request units
- 25 concurrent batches
- 2 requests per batch
- Total requests: 75

## Run

```bash
python evaluation/evaluation_runner.py \
  --config configs/eval_topics.json \
  --out results \
  --single-units 25 \
  --concurrent-batches 25 \
  --batch-size 2
```

## Outputs

Generated under `--out`:
- `run_level_results.csv`
- `topic_best_results.csv`
- `aggregate_summary.json`

CSV rows include paper-friendly fields:
- `Accpre`, `Accpost`, `Acccontrol`, `ΔAcc`, `ΔAccnet`
- `AlignSC`, `Coverage`, `OutOfScope`, `Consistency`, `Sync`
- plus robustness and system fields (`completion/silent/partial`, latency, tokens, per-agent tokens)
