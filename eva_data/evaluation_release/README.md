# Evaluation Package

This folder is a compact release bundle for the model evaluation work conducted on the `student_lessons` dataset.

## Contents

- `README.md` — this document
- `code/` — evaluation scripts used in the experiments
- `results/` — final experiment outputs used for analysis
- `supporting_files/` — auxiliary scripts and dataset metadata useful for reproducibility
- `results_summary.csv` — compact table of all final model results
- `results_summary.md` — paper-friendly summary table and notes

## Dataset and task

The evaluation dataset lives outside this package at:

- `C:\Users\Yue\Desktop\eva_data\student_lessons`

The dataset contains 20 quiz topics with 10 multiple-choice questions each, for a total of **200 questions**.
A `manifest.json` file maps each topic folder to a `student_input.json` payload.
A copy of the dataset `manifest.json` is included in `supporting_files/manifest.json` for reference.

## Evaluation settings

Two main settings were used for each model.

### 1. Pre-learn (closed-book)
The model is given:
- question
- answer options

It is **not** given the lesson text.
This measures baseline performance using the model's existing knowledge.

### 2. Post-learn (open-book)
The model is given:
- lesson title
- topic
- full lesson text
- question
- answer options

It is instructed to answer using the lesson content.
This measures performance when the model can directly use the provided lesson materials during inference.

## Core pipeline

Main script:
- `code/eval_student_lessons.py`

What it does:
1. Loads the dataset via `manifest.json`
2. Resolves each `student_input.json`
3. Builds prompts for either pre-learn or post-learn mode
4. Runs model inference through a local `transformers` backend (or other supported backends)
5. Parses final answers as `A/B/C/D`
6. Writes:
   - `summary.json`
   - `topic_summary.json`
   - `predictions.jsonl`

## Model execution environment

The evaluations were run on a Runpod server with:
- NVIDIA GeForce RTX 4090
- local Hugging Face model loading
- 4-bit quantized inference for most local-transformers runs

Model cache was redirected to `/workspace/hf_cache` on the server to avoid filling the root disk.

## Important implementation note

`Qwen/Qwen2.5-1.5B-Instruct` initially failed smoke testing because of inference/backend configuration issues, not because the model was inherently unable to answer the questions.
The `transformers` backend was later hardened to improve tokenizer/chat-template termination behavior and stabilize generation.
The final results in this package for Qwen 1.5B are from the **repaired pipeline**.

## TinyLlama note

TinyLlama completed the evaluation process successfully, but:
- pre-learn performance was low
- post-learn performance was substantially worse

An additional TinyLlama-specific summary-based post-learn experiment was created to test whether shorter instructional context would help. It did not materially improve outcomes. That experimental script is included in `code/` and `supporting_files/` for transparency.

## Included result folders

### Final result folders
- `results/phi3_prelearn_eval`
- `results/phi3_full_eval`
- `results/phi35_prelearn_eval`
- `results/phi35_postlearn_eval`
- `results/qwen25_3b_prelearn_eval`
- `results/qwen25_3b_postlearn_eval`
- `results/qwen25_15b_prelearn_eval`
- `results/qwen25_15b_postlearn_eval`
- `results/tinyllama_prelearn_eval`
- `results/tinyllama_postlearn_eval`

### Diagnostic / non-final outputs kept for transparency
- `supporting_files/qwen25_15b_repaired_smoke_summary.json`
- `supporting_files/qwen25_15b_prelearn_smokeeval_summary.json`
- `supporting_files/tinyllama_postlearn_summary_smoke_summary.json`
- `supporting_files/qwen15b_diag.py`
- `supporting_files/tinyllama_smoke.py`

## How to rerun

### Example: pre-learn
```bash
python code/eval_student_lessons.py \
  --data-dir "C:/Users/Yue/Desktop/eva_data/student_lessons" \
  --backend transformers \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --mode prelearn \
  --output-dir results/qwen25_15b_prelearn_eval
```

### Example: post-learn
```bash
python code/eval_student_lessons.py \
  --data-dir "C:/Users/Yue/Desktop/eva_data/student_lessons" \
  --backend transformers \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --mode postlearn \
  --output-dir results/qwen25_15b_postlearn_eval
```

## Notes for paper writing

The most defensible interpretation of the current experiments is:
- several small/mini instruct models benefit substantially from lesson-conditioned evaluation
- very small models such as TinyLlama may fail to benefit from direct lesson injection and can even degrade under long-context instructional prompting
- results for Qwen 1.5B should be cited using the repaired pipeline only

For quick citation-ready numbers, see:
- `results_summary.csv`
- `results_summary.md`
