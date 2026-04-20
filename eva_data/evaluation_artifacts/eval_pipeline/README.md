# Student Lessons Evaluation Pipeline

A small, inspectable evaluation pipeline for the `student_lessons` dataset.

## What it does

- Loads the dataset from `manifest.json`
- Builds a prompt from either:
  - lesson content + one multiple-choice question (`postlearn`)
  - question + options only (`prelearn`)
- Calls a model backend
- Extracts a final answer letter (`A`/`B`/`C`/`D`)
- Computes per-topic and overall accuracy
- Writes detailed outputs for review and later server-side runs

## Supported backends

### 1) `mock`
No model call. Returns deterministic placeholder answers so the whole pipeline can be tested locally.

### 2) `openai`
Uses an OpenAI-compatible chat-completions API.
This is useful once the server exposes a model endpoint.

Environment variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (optional for OpenAI-compatible servers)

### 3) `transformers`
Loads a Hugging Face causal LM directly on the machine running the script.
This is the fastest path when the server already has a usable local model environment.

## Expected dataset layout

```text
student_lessons/
  manifest.json
  Topic_A/student_input.json
  Topic_B/student_input.json
  ...
```

## Quick start

### Dry run with mock backend

```bash
python eval_student_lessons.py \
  --data-dir "C:/Users/Yue/Desktop/eva_data/student_lessons" \
  --backend mock \
  --output-dir outputs/mock_run
```

### Real run against an OpenAI-compatible endpoint

```bash
set OPENAI_API_KEY=your-key
set OPENAI_BASE_URL=http://127.0.0.1:8000/v1
python eval_student_lessons.py \
  --data-dir "C:/Users/Yue/Desktop/eva_data/student_lessons" \
  --backend openai \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output-dir outputs/qwen_eval
```

## Output files

- `summary.json` — overall metrics
- `predictions.jsonl` — one line per question with raw output and parsed answer
- `topic_summary.json` — per-topic accuracy

## Notes

- The parser prefers explicit final-answer patterns like `Final Answer: B`.
- If not found, it falls back to the last standalone option letter in the response.
- For best reliability, use an instruction-tuned model and keep temperature low.
