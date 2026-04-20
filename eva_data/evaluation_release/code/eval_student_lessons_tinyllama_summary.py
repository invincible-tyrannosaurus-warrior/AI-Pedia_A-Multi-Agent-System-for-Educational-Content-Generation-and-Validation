from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

FINAL_ANSWER_PATTERNS = [
    re.compile(r"final\s*answer\s*[:：]\s*([ABCD])\b", re.IGNORECASE),
    re.compile(r"answer\s*[:：]\s*([ABCD])\b", re.IGNORECASE),
    re.compile(r"correct\s*option\s*[:：]\s*([ABCD])\b", re.IGNORECASE),
    re.compile(r"option\s*([ABCD])\b", re.IGNORECASE),
]
FALLBACK_LETTER_RE = re.compile(r"\b([ABCD])\b")
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@dataclass
class EvalExample:
    topic: str
    lesson_title: str
    question_id: int
    question: str
    options: List[str]
    gold: str
    lesson_text: str
    source_file: str


def resolve_source(data_dir: Path, raw_output: str) -> Path:
    source = Path(raw_output)
    if source.exists():
        return source
    normalized = raw_output.replace("\\", "/")
    marker = "/student_lessons/"
    if marker in normalized:
        rel_part = normalized.split(marker, 1)[1]
        source = data_dir / Path(rel_part)
    else:
        source = data_dir / Path(Path(source.name))
    if not source.exists():
        raise FileNotFoundError(f"Could not resolve dataset file from manifest entry: {raw_output}")
    return source


def load_examples(data_dir: Path) -> List[EvalExample]:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    examples: List[EvalExample] = []
    for item in manifest:
        source = resolve_source(data_dir, item["output"])
        payload = json.loads(source.read_text(encoding="utf-8"))
        for q in payload["questions"]:
            examples.append(EvalExample(
                topic=payload["topic"],
                lesson_title=payload["lesson_title"],
                question_id=q["id"],
                question=q["question"],
                options=q["options"],
                gold=q["answer"].strip().upper(),
                lesson_text=payload["lesson_text"],
                source_file=str(source),
            ))
    return examples


def compress_lesson(text: str, max_chars: int = 700) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    kept: List[str] = []
    for ln in lines:
        if ln.startswith("Title:") or ln.startswith("Subtitle:") or ln.startswith("Content:") or ln.startswith("Formula:"):
            kept.append(ln)
    if not kept:
        kept = lines
    summary = []
    for ln in kept:
        ln = re.sub(r"^Title:\s*", "", ln)
        ln = re.sub(r"^Subtitle:\s*", "", ln)
        ln = re.sub(r"^Content:\s*", "", ln)
        summary.append(ln)
        joined = " | ".join(summary)
        if len(joined) >= max_chars:
            return joined[:max_chars].rsplit(" ", 1)[0]
    return " | ".join(summary)[:max_chars]


def build_prompt(ex: EvalExample, summary_text: str) -> str:
    option_lines = []
    for idx, opt in enumerate(ex.options):
        option_lines.append(f"{chr(ord('A') + idx)}. {opt}")
    return (
        f"Topic: {ex.topic}\n"
        f"Lesson summary:\n{summary_text}\n\n"
        f"Question {ex.question_id}: {ex.question}\n"
        f"Options:\n" + "\n".join(option_lines) + "\n\n"
        "Use only the short lesson summary above. Choose the best answer. End with exactly: Final Answer: <A/B/C/D>."
    )


def parse_answer_letter(text: str) -> Optional[str]:
    if not text:
        return None
    for pattern in FINAL_ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return matches[-1].upper()
    fallback = FALLBACK_LETTER_RE.findall(text.upper())
    return fallback[-1].upper() if fallback else None


def load_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb,
        device_map="auto",
        attn_implementation="eager",
    )
    return torch, tok, model


def generate(torch_mod, tok, model, prompt: str) -> str:
    full_prompt = (
        "<|system|>\n"
        "You are taking a multiple-choice quiz. Keep your reasoning short. End with exactly: Final Answer: <A/B/C/D>.\n"
        "<|user|>\n" + prompt + "\n<|assistant|>\n"
    )
    inputs = tok(full_prompt, return_tensors="pt").to("cuda")
    with torch_mod.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    correct = sum(1 for r in rows if r["is_correct"])
    parsed = sum(1 for r in rows if r["predicted_answer"] is not None)
    latencies = [r["latency_sec"] for r in rows]
    return {
        "total_questions": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "parsed_answer_count": parsed,
        "parsed_answer_rate": (parsed / total) if total else 0.0,
        "avg_latency_sec": statistics.mean(latencies) if latencies else 0.0,
        "backend": "transformers-summary",
        "model": MODEL_NAME,
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = load_examples(data_dir)
    if args.limit is not None:
        examples = examples[:args.limit]
    torch_mod, tok, model = load_model()
    rows: List[Dict[str, Any]] = []
    for idx, ex in enumerate(examples, start=1):
        summary_text = compress_lesson(ex.lesson_text)
        prompt = build_prompt(ex, summary_text)
        started = time.time()
        raw_output = generate(torch_mod, tok, model, prompt)
        latency = time.time() - started
        pred = parse_answer_letter(raw_output)
        row = {
            "index": idx,
            "topic": ex.topic,
            "lesson_title": ex.lesson_title,
            "question_id": ex.question_id,
            "question": ex.question,
            "options": ex.options,
            "gold_answer": ex.gold,
            "predicted_answer": pred,
            "is_correct": pred == ex.gold,
            "raw_output": raw_output,
            "lesson_summary": summary_text,
            "latency_sec": round(latency, 4),
            "backend": "transformers-summary",
            "model": MODEL_NAME,
            "source_file": ex.source_file,
            "mode": "postlearn_summary",
        }
        rows.append(row)
        print(f"[{idx}/{len(examples)}] {ex.topic} Q{ex.question_id} -> pred={pred} gold={ex.gold} correct={row['is_correct']}")
    write_json(out_dir / "summary.json", summarize(rows))
    write_jsonl(out_dir / "predictions.jsonl", rows)
    print("\n=== SUMMARY ===")
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
