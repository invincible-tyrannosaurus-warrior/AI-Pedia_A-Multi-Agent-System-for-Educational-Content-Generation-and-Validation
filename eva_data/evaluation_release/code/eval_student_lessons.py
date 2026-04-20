from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

FINAL_ANSWER_PATTERNS = [
    re.compile(r"final\s*answer\s*[:：]\s*([ABCD])\b", re.IGNORECASE),
    re.compile(r"answer\s*[:：]\s*([ABCD])\b", re.IGNORECASE),
    re.compile(r"correct\s*option\s*[:：]\s*([ABCD])\b", re.IGNORECASE),
    re.compile(r"option\s*([ABCD])\b", re.IGNORECASE),
]

FALLBACK_LETTER_RE = re.compile(r"\b([ABCD])\b")

SYSTEM_PROMPT = (
    "You are taking a multiple-choice quiz based only on the provided lesson. "
    "Think carefully, then answer with a brief explanation and end with exactly: Final Answer: <A/B/C/D>."
)


@dataclass
class EvalExample:
    topic: str
    lesson_title: str
    question_id: int
    question: str
    options: List[str]
    gold: str
    explanation: str
    lesson_text: str
    source_file: str


class Backend:
    def generate(self, prompt: str, model: str) -> str:
        raise NotImplementedError


class MockBackend(Backend):
    def generate(self, prompt: str, model: str) -> str:
        letters = ["A", "B", "C", "D"]
        seed = abs(hash(prompt)) % (2**32)
        rng = random.Random(seed)
        choice = rng.choice(letters)
        return f"I compared the lesson with the options. Final Answer: {choice}"


class OpenAIBackend(Backend):
    def __init__(self, api_key: Optional[str], base_url: Optional[str], temperature: float, max_tokens: int):
        if OpenAI is None:
            raise RuntimeError("openai package is not installed. Please `pip install -r requirements.txt`.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, model: str) -> str:
        resp = self.client.chat.completions.create(
            model=model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content or ""


class TransformersBackend(Backend):
    def __init__(self, temperature: float, max_tokens: int, load_in_4bit: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.AutoTokenizer = AutoTokenizer
        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.BitsAndBytesConfig = BitsAndBytesConfig
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.load_in_4bit = load_in_4bit
        self._model = None
        self._tokenizer = None
        self._loaded_model_name = None

    def _ensure_loaded(self, model: str) -> None:
        if self._model is not None and self._loaded_model_name == model:
            return
        self._tokenizer = self.AutoTokenizer.from_pretrained(model)
        load_kwargs = {"device_map": "auto"}
        if self.load_in_4bit:
            load_kwargs["quantization_config"] = self.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            load_kwargs["dtype"] = self.torch.float16
        self._model = self.AutoModelForCausalLM.from_pretrained(model, **load_kwargs)
        self._loaded_model_name = model

    def _terminators(self, tokenizer) -> List[int] | int | None:
        eos_ids: List[int] = []
        if tokenizer.eos_token_id is not None:
            eos_ids.append(tokenizer.eos_token_id)
        for token in ["<|im_end|>", "<|end_of_text|>"]:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0 and token_id not in eos_ids:
                eos_ids.append(token_id)
        if not eos_ids:
            return None
        return eos_ids if len(eos_ids) > 1 else eos_ids[0]

    def generate(self, prompt: str, model: str) -> str:
        self._ensure_loaded(model)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        tokenizer = self._tokenizer
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([rendered], return_tensors="pt", padding=True).to(self._model.device)
        do_sample = self.temperature > 0
        generate_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_tokens,
            do_sample=do_sample,
            pad_token_id=(tokenizer.pad_token_id or tokenizer.eos_token_id),
            eos_token_id=self._terminators(tokenizer),
            repetition_penalty=1.05,
        )
        if do_sample:
            generate_kwargs["temperature"] = self.temperature
        with self.torch.no_grad():
            outputs = self._model.generate(**generate_kwargs)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_manifest(data_dir: Path) -> List[Dict[str, Any]]:
    manifest_path = data_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_examples(data_dir: Path) -> List[EvalExample]:
    manifest = load_manifest(data_dir)
    examples: List[EvalExample] = []
    for item in manifest:
        raw_output = item["output"]
        source = Path(raw_output)
        if not source.exists():
            normalized = raw_output.replace("\\", "/")
            marker = "/student_lessons/"
            if marker in normalized:
                rel_part = normalized.split(marker, 1)[1]
                source = data_dir / Path(rel_part)
            else:
                source = data_dir / Path(source.name)
        if not source.exists():
            raise FileNotFoundError(f"Could not resolve dataset file from manifest entry: {raw_output}")
        with source.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for q in payload["questions"]:
            examples.append(
                EvalExample(
                    topic=payload["topic"],
                    lesson_title=payload["lesson_title"],
                    question_id=q["id"],
                    question=q["question"],
                    options=q["options"],
                    gold=q["answer"].strip().upper(),
                    explanation=q.get("explanation", ""),
                    lesson_text=payload["lesson_text"],
                    source_file=str(source),
                )
            )
    return examples


def build_prompt(ex: EvalExample, include_lesson: bool = True) -> str:
    option_lines = []
    for idx, opt in enumerate(ex.options):
        letter = chr(ord("A") + idx)
        option_lines.append(f"{letter}. {opt}")
    joined_options = "\n".join(option_lines)
    parts = [
        f"Lesson Title: {ex.lesson_title}",
        f"Topic: {ex.topic}",
        "",
    ]
    if include_lesson:
        parts.extend([
            "Lesson Content:",
            ex.lesson_text,
            "",
        ])
    parts.extend([
        f"Question {ex.question_id}: {ex.question}",
        "Options:",
        joined_options,
        "",
    ])
    if include_lesson:
        parts.append("Answer the question using the lesson only. If the lesson is flawed, still choose the best supported option from the lesson.")
    else:
        parts.append("Answer the multiple-choice question using your existing knowledge only. Do not assume access to the lesson text.")
    return "\n".join(parts)


def parse_answer_letter(text: str) -> Optional[str]:
    if not text:
        return None
    for pattern in FINAL_ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return matches[-1].upper()
    fallback = FALLBACK_LETTER_RE.findall(text.upper())
    return fallback[-1].upper() if fallback else None


def make_backend(name: str, args: argparse.Namespace) -> Backend:
    if name == "mock":
        return MockBackend()
    if name == "openai":
        return OpenAIBackend(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    if name == "transformers":
        return TransformersBackend(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            load_in_4bit=not args.no_4bit,
        )
    raise ValueError(f"Unsupported backend: {name}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
        "backend": rows[0]["backend"] if rows else None,
        "model": rows[0]["model"] if rows else None,
    }


def summarize_by_topic(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["topic"], []).append(row)
    out: Dict[str, Any] = {}
    for topic, topic_rows in grouped.items():
        total = len(topic_rows)
        correct = sum(1 for r in topic_rows if r["is_correct"])
        out[topic] = {
            "total_questions": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
        }
    return out


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate student lessons MCQ data")
    p.add_argument("--data-dir", required=True, help="Path to student_lessons directory")
    p.add_argument("--backend", choices=["mock", "openai", "transformers"], default="mock")
    p.add_argument("--model", default="mock-model")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading for transformers backend")
    p.add_argument("--mode", choices=["postlearn", "prelearn"], default="postlearn")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    examples = load_examples(data_dir)
    if args.limit is not None:
        examples = examples[: args.limit]

    backend = make_backend(args.backend, args)
    rows: List[Dict[str, Any]] = []

    include_lesson = args.mode == "postlearn"

    for idx, ex in enumerate(examples, start=1):
        prompt = build_prompt(ex, include_lesson=include_lesson)
        started = time.time()
        raw_output = backend.generate(prompt, args.model)
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
            "latency_sec": round(latency, 4),
            "backend": args.backend,
            "model": args.model,
            "source_file": ex.source_file,
            "mode": args.mode,
        }
        rows.append(row)
        print(f"[{idx}/{len(examples)}] {ex.topic} Q{ex.question_id} -> pred={pred} gold={ex.gold} correct={row['is_correct']}")

    summary = summarize(rows)
    topic_summary = summarize_by_topic(rows)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "topic_summary.json", topic_summary)
    write_jsonl(output_dir / "predictions.jsonl", rows)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
