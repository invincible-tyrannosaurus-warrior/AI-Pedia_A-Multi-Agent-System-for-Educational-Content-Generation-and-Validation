# Results Summary

| Model | Pre-learn | Post-learn | Delta |
|---|---:|---:|---:|
| microsoft/Phi-3-mini-4k-instruct | 68.5% | 96.0% | +27.5 pts |
| microsoft/Phi-3.5-mini-instruct | 92.5% | 96.5% | +4.0 pts |
| Qwen/Qwen2.5-3B-Instruct | 75.0% | 96.5% | +21.5 pts |
| Qwen/Qwen2.5-1.5B-Instruct | 79.0% | 98.0% | +19.0 pts |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 27.5% | 9.0% | -18.5 pts |

## Main takeaways

1. Lesson-conditioned evaluation substantially improved Phi-3 mini and both tested Qwen models.
2. Phi-3.5 mini started from a very high closed-book baseline, leaving less room for improvement.
3. TinyLlama did not benefit from lesson injection and instead degraded strongly under post-learn conditions.
4. Qwen 1.5B results should be interpreted using the repaired pipeline; earlier failing outputs came from inference/backend issues rather than inherent inability.

## Caution

Qwen 2.5 3B was evaluated before the final hardening pass of the `transformers` backend, while Qwen 1.5B final results were collected after the backend repair. If strict same-pipeline fairness is needed for publication, consider rerunning Qwen 3B with the repaired backend.
