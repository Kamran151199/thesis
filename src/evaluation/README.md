# `src/evaluation/` — score a trained model

## Mental model

Two phases, cleanly separated so you can add a metric without re-running the
model:

```
PREDICT   each example → Prediction        (run the model once)
SCORE     [Prediction…] → {metric: value}  (cheap; reuse predictions)
```

## Prediction depends on task type

- **Multiple-choice** (ScienceQA, A-OKVQA) → **generate-then-score**: the model
  writes its reasoning, then each option is likelihood-scored in that context;
  argmax wins. This replaced the gameable string-match eval that gave a bogus
  100% baseline.
- **Open-ended** (ChartQA, DocVQA, VQAv2) → generate the answer text, compare
  with EM / ANLS / relaxed accuracy.

The generated reasoning is kept on each `Prediction`, so explanation metrics
(ROUGE-L, BLEU) and faithfulness reuse it for free.

## Files

| File | Role |
|------|------|
| `scoring.py` | the proven primitives: `score_continuation` (likelihood) + `generate_continuation` + `split_reasoning_answer` |
| `base.py` | `Prediction` + `BaseMetric` (`applies_to` ∈ mc/open/explanation) |
| `evaluator.py` | runs predict→score; skips metrics that don't fit the dataset; adds the random baseline |
| `metrics/accuracy.py` | `mc_accuracy`, `exact_match`, `anls` (DocVQA), `relaxed_accuracy` (ChartQA) |
| `metrics/generation.py` | `rouge_l`, `bleu` — generated reasoning vs gold rationale (RQ3) |
| `metrics/retrieval.py` | `retrieval_recall` (R@K, MRR) for RQ2's contrastive eval |
| `faithfulness/` | RQ5 — see its own README |

## Add a metric

Subclass `BaseMetric`, set `name` + `applies_to`, implement
`compute(predictions) → {name: value}`, `@METRICS.register`, add to
`metrics/__init__.py`. Reference it from a config's `eval.metrics` list.
