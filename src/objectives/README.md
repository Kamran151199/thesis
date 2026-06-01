# `src/objectives/` — the training loss *is* the experiment

RQ2 (contrastive vs generative) and RQ3 (with vs without explanation) are
answered by swapping the objective and **changing nothing else**. Each objective
turns `(wrapper, batch)` into one scalar to backprop.

| Objective | Loss | Answers |
|-----------|------|---------|
| `generative` | next-token CE on the target (model's built-in masked loss) | RQ2 baseline |
| `explanation_aware` | `α·L_answer + (1−α)·L_explanation` | RQ3 (the core contribution) |
| `contrastive` | `L_generative + w·InfoNCE(image, answer)` | RQ2 contrastive arm |

## The explanation-aware loss (RQ3)

```
target:   " Reasoning: plants take in CO2 …   Answer: Carbon dioxide ."
span_ids:    1  1  1  1  1  1  1  1  1          2  2  2  2  2  2
             └────── L_explanation ──────┘     └──── L_answer ────┘
L = α·L_answer + (1−α)·L_explanation
```

`α` is the RQ3 dial: `1.0` = answer-only (≡ generative), `0.0` = explanation-only,
`0.5` = balanced. The span tags come from the collator (`tag_spans=True`, set
automatically because this objective sets `requires_span_ids = True`).

## Files

| File | Role |
|------|------|
| `base.py` | `BaseObjective`, `LossOutput`, and `masked_token_ce` (CE over a position subset) |
| `generative.py` | the baseline loss |
| `explanation_aware.py` | the α-weighted answer/explanation loss |
| `contrastive.py` | generative + InfoNCE; `info_nce()` is the same symmetric loss as the CLIP exploration |

## Add an objective

Subclass `BaseObjective`, implement `compute(wrapper, batch) → LossOutput`, set
`requires_span_ids` if you need answer/explanation tags, `@OBJECTIVES.register`,
add to `__init__.py`. Put per-token logic through `masked_token_ce` so the
autoregressive shift is handled once, correctly.
