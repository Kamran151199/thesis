# `configs/` — one YAML per experiment

Each file in `experiments/` is a complete experiment. Keys you omit fall back to
the defaults in `src/config/schema.py` (which already encode the proposal's
standard hyperparameters: AdamW β=(0.9, 0.999), wd 0.01, lr 2e-4 cosine, etc.),
so a file lists only what makes that run distinctive.

## Naming convention

```
rq<N>_<backbone>_<objective>_<dataset>.yaml
└─ which research question this run serves (provenance, not decoration)
```

e.g. `rq3_qwen2vl_explanation_aware_scienceqa.yaml`.

## The provided configs (a representative slice, not all ~30)

| File | RQ | Backbone | Objective | Dataset | Notes |
|------|----|----------|-----------|---------|-------|
| `rq2_blip2_generative_scienceqa` | 2 | BLIP-2 | generative | ScienceQA | **the proven run** (full Q-Former, no LoRA) |
| `rq2_blip2_contrastive_scienceqa` | 2 | BLIP-2 | contrastive | ScienceQA | generative loss + Q-Former/answer InfoNCE; sets `model.contrastive_projection=true` |
| `rq3_blip2_explanation_aware_scienceqa` | 3 | BLIP-2 | explanation-aware | ScienceQA | α=0.5; sweep α for the RQ3 curve |
| `rq2_qwen2vl_generative_scienceqa` | 2 | Qwen2-VL-2B | generative | ScienceQA | QLoRA baseline |
| `rq2_qwen2vl_generative_aokvqa` | 2/3 | Qwen2-VL-2B | generative | A-OKVQA | natural-image rationale control |
| `rq3_qwen2vl_explanation_aware_scienceqa` | 3 | Qwen2-VL-2B | explanation-aware | ScienceQA | **flagship** (primary backbone + core loss) |
| `rq3_qwen2vl_explanation_aware_aokvqa` | 3/4 | Qwen2-VL-2B | explanation-aware | A-OKVQA | second gold-rationale domain |
| `rq4_qwen2vl_explanation_aware_chartqa` | 4 | Qwen2-VL-2B | explanation-aware | ChartQA | chart domain (open-ended, relaxed acc) |
| `rq4_qwen2vl_explanation_aware_docvqa` | 4/7 | Qwen2-VL-2B | answer-only fallback | DocVQA | document domain (ANLS) |
| `rq4_qwen2vl_explanation_aware_vqav2` | 4/7 | Qwen2-VL-2B | answer-only fallback | VQAv2 subset | natural-image control; defaults to `Erland/VQAv2-sample` for Colab feasibility |
| `rq6_qwen2vl_7b_explanation_aware_scienceqa` | 6 | Qwen2-VL-7B | explanation-aware | ScienceQA | 2B-vs-7B scale ablation |

## Generating the rest of the grid

Most remaining runs are sweeps over an existing file — no new file needed:

```bash
# RQ3 α-sweep: {explanation-only, balanced, answer-only}
for a in 0.0 0.5 1.0; do
  python -m src.run --config configs/experiments/rq3_qwen2vl_explanation_aware_scienceqa.yaml \
                    --set objective.alpha=$a name=rq3_qwen2vl_alpha${a}_scienceqa
done

# RQ7 transfer: train on one domain, eval on another (swap data.name)
python -m src.run --config <chart config> --set data.split_eval=... data.name=docvqa
```

For a genuinely new combination (new backbone × dataset), copy the closest file
and change the `model.name` / `data.name` strings.

## Sweeping vs. new files

- **Sweep** (`--set k=v …`) for hyperparameter variants of the same experiment —
  α, learning rate, max_train. Override `name=` too so outputs don't collide.
- **New file** when the *story* differs (different RQ, backbone, or dataset) and
  you want it version-controlled and self-documenting.
