# `src/` — the thesis experiment framework

Everything here exists to answer the seven research questions by running
fine-tuning experiments. The design goal: **one experiment = one YAML file**, so
the ~30 runs behind the thesis are config changes, not code forks.

## The one idea: four pluggable axes

Every experiment is a choice on four independent axes. Each axis is a **registry**
(a name → class table) so a YAML string selects the implementation:

```
        backbone        ×      dataset       ×     objective      ×     metric
      (src/models)           (src/data)         (src/objectives)     (src/evaluation)
   ┌──────────────┐     ┌──────────────┐     ┌────────────────┐   ┌──────────────┐
   │ blip2        │     │ scienceqa    │     │ generative     │   │ mc_accuracy  │
   │ qwen2_vl     │     │ chartqa      │     │ explanation_   │   │ exact_match  │
   │ paligemma    │     │ docvqa       │     │   aware        │   │ anls         │
   │              │     │ aokvqa       │     │ contrastive    │   │ relaxed_acc  │
   │              │     │ vqav2        │     │                │   │ rouge_l/bleu │
   └──────────────┘     └──────────────┘     └────────────────┘   └──────────────┘
```

```yaml
model:     { name: qwen2_vl }      # ← picks the backbone class
data:      { name: scienceqa }     # ← picks the dataset loader
objective: { name: explanation_aware, alpha: 0.5 }
eval:      { metrics: [mc_accuracy, rouge_l] }
```

Add a new option anywhere = **one new file + one `@REGISTRY.register("name")`
decorator**. Nothing else changes — the trainer, evaluator and runner never name
a concrete class.

## How a run flows

```
  configs/experiments/foo.yaml
        │  load_config()                      src/config
        ▼
  ExperimentConfig (typed dataclasses)
        │  ExperimentRunner.run()             src/experiment
        ▼
  ┌─ build_model(cfg.model) ───────► wrapper  (4-bit base + LoRA + processor)
  ├─ build_dataset(cfg.data) ──────► VLMExample stream
  ├─ build_objective(cfg.objective)► loss fn
  ├─ build_collator(...) ──────────► masked, span-tagged batches
  ├─ build_metrics(cfg.eval) ──────► [metric…]
  │
  ├─ Evaluator.evaluate()  ── baseline (before training)
  ├─ Trainer.train()       ── fine-tune (grad-accum, callbacks, final eval)
  └─ write outputs/<name>/{config.resolved.yaml, results.json, checkpoint}
```

## File map

| Path | What lives there |
|------|------------------|
| `registry.py` | the generic `Registry` every axis uses |
| `config/` | `ExperimentConfig` dataclasses + YAML loader (`load_config`) |
| `data/` | `VLMExample`, prompt templates, the masking **collator**, dataset loaders |
| `models/` | backbone wrappers, 4-bit quantization, QLoRA application |
| `objectives/` | the losses — generative, explanation-aware, contrastive |
| `training/` | `Trainer` loop, optimizer/scheduler, callbacks, checkpointing |
| `evaluation/` | scoring, metrics, the `Evaluator`, faithfulness (RQ5) |
| `experiment/` | `ExperimentRunner` — config → finished run |
| `utils/` | seed, logging, device/dtype, output paths |
| `run.py` | the CLI: `python -m src.run --config <yaml>` |

## Run it

```bash
# list everything registered
python -m src.run --list

# run an experiment
python -m src.run --config configs/experiments/rq2_blip2_generative_scienceqa.yaml

# sweep without editing files (α arm of RQ3)
python -m src.run --config configs/experiments/rq3_blip2_explanation_aware_scienceqa.yaml \
                  --set objective.alpha=1.0
```

## What's proven vs. scaffolded

Honesty matters for a thesis — not everything is equally battle-tested:

| Component | Status |
|-----------|--------|
| BLIP-2 backbone, generative + explanation-aware, ScienceQA, mc_accuracy | **Proven** (ported from the working prototype) |
| Config / registry / collator / trainer / evaluator plumbing | **Tested** (`tests/test_core.py`, CPU) |
| Qwen2-VL & PaliGemma backbones | **Implemented**, marked `# VERIFY` — decode one label batch on first run to confirm image-token masking (see each wrapper's docstring) |
| Open-ended datasets (ChartQA/DocVQA/VQAv2/A-OKVQA) | **Implemented** with documented HF schemas — verify column names on first load |
| `contrastive` objective | **Implemented for BLIP-2** with Q-Former image features + answer-token projection; run the Colab preflight before training |
| Faithfulness: `masking_consistency` | **Runs today** (model-agnostic) |

Extension recipes live in each subpackage's `README.md`.
