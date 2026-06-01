# `notebooks/` — research, riding on top of `src/`

These are the **thin, interactive layer**. They do the work that doesn't belong
in a config-and-run pipeline: verifying a new backbone, eyeballing what the model
generates, probing faithfulness, turning results into figures. The rule:

> **Notebooks CALL `src`. They never re-implement it.**
> `from src.models import build_model` — not a fresh training loop.

So the repeatable 80% (training + eval) stays locked in the tested framework, and
the inherently-interactive 20% (exploration, debugging, plotting) lives here.

## Convention

- **`.py` files with `# %%` cell markers** (same as `exploration/` and `colab/`),
  not `.ipynb`. Runs as cells in VS Code, and over the VS Code ↔ Colab bridge.
- Numbered by workflow stage, e.g. `01_run_experiment.py`.

## Running

| Where | How |
|-------|-----|
| **Colab (real models)** | run `colab/setup_colab.py` once, then execute cells — the A100 has the GPU these VLMs need |
| **Local** | the framework imports fine, but the real backbones (BLIP-2, Qwen2-VL) need a GPU; locally, prefer `tests/test_core.py` for logic checks |

## Files

| Notebook | What it shows |
|----------|---------------|
| `01_run_experiment.py` | a full research cycle on the framework: load config → build the four axes → **verify the masking** → baseline → fine-tune → measure the lift → read the reasoning → faithfulness probe. The template for every experiment. |

*(Add more as you go — e.g. `02_analysis_figures.py` to turn `outputs/*/results.json` into the thesis figures, `03_faithfulness_deep_dive.py` for the RQ5 masking/attention study. Each stays a thin caller into `src`.)*

## The one-liner vs. the granular flow

- **Production** (run the whole thing): `python -m src.run --config <yaml>` — or
  `ExperimentRunner(load_config(...)).run()` in one cell.
- **Research** (see and poke each step): the granular `build_model` / `Trainer` /
  `Evaluator` flow in `01_run_experiment.py`. Same pieces, opened up.
