# Reproducibility Guide

This guide explains how to reproduce the experiment pipeline and paper artifacts
for the thesis repository:

> Reasoning and Grounded Explanation Generation Using Vision-Language Models

The project is designed as a controlled single-GPU study. It is not a
leaderboard reproduction. Most completed runs use capped training/evaluation
subsets and one seed because the goal is to keep the comparison auditable under
a student-level compute budget.

## 1. Hardware

Recommended:

- Google Colab Pro/Pro+ with an A100 GPU for the full final pipeline.
- An L4 can run many Qwen2-VL-2B and BLIP-2 jobs, but the Qwen2-VL-7B run is
  much more reliable on an A100.
- Persistent Google Drive storage for checkpoints and paper artifacts.

Local CPU use is mainly for unit tests, code inspection, and LaTeX builds.
The real VLM experiments require CUDA.

## 2. Clone And Bootstrap In Colab

Add a GitHub token to Colab Secrets as `GITHUB_TOKEN`, then run:

```python
import os, subprocess
from google.colab import userdata, drive

drive.mount("/content/drive")
tok = userdata.get("GITHUB_TOKEN")
assert tok, "Add GITHUB_TOKEN in Colab Secrets first."

REPO = "/content/thesis"
url = f"https://x-access-token:{tok}@github.com/Kamran151199/thesis.git"

if os.path.isdir(f"{REPO}/.git"):
    subprocess.run(["git", "-C", REPO, "pull", "--ff-only"], check=True)
else:
    subprocess.run(["rm", "-rf", REPO], check=True)
    subprocess.run(["git", "clone", url, REPO], check=True)

os.chdir(REPO)
exec(open("colab/setup_colab.py").read())
```

Set the final-pipeline mode:

```python
import os

os.environ["PAPER_FORCE_RETRAIN_STALE"] = "0"
os.environ["PAPER_EVAL_REFRESH_STALE"] = "1"
os.environ["PAPER_STRICT_FINAL"] = "1"
```

Then run `notebooks/00_paper_pipeline.ipynb` from top to bottom.

## 3. Output Locations

The Colab pipeline writes persistent state to Google Drive:

```text
/content/drive/MyDrive/thesis/checkpoints
/content/drive/MyDrive/thesis/paper_artifacts
```

The final paper artifact folder contains:

- run status logs;
- config and code digests;
- split/leakage audits;
- generated figures;
- generated tables;
- masking example artifacts;
- claim-to-evidence maps;
- final validation logs;
- an artifact manifest in CSV and JSON.

The repository mirrors the final paper figures in:

```text
clean-docs/Figures/generated
clean-docs/Figures/icons
```

## 4. Full Final Validation

The last notebook cell writes:

```text
paper_artifacts/logs/final_validation.json
paper_artifacts/data/artifact_manifest.csv
paper_artifacts/data/artifact_manifest.json
```

With `PAPER_STRICT_FINAL=1`, the final cell raises an error if expected runs,
expected figures, expected tables, code digests, stale-result checks, or final
manifest files are missing. Treat that failure as a real incomplete pipeline
state, not as a notebook annoyance.

## 5. Running One Experiment

For a single direct run:

```bash
python -m src.run --config configs/experiments/rq2_blip2_generative_scienceqa.yaml
```

For a selected paper-pipeline run in Colab:

```python
import os

os.environ["PAPER_RUNS"] = "rq6_qwen2vl_7b_explanation_aware_scienceqa"
os.environ["PAPER_EVAL_REFRESH_STALE"] = "1"
os.environ["PAPER_FORCE_RETRAIN_STALE"] = "0"
os.environ["PAPER_STRICT_FINAL"] = "0"
```

Use `PAPER_STRICT_FINAL=1` again only when running the full final pipeline.

## 6. Local Checks

On a local machine with Python 3.11:

```bash
uv sync
uv run python tests/test_core.py
uv run python -m src.run --list
```

These checks do not download the real VLM weights. They verify the code paths
that can be tested safely without a GPU.

## 7. Building The Documents

The clean thesis/journal sources are in `clean-docs/`.

```bash
cd clean-docs
latexmk -pdf thesis-paper.tex
latexmk -pdf journal-paper.tex
```

If local LaTeX is not available, upload `clean-docs/` to Overleaf and set
`thesis-paper.tex` or `journal-paper.tex` as the main file.

## 8. Important Interpretation Notes

- The project uses public datasets cited in the thesis/paper.
- ChartQA, DocVQA, and VQAv2 are used as answer-only fallback and transfer
  domains because gold rationales are not available in this setup.
- Masking drift is an evidence-sensitivity diagnostic. It is not treated as a
  complete proof of faithful rationale generation.
- The Qwen2-VL-7B result is a scale comparison under one selected objective, not
  a full efficiency benchmark across every objective.
- Results should be read as controlled framework results, not public leaderboard
  claims.
