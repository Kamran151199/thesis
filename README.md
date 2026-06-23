# Reasoning and Grounded Explanation Generation Using Vision-Language Models

Bachelor thesis repository for a controlled study of resource-constrained
vision-language model fine-tuning. The project compares answer-only,
rationale-generative, explanation-aware, BLIP-2 generative, and BLIP-2
contrastive-enhanced supervision while keeping the model/data/evaluation path as
fixed as possible inside each comparison group.

The repository has two jobs:

1. run the experiments and regenerate the paper artifacts; and
2. keep the final thesis and journal-style paper sources buildable.

The finished thesis source of truth is `clean-docs/`.

For citation and reuse, see `CITATION.cff` and `REPRODUCIBILITY.md`.
Journal availability wording and release metadata are kept in `docs/`.

## Repository Map

| Path | Purpose |
| --- | --- |
| `src/` | Reusable experiment framework: data loaders, model wrappers, objectives, training, evaluation, masking diagnostics, and run orchestration. |
| `configs/experiments/` | YAML experiment declarations. Each run fixes the backbone, dataset, objective, adapter settings, caps, and evaluation settings. |
| `notebooks/00_paper_pipeline.py` | Main Colab pipeline. It trains/resumes runs, audits inputs, builds tables/figures, validates artifacts, and writes the final paper artifact bundle. |
| `notebooks/00_paper_pipeline.ipynb` | Notebook version of the same pipeline for Colab execution. |
| `colab/setup_colab.py` | Idempotent Colab bootstrap. It installs this repo editable, mounts Drive, sets HuggingFace caches, and configures the persistent checkpoint directory. |
| `tests/` | CPU-safe unit tests for core framework behavior. These do not download the real VLM weights. |
| `clean-docs/` | Final thesis and journal-paper LaTeX sources, bibliography, submission figures, and UE template files. |
| `exploration/` | Learning/prototype scripts from the thesis preparation phase. Useful for understanding, not required to reproduce final tables. |
| `notes/` | Concept notes and video transcripts used during study. |

Generated checkpoints, model outputs, logs, Drive downloads, and rendered PDF
review images are intentionally not source files. Keep them outside git unless
you are making a separate artifact release.

## What The Experiments Cover

- BLIP-2 generative vs BLIP-2 contrastive-enhanced ScienceQA fine-tuning.
- Qwen2-VL answer-only, rationale-generative, explanation-aware, fixed-alpha,
  and length-aware controls on rationale-bearing datasets.
- Answer-only fallback runs for ChartQA, DocVQA, and the VQAv2 subset.
- Evidence-masking drift diagnostics and saved masking examples.
- Qwen2-VL-2B vs Qwen2-VL-7B scale comparison under the selected
  explanation-aware setup.
- Cross-domain transfer evaluation across the completed adapters.

The study is a controlled single-GPU protocol, not a leaderboard submission.
Most runs use capped subsets and one seed, so the repo is designed to make the
claim-to-evidence chain inspectable rather than to claim state-of-the-art
numbers.

## Local Setup

Use Python 3.11. The project uses `uv`, but a normal editable install also works.

```bash
uv sync
uv run python tests/test_core.py
uv run python -m src.run --list
```

The local machine is mainly for code checks and LaTeX work. Real Qwen2-VL and
BLIP-2 training/evaluation should be run on a CUDA GPU.

## Colab Reproduction

The easiest way to reproduce the research artifacts is Colab Pro/Pro+ with an
A100 when available. An L4 can run many 2B jobs, but the 7B run is much safer on
an A100.

Paste this into the first Colab cell:

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

Then set the pipeline mode before running the notebook:

```python
import os

os.environ["PAPER_FORCE_RETRAIN_STALE"] = "0"
os.environ["PAPER_EVAL_REFRESH_STALE"] = "1"
os.environ["PAPER_STRICT_FINAL"] = "1"
```

Run `notebooks/00_paper_pipeline.ipynb` top-to-bottom. The pipeline writes
persistent outputs to:

```text
/content/drive/MyDrive/thesis/checkpoints
/content/drive/MyDrive/thesis/paper_artifacts
```

The last validation cell fails loudly if expected runs, retrieval diagnostics,
figures, tables, claim-to-evidence files, or final manifest files are missing.
That failure is intentional; do not ignore it for final results.

## Running A Single Experiment

For a direct run outside the full paper pipeline:

```bash
python -m src.run --config configs/experiments/rq2_blip2_generative_scienceqa.yaml
```

To run only selected paper-pipeline jobs in Colab, set `PAPER_RUNS` before the
pipeline run:

```python
import os

os.environ["PAPER_RUNS"] = "rq6_qwen2vl_7b_explanation_aware_scienceqa"
os.environ["PAPER_EVAL_REFRESH_STALE"] = "1"
os.environ["PAPER_FORCE_RETRAIN_STALE"] = "0"
os.environ["PAPER_STRICT_FINAL"] = "0"  # use 1 only for the full final pipeline
```

## Final Documents

`clean-docs/` is the clean submission folder.

```text
clean-docs/
  thesis-paper.tex       # UE thesis template entry point
  journal-paper.tex      # journal-style paper source
  chapters/              # chapter files required by the thesis template
  misc/                  # title page, abstracts, declaration, glossary, annex
  bib/manual.bib         # thesis bibliography
  Bibliography.bib       # journal-paper bibliography
  Figures/               # final PDF figures used by both documents
  img/ue_logo.png        # UE logo used by the title page
```

Build from inside `clean-docs/`:

```bash
latexmk -pdf thesis-paper.tex
latexmk -pdf journal-paper.tex
```

If `latexmk` is not installed locally, upload the contents of `clean-docs/` to
Overleaf and set `thesis-paper.tex` as the main file.

## Citation And Availability

The recommended repository citation metadata is in `CITATION.cff`.

For the journal paper, the availability text is in:

```text
docs/JOURNAL_AVAILABILITY.md
```

For a stable public release, use:

```text
docs/RELEASE_NOTES_v1.0.0.md
.zenodo.json
```

The repository intentionally stores source code, configs, notebooks, final
figures, and LaTeX sources. Large generated outputs such as checkpoints,
downloaded dataset caches, and full Google Drive artifact bundles are excluded
from git and should be attached only as a separate release artifact if needed.

## What Should Not Be Committed

Do not commit:

- `.venv/`, `.ruff_cache/`, `.pytest_cache/`, `__pycache__/`, or `*.pyc`;
- model checkpoints, LoRA adapters, downloaded model weights, HuggingFace cache,
  `wandb/`, `runs/`, or local `outputs/`;
- Google Drive downloads, `gdrive_thesis/`, `paper_artifacts*.zip`, or
  `drive-download*.zip`;
- rendered PDF review folders under `tmp/`;
- LaTeX build products such as `.aux`, `.bbl`, `.blg`, `.fls`, `.fdb_latexmk`,
  `.log`, `.out`, `.toc`, `.lof`, `.lot`, glossary intermediates, and compiled
  PDFs unless you intentionally attach a final PDF release.

Commit source files, configs, notebooks, tests, final LaTeX sources, final
publication figures, and small audit tables that are needed to understand the
final paper.

## Suggested Final Sanity Checks Before Pushing

```bash
git status --short
uv run python tests/test_core.py
uv run python -m src.run --list
```

For the thesis PDF, also check:

- the PDF builds without undefined references/citations;
- every bibliography item is cited in the body;
- figures and tables render in the expected places;
- `clean-docs/thesis-paper.tex` still follows the UE chapter-based template.
