"""Where outputs go — and the local↔Colab bridge that makes it survive disconnects.

The "3-bucket sync model" from ``colab/setup_colab.py``::

    CODE          → git                (this repo)
    HEAVY INPUTS  → HuggingFace cache   (models + datasets, cached on Drive)
    OUTPUTS       → checkpoints + logs  (Drive, so a runtime disconnect can't
                                         vaporize a 40-minute fine-tune)

``setup_colab.py`` exports ``THESIS_CKPT_DIR`` pointing at Google Drive. Locally
that env var is unset, so we fall back to ``<repo>/outputs``. Either way code
just calls ``output_root()`` and never hard-codes ``/content/drive/...`` —
the same script runs unchanged on your laptop and on the A100.
"""

from __future__ import annotations

import os
from pathlib import Path

# repo root = three levels up from this file: src/utils/paths.py → repo/
REPO_ROOT = Path(__file__).resolve().parents[2]


def output_root() -> Path:
    """Root for all run outputs.

    ``$THESIS_CKPT_DIR`` if set (Colab → Drive), else ``<repo>/outputs`` (local).
    """
    env = os.environ.get("THESIS_CKPT_DIR")
    root = Path(env) if env else REPO_ROOT / "outputs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def experiment_dir(experiment_name: str) -> Path:
    """Per-experiment output folder, created on demand.

    >>> experiment_dir("rq2_blip2_generative_scienceqa")
    PosixPath('.../outputs/rq2_blip2_generative_scienceqa')

    Holds the checkpoint, the resolved config, metrics JSON and any figures —
    one self-contained, defensible folder per run.
    """
    d = output_root() / experiment_name
    d.mkdir(parents=True, exist_ok=True)
    return d
