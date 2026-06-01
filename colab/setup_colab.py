"""
setup_colab.py — idempotent Colab environment bootstrap.

Run this ONCE at the start of every Colab session (it's safe to re-run).
It assumes the repo is already cloned and that this file is being run from
the repo root. See the "FIRST CELL" snippet below for the one-time clone.

────────────────────────────────────────────────────────────────────────────
FIRST CELL (paste this into a fresh Colab notebook — it's the only thing you
type by hand; everything else lives in the repo):

    import os, subprocess
    from google.colab import userdata, drive
    drive.mount("/content/drive")                    # "already mounted" warning = harmless
    tok = userdata.get("GITHUB_TOKEN")               # 🔑 Colab secret (left sidebar)
    assert tok, "GITHUB_TOKEN secret missing/empty — add it in Colab Secrets"
    REPO = "/content/thesis"
    url = f"https://x-access-token:{tok}@github.com/Kamran151199/thesis.git"
    if os.path.isdir(f"{REPO}/.git"):                # only 'pull' if it's really a git repo
        subprocess.run(["git", "-C", REPO, "pull", "--ff-only"], check=True)
    else:
        subprocess.run(["rm", "-rf", REPO])          # clear any partial dir from a failed try
        subprocess.run(["git", "clone", url, REPO], check=True)   # check=True surfaces auth errors
    os.chdir(REPO)
    exec(open("colab/setup_colab.py").read())        # ← runs THIS file
────────────────────────────────────────────────────────────────────────────

What this does (the professional 3-bucket sync model):
  1. CODE        → git (single source of truth; the first cell already pulled it)
  2. HEAVY INPUTS→ HuggingFace Hub, cached on /content (fast local SSD). Re-fetched
                   each session (~minutes) — the Hub IS the persistent store, and
                   Drive is too slow/small for big model loads.
  3. OUTPUTS     → checkpoints on Drive (small + precious, survive disconnects);
                   metrics on wandb

IMPORTANT: run this BEFORE `import transformers` anywhere, because it sets the
HF cache env vars, which transformers only reads at import time.
"""

import os
import subprocess
import sys
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
# Hybrid cache model: heavy INPUTS (models/datasets) live on fast ephemeral
# /content and are re-fetched from the Hub each session; only small precious
# OUTPUTS (checkpoints/results) go to persistent Drive.
DRIVE_ROOT = "/content/drive/MyDrive/thesis"   # persistent: checkpoints + results ONLY
REPO_DIR = "/content/thesis"                    # code (git), on fast ephemeral SSD
HF_CACHE = "/content/hf_cache"                  # models   — fast SSD, re-fetched per session
HF_DATASETS = "/content/hf_datasets"            # datasets — fast SSD, re-fetched per session


def _run(cmd: list[str]) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ── 1. Drive (idempotent — skip if already mounted) ───────────────────────────
if not os.path.ismount("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
print("✓ Drive mounted")

# ── 2. Editable install so `import src.*` resolves on the remote kernel ────────
# --no-deps: install ONLY our `src` package; do NOT touch Colab's CUDA-matched
# torch/torchvision (reinstalling those can break the GPU build).
print("Installing the `src` package (editable, no-deps)...")
_run([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"])

# ML extras that Colab may not have / may have stale. These are pure-Python or
# CUDA-agnostic, safe to upgrade. torch/torchvision intentionally NOT listed.
# rouge-score + nltk back the explanation-quality metrics (rouge_l, bleu);
# einops is required by some VLM backbones (e.g. Qwen2-VL). All pure-Python.
print("Installing ML extras (transformers, peft, bitsandbytes, ...)...")
_run([sys.executable, "-m", "pip", "install", "-q", "-U",
      "transformers", "peft", "accelerate", "bitsandbytes",
      "datasets", "evaluate", "trl",
      "rouge-score", "nltk", "einops"])

# ── 3. Point HuggingFace caches at /content (fast local SSD) ───────────────────
# Models + datasets are re-fetched from the Hub each session (~minutes). Why not
# Drive: the Hub is already the persistent store, and Drive's FUSE mount is slow
# for big model loads + flaky for a dataset cache's many small files (and may be
# near-full). /content has ~80GB free and loads fast. Only OUTPUTS go to Drive (§4).
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"] = HF_DATASETS
Path(HF_CACHE).mkdir(parents=True, exist_ok=True)
Path(HF_DATASETS).mkdir(parents=True, exist_ok=True)
print(f"✓ HF cache → {HF_CACHE}  (ephemeral SSD; re-fetched each session)")

# ── 4. Checkpoint dir on Drive — the ONLY bucket we persist (small + precious) ──
# Trainable deltas only (LoRA adapter / Q-Former ≈ MBs) + results.json. Survives
# runtime disconnects; tiny, so it fits even a near-full Drive.
CKPT_DIR = f"{DRIVE_ROOT}/checkpoints"
Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)
os.environ["THESIS_CKPT_DIR"] = CKPT_DIR
print(f"✓ checkpoints → {CKPT_DIR}  (Drive, persistent)")

# ── 5. wandb (optional — only if you added the secret) ─────────────────────────
try:
    from google.colab import userdata
    os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
    print("✓ wandb key loaded (online tracking)")
except Exception:
    os.environ["WANDB_MODE"] = "offline"
    print("• no WANDB_API_KEY secret → wandb offline (add the secret to enable)")

# ── 6. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Colab environment ready.")
print(f"  cwd:          {os.getcwd()}")
print(f"  code:         {REPO_DIR}        (git — pull to refresh)")
print(f"  HF cache:     {os.environ['HF_HOME']}  (ephemeral, fast)")
print(f"  checkpoints:  {CKPT_DIR}  (Drive, persistent)")
print(f"  use in code:  os.environ['THESIS_CKPT_DIR']")
print("=" * 60)
