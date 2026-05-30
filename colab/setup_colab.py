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
    drive.mount("/content/drive")
    tok = userdata.get("GITHUB_TOKEN")                      # 🔑 Colab secret
    url = f"https://{tok}@github.com/Kamran151199/thesis.git"
    if os.path.exists("/content/thesis"):
        subprocess.run(["git", "-C", "/content/thesis", "pull", "--ff-only"])
    else:
        subprocess.run(["git", "clone", url, "/content/thesis"])
    os.chdir("/content/thesis")
    exec(open("colab/setup_colab.py").read())               # ← runs THIS file
────────────────────────────────────────────────────────────────────────────

What this does (the professional 3-bucket sync model):
  1. CODE        → git (single source of truth; the first cell already pulled it)
  2. HEAVY INPUTS→ HuggingFace Hub, cached on Drive (models + datasets persist
                   across sessions, so you don't re-download 4GB every time)
  3. OUTPUTS     → checkpoints on Drive (survive disconnects); metrics on wandb

IMPORTANT: run this BEFORE `import transformers` anywhere, because it sets the
HF cache env vars, which transformers only reads at import time.
"""

import os
import subprocess
import sys
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
DRIVE_ROOT = "/content/drive/MyDrive/thesis"   # persistent storage on your Drive
REPO_DIR = "/content/thesis"                    # code lives on fast ephemeral SSD


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
print("Installing ML extras (transformers, peft, bitsandbytes, ...)...")
_run([sys.executable, "-m", "pip", "install", "-q", "-U",
      "transformers", "peft", "accelerate", "bitsandbytes",
      "datasets", "evaluate", "trl"])

# ── 3. Point HuggingFace caches at Drive (persist across sessions) ─────────────
# First run still downloads; every run after reuses the Drive copy → no 4GB
# re-download. If you find Drive *loading* slow for big models, flip HF_HOME
# back to /content (fast SSD, but re-downloads each session).
os.environ["HF_HOME"] = f"{DRIVE_ROOT}/hf_cache"
os.environ["HF_DATASETS_CACHE"] = f"{DRIVE_ROOT}/hf_datasets"
Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)
print(f"✓ HF cache → {os.environ['HF_HOME']}")

# ── 4. Checkpoint dir on Drive (survives runtime disconnect) ───────────────────
CKPT_DIR = f"{DRIVE_ROOT}/checkpoints"
Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)
os.environ["THESIS_CKPT_DIR"] = CKPT_DIR
print(f"✓ checkpoints → {CKPT_DIR}")

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
print(f"  HF cache:     {os.environ['HF_HOME']}")
print(f"  checkpoints:  {CKPT_DIR}")
print(f"  use in code:  os.environ['THESIS_CKPT_DIR']")
print("=" * 60)
