"""Reproducibility — one knob to make a run repeatable.

The thesis reports **1 seed per configuration** (compute budget; see the
proposal). That makes deterministic seeding non-negotiable: a result you cannot
reproduce is a result you cannot defend in a viva. ``set_seed(42)`` pins every
RNG that touches a run (Python, NumPy, PyTorch CPU + CUDA).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU + all CUDA devices).

    Parameters
    ----------
    seed:
        The integer to seed every RNG with.
    deterministic:
        If ``True``, also force cuDNN into deterministic mode. This makes
        convolutions bit-for-bit reproducible at a speed cost — leave it
        ``False`` for training (fast), flip it ``True`` only when chasing a
        non-reproducible bug.

    Example
    -------
    >>> set_seed(42)
    >>> torch.randn(2).tolist()        # identical every process restart
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
