"""
src.training — the fine-tuning loop and its moving parts.

    from src.training import Trainer, build_optimizer
    from src.training.callbacks import ConsoleCallback, WandbCallback, CheckpointCallback
    from src.training.checkpoint import save_checkpoint, load_checkpoint

The :class:`Trainer` is backbone/dataset/objective-agnostic — it asks the
objective for a loss and fires callbacks for everything else. See the README.
"""

from src.training.callbacks import (
    Callback,
    CheckpointCallback,
    ConsoleCallback,
    WandbCallback,
)
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.optim import build_optimizer, build_scheduler
from src.training.trainer import Trainer

__all__ = [
    "Trainer",
    "build_optimizer",
    "build_scheduler",
    "Callback",
    "ConsoleCallback",
    "WandbCallback",
    "CheckpointCallback",
    "save_checkpoint",
    "load_checkpoint",
]
