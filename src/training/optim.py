"""
src.training.optim — build the AdamW optimizer and the LR schedule.

Defaults follow proposal §7.4: AdamW (β1=0.9, β2=0.999, wd=0.01), learning rate
with a cosine decay and a short warm-up.

THE WARM-UP + COSINE SHAPE
--------------------------
::

    lr
     │      ╭──────╮___
     │     ╱            ‾‾‾───╮___
     │    ╱                       ‾‾──╮____
     │   ╱  warmup                          ‾‾──╮___
     └──┴───────────────────────────────────────────▶ step
        3%          cosine decay to ~0

Warm-up stops the first few (large, noisy) updates from blowing up freshly
initialized LoRA adapters; cosine decay anneals to a near-zero LR so the model
settles into a minimum instead of bouncing around it.
"""

from __future__ import annotations

from torch.optim import AdamW
from transformers import get_scheduler

from src.config.schema import TrainConfig


def build_optimizer(trainable_params, cfg: TrainConfig) -> AdamW:
    """AdamW over the trainable params only (LoRA adapters / unfrozen modules).

    Passing only ``requires_grad`` params is what makes QLoRA cheap: Adam keeps
    ~2 extra state tensors per parameter, so restricting it to the ~1% trainable
    adapters is most of the memory win.
    """
    return AdamW(
        trainable_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
    )


def build_scheduler(optimizer: AdamW, cfg: TrainConfig, num_training_steps: int):
    """LR scheduler with ``warmup_ratio`` warm-up then ``cfg.scheduler`` decay.

    ``num_training_steps`` is the number of *optimizer* steps (after gradient
    accumulation), so the cosine curve lands at ~0 exactly at the end of training.
    """
    warmup_steps = int(cfg.warmup_ratio * num_training_steps)
    return get_scheduler(
        name=cfg.scheduler,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
