"""
src.training.trainer — the loop, with gradient accumulation and callbacks.

Backbone-, dataset-, and objective-agnostic: it asks the OBJECTIVE for a loss
and never looks inside it. That's what lets one loop serve every RQ.

GRADIENT ACCUMULATION (how batch-4 becomes effective-batch-32)
--------------------------------------------------------------
The A100 fits ~4 examples at a time, but the proposal wants an effective batch
of 32. So we forward/backward 8 micro-batches, *summing* their gradients, and
only then take ONE optimizer step::

    micro:  1   2   3   4   5   6   7   8     1   2 …
    grad:   +   +   +   +   +   +   +   +     (accumulating)
    step:   ·   ·   ·   ·   ·   ·   ·  ✓step  ·   · …
                                       ↑ optimizer.step + scheduler.step + zero_grad

We divide each micro-batch loss by ``grad_accum_steps`` so the summed gradient
equals the true batch-32 gradient (mean, not sum).

ONE OPTIMIZER STEP = ONE "step" everywhere
------------------------------------------
Logging, the LR schedule, and ``max_steps`` all count *optimizer* steps, not
micro-batches — so "5,000 steps" means 5,000 real updates regardless of accum.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from src.config.schema import ExperimentConfig
from src.training.callbacks import Callback
from src.training.optim import build_optimizer, build_scheduler
from src.utils.devices import move_to_device
from src.utils.logging import get_logger

log = get_logger(__name__)


class Trainer:
    """Fine-tune ``wrapper`` on ``dataset`` under ``objective``.

    Parameters
    ----------
    wrapper:       a built :class:`~src.models.base.BaseVLMWrapper`.
    objective:     a :class:`~src.objectives.base.BaseObjective`.
    dataset:       the training :class:`~src.data.base.BaseVLMDataset`.
    collator:      a :class:`~src.data.collator.VLMCollator`.
    cfg:           the full :class:`ExperimentConfig`.
    callbacks:     list of :class:`Callback` (console/wandb/checkpoint).
    evaluator_fn:  optional ``() -> dict[str, float]`` run mid-training and at
                   the end (the runner passes the Evaluator's bound method).

    Example
    -------
    >>> trainer = Trainer(wrapper, objective, train_ds, collator, cfg, callbacks)
    >>> summary = trainer.train()
    >>> summary["final_loss"]
    0.83
    """

    def __init__(
        self,
        wrapper,
        objective,
        dataset,
        collator,
        cfg: ExperimentConfig,
        callbacks: list[Callback] | None = None,
        evaluator_fn: Callable[[], dict[str, float]] | None = None,
    ):
        self.wrapper = wrapper
        self.objective = objective
        self.dataset = dataset
        self.collator = collator
        self.cfg = cfg
        self.callbacks = callbacks or []
        self.evaluator_fn = evaluator_fn

        self.loader = DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0,  # PIL images + HF processor: keep collation in-process
        )

        accum = cfg.train.grad_accum_steps
        optim_steps_per_epoch = math.ceil(len(self.loader) / accum)
        total = cfg.train.epochs * optim_steps_per_epoch
        if cfg.train.max_steps is not None:
            total = min(total, cfg.train.max_steps)
        self.total_steps = total

        self.optimizer = build_optimizer(wrapper.trainable_parameters(), cfg.train)
        self.scheduler = build_scheduler(self.optimizer, cfg.train, total)

    def _fire(self, hook: str, *args) -> None:
        for cb in self.callbacks:
            getattr(cb, hook)(self, *args)

    def train(self) -> dict[str, Any]:
        cfg, accum = self.cfg.train, self.cfg.train.grad_accum_steps
        device, dtype = self.wrapper.device, self.wrapper.dtype
        self.wrapper.train()
        self._fire("on_train_begin")

        log.info(
            "training %s: %d optimizer steps (batch %d × accum %d = %d effective)",
            self.cfg.name,
            self.total_steps,
            cfg.batch_size,
            accum,
            cfg.batch_size * accum,
        )

        t0 = time.time()
        optim_step, micro, last = 0, 0, {}
        self.optimizer.zero_grad()
        done = False
        for _epoch in range(cfg.epochs):
            for batch in self.loader:
                batch = move_to_device(batch, device, dtype)
                out = self.objective.compute(self.wrapper, batch)
                (out.loss / accum).backward()
                last = out.components
                micro += 1

                if micro % accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.wrapper.trainable_parameters()), cfg.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    optim_step += 1

                    self._fire(
                        "on_step_end", optim_step, self.scheduler.get_last_lr()[0], last
                    )
                    if (
                        cfg.eval_every
                        and self.evaluator_fn
                        and optim_step % cfg.eval_every == 0
                    ):
                        self._run_eval(optim_step)

                    if optim_step >= self.total_steps:
                        done = True
                        break
            if done:
                break

        summary: dict[str, Any] = {
            "final_loss": last.get("loss"),
            "steps": optim_step,
            "train_seconds": round(time.time() - t0, 1),
        }
        if self.evaluator_fn:
            summary["final_metrics"] = self._run_eval(optim_step)
        self._fire("on_train_end", summary)
        log.info("done in %.0fs | %s", summary["train_seconds"], summary)
        return summary

    def _run_eval(self, step: int) -> dict[str, float]:
        self.wrapper.eval()
        metrics = self.evaluator_fn()
        self._fire("on_evaluate", step, metrics)
        self.wrapper.train()
        return metrics
