"""
src.training.callbacks — pluggable hooks fired by the Trainer.

Keeps the training loop clean: the loop emits events, callbacks decide what to do
with them (print, log to wandb, checkpoint). Add behavior without touching the
loop. Event flow::

    on_train_begin
      └─ per optimizer step ──▶ on_step_end(step, lr, components)
      └─ per eval ────────────▶ on_evaluate(step, metrics)
    on_train_end(summary)

Provided callbacks: console logging, Weights & Biases, periodic checkpointing.
"""

from __future__ import annotations

from typing import Any

from src.training.checkpoint import save_checkpoint
from src.utils.logging import get_logger

log = get_logger(__name__)


class Callback:
    """No-op base. Override the hooks you care about."""

    def on_train_begin(self, trainer) -> None: ...
    def on_step_end(
        self, trainer, step: int, lr: float, components: dict[str, float]
    ) -> None: ...
    def on_evaluate(self, trainer, step: int, metrics: dict[str, float]) -> None: ...
    def on_train_end(self, trainer, summary: dict[str, Any]) -> None: ...


class ConsoleCallback(Callback):
    """Print a tidy progress line every ``log_every`` steps."""

    def on_step_end(self, trainer, step, lr, components):
        if step % trainer.cfg.train.log_every == 0:
            def _fmt(key, value):
                if isinstance(value, (int, float)):
                    return f"{key}={value:.4f}"
                return f"{key}={value}"

            comp = "  ".join(
                _fmt(k, v) for k, v in components.items() if k != "alpha"
            )
            log.info("step %d/%d | lr %.2e | %s", step, trainer.total_steps, lr, comp)

    def on_evaluate(self, trainer, step, metrics):
        pretty = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        log.info("eval @ step %d | %s", step, pretty)


class WandbCallback(Callback):
    """Log losses, LR and eval metrics to Weights & Biases.

    Initialized lazily on ``on_train_begin`` so importing this module never
    requires wandb to be installed/configured. Respects ``WandbConfig.enabled``
    and ``mode`` (online/offline/disabled).
    """

    def __init__(self):
        self._run = None

    def on_train_begin(self, trainer):
        wcfg = trainer.cfg.wandb
        if not wcfg.enabled or wcfg.mode == "disabled":
            return
        import wandb  # local import: optional dependency at runtime

        self._run = wandb.init(
            project=wcfg.project,
            entity=wcfg.entity,
            name=trainer.cfg.name,
            tags=[trainer.cfg.rq, *wcfg.tags],
            mode=wcfg.mode,
            config=trainer.cfg.to_dict(),
        )

    def on_step_end(self, trainer, step, lr, components):
        if self._run is not None:
            self._run.log(
                {"lr": lr, **{f"train/{k}": v for k, v in components.items()}},
                step=step,
            )

    def on_evaluate(self, trainer, step, metrics):
        if self._run is not None:
            self._run.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)

    def on_train_end(self, trainer, summary):
        if self._run is not None:
            self._run.summary.update(summary)
            self._run.finish()


class CheckpointCallback(Callback):
    """Save the trainable delta every ``save_every`` steps and once at the end."""

    def __init__(self, out_dir):
        self.out_dir = out_dir

    def on_step_end(self, trainer, step, lr, components):
        every = trainer.cfg.train.save_every
        if every and step > 0 and step % every == 0:
            save_checkpoint(
                trainer.wrapper,
                f"{self.out_dir}/step_{step}",
                meta={"step": step, **components},
            )

    def on_train_end(self, trainer, summary):
        save_checkpoint(trainer.wrapper, self.out_dir, meta=summary)
