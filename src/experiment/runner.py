"""
src.experiment.runner — config in, finished experiment out.

This is where the four axes snap together. Given ONE ``ExperimentConfig`` it
builds the backbone, the data, the objective, the trainer and the evaluator,
measures a baseline, fine-tunes, measures the lift, and writes everything to the
experiment folder. The whole thesis is "write a YAML, call this."

::

    ExperimentConfig
        │
        ├─ set_seed
        ├─ build_model(model)         → wrapper        (backbone axis)
        ├─ build_dataset(data, …)     → train / eval   (data axis)
        ├─ build_objective(objective) → loss           (objective axis)
        ├─ build_collator(data, …)    → batches (span-tagged iff expl-aware)
        ├─ build_metrics(eval)        → [metric…]       (metric axis)
        │
        ├─ baseline = Evaluator.evaluate()   ← BEFORE training
        ├─ Trainer.train()                   ← fine-tune (final eval at the end)
        └─ write config + results.json + checkpoint to outputs/<name>/

The baseline-then-finetune-then-lift flow is exactly the proven prototype,
generalized so it runs for any (backbone, dataset, objective, metric) combo.
"""

from __future__ import annotations

import json
from typing import Any

from src.config.schema import ExperimentConfig
from src.config import dump_config
from src.data import build_collator, build_dataset, build_template
from src.evaluation import Evaluator, build_metrics
from src.models import build_model
from src.objectives import build_objective
from src.training import CheckpointCallback, ConsoleCallback, Trainer, WandbCallback
from src.utils import describe_device, experiment_dir, get_logger, set_seed

log = get_logger(__name__)


class ExperimentRunner:
    """Run one experiment end-to-end from its config."""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.out_dir = experiment_dir(cfg.output_dir or cfg.name)

    def run(self) -> dict[str, Any]:
        cfg = self.cfg
        set_seed(cfg.seed)
        log.info("=== %s (%s) on %s ===", cfg.name, cfg.rq, describe_device())

        # ── build the four axes ────────────────────────────────────────────────
        wrapper = build_model(cfg.model)
        objective = build_objective(cfg.objective)
        template = build_template(cfg.data.prompt_variant)

        train_ds = build_dataset(cfg.data, split=cfg.data.split_train)
        eval_ds = build_dataset(cfg.data, split=cfg.data.split_eval)
        collator = build_collator(
            cfg.data, wrapper, tag_spans=objective.requires_span_ids
        )

        metrics = build_metrics(cfg.eval.metrics)
        evaluator = Evaluator(wrapper, eval_ds, template, cfg.eval, metrics)

        # ── baseline (before any training) ──────────────────────────────────────
        log.info("measuring baseline (pre-fine-tune)...")
        baseline = evaluator.evaluate()
        log.info("baseline: %s", baseline)

        # ── train (final eval runs inside Trainer.train) ────────────────────────
        callbacks = [
            ConsoleCallback(),
            WandbCallback(),
            CheckpointCallback(self.out_dir),
        ]
        trainer = Trainer(
            wrapper,
            objective,
            train_ds,
            collator,
            cfg,
            callbacks,
            evaluator_fn=evaluator.evaluate,
        )
        summary = trainer.train()
        final = summary.get("final_metrics", {})

        # ── persist + report lift ────────────────────────────────────────────────
        results = {"baseline": baseline, "final": final, "summary": summary}
        dump_config(cfg, self.out_dir / "config.resolved.yaml")
        (self.out_dir / "results.json").write_text(
            json.dumps(results, indent=2, default=str)
        )
        self._log_lift(baseline, final)
        log.info("artifacts → %s", self.out_dir)
        return results

    @staticmethod
    def _log_lift(baseline: dict, final: dict) -> None:
        for k in sorted(set(baseline) & set(final)):
            if k == "random_baseline":
                continue
            log.info(
                "LIFT %-18s %.4f → %.4f  (%+.4f)",
                k,
                baseline[k],
                final[k],
                final[k] - baseline[k],
            )
