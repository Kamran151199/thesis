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

import hashlib
import json
import os
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
        objective = build_objective(cfg.objective)
        template = build_template(cfg.data.prompt_variant)
        train_ds = build_dataset(cfg.data, split=cfg.data.split_train)
        eval_ds = build_dataset(cfg.data, split=cfg.data.split_eval)
        self._sync_eval_length()

        wrapper = build_model(cfg.model)
        collator = build_collator(
            cfg.data, wrapper, tag_spans=objective.requires_span_ids
        )

        metrics = build_metrics(cfg.eval.metrics)
        evaluator = Evaluator(wrapper, eval_ds, template, cfg.eval, metrics)

        # ── baseline (before any training) ──────────────────────────────────────
        baseline = self._load_or_measure_baseline(evaluator)
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
            train_ds,  # type: ignore (BaseVLMDataset is a Dataset's duck type, but mypy doesn't see it)
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

    def _load_or_measure_baseline(self, evaluator: Evaluator) -> dict[str, float]:
        """Reuse identical zero-shot baselines across objective sweeps.

        Alpha sweeps change only the training objective, so the pre-fine-tune
        baseline is identical. Caching avoids repeated generation/scoring while
        keeping each run's ``results.json`` self-contained.
        """
        if os.environ.get("THESIS_DISABLE_BASELINE_CACHE") == "1":
            log.info("measuring baseline (cache disabled)...")
            return evaluator.evaluate()

        key, payload = self._baseline_cache_key()
        cache_dir = experiment_dir("_baseline_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{key}.json"

        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text())
                if cached.get("key_payload") == payload:
                    log.info("baseline cache hit: %s", cache_path)
                    return cached["baseline"]
                log.warning("baseline cache key collision or stale payload: %s", cache_path)
            except Exception as exc:  # noqa: BLE001
                log.warning("could not read baseline cache %s: %s", cache_path, exc)

        log.info("measuring baseline (pre-fine-tune)...")
        baseline = evaluator.evaluate()
        record = {
            "key": key,
            "key_payload": payload,
            "source_run": self.cfg.name,
            "baseline": baseline,
        }
        tmp = cache_path.with_name(cache_path.name + ".tmp")
        tmp.write_text(json.dumps(record, indent=2, default=str))
        tmp.replace(cache_path)
        log.info("baseline cache saved: %s", cache_path)
        return baseline

    def _baseline_cache_key(self) -> tuple[str, dict[str, Any]]:
        cfg = self.cfg.to_dict()
        model = dict(cfg["model"])
        # The BLIP-2 contrastive projection is an auxiliary training head. It is
        # not read during generation/scoring, so it should not split the baseline
        # cache between generative and contrastive RQ2 runs.
        model["contrastive_projection"] = False
        payload = {
            "version": 1,
            "seed": cfg["seed"],
            "model": model,
            "data": cfg["data"],
            "eval": cfg["eval"],
        }
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], payload

    def _sync_eval_length(self) -> None:
        """Keep eval encoding at least as long as train/data encoding.

        Qwen2-VL validates that the number of image placeholder tokens in the
        text matches the encoded ``input_ids``. For high-resolution chart/doc
        images, using the default eval max_length=1024 can truncate the image
        token block even when training uses data.max_length=2048. Raise eval
        length here so configs cannot silently diverge.
        """
        if self.cfg.eval.max_length < self.cfg.data.max_length:
            log.info(
                "raising eval.max_length %d → %d to match data.max_length",
                self.cfg.eval.max_length,
                self.cfg.data.max_length,
            )
            self.cfg.eval.max_length = self.cfg.data.max_length
