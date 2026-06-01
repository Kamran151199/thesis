"""
src.evaluation — score a trained model: accuracy, explanation quality, faithfulness.

Public API
----------
    from src.evaluation import Evaluator, build_metrics
    metrics = build_metrics(cfg.eval.metrics)            # [BaseMetric, …]
    ev = Evaluator(wrapper, eval_ds, template, cfg.eval, metrics)
    scores = ev.evaluate()                                # {"mc_accuracy": 0.61, …}

Layers: ``scoring`` (likelihood + generation primitives), ``metrics`` (accuracy/
generation/retrieval), ``faithfulness`` (RQ5), and the ``Evaluator`` that runs
the model once and applies every applicable metric. See the README.
"""

from src.evaluation.base import BaseMetric, Prediction
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import METRICS


def build_metrics(names: list[str]) -> list[BaseMetric]:
    """Instantiate the metrics named in ``EvalConfig.metrics`` (skips unknown
    names with a clear registry error)."""
    return [METRICS.build(name) for name in names]


__all__ = ["Evaluator", "build_metrics", "BaseMetric", "Prediction", "METRICS"]
