"""Metric registry + auto-registration.

``METRICS`` maps a name (used in ``EvalConfig.metrics``) to a :class:`BaseMetric`.
Defined first, then the metric modules import for their registration side
effects. Retrieval (``retrieval.py``) is a standalone function, not a
prediction-metric, so it isn't registered here — import it directly.
"""

from src.evaluation.base import BaseMetric
from src.registry import Registry

#: name → BaseMetric subclass. Keyed by entries in ``EvalConfig.metrics``.
METRICS: Registry[BaseMetric] = Registry("metric")

# Import for side effects (registration). Must come AFTER METRICS is defined.
from src.evaluation.metrics import accuracy, generation  # noqa: E402,F401

__all__ = ["METRICS"]
