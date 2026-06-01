"""
src.evaluation.base — what a prediction and a metric are.

The evaluator produces one :class:`Prediction` per item (computed once), then
each configured :class:`BaseMetric` reduces the list to a few numbers. Splitting
"run the model" from "score the outputs" means you can add a metric without
re-running inference, and reuse the same predictions across several metrics.

    dataset ──Evaluator──▶ [Prediction, Prediction, …] ──metric.compute──▶ {"mc_accuracy": 0.61}
                                                         ──metric.compute──▶ {"rouge_l": 0.44}
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.data.example import VLMExample


@dataclass
class Prediction:
    """Everything the evaluator computed for one example.

    Fields are populated according to task type — MC items get
    ``predicted_index`` + ``choice_scores``; open-ended items get
    ``predicted_text``; CoT runs also fill ``reasoning``.
    """

    example: VLMExample
    predicted_index: int | None = None  # multiple-choice argmax
    predicted_text: str = ""  # generated answer (open-ended)
    reasoning: str = ""  # generated chain-of-thought (RQ5 source)
    choice_scores: list[float] = field(default_factory=list)


class BaseMetric(ABC):
    """Reduce a list of predictions to one or more named scores.

    Class attributes
    ----------------
    name:
        Registry key (``EvalConfig.metrics`` references it).
    applies_to:
        ``"mc"`` (needs ``predicted_index``), ``"open"`` (needs
        ``predicted_text``), or ``"explanation"`` (needs ``reasoning`` +
        gold ``explanation``). The evaluator skips metrics whose inputs aren't
        available for the current dataset.
    """

    name: str = "metric"
    applies_to: str = "mc"

    @abstractmethod
    def compute(self, predictions: list[Prediction]) -> dict[str, float]:
        """Return ``{score_name: value}`` aggregated over ``predictions``."""
