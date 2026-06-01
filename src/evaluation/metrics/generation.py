"""Explanation-quality metrics — how good is the GENERATED reasoning? (RQ3)

Compares the model's generated rationale (``prediction.reasoning``) against the
gold rationale (``example.explanation``):

    rouge_l   longest-common-subsequence overlap (recall-oriented; fluent order)
    bleu      n-gram precision with brevity penalty (BLEU-4)

These quantify "did the explanation say the right things in roughly the right
way?" — the surface-level half of RQ3 (the faithfulness half is in
``src.evaluation.faithfulness``). BERTScore (semantic similarity) is a natural
third metric; add it as another class here once ``bert-score`` is wired in.
"""

from __future__ import annotations

from src.evaluation.base import BaseMetric, Prediction
from src.evaluation.metrics import METRICS


def _pairs(predictions: list[Prediction]) -> list[tuple[str, str]]:
    """(hypothesis reasoning, gold explanation) pairs where both exist."""
    out = []
    for p in predictions:
        gold = p.example.explanation
        if gold and p.reasoning:
            out.append((p.reasoning, gold.strip()))
    return out


@METRICS.register("rouge_l")
class RougeL(BaseMetric):
    name = "rouge_l"
    applies_to = "explanation"

    def compute(self, predictions: list[Prediction]) -> dict[str, float]:
        from rouge_score import rouge_scorer  # local import (optional dep)

        pairs = _pairs(predictions)
        if not pairs:
            return {"rouge_l": 0.0}
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        f = [scorer.score(gold, hyp)["rougeL"].fmeasure for hyp, gold in pairs]
        return {"rouge_l": sum(f) / len(f)}


@METRICS.register("bleu")
class Bleu4(BaseMetric):
    name = "bleu"
    applies_to = "explanation"

    def compute(self, predictions: list[Prediction]) -> dict[str, float]:
        from nltk.translate.bleu_score import (  # local import (optional dep)
            SmoothingFunction,
            sentence_bleu,
        )

        pairs = _pairs(predictions)
        if not pairs:
            return {"bleu": 0.0}
        smooth = SmoothingFunction().method1
        scores = [
            sentence_bleu([gold.split()], hyp.split(), smoothing_function=smooth)
            for hyp, gold in pairs
        ]
        return {"bleu": sum(scores) / len(scores)}
