"""Answer-accuracy metrics — one per benchmark's scoring convention (§7.5).

    mc_accuracy        ScienceQA / A-OKVQA   exact index match (argmax == gold)
    exact_match        generic open-ended    normalized string equality (any gold variant)
    anls               DocVQA                edit-distance similarity (OCR-tolerant)
    relaxed_accuracy   ChartQA               numeric within 5%, else exact match

Why several? "What counts as correct" genuinely differs: a document answer
that's one OCR character off should get partial credit (ANLS), a chart value of
41.8 vs gold 42 should count (relaxed), but a multiple-choice index is exact.
"""

from __future__ import annotations

import re

from src.evaluation.base import BaseMetric, Prediction
from src.evaluation.metrics import METRICS


def _normalize(text: str) -> str:
    """Lowercase, strip articles/punctuation/extra spaces — standard VQA norm."""
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _gold_variants(p: Prediction) -> list[str]:
    """All acceptable gold strings (metadata answer list, else the single gold)."""
    variants = p.example.metadata.get("answers")
    return list(variants) if variants else [p.example.answer]


def _normalized_levenshtein(a: str, b: str) -> float:
    """Levenshtein distance / max(len) ∈ [0,1]; 0 = identical."""
    if not a and not b:
        return 0.0
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n] / max(m, n)


@METRICS.register("mc_accuracy")
class MultipleChoiceAccuracy(BaseMetric):
    name = "mc_accuracy"
    applies_to = "mc"

    def compute(self, predictions: list[Prediction]) -> dict[str, float]:
        correct = sum(
            int(p.predicted_index == p.example.answer_index) for p in predictions
        )
        return {"mc_accuracy": correct / max(len(predictions), 1)}


@METRICS.register("exact_match")
class ExactMatch(BaseMetric):
    name = "exact_match"
    applies_to = "open"

    def compute(self, predictions: list[Prediction]) -> dict[str, float]:
        hits = 0
        for p in predictions:
            pred = _normalize(p.predicted_text)
            hits += int(any(pred == _normalize(g) for g in _gold_variants(p)))
        return {"exact_match": hits / max(len(predictions), 1)}


@METRICS.register("anls")
class ANLS(BaseMetric):
    """DocVQA's Average Normalized Levenshtein Similarity (threshold 0.5)."""

    name = "anls"
    applies_to = "open"

    def compute(self, predictions: list[Prediction]) -> dict[str, float]:
        total = 0.0
        for p in predictions:
            pred = _normalize(p.predicted_text)
            best = max(
                (
                    1.0 - _normalized_levenshtein(pred, _normalize(g))
                    for g in _gold_variants(p)
                ),
                default=0.0,
            )
            total += best if best >= 0.5 else 0.0  # ANLS zeros out poor matches
        return {"anls": total / max(len(predictions), 1)}


@METRICS.register("relaxed_accuracy")
class RelaxedAccuracy(BaseMetric):
    """ChartQA relaxed accuracy: numeric answers within 5%, else exact match."""

    name = "relaxed_accuracy"
    applies_to = "open"

    @staticmethod
    def _as_float(s: str) -> float | None:
        m = re.search(r"-?\d+\.?\d*", s.replace(",", ""))
        return float(m.group()) if m else None

    def compute(self, predictions: list[Prediction]) -> dict[str, float]:
        hits = 0
        for p in predictions:
            ok = False
            pf = self._as_float(p.predicted_text)
            for g in _gold_variants(p):
                gf = self._as_float(g)
                if pf is not None and gf is not None:
                    ok = gf != 0 and abs(pf - gf) / abs(gf) <= 0.05
                else:
                    ok = _normalize(p.predicted_text) == _normalize(g)
                if ok:
                    break
            hits += int(ok)
        return {"relaxed_accuracy": hits / max(len(predictions), 1)}
