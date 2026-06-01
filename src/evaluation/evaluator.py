"""
src.evaluation.evaluator — run a model over a dataset, then score the outputs.

Two phases, cleanly separated::

    PREDICT   each example → a Prediction (run the model once)
    SCORE     [Prediction…] → metrics      (cheap; reuse predictions)

PREDICTION DEPENDS ON TASK TYPE
-------------------------------
Multiple-choice (ScienceQA, A-OKVQA) → **generate-then-score** (the proven CoT
eval): let the model write its reasoning, then likelihood-score each option in
that context and take the argmax. This is what fixed the bogus "100% baseline"
(string-matching the generated text was gameable).

Open-ended (ChartQA, DocVQA, VQAv2) → **generate** the answer text and compare
with EM / ANLS / relaxed accuracy.

Either way the generated reasoning is kept on the Prediction, so explanation
metrics (ROUGE-L, BLEU) and faithfulness analysis can use it without re-running.
"""

from __future__ import annotations

import numpy as np
import torch

from src.config.schema import EvalConfig
from src.evaluation.base import BaseMetric, Prediction
from src.evaluation.scoring import (
    generate_continuation,
    score_continuation,
    split_reasoning_answer,
)
from src.utils.logging import get_logger

log = get_logger(__name__)


class Evaluator:
    """Evaluate ``wrapper`` on ``dataset`` with the given metrics.

    Parameters
    ----------
    wrapper:   the (trained) backbone wrapper.
    dataset:   a :class:`~src.data.base.BaseVLMDataset`.
    template:  the prompt template (must match training for clean scoring).
    cfg:       the :class:`EvalConfig` (cot flag, max_new_tokens).
    metrics:   list of :class:`BaseMetric` to compute (built from cfg.metrics).

    Example
    -------
    >>> ev = Evaluator(wrapper, eval_ds, template, cfg.eval, metrics)
    >>> ev.evaluate()
    {'mc_accuracy': 0.61, 'random_baseline': 0.27}
    """

    def __init__(
        self, wrapper, dataset, template, cfg: EvalConfig, metrics: list[BaseMetric]
    ):
        self.wrapper = wrapper
        self.dataset = dataset
        self.template = template
        self.cfg = cfg
        self.metrics = metrics

    @torch.no_grad()
    def predict(self) -> list[Prediction]:
        self.wrapper.eval()
        preds: list[Prediction] = []
        for i in range(len(self.dataset)):
            ex = self.dataset[i]
            preds.append(self._predict_one(ex))
            if (i + 1) % 50 == 0:
                log.info("predicted %d/%d", i + 1, len(self.dataset))
        return preds

    def _predict_one(self, ex) -> Prediction:
        image = ex.image.convert("RGB")
        prompt = self.template.prompt(ex)

        # Generate the model's continuation (reasoning + tentative answer).
        reasoning = ""
        if self.cfg.cot and self.template.produces_reasoning:
            cont = generate_continuation(
                self.wrapper, ex, self.template, self.cfg.max_new_tokens
            )
            reasoning, _ = split_reasoning_answer(cont)

        if ex.is_multiple_choice:
            # Build the context the choices are scored in (CoT-conditioned if any).
            context = (
                f"{prompt} Reasoning: {reasoning} Answer:"
                if reasoning
                else f"{prompt} Answer:"
            )
            scores = [
                score_continuation(
                    self.wrapper, image, context, c, self.cfg.max_new_tokens
                )
                for c in ex.choices
            ]
            idx = int(np.argmax(scores))
            return Prediction(
                example=ex,
                predicted_index=idx,
                predicted_text=ex.choices[idx],
                reasoning=reasoning,
                choice_scores=scores,
            )

        # Open-ended: the generated text IS the answer.
        cont = generate_continuation(
            self.wrapper, ex, self.template, self.cfg.max_new_tokens
        )
        gen_reasoning, answer = split_reasoning_answer(cont)
        return Prediction(
            example=ex,
            predicted_text=answer or cont.strip(),
            reasoning=reasoning or gen_reasoning,
        )

    def evaluate(self) -> dict[str, float]:
        """Predict, then compute every applicable configured metric."""
        preds = self.predict()
        results: dict[str, float] = {}
        for metric in self.metrics:
            if not self._applicable(metric, preds):
                log.info(
                    "skipping metric %r (not applicable to this dataset)", metric.name
                )
                continue
            results.update(metric.compute(preds))

        if self.dataset.is_multiple_choice:
            results["random_baseline"] = float(
                np.mean(
                    [
                        1.0 / len(self.dataset[i].choices)
                        for i in range(len(self.dataset))
                    ]
                )
            )
        return results

    def _applicable(self, metric: BaseMetric, preds: list[Prediction]) -> bool:
        if metric.applies_to == "mc":
            return self.dataset.is_multiple_choice
        if metric.applies_to == "open":
            return not self.dataset.is_multiple_choice
        if metric.applies_to == "explanation":
            return any(p.reasoning and p.example.has_explanation for p in preds)
        return True
