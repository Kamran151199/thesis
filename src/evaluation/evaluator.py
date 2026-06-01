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
from src.data.base import BaseVLMDataset
from src.data.example import VLMExample
from src.data.prompts import PromptTemplate
from src.evaluation.base import BaseMetric, Prediction
from src.evaluation.scoring import (
    generate_batch,
    score_batch,
    split_reasoning_answer,
)
from src.models.base import BaseVLMWrapper
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
        self,
        wrapper: BaseVLMWrapper,
        dataset: BaseVLMDataset,
        template: PromptTemplate,
        cfg: EvalConfig,
        metrics: list[BaseMetric]
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
        n = len(self.dataset)
        bs = max(1, self.cfg.batch_size)
        for start in range(0, n, bs):
            chunk = [self.dataset[i] for i in range(start, min(start + bs, n))]
            preds.extend(self._predict_chunk(chunk))
            log.info("predicted %d/%d", min(start + bs, n), n)
        return preds

    def _predict_one(self, ex: VLMExample) -> Prediction:
        """Single-example prediction — just a one-row :meth:`_predict_chunk`."""
        return self._predict_chunk([ex])[0]

    def _predict_chunk(self, chunk: list[VLMExample]) -> list[Prediction]:
        """Predict a batch of examples in as few forward passes as possible.

        One batched generation for the whole chunk (the autoregressive
        bottleneck), then — for multiple-choice — one batched likelihood pass
        over every (example × choice) row. The per-item results are identical to
        the single-item ``generate_continuation`` / ``score_continuation`` path;
        only the batching differs (so the GPU stays busy instead of idling at
        batch size 1).
        """
        is_mc = self.dataset.is_multiple_choice
        prompts = [self.template.prompt(ex) for ex in chunk]
        images = [ex.image.convert("RGB") for ex in chunk]

        # Batched generation: reasoning for MC+CoT, or the answer for open-ended.
        conts: list[str] | None = None
        if (not is_mc) or (self.cfg.cot and self.template.produces_reasoning):
            conts = generate_batch(
                self.wrapper,
                chunk,
                self.template,
                self.cfg.max_new_tokens,
                self.cfg.max_length,
            )

        if not is_mc:
            # Open-ended: the generated text IS the answer.
            assert conts is not None  # always generated on the open-ended path
            out: list[Prediction] = []
            for ex, cont in zip(chunk, conts):
                gen_reasoning, answer = split_reasoning_answer(cont)
                out.append(
                    Prediction(
                        example=ex,
                        predicted_text=answer or cont.strip(),
                        reasoning=gen_reasoning,
                    )
                )
            return out

        # Multiple-choice: build each example's (CoT-conditioned) scoring context…
        reasonings = (
            [split_reasoning_answer(c)[0] for c in conts]
            if conts is not None
            else [""] * len(chunk)
        )
        contexts = [
            f"{p} Reasoning: {r} Answer:" if r else f"{p} Answer:"
            for p, r in zip(prompts, reasonings)
        ]

        # …then flatten every (example, choice) pair into one scoring batch.
        row_images: list = []
        row_contexts: list[str] = []
        row_choices: list[str] = []
        owner: list[int] = []
        for i, ex in enumerate(chunk):
            if not ex.choices:
                raise ValueError(f"Multiple-choice example has no choices: {ex!r}")
            for choice in ex.choices:
                row_images.append(images[i])
                row_contexts.append(contexts[i])
                row_choices.append(choice)
                owner.append(i)

        # Score in sub-batches of batch_size rows (bounds the logits tensor).
        bs = max(1, self.cfg.batch_size)
        flat_scores: list[float] = []
        for s in range(0, len(row_images), bs):
            flat_scores.extend(
                score_batch(
                    self.wrapper,
                    row_images[s : s + bs],
                    row_contexts[s : s + bs],
                    row_choices[s : s + bs],
                    self.cfg.max_length,
                )
            )

        # Regroup scores by example and take the argmax choice.
        out = []
        for i, ex in enumerate(chunk):
            ex_scores = [flat_scores[j] for j, o in enumerate(owner) if o == i]
            idx = int(np.argmax(ex_scores))
            out.append(
                Prediction(
                    example=ex,
                    predicted_index=idx,
                    predicted_text=ex.choices[idx],
                    reasoning=reasonings[i],
                    choice_scores=ex_scores,
                )
            )
        return out

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
                        1.0 / len(self.dataset[i].choices)  # type: ignore[union-attr]
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
