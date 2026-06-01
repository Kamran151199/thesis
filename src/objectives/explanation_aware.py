"""Explanation-aware objective — the thesis's central contribution (RQ3).

THE LOSS
--------
::

    L = α · L_answer  +  (1 − α) · L_explanation

Two cross-entropies over the SAME forward pass, on disjoint token spans the
collator tagged for us::

    target:   " Reasoning: plants take in CO2 …   Answer: Carbon dioxide ."
    span_ids:    1  1  1  1  1  1  1  1  1  1        2  2  2  2  2  2  2
                 └──────── L_explanation ─────┘     └──── L_answer ────┘

α is the dial RQ3 sweeps over {0.0, 0.5, 1.0}:

    α = 1.0  →  pure answer supervision      (≡ the generative / answer-only arm)
    α = 0.5  →  balanced: learn to reason AND answer
    α = 0.0  →  pure explanation supervision  (learns to reason, answer unweighted)

Holding the data, backbone and hyperparameters fixed and moving only α turns
"does explaining help?" into a clean, measurable curve.

WHY NOT JUST USE THE MODEL'S BUILT-IN LOSS? Because that averages the whole
target uniformly — it can't put different weights on the reasoning vs the answer.
We compute the two spans ourselves from the logits + ``span_ids``.
"""

from __future__ import annotations

from src.data.collator import SPAN_ANSWER, SPAN_EXPLANATION
from src.objectives.base import BaseObjective, LossOutput, masked_token_ce
from src.objectives import OBJECTIVES


@OBJECTIVES.register("explanation_aware")
class ExplanationAwareObjective(BaseObjective):
    requires_span_ids = True  # tells the Trainer to enable span tagging

    def compute(self, wrapper, batch: dict) -> LossOutput:
        alpha = self.cfg.alpha
        labels = batch["labels"]
        span_ids = batch["span_ids"]

        # Forward WITHOUT labels/span_ids (model would otherwise reject span_ids
        # and compute a redundant uniform loss). We only need the logits.
        forward_batch = {
            k: v for k, v in batch.items() if k not in ("labels", "span_ids")
        }
        logits = wrapper.forward(forward_batch).logits

        # here we take logits (the unnormalized scores for every token),
        # take the labels (the target token ids),
        # take the span_ids (which token belongs to which loss component),
        # and compute the loss by masking out the irrelevant tokens for each component and applying cross-entropy to the rest.
        # How filtering out the irrelevant tokens works: masked_token_ce applies the CE loss only to the tokens where the mask is True, effectively ignoring the others.
        l_answer = masked_token_ce(logits, labels, mask=span_ids == SPAN_ANSWER)
        l_expl = masked_token_ce(logits, labels, mask=span_ids == SPAN_EXPLANATION)
        loss = alpha * l_answer + (1.0 - alpha) * l_expl

        return LossOutput(
            loss=loss,
            components={
                "loss": loss.item(),
                "l_answer": l_answer.item(),
                "l_explanation": l_expl.item(),
                "alpha": alpha,
            },
        )
