"""Explanation-aware objective — span-weighted rationale/answer supervision.

THE LOSS
--------
::

    L = α · L_answer  +  (1 − α) · L_explanation

Two cross-entropies over the SAME forward pass, on disjoint token spans the
collator tagged for us::

    target:   " Reasoning: plants take in CO2 …   Answer: Carbon dioxide ."
    span_ids:    1  1  1  1  1  1  1  1  1  1        2  2  2  2  2  2  2
                 └──────── L_explanation ─────┘     └──── L_answer ────┘

With ``alpha_mode="fixed"``, α is the answer-span weight swept in the ablation:

    α = 1.0  →  answer-span-only supervision
    α = 0.5  →  balanced: learn to reason AND answer
    α = 0.0  →  pure explanation supervision  (learns to reason, answer unweighted)

With ``alpha_mode="length_aware"``, α is derived from the number of supervised
answer and explanation tokens. With multiplier=1.0 this is the natural
token-weighted rationale CE control; increasing ``answer_weight_multiplier`` is
the explicit way to upweight answer tokens beyond their natural length ratio.

If the target template contains a gold rationale, α=1 still predicts answer
tokens after rationale tokens in the teacher-forced sequence. It is therefore
not the same as a clean answer-only control; that control uses the
``answer_only`` prompt template and the plain generative objective.

WHY NOT JUST USE THE MODEL'S BUILT-IN LOSS? Because that averages the whole
target uniformly — it can't put different weights on the reasoning vs the answer.
We compute the two spans ourselves from the logits + ``span_ids``.
"""

from __future__ import annotations

import torch

from src.data.constants import SPAN_ANSWER, SPAN_EXPLANATION
from src.objectives.base import BaseObjective, LossOutput, masked_token_ce
from src.objectives import OBJECTIVES


def supervised_span_counts(labels, span_ids) -> tuple[int, int]:
    """Return shifted supervised token counts for explanation and answer spans."""
    valid = labels[:, 1:] != -100
    shifted_spans = span_ids[:, 1:]
    n_expl = int(((shifted_spans == SPAN_EXPLANATION) & valid).sum().item())
    n_answer = int(((shifted_spans == SPAN_ANSWER) & valid).sum().item())
    return n_expl, n_answer


def effective_answer_alpha(cfg, labels, span_ids):
    """Scalar answer-span weight used by the explanation-aware loss.

    ``fixed`` returns ``cfg.alpha``. ``length_aware`` computes the natural answer
    token proportion after the autoregressive shift, optionally multiplied by
    ``answer_weight_multiplier`` to intentionally upweight answer tokens.
    """
    mode = getattr(cfg, "alpha_mode", "fixed")
    if mode == "fixed":
        return labels.new_tensor(float(cfg.alpha), dtype=torch.float32)

    if mode != "length_aware":
        raise ValueError(f"unknown objective.alpha_mode: {mode!r}")

    n_expl, n_answer = supervised_span_counts(labels, span_ids)
    answer_weight = float(getattr(cfg, "answer_weight_multiplier", 1.0))
    weighted_answer = answer_weight * n_answer
    denom = weighted_answer + n_expl
    if denom <= 0:
        return labels.new_tensor(0.5, dtype=torch.float32)
    return labels.new_tensor(weighted_answer / denom, dtype=torch.float32)


@OBJECTIVES.register("explanation_aware")
class ExplanationAwareObjective(BaseObjective):
    requires_span_ids = True  # tells the Trainer to enable span tagging

    def compute(self, wrapper, batch: dict) -> LossOutput:
        labels = batch["labels"]
        span_ids = batch["span_ids"]
        alpha = effective_answer_alpha(self.cfg, labels, span_ids)
        n_expl, n_answer = supervised_span_counts(labels, span_ids)

        # Forward WITHOUT labels/span_ids (model would otherwise reject span_ids
        # and compute a redundant uniform loss). We only need the logits.
        forward_batch = {
            k: v for k, v in batch.items() if k not in ("labels", "span_ids")
        }
        logits = wrapper.forward(forward_batch).logits

        l_answer = masked_token_ce(logits, labels, mask=span_ids == SPAN_ANSWER)
        l_expl = masked_token_ce(logits, labels, mask=span_ids == SPAN_EXPLANATION)
        loss = alpha * l_answer + (1.0 - alpha) * l_expl

        return LossOutput(
            loss=loss,
            components={
                "loss": loss.item(),
                "l_answer": l_answer.item(),
                "l_explanation": l_expl.item(),
                "alpha": float(alpha.detach().item()),
                "configured_alpha": float(self.cfg.alpha),
                "alpha_mode": getattr(self.cfg, "alpha_mode", "fixed"),
                "answer_weight_multiplier": float(
                    getattr(self.cfg, "answer_weight_multiplier", 1.0)
                ),
                "n_answer_tokens": float(n_answer),
                "n_explanation_tokens": float(n_expl),
            },
        )
