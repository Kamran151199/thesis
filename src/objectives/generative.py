"""Generative objective — plain next-token cross-entropy on the target.

The RQ2 "generative alignment" arm and the simplest possible loss: let the model
compute its own masked cross-entropy. The collator already masked everything but
the target span to ``-100``, so ``model(**batch).loss`` is exactly the loss over
"Reasoning: … Answer: X." and nothing else.

    batch (input_ids, attention_mask, pixel_values, labels) ──▶ model ──▶ out.loss
                                                                            │
                                                              backprop ◀────┘

This is the baseline every other objective is compared against.
"""

from __future__ import annotations

from src.objectives.base import BaseObjective, LossOutput
from src.objectives import OBJECTIVES


@OBJECTIVES.register("generative")
class GenerativeObjective(BaseObjective):
    requires_span_ids = False

    def compute(self, wrapper, batch: dict) -> LossOutput:
        out = wrapper.forward(batch)  # model computes masked CE from `labels`
        return LossOutput(loss=out.loss, components={"loss": out.loss.item()})
