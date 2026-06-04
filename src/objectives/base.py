"""
src.objectives.base — what a training objective is, and the CE helper they share.

An objective answers one question: *given the model and a batch, what scalar do
we backprop?* That scalar IS the experiment for RQ2 (contrastive vs generative)
and RQ3 (with vs without explanation). Swapping objectives — nothing else —
isolates the effect of the training signal.

    LossOutput
      .loss        ← the scalar to .backward()
      .components  ← {"loss": .., "l_answer": .., "l_explanation": ..} for logging

The shared helper ``masked_token_ce`` computes next-token cross-entropy over a
*subset* of positions (a boolean mask), which is what lets the explanation-aware
objective score the answer span and the explanation span independently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from src.config.schema import ObjectiveConfig


@dataclass
class LossOutput:
    """The result of one objective evaluation."""

    loss: torch.Tensor  #: scalar tensor with grad — call ``.backward()`` on it
    components: dict[str, object] = field(default_factory=dict)  #: scalars/labels for logging


class BaseObjective(ABC):
    """Compute a training loss from a model + a collated batch.

    Class attribute
    ---------------
    requires_span_ids:
        If True, the Trainer tells the collator to emit ``span_ids`` (answer vs
        explanation tags). Only the explanation-aware objective needs it.
    """

    requires_span_ids: bool = False

    def __init__(self, cfg: ObjectiveConfig):
        self.cfg = cfg

    @abstractmethod
    def compute(self, wrapper, batch: dict) -> LossOutput:
        """Return a :class:`LossOutput` for ``batch`` (already on-device)."""


def masked_token_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean next-token cross-entropy over selected positions.

    The autoregressive shift in one place::

        logits[:, t]  predicts  token[t+1]
        ────────────  ───────   ──────────
        so we line up  logits[:, :-1]  with  labels[:, 1:]

    Parameters
    ----------
    logits:
        ``(B, T, vocab)`` raw scores from the model.
    labels:
        ``(B, T)`` target ids with ``-100`` at ignored positions.
    mask:
        Optional ``(B, T)`` boolean. If given, only positions where ``mask`` is
        True **and** the label isn't ``-100`` are averaged — this is how the
        answer span and explanation span get separate losses. If ``None``, all
        non-ignored positions count (plain generative loss).

    Returns
    -------
    Scalar tensor. Returns ``0.0`` (with grad) if the selection is empty — so an
    answer-only example contributes no explanation loss instead of a NaN.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    valid = shift_labels != -100
    if mask is not None:
        valid = valid & mask[:, 1:].bool()

    if valid.sum() == 0:
        # keep it differentiable & on-device, but contribute nothing
        return shift_logits.sum() * 0.0

    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1).clone()
    flat_labels[~valid.view(-1)] = -100  # let CE's ignore_index drop them
    return F.cross_entropy(flat_logits.float(), flat_labels, ignore_index=-100)
