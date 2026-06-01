"""Contrastive-enhanced objective — RQ2's "contrastive alignment" arm.

THE IDEA
--------
Take the generative loss and ADD an auxiliary InfoNCE (Info Noise-Contrastive
Estimation) term that pulls each image's projected features toward its own
answer's sentence embedding and pushes them away from the other answers in the
batch (proposal §7.3, "contrastive-enhanced")::

    L = L_generative  +  contrastive_weight · L_InfoNCE

InfoNCE in one picture (a batch of N image/answer pairs)::

        answers →   a0  a1  a2  a3
    images ↓      ┌────────────────┐
        i0        │ ✓   .   .   .  │   similarity matrix S = img · ansᵀ / τ
        i1        │ .   ✓   .   .  │   the diagonal is the TRUE pair
        i2        │ .   .   ✓   .  │   → cross-entropy with labels = [0,1,2,3]
        i3        │ .   .   .   ✓  │   in BOTH directions (img→ans, ans→img)
                  └────────────────┘

(Identical to the CLIP loss you built from scratch — symmetric cross-entropy
over the batch with a temperature τ.)

WHY A BACKBONE HOOK FOR THE FEATURES?
-------------------------------------
"Projected visual features" and "answer sentence embedding" live in different
places in BLIP-2 vs Qwen2-VL vs PaliGemma. So this objective asks the wrapper
for them via ``wrapper.contrastive_features(batch)`` — which each backbone must
implement (the base class raises a clear NotImplementedError telling you what to
return). The generative + explanation-aware objectives need no such hook.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.objectives.base import BaseObjective, LossOutput
from src.objectives import OBJECTIVES


def info_nce(
    image_embeds: torch.Tensor, text_embeds: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Symmetric InfoNCE over a batch of paired embeddings.

    Parameters
    ----------
    image_embeds, text_embeds:
        ``(N, D)`` each; row ``i`` of one is the true match for row ``i`` of the
        other. L2-normalized inside, so the dot product is a cosine similarity.
    temperature:
        Softmax temperature τ (smaller = sharper). 0.07 is the CLIP default.

    Returns
    -------
    Scalar loss = ½·(CE(img→txt) + CE(txt→img)) with targets = ``arange(N)``.
    """
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    logits = image_embeds @ text_embeds.t() / temperature  # (N, N)
    targets = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)
    )


@OBJECTIVES.register("contrastive")
class ContrastiveObjective(BaseObjective):
    requires_span_ids = False

    def compute(self, wrapper, batch: dict) -> LossOutput:
        # 1. The usual generative loss (the model still learns to answer).
        out = wrapper.forward(batch)
        l_gen = out.loss

        # 2. Auxiliary InfoNCE between image features and answer embeddings.
        image_embeds, text_embeds = wrapper.contrastive_features(batch)
        l_nce = info_nce(image_embeds, text_embeds, self.cfg.temperature)

        loss = l_gen + self.cfg.contrastive_weight * l_nce
        return LossOutput(
            loss=loss,
            components={
                "loss": loss.item(),
                "l_generative": l_gen.item(),
                "l_infonce": l_nce.item(),
            },
        )
