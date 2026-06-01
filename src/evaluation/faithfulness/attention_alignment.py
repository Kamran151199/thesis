"""
Attention-alignment faithfulness (RQ5) — STRUCTURED SCAFFOLD.

THE METRIC (proposal §faithfulness)
-----------------------------------
"Attention-alignment score between the model's attention over image patches and
the regions referenced by the explanation." Operationally::

    1. Run a forward pass with output_attentions=True; collect cross-modal
       attention from the generated explanation tokens → image patch tokens.
    2. Aggregate per patch → a saliency map A over the image grid.
    3. Compare A against an "evidence map" E (where the explanation actually
       refers — from grounding boxes, or the high-drift regions from
       masking_consistency as a proxy).
    4. Score = alignment(A, E)  (e.g. cosine, or IoU of top-k patches).

WHY A SCAFFOLD AND NOT FULLY IMPLEMENTED
----------------------------------------
Step 1 is backbone-specific: where cross-modal attention lives, how image patch
tokens are indexed in the sequence, and which layers to read differ across
BLIP-2 (Q-Former cross-attention), Qwen2-VL (merged self-attention), and
PaliGemma. Implementing it blind would be a guess. The function signatures and
the aggregation/alignment math below are real; fill in ``extract_image_attention``
for your chosen backbone (start with BLIP-2's Q-Former cross-attentions, which
are the cleanest), then the rest runs.

A model-agnostic alternative that works TODAY is in ``masking_consistency.py``
(perturb regions, measure answer drift) — use it to make progress on RQ5 while
the attention path is wired up.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def extract_image_attention(wrapper, batch: dict) -> torch.Tensor:
    """Return a per-patch attention saliency map ``(B, n_patches)``.

    BACKBONE-SPECIFIC — not implemented generically. To implement for BLIP-2:
      - call the model with ``output_attentions=True``,
      - take the Q-Former cross-attentions (queries → image patches),
      - average over heads and query tokens → one weight per patch.
    """
    raise NotImplementedError(
        "extract_image_attention is backbone-specific. Implement it for your "
        "backbone (BLIP-2 Q-Former cross-attentions are the easiest entry point), "
        "or use masking_consistency.region_importance for a model-agnostic "
        "faithfulness signal in the meantime."
    )


def alignment_score(attention_map: torch.Tensor, evidence_map: torch.Tensor) -> float:
    """Cosine alignment between an attention map and an evidence map.

    Both ``(n_patches,)`` (or broadcastable). 1.0 = the model attends exactly
    where the evidence is; ~0 = attention is unrelated to the cited evidence.
    This part IS implemented — it's the comparison once you have both maps.
    """
    a = F.normalize(attention_map.flatten().float(), dim=0)
    e = F.normalize(evidence_map.flatten().float(), dim=0)
    return float((a * e).sum())
