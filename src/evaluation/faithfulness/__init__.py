"""
src.evaluation.faithfulness — RQ5: are the explanations grounded or hallucinated?

Accuracy says whether the answer is right; faithfulness asks whether the
*reasoning* is honestly tied to the image. Two complementary probes:

    masking_consistency   model-agnostic, runs TODAY — mask image regions,
                          measure how much the answer's likelihood drops. Big
                          drop on relevant regions ⇒ grounded.
    attention_alignment   structured scaffold — compare attention-over-patches
                          to where the explanation actually refers. Needs
                          per-backbone attention extraction wired in.

See ``README.md`` in this folder for how these map to the proposal's metrics.
"""

from src.evaluation.faithfulness.attention_alignment import (
    alignment_score,
    extract_image_attention,
)
from src.evaluation.faithfulness.masking_consistency import (
    MaskingResult,
    mask_region,
    region_importance,
)

__all__ = [
    "region_importance",
    "mask_region",
    "MaskingResult",
    "alignment_score",
    "extract_image_attention",
]
