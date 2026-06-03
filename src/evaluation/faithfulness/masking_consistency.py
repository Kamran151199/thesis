"""
Evidence-masking consistency — does the model actually LOOK where it claims? (RQ5)

THE TEST (proposal §faithfulness, masking-consistency)
------------------------------------------------------
Hide a region of the image and see whether the answer's likelihood drops:

    drift(region) = score(answer | full image) − score(answer | region masked)

::

    full image          mask top-left        mask bottom-right
    ┌────┬────┐          ┌────┬────┐          ┌────┬────┐
    │ ░░ │ ░░ │          │ ▓▓ │ ░░ │          │ ░░ │ ░░ │
    ├────┼────┤   →      ├────┼────┤    →     ├────┼────┤
    │ ░░ │ ░░ │          │ ░░ │ ░░ │          │ ░░ │ ▓▓ │
    └────┴────┘          └────┴────┘          └────┴────┘
    score = -1.2          score = -3.8         score = -1.3
                          drift = +2.6 ← model RELIED on this region
                                               drift = +0.1 ← ignored it

A faithful model's high-drift regions should be the ones its explanation refers
to. A model whose answer barely moves when you blank the relevant evidence is
**hallucinating** from the language prior, not reading the image.

This core is model-agnostic (just masking + likelihood, reusing
``score_continuation``), so it runs on any backbone today. Linking high-drift
regions to explanation *tokens* is the attention-alignment side
(``attention_alignment.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image

from src.data.example import VLMExample
from src.data.prompts import PromptTemplate
from src.evaluation.scoring import score_continuation


@dataclass
class MaskingResult:
    """Per-region importance and a single consistency summary for one example."""

    drifts: list[float]  #: drift per region (row-major over the grid)
    grid: tuple[int, int]  #: (rows, cols)
    max_drift: float  #: the most-relied-on region's drift
    mean_drift: float  #: average reliance across regions


def _mask_region(img: Image.Image, r: int, c: int, rows: int, cols: int) -> Image.Image:
    """Return a copy of ``img`` with grid cell ``(r, c)`` filled with mid-gray."""
    w, h = img.size
    x0, y0 = c * w // cols, r * h // rows
    x1, y1 = (c + 1) * w // cols, (r + 1) * h // rows
    out = img.copy()
    out.paste((128, 128, 128), (x0, y0, x1, y1))
    return out


@torch.no_grad()
def region_importance(
    wrapper,
    ex: VLMExample,
    template: PromptTemplate,
    grid: tuple[int, int] = (2, 2),
    max_length: int = 1024,
) -> MaskingResult:
    """Drift in the gold answer's likelihood when each image region is masked.

    Parameters
    ----------
    wrapper:   the trained backbone wrapper.
    ex:        the example (uses its gold ``answer`` as the scored continuation).
    template:  prompt template (the context the answer is scored in).
    grid:      (rows, cols) to divide the image into.
    max_length:
        Token cap for image+text encoding. High-resolution Qwen2-VL chart/doc
        images often need 2048; using 1024 can truncate image placeholders and
        trigger an image-token mismatch.

    Returns
    -------
    :class:`MaskingResult` with per-region drift (baseline − masked score).
    """
    rows, cols = grid
    image = ex.image.convert("RGB")
    context = f"{template.prompt(ex)} Answer:"

    baseline = score_continuation(wrapper, image, context, ex.answer, max_length=max_length)
    drifts: list[float] = []
    for r in range(rows):
        for c in range(cols):
            masked = _mask_region(image, r, c, rows, cols)
            masked_score = score_continuation(
                wrapper, masked, context, ex.answer, max_length=max_length
            )
            drifts.append(baseline - masked_score)  # >0 ⇒ region mattered

    return MaskingResult(
        drifts=drifts,
        grid=grid,
        max_drift=max(drifts),
        mean_drift=sum(drifts) / len(drifts),
    )
