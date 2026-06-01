"""Cross-modal retrieval metrics — R@K and MRR (RQ2's contrastive evaluation).

Retrieval doesn't fit the per-prediction metric interface (it scores a whole
similarity matrix at once), so it's a standalone function. The RQ2 retrieval
eval builds an ``(N, D)`` image-embedding matrix and an ``(N, D)`` text-embedding
matrix (via ``wrapper.contrastive_features``), then calls this.

WHAT "RANK" MEANS HERE (not matrix rank!)
-----------------------------------------
For image ``i``, sort all ``N`` texts by similarity. The *rank* is the position
of the TRUE text ``i`` in that sorted list (1 = top). Then:

    R@K  = fraction of images whose true text is in the top-K   (higher = better)
    MRR  = mean of 1/rank                                        (higher = better)

::

    image 0 → sorted texts: [t0 ✓, t7, t3, …]   rank 1  → in R@1, R@5, R@10
    image 1 → sorted texts: [t4, t9, t1 ✓, …]   rank 3  → in R@5, R@10 (not R@1)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def retrieval_recall(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Image→text retrieval R@K and MRR over a batch of paired embeddings.

    Parameters
    ----------
    image_embeds, text_embeds:
        ``(N, D)`` each; row ``i`` of one is the true match for row ``i`` of the
        other. L2-normalized inside.
    ks:
        Which cutoffs to report (default 1, 5, 10 per §7.5).

    Returns
    -------
    ``{"R@1": .., "R@5": .., "R@10": .., "MRR": ..}``.
    """
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    sims = image_embeds @ text_embeds.t()  # (N, N)
    n = sims.size(0)

    # rank of the diagonal (true) entry within each row, sorted descending.
    order = sims.argsort(dim=-1, descending=True)
    truth = torch.arange(n, device=sims.device).unsqueeze(1)
    ranks = (order == truth).float().argmax(dim=-1) + 1  # 1-indexed rank per row

    out = {f"R@{k}": float((ranks <= k).float().mean()) for k in ks}
    out["MRR"] = float((1.0 / ranks.float()).mean())
    return out
