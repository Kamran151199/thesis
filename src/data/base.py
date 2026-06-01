"""
src.data.base — the contract every dataset loader fulfills.

A loader's whole job: take a raw HuggingFace dataset and hand back a list of
:class:`VLMExample`. The base class does the parts that are identical for every
dataset (filtering image-less rows, shuffling, subsampling to ``max_train``);
subclasses supply only the two dataset-specific pieces:

    _load_raw(split)   →  the HF dataset object   (which hub id, which config)
    to_example(row)    →  a VLMExample            (which columns map to what)

VISUAL
------
::

    BaseVLMDataset(cfg, split)
        │  _load_raw(split)         ← subclass: load_dataset("derek-thomas/ScienceQA", ...)
        ▼
    raw HF dataset
        │  .filter(_keep)           ← drop rows with no image / no rationale
        │  .shuffle(seed).select(:max)
        ▼
    subsampled rows
        │  to_example(row)          ← subclass: row["solution"] → ex.explanation, …
        ▼
    list[VLMExample]   ── consumed by VLMCollator ──▶ model
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.config.schema import DataConfig
from src.data.example import VLMExample
from src.utils.logging import get_logger

log = get_logger(__name__)


class BaseVLMDataset(ABC):
    """Abstract loader: HF rows → ``list[VLMExample]``, with shared plumbing.

    Class attributes
    ----------------
    hf_path:
        Default HuggingFace hub id. ``DataConfig.hf_path`` overrides it.
    is_multiple_choice:
        Whether this dataset has discrete options (selects the eval mode).
    has_gold_explanation:
        Whether rows carry rationales (enables explanation-aware training).
    """

    hf_path: str = ""
    is_multiple_choice: bool = False
    has_gold_explanation: bool = False

    def __init__(self, cfg: DataConfig, split: str):
        self.cfg = cfg
        self.split = split
        self.path = cfg.hf_path or self.hf_path

        raw = self._load_raw(split)
        n_raw = len(raw)
        raw = raw.filter(self._keep, num_proc=cfg.num_proc)

        max_n = cfg.max_train if split == cfg.split_train else cfg.max_eval
        if max_n is not None and max_n < len(raw):
            # Seed offset by split keeps train/eval subsets disjoint-ish and
            # reproducible (the proven prototype used seed and seed+1).
            offset = 0 if split == cfg.split_train else 1
            raw = raw.shuffle(seed=42 + offset).select(range(max_n))

        self._rows = raw
        log.info(
            "%s[%s]: %d rows (from %d raw; kept after filter, capped at %s)",
            type(self).__name__, split, len(raw), n_raw, max_n,
        )

    # ── subclass hooks ────────────────────────────────────────────────────────
    @abstractmethod
    def _load_raw(self, split: str) -> Any:
        """Return the raw HF dataset for ``split`` (a ``datasets.Dataset``)."""

    @abstractmethod
    def to_example(self, row: dict[str, Any]) -> VLMExample:
        """Map one raw row into a :class:`VLMExample`."""

    def _keep(self, row: dict[str, Any]) -> bool:
        """Filter predicate. Default: must have a non-null image. Override to
        also require a rationale (ScienceQA does)."""
        return row.get(self.cfg.image_field) is not None

    # ── sequence protocol → list[VLMExample] ───────────────────────────────────
    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> VLMExample:
        return self.to_example(self._rows[idx])

    def examples(self) -> list[VLMExample]:
        """Materialize the whole split as a list (handy for eval loops)."""
        return [self[i] for i in range(len(self))]
