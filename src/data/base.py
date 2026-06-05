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
import hashlib
import json
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
            # Seed offset by split keeps train/eval subset choice reproducible.
            offset = 0 if split == cfg.split_train else 1
            raw = raw.shuffle(seed=42 + offset)
            if (
                split != cfg.split_train
                and cfg.avoid_train_eval_overlap
                and cfg.split_train
            ):
                raw = self._select_eval_without_train_overlap(raw, max_n)
            else:
                raw = raw.select(range(max_n))

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

    def _select_eval_without_train_overlap(self, eval_raw, max_n: int):
        """Select eval rows while skipping examples duplicated in capped train.

        Some public mirrors contain duplicate-looking VQA rows across official
        splits. With small capped subsets, even a few duplicated questions can
        noticeably bias a 200-example evaluation. We therefore compare the eval
        candidates against the exact shuffled/capped train subset used by the
        run and skip matching rows before taking ``max_eval`` examples.
        """
        train_raw = self._load_raw(self.cfg.split_train)
        train_raw = train_raw.filter(self._keep, num_proc=self.cfg.num_proc)
        if self.cfg.max_train is not None and self.cfg.max_train < len(train_raw):
            train_raw = train_raw.shuffle(seed=42).select(range(self.cfg.max_train))

        train_keys = {self._example_key(self.to_example(row)) for row in train_raw}
        selected: list[int] = []
        skipped = 0
        for idx, row in enumerate(eval_raw):
            if self._example_key(self.to_example(row)) in train_keys:
                skipped += 1
                continue
            selected.append(idx)
            if len(selected) >= max_n:
                break

        if len(selected) < max_n:
            log.warning(
                "%s[%s]: only %d non-overlapping eval rows available for cap %d",
                type(self).__name__,
                self.split,
                len(selected),
                max_n,
            )
        if skipped:
            log.info(
                "%s[%s]: skipped %d eval rows duplicated in capped train subset",
                type(self).__name__,
                self.split,
                skipped,
            )
        return eval_raw.select(selected)

    @staticmethod
    def _example_key(ex: VLMExample) -> str:
        meta = getattr(ex, "metadata", None) or {}
        for field in ("id", "question_id", "qid", "questionId", "question_id_str"):
            value = meta.get(field)
            if value is not None:
                return f"{field}:{value}"
        payload = {
            "question": ex.question,
            "answer": ex.answer,
            "choices": ex.choices,
            "explanation": ex.explanation,
            "image_size": getattr(ex.image, "size", None),
        }
        raw = json.dumps(payload, sort_keys=True, default=str)
        return "fingerprint:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]

    # ── sequence protocol → list[VLMExample] ───────────────────────────────────
    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> VLMExample:
        return self.to_example(self._rows[idx])

    def examples(self) -> list[VLMExample]:
        """Materialize the whole split as a list (handy for eval loops)."""
        return [self[i] for i in range(len(self))]
