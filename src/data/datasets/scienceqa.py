"""ScienceQA — the proven, fully-working loader (RQ2 / RQ3 / RQ5 workhorse).

~21K multimodal science questions, each multiple-choice **with a gold
natural-language explanation** (``solution``). That explanation is gold dust for
this thesis: it's the supervision signal for explanation-aware training (RQ3)
and the reference for faithfulness analysis (RQ5).

Schema (``derek-thomas/ScienceQA``)::

    question:  str
    choices:   list[str]          # the options
    answer:    int                # index of the correct option
    solution:  str                # natural-language rationale  ← explanation
    image:     PIL.Image | None   # many rows are text-only → filtered out
"""

from __future__ import annotations

from typing import Any

from datasets import load_dataset

from src.data.base import BaseVLMDataset
from src.data.example import VLMExample
from src.data.datasets import DATASETS


@DATASETS.register("scienceqa")
class ScienceQADataset(BaseVLMDataset):
    hf_path = "derek-thomas/ScienceQA"
    is_multiple_choice = True
    has_gold_explanation = True

    def _load_raw(self, split: str):
        return load_dataset(self.path, split=split)

    def _keep(self, row: dict[str, Any]) -> bool:
        # Must have an image AND a non-empty rationale — the prototype's filter.
        return (
            row.get("image") is not None
            and (row.get("solution") or "").strip() != ""
        )

    def to_example(self, row: dict[str, Any]) -> VLMExample:
        answer_idx = int(row["answer"])
        choices = list(row["choices"])
        return VLMExample(
            image=row["image"],
            question=row["question"],
            choices=choices,
            answer_index=answer_idx,
            answer=choices[answer_idx],
            explanation=row.get("solution"),
            metadata={
                "id": row.get("pid") or row.get("id") or row.get("question_id"),
                "subject": row.get("subject"),
                "topic": row.get("topic"),
            },
        )
