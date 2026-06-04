"""A-OKVQA — multiple-choice commonsense VQA **with rationales** (RQ3 explanation source).

~25K questions, each with 4 choices, a correct index, AND ``rationales`` (a list
of short human explanations). Like ScienceQA, this is a full explanation-aware
dataset — choices + gold rationale — so it doubles the data available for the
α-sweep beyond ScienceQA alone (proposal §explanation harvesting).

Schema (``HuggingFaceM4/A-OKVQA``)::

    image:              PIL.Image
    question:           str
    choices:            list[str]   # 4 options
    correct_choice_idx: int
    rationales:         list[str]   # one or more human rationales

If the hub schema differs on first load, adjust the column names in
``to_example`` — the structure (MC + rationale) is what matters.
"""

from __future__ import annotations

from typing import Any

from datasets import load_dataset

from src.data.base import BaseVLMDataset
from src.data.example import VLMExample
from src.data.datasets import DATASETS


@DATASETS.register("aokvqa")
class AOKVQADataset(BaseVLMDataset):
    hf_path = "HuggingFaceM4/A-OKVQA"
    is_multiple_choice = True
    has_gold_explanation = True

    def _load_raw(self, split: str):
        return load_dataset(self.path, split=split)

    def to_example(self, row: dict[str, Any]) -> VLMExample:
        choices = list(row["choices"])
        answer_idx = int(row["correct_choice_idx"])
        rationales = row.get("rationales") or []
        explanation = rationales[0] if rationales else None
        return VLMExample(
            image=row["image"],
            question=row["question"],
            choices=choices,
            answer_index=answer_idx,
            answer=choices[answer_idx],
            explanation=explanation,
            metadata={
                "domain": "natural_image",
                "id": row.get("question_id") or row.get("questionId") or row.get("id"),
                "image_id": row.get("image_id"),
            },
        )
