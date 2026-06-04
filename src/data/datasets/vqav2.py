"""VQAv2 — open-ended natural-image VQA (RQ2 baseline "natural image" domain).

A ~50K subset of VQAv2 for baseline reasoning evaluation (proposal §7.1). Each
question has 10 human answers; the canonical target is ``multiple_choice_answer``
(the majority vote). Open-ended, no rationale.

Schema (``HuggingFaceM4/VQAv2``)::

    image:                  PIL.Image
    question:               str
    multiple_choice_answer: str          # majority human answer → the target
    answers:                list[dict]    # 10 annotator answers (for VQA-accuracy)
"""

from __future__ import annotations

from typing import Any

from datasets import load_dataset

from src.data.base import BaseVLMDataset
from src.data.example import VLMExample
from src.data.datasets import DATASETS


@DATASETS.register("vqav2")
class VQAv2Dataset(BaseVLMDataset):
    hf_path = "HuggingFaceM4/VQAv2"
    is_multiple_choice = False
    has_gold_explanation = False

    def _load_raw(self, split: str):
        return load_dataset(self.path, split=split)

    def to_example(self, row: dict[str, Any]) -> VLMExample:
        answer = row.get("multiple_choice_answer") or ""
        raw_answers = row.get("answers") or []
        # answers may be a list[dict{answer:..}] or list[str] depending on mirror
        variants = [
            a["answer"] if isinstance(a, dict) else a for a in raw_answers
        ]
        return VLMExample(
            image=row["image"],
            question=row["question"],
            answer=answer,
            metadata={
                "domain": "natural_image",
                "id": row.get("question_id") or row.get("questionId") or row.get("id"),
                "image_id": row.get("image_id"),
                "answers": variants,
            },
        )
