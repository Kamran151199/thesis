"""DocVQA — open-ended question answering over document images (RQ2/RQ4 "document").

~50K questions over scanned documents (forms, reports, letters). Each question
has several acceptable answer strings; scored by **ANLS** (Average Normalized
Levenshtein Similarity), which rewards near-misses in OCR'd text. Open-ended, no
gold rationale (synthesize for RQ3 per proposal §7.2).

Schema (``lmms-lab/DocVQA``, config ``"DocVQA"``)::

    image:    PIL.Image
    question: str
    answers:  list[str]   # several acceptable variants → keep all for ANLS

VERIFY on first load: the ``lmms-lab/DocVQA`` build needs the config name
``"DocVQA"`` (vs ``"InfographicVQA"``). Adjust ``_load_raw`` if needed.
"""

from __future__ import annotations

from typing import Any

from datasets import load_dataset

from src.data.base import BaseVLMDataset
from src.data.example import VLMExample
from src.data.datasets import DATASETS


@DATASETS.register("docvqa")
class DocVQADataset(BaseVLMDataset):
    hf_path = "lmms-lab/DocVQA"
    is_multiple_choice = False
    has_gold_explanation = False

    def _load_raw(self, split: str):
        # lmms-lab/DocVQA ships multiple configs; "DocVQA" is the standard one.
        return load_dataset(self.path, "DocVQA", split=split)

    def to_example(self, row: dict[str, Any]) -> VLMExample:
        answers = row.get("answers") or []
        answer = answers[0] if answers else ""
        return VLMExample(
            image=row["image"],
            question=row["question"],
            answer=answer,
            metadata={
                "domain": "document",
                "id": row.get("questionId")
                or row.get("question_id")
                or row.get("id")
                or row.get("image_id"),
                "answers": list(answers),
            },
        )
