"""ChartQA — open-ended question answering over charts (RQ2/RQ4 "chart" domain).

~32K questions over bar/line/pie charts. Answers are short strings, often
numeric ("42", "3.5%"). No options, no gold rationale → open-ended task scored
by **relaxed accuracy** (numeric answers within 5%; see metrics). For RQ3 the
explanation must be synthesized by a teacher model (proposal §7.2) — until then
this dataset trains the generative/answer-only objective only.

Schema (``HuggingFaceM4/ChartQA``)::

    image: PIL.Image
    query: str          # the question
    label: list[str]    # the answer(s), usually one element

VERIFY on first load: some mirrors name the columns ``question`` / ``answer``
instead. Adjust ``to_example`` if a KeyError fires.
"""

from __future__ import annotations

from typing import Any

from datasets import load_dataset

from src.data.base import BaseVLMDataset
from src.data.example import VLMExample
from src.data.datasets import DATASETS


@DATASETS.register("chartqa")
class ChartQADataset(BaseVLMDataset):
    hf_path = "HuggingFaceM4/ChartQA"
    is_multiple_choice = False
    has_gold_explanation = False

    def _load_raw(self, split: str):
        return load_dataset(self.path, split=split)

    def to_example(self, row: dict[str, Any]) -> VLMExample:
        question = row.get("query") or row.get("question")
        label = row.get("label") or row.get("answer")
        answer = label[0] if isinstance(label, (list, tuple)) else str(label)
        return VLMExample(
            image=row["image"],
            question=question,
            answer=answer,
            metadata={
                "domain": "chart",
                "id": row.get("id") or row.get("question_id") or row.get("image_id"),
                # keep all gold variants for relaxed-accuracy scoring
                "answers": list(label) if isinstance(label, (list, tuple)) else [answer],
            },
        )
