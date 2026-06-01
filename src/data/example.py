"""
src.data.example — ONE sample shape for FIVE very different datasets.

The thesis touches five benchmarks with incompatible columns::

    ScienceQA   question, choices[], answer(idx), solution(=explanation), image
    A-OKVQA     question, choices[], correct_choice_idx, rationales[],    image
    ChartQA     query,    label[] (open-ended, no choices, no rationale),  image
    DocVQA      question, answers[] (open-ended, ANLS-scored),             image
    VQAv2       question, answers[] (open-ended),                          image

If the training loop branched on dataset name, it would be a mess. Instead every
loader maps its rows into ONE neutral container — ``VLMExample`` — and the rest
of the pipeline (prompts, collator, trainer, evaluator) only ever sees this.

    ┌─ ScienceQA row ─┐
    │ choices=[a,b,c] │──┐
    │ answer=1        │  │   to_example()      ┌──────────────────────┐
    └─────────────────┘  ├───────────────────▶ │     VLMExample       │
    ┌─ ChartQA row ───┐  │                     │ image, question,     │
    │ query="...",    │──┘                     │ answer, choices?,    │
    │ label=["42"]    │                        │ answer_index?,       │
    └─────────────────┘                        │ explanation?         │
                                               └──────────────────────┘

The ``?`` fields are the only branch the downstream code needs: if ``choices``
is present it's a multiple-choice task (score each option); if not it's
open-ended (generate + string-match).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass
class VLMExample:
    """One vision-language QA sample, dataset-agnostic.

    Attributes
    ----------
    image:
        The PIL image (already RGB, or convertible). Never ``None`` — loaders
        filter image-less rows out.
    question:
        The natural-language question.
    answer:
        The gold answer **as text** (always present). For multiple-choice this
        equals ``choices[answer_index]``; for open-ended it's the reference
        string (e.g. ``"42"`` for a ChartQA value).
    choices:
        The option strings for a multiple-choice task, else ``None``. Presence
        of this field is the single signal that flips evaluation between
        "score each choice" (MC) and "generate then string-match" (open-ended).
    answer_index:
        Index of the correct option in ``choices`` (MC only). The integer the
        evaluator compares its ``argmax`` against.
    explanation:
        Gold rationale / chain-of-thought, if the dataset provides one
        (ScienceQA ``solution``, A-OKVQA ``rationales``). This is the
        supervision signal for explanation-aware training (RQ3) and the
        reference for explanation-quality metrics. ``None`` ⇒ no rationale, so
        only answer-only / generative objectives apply.
    metadata:
        Free-form passthrough (question id, domain tag, original answer list…)
        used by domain-specific metrics like DocVQA ANLS, which needs the full
        answer-variant list.

    Example
    -------
    >>> ex = VLMExample(
    ...     image=img, question="What gas do plants absorb?",
    ...     choices=["Oxygen", "Carbon dioxide", "Nitrogen"], answer_index=1,
    ...     answer="Carbon dioxide",
    ...     explanation="Plants take in CO2 for photosynthesis.")
    >>> ex.is_multiple_choice
    True
    """

    image: "Image"
    question: str
    answer: str
    choices: list[str] | None = None
    answer_index: int | None = None
    explanation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_multiple_choice(self) -> bool:
        """True iff the sample carries discrete options to score against."""
        return self.choices is not None and len(self.choices) > 0

    @property
    def has_explanation(self) -> bool:
        """True iff a gold rationale is available for explanation-aware training."""
        return bool(self.explanation and self.explanation.strip())
