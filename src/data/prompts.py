"""
src.data.prompts — the prompt template IS the RQ3 experiment.

RQ3 asks: *does training the model to explain before answering improve accuracy
and explanation quality?* You answer it by changing ONE thing — the supervision
target — while holding the dataset, backbone, and hyperparameters fixed:

    answer_only            target = " Answer: Carbon dioxide."
    explanation_then_answer target = " Reasoning: Plants take in CO2 … Answer: Carbon dioxide."

Same image, same question, same everything else. The only difference is whether
the gold target contains a reasoning span. That clean swap is what makes the
comparison a controlled experiment instead of a confound.

THE PROMPT / TARGET SPLIT (and why it matters for the loss)
-----------------------------------------------------------
Every template produces two strings::

    prompt  = "Question: <q> Options: (A) .. (B) .. (C) .."   ← model SEES this (masked in loss)
    target  = " Reasoning: <r> Answer: <a>."                  ← model PREDICTS this (supervised)

The collator concatenates them, then masks the prompt span to ``-100`` so the
loss only scores the target. (See ``src.data.collator``.)

WHY "Answer:" COMES LAST, EVEN WITH REASONING
---------------------------------------------
A subtle bug we hit: an earlier template put "Answer:" *before* the reasoning,
so the cue word "Answer:" was followed by an explanation, not the answer. At
eval we scored a bare choice right after "Answer:" → train/eval mismatch.
Fix: "Answer:" always *immediately precedes the real answer*. The model learns
reasoning → "Answer:" → answer, and eval mirrors it exactly.
"""

from __future__ import annotations

from src.data.example import VLMExample
from src.registry import Registry

#: name → PromptTemplate subclass. Selected by ``DataConfig.prompt_variant``.
PROMPT_VARIANTS: Registry["PromptTemplate"] = Registry("prompt_variant")


def _format_options(choices: list[str]) -> str:
    """``["Oxygen", "CO2"]`` → ``"(A) Oxygen, (B) CO2"`` (the proven format)."""
    return ", ".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))


class PromptTemplate:
    """Base class: turn a :class:`VLMExample` into (prompt, target) text.

    Subclasses override ``target`` (and rarely ``prompt``). The two are kept
    separate so the collator knows exactly which span to supervise.
    """

    #: True if this template's target contains a reasoning span — drives whether
    #: eval must generate-then-score (CoT) or can score directly.
    produces_reasoning: bool = False

    def prompt(self, ex: VLMExample) -> str:
        """Question (+ options for MC). Ends *without* a cue word; the target
        supplies the first cue ("Reasoning:" or "Answer:")."""
        if ex.is_multiple_choice:
            return f"Question: {ex.question} Options: {_format_options(ex.choices)}."
        return f"Question: {ex.question}"

    def target(self, ex: VLMExample) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def target_spans(self, ex: VLMExample) -> list[tuple[str, str]]:
        """Ordered ``(span_name, span_text)`` pairs whose concatenation == target.

        Lets the collator tag each target token as "explanation" or "answer" so
        the explanation-aware objective can weight them separately
        (``L = α·L_answer + (1−α)·L_explanation``). Default: the whole target is
        one ``"answer"`` span (correct for answer-only training).

        Span names must be in ``{"explanation", "answer"}``.
        """
        return [("answer", self.target(ex))]

    def __call__(self, ex: VLMExample) -> tuple[str, str]:
        return self.prompt(ex), self.target(ex)


@PROMPT_VARIANTS.register("answer_only")
class AnswerOnlyTemplate(PromptTemplate):
    """Target = just the answer. This is the clean no-rationale control.

    >>> AnswerOnlyTemplate().target(ex)
    ' Answer: Carbon dioxide.'
    """

    produces_reasoning = False

    def target(self, ex: VLMExample) -> str:
        return f" Answer: {ex.answer}."


@PROMPT_VARIANTS.register("explanation_then_answer")
class ExplanationThenAnswerTemplate(PromptTemplate):
    """Target = reasoning, then answer. The explanation-aware arm (α < 1.0).

    Falls back to answer-only if the example has no gold rationale, so a mixed
    dataset (some rows with explanations, some without) trains without crashing.

    >>> ExplanationThenAnswerTemplate().target(ex)
    ' Reasoning: Plants take in CO2 for photosynthesis. Answer: Carbon dioxide.'
    """

    produces_reasoning = True

    def target(self, ex: VLMExample) -> str:
        if not ex.has_explanation:
            return f" Answer: {ex.answer}."
        return f" Reasoning: {ex.explanation.strip()} Answer: {ex.answer}."

    def target_spans(self, ex: VLMExample) -> list[tuple[str, str]]:
        # Split at " Answer:" → reasoning tokens vs answer tokens, so the
        # explanation-aware loss can weight them with α. The concatenation of
        # the two span texts must exactly reproduce ``target(ex)``.
        if not ex.has_explanation:
            return [("answer", f" Answer: {ex.answer}.")]
        return [
            ("explanation", f" Reasoning: {ex.explanation.strip()}"),
            ("answer", f" Answer: {ex.answer}."),
        ]


def build_template(variant: str) -> PromptTemplate:
    """Instantiate the template named by ``DataConfig.prompt_variant``."""
    return PROMPT_VARIANTS.build(variant)
