"""
src.evaluation.scoring — likelihood scoring + generation (the proven eval core).

Two primitives the evaluator builds on, both ported from the working prototype.

1. SCORING — "how likely is this continuation?"
   ::

       score = −(mean cross-entropy of `continuation` given `context`)

   Higher = the model finds the continuation more probable. For multiple-choice
   we score each option in the same context and take the argmax. This is robust
   where string-matching the generated answer is gameable (the model echoes the
   prompt, which lists all the options).

2. GENERATION — "let the model write its chain of thought."
   We feed the prompt, generate, strip the echoed prompt, and split on "Answer:"
   into (reasoning, answer). The reasoning conditions the choice scoring (CoT
   eval) and is itself the object RQ5 analyzes for faithfulness.

THE MASKING MUST MATCH TRAINING. ``context`` length is measured with the same
image-token-aware ``wrapper.input_length`` used by the collator — otherwise the
score includes context tokens and the comparison is meaningless.
"""

from __future__ import annotations

import torch

from src.data.example import VLMExample
from src.data.prompts import PromptTemplate
from src.utils.devices import move_to_device


@torch.no_grad()
def score_continuation(
    wrapper, image, context_text: str, continuation_text: str, max_length: int = 1024
) -> float:
    """Length-normalized log-likelihood of ``continuation_text`` after ``context_text``.

    Returns ``−loss`` (higher = more likely). Only the continuation tokens are
    scored; the context (image + prompt + any reasoning) is masked to ``-100``.
    """
    full = context_text + " " + continuation_text
    enc = wrapper.build_inputs(
        [image], [full], padding=False, truncation=True, max_length=max_length
    )
    enc = move_to_device(dict(enc), wrapper.device, wrapper.dtype)

    labels = enc["input_ids"].clone()
    ctx_len = wrapper.input_length(
        image, context_text, max_length
    )  # includes image tokens
    labels[:, :ctx_len] = -100
    pad_id = wrapper.processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100

    loss = wrapper.model(**enc, labels=labels).loss
    return -float(loss.item())


@torch.no_grad()
def generate_continuation(
    wrapper,
    ex: VLMExample,
    template: PromptTemplate,
    max_new_tokens: int = 160,
    max_length: int | None = None,
) -> str:
    """Generate the model's continuation of the prompt (its reasoning + answer).

    Returns the text AFTER the prompt, with any echoed prompt stripped. The
    caller splits it into reasoning / answer on the "Answer:" cue.

    ``max_length`` caps the *prompt* encoding (image tokens + text); pass the
    ``EvalConfig.max_length`` so it exceeds the backbone's image-token count
    (Qwen2-VL ≈ 320). Falls back to ``template_max_length`` when not given.
    """
    prompt = template.prompt(ex)
    image = ex.image.convert("RGB")
    if max_length is None:
        max_length = template_max_length(template)
    enc = wrapper.build_inputs(
        [image],
        [prompt],
        padding=False,
        truncation=True,
        max_length=max_length,
        for_generation=True,
    )
    enc = move_to_device(dict(enc), wrapper.device, wrapper.dtype)
    out_ids = wrapper.model.generate(**enc, max_new_tokens=max_new_tokens)
    text = wrapper.processor.tokenizer.decode(out_ids[0], skip_special_tokens=True)
    # Some models echo the prompt in the output; strip it if so.
    return text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()


def template_max_length(template: PromptTemplate) -> int:
    """Fallback prompt-encoding length cap (separate from the new-token budget).
    1024 holds a question + options + the image-token block for every backbone
    (BLIP-2 ≈ 32 image tokens, Qwen2-VL ≈ 320). Callers normally pass the
    ``EvalConfig.max_length`` instead of relying on this default."""
    return 1024


def split_reasoning_answer(continuation: str) -> tuple[str, str]:
    """Split a generated continuation into (reasoning, answer) on the "Answer:" cue.

    >>> split_reasoning_answer("Reasoning: plants need CO2. Answer: Carbon dioxide.")
    ('plants need CO2.', 'Carbon dioxide.')
    """
    if "Answer:" in continuation:
        reasoning, answer = continuation.split("Answer:", 1)
    else:
        reasoning, answer = continuation, ""
    reasoning = reasoning.strip()
    if reasoning.lower().startswith("reasoning:"):
        reasoning = reasoning[len("reasoning:") :].strip()
    return reasoning, answer.strip().rstrip(".").strip()
