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
import torch.nn.functional as F

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


def per_row_neg_ce(logits: torch.Tensor, labels: torch.Tensor) -> list[float]:
    """Per-row, length-normalized −cross-entropy — the batched analogue of
    ``score_continuation``'s ``−loss``.

    Why not just use ``model(labels=…).loss``? Because HF averages that loss over
    **all** non-masked tokens in the *whole batch* — fine for one row, but for a
    batch of independent items it blends their scores into one number. So we
    compute the mean CE **per row** by hand::

        shift one position (predict tokenₜ₊₁ from posₜ)
          → token CE with reduction="none", ignore_index=-100
          → sum over each row ÷ that row's scored-token count
          → negate

    With a single row this returns exactly what ``score_continuation`` does, so
    batched and per-item evaluation agree.
    """
    shift_logits = logits[:, :-1, :].float()  # float32 for a stable softmax
    shift_labels = labels[:, 1:]
    b, tm1, vocab = shift_logits.shape
    ce = F.cross_entropy(
        shift_logits.reshape(-1, vocab),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view(b, tm1)
    valid = (shift_labels != -100).sum(dim=1).clamp(min=1)  # tokens scored per row
    per_row = ce.sum(dim=1) / valid
    return (-per_row).tolist()


@torch.no_grad()
def score_batch(
    wrapper,
    images: list,
    contexts: list[str],
    continuations: list[str],
    max_length: int = 1024,
) -> list[float]:
    """Batched :func:`score_continuation`: one continuation scored per row.

    Row ``i`` is ``contexts[i] + " " + continuations[i]`` conditioned on
    ``images[i]``. We **right-pad** (forced, regardless of the tokenizer default)
    so every context stays flush at the front — then masking each row's context
    prefix is just ``labels[i, :ctx_len] = -100``, exactly as the single-item
    path. One forward, one score per row.
    """
    fulls = [c + " " + cont for c, cont in zip(contexts, continuations)]

    tok = wrapper.processor.tokenizer
    old_side = tok.padding_side
    tok.padding_side = "right"  # context-then-continuation must sit at the front
    try:
        enc = wrapper.build_inputs(
            images, fulls, padding=True, truncation=True, max_length=max_length
        )
    finally:
        tok.padding_side = old_side
    enc = move_to_device(dict(enc), wrapper.device, wrapper.dtype)

    labels = enc["input_ids"].clone()
    for i, ctx in enumerate(contexts):
        ctx_len = wrapper.input_length(images[i], ctx, max_length)  # incl. image tokens
        labels[i, :ctx_len] = -100
    pad_id = tok.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100

    logits = wrapper.model(**enc).logits
    return per_row_neg_ce(logits, labels)


@torch.no_grad()
def generate_continuation(
    wrapper,
    ex: VLMExample,
    template: PromptTemplate,
    max_new_tokens: int = 160,
    max_length: int = 1024,
) -> str:
    """Generate the model's continuation of the prompt (its reasoning + answer).

    Returns the text AFTER the prompt, with any echoed prompt stripped. The
    caller splits it into reasoning / answer on the "Answer:" cue.

    ``max_length`` caps the *prompt* encoding (image tokens + text); the default
    1024 clears every backbone's image-token block (BLIP-2 ≈ 32, Qwen2-VL ≈ 320).
    The evaluator passes ``EvalConfig.max_length`` here.
    """
    prompt = template.prompt(ex)
    image = ex.image.convert("RGB")
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


@torch.no_grad()
def generate_batch(
    wrapper,
    examples: list[VLMExample],
    template: PromptTemplate,
    max_new_tokens: int = 160,
    max_length: int = 1024,
) -> list[str]:
    """Batched :func:`generate_continuation` — the throughput win.

    Autoregressive decoding is the eval bottleneck; one batched ``generate`` keeps
    the GPU busy where the per-item loop left it idle. Generation must
    **left-pad** (every row's real tokens end flush against the first generated
    token, so the model never decodes from a pad), so we flip ``padding_side`` for
    the encode and restore it after. Each row is prompt-stripped exactly as the
    single-item path, keeping the two consistent.
    """
    prompts = [template.prompt(ex) for ex in examples]
    images = [ex.image.convert("RGB") for ex in examples]

    tok = wrapper.processor.tokenizer
    old_side = tok.padding_side
    tok.padding_side = "left"  # decoder-only generation pads on the LEFT
    try:
        enc = wrapper.build_inputs(
            images,
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            for_generation=True,
        )
        enc = move_to_device(dict(enc), wrapper.device, wrapper.dtype)
        out_ids = wrapper.model.generate(**enc, max_new_tokens=max_new_tokens)
    finally:
        tok.padding_side = old_side

    texts = tok.batch_decode(out_ids, skip_special_tokens=True)
    return [
        (t[len(p) :].strip() if t.startswith(p) else t.strip())
        for p, t in zip(prompts, texts)
    ]


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
