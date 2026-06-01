"""
src.data.collator — turn ``list[VLMExample]`` into a masked, model-ready batch.

This is the most bug-prone 30 lines in the whole pipeline, so it is also the
most heavily commented. It encodes the proven logic from the BLIP-2 prototype.

WHAT IT BUILDS (one row of the batch)
-------------------------------------
::

    input_ids:  [<img><img>…<img>  Question: …  Options: …  Reasoning: …  Answer: X .  <pad><pad>]
                └──── image ─────┘└──────── prompt ────────┘└─────────── target ──────┘└─ padding ─┘
    labels:     [ -100  -100 … -100  -100  -100 …  -100      Reasoning: …  Answer: X .   -100 -100 ]
                └──────────────── MASKED (not scored) ──────┘└──── SUPERVISED ─────────┘└─ MASKED ─┘

So cross-entropy is computed ONLY on the target span. The model is graded on
"can you produce the reasoning + answer?", never on "can you echo the prompt?".

THE BUG THIS FILE EXISTS TO PREVENT
-----------------------------------
``prompt_len`` must count the image placeholder tokens the processor silently
prepends (32 for BLIP-2; a variable count for Qwen2-VL). If you compute it from
``tokenizer(prompt)`` — text only — you undercount by the image-token count, the
mask ends early, and the **tail of the prompt leaks into the supervised target**.
We hit exactly this (decoded a label and saw "…(C) Answer:" in it). The fix:
compute ``prompt_len`` by running the prompt **through the wrapper's encoder,
with the image** (``wrapper.input_length``), so the count includes image tokens.

WHY THE WRAPPER, NOT THE PROCESSOR?
-----------------------------------
Image-token insertion is backbone-specific (BLIP-2's processor does it from
``images=``; Qwen2-VL needs a chat template). The collator therefore calls
``wrapper.build_inputs`` / ``wrapper.input_length`` and stays backbone-agnostic.

PADDING SIDE
------------
Training assumes **right-padding** (pads after the target, so target positions
keep their true indices). The collator enforces it. Generation uses left-padding
instead — see ``src.evaluation.scoring``.
"""

from __future__ import annotations

import torch

from src.data.example import VLMExample
from src.data.prompts import PromptTemplate

LABEL_IGNORE = -100  # HF's "ignore this position in the loss" sentinel

# Span codes for explanation-aware loss weighting (the optional ``span_ids``
# tensor). Each TARGET token is tagged so the objective can apply α to the
# answer span and (1−α) to the explanation span.
SPAN_IGNORE = 0  # prompt / image / padding — never scored
SPAN_EXPLANATION = 1  # the " Reasoning: …" tokens
SPAN_ANSWER = 2  # the " Answer: X." tokens
_SPAN_CODE = {"explanation": SPAN_EXPLANATION, "answer": SPAN_ANSWER}


class VLMCollator:
    """Collate examples into a padded batch with prompt-masked labels.

    Parameters
    ----------
    wrapper:
        A :class:`~src.models.base.BaseVLMWrapper` — used for its backbone-aware
        ``build_inputs`` / ``input_length`` and its processor (for the pad id).
    template:
        A :class:`~src.data.prompts.PromptTemplate` giving (prompt, target).
    max_length:
        Truncation length for the full (prompt + target) sequence.

    Returns (when called)
    ---------------------
    A dict of CPU tensors: everything the processor emits (``input_ids``,
    ``attention_mask``, ``pixel_values``, and any backbone-specific extras like
    Qwen2-VL's ``image_grid_thw``) **plus** ``labels``. The Trainer moves it to
    the device and casts floats to the model dtype.

    Example
    -------
    >>> collator = VLMCollator(wrapper, ExplanationThenAnswerTemplate(), 512)
    >>> batch = collator([ex0, ex1, ex2, ex3])
    >>> batch["labels"].shape, (batch["labels"] != -100).sum().item()
    (torch.Size([4, 96]), 41)   # only 41 target tokens are supervised
    """

    def __init__(
        self,
        wrapper,
        template: PromptTemplate,
        max_length: int = 512,
        tag_spans: bool = False,
    ):
        self.wrapper = wrapper
        self.processor = wrapper.processor
        self.template = template
        self.max_length = max_length
        #: emit a ``span_ids`` tensor tagging answer vs explanation tokens.
        #: Turned on by the Trainer when the objective needs it (explanation-aware).
        self.tag_spans = tag_spans
        # Training relies on right-padding; enforce it so a stray global setting
        # (e.g. left-padding left over from a generation cell) can't corrupt the
        # label alignment silently.
        tok = getattr(self.processor, "tokenizer", None)
        if tok is not None and hasattr(tok, "padding_side"):
            tok.padding_side = "right"

    def __call__(self, examples: list[VLMExample]) -> dict[str, torch.Tensor]:
        images = [ex.image.convert("RGB") for ex in examples]
        prompts, targets = zip(*(self.template(ex) for ex in examples))
        full_texts = [p + t for p, t in zip(prompts, targets)]

        # Encode the whole batch (image + prompt + target), padded to the longest.
        enc = self.wrapper.build_inputs(
            images, list(full_texts), padding=True, truncation=True,
            max_length=self.max_length,
        )

        labels = enc["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        seq_len = labels.shape[1]

        # prompt_len[i] = where the supervised target begins (after image+prompt).
        prompt_lens = [
            min(self.wrapper.input_length(images[i], prompts[i], self.max_length), seq_len)
            for i in range(len(examples))
        ]
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = LABEL_IGNORE  # mask image + prompt prefix
        if pad_id is not None:
            labels[labels == pad_id] = LABEL_IGNORE  # ignore padding too
        enc["labels"] = labels

        if self.tag_spans:
            enc["span_ids"] = self._build_span_ids(examples, images, prompts, prompt_lens, labels)
        return dict(enc)

    def _build_span_ids(self, examples, images, prompts, prompt_lens, labels) -> torch.Tensor:
        """Tag each TARGET token as explanation (1) or answer (2); rest = 0.

        Boundaries are found by re-encoding cumulative prefixes
        (prompt → prompt+explanation → prompt+explanation+answer) and reading
        the lengths — the same image-token-aware trick used for prompt masking.
        A token may drift by one at a span seam (subword tokenization); for an
        α-weighted *mean* loss that is negligible.
        """
        span_ids = torch.zeros_like(labels)
        seq_len = labels.shape[1]
        for i, ex in enumerate(examples):
            cursor = prompt_lens[i]
            acc = prompts[i]
            for name, text in self.template.target_spans(ex):
                acc = acc + text
                end = min(self.wrapper.input_length(images[i], acc, self.max_length), seq_len)
                span_ids[i, cursor:end] = _SPAN_CODE[name]
                cursor = end
        span_ids[labels == LABEL_IGNORE] = SPAN_IGNORE  # never tag masked positions
        return span_ids
