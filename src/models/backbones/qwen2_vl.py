"""Qwen2-VL — the proposal's PRIMARY backbone (2B primary, 7B scale ablation).

Why primary: native **dynamic resolution** (no fixed 224×224 crop), strong OCR
and chart reading, and a 2B size that fine-tunes comfortably under QLoRA on a
single A100-40GB.

THE ONE TRICKY BIT: how the image gets into the token stream
------------------------------------------------------------
BLIP-2's processor inserts image tokens automatically from ``images=``. Qwen2-VL
does NOT — it expects the text to already contain a vision-token block, which
its chat template normally adds::

    <|vision_start|><|image_pad|><|vision_end|>{your text}

The processor then expands that single ``<|image_pad|>`` into the right number
of patch tokens for the image's resolution. We replicate exactly that by
prepending the block in ``prompt_to_model_text`` — deliberately WITHOUT the
chat role markers. Why no roles? Because the collator masks the loss by
"prompt length is a prefix of the full text", and a plain ``block + prompt +
target`` keeps ``block + prompt`` a TRUE prefix of ``block + prompt + target``.
A 2-turn chat format would insert ``<|im_end|>`` between prompt and target and
break that prefix property. (If you later want full instruct formatting, make
the collator turn-aware; for QLoRA fine-tuning the raw form trains fine.)

# VERIFY on first Qwen2-VL run: decode one label row
#   processor.tokenizer.decode(batch["labels"][0][batch["labels"][0] != -100])
# and confirm it shows ONLY " Reasoning: … Answer: X." with no image/prompt leak.
"""

from __future__ import annotations

from typing import Any

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from src.models.backbones import BACKBONES
from src.models.base import BaseVLMWrapper

# Qwen2-VL's vision-token block. The lone <|image_pad|> is expanded by the
# processor into one token per visual patch (count depends on image resolution).
_VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"


@BACKBONES.register("qwen2_vl")
class Qwen2VLWrapper(BaseVLMWrapper):
    model_cls = Qwen2VLForConditionalGeneration
    processor_cls = AutoProcessor

    # Freeze the vision tower (named ``visual``); adapt the language model.
    default_freeze = ["visual"]

    # Qwen2 LLM attention + MLP projection names (proposal §7.3: attn + MLP).
    default_lora_targets = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",  # attention
        "gate_proj",
        "up_proj",
        "down_proj",  # MLP
    ]

    def _load_processor(self) -> Any:
        # Cap pixels to bound the per-image token count (and thus GPU memory).
        # 256–1280 visual tokens per image is a sane window for A100-40GB.
        return AutoProcessor.from_pretrained(
            self.cfg.pretrained,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )

    def prompt_to_model_text(
        self, text: str, add_generation_prompt: bool = False
    ) -> str:
        # Prepend the vision block so the processor knows where the image goes.
        # (add_generation_prompt is unused here: the raw format has no assistant
        # turn to prompt — the model simply continues after the prompt text.)
        return f"{_VISION_BLOCK}{text}"
