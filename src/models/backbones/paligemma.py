"""PaliGemma — Google's SigLIP + Gemma VLM (third backbone, cross-arch breadth).

Adds architectural diversity to the backbone axis (RQ4): a SigLIP vision encoder
+ Gemma language model, with a simple linear projection (no Q-Former, no dynamic
resolution). Useful precisely because it differs from BLIP-2 and Qwen2-VL.

Encoding: PaliGemma's processor inserts the image tokens from ``images=`` (like
BLIP-2), so the default encoding hooks apply — no override needed. It also
supports a native ``suffix=`` (answer) argument for label masking; we instead
use the shared collator masking for consistency across backbones.

# VERIFY on first run: confirm image-token count and that prefix-masking lands
# on the target only (decode a label row, as in the Qwen2-VL note).
"""

from __future__ import annotations

from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

from src.models.backbones import BACKBONES
from src.models.base import BaseVLMWrapper


@BACKBONES.register("paligemma")
class PaliGemmaWrapper(BaseVLMWrapper):
    model_cls = PaliGemmaForConditionalGeneration
    processor_cls = PaliGemmaProcessor

    # SigLIP vision tower frozen; adapt the Gemma language model.
    default_freeze = ["vision_tower"]

    # Gemma LLM attention + MLP projections.
    default_lora_targets = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
