"""BLIP-2 — the proven backbone (this is the one that produced real results).

ARCHITECTURE RECAP
------------------
::

    image → [ViT vision tower] → 257 patch features
                                    │  cross-attention
                            [Q-Former: 32 learnable queries]   ← compresses 257 → 32
                                    │  language_projection (Linear)
            "Question: …" tokens ── concat ──▶ [OPT-2.7B LLM] → answer tokens

TWO TRAINING MODES (chosen by ``use_qlora`` in the config)
----------------------------------------------------------
1. **Proven full mode** (``use_qlora: false``): freeze the vision tower and the
   OPT LLM, train the Q-Former + ``language_projection`` (≈110M params, ~3%).
   This is exactly the prototype that worked — use it as the BLIP-2 baseline.
2. **QLoRA mode** (``use_qlora: true``): 4-bit base + rank-16 adapters on the
   OPT attention/MLP projections, vision frozen — the thesis-spec PEFT path.

The processor inserts the 32 image placeholder tokens from ``images=`` directly,
so the default encoding hooks in the base class need no override here.
"""

from __future__ import annotations

from transformers import Blip2ForConditionalGeneration, Blip2Processor

from src.models.backbones import BACKBONES
from src.models.base import BaseVLMWrapper


@BACKBONES.register("blip2")
class Blip2Wrapper(BaseVLMWrapper):
    model_cls = Blip2ForConditionalGeneration
    processor_cls = Blip2Processor

    # Proven freeze policy: vision tower + LLM frozen → train Q-Former + projection.
    default_freeze = ["vision_model", "language_model"]

    # For QLoRA mode: adapters on the OPT LLM attention + MLP (proposal §7.3).
    # OPT module names: q/k/v/out_proj (attention), fc1/fc2 (MLP).
    default_lora_targets = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
