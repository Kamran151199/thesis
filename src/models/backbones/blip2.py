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

The RQ2 contrastive arm keeps mode 1, but sets
``model.contrastive_projection: true`` to add one trainable OPT→Q-Former
projection for the auxiliary InfoNCE term. Plain BLIP-2 generative controls
leave that head disabled.

The processor inserts the 32 image placeholder tokens from ``images=`` directly,
so the default encoding hooks in the base class need no override here.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from src.data.constants import LABEL_IGNORE, SPAN_ANSWER
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

    def _build_model(self) -> nn.Module:
        """Load BLIP-2 and attach the trainable text projection used by RQ2.

        The projection is registered under ``model`` so the existing optimizer
        and checkpoint code include it automatically in the non-LoRA BLIP-2
        training path.
        """
        model = super()._build_model()
        if not self.cfg.contrastive_projection:
            return model

        qformer_dim = int(model.config.qformer_config.hidden_size)
        text_dim = int(getattr(model.language_model.config, "hidden_size"))
        projection = nn.Linear(text_dim, qformer_dim, bias=False)
        nn.init.normal_(projection.weight, mean=0.0, std=qformer_dim ** -0.5)

        ref = next(model.qformer.parameters())
        projection.to(device=ref.device, dtype=ref.dtype)

        model.contrastive_text_projection = projection
        self._log_param_counts(model)
        return model

    def contrastive_features(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return paired BLIP-2 image and answer embeddings for InfoNCE.

        Image representation:
            mean of the Q-Former's 32 query outputs after attending to the frozen
            ViT image features.

        Text representation:
            mean of the supervised answer-span token embeddings, projected from
            OPT hidden size into Q-Former hidden size by the trainable
            ``contrastive_text_projection`` module.
        """
        model = self.model
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        span_ids = batch.get("span_ids")
        if span_ids is None:
            raise ValueError(
                "BLIP-2 contrastive features require batch['span_ids']; "
                "ContrastiveObjective.requires_span_ids must be True."
            )
        if not hasattr(model, "contrastive_text_projection"):
            raise ValueError(
                "BLIP-2 contrastive objective requires "
                "model.contrastive_projection=true in the experiment config."
            )

        image_embeds = self._qformer_image_embedding(model, pixel_values)
        text_embeds = self._answer_text_embedding(model, labels, span_ids)
        return image_embeds, text_embeds

    @staticmethod
    def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
        ref = next(module.parameters())
        return ref.device, ref.dtype

    def _qformer_image_embedding(
        self, model: nn.Module, pixel_values: torch.Tensor
    ) -> torch.Tensor:
        vision_device, vision_dtype = self._module_device_dtype(model.vision_model)
        qformer_device, _ = self._module_device_dtype(model.qformer)

        pixel_values = pixel_values.to(device=vision_device, dtype=vision_dtype)
        with torch.no_grad():
            vision_outputs = model.vision_model(pixel_values=pixel_values, return_dict=True)
            image_embeds = vision_outputs.last_hidden_state.detach()

        image_embeds = image_embeds.to(qformer_device)
        image_attention_mask = torch.ones(
            image_embeds.shape[:-1], dtype=torch.long, device=qformer_device
        )
        query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        return query_outputs.last_hidden_state.mean(dim=1)

    def _answer_text_embedding(
        self,
        model: nn.Module,
        labels: torch.Tensor,
        span_ids: torch.Tensor,
    ) -> torch.Tensor:
        embed_layer = model.language_model.get_input_embeddings()
        embed_device, _ = self._module_device_dtype(embed_layer)
        projection = model.contrastive_text_projection
        projection_device, projection_dtype = self._module_device_dtype(projection)

        valid = labels != LABEL_IGNORE
        answer_mask = (span_ids == SPAN_ANSWER) & valid
        missing_answer = answer_mask.sum(dim=1) == 0
        if missing_answer.any():
            answer_mask = answer_mask.clone()
            answer_mask[missing_answer] = valid[missing_answer]
        if (answer_mask.sum(dim=1) == 0).any():
            raise ValueError(
                "At least one batch row has no supervised answer tokens after truncation."
            )

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        token_ids = labels.masked_fill(~valid, pad_id).to(embed_device)
        mask = answer_mask.to(embed_device)

        token_embeds = embed_layer(token_ids)
        lengths = mask.sum(dim=1).clamp_min(1).to(token_embeds.dtype)
        pooled = (token_embeds * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1)
        pooled = pooled.to(device=projection_device, dtype=projection_dtype)
        return projection(pooled)
