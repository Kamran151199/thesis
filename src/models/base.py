"""
src.models.base — the wrapper contract every backbone implements.

WHY A WRAPPER AT ALL?
---------------------
BLIP-2, Qwen2-VL and PaliGemma have different classes, processors, module names,
and freeze policies. The trainer must not care. So each backbone is wrapped to
expose ONE interface::

    wrapper.model       # the (quantized + LoRA) HF module the optimizer trains
    wrapper.processor   # image preprocessing + tokenizer for the collator
    wrapper.forward(batch)        → outputs with .loss / .logits
    wrapper.generate(batch, ...)  → token ids
    wrapper.device / wrapper.dtype

THE SHARED LOAD PIPELINE (implemented here once, for all backbones)
-------------------------------------------------------------------
::

    ModelConfig
        │  build_bnb_config()        ← 4-bit NF4   (if use_qlora)
        ▼
    from_pretrained(quantization_config=…, torch_dtype=bf16, device_map=auto)
        │  _apply_freeze()           ← hard-freeze vision tower etc.
        │  apply_qlora()             ← rank-16 adapters on target_modules
        ▼
    ready-to-train model

A subclass supplies only the backbone-specific facts: the model class, the
processor class, the default LoRA target module names, and the default freeze
list. Everything above is inherited.
"""

from __future__ import annotations

from abc import ABC
from typing import Any

import torch
import torch.nn as nn

from src.config.schema import ModelConfig
from src.models.peft_lora import apply_qlora
from src.models.quantization import build_bnb_config
from src.utils.devices import resolve_dtype
from src.utils.logging import get_logger

log = get_logger(__name__)


class BaseVLMWrapper(ABC):
    """Base class for backbone wrappers. Subclass and set the four class attrs.

    Class attributes (set by each backbone)
    ----------------------------------------
    model_cls:
        The HF model class, e.g. ``Blip2ForConditionalGeneration``.
    processor_cls:
        The HF processor class, e.g. ``Blip2Processor``.
    default_lora_targets:
        Module-name substrings to attach LoRA adapters to when the config says
        ``target_modules: auto``.
    default_freeze:
        Module-name prefixes to hard-freeze (e.g. the vision encoder) regardless
        of LoRA — applied before adapters are added.
    """

    model_cls: type = None  # type: ignore[assignment]
    processor_cls: type = None  # type: ignore[assignment]
    default_lora_targets: list[str] = []
    default_freeze: list[str] = []

    def __init__(self, cfg: ModelConfig):
        if self.model_cls is None or self.processor_cls is None:
            raise TypeError(
                f"{type(self).__name__} must set model_cls and processor_cls."
            )
        self.cfg = cfg
        self.processor = self._load_processor()
        self.model = self._build_model()

    # ── overridable hooks ──────────────────────────────────────────────────────
    def _load_processor(self) -> Any:
        return self.processor_cls.from_pretrained(self.cfg.pretrained)

    def _from_pretrained_kwargs(self) -> dict[str, Any]:
        """Extra kwargs for ``from_pretrained`` (e.g. Qwen2-VL pixel bounds).
        Override in a backbone that needs them; default is none."""
        return {}

    def resolve_lora_targets(self) -> list[str]:
        """``"auto"`` → the backbone's defaults; otherwise the configured list."""
        t = self.cfg.lora.target_modules
        return self.default_lora_targets if t == "auto" else list(t)

    # ── the shared load pipeline ────────────────────────────────────────────────
    def _build_model(self) -> nn.Module:
        cfg = self.cfg
        is_quantized = cfg.use_qlora and cfg.quantization.load_in_4bit

        kwargs: dict[str, Any] = {
            "torch_dtype": resolve_dtype(cfg.dtype),
            "device_map": cfg.device_map,
        }
        if is_quantized:
            kwargs["quantization_config"] = build_bnb_config(cfg.quantization)
        kwargs.update(self._from_pretrained_kwargs())

        log.info(
            "loading %s (qlora=%s, 4bit=%s)",
            cfg.pretrained,
            cfg.use_qlora,
            is_quantized,
        )
        model = self.model_cls.from_pretrained(cfg.pretrained, **kwargs)

        self._apply_freeze(model)

        if cfg.use_qlora and cfg.lora.enabled:
            model = apply_qlora(
                model, cfg.lora, self.resolve_lora_targets(), is_quantized=is_quantized
            )
        else:
            # Non-LoRA path (e.g. the proven BLIP-2 "train the Q-Former" mode):
            # whatever wasn't frozen above trains at full precision.
            self._log_param_counts(model)
        return model

    def _apply_freeze(self, model: nn.Module) -> None:
        """Set ``requires_grad=False`` on params whose name starts with any
        configured/default freeze prefix (e.g. ``vision_model``)."""
        prefixes = self.cfg.freeze or self.default_freeze
        if not prefixes:
            return
        for name, p in model.named_parameters():
            if any(name.startswith(pre) for pre in prefixes):
                p.requires_grad_(False)
        log.info("hard-froze module prefixes: %s", prefixes)

    @staticmethod
    def _log_param_counts(model: nn.Module) -> None:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log.info(
            "trainable %s / %s (%.2f%%)",
            f"{trainable:,}",
            f"{total:,}",
            100 * trainable / max(total, 1),
        )

    # ── interface used by trainer / evaluator ──────────────────────────────────
    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return resolve_dtype(self.cfg.dtype)

    # ── encoding hooks (the backbone-specific part of "examples → tensors") ─────
    # The collator and evaluator call these instead of the processor directly,
    # because HOW image tokens get inserted differs per backbone:
    #   • BLIP-2 / PaliGemma: the processor inserts image tokens from images=+text=
    #     → the default hooks below "just work".
    #   • Qwen2-VL: image tokens come from a chat template → it overrides
    #     prompt_to_model_text to wrap the text accordingly.

    def prompt_to_model_text(
        self, text: str, add_generation_prompt: bool = False
    ) -> str:
        """Transform plain text into the exact string the processor expects.

        Default: identity. Override in backbones whose processor needs a chat
        template (Qwen2-VL) to place the image placeholder token.
        """
        return text

    def build_inputs(
        self,
        images: list,
        texts: list[str],
        *,
        padding: bool = False,
        truncation: bool = True,
        max_length: int = 512,
        for_generation: bool = False,
    ):
        """Encode (images, texts) into model-ready tensors (a ``BatchEncoding``).

        Single place that turns text+image into ``input_ids``/``pixel_values``/…
        so the collator and scorer stay backbone-agnostic. ``for_generation``
        toggles the assistant-turn marker for chat-template backbones.
        """
        proc_texts = [
            self.prompt_to_model_text(t, add_generation_prompt=for_generation)
            for t in texts
        ]
        return self.processor(
            images=images,
            text=proc_texts,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )

    def input_length(
        self, image, text: str, max_length: int = 512, for_generation: bool = False
    ) -> int:
        """Token length of ``text`` **including** image placeholder tokens — the
        single source of truth for prompt/context masking (see the collator)."""
        enc = self.build_inputs(
            [image],
            [text],
            padding=False,
            truncation=True,
            max_length=max_length,
            for_generation=for_generation,
        )
        return int(enc["input_ids"].shape[1])

    def contrastive_features(self, batch: dict[str, Any]):
        """Return ``(image_embeds, text_embeds)`` of shape ``(B, D)`` for the
        contrastive (InfoNCE) objective — pooled visual features and answer
        sentence embeddings.

        Backbone-specific (the feature locations differ across BLIP-2 / Qwen2-VL
        / PaliGemma), so the base class refuses rather than guess. Implement it
        in your wrapper to run the RQ2 contrastive arm; the generative and
        explanation-aware objectives do not need it.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.contrastive_features is not implemented. "
            "Return (image_embeds, text_embeds) as (B, D) tensors — e.g. the "
            "pooled vision-encoder output and the answer's mean token embedding. "
            "Only the 'contrastive' objective (RQ2) requires this hook."
        )

    def forward(self, batch: dict[str, Any]):
        """Standard forward. ``batch`` must contain ``labels`` for a loss."""
        return self.model(**batch)

    @torch.no_grad()
    def generate(self, batch: dict[str, Any], **gen_kwargs):
        return self.model.generate(**batch, **gen_kwargs)  # type: ignore[attr-defined]

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def trainable_parameters(self):
        """Iterator over params with ``requires_grad`` — feed to the optimizer."""
        return (p for p in self.model.parameters() if p.requires_grad)

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())
