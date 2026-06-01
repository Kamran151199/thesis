"""
src.models.quantization — the "Q" in QLoRA (4-bit NF4 base weights).

WHAT QUANTIZATION BUYS YOU
--------------------------
A 7B model in bf16 is ~14 GB just for weights — before activations, gradients,
or optimizer state. That blows the A100-40GB budget once you add training
overhead. Store those frozen weights in **4-bit** instead and they shrink to
~3.5 GB. Since the base is frozen (only LoRA adapters train), 4-bit storage
costs almost no accuracy: weights are de-quantized to bf16 on the fly for each
matmul, so the *math* is still bf16 — only *storage* is 4-bit.

::

    bf16 weight (16 bits)   ████████████████   2 bytes
    NF4  weight ( 4 bits)   ████               0.5 bytes   ← 4× smaller, frozen

NF4 = NormalFloat-4: a 4-bit datatype whose 16 levels are spaced to match a
normal distribution (which is how trained weights are actually distributed), so
it loses less than naive linear 4-bit. ``double_quant`` then quantizes the
per-block scale constants too — a few more % saved for free.
"""

from __future__ import annotations

from transformers import BitsAndBytesConfig

from src.config.schema import QuantizationConfig
from src.utils.devices import resolve_dtype


def build_bnb_config(cfg: QuantizationConfig) -> BitsAndBytesConfig:
    """Translate our :class:`QuantizationConfig` into a HF ``BitsAndBytesConfig``.

    Pass the result as ``quantization_config=`` to ``from_pretrained`` and the
    base model loads in 4-bit. ``bnb_4bit_compute_dtype`` is the dtype weights
    are de-quantized to for each forward matmul (keep it bf16 on A100).

    >>> build_bnb_config(QuantizationConfig())
    BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', ...)
    """
    return BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.quant_type,
        bnb_4bit_compute_dtype=resolve_dtype(cfg.compute_dtype),
        bnb_4bit_use_double_quant=cfg.double_quant,
    )
