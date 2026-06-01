"""
src.models.peft_lora — the "LoRA" in QLoRA (rank-16 trainable adapters).

THE IDEA IN ONE PICTURE
-----------------------
Freeze the giant pretrained weight ``W`` (d×d). Learn a low-rank *correction*::

    h = W·x  +  (B·A)·x          A: r×d,  B: d×r,  r = 16 ≪ d

::

        W (frozen, 4-bit)            ΔW = B·A  (trainable, bf16)
        ┌───────────────┐           ┌──┐
        │   d × d        │     +     │d │ · ┌──────────┐
        │   ~millions    │           │×r│   │  r × d   │   = a d×d update
        └───────────────┘           └──┘   └──────────┘     built from only
                                                              2·d·r numbers
    For d=2560, r=16:  ΔW is described by 2·2560·16 ≈ 82K params, not 6.5M.

Only ``A`` and ``B`` get gradients and optimizer state, so ~0.5–3% of the model
trains. ``alpha`` scales the update (effective scale = ``alpha / r`` = 32/16 = 2).

WHICH LAYERS GET ADAPTERS (``target_modules``)?
-----------------------------------------------
Per proposal §7.3: the attention projections and MLP of the language model, plus
the alignment module. Each backbone wrapper supplies the right module *names*
(they differ: OPT uses ``q_proj``/``v_proj``; Qwen uses the same names but in a
different tree). ``"auto"`` in the config means "ask the wrapper".
"""

from __future__ import annotations

import torch.nn as nn
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model, prepare_model_for_kbit_training

from src.config.schema import LoraConfig
from src.utils.logging import get_logger

log = get_logger(__name__)


def apply_qlora(
    model: nn.Module,
    cfg: LoraConfig,
    target_modules: list[str],
    is_quantized: bool,
    task_type: str = "CAUSAL_LM",
) -> nn.Module:
    """Wrap ``model`` with LoRA adapters on ``target_modules``; return the PEFT model.

    Parameters
    ----------
    model:
        The (optionally 4-bit) base model.
    cfg:
        Our :class:`LoraConfig` (r, alpha, dropout, bias).
    target_modules:
        Concrete module-name substrings to inject adapters into (from the
        wrapper's ``default_lora_targets`` when the config says ``"auto"``).
    is_quantized:
        If the base was loaded in 4-bit, run ``prepare_model_for_kbit_training``
        first — it casts norms/embeddings to fp32 and enables input-grad routing
        so gradients flow through the frozen quantized trunk to the adapters.

    Example
    -------
    >>> model = apply_qlora(base, LoraConfig(r=16, alpha=32),
    ...                     target_modules=["q_proj", "v_proj"], is_quantized=True)
    >>> model.print_trainable_parameters()
    trainable params: 4,718,592 || all params: 2,705,..., || trainable%: 0.17
    """
    if is_quantized:
        model = prepare_model_for_kbit_training(model)

    peft_cfg = PeftLoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        bias=cfg.bias,
        target_modules=target_modules,
        task_type=task_type,
    )
    model = get_peft_model(model, peft_cfg)  # type: ignore[return-value]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(
        "LoRA applied: r=%d alpha=%d on %s | trainable %s (%.2f%%)",
        cfg.r,
        cfg.alpha,
        target_modules,
        f"{trainable:,}",
        100 * trainable / total,
    )
    return model
