"""
src.training.checkpoint — save/load ONLY what was trained.

The frozen 4-bit base (gigabytes) never changed, so we never save it — we save
just the trainable delta, which is tiny and reattaches to a fresh base later:

    QLoRA model     → ``model.save_pretrained(dir)``  writes the LoRA adapter
                       (a few MB of A/B matrices) + adapter_config.json
    full-tune model → a ``state_dict`` of the ``requires_grad`` params only
                       (the proven BLIP-2 "trainable_state" pattern, ~110MB)

Either way a ``meta.json`` (resolved config + final metrics) lands beside it, so
months later a checkpoint folder fully explains itself — what RQ, what data,
what numbers. This is reproducibility hygiene the viva will thank you for.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel

from src.utils.logging import get_logger

log = get_logger(__name__)


def save_checkpoint(
    wrapper, out_dir: str | Path, meta: dict[str, Any] | None = None
) -> Path:
    """Save the trainable delta + a ``meta.json`` into ``out_dir``.

    Returns the directory path. Works for both QLoRA (adapter) and full-tune
    (trainable state dict) models — it inspects the model type and does the
    right thing.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(wrapper.model, PeftModel):
        wrapper.model.save_pretrained(
            str(out_dir)
        )  # adapter_model.safetensors + config
        log.info("saved LoRA adapter → %s", out_dir)
    else:
        trainable = {
            name: p.detach().cpu().clone()
            for name, p in wrapper.model.named_parameters()
            if p.requires_grad
        }
        torch.save(trainable, out_dir / "trainable_state.pt")
        log.info("saved %d trainable tensors → %s", len(trainable), out_dir)

    if meta is not None:
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return out_dir


def checkpoint_exists(ckpt_dir: str | Path) -> bool:
    """Return whether ``ckpt_dir`` contains a loadable trained delta."""
    ckpt_dir = Path(ckpt_dir)
    has_peft = (ckpt_dir / "adapter_config.json").exists() and (
        (ckpt_dir / "adapter_model.safetensors").exists()
        or (ckpt_dir / "adapter_model.bin").exists()
    )
    has_trainable_state = (ckpt_dir / "trainable_state.pt").exists()
    return has_peft or has_trainable_state


def load_checkpoint(wrapper, ckpt_dir: str | Path) -> None:
    """Load a saved delta back into an already-built ``wrapper`` (in place).

    The wrapper must be constructed from the SAME base + config first; this only
    restores the trained part on top.
    """
    ckpt_dir = Path(ckpt_dir)
    if not checkpoint_exists(ckpt_dir):
        raise FileNotFoundError(f"no loadable checkpoint delta found in {ckpt_dir}")
    if isinstance(wrapper.model, PeftModel):
        # The wrapper is already constructed with a freshly initialized PEFT
        # adapter named "default". Loading the trained adapter with the same
        # name can collide in PEFT, so attach it under a stable evaluation name
        # and activate it explicitly.
        adapter_name = "trained"
        if adapter_name not in getattr(wrapper.model, "peft_config", {}):
            wrapper.model.load_adapter(
                str(ckpt_dir), adapter_name=adapter_name, is_trainable=False
            )
        wrapper.model.set_adapter(adapter_name)
        log.info("loaded LoRA adapter ← %s", ckpt_dir)
    else:
        state = torch.load(ckpt_dir / "trainable_state.pt", map_location="cpu")
        missing, unexpected = wrapper.model.load_state_dict(state, strict=False)
        log.info(
            "loaded %d trainable tensors ← %s (missing=%d, unexpected=%d)",
            len(state),
            ckpt_dir,
            len(missing),
            len(unexpected),
        )
