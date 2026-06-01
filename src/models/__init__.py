"""
src.models — backbone wrappers + QLoRA, behind one factory.

The backbone axis of the experiment grid. Every VLM is wrapped to a single
interface (:class:`BaseVLMWrapper`) so the trainer/evaluator are backbone-blind.

Public API
----------
    from src.models import build_model
    wrapper = build_model(cfg.model)     # loads (quantized) base + LoRA + processor
    out = wrapper.forward(batch)         # out.loss, out.logits
    ids = wrapper.generate(batch, max_new_tokens=64)

See the subpackage README for the load pipeline and how to add a backbone.
"""

from src.config.schema import ModelConfig
from src.models.backbones import BACKBONES
from src.models.base import BaseVLMWrapper


def build_model(cfg: ModelConfig) -> BaseVLMWrapper:
    """Instantiate the backbone named by ``cfg.name`` (loads + adapts it)."""
    return BACKBONES.build(cfg.name, cfg)


__all__ = ["build_model", "BaseVLMWrapper", "BACKBONES"]
