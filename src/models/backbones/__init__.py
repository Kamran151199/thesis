"""Backbone registry + auto-registration of every VLM wrapper.

``BACKBONES`` is defined first, then each wrapper module is imported so its
``@BACKBONES.register(...)`` decorator runs (same bootstrap as the dataset
registry). Add a backbone = new file + one decorator + one import line below.

Note: importing a wrapper module imports its HF model class (e.g.
``Qwen2VLForConditionalGeneration``). If a given ``transformers`` version lacks
one, that single import fails — wrap the import below in try/except only if you
need the others to keep working on an older install.
"""

from src.models.base import BaseVLMWrapper
from src.registry import Registry

#: name → BaseVLMWrapper subclass. Key it with ``ModelConfig.name``.
BACKBONES: Registry[BaseVLMWrapper] = Registry("backbone")

# Import for side effects (registration). Must come AFTER BACKBONES is defined.
from src.models.backbones import (  # noqa: E402,F401
    blip2,
    paligemma,
    qwen2_vl,
)

__all__ = ["BACKBONES"]
