"""
src.data — datasets, prompt templates, and the masking collator.

The data axis of the experiment grid. Everything funnels through one neutral
sample type (:class:`VLMExample`) so the trainer never branches on dataset name.

Public API
----------
    from src.data import build_dataset, build_collator, VLMExample
    train = build_dataset(cfg.data, split=cfg.data.split_train)   # BaseVLMDataset
    collate = build_collator(cfg.data, processor)                 # VLMCollator

See the subpackage README for the full picture.
"""

from src.config.schema import DataConfig
from src.data.base import BaseVLMDataset
from src.data.collator import VLMCollator
from src.data.datasets import DATASETS
from src.data.example import VLMExample
from src.data.prompts import PROMPT_VARIANTS, PromptTemplate, build_template


def build_dataset(cfg: DataConfig, split: str) -> BaseVLMDataset:
    """Instantiate the dataset named by ``cfg.name`` for the given split."""
    return DATASETS.build(cfg.name, cfg, split)


def build_collator(cfg: DataConfig, wrapper, tag_spans: bool = False) -> VLMCollator:
    """Build the collator with the prompt template named by ``cfg.prompt_variant``.

    ``wrapper`` is the backbone (:class:`~src.models.base.BaseVLMWrapper`); the
    collator uses its backbone-aware encoder so image tokens are inserted
    correctly for any model. ``tag_spans`` adds the answer/explanation span tags
    the explanation-aware objective needs (the Trainer sets this from the
    objective's ``requires_span_ids``).
    """
    return VLMCollator(
        wrapper, build_template(cfg.prompt_variant), cfg.max_length, tag_spans=tag_spans
    )


__all__ = [
    "VLMExample",
    "BaseVLMDataset",
    "VLMCollator",
    "PromptTemplate",
    "build_template",
    "build_dataset",
    "build_collator",
    "DATASETS",
    "PROMPT_VARIANTS",
]
