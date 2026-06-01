"""Typed experiment configuration: YAML ⇄ dataclasses.

Public API::

    from src.config import load_config, dump_config, ExperimentConfig
    cfg = load_config("configs/experiments/rq2_blip2_generative_scienceqa.yaml")

See ``schema.py`` for the dataclass tree and ``io.py`` for loading/overrides.
"""

from src.config.io import dump_config, load_config
from src.config.schema import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    LoraConfig,
    ModelConfig,
    ObjectiveConfig,
    QuantizationConfig,
    TrainConfig,
    WandbConfig,
)

__all__ = [
    "load_config",
    "dump_config",
    "ExperimentConfig",
    "ModelConfig",
    "QuantizationConfig",
    "LoraConfig",
    "DataConfig",
    "ObjectiveConfig",
    "TrainConfig",
    "EvalConfig",
    "WandbConfig",
]
