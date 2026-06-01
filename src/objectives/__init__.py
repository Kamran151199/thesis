"""
src.objectives — the training loss IS the experiment (RQ2 & RQ3).

The objective axis. Each objective turns (model, batch) into one scalar to
backprop. Swap the objective, change nothing else → isolate the effect of the
training signal.

    generative          plain next-token CE on the target            (RQ2 baseline)
    explanation_aware   α·L_answer + (1−α)·L_explanation              (RQ3 core)
    contrastive         generative + weight·InfoNCE(image, answer)    (RQ2 contrastive arm)

Public API
----------
    from src.objectives import build_objective
    objective = build_objective(cfg.objective)
    out = objective.compute(wrapper, batch)   # out.loss, out.components

The registry is defined first, then the objective modules are imported for their
registration side effects (same bootstrap as the other registries).
"""

from src.config.schema import ObjectiveConfig
from src.objectives.base import BaseObjective, LossOutput, masked_token_ce
from src.registry import Registry

#: name → BaseObjective subclass. Key it with ``ObjectiveConfig.name``.
OBJECTIVES: Registry[BaseObjective] = Registry("objective")

# Import for side effects (registration). Must come AFTER OBJECTIVES is defined.
from src.objectives import (  # noqa: E402,F401
    contrastive,
    explanation_aware,
    generative,
)


def build_objective(cfg: ObjectiveConfig) -> BaseObjective:
    """Instantiate the objective named by ``cfg.name``."""
    return OBJECTIVES.build(cfg.name, cfg)


__all__ = [
    "OBJECTIVES",
    "build_objective",
    "BaseObjective",
    "LossOutput",
    "masked_token_ce",
]
