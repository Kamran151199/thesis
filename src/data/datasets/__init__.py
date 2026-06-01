"""Dataset registry + auto-registration of every loader.

The ``DATASETS`` registry is defined here FIRST, then each loader module is
imported so its ``@DATASETS.register(...)`` decorator runs. That import order is
deliberate: a loader does ``from src.data.datasets import DATASETS`` at its top,
which only resolves because ``DATASETS`` is bound before the submodule imports
below execute (the standard registry bootstrap).

Add a dataset = drop a file in this folder + one ``@DATASETS.register("name")``
decorator + one line in the import block below.
"""

from src.data.base import BaseVLMDataset
from src.registry import Registry

#: name → BaseVLMDataset subclass. Key it with ``DataConfig.name``.
DATASETS: Registry[BaseVLMDataset] = Registry("dataset")

# Import for side effects (registration). Must come AFTER DATASETS is defined.
from src.data.datasets import (  # noqa: E402,F401
    aokvqa,
    chartqa,
    docvqa,
    scienceqa,
    vqav2,
)

__all__ = ["DATASETS"]
