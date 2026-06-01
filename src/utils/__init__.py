"""Cross-cutting helpers used everywhere: seeding, logging, device/dtype, paths.

These have **no dependencies on the rest of ``src``** — they sit at the bottom
of the import graph so anything may import them without a cycle.
"""

from src.utils.devices import (
    best_device,
    describe_device,
    move_to_device,
    resolve_dtype,
)
from src.utils.logging import get_logger
from src.utils.paths import REPO_ROOT, experiment_dir, output_root
from src.utils.seed import set_seed

__all__ = [
    "set_seed",
    "get_logger",
    "resolve_dtype",
    "best_device",
    "describe_device",
    "move_to_device",
    "output_root",
    "experiment_dir",
    "REPO_ROOT",
]
