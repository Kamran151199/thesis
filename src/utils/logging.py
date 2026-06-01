"""A single, consistent logger so every module's output looks the same.

Use ``get_logger(__name__)`` at the top of any module instead of bare
``print``. Benefits over ``print``: timestamps, the source module name, and a
level you can turn down to WARNING when a run gets noisy — without deleting
lines.

    >>> log = get_logger(__name__)
    >>> log.info("loaded %d examples", 2000)
    2026-05-31 14:02:11 | INFO | src.data.scienceqa | loaded 2000 examples
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _configure_root() -> None:
    """Attach a single stdout handler to the package root logger, once."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root = logging.getLogger("src")
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    root.propagate = False  # don't double-print through the global root logger
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger. Pass ``__name__`` from the calling module."""
    _configure_root()
    return logging.getLogger(name if name.startswith("src") else f"src.{name}")
