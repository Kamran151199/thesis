"""
src.config.io — turn a YAML file into a typed ``ExperimentConfig`` (and back).

    configs/experiments/foo.yaml                 ExperimentConfig
    ┌──────────────────────────┐   load_config   ┌────────────────────────┐
    │ model:                   │   ───────────▶   │ cfg.model.name='blip2' │
    │   name: blip2            │                  │ cfg.train.lr=2e-4      │
    │ train:                   │                  │ … fully typed …        │
    │   lr: 2.0e-4             │                  └────────────────────────┘
    └──────────────────────────┘

Two jobs:
  1. **Nested construction** — recursively build the dataclass tree so
     ``model.lora.r`` in YAML lands in ``cfg.model.lora.r`` as an ``int``.
  2. **Overrides** — apply dotted ``key=value`` strings from the CLI so you can
     sweep without editing files: ``--set train.lr=1e-4 objective.alpha=0.0``.

No third-party config framework (Hydra/OmegaConf) — just PyYAML + dataclasses.
Fewer moving parts, and the whole config still round-trips to a plain dict.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, get_type_hints

import yaml

from src.config.schema import ExperimentConfig


def _is_dataclass_type(tp: Any) -> bool:
    return isinstance(tp, type) and dataclasses.is_dataclass(tp)


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build dataclass ``cls`` from a (possibly nested) dict.

    For each field whose declared type is itself a dataclass, recurse so the
    sub-dict becomes a sub-dataclass. Unknown keys raise — a typo like
    ``learnign_rate`` fails loudly instead of being silently ignored.

    Note: ``schema.py`` uses ``from __future__ import annotations``, so
    ``dataclasses.fields(cls).type`` is a *string* (e.g. ``"LoraConfig"``).
    ``get_type_hints`` evaluates those strings back into real classes using the
    schema module's namespace — that's how we detect nested dataclasses.
    """
    if not _is_dataclass_type(cls):
        return data

    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - field_names
    if unknown:
        raise KeyError(
            f"unknown key(s) {sorted(unknown)} for {cls.__name__}. "
            f"Valid keys: {sorted(field_names)}"
        )

    hints = get_type_hints(cls)  # resolves the stringized annotations
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        field_type = hints.get(name)
        # A field typed as a dataclass given a dict → recurse.
        if _is_dataclass_type(field_type) and isinstance(value, dict):
            kwargs[name] = _from_dict(field_type, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _coerce_scalar(text: str) -> Any:
    """Parse a CLI override value: ``42``→int, ``1e-4``→float, ``true``→bool, ``[a,b]``→list.

    We try ``int`` then ``float`` BEFORE ``yaml.safe_load`` because PyYAML's
    float resolver rejects unsigned-exponent forms like ``1e-4`` (it requires a
    decimal point), parsing them as strings — a nasty silent ``lr="1e-4"`` bug.
    ``float()`` handles them correctly; YAML then covers bools/null/lists.
    """
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return yaml.safe_load(text)


def _apply_override(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    """In-place set ``d["a"]["b"]["c"] = value`` for ``dotted_key="a.b.c"``."""
    keys = dotted_key.split(".")
    node = d
    for k in keys[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise TypeError(f"override path {dotted_key!r} hits a non-mapping at {k!r}")
    node[keys[-1]] = value


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
) -> ExperimentConfig:
    """Load a YAML experiment file into a typed ``ExperimentConfig``.

    Parameters
    ----------
    path:
        Path to a YAML file under ``configs/`` (any structure matching the
        schema; omitted keys take their dataclass defaults).
    overrides:
        Optional ``["train.lr=1e-4", "objective.alpha=0.0"]`` from the CLI,
        applied on top of the file — for sweeps without new files.

    Example
    -------
    >>> cfg = load_config("configs/experiments/rq2_blip2_generative_scienceqa.yaml",
    ...                   overrides=["train.lr=1e-5", "data.max_train=500"])
    >>> cfg.train.lr, cfg.data.max_train
    (1e-05, 500)
    """
    path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}

    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override {ov!r} must be 'dotted.key=value'")
        key, _, val = ov.partition("=")
        _apply_override(raw, key.strip(), _coerce_scalar(val.strip()))

    return _from_dict(ExperimentConfig, raw)


def dump_config(cfg: ExperimentConfig, path: str | Path) -> None:
    """Write the *resolved* config (defaults filled in) next to a checkpoint.

    Critical for reproducibility: the saved YAML is the exact, fully-expanded
    config the run used — not the sparse file you launched with.
    """
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
