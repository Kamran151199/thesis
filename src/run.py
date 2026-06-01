"""
src.run — the command-line entrypoint for every experiment.

    # run one experiment from a YAML config
    python -m src.run --config configs/experiments/rq2_blip2_generative_scienceqa.yaml

    # sweep a hyperparameter without editing the file
    python -m src.run --config <cfg> --set objective.alpha=0.0 train.lr=1e-4

    # discover what's registered (backbones / datasets / objectives / metrics)
    python -m src.run --list

One config = one run = one folder under ``outputs/``. That's the whole workflow:
the ~30 thesis experiments are ~30 YAML files, looped over by a shell script or
launched one at a time on Colab.
"""

from __future__ import annotations

import argparse

from src.config import load_config
from src.experiment import ExperimentRunner


def _print_registries() -> None:
    """List everything registered on each axis (imports populate the registries)."""
    from src.data import DATASETS, PROMPT_VARIANTS
    from src.evaluation import METRICS
    from src.models import BACKBONES
    from src.objectives import OBJECTIVES

    for reg in (BACKBONES, DATASETS, OBJECTIVES, METRICS, PROMPT_VARIANTS):
        print(f"  {reg.name:14s}: {reg.available()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="src.run", description="Run a thesis experiment."
    )
    parser.add_argument("--config", "-c", help="Path to a YAML experiment config.")
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        help="Dotted overrides, e.g. train.lr=1e-4 objective.alpha=0.0",
    )
    parser.add_argument(
        "--list", action="store_true", help="List registered components and exit."
    )
    args = parser.parse_args()

    if args.list:
        _print_registries()
        return
    if not args.config:
        parser.error("--config is required (or use --list)")

    cfg = load_config(args.config, overrides=args.overrides)
    ExperimentRunner(cfg).run()


if __name__ == "__main__":
    main()
