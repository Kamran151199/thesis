"""
src.experiment — the orchestrator that runs one config end-to-end.

    from src.experiment import ExperimentRunner
    from src.config import load_config
    ExperimentRunner(load_config("configs/experiments/foo.yaml")).run()

Usually invoked via the CLI instead: ``python -m src.run --config <yaml>``.
"""

from src.experiment.runner import ExperimentRunner

__all__ = ["ExperimentRunner"]
