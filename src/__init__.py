"""
src — the thesis experiment framework.

"Cross-Modal Generative Alignment for Document, Chart, and Image Reasoning with
Explanation-Aware Vision-Language Models."

A registry-driven, config-first harness for running the ~30 fine-tuning
experiments behind the thesis's research questions. Four pluggable axes::

    backbone   ×   dataset   ×   objective   ×   metric
    (models/)      (data/)       (objectives/)   (evaluation/)

Pick one of each in a YAML file; the runner (``experiment/``) assembles and runs
it. Add a new option = one file + one ``@REGISTRY.register(...)`` decorator.

Quick start
-----------
    from src.config import load_config
    from src.experiment import ExperimentRunner
    ExperimentRunner(load_config("configs/experiments/rq2_blip2_generative_scienceqa.yaml")).run()

Or from the shell: ``python -m src.run --config <yaml>``.  See ``src/README.md``.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
