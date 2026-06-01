"""
src.config.schema — the typed shape of one experiment.

Every run is described by ONE ``ExperimentConfig``, normally loaded from a YAML
file in ``configs/experiments/``. The dataclasses below are the contract: if a
YAML key is missing it gets the default here; if it has the wrong type you find
out at load time, not 40 minutes into training on the A100.

THE TREE (mirrors the four pluggable axes + the training/eval knobs)::

    ExperimentConfig
    ├── name, rq, seed, output_dir          # provenance: which RQ does this serve?
    ├── model:     ModelConfig              # WHICH backbone + how it's adapted
    │   ├── quantization: QuantizationConfig   #   4-bit NF4 (the "Q" in QLoRA)
    │   └── lora:         LoraConfig           #   rank-16 adapters (the "LoRA")
    ├── data:      DataConfig               # WHICH dataset + prompt template
    ├── objective: ObjectiveConfig          # WHICH loss (generative / expl-aware / contrastive)
    ├── train:     TrainConfig              # optimizer, lr schedule, steps, grad-accum
    ├── eval:      EvalConfig               # which metrics, CoT-eval or not
    └── wandb:     WandbConfig              # experiment tracking

WHY DATACLASSES AND NOT A DICT? Autocomplete, type-checking, defaults in one
place, and ``cfg.train.lr`` reads better than ``cfg["train"]["lr"]``. The whole
config also round-trips to a plain dict (``to_dict``) so it can be dumped next
to a checkpoint and logged to wandb verbatim — full provenance for the thesis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from typing_extensions import Literal


# ════════════════════════════════════════════════════════════════════════════
# Model — backbone + QLoRA (the "how do we adapt a frozen VLM" axis)
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class QuantizationConfig:
    """4-bit NormalFloat (NF4) quantization — the "Q" in QLoRA.

    Freezes the big base model in 4-bit so it costs ~1/4 the memory and needs
    no gradients/optimizer state. ``compute_dtype`` is what 4-bit weights are
    *de-quantized to* for each matmul (bf16), so math stays accurate even though
    storage is 4-bit. ``double_quant`` quantizes the quantization constants too
    (a few extra % memory saved, free).
    """

    load_in_4bit: bool = True
    quant_type: str = "nf4"  # "nf4" (NormalFloat) or "fp4"
    compute_dtype: str = "bfloat16"
    double_quant: bool = True


@dataclass
class LoraConfig:
    """Low-Rank Adaptation — the "LoRA" in QLoRA.

    Freeze base weight ``W``; learn a low-rank update ``ΔW = B @ A`` with
    ``rank=r``. Only ``A`` and ``B`` get gradients → ~0.5–3% of params train.
    Proposal spec: rank 16, alpha 32 (scale = alpha/r = 2.0).

    ``target_modules="auto"`` defers to the backbone wrapper, which knows the
    right module names (e.g. ``q_proj``/``v_proj`` for an OPT/Qwen LLM) — so the
    YAML stays backbone-agnostic.
    """

    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: Literal["none", "lora_only", "all"] = "none"  # "none" | "lora_only" | "all"
    target_modules: list[str] | str = "auto"


@dataclass
class ModelConfig:
    """Which VLM, in what precision, adapted how."""

    name: str = "blip2"  # registry key → src/models/backbones/
    pretrained: str = "Salesforce/blip2-opt-2.7b"  # HF hub id
    dtype: str = "bfloat16"
    device_map: str = "auto"
    use_qlora: bool = True
    #: Module-name prefixes to hard-freeze regardless of LoRA (e.g. the vision
    #: tower). The wrapper supplies sane defaults if left empty.
    freeze: list[str] = field(default_factory=list)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)


# ════════════════════════════════════════════════════════════════════════════
# Data — dataset + which prompt template (the "what does the model see" axis)
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class DataConfig:
    """Which dataset, how much of it, and which prompt template.

    ``prompt_variant`` is the lever for RQ3 (explanation-aware training):
      - ``"answer_only"``        → target is just the answer        (α = 1.0 arm)
      - ``"explanation_then_answer"`` → target is "Reasoning: … Answer: …" (α < 1)
    Same underlying data, different supervision — a controlled comparison.
    """

    name: str = "scienceqa"  # registry key → src/data/datasets/
    hf_path: str | None = None  # override the loader's default HF path
    split_train: str = "train"
    split_eval: str = "validation"
    max_train: int | None = 2000  # None → use the whole split
    max_eval: int | None = 200
    prompt_variant: str = "explanation_then_answer"
    max_length: int = 512  # truncate (image tokens + text) to this many tokens
    image_field: str = "image"
    num_proc: int = 4  # parallelism for the datasets .filter/.map


# ════════════════════════════════════════════════════════════════════════════
# Objective — the loss (the central RQ2/RQ3 axis)
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class ObjectiveConfig:
    """Which training signal.

    - ``"generative"``        — plain next-token cross-entropy on the target.
    - ``"explanation_aware"`` — ``L = α·L_answer + (1−α)·L_explanation``.
      ``alpha`` swept over {0.0, 0.5, 1.0} per the proposal. α=1 ⇒ answer-only;
      α=0 ⇒ explanation-only; 0.5 ⇒ balanced.
    - ``"contrastive"``       — generative loss + ``contrastive_weight`` × InfoNCE
      between projected image features and answer sentence embeddings (RQ2's
      contrastive-enhanced arm).
    """

    name: str = "generative"  # registry key → src/objectives/
    alpha: float = 0.5  # explanation_aware: weight on the ANSWER span
    contrastive_weight: float = 0.0  # contrastive: weight on the auxiliary InfoNCE
    temperature: float = 0.07  # InfoNCE temperature (contrastive only)


# ════════════════════════════════════════════════════════════════════════════
# Training — optimizer + schedule + budget
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class TrainConfig:
    """Optimizer, LR schedule, and the step/epoch budget.

    Defaults follow proposal §7.4: AdamW (β1=0.9, β2=0.999, wd=0.01), lr 2e-4
    cosine with 3% warm-up, effective batch 32 via gradient accumulation.

        effective_batch = batch_size × grad_accum_steps

    ``max_steps`` (if set) caps training regardless of ``epochs`` — the
    "5,000 steps per domain or 2 epochs, whichever is fewer" rule.
    """

    epochs: int = 1
    max_steps: int | None = None
    batch_size: int = 4  # per-step micro-batch (fits A100 40GB at bf16)
    grad_accum_steps: int = 8  # 4 × 8 = 32 effective
    lr: float = 2e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    max_grad_norm: float = 1.0
    scheduler: str = "cosine"  # "cosine" | "linear" | "constant"
    warmup_ratio: float = 0.03
    log_every: int = 25  # steps between loss logs
    eval_every: int | None = None  # steps between mid-train evals (None → only at end)
    save_every: int | None = None  # steps between checkpoints (None → only at end)


# ════════════════════════════════════════════════════════════════════════════
# Evaluation — which metrics, and how to query the model
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class EvalConfig:
    """Which metrics to compute and how to elicit answers.

    ``cot=True`` ⇒ generate-then-score (the model writes its reasoning, then we
    likelihood-score each choice conditioned on it) — required when the model is
    trained to reason before answering. ``cot=False`` ⇒ score choices directly
    after the prompt (answer-only models).
    """

    metrics: list[str] = field(default_factory=lambda: ["mc_accuracy"])
    cot: bool = True
    max_new_tokens: int = 160  # generation budget for the reasoning span
    # Sequence-truncation cap for scoring/encoding. MUST exceed the backbone's
    # image-token count (BLIP-2 ≈ 32, Qwen2-VL ≈ 320) — a smaller cap slices the
    # image-token block mid-way and the processor raises a token-count mismatch.
    # This is NOT max_new_tokens (a generation budget); the two are unrelated.
    max_length: int = 1024
    # How many examples per batched generation, and how many (example × choice)
    # rows per batched scoring forward. Higher → better GPU use, more memory. 8 is
    # comfortable for a 2B model on an A100-40GB; lower it if you OOM, raise it to
    # go faster. (batch_size=1 reproduces the old per-item path bit-for-bit.)
    batch_size: int = 8


# ════════════════════════════════════════════════════════════════════════════
# Tracking
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class WandbConfig:
    """Weights & Biases settings. ``enabled=False`` ⇒ no network calls at all."""

    enabled: bool = True
    project: str = "thesis-vlm-alignment"
    entity: str | None = None
    mode: str = "online"  # "online" | "offline" | "disabled"
    tags: list[str] = field(default_factory=list)


# ════════════════════════════════════════════════════════════════════════════
# The whole thing
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class ExperimentConfig:
    """One complete, self-describing experiment.

    The ``rq`` field is provenance, not decoration: every run must trace to a
    research question, or it is scope creep. It also ends up in the wandb run
    name and the output folder, so months later you know *why* a checkpoint
    exists.

    Example
    -------
    >>> from src.config import load_config
    >>> cfg = load_config("configs/experiments/rq2_blip2_generative_scienceqa.yaml")
    >>> cfg.model.name, cfg.objective.name, cfg.train.lr
    ('blip2', 'generative', 0.0002)
    """

    name: str = "unnamed_experiment"
    rq: str = "RQ?"  # e.g. "RQ2" — which research question this serves
    seed: int = 42
    output_dir: str | None = None  # None → utils.paths.experiment_dir(name)

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def to_dict(self) -> dict[str, Any]:
        """Plain nested dict — dump beside the checkpoint and log to wandb."""
        return asdict(self)
