# %% [markdown]
# # 00 - Journal-paper pipeline
#
# This notebook is the paper artifact factory. Run it top-to-bottom in Colab
# after `colab/setup_colab.py`. It is designed to be:
#
# - fail-loud: config typos, masking leaks, missing expected runs, and missing
#   required paper artifacts are reported explicitly;
# - resumable: completed runs are skipped, and every expensive post-processing
#   stage writes JSON/CSV/PDF files after each unit of work;
# - paper-facing: by the last cell, the tables, figures, qualitative samples,
#   transfer matrix, faithfulness scores, and artifact manifest are written.
#
# Coverage:
#
# - RQ2: BLIP-2 generative vs BLIP-2 contrastive-enhanced alignment.
# - RQ3: Qwen2-VL ScienceQA alpha sweep, plus A-OKVQA rationale-domain check.
# - RQ4: ScienceQA, A-OKVQA, ChartQA, DocVQA, and VQAv2 domain comparison.
# - RQ5: evidence-masking drift per completed model.
# - RQ6: Qwen2-VL 2B vs 7B QLoRA scale comparison.
# - RQ7: train-on-domain -> eval-on-domain transfer matrix.
# - RQ1: literature review only; no experiment run is needed here.

# %% Cell 1 - setup, paths, and artifact helpers
import gc
import hashlib
import json
import math
import os
import shutil
import textwrap
import traceback
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import Markdown, display

from src.config import load_config
from src.data import build_collator, build_dataset, build_template
from src.data.constants import LABEL_IGNORE, SPAN_ANSWER, SPAN_EXPLANATION, SPAN_IGNORE
from src.evaluation import Evaluator, build_metrics
from src.evaluation.faithfulness import mask_region, region_importance
from src.evaluation.metrics.retrieval import retrieval_recall
from src.evaluation.scoring import clean_generated_answer, generate_continuation, score_continuation, split_reasoning_answer
from src.experiment.runner import ExperimentRunner
from src.models import build_model
from src.objectives import build_objective
from src.objectives.contrastive import info_nce
from src.training.checkpoint import load_checkpoint
from src.utils import describe_device, experiment_dir, move_to_device, set_seed

print("device:", describe_device())
set_seed(42)

REPO_ROOT = Path.cwd().resolve()
BASE = "configs/experiments"

# Finished checkpoints/results live under THESIS_CKPT_DIR on Colab Drive.
RESULTS_ROOT = experiment_dir("_").parent
DRIVE_ROOT = Path(os.environ.get("THESIS_CKPT_DIR", "outputs")).resolve().parent
PAPER_OUT = DRIVE_ROOT / "paper_artifacts"

FIG_DIR = PAPER_OUT / "figures"
TAB_DIR = PAPER_OUT / "tables"
ART_DIR = PAPER_OUT / "data"
LOG_DIR = PAPER_OUT / "logs"

# Mirror final figures/tables into the repo paper folder when this notebook is
# run from the checked-out thesis repo. This makes LaTeX inclusion easy.
REPO_PAPER_DIR = REPO_ROOT / "thesis_doc" / "paper"
REPO_FIG_DIR = REPO_PAPER_DIR / "Figures" / "generated"
REPO_TAB_DIR = REPO_PAPER_DIR / "Tables" / "generated"

for _d in (FIG_DIR, TAB_DIR, ART_DIR, LOG_DIR, REPO_FIG_DIR, REPO_TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

print("results root :", RESULTS_ROOT)
print("paper output :", PAPER_OUT)
print("repo figures :", REPO_FIG_DIR)


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def mirror_file(path: Path, mirror_dir: Path | None) -> None:
    if mirror_dir is None:
        return
    mirror_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, mirror_dir / path.name)


def save_json(obj: Any, name: str, directory: Path = ART_DIR) -> Path:
    path = directory / f"{name}.json"
    atomic_write_text(path, json.dumps(obj, indent=2, default=str))
    print("json ->", path)
    return path


def save_markdown(text: str, name: str) -> Path:
    path = ART_DIR / f"{name}.md"
    atomic_write_text(path, text)
    print("md   ->", path)
    return path


def save_csv(df: pd.DataFrame, name: str, directory: Path = ART_DIR) -> Path:
    path = directory / f"{name}.csv"
    tmp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    print("csv  ->", path)
    return path


def latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    return df.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        column_format="l" + "r" * (len(df.columns) - 1),
        na_rep="--",
    )


def save_table(df: pd.DataFrame, name: str, caption: str, label: str) -> None:
    tex = TAB_DIR / f"{name}.tex"
    csv = TAB_DIR / f"{name}.csv"
    atomic_write_text(tex, latex_table(df, caption, label))
    tmp = csv.with_name(csv.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(csv)
    mirror_file(tex, REPO_TAB_DIR)
    mirror_file(csv, REPO_TAB_DIR)
    print(f"table -> {tex} and {csv}")


def save_fig(fig, name: str, *, data: pd.DataFrame | None = None) -> None:
    pdf = FIG_DIR / f"{name}.pdf"
    png = FIG_DIR / f"{name}.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    mirror_file(pdf, REPO_FIG_DIR)
    mirror_file(png, REPO_FIG_DIR)
    if data is not None:
        save_csv(data, f"{name}_plot_data", ART_DIR)
    print(f"fig   -> {pdf} and {png}")


def result_path(run_name: str) -> Path:
    return experiment_dir(run_name) / "results.json"


def resolved_config_path(run_name: str) -> Path:
    return experiment_dir(run_name) / "config.resolved.yaml"


def run_meta_path(run_name: str) -> Path:
    return experiment_dir(run_name) / "paper_run_meta.json"


def sync_eval_length(cfg) -> None:
    if cfg.eval.max_length < cfg.data.max_length:
        print(
            f"raising {cfg.name} eval.max_length "
            f"{cfg.eval.max_length} -> {cfg.data.max_length}"
        )
        cfg.eval.max_length = cfg.data.max_length


def config_digest(cfg) -> str:
    raw = json.dumps(cfg.to_dict(), sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


PIPELINE_CODE_VERSION = 2
CODE_DIGEST_FILES = [
    "notebooks/00_paper_pipeline.py",
    "src/config/schema.py",
    "src/data/prompts.py",
    "src/data/collator.py",
    "src/objectives/base.py",
    "src/objectives/explanation_aware.py",
    "src/objectives/contrastive.py",
    "src/evaluation/evaluator.py",
    "src/evaluation/scoring.py",
    "src/evaluation/faithfulness/masking_consistency.py",
    "src/models/backbones/blip2.py",
    "src/models/backbones/qwen2_vl.py",
    "src/training/trainer.py",
]


def code_digest() -> str:
    h = hashlib.sha256(f"pipeline-code-version:{PIPELINE_CODE_VERSION}".encode("utf-8"))
    missing: list[str] = []
    for rel in CODE_DIGEST_FILES:
        path = REPO_ROOT / rel
        h.update(rel.encode("utf-8"))
        if path.exists():
            h.update(path.read_bytes())
        else:
            missing.append(rel)
            h.update(b"<missing>")
    if missing:
        print("code digest warning: missing files", missing)
    return h.hexdigest()[:16]


PIPELINE_CODE_DIGEST = code_digest()
save_json(
    {
        "pipeline_code_version": PIPELINE_CODE_VERSION,
        "code_digest": PIPELINE_CODE_DIGEST,
        "files": CODE_DIGEST_FILES,
    },
    "paper_pipeline_code_digest",
    LOG_DIR,
)


def materialize_exp_config(exp: dict[str, Any]):
    cfg = load_config(exp["config"], overrides=exp["overrides"])
    cfg.name = exp["name"]
    if exp["metrics"] is not None:
        cfg.eval.metrics = exp["metrics"]
    sync_eval_length(cfg)
    return cfg


def config_is_current(exp: dict[str, Any]) -> bool:
    cfg_path = resolved_config_path(exp["name"])
    if not result_path(exp["name"]).exists() or not cfg_path.exists():
        return False
    current = materialize_exp_config(exp).to_dict()
    previous = load_config(cfg_path).to_dict()
    if current != previous:
        return False
    meta_path = run_meta_path(exp["name"])
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return False
    return (
        meta.get("config_digest") == config_digest(materialize_exp_config(exp))
        and meta.get("code_digest") == PIPELINE_CODE_DIGEST
        and int(meta.get("pipeline_code_version", -1)) == PIPELINE_CODE_VERSION
    )


def result_state(exp: dict[str, Any]) -> str:
    if not result_path(exp["name"]).exists():
        return "missing_results"
    if not resolved_config_path(exp["name"]).exists():
        return "stale_missing_resolved_config"
    if not run_meta_path(exp["name"]).exists():
        return "stale_missing_run_meta"
    return "current" if config_is_current(exp) else "stale_config_or_code_changed"


def load_run(name: str):
    out = experiment_dir(name)
    cfg_path = out / "config.resolved.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing resolved config for {name}: {cfg_path}")
    cfg = load_config(cfg_path)
    sync_eval_length(cfg)
    wrapper = build_model(cfg.model)
    load_checkpoint(wrapper, out)
    template = build_template(cfg.data.prompt_variant)
    eval_ds = build_dataset(cfg.data, split=cfg.data.split_eval)
    wrapper.eval()
    return cfg, wrapper, template, eval_ds


def headline_from_metrics(metrics: dict[str, Any]) -> float | None:
    for key in ("mc_accuracy", "relaxed_accuracy", "anls", "vqa_accuracy", "exact_match"):
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return None


def md(text: str) -> None:
    display(Markdown(textwrap.dedent(text).strip()))


def short(text: Any, n: int = 360) -> str:
    s = "" if text is None else str(text).replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def wrap(text: Any, width: int = 88) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width)) if text else ""


def sample_record(ex, idx: int, dataset: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "idx": idx,
        "question": ex.question,
        "answer": ex.answer,
        "choices": ex.choices,
        "has_explanation": ex.has_explanation,
        "explanation": ex.explanation,
        "image_size": getattr(ex.image, "size", None),
    }


def save_dataset_sample_figure(ex, dataset: str, idx: int, prefix: str = "dataset_sample") -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.axis("off")
    if ex.image is not None:
        ax.imshow(ex.image.convert("RGB"))
    title = (
        f"{dataset} sample {idx}\n"
        f"Q: {short(ex.question, 110)}\n"
        f"Gold: {short(ex.answer, 80)}"
    )
    if ex.choices:
        title += f"\nChoices: {short(ex.choices, 120)}"
    if ex.explanation:
        title += f"\nRationale: {short(ex.explanation, 120)}"
    ax.set_title(title, fontsize=9)
    save_fig(fig, f"{prefix}_{dataset}_{idx}")
    plt.show()


# %% Cell 2 - the full paper experiment matrix
MC_METRICS = ["mc_accuracy", "rouge_l", "bleu"]


def spec(
    *,
    name: str,
    config: str,
    rq: str,
    domain: str,
    dataset: str,
    backbone: str,
    objective: str,
    role: str,
    overrides: list[str] | None = None,
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "config": config,
        "overrides": overrides or [],
        "metrics": metrics,
        "rq": rq,
        "domain": domain,
        "dataset": dataset,
        "backbone": backbone,
        "objective": objective,
        "role": role,
    }


EXPERIMENTS: list[dict[str, Any]] = []

# RQ3 alpha sweep on ScienceQA. These runs all use the rationale+answer target;
# alpha=1.0 is answer-span-only inside that sequence, not the clean answer-only
# baseline. The clean answer-only controls are separate configs below.
for a in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
    EXPERIMENTS.append(
        spec(
            name=f"rq3_alpha_{int(round(a * 100)):03d}",
            config=f"{BASE}/rq3_qwen2vl_explanation_aware_scienceqa.yaml",
            overrides=[f"objective.alpha={a}"],
            metrics=MC_METRICS,
            rq="RQ3",
            domain="science",
            dataset="scienceqa",
            backbone="Qwen2-VL-2B",
            objective=f"explanation_aware_alpha_{a:g}",
            role="scienceqa_alpha_sweep",
        )
    )

# RQ2 BLIP-2 comparison: same backbone/data/budget, objective changes.
EXPERIMENTS.extend(
    [
        spec(
            name="rq2_blip2_generative_scienceqa",
            config=f"{BASE}/rq2_blip2_generative_scienceqa.yaml",
            metrics=["mc_accuracy"],
            rq="RQ2",
            domain="science",
            dataset="scienceqa",
            backbone="BLIP-2 OPT-2.7B",
            objective="generative",
            role="blip2_rq2",
        ),
        spec(
            name="rq2_blip2_contrastive_scienceqa",
            config=f"{BASE}/rq2_blip2_contrastive_scienceqa.yaml",
            metrics=["mc_accuracy"],
            rq="RQ2",
            domain="science",
            dataset="scienceqa",
            backbone="BLIP-2 OPT-2.7B",
            objective="contrastive_enhanced",
            role="blip2_rq2",
        ),
    ]
)

# Qwen2-VL controls with rationale-bearing datasets.
EXPERIMENTS.extend(
    [
        spec(
            name="rq3_qwen2vl_answer_only_scienceqa",
            config=f"{BASE}/rq3_qwen2vl_answer_only_scienceqa.yaml",
            metrics=["mc_accuracy"],
            rq="RQ3/RQ7",
            domain="science",
            dataset="scienceqa",
            backbone="Qwen2-VL-2B",
            objective="answer_only",
            role="qwen_answer_only_control",
        ),
        spec(
            name="rq2_qwen2vl_generative_scienceqa",
            config=f"{BASE}/rq2_qwen2vl_generative_scienceqa.yaml",
            metrics=MC_METRICS,
            rq="RQ2/RQ3",
            domain="science",
            dataset="scienceqa",
            backbone="Qwen2-VL-2B",
            objective="rationale_generative",
            role="qwen_rationale_generative_control",
        ),
        spec(
            name="rq3_qwen2vl_answer_only_aokvqa",
            config=f"{BASE}/rq3_qwen2vl_answer_only_aokvqa.yaml",
            metrics=["mc_accuracy"],
            rq="RQ3/RQ7",
            domain="commonsense",
            dataset="aokvqa",
            backbone="Qwen2-VL-2B",
            objective="answer_only",
            role="qwen_answer_only_control",
        ),
        spec(
            name="rq2_qwen2vl_generative_aokvqa",
            config=f"{BASE}/rq2_qwen2vl_generative_aokvqa.yaml",
            metrics=MC_METRICS,
            rq="RQ2/RQ3",
            domain="commonsense",
            dataset="aokvqa",
            backbone="Qwen2-VL-2B",
            objective="rationale_generative",
            role="qwen_rationale_generative_control",
        ),
        spec(
            name="rq3_qwen2vl_explanation_aware_aokvqa",
            config=f"{BASE}/rq3_qwen2vl_explanation_aware_aokvqa.yaml",
            metrics=MC_METRICS,
            rq="RQ3/RQ4",
            domain="commonsense",
            dataset="aokvqa",
            backbone="Qwen2-VL-2B",
            objective="explanation_aware_alpha_0.5",
            role="domain_explanation_aware",
        ),
        spec(
            name="rq3_qwen2vl_length_aware_scienceqa",
            config=f"{BASE}/rq3_qwen2vl_explanation_aware_scienceqa.yaml",
            overrides=["objective.alpha_mode=length_aware", "objective.answer_weight_multiplier=1.0"],
            metrics=MC_METRICS,
            rq="RQ3",
            domain="science",
            dataset="scienceqa",
            backbone="Qwen2-VL-2B",
            objective="explanation_aware_length_aware",
            role="length_aware_alpha_control",
        ),
        spec(
            name="rq3_qwen2vl_length_aware_aokvqa",
            config=f"{BASE}/rq3_qwen2vl_explanation_aware_aokvqa.yaml",
            overrides=["objective.alpha_mode=length_aware", "objective.answer_weight_multiplier=1.0"],
            metrics=MC_METRICS,
            rq="RQ3",
            domain="commonsense",
            dataset="aokvqa",
            backbone="Qwen2-VL-2B",
            objective="explanation_aware_length_aware",
            role="length_aware_alpha_control",
        ),
    ]
)

# RQ4/RQ7 domain coverage. ChartQA/DocVQA/VQAv2 have no gold rationales, so
# these are explicit answer-only/generative fallback runs. The historical run
# names are kept for continuity, but they must not be interpreted as
# rationale-supervised training.
EXPERIMENTS.extend(
    [
        spec(
            name="rq4_qwen2vl_explanation_aware_chartqa",
            config=f"{BASE}/rq4_qwen2vl_explanation_aware_chartqa.yaml",
            metrics=None,
            rq="RQ4/RQ7",
            domain="charts",
            dataset="chartqa",
            backbone="Qwen2-VL-2B",
            objective="answer_only_fallback",
            role="domain_run",
        ),
        spec(
            name="rq4_qwen2vl_explanation_aware_docvqa",
            config=f"{BASE}/rq4_qwen2vl_explanation_aware_docvqa.yaml",
            metrics=None,
            rq="RQ4/RQ7",
            domain="documents",
            dataset="docvqa",
            backbone="Qwen2-VL-2B",
            objective="answer_only_fallback",
            role="domain_run",
        ),
        spec(
            name="rq4_qwen2vl_explanation_aware_vqav2",
            config=f"{BASE}/rq4_qwen2vl_explanation_aware_vqav2.yaml",
            metrics=None,
            rq="RQ4/RQ7",
            domain="natural",
            dataset="vqav2",
            backbone="Qwen2-VL-2B",
            objective="answer_only_fallback",
            role="domain_run",
        ),
    ]
)

# RQ6 scale comparison: 2B alpha=0.5 above vs 7B alpha=0.5 below.
EXPERIMENTS.append(
    spec(
        name="rq6_qwen2vl_7b_explanation_aware_scienceqa",
        config=f"{BASE}/rq6_qwen2vl_7b_explanation_aware_scienceqa.yaml",
        metrics=MC_METRICS,
        rq="RQ6",
        domain="science",
        dataset="scienceqa",
        backbone="Qwen2-VL-7B",
        objective="explanation_aware_alpha_0.5",
        role="scale",
    )
)

RUN_BY_NAME = {e["name"]: e for e in EXPERIMENTS}
EXPECTED_RUNS = list(RUN_BY_NAME)
duplicate_names = [n for n in EXPECTED_RUNS if EXPECTED_RUNS.count(n) > 1]
if duplicate_names:
    raise ValueError(f"duplicate experiment names: {sorted(set(duplicate_names))}")


def result_ready(run_name: str) -> bool:
    exp = RUN_BY_NAME.get(run_name)
    return bool(exp) and result_state(exp) == "current"


DATASET_LABELS = {
    "science": "ScienceQA",
    "scienceqa": "ScienceQA",
    "aokvqa": "A-OKVQA",
    "charts": "ChartQA",
    "chartqa": "ChartQA",
    "documents": "DocVQA",
    "docvqa": "DocVQA",
    "natural": "VQAv2",
    "vqav2": "VQAv2",
}

OBJECTIVE_LABELS = {
    "generative": "Generative",
    "rationale_generative": "Rationale+answer CE",
    "answer_only": "Answer-only",
    "contrastive_enhanced": "Generative + contrastive",
    "explanation_aware_alpha_0": "Explanation only",
    "explanation_aware_alpha_0.1": "Expl.-aware alpha=0.10",
    "explanation_aware_alpha_0.25": "Expl.-aware alpha=0.25",
    "explanation_aware_alpha_0.5": "Expl.-aware alpha=0.50",
    "explanation_aware_alpha_0.75": "Expl.-aware alpha=0.75",
    "explanation_aware_alpha_1": "Answer-span only",
    "explanation_aware_length_aware": "Expl.-aware length-aware",
    "answer_only_fallback": "Answer-only fallback",
}

HEADLINE_METRIC_BY_DATASET = {
    "scienceqa": "MC accuracy",
    "aokvqa": "MC accuracy",
    "chartqa": "relaxed accuracy",
    "docvqa": "ANLS",
    "vqav2": "VQA accuracy",
}

TRANSFER_LABELS = {
    "science_answer_only": "ScienceQA\nanswer-only",
    "science_rationale_ce": "ScienceQA\nrationale CE",
    "science_expl_aware": "ScienceQA\nexpl.-aware",
    "aokvqa_answer_only": "A-OKVQA\nanswer-only",
    "aokvqa_rationale_ce": "A-OKVQA\nrationale CE",
    "aokvqa_expl_aware": "A-OKVQA\nexpl.-aware",
    "charts_answer_only": "ChartQA\nanswer-only",
    "documents_answer_only": "DocVQA\nanswer-only",
    "vqav2_answer_only": "VQAv2\nanswer-only",
}


def pretty_dataset(name: str) -> str:
    return DATASET_LABELS.get(name, name)


def pretty_objective(name: str) -> str:
    return OBJECTIVE_LABELS.get(name, name.replace("_", " "))


def pretty_run_label(row: pd.Series | dict[str, Any]) -> str:
    return f"{pretty_dataset(row['dataset'])} · {pretty_objective(row['objective'])}"


def headline_metric_name(dataset: str) -> str:
    return HEADLINE_METRIC_BY_DATASET.get(dataset, "native score")


def rationale_availability(dataset: str) -> str:
    return "gold rationales" if dataset in {"scienceqa", "aokvqa"} else "answer labels only"


def supported_question_label(rq: str) -> str:
    return rq.replace("RQ", "question ")


def prompt_family_name(prompt_variant: str) -> str:
    if prompt_variant == "answer_only":
        return "answer-only"
    if prompt_variant == "explanation_then_answer":
        return "rationale+answer"
    return prompt_variant.replace("_", " ")


def cfg_alpha_summary(cfg) -> dict[str, Any]:
    return {
        "alpha": getattr(cfg.objective, "alpha", None),
        "alpha_mode": getattr(cfg.objective, "alpha_mode", "fixed"),
        "answer_weight_multiplier": getattr(cfg.objective, "answer_weight_multiplier", 1.0),
    }


def pretty_transfer_source(name: str) -> str:
    return TRANSFER_LABELS.get(name, name.replace("_", " "))


matrix_df = pd.DataFrame(EXPERIMENTS)
save_csv(matrix_df, "paper_run_matrix")
save_json(EXPERIMENTS, "paper_run_matrix")

print(f"{len(EXPERIMENTS)} runs queued:")
for e in EXPERIMENTS:
    print(f"  {e['name']:48s} | {e['backbone']:14s} | {e['dataset']:9s} | {e['objective']}")

ledger_rows: list[dict[str, Any]] = []
for e in EXPERIMENTS:
    cfg = materialize_exp_config(e)
    ledger_rows.append(
        {
            "run": e["name"],
            "display_label": pretty_run_label(e),
            "backbone": e["backbone"],
            "dataset": pretty_dataset(e["dataset"]),
            "domain": e["domain"],
            "train_cap": cfg.data.max_train,
            "eval_cap": cfg.data.max_eval,
            "rationale_availability": rationale_availability(e["dataset"]),
            "objective": pretty_objective(e["objective"]),
            "objective_impl": cfg.objective.name,
            "alpha_mode": getattr(cfg.objective, "alpha_mode", "fixed"),
            "alpha": getattr(cfg.objective, "alpha", np.nan),
            "answer_weight_multiplier": getattr(cfg.objective, "answer_weight_multiplier", 1.0),
            "prompt_family": prompt_family_name(cfg.data.prompt_variant),
            "metric": headline_metric_name(e["dataset"]),
            "supported_question": supported_question_label(e["rq"]),
        }
    )

experiment_ledger = pd.DataFrame(ledger_rows)
save_csv(experiment_ledger, "experiment_ledger")
save_json(ledger_rows, "experiment_ledger")
save_table(
    experiment_ledger[
        [
            "display_label",
            "backbone",
            "dataset",
            "train_cap",
            "eval_cap",
            "rationale_availability",
            "objective",
            "alpha_mode",
            "prompt_family",
            "metric",
            "supported_question",
        ]
    ],
    "experiment_ledger",
    "Experiment ledger: datasets, caps, prompt family, objective, and supported research question.",
    "tab:experiment_ledger",
)


# %% Cell 3 - config validation without loading any model
config_errors: dict[str, str] = {}
for exp in EXPERIMENTS:
    try:
        cfg = materialize_exp_config(exp)
    except Exception as exc:  # noqa: BLE001
        config_errors[exp["name"]] = "".join(
            traceback.format_exception_only(type(exc), exc)
        ).strip()

save_json(config_errors, "config_validation_errors", LOG_DIR)
if config_errors:
    raise RuntimeError(f"Config validation failed for {len(config_errors)} run(s): {config_errors}")
print("All configs load cleanly.")


# %% [markdown]
# ## What actually happens in the pipeline?
#
# Before running expensive training, this section makes the pipeline inspectable.
# It answers the questions that are easy to miss when everything is hidden behind
# `ExperimentRunner`:
#
# ```text
# dataset row
#   └─ image/question/answer/rationale
#       └─ prompt template
#           ├─ prompt  = what the model is conditioned on
#           └─ target  = what the model is trained to produce
#               └─ collator
#                   ├─ input_ids / image tensors / masks
#                   ├─ labels: prompt tokens = -100, target tokens = supervised
#                   └─ span_ids: explanation vs answer tokens when needed
#                       └─ objective
#                           ├─ generative: CE(target)
#                           ├─ explanation-aware: alpha*CE(answer)+(1-alpha)*CE(expl)
#                           └─ contrastive: CE(target)+w*InfoNCE(image, answer)
#                               └─ evaluator
#                                   ├─ raw model generation
#                                   ├─ parsed answer
#                                   └─ metric calculation
# ```
#
# The sample trace cell below goes one level deeper: it records the exact text
# provided to the template, the model-facing text after backbone wrapping, every
# collated tensor, every token id, every label id, which positions are masked
# with `-100`, the span tags used for explanation-aware loss, the forward logits
# shape, the generated token ids, and the decoded/evaluated output. The notebook
# displays the boundary window where supervision starts; the full token table is
# saved as `sample_trace_tokens.csv`.
#
# The generated markdown/CSV/JSON/figures are saved into `paper_artifacts/data`
# and `paper_artifacts/figures`. The sample figures can be included in the paper
# if the journal page budget allows it.

# %% Cell 4 - dataset, prompt, collator, model-input, loss, and output diagnostics
EXPLAIN_RUNS = [
    "rq3_qwen2vl_answer_only_scienceqa",
    "rq2_qwen2vl_generative_scienceqa",
    "rq3_alpha_050",
    "rq3_qwen2vl_length_aware_scienceqa",
    "rq3_qwen2vl_answer_only_aokvqa",
    "rq3_qwen2vl_explanation_aware_aokvqa",
    "rq3_qwen2vl_length_aware_aokvqa",
    "rq4_qwen2vl_explanation_aware_chartqa",
    "rq4_qwen2vl_explanation_aware_docvqa",
    "rq4_qwen2vl_explanation_aware_vqav2",
    "rq2_blip2_contrastive_scienceqa",
    "rq2_blip2_generative_scienceqa",
]


def explain_dataset_and_prompt(run_names: list[str]) -> None:
    sample_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    seen_datasets: set[str] = set()

    for run_name in run_names:
        exp = RUN_BY_NAME[run_name]
        cfg = materialize_exp_config(exp)
        ds = build_dataset(cfg.data, split=cfg.data.split_train)
        if len(ds) == 0:
            raise ValueError(f"{run_name}: no rows available for explanation preview")
        ex = ds[0]
        template = build_template(cfg.data.prompt_variant)
        prompt, target = template(ex)
        spans = template.target_spans(ex)

        sample_rows.append(sample_record(ex, 0, cfg.data.name) | {"run": run_name})
        prompt_rows.append(
            {
                "run": run_name,
                "dataset": cfg.data.name,
                "prompt_variant": cfg.data.prompt_variant,
                "objective": cfg.objective.name,
                "has_gold_rationale": ex.has_explanation,
                "prompt": prompt,
                "target": target,
                "target_spans": spans,
                "allowed_interpretation": (
                    "rationale-supervised"
                    if ex.has_explanation and cfg.objective.name == "explanation_aware"
                    else "answer-only / no gold rationale"
                    if not ex.has_explanation
                    else cfg.objective.name
                ),
            }
        )

        if cfg.data.name not in seen_datasets:
            seen_datasets.add(cfg.data.name)
            save_dataset_sample_figure(ex, cfg.data.name, 0)

    sample_df = pd.DataFrame(sample_rows)
    prompt_df = pd.DataFrame(prompt_rows)
    save_csv(sample_df, "dataset_sample_preview")
    save_csv(prompt_df, "prompt_template_preview")
    save_json(sample_rows, "dataset_sample_preview")
    save_json(prompt_rows, "prompt_template_preview")

    md_lines = ["# Dataset and prompt preview\n"]
    for row in prompt_rows:
        md_lines.extend(
            [
                f"## {row['run']} ({row['dataset']})",
                f"- prompt variant: `{row['prompt_variant']}`",
                f"- objective: `{row['objective']}`",
                f"- interpretation: {row['allowed_interpretation']}",
                f"- has gold rationale: `{row['has_gold_rationale']}`",
                "",
                "### Prompt",
                "```text",
                wrap(row["prompt"], 100),
                "```",
                "### Target",
                "```text",
                wrap(row["target"], 100),
                "```",
                f"- spans: `{row['target_spans']}`",
                "",
            ]
        )
    save_markdown("\n".join(md_lines), "dataset_prompt_preview")

    display(sample_df[["run", "dataset", "question", "answer", "has_explanation", "image_size"]])
    display(prompt_df[["run", "dataset", "prompt_variant", "objective", "has_gold_rationale", "allowed_interpretation"]])


def decoded_supervised_target(wrapper, labels) -> str:
    supervised = labels[labels != -100]
    return wrapper.processor.tokenizer.decode(supervised, skip_special_tokens=False)


SPAN_NAMES = {
    SPAN_IGNORE: "ignore",
    SPAN_EXPLANATION: "explanation",
    SPAN_ANSWER: "answer",
}


def tensor_summary(batch: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in batch.items():
        if torch.is_tensor(value):
            row = {
                "tensor": key,
                "shape": tuple(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
            if value.numel() and value.is_floating_point():
                row["min"] = float(value.detach().float().min().item())
                row["max"] = float(value.detach().float().max().item())
            elif value.numel():
                row["min"] = int(value.detach().min().item())
                row["max"] = int(value.detach().max().item())
            rows.append(row)
        else:
            rows.append({"tensor": key, "type": type(value).__name__})
    return rows


def token_label_table(
    wrapper,
    batch: dict[str, torch.Tensor],
    span_ids: torch.Tensor | None,
    prompt_len: int,
    run_name: str,
) -> list[dict[str, Any]]:
    tok = wrapper.processor.tokenizer
    input_ids = batch["input_ids"][0].detach().cpu().tolist()
    labels = batch["labels"][0].detach().cpu().tolist()
    attention = batch.get("attention_mask")
    attention_values = (
        attention[0].detach().cpu().tolist()
        if torch.is_tensor(attention)
        else [None] * len(input_ids)
    )
    span_values = (
        span_ids[0].detach().cpu().tolist()
        if span_ids is not None
        else [SPAN_IGNORE] * len(input_ids)
    )
    tokens = tok.convert_ids_to_tokens(input_ids)
    pad_id = tok.pad_token_id
    rows: list[dict[str, Any]] = []
    for pos, (input_id, label_id, attn, span_id, token) in enumerate(
        zip(input_ids, labels, attention_values, span_values, tokens)
    ):
        supervised = label_id != LABEL_IGNORE
        if attn == 0 or input_id == pad_id:
            segment = "padding"
        elif pos < prompt_len:
            segment = "image+prompt masked"
        elif supervised:
            segment = "target supervised"
        else:
            segment = "target/special masked"
        label_token = "IGNORE" if not supervised else tok.convert_ids_to_tokens([label_id])[0]
        rows.append(
            {
                "run": run_name,
                "position": pos,
                "segment": segment,
                "input_id": int(input_id),
                "input_token": token,
                "input_decoded": tok.decode([input_id], skip_special_tokens=False),
                "attention_mask": attn,
                "label_id": int(label_id),
                "label_token": label_token,
                "label_decoded": "" if not supervised else tok.decode([label_id], skip_special_tokens=False),
                "supervised": bool(supervised),
                "span_id": int(span_id),
                "span_name": SPAN_NAMES.get(int(span_id), f"unknown_{span_id}"),
            }
        )
    return rows


def boundary_window(rows: list[dict[str, Any]], radius: int = 45) -> pd.DataFrame:
    supervised_positions = [r["position"] for r in rows if r["supervised"]]
    if supervised_positions:
        start = max(min(supervised_positions) - radius, 0)
        end = min(min(supervised_positions) + radius, len(rows) - 1)
    else:
        start = 0
        end = min(2 * radius, len(rows) - 1)
    return pd.DataFrame([r for r in rows if start <= r["position"] <= end])


def generation_trace(wrapper, ex, template, cfg) -> dict[str, Any]:
    prompt = template.prompt(ex)
    enc = wrapper.build_inputs(
        [ex.image.convert("RGB")],
        [prompt],
        padding=False,
        truncation=True,
        max_length=cfg.eval.max_length,
        for_generation=True,
    )
    enc_device = move_to_device(dict(enc), wrapper.device, wrapper.dtype)
    with torch.no_grad():
        out_ids = wrapper.model.generate(
            **enc_device,
            max_new_tokens=cfg.eval.max_new_tokens,
            do_sample=False,
        )
    tok = wrapper.processor.tokenizer
    full_decoded = tok.decode(out_ids[0], skip_special_tokens=True)
    input_len = int(enc["input_ids"].shape[1])
    new_ids = out_ids[0][input_len:] if out_ids.shape[1] > input_len else out_ids[0]
    continuation = tok.decode(new_ids, skip_special_tokens=True).strip()
    if full_decoded.startswith(prompt):
        continuation = full_decoded[len(prompt) :].strip()
    reasoning, answer = split_reasoning_answer(continuation)
    if not answer:
        answer = clean_generated_answer(continuation)
    return {
        "prompt": prompt,
        "model_prompt": wrapper.prompt_to_model_text(prompt, add_generation_prompt=True),
        "generation_input_shape": tuple(enc["input_ids"].shape),
        "generated_full_token_count": int(out_ids.shape[1]),
        "generated_new_token_count": int(new_ids.shape[0]),
        "new_token_ids": [int(x) for x in new_ids.detach().cpu().tolist()[:80]],
        "decoded_full": full_decoded,
        "decoded_continuation": continuation,
        "parsed_reasoning": reasoning,
        "parsed_answer": answer,
    }


def build_sample_trace(run_names: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    tensor_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []

    groups: dict[tuple[str, str], list[str]] = {}
    for run_name in run_names:
        cfg = materialize_exp_config(RUN_BY_NAME[run_name])
        groups.setdefault((cfg.model.name, cfg.model.pretrained), []).append(run_name)

    for _, group_runs in groups.items():
        first_cfg = materialize_exp_config(RUN_BY_NAME[group_runs[0]])
        print(f"\nloading trace model: {first_cfg.model.name} / {first_cfg.model.pretrained}")
        wrapper = build_model(first_cfg.model)
        wrapper.eval()
        try:
            for run_name in group_runs:
                exp = RUN_BY_NAME[run_name]
                cfg = materialize_exp_config(exp)
                obj = build_objective(cfg.objective)
                template = build_template(cfg.data.prompt_variant)
                train_ds = build_dataset(cfg.data, split=cfg.data.split_train)
                eval_ds = build_dataset(cfg.data, split=cfg.data.split_eval)
                ex = train_ds[0]
                prompt, target = template(ex)
                full_text = prompt + target
                image = ex.image.convert("RGB")
                prompt_len = wrapper.input_length(image, prompt, cfg.data.max_length)
                full_len = wrapper.input_length(image, full_text, cfg.data.max_length)

                actual_collator = build_collator(cfg.data, wrapper, tag_spans=obj.requires_span_ids)
                batch_cpu = actual_collator([ex])
                span_cpu = batch_cpu.get("span_ids")
                span_used_by_objective = span_cpu is not None
                if span_cpu is None:
                    span_preview_collator = build_collator(cfg.data, wrapper, tag_spans=True)
                    span_cpu = span_preview_collator([ex])["span_ids"]

                batch = move_to_device(batch_cpu, wrapper.device, wrapper.dtype)
                with torch.no_grad():
                    loss_out = obj.compute(wrapper, batch)
                    forward_batch = {k: v for k, v in batch.items() if k != "span_ids"}
                    out = wrapper.forward(forward_batch)

                trace_tokens = token_label_table(wrapper, batch_cpu, span_cpu, prompt_len, run_name)
                token_rows.extend(trace_tokens)
                for row in tensor_summary(batch_cpu):
                    tensor_rows.append({"run": run_name, "stage": "collator_cpu", **row})
                tensor_rows.append(
                    {
                        "run": run_name,
                        "stage": "model_forward",
                        "tensor": "logits",
                        "shape": tuple(out.logits.shape),
                        "dtype": str(out.logits.dtype),
                        "device": str(out.logits.device),
                    }
                )

                gen = generation_trace(wrapper, eval_ds[0], template, cfg)
                ev = Evaluator(wrapper, eval_ds, template, cfg.eval, build_metrics(cfg.eval.metrics))
                pred = ev._predict_one(eval_ds[0])
                one_metrics: dict[str, float] = {}
                for metric in ev.metrics:
                    if ev._applicable(metric, [pred]):
                        one_metrics.update(metric.compute([pred]))

                supervised_decoded = decoded_supervised_target(wrapper, batch_cpu["labels"][0])
                summary_rows.append(
                    {
                        "run": run_name,
                        "dataset": cfg.data.name,
                        "objective_label": exp["objective"],
                        "objective_impl": cfg.objective.name,
                        "prompt_variant": cfg.data.prompt_variant,
                        "span_ids_used_by_objective": span_used_by_objective,
                        "question": ex.question,
                        "gold_answer": ex.answer,
                        "gold_explanation": ex.explanation,
                        "prompt_text": prompt,
                        "target_text": target,
                        "full_training_text": full_text,
                        "model_full_text_prefix": short(wrapper.prompt_to_model_text(full_text), 1500),
                        "image_size": getattr(image, "size", None),
                        "prompt_token_len": prompt_len,
                        "full_token_len": full_len,
                        "collated_seq_len": int(batch_cpu["input_ids"].shape[1]),
                        "supervised_token_count": int((batch_cpu["labels"][0] != LABEL_IGNORE).sum().item()),
                        "decoded_supervised_target": supervised_decoded,
                        "loss_components": loss_out.components,
                        "generation_prompt": gen["prompt"],
                        "generation_input_shape": gen["generation_input_shape"],
                        "generated_new_token_count": gen["generated_new_token_count"],
                        "generation_new_token_ids_first80": gen["new_token_ids"],
                        "decoded_generation_full": gen["decoded_full"],
                        "decoded_generation_continuation": gen["decoded_continuation"],
                        "parsed_reasoning": gen["parsed_reasoning"],
                        "parsed_answer": gen["parsed_answer"],
                        "evaluator_prediction": pred.predicted_text,
                        "evaluator_reasoning": pred.reasoning,
                        "one_example_metrics": one_metrics,
                    }
                )
                print(f"trace complete: {run_name}")
        finally:
            del wrapper
            free_gpu()
    return summary_rows, tensor_rows, token_rows


def sample_trace_markdown(
    summary_rows: list[dict[str, Any]],
    tensor_rows: list[dict[str, Any]],
    token_rows: list[dict[str, Any]],
) -> str:
    lines = ["# End-to-end sample trace\n"]
    tensor_df = pd.DataFrame(tensor_rows)
    for row in summary_rows:
        run = row["run"]
        token_window = boundary_window([r for r in token_rows if r["run"] == run], radius=35)
        tensor_sub = tensor_df[tensor_df["run"] == run] if not tensor_df.empty else pd.DataFrame()
        lines.extend(
            [
                f"## {run}",
                f"- dataset: `{row['dataset']}`",
                f"- objective label: `{row['objective_label']}`",
                f"- objective implementation: `{row['objective_impl']}`",
                f"- prompt variant: `{row['prompt_variant']}`",
                f"- span IDs used by objective: `{row['span_ids_used_by_objective']}`",
                f"- image size: `{row['image_size']}`",
                f"- prompt tokens: `{row['prompt_token_len']}`",
                f"- full training tokens: `{row['full_token_len']}`",
                f"- collated sequence length: `{row['collated_seq_len']}`",
                f"- supervised tokens: `{row['supervised_token_count']}`",
                "",
                "### 1. Raw dataset example",
                "```text",
                wrap(f"Question: {row['question']}", 100),
                wrap(f"Gold answer: {row['gold_answer']}", 100),
                wrap(f"Gold rationale: {row['gold_explanation']}", 100),
                "```",
                "### 2. Prompt and target",
                "```text",
                "PROMPT:",
                wrap(row["prompt_text"], 100),
                "",
                "TARGET:",
                wrap(row["target_text"], 100),
                "```",
                "### 3. Model-facing full text",
                "```text",
                wrap(row["model_full_text_prefix"], 100),
                "```",
                "### 4. Collator tensors",
                "```text",
                tensor_sub[["stage", "tensor", "shape", "dtype", "device"]].to_string(index=False)
                if not tensor_sub.empty
                else "(no tensor summary)",
                "```",
                "### 5. Token/label boundary window",
                "```text",
                token_window[
                    [
                        "position",
                        "segment",
                        "input_id",
                        "input_token",
                        "label_id",
                        "label_token",
                        "supervised",
                        "span_name",
                    ]
                ].to_string(index=False),
                "```",
                "### 6. Decoded supervised target",
                "```text",
                wrap(row["decoded_supervised_target"], 100),
                "```",
                "### 7. Objective and forward pass",
                f"- loss components: `{row['loss_components']}`",
                "",
                "### 8. Generation and decoding",
                "```text",
                "PROMPT:",
                wrap(row["generation_prompt"], 100),
                "",
                "NEW TOKEN IDS (first 80):",
                str(row["generation_new_token_ids_first80"]),
                "",
                "DECODED CONTINUATION:",
                wrap(row["decoded_generation_continuation"], 100),
                "```",
                f"- parsed reasoning: `{short(row['parsed_reasoning'], 300)}`",
                f"- parsed answer: `{row['parsed_answer']}`",
                f"- evaluator prediction: `{row['evaluator_prediction']}`",
                f"- one-example metrics: `{row['one_example_metrics']}`",
                "",
            ]
        )
    return "\n".join(lines)


def inspect_model_group(run_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[str]] = {}
    for run_name in run_names:
        cfg = materialize_exp_config(RUN_BY_NAME[run_name])
        key = (cfg.model.name, cfg.model.pretrained)
        groups.setdefault(key, []).append(run_name)

    for _, group_runs in groups.items():
        first_cfg = materialize_exp_config(RUN_BY_NAME[group_runs[0]])
        print(f"\nloading diagnostic model: {first_cfg.model.name} / {first_cfg.model.pretrained}")
        wrapper = build_model(first_cfg.model)
        wrapper.eval()
        try:
            for run_name in group_runs:
                exp = RUN_BY_NAME[run_name]
                cfg = materialize_exp_config(exp)
                obj = build_objective(cfg.objective)
                template = build_template(cfg.data.prompt_variant)
                train_ds = build_dataset(cfg.data, split=cfg.data.split_train)
                eval_ds = build_dataset(cfg.data, split=cfg.data.split_eval)
                n_batch = min(4 if cfg.objective.name == "contrastive" else 1, len(train_ds))
                examples = [train_ds[i] for i in range(n_batch)]
                collator = build_collator(cfg.data, wrapper, tag_spans=obj.requires_span_ids)
                batch_cpu = collator(examples)
                batch = move_to_device(batch_cpu, wrapper.device, wrapper.dtype)
                with torch.no_grad():
                    loss_out = obj.compute(wrapper, batch)

                ex = examples[0]
                prompt, target = template(ex)
                decoded_input = wrapper.processor.tokenizer.decode(
                    batch_cpu["input_ids"][0], skip_special_tokens=False
                )
                span_summary: dict[str, int] = {}
                if "span_ids" in batch_cpu:
                    vals, counts = torch.unique(batch_cpu["span_ids"][0], return_counts=True)
                    span_summary = {str(int(v)): int(c) for v, c in zip(vals, counts)}

                eval_ex = eval_ds[0]
                raw = generate_continuation(
                    wrapper,
                    eval_ex,
                    template,
                    max_new_tokens=cfg.eval.max_new_tokens,
                    max_length=cfg.eval.max_length,
                )
                reasoning, parsed_answer = split_reasoning_answer(raw)
                if not parsed_answer:
                    parsed_answer = clean_generated_answer(raw)

                ev = Evaluator(wrapper, eval_ds, template, cfg.eval, build_metrics(cfg.eval.metrics))
                pred = ev._predict_one(eval_ex)
                one_metrics: dict[str, float] = {}
                for metric in ev.metrics:
                    if ev._applicable(metric, [pred]):
                        one_metrics.update(metric.compute([pred]))

                rows.append(
                    {
                        "run": run_name,
                        "dataset": cfg.data.name,
                        "backbone": exp["backbone"],
                        "objective": cfg.objective.name,
                        "prompt_variant": cfg.data.prompt_variant,
                        "train_question": ex.question,
                        "train_target": target,
                        "decoded_supervised_target": decoded_supervised_target(wrapper, batch_cpu["labels"][0]),
                        "input_shape": tuple(batch_cpu["input_ids"].shape),
                        "supervised_tokens": int((batch_cpu["labels"][0] != -100).sum().item()),
                        "span_summary": span_summary,
                        "loss_components": loss_out.components,
                        "model_input_preview": short(decoded_input, 1200),
                        "eval_question": eval_ex.question,
                        "gold": eval_ex.answer,
                        "raw_generation": raw,
                        "parsed_reasoning": reasoning,
                        "parsed_answer": parsed_answer,
                        "evaluator_prediction": pred.predicted_text,
                        "one_example_metrics": one_metrics,
                    }
                )
                print(f"diagnostic complete: {run_name}")
        finally:
            del wrapper
            free_gpu()
    return rows


explain_dataset_and_prompt(EXPLAIN_RUNS)


def stable_example_id(ex) -> str | None:
    meta = getattr(ex, "metadata", None) or {}
    for key in ("id", "question_id", "qid", "image_id", "questionId", "question_id_str"):
        value = meta.get(key)
        if value is not None:
            return f"{key}:{value}"
    return None


def fallback_example_fingerprint(ex) -> str:
    payload = {
        "question": ex.question,
        "answer": ex.answer,
        "choices": ex.choices,
        "image_size": getattr(ex.image, "size", None),
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def collect_split_records(ds, cap: int, *, dataset: str, split: str) -> tuple[list[dict[str, Any]], set[str], bool]:
    records: list[dict[str, Any]] = []
    keys: set[str] = set()
    has_native_id = True
    for i in range(min(cap, len(ds))):
        ex = ds[i]
        native_key = stable_example_id(ex)
        key = native_key
        if native_key is None:
            has_native_id = False
            key = f"fingerprint:{fallback_example_fingerprint(ex)}"
        keys.add(key)
        records.append(
            {
                "dataset": dataset,
                "split": split,
                "selected_index": i,
                "example_key": key,
                "native_id_available": native_key is not None,
                "question_hash": hashlib.sha256(str(ex.question).encode("utf-8")).hexdigest()[:16],
                "answer_hash": hashlib.sha256(str(ex.answer).encode("utf-8")).hexdigest()[:16],
                "image_size": getattr(ex.image, "size", None),
            }
        )
    return records, keys, has_native_id


split_audit_rows: list[dict[str, Any]] = []
split_record_rows: list[dict[str, Any]] = []
seen_split_sigs: set[tuple[str, str, str, int, int]] = set()
for exp in EXPERIMENTS:
    cfg = materialize_exp_config(exp)
    sig = (
        cfg.data.name,
        cfg.data.split_train,
        cfg.data.split_eval,
        int(cfg.data.max_train),
        int(cfg.data.max_eval),
    )
    if sig in seen_split_sigs:
        continue
    seen_split_sigs.add(sig)
    try:
        train_ds = build_dataset(cfg.data, split=cfg.data.split_train)
        eval_ds = build_dataset(cfg.data, split=cfg.data.split_eval)
        train_records, train_keys, train_native = collect_split_records(
            train_ds,
            cfg.data.max_train,
            dataset=cfg.data.name,
            split=cfg.data.split_train,
        )
        eval_records, eval_keys, eval_native = collect_split_records(
            eval_ds,
            cfg.data.max_eval,
            dataset=cfg.data.name,
            split=cfg.data.split_eval,
        )
        split_record_rows.extend(train_records)
        split_record_rows.extend(eval_records)
        overlap = sorted(train_keys & eval_keys)
        split_audit_rows.append(
            {
                "dataset": cfg.data.name,
                "train_split": cfg.data.split_train,
                "eval_split": cfg.data.split_eval,
                "train_cap": cfg.data.max_train,
                "eval_cap": cfg.data.max_eval,
                "train_rows_loaded": len(train_ds),
                "eval_rows_loaded": len(eval_ds),
                "native_ids_available": bool(train_native and eval_native),
                "overlap_count": len(overlap),
                "overlap_keys_preview": overlap[:10],
                "audit_method": "native ids" if train_native and eval_native else "question/answer/image fingerprint fallback",
            }
        )
    except Exception as exc:  # noqa: BLE001
        split_audit_rows.append(
            {
                "dataset": cfg.data.name,
                "train_split": cfg.data.split_train,
                "eval_split": cfg.data.split_eval,
                "train_cap": cfg.data.max_train,
                "eval_cap": cfg.data.max_eval,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

split_audit = pd.DataFrame(split_audit_rows)
split_records = pd.DataFrame(split_record_rows)
save_csv(split_audit, "split_leakage_audit")
save_json(split_audit_rows, "split_leakage_audit")
save_csv(split_records, "selected_split_records")
save_json(split_record_rows, "selected_split_records")
save_table(
    split_audit.reindex(
        columns=[
            "dataset",
            "train_split",
            "eval_split",
            "train_cap",
            "eval_cap",
            "native_ids_available",
            "overlap_count",
            "audit_method",
        ]
    ),
    "split_leakage_audit",
    "Split and leakage audit for capped training/evaluation data.",
    "tab:split_leakage_audit",
)

TRACE_RUNS = [
    r.strip()
    for r in os.environ.get(
        "PAPER_TRACE_RUNS",
        ",".join(
            [
                "rq3_qwen2vl_answer_only_scienceqa",
                "rq2_qwen2vl_generative_scienceqa",
                "rq3_alpha_050",
                "rq4_qwen2vl_explanation_aware_chartqa",
            ]
        ),
    ).split(",")
    if r.strip()
]

if os.environ.get("PAPER_SKIP_MODEL_TRACE", "0") != "1":
    trace_summary_rows, trace_tensor_rows, trace_token_rows = build_sample_trace(TRACE_RUNS)
    trace_summary_df = pd.DataFrame(trace_summary_rows)
    trace_tensor_df = pd.DataFrame(trace_tensor_rows)
    trace_token_df = pd.DataFrame(trace_token_rows)
    save_json(trace_summary_rows, "sample_trace_summary")
    save_json(trace_tensor_rows, "sample_trace_tensors")
    save_csv(trace_summary_df, "sample_trace_summary")
    save_csv(trace_tensor_df, "sample_trace_tensors")
    save_csv(trace_token_df, "sample_trace_tokens")
    save_markdown(
        sample_trace_markdown(trace_summary_rows, trace_tensor_rows, trace_token_rows),
        "sample_trace_walkthrough",
    )
    display(
        trace_summary_df[
            [
                "run",
                "dataset",
                "objective_label",
                "prompt_variant",
                "prompt_token_len",
                "collated_seq_len",
                "supervised_token_count",
                "loss_components",
                "parsed_answer",
                "one_example_metrics",
            ]
        ]
    )
    for run_name in TRACE_RUNS:
        run_tokens = [r for r in trace_token_rows if r["run"] == run_name]
        if run_tokens:
            display(Markdown(f"### Token/label boundary window: `{run_name}`"))
            display(
                boundary_window(run_tokens, radius=int(os.environ.get("PAPER_TRACE_TOKEN_RADIUS", "45")))[
                    [
                        "position",
                        "segment",
                        "input_id",
                        "input_token",
                        "label_id",
                        "label_token",
                        "supervised",
                        "span_name",
                    ]
                ]
            )
else:
    print("Skipping model trace because PAPER_SKIP_MODEL_TRACE=1")

if os.environ.get("PAPER_SKIP_MODEL_DIAGNOSTICS", "0") != "1":
    diagnostic_rows = inspect_model_group(EXPLAIN_RUNS)
    save_json(diagnostic_rows, "model_io_loss_metric_preview")
    save_markdown(
        "\n\n".join(
            [
                f"# Model I/O, loss, and metric preview",
                *[
                    "\n".join(
                        [
                            f"## {row['run']} ({row['dataset']})",
                            f"- objective: `{row['objective']}`",
                            f"- prompt variant: `{row['prompt_variant']}`",
                            f"- input shape: `{row['input_shape']}`",
                            f"- supervised tokens: `{row['supervised_tokens']}`",
                            f"- span summary: `{row['span_summary']}`",
                            f"- loss components: `{row['loss_components']}`",
                            "",
                            "### Decoded supervised target",
                            "```text",
                            wrap(row["decoded_supervised_target"], 100),
                            "```",
                            "### Model input preview",
                            "```text",
                            wrap(row["model_input_preview"], 100),
                            "```",
                            "### Raw generation and parsed answer",
                            "```text",
                            wrap(row["raw_generation"], 100),
                            "```",
                            f"- parsed answer: `{row['parsed_answer']}`",
                            f"- evaluator prediction: `{row['evaluator_prediction']}`",
                            f"- one-example metrics: `{row['one_example_metrics']}`",
                        ]
                    )
                    for row in diagnostic_rows
                ],
            ]
        ),
        "model_io_loss_metric_preview",
    )
    display(pd.DataFrame(diagnostic_rows)[["run", "dataset", "objective", "prompt_variant", "supervised_tokens", "loss_components", "parsed_answer", "one_example_metrics"]])
else:
    print("Skipping model diagnostics because PAPER_SKIP_MODEL_DIAGNOSTICS=1")


# %% [markdown]
# ## Pre-flight checks
#
# These checks are intentionally run before training:
#
# - masking preflight catches prompt leakage and empty supervised targets;
# - contrastive preflight checks that BLIP-2 InfoNCE reaches both the text
#   projection and trainable Q-Former parameters.

# %% Cell 5 - masking and contrastive preflight
def preflight_masking(cfg) -> dict[str, Any]:
    obj = build_objective(cfg.objective)
    ds = build_dataset(cfg.data, split=cfg.data.split_train)
    if len(ds) == 0:
        raise ValueError(f"{cfg.name}: dataset is empty after filtering")

    wrapper = build_model(cfg.model)
    try:
        n = min(2, len(ds))
        collator = build_collator(cfg.data, wrapper, tag_spans=obj.requires_span_ids)
        batch = collator([ds[i] for i in range(n)])
        decoded_targets = []
        for row in batch["labels"]:
            supervised = row[row != -100]
            decoded = wrapper.processor.tokenizer.decode(supervised, skip_special_tokens=True)
            if len(supervised) == 0:
                raise AssertionError(f"EMPTY TARGET in {cfg.name}")
            if "Question:" in decoded or "Options:" in decoded:
                raise AssertionError(f"PROMPT LEAK in {cfg.name}: {decoded!r}")
            if "Answer:" not in decoded:
                raise AssertionError(f"NO ANSWER CUE in {cfg.name}: {decoded!r}")
            decoded_targets.append(decoded[:180])
        return {
            "dataset_len": len(ds),
            "requires_span_ids": obj.requires_span_ids,
            "decoded_targets": decoded_targets,
        }
    finally:
        del wrapper
        free_gpu()


def preflight_contrastive(cfg) -> dict[str, Any]:
    wrapper = build_model(cfg.model)
    try:
        obj = build_objective(cfg.objective)
        assert obj.requires_span_ids, "contrastive objective must request span_ids"
        assert cfg.objective.contrastive_weight > 0, "contrastive_weight must be nonzero"

        ds = build_dataset(cfg.data, split=cfg.data.split_train)
        collator = build_collator(cfg.data, wrapper, tag_spans=obj.requires_span_ids)
        batch_size = min(4, len(ds))
        batch = collator([ds[i] for i in range(batch_size)])
        batch = move_to_device(batch, wrapper.device, wrapper.dtype)

        wrapper.train()
        forward_batch = {k: v for k, v in batch.items() if k != "span_ids"}
        l_gen = wrapper.forward(forward_batch).loss
        image_embeds, text_embeds = wrapper.contrastive_features(batch)

        assert image_embeds.ndim == text_embeds.ndim == 2
        assert image_embeds.shape == text_embeds.shape
        assert image_embeds.shape[0] == batch_size
        assert torch.isfinite(image_embeds).all()
        assert torch.isfinite(text_embeds).all()

        l_nce = info_nce(image_embeds, text_embeds, cfg.objective.temperature)
        loss = l_gen + cfg.objective.contrastive_weight * l_nce
        assert torch.isfinite(loss)
        loss.backward()

        projection = wrapper.model.contrastive_text_projection
        projection_grad = projection.weight.grad
        assert projection_grad is not None
        assert torch.isfinite(projection_grad).all()
        assert projection_grad.abs().sum().item() > 0

        qformer_grad = sum(
            0.0 if p.grad is None else float(p.grad.detach().abs().sum().item())
            for n, p in wrapper.model.named_parameters()
            if p.requires_grad and "qformer" in n
        )
        assert qformer_grad > 0, "InfoNCE did not reach trainable Q-Former parameters"

        with torch.no_grad():
            sim = (
                torch.nn.functional.normalize(image_embeds.float(), dim=-1)
                @ torch.nn.functional.normalize(text_embeds.float(), dim=-1).t()
            )
            diag = float(sim.diag().mean().item())
            denom = max(sim.numel() - batch_size, 1)
            off_diag = float((sim.sum() - sim.diag().sum()).item() / denom)

        wrapper.model.zero_grad(set_to_none=True)
        return {
            "shape": tuple(image_embeds.shape),
            "l_generative": float(l_gen.detach().float().item()),
            "l_infonce": float(l_nce.detach().float().item()),
            "diag_cosine": diag,
            "off_diag_cosine": off_diag,
            "qformer_grad_l1": qformer_grad,
            "projection_grad_l1": float(projection_grad.detach().abs().sum().item()),
        }
    finally:
        del wrapper
        free_gpu()


masking_log_path = LOG_DIR / "preflight_masking.json"
masking_log = json.loads(masking_log_path.read_text()) if masking_log_path.exists() else {}

seen_signatures: set[tuple[str, str, str, str, str]] = set()
for exp in EXPERIMENTS:
    cfg = materialize_exp_config(exp)
    sig = (
        cfg.model.name,
        cfg.model.pretrained,
        cfg.data.name,
        cfg.data.prompt_variant,
        cfg.objective.name,
    )
    if sig in seen_signatures:
        continue
    seen_signatures.add(sig)
    cached = masking_log.get(exp["name"])
    digest = config_digest(cfg)
    sig_key = "|".join(sig)
    if (
        isinstance(cached, dict)
        and cached.get("config_signature") == sig_key
        and cached.get("config_digest") == digest
    ):
        print(f"skip masking preflight {exp['name']} (cached)")
        continue
    print("masking preflight:", exp["name"], sig)
    masking_log[exp["name"]] = {
        **preflight_masking(cfg),
        "config_signature": sig_key,
        "config_digest": digest,
    }
    save_json(masking_log, "preflight_masking", LOG_DIR)

contrastive_log_path = LOG_DIR / "preflight_contrastive.json"
contrastive_log = json.loads(contrastive_log_path.read_text()) if contrastive_log_path.exists() else {}
for exp in EXPERIMENTS:
    cfg = materialize_exp_config(exp)
    if cfg.objective.name != "contrastive":
        continue
    cached = contrastive_log.get(exp["name"])
    digest = config_digest(cfg)
    if isinstance(cached, dict) and cached.get("config_digest") == digest:
        print(f"skip contrastive preflight {exp['name']} (cached)")
        continue
    print("contrastive preflight:", exp["name"])
    contrastive_log[exp["name"]] = {
        **preflight_contrastive(cfg),
        "config_digest": digest,
    }
    save_json(contrastive_log, "preflight_contrastive", LOG_DIR)

print("All pre-flight checks completed.")


# %% [markdown]
# ## Run the matrix
#
# This cell is crash-safe. It writes `run_status.json` after each experiment.
# If a run fails, the loop records the stack trace and keeps moving so already
# completed outputs are not lost. The final manifest cell fails loudly if any
# required run is still missing or failed.

# %% Cell 6 - run every experiment, resumably
RUN_STATUS_PATH = LOG_DIR / "run_status.json"
run_status = json.loads(RUN_STATUS_PATH.read_text()) if RUN_STATUS_PATH.exists() else {}

selected = os.environ.get("PAPER_RUNS")
selected_runs = {x.strip() for x in selected.split(",")} if selected else None

for exp in EXPERIMENTS:
    name = exp["name"]
    if selected_runs and name not in selected_runs:
        continue
    state = result_state(exp)
    if state == "current":
        cfg = materialize_exp_config(exp)
        run_status[name] = {
            "status": "done",
            "path": str(result_path(name)),
            "config_digest": config_digest(cfg),
            "code_digest": PIPELINE_CODE_DIGEST,
            "pipeline_code_version": PIPELINE_CODE_VERSION,
        }
        save_json(run_status, "run_status", LOG_DIR)
        print(f"skip {name} (current results.json exists)")
        continue
    if state.startswith("stale"):
        print(f"rerun {name}: {state}")

    try:
        cfg = materialize_exp_config(exp)
        print(f"\n===== RUN {name} | metrics={cfg.eval.metrics} =====")
        results = ExperimentRunner(cfg).run()
        run_meta = {
            "run": name,
            "path": str(result_path(name)),
            "config_digest": config_digest(cfg),
            "code_digest": PIPELINE_CODE_DIGEST,
            "pipeline_code_version": PIPELINE_CODE_VERSION,
            "code_digest_files": CODE_DIGEST_FILES,
        }
        atomic_write_text(run_meta_path(name), json.dumps(run_meta, indent=2, default=str))
        run_status[name] = {
            "status": "done",
            "path": str(result_path(name)),
            "headline": headline_from_metrics(results.get("final", {})),
            "config_digest": config_digest(cfg),
            "code_digest": PIPELINE_CODE_DIGEST,
            "pipeline_code_version": PIPELINE_CODE_VERSION,
        }
    except Exception as exc:  # noqa: BLE001
        run_status[name] = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        print(f"FAILED {name}; continuing.\n{run_status[name]['traceback']}")
    finally:
        save_json(run_status, "run_status", LOG_DIR)
        free_gpu()

print("Run status:")
for name in EXPECTED_RUNS:
    print(f"  {name:48s} {run_status.get(name, {}).get('status', 'not_started')}")


# %% [markdown]
# ## Load results and write master tables
#
# Only expected run names are loaded. This prevents stale output folders from
# contaminating the paper tables.

# %% Cell 7 - collect expected results only
METRIC_KEYS = [
    "mc_accuracy",
    "rouge_l",
    "bleu",
    "relaxed_accuracy",
    "anls",
    "vqa_accuracy",
    "exact_match",
    "random_baseline",
]


def alpha_of(run_name: str) -> float:
    if "alpha_" in run_name:
        return int(run_name.split("alpha_")[-1]) / 100.0
    return np.nan


rows: list[dict[str, Any]] = []
missing_runs: list[str] = []
stale_runs: dict[str, str] = {}
for exp in EXPERIMENTS:
    name = exp["name"]
    state = result_state(exp)
    if state != "current":
        if state == "missing_results":
            missing_runs.append(name)
        else:
            stale_runs[name] = state
        continue
    cfg = materialize_exp_config(exp)
    path = result_path(name)
    raw = json.loads(path.read_text())
    base = raw.get("baseline", {})
    final = raw.get("final", {})
    row = {
        "run": name,
        "rq": exp["rq"],
        "domain": exp["domain"],
        "dataset": exp["dataset"],
        "backbone": exp["backbone"],
        "objective": exp["objective"],
        "role": exp["role"],
        "alpha": alpha_of(name),
        "configured_alpha": getattr(cfg.objective, "alpha", np.nan),
        "alpha_mode": getattr(cfg.objective, "alpha_mode", "fixed"),
        "answer_weight_multiplier": getattr(cfg.objective, "answer_weight_multiplier", 1.0),
        "train_cap": cfg.data.max_train,
        "eval_cap": cfg.data.max_eval,
        "prompt_family": prompt_family_name(cfg.data.prompt_variant),
        "headline_metric": headline_metric_name(exp["dataset"]),
    }
    for key in METRIC_KEYS:
        out_key = {
            "mc_accuracy": "mc_acc",
            "relaxed_accuracy": "relaxed_acc",
        }.get(key, key)
        row[out_key] = final.get(key)
        row[f"{out_key}_base"] = base.get(key)
    row["headline"] = headline_from_metrics(final)
    row["headline_base"] = headline_from_metrics(base)
    rows.append(row)

df = pd.DataFrame(rows)
if not df.empty:
    df = df.sort_values(["role", "dataset", "alpha", "run"], na_position="last").reset_index(drop=True)
save_csv(df, "all_results")
save_json(missing_runs, "missing_expected_runs", LOG_DIR)
save_json(stale_runs, "stale_expected_runs", LOG_DIR)


ACCURACY_LIKE_HEADLINES = {"MC accuracy", "relaxed accuracy", "VQA accuracy"}


def normal_approx_ci(score: float | None, n: int | float | None, z: float = 1.96) -> tuple[float, float, float]:
    if score is None or pd.isna(score) or n is None or pd.isna(n) or float(n) <= 0:
        return (np.nan, np.nan, np.nan)
    p = min(max(float(score), 0.0), 1.0)
    se = math.sqrt(p * (1.0 - p) / float(n))
    return se, max(0.0, p - z * se), min(1.0, p + z * se)


uncertainty_rows: list[dict[str, Any]] = []
if not df.empty:
    for row in df.to_dict(orient="records"):
        metric = row.get("headline_metric", headline_metric_name(row["dataset"]))
        n_eval = row.get("eval_cap")
        if metric in ACCURACY_LIKE_HEADLINES:
            se, lo, hi = normal_approx_ci(row.get("headline"), n_eval)
            base_se, base_lo, base_hi = normal_approx_ci(row.get("headline_base"), n_eval)
            method_note = "binomial normal approximation from configured evaluation cap"
        else:
            se = lo = hi = base_se = base_lo = base_hi = np.nan
            method_note = "aggregate-only metric; bootstrap needs per-example retained scores"
        uncertainty_rows.append(
            {
                "run": row["run"],
                "dataset": row["dataset"],
                "metric": metric,
                "n_eval_config": n_eval,
                "headline": row.get("headline"),
                "headline_se_approx": se,
                "headline_ci95_low_approx": lo,
                "headline_ci95_high_approx": hi,
                "baseline": row.get("headline_base"),
                "baseline_se_approx": base_se,
                "baseline_ci95_low_approx": base_lo,
                "baseline_ci95_high_approx": base_hi,
                "method_note": method_note,
            }
        )

uncertainty = pd.DataFrame(uncertainty_rows)
save_csv(uncertainty, "uncertainty_estimates")
save_json(uncertainty_rows, "uncertainty_estimates")
if not uncertainty.empty:
    save_table(
        uncertainty[
            [
                "run",
                "metric",
                "n_eval_config",
                "headline",
                "headline_se_approx",
                "headline_ci95_low_approx",
                "headline_ci95_high_approx",
            ]
        ].round(4),
        "uncertainty_estimates",
        "Approximate uncertainty estimates for headline metrics.",
        "tab:uncertainty",
    )
df


# %% Cell 8 - paper tables
if df.empty:
    raise RuntimeError("No current expected results were found. Run Cell 6 first.")

master_cols = [
    "run",
    "rq",
    "domain",
    "backbone",
    "objective",
    "alpha",
    "alpha_mode",
    "answer_weight_multiplier",
    "train_cap",
    "eval_cap",
    "headline_metric",
    "headline",
    "mc_acc",
    "rouge_l",
    "bleu",
    "relaxed_acc",
    "anls",
    "vqa_accuracy",
    "exact_match",
]
master = df[[c for c in master_cols if c in df.columns]].round(4)
save_table(
    master,
    "all_results",
    "All completed expected experiments. Headline uses the native task metric.",
    "tab:all_results",
)

rq2 = df[df["run"].isin(["rq2_blip2_generative_scienceqa", "rq2_blip2_contrastive_scienceqa"])].copy()
if not rq2.empty:
    t = rq2[["objective", "headline_base", "headline", "mc_acc"]].round(4)
    save_table(t, "rq2_blip2", "RQ2 BLIP-2 generative vs contrastive-enhanced alignment.", "tab:rq2_blip2")

sweep = df[df["role"] == "scienceqa_alpha_sweep"].copy()
if not sweep.empty:
    t = sweep[["alpha", "mc_acc", "rouge_l", "bleu"]].sort_values("alpha").round(4)
    t.columns = ["alpha", "MC acc.", "ROUGE-L", "BLEU"]
    save_table(t, "rq3_alpha_sweep", "RQ3 ScienceQA alpha sweep with Qwen2-VL-2B.", "tab:rq3_alpha")

rq3_cmp_runs = [
    "rq3_qwen2vl_answer_only_scienceqa",
    "rq2_qwen2vl_generative_scienceqa",
    "rq3_alpha_050",
    "rq3_qwen2vl_length_aware_scienceqa",
    "rq3_qwen2vl_answer_only_aokvqa",
    "rq2_qwen2vl_generative_aokvqa",
    "rq3_qwen2vl_explanation_aware_aokvqa",
    "rq3_qwen2vl_length_aware_aokvqa",
]
rq3_cmp = df[df["run"].isin(rq3_cmp_runs)].copy()
if not rq3_cmp.empty:
    t = rq3_cmp[["dataset", "objective", "mc_acc", "rouge_l", "bleu"]].round(4)
    save_table(t, "rq3_objective_comparison", "RQ3 generative vs explanation-aware rationale datasets.", "tab:rq3_objective")

rq4_runs = [
    "rq3_alpha_050",
    "rq3_qwen2vl_explanation_aware_aokvqa",
    "rq4_qwen2vl_explanation_aware_chartqa",
    "rq4_qwen2vl_explanation_aware_docvqa",
    "rq4_qwen2vl_explanation_aware_vqav2",
]
cross = df[df["run"].isin(rq4_runs)].copy()
if not cross.empty:
    t = cross[["domain", "dataset", "objective", "headline", "mc_acc", "relaxed_acc", "anls", "vqa_accuracy"]].round(4)
    save_table(t, "rq4_cross_domain", "RQ4 native headline score by visual domain.", "tab:rq4_domain")

scale = df[df["run"].isin(["rq3_alpha_050", "rq6_qwen2vl_7b_explanation_aware_scienceqa"])].copy()
if len(scale) == 2:
    scale["model_size"] = scale["run"].map({
        "rq3_alpha_050": "2B",
        "rq6_qwen2vl_7b_explanation_aware_scienceqa": "7B",
    })
    t = scale[["model_size", "mc_acc", "rouge_l", "bleu"]].round(4)
    save_table(t, "rq6_scale", "RQ6 QLoRA scale comparison on ScienceQA.", "tab:rq6_scale")

comparison_design = pd.DataFrame(
    [
        {
            "comparison": "RQ2 BLIP-2 ScienceQA",
            "zero_shot": "same frozen BLIP-2",
            "control": "BLIP-2 generative fine-tune",
            "candidate": "BLIP-2 generative + contrastive loss",
            "valid_claim": "effect of the contrastive auxiliary objective under matched backbone/data",
        },
        {
            "comparison": "RQ3 Qwen2-VL ScienceQA",
            "zero_shot": "same frozen Qwen2-VL-2B",
            "control": "true answer-only and rationale+answer generative fine-tunes",
            "candidate": "Qwen2-VL explanation-aware alpha sweep",
            "valid_claim": "effect of adding rationale targets and answer/explanation span weighting",
        },
        {
            "comparison": "RQ3 Qwen2-VL A-OKVQA",
            "zero_shot": "same frozen Qwen2-VL-2B",
            "control": "true answer-only and rationale+answer generative fine-tunes",
            "candidate": "Qwen2-VL explanation-aware alpha=0.50",
            "valid_claim": "second rationale-bearing domain check for explanation-aware training",
        },
        {
            "comparison": "RQ4/RQ7 ChartQA DocVQA VQAv2",
            "zero_shot": "per-domain frozen Qwen2-VL-2B",
            "control": "answer-only fine-tuned adapter",
            "candidate": "no rationale-supervised candidate; no gold rationales",
            "valid_claim": "domain behavior and transfer only, not explanation-aware improvement",
        },
        {
            "comparison": "RQ6 Qwen2-VL scale",
            "zero_shot": "2B/7B frozen baselines",
            "control": "Qwen2-VL-2B explanation-aware alpha=0.50",
            "candidate": "Qwen2-VL-7B explanation-aware alpha=0.50",
            "valid_claim": "scale behavior under the same QLoRA/objective/data protocol",
        },
    ]
)
save_csv(comparison_design, "comparison_design")
save_table(
    comparison_design,
    "comparison_design",
    "Controlled comparisons and claim boundaries used in the paper.",
    "tab:comparison_design",
)


def controlled_block(block: str, run_names: list[str]) -> list[dict[str, Any]]:
    by_run = df.set_index("run")
    available = [run_name for run_name in run_names if run_name in by_run.index]
    if not available:
        return []
    first = by_run.loc[available[0]]
    rows_out = [
        {
            "comparison_group": block,
            "method": "Zero-shot",
            "score": first["headline_base"],
            "metric": headline_metric_name(first["dataset"]),
            "run": "baseline",
        }
    ]
    for run_name in available:
        row = by_run.loc[run_name]
        rows_out.append(
            {
                "comparison_group": block,
                "method": pretty_objective(row["objective"]),
                "score": row["headline"],
                "metric": headline_metric_name(row["dataset"]),
                "run": run_name,
            }
        )
    return rows_out


controlled_rows = []
controlled_rows.extend(
    controlled_block(
        "BLIP-2 / ScienceQA",
        ["rq2_blip2_generative_scienceqa", "rq2_blip2_contrastive_scienceqa"],
    )
)
controlled_rows.extend(
    controlled_block(
        "Qwen2-VL / ScienceQA",
        [
            "rq3_qwen2vl_answer_only_scienceqa",
            "rq2_qwen2vl_generative_scienceqa",
            "rq3_alpha_050",
            "rq3_qwen2vl_length_aware_scienceqa",
        ],
    )
)
controlled_rows.extend(
    controlled_block(
        "Qwen2-VL / A-OKVQA",
        [
            "rq3_qwen2vl_answer_only_aokvqa",
            "rq2_qwen2vl_generative_aokvqa",
            "rq3_qwen2vl_explanation_aware_aokvqa",
            "rq3_qwen2vl_length_aware_aokvqa",
        ],
    )
)
controlled = pd.DataFrame(controlled_rows)
if not controlled.empty:
    save_csv(controlled, "controlled_strategy_comparison")
    save_table(
        controlled.round(4),
        "controlled_strategy_comparison",
        "Controlled zero-shot, generative, and aligned strategy comparisons.",
        "tab:controlled_strategy",
    )


# %% Cell 9 - paper figures from standard results
if not controlled.empty:
    plot = controlled.dropna(subset=["score"]).copy()
    groups = list(dict.fromkeys(plot["comparison_group"]))
    fig, axes = plt.subplots(1, len(groups), figsize=(11.8, 3.9), sharey=True)
    axes = np.array(axes).reshape(-1)
    palette = {
        "Zero-shot": "#F8FAFC",
        "Generative": "#94A3B8",
        "Answer-only": "#CBD5E1",
        "Rationale+answer CE": "#64748B",
        "Generative + contrastive": "#0F766E",
        "Expl.-aware alpha=0.50": "#0F766E",
        "Expl.-aware length-aware": "#7C3AED",
    }
    for ax, group in zip(axes, groups):
        sub = plot[plot["comparison_group"] == group].reset_index(drop=True)
        colors = [palette.get(m, "#64748B") for m in sub["method"]]
        ax.bar(np.arange(len(sub)), sub["score"], color=colors, edgecolor="#111827", linewidth=0.8)
        ax.set_title(group, fontsize=10)
        ax.set_xticks(np.arange(len(sub)))
        ax.set_xticklabels(sub["method"], rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.2)
        ax.set_xlabel(sub["metric"].iloc[0])
    axes[0].set_ylabel("score")
    fig.suptitle("Controlled strategy comparisons", fontsize=12)
    save_fig(fig, "controlled_strategy_comparison", data=plot)
    plt.show()

if not sweep.empty:
    s = sweep.sort_values("alpha")
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    for col, lab, mk, ls in [
        ("mc_acc", "MC accuracy", "o", "-"),
        ("rouge_l", "ROUGE-L", "s", "--"),
        ("bleu", "BLEU", "^", ":"),
    ]:
        sub = s.dropna(subset=[col])
        if not sub.empty:
            ax.plot(sub["alpha"], sub[col], marker=mk, linestyle=ls, color="black", label=lab)
    ax.set_xlabel("alpha (answer weight; 1-alpha on explanation)")
    ax.set_ylabel("score")
    ax.set_title("RQ3: explanation-aware alpha sweep")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    save_fig(fig, "rq3_alpha_sweep", data=s[["alpha", "mc_acc", "rouge_l", "bleu"]])
    plt.show()

if not rq2.empty:
    plot = rq2[["objective", "headline_base", "headline"]].copy()
    plot["method"] = plot["objective"].map(pretty_objective)
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    x = np.arange(len(plot))
    ax.bar(x - 0.18, plot["headline_base"], 0.36, label="zero-shot", color="#F8FAFC", edgecolor="#111827")
    ax.bar(x + 0.18, plot["headline"], 0.36, label="fine-tuned", color="#64748B", edgecolor="#111827")
    ax.set_xticks(x)
    ax.set_xticklabels(plot["method"], rotation=10, ha="right")
    ax.set_ylabel("ScienceQA MC accuracy")
    ax.set_title("RQ2: BLIP-2 objective comparison")
    ax.legend(frameon=False)
    save_fig(fig, "rq2_blip2_objective", data=plot)
    plt.show()

if not rq3_cmp.empty:
    plot = rq3_cmp[["dataset", "objective", "mc_acc", "rouge_l"]].copy()
    plot["label"] = plot.apply(lambda r: f"{pretty_dataset(r['dataset'])}\n{pretty_objective(r['objective'])}", axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    axes[0].bar(plot["label"], plot["mc_acc"], color="#64748B", edgecolor="#111827")
    axes[0].set_title("Answer accuracy")
    axes[0].set_ylabel("MC accuracy")
    axes[1].bar(plot["label"], plot["rouge_l"], color="#CBD5E1", edgecolor="#111827")
    axes[1].set_title("Rationale overlap")
    axes[1].set_ylabel("ROUGE-L")
    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=18, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("RQ3: generative vs explanation-aware training")
    save_fig(fig, "rq3_objective_comparison", data=plot)
    plt.show()

piv = df.dropna(subset=["headline", "headline_base"]).copy()
if not piv.empty:
    plot = piv[["run", "dataset", "objective", "headline_base", "headline"]].copy()
    plot["label"] = plot.apply(pretty_run_label, axis=1)
    fig, ax = plt.subplots(figsize=(7.8, 0.38 * len(plot) + 1.8))
    y = np.arange(len(plot))
    ax.barh(y - 0.18, plot["headline_base"], 0.36, label="zero-shot", color="#F8FAFC", edgecolor="#111827")
    ax.barh(y + 0.18, plot["headline"], 0.36, label="fine-tuned", color="#64748B", edgecolor="#111827")
    ax.set_yticks(y)
    ax.set_yticklabels(plot["label"], fontsize=7)
    ax.set_xlabel("native headline score")
    ax.set_title("Fine-tuning lift by run")
    ax.legend(frameon=False)
    save_fig(fig, "headline_lift", data=plot)
    plt.show()

if not cross.empty:
    plot = cross[["domain", "dataset", "objective", "headline_base", "headline"]].dropna(subset=["headline"]).copy()
    plot["label"] = plot.apply(lambda r: f"{pretty_dataset(r['dataset'])}\n{pretty_objective(r['objective'])}", axis=1)
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    x = np.arange(len(plot))
    ax.bar(x - 0.18, plot["headline_base"], 0.36, label="zero-shot", color="#F8FAFC", edgecolor="#111827")
    ax.bar(x + 0.18, plot["headline"], 0.36, label="fine-tuned", color="#64748B", edgecolor="#111827")
    ax.set_xticks(x)
    ax.set_xticklabels(plot["label"])
    ax.set_ylabel("native headline score")
    ax.set_title("RQ4: cross-domain performance (native metric)")
    ax.legend(frameon=False, fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.2)
    save_fig(fig, "rq4_cross_domain", data=plot)
    plt.show()

if len(scale) == 2:
    plot = scale[["model_size", "mc_acc", "rouge_l", "bleu"]].copy()
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    ax.bar(plot["model_size"], plot["mc_acc"], color=["#F8FAFC", "#64748B"], edgecolor="#111827")
    ax.set_ylabel("ScienceQA MC accuracy")
    ax.set_title("RQ6: Qwen2-VL QLoRA scale")
    ax.grid(axis="y", alpha=0.2)
    save_fig(fig, "rq6_scale", data=plot)
    plt.show()


# %% [markdown]
# ## RQ2 contrastive retrieval
#
# The answer metric tests whether contrastive-enhanced BLIP-2 answers better.
# This stage additionally tests whether the learned image-answer representation
# is better aligned using image-to-answer retrieval R@K/MRR.

# %% Cell 10 - BLIP-2 retrieval R@K/MRR with controls
def blip_retrieval_eval(
    run_name: str,
    *,
    checkpoint: bool,
    n_eval: int = 200,
    batch_size: int = 16,
) -> dict[str, float]:
    if checkpoint:
        cfg, wrapper, _, eval_ds = load_run(run_name)
    else:
        cfg = materialize_exp_config(RUN_BY_NAME[run_name])
        wrapper = build_model(cfg.model)
        eval_ds = build_dataset(cfg.data, split=cfg.data.split_eval)
        wrapper.eval()
    try:
        collator = build_collator(cfg.data, wrapper, tag_spans=True)
        image_chunks = []
        text_chunks = []
        n = min(n_eval, len(eval_ds))
        for start in range(0, n, batch_size):
            examples = [eval_ds[i] for i in range(start, min(start + batch_size, n))]
            batch = collator(examples)
            batch = move_to_device(batch, wrapper.device, wrapper.dtype)
            image_embeds, text_embeds = wrapper.contrastive_features(batch)
            image_chunks.append(image_embeds.detach().float().cpu())
            text_chunks.append(text_embeds.detach().float().cpu())
        image = torch.cat(image_chunks, dim=0)
        text = torch.cat(text_chunks, dim=0)
        out = retrieval_recall(image, text, ks=(1, 5, 10))
        out["n_eval"] = int(n)
        return out
    finally:
        del wrapper
        free_gpu()


def random_retrieval_reference(n: int, ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    n = max(int(n), 1)
    out = {f"R@{k}": min(int(k), n) / n for k in ks}
    out["MRR"] = float(sum(1.0 / r for r in range(1, n + 1)) / n)
    out["n_eval"] = n
    return out


RET_PATH = ART_DIR / "rq2_retrieval_comparison.json"
retrieval_scores = json.loads(RET_PATH.read_text()) if RET_PATH.exists() else {}
retrieval_n = int(os.environ.get("PAPER_RETRIEVAL_N", "200"))
retrieval_batch = int(os.environ.get("PAPER_RETRIEVAL_BATCH", "16"))
retrieval_specs = {
    "frozen_blip2": {
        "display": "Frozen BLIP-2",
        "run": "rq2_blip2_generative_scienceqa",
        "checkpoint": False,
    },
    "blip2_generative": {
        "display": "BLIP-2 generative",
        "run": "rq2_blip2_generative_scienceqa",
        "checkpoint": True,
    },
    "blip2_contrastive": {
        "display": "BLIP-2 contrastive-enhanced",
        "run": "rq2_blip2_contrastive_scienceqa",
        "checkpoint": True,
    },
}

for key, spec_row in retrieval_specs.items():
    run_name = spec_row["run"]
    if spec_row["checkpoint"] and not result_ready(run_name):
        retrieval_scores[f"{key}__status"] = result_state(RUN_BY_NAME[run_name])
        save_json(retrieval_scores, "rq2_retrieval_comparison")
        continue
    digest_payload = {
        "run_digest": config_digest(materialize_exp_config(RUN_BY_NAME[run_name])),
        "checkpoint": spec_row["checkpoint"],
        "n_eval": retrieval_n,
    }
    digest = hashlib.sha256(json.dumps(digest_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    cached = retrieval_scores.get(key)
    if isinstance(cached, dict) and cached.get("diagnostic_digest") == digest:
        continue
    try:
        retrieval_scores[key] = {
            **blip_retrieval_eval(
                run_name,
                checkpoint=bool(spec_row["checkpoint"]),
                n_eval=retrieval_n,
                batch_size=retrieval_batch,
            ),
            "display": spec_row["display"],
            "source_run": run_name,
            "diagnostic_digest": digest,
        }
        save_json(retrieval_scores, "rq2_retrieval_comparison")
    except Exception as exc:  # noqa: BLE001
        retrieval_scores[f"{key}__error"] = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        save_json(retrieval_scores, "rq2_retrieval_comparison")

if any(isinstance(v, dict) and "n_eval" in v for v in retrieval_scores.values()):
    n_for_random = max(
        [int(v["n_eval"]) for v in retrieval_scores.values() if isinstance(v, dict) and "n_eval" in v]
        or [retrieval_n]
    )
    retrieval_scores["random_rank"] = {
        **random_retrieval_reference(n_for_random),
        "display": "Random rank",
        "source_run": "random",
        "diagnostic_digest": f"random-{n_for_random}",
    }
    save_json(retrieval_scores, "rq2_retrieval_comparison")
    save_json(retrieval_scores, "rq2_contrastive_retrieval")

ret_rows = []
for key, value in retrieval_scores.items():
    if not isinstance(value, dict) or "n_eval" not in value:
        continue
    for metric in ("R@1", "R@5", "R@10", "MRR"):
        if metric in value:
            ret_rows.append(
                {
                    "method": value.get("display", key),
                    "metric": metric,
                    "score": value[metric],
                    "n_eval": value.get("n_eval"),
                    "source_run": value.get("source_run"),
                }
            )

if ret_rows:
    ret = pd.DataFrame(ret_rows)
    save_table(
        ret.pivot(index="method", columns="metric", values="score").reset_index().round(4),
        "rq2_retrieval_comparison",
        "BLIP-2 image-to-answer retrieval diagnostic with frozen, generative, contrastive, and random controls.",
        "tab:rq2_retrieval",
    )
    # Backward-compatible artifact names used by earlier paper drafts.
    save_table(
        ret.pivot(index="method", columns="metric", values="score").reset_index().round(4),
        "rq2_contrastive_retrieval",
        "BLIP-2 image-to-answer retrieval diagnostic with frozen, generative, contrastive, and random controls.",
        "tab:rq2_retrieval_old",
    )
    metrics = ["R@1", "R@5", "R@10", "MRR"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    pivot = ret.pivot(index="method", columns="metric", values="score").reindex(columns=metrics)
    x = np.arange(len(pivot.index))
    width = 0.18
    colors = ["#E5E7EB", "#CBD5E1", "#0F766E", "#111827"]
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, pivot[metric], width, label=metric, color=colors[i], edgecolor="#111827", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("retrieval score")
    ax.set_title("BLIP-2 image-to-answer retrieval diagnostic")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=4, fontsize=8)
    save_fig(fig, "rq2_retrieval_comparison", data=ret)
    save_fig(fig, "rq2_contrastive_retrieval", data=ret)
    plt.show()


# %% [markdown]
# ## RQ5 faithfulness
#
# Mean masking drift is a causal evidence-sensitivity diagnostic. It does not
# prove rationale faithfulness alone, but it tells us whether the gold answer
# likelihood drops when image evidence is occluded.

# %% Cell 11 - evidence-masking drift scores
def bootstrap_ci(values: list[float], n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    arr = np.array([v for v in values if not pd.isna(v)], dtype=float)
    if len(arr) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def faithfulness_distribution(wrapper, eval_ds, template, n: int = 30, grid: tuple[int, int] = (3, 3)) -> list[float]:
    drifts = []
    for i in range(min(n, len(eval_ds))):
        drifts.append(
            region_importance(
                wrapper,
                eval_ds[i],
                template,
                grid=grid,
                max_length=getattr(wrapper, "_paper_eval_max_length", 1024),
            ).mean_drift
        )
    return [float(x) for x in drifts]


FAITH_PATH = ART_DIR / "faithfulness_scores.json"
faith = json.loads(FAITH_PATH.read_text()) if FAITH_PATH.exists() else {}

faith_runs = [
    exp["name"]
    for exp in EXPERIMENTS
    if exp["backbone"].startswith("Qwen2-VL") and result_ready(exp["name"])
]
for name in faith_runs:
    digest = config_digest(materialize_exp_config(RUN_BY_NAME[name]))
    cached = faith.get(name)
    if isinstance(cached, dict) and cached.get("config_digest") == digest:
        continue
    try:
        cfg, wrapper, template, eval_ds = load_run(name)
        wrapper._paper_eval_max_length = cfg.eval.max_length
        drift_values = faithfulness_distribution(
            wrapper,
            eval_ds,
            template,
            n=int(os.environ.get("PAPER_FAITH_N", "30")),
            grid=(3, 3),
        )
        ci_low, ci_high = bootstrap_ci(drift_values)
        faith[name] = {
            "mean_drift": float(np.mean(drift_values)) if drift_values else float("nan"),
            "bootstrap_ci95_low": ci_low,
            "bootstrap_ci95_high": ci_high,
            "drift_values": drift_values,
            "n_eval": len(drift_values),
            "grid": [3, 3],
            "config_digest": digest,
        }
        save_json(faith, "faithfulness_scores")
        del wrapper
    except Exception as exc:  # noqa: BLE001
        faith[f"{name}__error"] = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        save_json(faith, "faithfulness_scores")
    finally:
        free_gpu()

faith_rows = [
    {
        "run": k,
        "mean_drift": v["mean_drift"],
        "ci95_low": v.get("bootstrap_ci95_low"),
        "ci95_high": v.get("bootstrap_ci95_high"),
        "n_eval": v["n_eval"],
    }
    for k, v in faith.items()
    if isinstance(v, dict) and "mean_drift" in v
]
if faith_rows:
    fs = pd.DataFrame(faith_rows).sort_values("run").reset_index(drop=True)
    save_table(fs.round(4), "rq5_faithfulness", "RQ5 evidence-masking drift by model.", "tab:rq5_faithfulness")
    fig, ax = plt.subplots(figsize=(7.8, 0.42 * len(fs) + 1.8))
    ax.barh(fs["run"], fs["mean_drift"], color="dimgray", edgecolor="black")
    if {"ci95_low", "ci95_high"}.issubset(fs.columns):
        xerr = np.vstack(
            [
                (fs["mean_drift"] - fs["ci95_low"]).clip(lower=0).fillna(0).to_numpy(),
                (fs["ci95_high"] - fs["mean_drift"]).clip(lower=0).fillna(0).to_numpy(),
            ]
        )
        ax.errorbar(fs["mean_drift"], fs["run"], xerr=xerr, fmt="none", ecolor="#111827", capsize=2, linewidth=0.8)
    ax.set_xlabel("mean masking drift")
    ax.set_title("RQ5: visual evidence sensitivity")
    plt.setp(ax.get_yticklabels(), fontsize=7)
    ax.grid(axis="x", alpha=0.2)
    save_fig(fig, "rq5_faithfulness", data=fs)
    plt.show()


def image_preprocess_diff(wrapper, img_a, img_b, prompt: str, max_length: int) -> float:
    enc_a = wrapper.build_inputs([img_a], [prompt], padding=False, truncation=True, max_length=max_length)
    enc_b = wrapper.build_inputs([img_b], [prompt], padding=False, truncation=True, max_length=max_length)
    diffs: list[float] = []
    for key in sorted(set(enc_a) & set(enc_b)):
        va, vb = enc_a[key], enc_b[key]
        if torch.is_tensor(va) and torch.is_tensor(vb) and va.shape == vb.shape and va.is_floating_point():
            diffs.append(float((va.float() - vb.float()).abs().sum().item()))
    return float(sum(diffs))


def blank_image_like(img):
    return img.copy().point(lambda _: 128)


def save_masking_example(run_name: str, idx: int = 0, grid: tuple[int, int] = (3, 3)) -> dict[str, Any]:
    cfg, wrapper, template, eval_ds = load_run(run_name)
    try:
        if idx >= len(eval_ds):
            idx = 0
        ex = eval_ds[idx]
        image = ex.image.convert("RGB")
        context = f"{template.prompt(ex)} Answer:"
        res = region_importance(wrapper, ex, template, grid=grid, max_length=cfg.eval.max_length)
        top_idx = int(np.argmax(res.drifts))
        rows, cols = grid
        top_r, top_c = divmod(top_idx, cols)
        random_idx = (top_idx + max(1, (rows * cols) // 2)) % (rows * cols)
        random_r, random_c = divmod(random_idx, cols)

        top_mask = mask_region(image, top_r, top_c, rows, cols)
        random_mask = mask_region(image, random_r, random_c, rows, cols)
        blank = blank_image_like(image)

        full_score = score_continuation(wrapper, image, context, ex.answer, max_length=cfg.eval.max_length)
        top_score = score_continuation(wrapper, top_mask, context, ex.answer, max_length=cfg.eval.max_length)
        random_score = score_continuation(wrapper, random_mask, context, ex.answer, max_length=cfg.eval.max_length)
        blank_score = score_continuation(wrapper, blank, context, ex.answer, max_length=cfg.eval.max_length)

        top_tensor_diff = image_preprocess_diff(wrapper, image, top_mask, template.prompt(ex), cfg.eval.max_length)
        random_tensor_diff = image_preprocess_diff(wrapper, image, random_mask, template.prompt(ex), cfg.eval.max_length)
        if top_tensor_diff <= 0:
            raise AssertionError(f"{run_name}: top masked image did not alter preprocessed image tensor")
        if random_tensor_diff <= 0:
            raise AssertionError(f"{run_name}: random masked image did not alter preprocessed image tensor")

        fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.8))
        panels = [
            ("original", image, full_score, 0.0),
            (f"top-drift cell ({top_r},{top_c})", top_mask, top_score, full_score - top_score),
            (f"random cell ({random_r},{random_c})", random_mask, random_score, full_score - random_score),
            ("blank image", blank, blank_score, full_score - blank_score),
        ]
        for ax, (title, panel_img, score, drift) in zip(axes, panels):
            ax.imshow(panel_img)
            ax.axis("off")
            ax.set_title(f"{title}\nscore={score:.3f}, drift={drift:.3f}", fontsize=8)
        fig.suptitle(f"Masking example - {run_name}\nQ: {short(ex.question, 150)} | Gold: {short(ex.answer, 80)}", fontsize=10)
        save_fig(fig, f"masking_example_{run_name}")
        plt.show()

        return {
            "run": run_name,
            "idx": idx,
            "dataset": cfg.data.name,
            "question": ex.question,
            "gold_answer": ex.answer,
            "grid": list(grid),
            "full_score": full_score,
            "top_region": [top_r, top_c],
            "top_score": top_score,
            "top_drift": full_score - top_score,
            "random_region": [random_r, random_c],
            "random_score": random_score,
            "random_drift": full_score - random_score,
            "blank_score": blank_score,
            "blank_drift": full_score - blank_score,
            "mean_drift": res.mean_drift,
            "max_drift": res.max_drift,
            "top_tensor_diff_l1": top_tensor_diff,
            "random_tensor_diff_l1": random_tensor_diff,
            "interpretation": "masking drift measures evidence sensitivity, not direct proof of rationale faithfulness",
        }
    finally:
        del wrapper
        free_gpu()


masking_example_runs = [
    r.strip()
    for r in os.environ.get(
        "PAPER_MASKING_EXAMPLE_RUNS",
        "rq4_qwen2vl_explanation_aware_chartqa,rq4_qwen2vl_explanation_aware_docvqa,rq3_alpha_050",
    ).split(",")
    if r.strip()
]
masking_example_log_path = ART_DIR / "masking_examples.json"
masking_examples = json.loads(masking_example_log_path.read_text()) if masking_example_log_path.exists() else {}
for run_name in masking_example_runs:
    exp = RUN_BY_NAME.get(run_name)
    if exp is None:
        masking_examples[run_name] = {"status": "unknown_run"}
        save_json(masking_examples, "masking_examples")
        continue
    if not result_ready(run_name):
        masking_examples[run_name] = {"status": result_state(exp)}
        save_json(masking_examples, "masking_examples")
        continue
    digest = config_digest(materialize_exp_config(exp))
    cached = masking_examples.get(run_name)
    if isinstance(cached, dict) and cached.get("status") == "done" and cached.get("config_digest") == digest:
        continue
    try:
        masking_examples[run_name] = {
            "status": "done",
            "config_digest": digest,
            **save_masking_example(run_name, idx=int(os.environ.get("PAPER_MASKING_EXAMPLE_IDX", "0"))),
        }
    except Exception as exc:  # noqa: BLE001
        masking_examples[run_name] = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    finally:
        save_json(masking_examples, "masking_examples")

masking_example_rows = [
    v for v in masking_examples.values() if isinstance(v, dict) and v.get("status") == "done"
]
if masking_example_rows:
    masking_example_df = pd.DataFrame(masking_example_rows)
    save_csv(masking_example_df, "masking_examples")
    save_table(
        masking_example_df[
            [
                "run",
                "dataset",
                "full_score",
                "top_drift",
                "random_drift",
                "blank_drift",
                "top_tensor_diff_l1",
            ]
        ].round(4),
        "masking_examples",
        "Masking example controls and preprocessing validation.",
        "tab:masking_examples",
    )


# %% [markdown]
# ## Qualitative samples and faithfulness heatmaps
#
# These are optional for the final journal if page limits are tight, but they are
# saved because they are useful for interpreting failures and writing discussion.

# %% Cell 12 - qualitative sample grids and heatmaps
def save_qualitative_views(run_name: str, n: int = 6) -> None:
    cfg, wrapper, template, eval_ds = load_run(run_name)
    try:
        ev = Evaluator(wrapper, eval_ds, template, cfg.eval, build_metrics(cfg.eval.metrics))
        count = min(n, len(eval_ds))
        cols = 3
        rows = int(np.ceil(count / cols))

        sample_lines = [f"# Qualitative samples - {run_name}\n"]
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4.8 * rows))
        axes = np.array(axes).reshape(-1)
        for i, ax in enumerate(axes):
            ax.axis("off")
            if i >= count:
                continue
            ex = eval_ds[i]
            pred = ev._predict_one(ex)
            hit = ""
            if ex.is_multiple_choice:
                hit = "hit" if pred.predicted_index == ex.answer_index else "miss"
            if ex.image is not None:
                ax.imshow(ex.image.convert("RGB"))
            ax.set_title(
                "\n".join(
                    [
                        f"[{i}] {hit}".strip(),
                        f"Gold: {short(ex.answer, 54)}",
                        f"Pred: {short(pred.predicted_text, 72)}",
                        f"Q: {short(ex.question, 92)}",
                    ]
                ),
                fontsize=8,
            )
            sample_lines.append(
                f"## [{i}] {hit}\n"
                f"- question: {short(ex.question, 500)}\n"
                f"- gold: {short(ex.answer, 200)}\n"
                f"- prediction: {short(pred.predicted_text, 400)}\n"
                f"- reasoning: {short(pred.reasoning, 700)}\n"
            )
        fig.suptitle(f"Qualitative samples - {run_name}", fontsize=12)
        save_fig(fig, f"samples_{run_name}")
        save_markdown("\n".join(sample_lines), f"samples_{run_name}")
        plt.show()

        fig, axes = plt.subplots(rows, cols, figsize=(15, 4.8 * rows))
        axes = np.array(axes).reshape(-1)
        heat_rows = []
        for i, ax in enumerate(axes):
            ax.axis("off")
            if i >= count:
                continue
            ex = eval_ds[i]
            res = region_importance(
                wrapper, ex, template, grid=(4, 4), max_length=cfg.eval.max_length
            )
            heat = np.array(res.drifts).reshape(res.grid)
            ax.imshow(ex.image.convert("RGB"))
            ax.imshow(
                heat,
                cmap="hot",
                alpha=0.5,
                extent=[0, ex.image.width, ex.image.height, 0],
                interpolation="bilinear",
            )
            ax.set_title(f"[{i}] max drift={res.max_drift:.3f}\n{short(ex.question, 92)}", fontsize=8)
            heat_rows.append({"idx": i, "max_drift": res.max_drift, "mean_drift": res.mean_drift})
        fig.suptitle(f"Evidence-masking heatmaps - {run_name}", fontsize=12)
        save_fig(fig, f"faithfulness_heatmaps_{run_name}", data=pd.DataFrame(heat_rows))
        plt.show()
    finally:
        del wrapper
        free_gpu()


INSPECT_RUNS = [
    r.strip()
    for r in os.environ.get(
        "PAPER_INSPECT_RUNS",
        "rq3_alpha_050,rq3_qwen2vl_explanation_aware_aokvqa,rq4_qwen2vl_explanation_aware_chartqa,rq4_qwen2vl_explanation_aware_docvqa",
    ).split(",")
    if r.strip()
]

qual_status_path = LOG_DIR / "qualitative_status.json"
qual_status = json.loads(qual_status_path.read_text()) if qual_status_path.exists() else {}
for run_name in INSPECT_RUNS:
    exp = RUN_BY_NAME.get(run_name)
    if exp is None:
        qual_status[run_name] = {"status": "unknown_run"}
        save_json(qual_status, "qualitative_status", LOG_DIR)
        continue
    digest = config_digest(materialize_exp_config(exp))
    cached = qual_status.get(run_name)
    if isinstance(cached, dict) and cached.get("status") == "done" and cached.get("config_digest") == digest:
        continue
    if not result_ready(run_name):
        qual_status[run_name] = {"status": result_state(exp)}
        save_json(qual_status, "qualitative_status", LOG_DIR)
        continue
    try:
        save_qualitative_views(run_name, n=int(os.environ.get("PAPER_QUAL_N", "6")))
        qual_status[run_name] = {"status": "done", "config_digest": digest}
    except Exception as exc:  # noqa: BLE001
        qual_status[run_name] = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    finally:
        save_json(qual_status, "qualitative_status", LOG_DIR)


# %% [markdown]
# ## RQ7 transfer
#
# This evaluates each Qwen2-VL-2B source adapter on each target domain without
# retraining. For rationale-bearing source domains, answer-only, rationale-CE,
# and explanation-aware adapters are transferred separately. Evaluation uses the
# source adapter's prompt family so answer-only models are not forced through a
# rationale-generation prompt at test time.

# %% Cell 13 - full transfer matrix
TRANSFER_SOURCES = {
    "science_answer_only": {
        "run": "rq3_qwen2vl_answer_only_scienceqa",
        "prompt_variant": "answer_only",
        "cot": False,
    },
    "science_rationale_ce": {
        "run": "rq2_qwen2vl_generative_scienceqa",
        "prompt_variant": "explanation_then_answer",
        "cot": True,
    },
    "science_expl_aware": {
        "run": "rq3_alpha_050",
        "prompt_variant": "explanation_then_answer",
        "cot": True,
    },
    "aokvqa_answer_only": {
        "run": "rq3_qwen2vl_answer_only_aokvqa",
        "prompt_variant": "answer_only",
        "cot": False,
    },
    "aokvqa_rationale_ce": {
        "run": "rq2_qwen2vl_generative_aokvqa",
        "prompt_variant": "explanation_then_answer",
        "cot": True,
    },
    "aokvqa_expl_aware": {
        "run": "rq3_qwen2vl_explanation_aware_aokvqa",
        "prompt_variant": "explanation_then_answer",
        "cot": True,
    },
    "charts_answer_only": {
        "run": "rq4_qwen2vl_explanation_aware_chartqa",
        "prompt_variant": "answer_only",
        "cot": False,
    },
    "documents_answer_only": {
        "run": "rq4_qwen2vl_explanation_aware_docvqa",
        "prompt_variant": "answer_only",
        "cot": False,
    },
    "vqav2_answer_only": {
        "run": "rq4_qwen2vl_explanation_aware_vqav2",
        "prompt_variant": "answer_only",
        "cot": False,
    },
}

TRANSFER_TARGETS = {
    "science": f"{BASE}/rq3_qwen2vl_explanation_aware_scienceqa.yaml",
    "aokvqa": f"{BASE}/rq3_qwen2vl_explanation_aware_aokvqa.yaml",
    "charts": f"{BASE}/rq4_qwen2vl_explanation_aware_chartqa.yaml",
    "documents": f"{BASE}/rq4_qwen2vl_explanation_aware_docvqa.yaml",
    "vqav2": f"{BASE}/rq4_qwen2vl_explanation_aware_vqav2.yaml",
}


def transfer_target_config(target_cfg_path: str, source_spec: dict[str, Any], n_eval: int):
    cfg_t = load_config(target_cfg_path)
    cfg_t.data.max_eval = n_eval
    cfg_t.data.prompt_variant = source_spec["prompt_variant"]
    cfg_t.eval.cot = bool(source_spec["cot"])
    sync_eval_length(cfg_t)
    return cfg_t


def transfer_eval_loaded(wrapper, target_cfg_path: str, source_spec: dict[str, Any], n_eval: int = 200) -> dict[str, float]:
    cfg_t = transfer_target_config(target_cfg_path, source_spec, n_eval)
    target_template = build_template(cfg_t.data.prompt_variant)
    ds_t = build_dataset(cfg_t.data, split=cfg_t.data.split_eval)
    ev = Evaluator(wrapper, ds_t, target_template, cfg_t.eval, build_metrics(cfg_t.eval.metrics))
    return ev.evaluate()


def transfer_target_digest(target_cfg_path: str, source_spec: dict[str, Any], n_eval: int) -> str:
    return config_digest(transfer_target_config(target_cfg_path, source_spec, n_eval))


def transfer_entry_current(entry: Any, source_digest: str, target_digest: str, n_eval: int) -> bool:
    if not isinstance(entry, dict):
        return False
    if not isinstance(entry.get("metrics"), dict):
        return False
    meta = entry.get("_meta")
    if not isinstance(meta, dict):
        return False
    return (
        meta.get("source_config_digest") == source_digest
        and meta.get("target_config_digest") == target_digest
        and int(meta.get("n_eval", -1)) == int(n_eval)
    )


TX_PATH = ART_DIR / "transfer_matrix.json"
tx = json.loads(TX_PATH.read_text()) if TX_PATH.exists() else {}
transfer_n = int(os.environ.get("PAPER_TRANSFER_N", "200"))

for source_label, source_spec in TRANSFER_SOURCES.items():
    source_run = source_spec["run"]
    source_exp = RUN_BY_NAME[source_run]
    if not result_ready(source_run):
        print(f"skip transfer source {source_label}: {result_state(source_exp)}")
        continue
    source_digest = config_digest(materialize_exp_config(source_exp))
    pending_targets = [
        (target_label, target_cfg)
        for target_label, target_cfg in TRANSFER_TARGETS.items()
        if not transfer_entry_current(
            tx.get(f"{source_label}->{target_label}"),
            source_digest,
            transfer_target_digest(target_cfg, source_spec, transfer_n),
            transfer_n,
        )
    ]
    if not pending_targets:
        continue
    _, source_wrapper, _, _ = load_run(source_run)
    for target_label, target_cfg in pending_targets:
        label = f"{source_label}->{target_label}"
        target_digest = transfer_target_digest(target_cfg, source_spec, transfer_n)
        try:
            tx[label] = {
                "metrics": transfer_eval_loaded(source_wrapper, target_cfg, source_spec, n_eval=transfer_n),
                "_meta": {
                    "source_run": source_run,
                    "source_prompt_variant": source_spec["prompt_variant"],
                    "source_eval_cot": bool(source_spec["cot"]),
                    "source_config_digest": source_digest,
                    "target_config_digest": target_digest,
                    "n_eval": transfer_n,
                },
            }
            save_json(tx, "transfer_matrix")
        except Exception as exc:  # noqa: BLE001
            tx[f"{label}__error"] = {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            save_json(tx, "transfer_matrix")
    del source_wrapper
    free_gpu()

transfer_rows = []
for label, entry in tx.items():
    if label.endswith("__error") or not isinstance(entry, dict):
        continue
    if "->" not in label:
        continue
    src, tgt = label.split("->")
    if src not in TRANSFER_SOURCES or tgt not in TRANSFER_TARGETS:
        continue
    source_spec = TRANSFER_SOURCES[src]
    source_run = source_spec["run"]
    if not result_ready(source_run):
        continue
    source_digest = config_digest(materialize_exp_config(RUN_BY_NAME[source_run]))
    target_digest = transfer_target_digest(TRANSFER_TARGETS[tgt], source_spec, transfer_n)
    if not transfer_entry_current(entry, source_digest, target_digest, transfer_n):
        continue
    metrics = entry["metrics"]
    metrics = {k: v for k, v in metrics.items() if not str(k).startswith("_")}
    transfer_rows.append(
        {
            "trained_on": src,
            "source_run": source_run,
            "source_prompt_variant": source_spec["prompt_variant"],
            "eval_on": tgt,
            "score": headline_from_metrics(metrics),
            **metrics,
        }
    )

if transfer_rows:
    transfer_df = pd.DataFrame(transfer_rows)
    transfer_df["trained_on_label"] = transfer_df["trained_on"].map(pretty_transfer_source)
    save_table(transfer_df.round(4), "rq7_transfer_long", "RQ7 transfer results in long format.", "tab:rq7_transfer_long")
    tmat = transfer_df.pivot(index="trained_on", columns="eval_on", values="score")
    row_order = [x for x in TRANSFER_SOURCES if x in tmat.index]
    col_order = [x for x in TRANSFER_TARGETS if x in tmat.columns]
    tmat = tmat.reindex(index=row_order, columns=col_order)
    table_tmat = tmat.round(4).reset_index()
    table_tmat.insert(1, "trained_on_label", table_tmat["trained_on"].map(pretty_transfer_source))
    save_table(table_tmat, "rq7_transfer", "RQ7 transfer matrix: train-on-row, eval-on-column.", "tab:rq7_transfer")

    fig, ax = plt.subplots(figsize=(6.8, 0.46 * len(tmat.index) + 2.1))
    vals = tmat.values.astype(float)
    im = ax.imshow(vals, cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(len(tmat.columns)))
    ax.set_xticklabels([pretty_dataset(c) for c in tmat.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(tmat.index)))
    ax.set_yticklabels([pretty_transfer_source(x) for x in tmat.index], fontsize=7)
    for i in range(len(tmat.index)):
        for j in range(len(tmat.columns)):
            v = vals[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.55 else "black")
    ax.set_xlabel("eval on")
    ax.set_ylabel("trained on")
    ax.set_title("RQ7: cross-domain transfer")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_fig(fig, "rq7_transfer", data=tmat.reset_index())
    plt.show()


# %% [markdown]
# ## Claim-to-evidence map
#
# This artifact is a writing guardrail. A claim can stay in the abstract,
# highlights, discussion, or conclusion only if this table points to the result
# table/figure/artifact that supports it.

# %% Cell 14 - claim-to-evidence map
claim_evidence_rows = [
    {
        "paper_location": "Abstract / Highlights",
        "claim": "The study compares BLIP-2 generative and BLIP-2 contrastive-enhanced alignment under the same ScienceQA setting.",
        "evidence": "tab:rq2_blip2; tab:rq2_retrieval; Figures/generated/rq2_blip2_objective.pdf; Figures/generated/rq2_retrieval_comparison.pdf",
        "claim_strength": "direct controlled comparison",
    },
    {
        "paper_location": "Abstract / Results",
        "claim": "Qwen2-VL answer-only, rationale-generative, and explanation-aware controls are separated on ScienceQA and A-OKVQA.",
        "evidence": "tab:controlled_strategy; tab:rq3_objective; Figures/generated/controlled_strategy_comparison.pdf",
        "claim_strength": "direct controlled comparison when all expected runs are current",
    },
    {
        "paper_location": "Methodology",
        "claim": "Answer-only adapters do not receive gold rationale targets, while rationale adapters train on reasoning plus answer.",
        "evidence": "data/prompt_template_preview.csv; data/sample_trace_walkthrough.md; tab:experiment_ledger",
        "claim_strength": "pipeline audit artifact",
    },
    {
        "paper_location": "Methodology / Results",
        "claim": "Length-aware alpha is audited as token-weighted rationale CE, while the fixed-alpha sweep remains an ablation.",
        "evidence": "data/experiment_ledger.csv; data/all_results.csv; tab:rq3_alpha; tab:controlled_strategy",
        "claim_strength": "implementation and result artifact",
    },
    {
        "paper_location": "Results",
        "claim": "ChartQA, DocVQA, and VQAv2 rows are answer-only fallback/domain rows, not rationale-supervised explanation-aware rows.",
        "evidence": "tab:experiment_ledger; tab:comparison_design; tab:rq4_domain",
        "claim_strength": "scope boundary",
    },
    {
        "paper_location": "Results / Discussion",
        "claim": "Masking drift measures evidence sensitivity and should not be described as proof that rationales are faithful.",
        "evidence": "tab:rq5_faithfulness; tab:masking_examples; data/masking_examples.json; Figures/generated/masking_example_*.pdf",
        "claim_strength": "diagnostic, not causal proof of rationale text",
    },
    {
        "paper_location": "Results",
        "claim": "The scale comparison is Qwen2-VL-2B vs Qwen2-VL-7B under the selected explanation-aware QLoRA setup.",
        "evidence": "tab:rq6_scale; Figures/generated/rq6_scale.pdf",
        "claim_strength": "narrow scale comparison",
    },
    {
        "paper_location": "Results",
        "claim": "Transfer is evaluated by applying each source-domain adapter to target domains while preserving the adapter prompt family.",
        "evidence": "tab:rq7_transfer; tab:rq7_transfer_long; data/transfer_matrix.json",
        "claim_strength": "cross-domain transfer artifact",
    },
]
claim_evidence = pd.DataFrame(claim_evidence_rows)
save_csv(claim_evidence, "claim_to_evidence_map")
save_json(claim_evidence_rows, "claim_to_evidence_map")
save_table(
    claim_evidence,
    "claim_to_evidence_map",
    "Claim-to-evidence map used to keep paper claims within the completed artifacts.",
    "tab:claim_evidence",
)

visual_qa_rows = [
    {
        "artifact": "controlled_strategy_comparison.pdf",
        "check": "All bars and labels readable; answer-only, rationale-generative, explanation-aware, BLIP-2 controls visible.",
        "status": "requires_manual_review_after_colab_run",
    },
    {
        "artifact": "rq2_retrieval_comparison.pdf",
        "check": "Frozen/generative/contrastive/random methods present; axis and legend readable.",
        "status": "requires_manual_review_after_colab_run",
    },
    {
        "artifact": "rq3_alpha_sweep.pdf",
        "check": "Alpha axis, all metric series, legend, and random baseline are readable.",
        "status": "requires_manual_review_after_colab_run",
    },
    {
        "artifact": "rq4_cross_domain.pdf",
        "check": "Caption and labels make clear these are in-domain adaptation/fallback rows, not transfer.",
        "status": "requires_manual_review_after_colab_run",
    },
    {
        "artifact": "rq5_faithfulness.pdf",
        "check": "Drift values and confidence intervals are readable; figure is not used as rationale-faithfulness proof.",
        "status": "requires_manual_review_after_colab_run",
    },
    {
        "artifact": "masking_example_*.pdf",
        "check": "Original, top-mask, random-mask, and blank-control panels are readable and not pixelated.",
        "status": "requires_manual_review_after_colab_run",
    },
    {
        "artifact": "rq7_transfer.pdf",
        "check": "Heatmap labels fit; rows are source adapters and columns are target domains.",
        "status": "requires_manual_review_after_colab_run",
    },
]
visual_qa = pd.DataFrame(visual_qa_rows)
save_csv(visual_qa, "visual_qa_checklist")
save_json(visual_qa_rows, "visual_qa_checklist")
save_markdown(
    "# Visual QA checklist\n\n"
    + "\n".join(
        f"- `{row['artifact']}`: {row['check']} **Status:** {row['status']}."
        for row in visual_qa_rows
    ),
    "visual_qa_checklist",
)


# %% [markdown]
# ## Final manifest and strict validation
#
# The final cell writes a manifest and raises if expected runs or required paper
# artifacts are missing. Set `PAPER_STRICT_FINAL=0` only when you intentionally
# want a partial artifact pass.

# %% Cell 15 - manifest and final validation
def manifest_for(root: Path) -> pd.DataFrame:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            records.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "directory": str(path.parent.relative_to(root)),
                }
            )
    return pd.DataFrame(records)


manifest_parts = []
for root, label in [(FIG_DIR, "figures"), (TAB_DIR, "tables"), (ART_DIR, "data"), (LOG_DIR, "logs")]:
    part = manifest_for(root)
    if not part.empty:
        part.insert(0, "artifact_type", label)
        manifest_parts.append(part)

manifest = pd.concat(manifest_parts, ignore_index=True) if manifest_parts else pd.DataFrame()
save_csv(manifest, "artifact_manifest")
save_json(manifest.to_dict(orient="records"), "artifact_manifest")

required_files = [
    FIG_DIR / "controlled_strategy_comparison.pdf",
    FIG_DIR / "dataset_sample_aokvqa_0.pdf",
    FIG_DIR / "dataset_sample_chartqa_0.pdf",
    FIG_DIR / "dataset_sample_docvqa_0.pdf",
    FIG_DIR / "dataset_sample_scienceqa_0.pdf",
    FIG_DIR / "dataset_sample_vqav2_0.pdf",
    FIG_DIR / "rq2_blip2_objective.pdf",
    FIG_DIR / "rq2_contrastive_retrieval.pdf",
    FIG_DIR / "rq2_retrieval_comparison.pdf",
    FIG_DIR / "rq3_alpha_sweep.pdf",
    FIG_DIR / "rq3_objective_comparison.pdf",
    FIG_DIR / "rq4_cross_domain.pdf",
    FIG_DIR / "rq5_faithfulness.pdf",
    FIG_DIR / "rq6_scale.pdf",
    FIG_DIR / "rq7_transfer.pdf",
    TAB_DIR / "all_results.tex",
    TAB_DIR / "claim_to_evidence_map.tex",
    TAB_DIR / "comparison_design.tex",
    TAB_DIR / "controlled_strategy_comparison.tex",
    TAB_DIR / "experiment_ledger.tex",
    TAB_DIR / "split_leakage_audit.tex",
    TAB_DIR / "uncertainty_estimates.tex",
    TAB_DIR / "rq2_blip2.tex",
    TAB_DIR / "rq2_contrastive_retrieval.tex",
    TAB_DIR / "rq2_retrieval_comparison.tex",
    TAB_DIR / "rq3_alpha_sweep.tex",
    TAB_DIR / "rq4_cross_domain.tex",
    TAB_DIR / "rq5_faithfulness.tex",
    TAB_DIR / "masking_examples.tex",
    TAB_DIR / "rq6_scale.tex",
    TAB_DIR / "rq7_transfer.tex",
    ART_DIR / "all_results.csv",
    ART_DIR / "claim_to_evidence_map.csv",
    ART_DIR / "comparison_design.csv",
    ART_DIR / "controlled_strategy_comparison.csv",
    ART_DIR / "dataset_prompt_preview.md",
    ART_DIR / "dataset_sample_preview.csv",
    ART_DIR / "dataset_sample_preview.json",
    ART_DIR / "prompt_template_preview.csv",
    ART_DIR / "prompt_template_preview.json",
    ART_DIR / "experiment_ledger.csv",
    ART_DIR / "split_leakage_audit.csv",
    ART_DIR / "selected_split_records.csv",
    ART_DIR / "selected_split_records.json",
    ART_DIR / "uncertainty_estimates.csv",
    ART_DIR / "visual_qa_checklist.csv",
    ART_DIR / "visual_qa_checklist.json",
    ART_DIR / "visual_qa_checklist.md",
    ART_DIR / "paper_pipeline_code_digest.json",
    ART_DIR / "rq2_contrastive_retrieval.json",
    ART_DIR / "rq2_retrieval_comparison.json",
    ART_DIR / "faithfulness_scores.json",
    ART_DIR / "masking_examples.json",
    ART_DIR / "transfer_matrix.json",
]
if os.environ.get("PAPER_SKIP_MODEL_DIAGNOSTICS", "0") != "1":
    required_files.extend(
        [
            ART_DIR / "model_io_loss_metric_preview.json",
            ART_DIR / "model_io_loss_metric_preview.md",
        ]
    )
if os.environ.get("PAPER_SKIP_MODEL_TRACE", "0") != "1":
    required_files.extend(
        [
            ART_DIR / "sample_trace_summary.csv",
            ART_DIR / "sample_trace_summary.json",
            ART_DIR / "sample_trace_tensors.csv",
            ART_DIR / "sample_trace_tensors.json",
            ART_DIR / "sample_trace_tokens.csv",
            ART_DIR / "sample_trace_walkthrough.md",
        ]
    )

missing_artifacts = [str(p) for p in required_files if not p.exists()]
if "run_status" not in globals():
    RUN_STATUS_PATH = LOG_DIR / "run_status.json"
    run_status = json.loads(RUN_STATUS_PATH.read_text()) if RUN_STATUS_PATH.exists() else {}
failed_runs = {
    name: status
    for name, status in run_status.items()
    if isinstance(status, dict)
    and status.get("status") == "failed"
    and name in RUN_BY_NAME
    and not result_ready(name)
}
states = {exp["name"]: result_state(exp) for exp in EXPERIMENTS}
missing_expected = [name for name, state in states.items() if state == "missing_results"]
stale_expected = {name: state for name, state in states.items() if state.startswith("stale")}

final_report = {
    "paper_out": str(PAPER_OUT),
    "repo_figures": str(REPO_FIG_DIR),
    "repo_tables": str(REPO_TAB_DIR),
    "expected_runs": EXPECTED_RUNS,
    "missing_expected_runs": missing_expected,
    "stale_expected_runs": stale_expected,
    "failed_runs": failed_runs,
    "missing_required_artifacts": missing_artifacts,
    "pipeline_code_version": PIPELINE_CODE_VERSION,
    "code_digest": PIPELINE_CODE_DIGEST,
}
save_json(final_report, "final_validation", LOG_DIR)

print("Artifact manifest:", ART_DIR / "artifact_manifest.csv")
print("Figures:", FIG_DIR)
print("Tables:", TAB_DIR)
print("Repo figure mirror:", REPO_FIG_DIR)
print("Repo table mirror:", REPO_TAB_DIR)

strict_final = os.environ.get("PAPER_STRICT_FINAL", "1") != "0"
if strict_final and (missing_expected or stale_expected or failed_runs or missing_artifacts):
    raise RuntimeError(
        "Paper pipeline incomplete. See logs/final_validation.json. "
        f"missing_runs={missing_expected}, stale_runs={list(stale_expected)}, "
        f"failed_runs={list(failed_runs)}, "
        f"missing_artifacts={missing_artifacts}"
    )

print("Paper pipeline complete.")
