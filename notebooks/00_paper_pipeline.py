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
import json
import os
import shutil
import traceback
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.data import build_collator, build_dataset, build_template
from src.evaluation import Evaluator, build_metrics
from src.evaluation.faithfulness import region_importance
from src.evaluation.metrics.retrieval import retrieval_recall
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


def load_run(name: str):
    out = experiment_dir(name)
    cfg_path = out / "config.resolved.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing resolved config for {name}: {cfg_path}")
    cfg = load_config(cfg_path)
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

# RQ3 alpha sweep on ScienceQA. Alpha=1.0 is the answer-only endpoint; alpha=0.5
# is the flagship balanced explanation-aware run.
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

# Qwen2-VL generative controls with rationale-bearing datasets.
EXPERIMENTS.extend(
    [
        spec(
            name="rq2_qwen2vl_generative_scienceqa",
            config=f"{BASE}/rq2_qwen2vl_generative_scienceqa.yaml",
            metrics=MC_METRICS,
            rq="RQ2/RQ3",
            domain="science",
            dataset="scienceqa",
            backbone="Qwen2-VL-2B",
            objective="generative",
            role="qwen_generative_control",
        ),
        spec(
            name="rq2_qwen2vl_generative_aokvqa",
            config=f"{BASE}/rq2_qwen2vl_generative_aokvqa.yaml",
            metrics=MC_METRICS,
            rq="RQ2/RQ3",
            domain="commonsense",
            dataset="aokvqa",
            backbone="Qwen2-VL-2B",
            objective="generative",
            role="qwen_generative_control",
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
    ]
)

# RQ4/RQ7 domain coverage. ChartQA/DocVQA/VQAv2 have no gold rationales, so the
# explanation-aware template falls back to answer-only targets; do not interpret
# those runs as rationale-supervised training.
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

matrix_df = pd.DataFrame(EXPERIMENTS)
save_csv(matrix_df, "paper_run_matrix")
save_json(EXPERIMENTS, "paper_run_matrix")

print(f"{len(EXPERIMENTS)} runs queued:")
for e in EXPERIMENTS:
    print(f"  {e['name']:48s} | {e['backbone']:14s} | {e['dataset']:9s} | {e['objective']}")


# %% Cell 3 - config validation without loading any model
config_errors: dict[str, str] = {}
for exp in EXPERIMENTS:
    try:
        cfg = load_config(exp["config"], overrides=exp["overrides"])
        cfg.name = exp["name"]
        if exp["metrics"] is not None:
            cfg.eval.metrics = exp["metrics"]
    except Exception as exc:  # noqa: BLE001
        config_errors[exp["name"]] = "".join(
            traceback.format_exception_only(type(exc), exc)
        ).strip()

save_json(config_errors, "config_validation_errors", LOG_DIR)
if config_errors:
    raise RuntimeError(f"Config validation failed for {len(config_errors)} run(s): {config_errors}")
print("All configs load cleanly.")


# %% [markdown]
# ## Pre-flight checks
#
# These checks are intentionally run before training:
#
# - masking preflight catches prompt leakage and empty supervised targets;
# - contrastive preflight checks that BLIP-2 InfoNCE reaches both the text
#   projection and trainable Q-Former parameters.

# %% Cell 4 - masking and contrastive preflight
def preflight_masking(cfg) -> dict[str, Any]:
    wrapper = build_model(cfg.model)
    try:
        obj = build_objective(cfg.objective)
        ds = build_dataset(cfg.data, split=cfg.data.split_train)
        if len(ds) == 0:
            raise ValueError(f"{cfg.name}: dataset is empty after filtering")
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

seen_signatures: set[tuple[str, str, str, str]] = set()
for exp in EXPERIMENTS:
    cfg = load_config(exp["config"], overrides=exp["overrides"])
    cfg.name = exp["name"]
    if exp["metrics"] is not None:
        cfg.eval.metrics = exp["metrics"]
    sig = (cfg.model.name, cfg.model.pretrained, cfg.data.name, cfg.data.prompt_variant)
    if sig in seen_signatures:
        continue
    seen_signatures.add(sig)
    if exp["name"] in masking_log:
        print(f"skip masking preflight {exp['name']} (cached)")
        continue
    print("masking preflight:", exp["name"], sig)
    masking_log[exp["name"]] = preflight_masking(cfg)
    save_json(masking_log, "preflight_masking", LOG_DIR)

contrastive_log_path = LOG_DIR / "preflight_contrastive.json"
contrastive_log = json.loads(contrastive_log_path.read_text()) if contrastive_log_path.exists() else {}
for exp in EXPERIMENTS:
    cfg = load_config(exp["config"], overrides=exp["overrides"])
    cfg.name = exp["name"]
    if cfg.objective.name != "contrastive":
        continue
    if exp["name"] in contrastive_log:
        print(f"skip contrastive preflight {exp['name']} (cached)")
        continue
    print("contrastive preflight:", exp["name"])
    contrastive_log[exp["name"]] = preflight_contrastive(cfg)
    save_json(contrastive_log, "preflight_contrastive", LOG_DIR)

print("All pre-flight checks completed.")


# %% [markdown]
# ## Run the matrix
#
# This cell is crash-safe. It writes `run_status.json` after each experiment.
# If a run fails, the loop records the stack trace and keeps moving so already
# completed outputs are not lost. The final manifest cell fails loudly if any
# required run is still missing or failed.

# %% Cell 5 - run every experiment, resumably
RUN_STATUS_PATH = LOG_DIR / "run_status.json"
run_status = json.loads(RUN_STATUS_PATH.read_text()) if RUN_STATUS_PATH.exists() else {}

selected = os.environ.get("PAPER_RUNS")
selected_runs = {x.strip() for x in selected.split(",")} if selected else None

for exp in EXPERIMENTS:
    name = exp["name"]
    if selected_runs and name not in selected_runs:
        continue
    if result_path(name).exists():
        run_status[name] = {"status": "done", "path": str(result_path(name))}
        save_json(run_status, "run_status", LOG_DIR)
        print(f"skip {name} (results.json exists)")
        continue

    try:
        cfg = load_config(exp["config"], overrides=exp["overrides"])
        cfg.name = name
        if exp["metrics"] is not None:
            cfg.eval.metrics = exp["metrics"]
        print(f"\n===== RUN {name} | metrics={cfg.eval.metrics} =====")
        results = ExperimentRunner(cfg).run()
        run_status[name] = {
            "status": "done",
            "path": str(result_path(name)),
            "headline": headline_from_metrics(results.get("final", {})),
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

# %% Cell 6 - collect expected results only
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
for exp in EXPERIMENTS:
    name = exp["name"]
    path = result_path(name)
    if not path.exists():
        missing_runs.append(name)
        continue
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
df


# %% Cell 7 - paper tables
if df.empty:
    raise RuntimeError("No expected results were found. Run Cell 5 first.")

master_cols = [
    "run",
    "rq",
    "domain",
    "backbone",
    "objective",
    "alpha",
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
    "rq2_qwen2vl_generative_scienceqa",
    "rq3_alpha_050",
    "rq2_qwen2vl_generative_aokvqa",
    "rq3_qwen2vl_explanation_aware_aokvqa",
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


# %% Cell 8 - paper figures from standard results
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
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    x = np.arange(len(plot))
    ax.bar(x - 0.18, plot["headline_base"], 0.36, label="zero-shot", color="white", edgecolor="black")
    ax.bar(x + 0.18, plot["headline"], 0.36, label="fine-tuned", color="dimgray", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(plot["objective"], rotation=10, ha="right")
    ax.set_ylabel("ScienceQA MC accuracy")
    ax.set_title("RQ2: BLIP-2 objective comparison")
    ax.legend(frameon=False)
    save_fig(fig, "rq2_blip2_objective", data=plot)
    plt.show()

if not rq3_cmp.empty:
    plot = rq3_cmp[["dataset", "objective", "mc_acc", "rouge_l"]].copy()
    plot["label"] = plot["dataset"] + "\n" + plot["objective"].str.replace("_", " ")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    axes[0].bar(plot["label"], plot["mc_acc"], color="dimgray", edgecolor="black")
    axes[0].set_title("Answer accuracy")
    axes[0].set_ylabel("MC accuracy")
    axes[1].bar(plot["label"], plot["rouge_l"], color="lightgray", edgecolor="black")
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
    plot = piv[["run", "headline_base", "headline"]].copy()
    fig, ax = plt.subplots(figsize=(7.8, 0.38 * len(plot) + 1.8))
    y = np.arange(len(plot))
    ax.barh(y - 0.18, plot["headline_base"], 0.36, label="zero-shot", color="white", edgecolor="black")
    ax.barh(y + 0.18, plot["headline"], 0.36, label="fine-tuned", color="dimgray", edgecolor="black")
    ax.set_yticks(y)
    ax.set_yticklabels(plot["run"], fontsize=7)
    ax.set_xlabel("native headline score")
    ax.set_title("Fine-tuning lift by run")
    ax.legend(frameon=False)
    save_fig(fig, "headline_lift", data=plot)
    plt.show()

if not cross.empty:
    plot = cross[["domain", "dataset", "headline"]].dropna().copy()
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    ax.bar(plot["domain"], plot["headline"], color="dimgray", edgecolor="black")
    ax.set_ylabel("native headline score")
    ax.set_title("RQ4: cross-domain performance")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.2)
    save_fig(fig, "rq4_cross_domain", data=plot)
    plt.show()

if len(scale) == 2:
    plot = scale[["model_size", "mc_acc", "rouge_l", "bleu"]].copy()
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    ax.bar(plot["model_size"], plot["mc_acc"], color=["white", "dimgray"], edgecolor="black")
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

# %% Cell 9 - BLIP-2 contrastive retrieval R@K/MRR
def contrastive_retrieval_eval(run_name: str, n_eval: int = 200, batch_size: int = 16) -> dict[str, float]:
    cfg, wrapper, _, eval_ds = load_run(run_name)
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


RET_PATH = ART_DIR / "rq2_contrastive_retrieval.json"
retrieval_scores = json.loads(RET_PATH.read_text()) if RET_PATH.exists() else {}
if result_path("rq2_blip2_contrastive_scienceqa").exists() and "rq2_blip2_contrastive_scienceqa" not in retrieval_scores:
    try:
        retrieval_scores["rq2_blip2_contrastive_scienceqa"] = contrastive_retrieval_eval(
            "rq2_blip2_contrastive_scienceqa",
            n_eval=int(os.environ.get("PAPER_RETRIEVAL_N", "200")),
            batch_size=int(os.environ.get("PAPER_RETRIEVAL_BATCH", "16")),
        )
        save_json(retrieval_scores, "rq2_contrastive_retrieval")
    except Exception as exc:  # noqa: BLE001
        retrieval_scores["rq2_blip2_contrastive_scienceqa_error"] = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        save_json(retrieval_scores, "rq2_contrastive_retrieval")

if "rq2_blip2_contrastive_scienceqa" in retrieval_scores:
    ret = pd.DataFrame(
        [{"metric": k, "score": v} for k, v in retrieval_scores["rq2_blip2_contrastive_scienceqa"].items() if k != "n_eval"]
    )
    save_table(ret.round(4), "rq2_contrastive_retrieval", "RQ2 BLIP-2 contrastive image-to-answer retrieval.", "tab:rq2_retrieval")
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.bar(ret["metric"], ret["score"], color="dimgray", edgecolor="black")
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("RQ2: contrastive retrieval")
    ax.grid(axis="y", alpha=0.2)
    save_fig(fig, "rq2_contrastive_retrieval", data=ret)
    plt.show()


# %% [markdown]
# ## RQ5 faithfulness
#
# Mean masking drift is a causal evidence-sensitivity diagnostic. It does not
# prove rationale faithfulness alone, but it tells us whether the gold answer
# likelihood drops when image evidence is occluded.

# %% Cell 10 - evidence-masking drift scores
def faithfulness_score(wrapper, eval_ds, template, n: int = 30, grid: tuple[int, int] = (3, 3)) -> float:
    drifts = []
    for i in range(min(n, len(eval_ds))):
        drifts.append(region_importance(wrapper, eval_ds[i], template, grid=grid).mean_drift)
    return float(np.mean(drifts)) if drifts else float("nan")


FAITH_PATH = ART_DIR / "faithfulness_scores.json"
faith = json.loads(FAITH_PATH.read_text()) if FAITH_PATH.exists() else {}

faith_runs = [
    exp["name"]
    for exp in EXPERIMENTS
    if exp["backbone"].startswith("Qwen2-VL") and result_path(exp["name"]).exists()
]
for name in faith_runs:
    if name in faith:
        continue
    try:
        cfg, wrapper, template, eval_ds = load_run(name)
        faith[name] = {
            "mean_drift": faithfulness_score(
                wrapper,
                eval_ds,
                template,
                n=int(os.environ.get("PAPER_FAITH_N", "30")),
                grid=(3, 3),
            ),
            "n_eval": int(os.environ.get("PAPER_FAITH_N", "30")),
            "grid": [3, 3],
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
    {"run": k, "mean_drift": v["mean_drift"], "n_eval": v["n_eval"]}
    for k, v in faith.items()
    if isinstance(v, dict) and "mean_drift" in v
]
if faith_rows:
    fs = pd.DataFrame(faith_rows).sort_values("run").reset_index(drop=True)
    save_table(fs.round(4), "rq5_faithfulness", "RQ5 evidence-masking drift by model.", "tab:rq5_faithfulness")
    fig, ax = plt.subplots(figsize=(7.8, 0.42 * len(fs) + 1.8))
    ax.barh(fs["run"], fs["mean_drift"], color="dimgray", edgecolor="black")
    ax.set_xlabel("mean masking drift")
    ax.set_title("RQ5: visual evidence sensitivity")
    plt.setp(ax.get_yticklabels(), fontsize=7)
    ax.grid(axis="x", alpha=0.2)
    save_fig(fig, "rq5_faithfulness", data=fs)
    plt.show()


# %% [markdown]
# ## Qualitative samples and faithfulness heatmaps
#
# These are optional for the final journal if page limits are tight, but they are
# saved because they are useful for interpreting failures and writing discussion.

# %% Cell 11 - qualitative sample grids and heatmaps
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
                f"[{i}] {hit} model: {pred.predicted_text} | gold: {ex.answer}\n{ex.question[:70]}",
                fontsize=8,
            )
            sample_lines.append(
                f"## [{i}] {hit}\n"
                f"- question: {ex.question}\n"
                f"- gold: {ex.answer}\n"
                f"- prediction: {pred.predicted_text}\n"
                f"- reasoning: {pred.reasoning}\n"
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
            res = region_importance(wrapper, ex, template, grid=(4, 4))
            heat = np.array(res.drifts).reshape(res.grid)
            ax.imshow(ex.image.convert("RGB"))
            ax.imshow(
                heat,
                cmap="hot",
                alpha=0.5,
                extent=[0, ex.image.width, ex.image.height, 0],
                interpolation="bilinear",
            )
            ax.set_title(f"[{i}] max drift={res.max_drift:.3f}\n{ex.question[:70]}", fontsize=8)
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
    if run_name in qual_status:
        continue
    if not result_path(run_name).exists():
        qual_status[run_name] = {"status": "missing_run"}
        save_json(qual_status, "qualitative_status", LOG_DIR)
        continue
    try:
        save_qualitative_views(run_name, n=int(os.environ.get("PAPER_QUAL_N", "6")))
        qual_status[run_name] = {"status": "done"}
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
# This evaluates each Qwen2-VL-2B domain adapter on each target domain without
# retraining. Sources and targets use the same prompt family, but target metrics
# are loaded from the target config.

# %% Cell 12 - full transfer matrix
TRANSFER_SOURCES = {
    "science": "rq3_alpha_050",
    "aokvqa": "rq3_qwen2vl_explanation_aware_aokvqa",
    "charts": "rq4_qwen2vl_explanation_aware_chartqa",
    "documents": "rq4_qwen2vl_explanation_aware_docvqa",
    "vqav2": "rq4_qwen2vl_explanation_aware_vqav2",
}

TRANSFER_TARGETS = {
    "science": f"{BASE}/rq3_qwen2vl_explanation_aware_scienceqa.yaml",
    "aokvqa": f"{BASE}/rq3_qwen2vl_explanation_aware_aokvqa.yaml",
    "charts": f"{BASE}/rq4_qwen2vl_explanation_aware_chartqa.yaml",
    "documents": f"{BASE}/rq4_qwen2vl_explanation_aware_docvqa.yaml",
    "vqav2": f"{BASE}/rq4_qwen2vl_explanation_aware_vqav2.yaml",
}


def transfer_eval(source_run: str, target_cfg_path: str, n_eval: int = 200) -> dict[str, float]:
    _, wrapper, _, _ = load_run(source_run)
    try:
        cfg_t = load_config(target_cfg_path)
        cfg_t.data.max_eval = n_eval
        target_template = build_template(cfg_t.data.prompt_variant)
        ds_t = build_dataset(cfg_t.data, split=cfg_t.data.split_eval)
        ev = Evaluator(wrapper, ds_t, target_template, cfg_t.eval, build_metrics(cfg_t.eval.metrics))
        return ev.evaluate()
    finally:
        del wrapper
        free_gpu()


TX_PATH = ART_DIR / "transfer_matrix.json"
tx = json.loads(TX_PATH.read_text()) if TX_PATH.exists() else {}
transfer_n = int(os.environ.get("PAPER_TRANSFER_N", "200"))

for source_label, source_run in TRANSFER_SOURCES.items():
    if not result_path(source_run).exists():
        print(f"skip transfer source {source_label}: missing {source_run}")
        continue
    for target_label, target_cfg in TRANSFER_TARGETS.items():
        label = f"{source_label}->{target_label}"
        if label in tx:
            continue
        try:
            tx[label] = transfer_eval(source_run, target_cfg, n_eval=transfer_n)
            save_json(tx, "transfer_matrix")
        except Exception as exc:  # noqa: BLE001
            tx[f"{label}__error"] = {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            save_json(tx, "transfer_matrix")

transfer_rows = []
for label, metrics in tx.items():
    if label.endswith("__error") or not isinstance(metrics, dict):
        continue
    src, tgt = label.split("->")
    transfer_rows.append({"trained_on": src, "eval_on": tgt, "score": headline_from_metrics(metrics), **metrics})

if transfer_rows:
    transfer_df = pd.DataFrame(transfer_rows)
    save_table(transfer_df.round(4), "rq7_transfer_long", "RQ7 transfer results in long format.", "tab:rq7_transfer_long")
    tmat = transfer_df.pivot(index="trained_on", columns="eval_on", values="score")
    ordered = [x for x in TRANSFER_TARGETS if x in tmat.index or x in tmat.columns]
    tmat = tmat.reindex(index=ordered, columns=ordered)
    save_table(tmat.round(4).reset_index(), "rq7_transfer", "RQ7 transfer matrix: train-on-row, eval-on-column.", "tab:rq7_transfer")

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    vals = tmat.values.astype(float)
    im = ax.imshow(vals, cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(len(tmat.columns)))
    ax.set_xticklabels(tmat.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(tmat.index)))
    ax.set_yticklabels(tmat.index)
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
# ## Final manifest and strict validation
#
# The final cell writes a manifest and raises if expected runs or required paper
# artifacts are missing. Set `PAPER_STRICT_FINAL=0` only when you intentionally
# want a partial artifact pass.

# %% Cell 13 - manifest and final validation
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
    FIG_DIR / "rq2_blip2_objective.pdf",
    FIG_DIR / "rq2_contrastive_retrieval.pdf",
    FIG_DIR / "rq3_alpha_sweep.pdf",
    FIG_DIR / "rq3_objective_comparison.pdf",
    FIG_DIR / "rq4_cross_domain.pdf",
    FIG_DIR / "rq5_faithfulness.pdf",
    FIG_DIR / "rq6_scale.pdf",
    FIG_DIR / "rq7_transfer.pdf",
    TAB_DIR / "all_results.tex",
    TAB_DIR / "rq2_blip2.tex",
    TAB_DIR / "rq2_contrastive_retrieval.tex",
    TAB_DIR / "rq3_alpha_sweep.tex",
    TAB_DIR / "rq4_cross_domain.tex",
    TAB_DIR / "rq5_faithfulness.tex",
    TAB_DIR / "rq6_scale.tex",
    TAB_DIR / "rq7_transfer.tex",
    ART_DIR / "all_results.csv",
    ART_DIR / "rq2_contrastive_retrieval.json",
    ART_DIR / "faithfulness_scores.json",
    ART_DIR / "transfer_matrix.json",
]

missing_artifacts = [str(p) for p in required_files if not p.exists()]
failed_runs = {
    name: status
    for name, status in run_status.items()
    if isinstance(status, dict) and status.get("status") == "failed"
}
missing_expected = [name for name in EXPECTED_RUNS if not result_path(name).exists()]

final_report = {
    "paper_out": str(PAPER_OUT),
    "repo_figures": str(REPO_FIG_DIR),
    "repo_tables": str(REPO_TAB_DIR),
    "expected_runs": EXPECTED_RUNS,
    "missing_expected_runs": missing_expected,
    "failed_runs": failed_runs,
    "missing_required_artifacts": missing_artifacts,
}
save_json(final_report, "final_validation", LOG_DIR)

print("Artifact manifest:", ART_DIR / "artifact_manifest.csv")
print("Figures:", FIG_DIR)
print("Tables:", TAB_DIR)
print("Repo figure mirror:", REPO_FIG_DIR)
print("Repo table mirror:", REPO_TAB_DIR)

strict_final = os.environ.get("PAPER_STRICT_FINAL", "1") != "0"
if strict_final and (missing_expected or failed_runs or missing_artifacts):
    raise RuntimeError(
        "Paper pipeline incomplete. See logs/final_validation.json. "
        f"missing_runs={missing_expected}, failed_runs={list(failed_runs)}, "
        f"missing_artifacts={missing_artifacts}"
    )

print("Paper pipeline complete.")
