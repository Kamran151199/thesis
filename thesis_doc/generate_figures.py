"""Generate all 14 expected-results figures for the thesis proposal.

Run from the repo root:
    uv run python thesis_doc/generate_figures.py

Figures are saved to thesis_doc/figures/ as PNG (publication-quality).
Numbers embedded here MUST match the dummy data in thesis_doc/proposal.md.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "sans-serif",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)

PALETTE = {
    "contrastive": "#4C72B0",
    "generative": "#DD8452",
    "explanation": "#C44E52",
    "baseline": "#8C8C8C",
    "accent": "#55A868",
    "purple": "#8172B2",
    "teal": "#64B5CD",
    "gold": "#CCB974",
}


def save(fig, name: str) -> None:
    out = FIG_DIR / name
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------- #
# RQ1                                                                         #
# --------------------------------------------------------------------------- #


def fig_1_1_taxonomy() -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def node(x, y, w, h, text, color):
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.08",
            facecolor=color,
            edgecolor="#333",
            linewidth=1.3,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white" if color != "#F5F5F5" else "#222",
        )
        return (x + w / 2, y + h / 2)

    def arrow(src, dst):
        ax.add_patch(
            FancyArrowPatch(
                src,
                dst,
                arrowstyle="-",
                color="#666",
                linewidth=1.2,
                connectionstyle="arc3,rad=0",
            )
        )

    root = node(4.5, 5.8, 3, 0.8, "Vision–Language Alignment", "#34495E")

    contr = node(0.3, 4.2, 3, 0.8, "Contrastive (CLIP, ALIGN)", PALETTE["contrastive"])
    hybrid = node(4.5, 4.2, 3, 0.8, "Hybrid (BLIP, BLIP-2)", PALETTE["purple"])
    gener = node(8.7, 4.2, 3, 0.8, "Generative (LLaVA, Qwen2-VL)", PALETTE["generative"])
    arrow(root, contr)
    arrow(root, hybrid)
    arrow(root, gener)

    c1 = node(0.1, 2.7, 1.6, 0.7, "Dual Encoder", PALETTE["teal"])
    c2 = node(1.8, 2.7, 1.6, 0.7, "InfoNCE Loss", PALETTE["teal"])
    h1 = node(4.3, 2.7, 1.6, 0.7, "Q-Former", PALETTE["gold"])
    h2 = node(6.0, 2.7, 1.6, 0.7, "Frozen LLM", PALETTE["gold"])
    g1 = node(8.5, 2.7, 1.6, 0.7, "MLP Projection", PALETTE["accent"])
    g2 = node(10.2, 2.7, 1.6, 0.7, "Cross-Attention", PALETTE["accent"])
    for src, dst in [(contr, c1), (contr, c2), (hybrid, h1), (hybrid, h2), (gener, g1), (gener, g2)]:
        arrow(src, dst)

    thesis = node(4.5, 0.7, 3, 1.0, "THIS THESIS:\nExplanation-Aware\nGenerative Alignment", PALETTE["explanation"])
    ax.add_patch(
        FancyArrowPatch(
            (10.0, 2.7),
            (6.0, 1.2),
            arrowstyle="->",
            color=PALETTE["explanation"],
            linewidth=2.0,
            linestyle="--",
            connectionstyle="arc3,rad=-0.15",
        )
    )
    ax.text(8.4, 1.8, "rationale-grounded\ntoken supervision", fontsize=8, style="italic", color=PALETTE["explanation"])

    ax.set_title("Figure 1.1 — Taxonomy of Vision–Language Alignment Methods", pad=12)
    save(fig, "fig_1_1_taxonomy.png")


def fig_1_2_timeline() -> None:
    rows = [
        ("CLIP", 2021.1, 41.5, "contrastive"),
        ("ALIGN", 2021.5, 42.8, "contrastive"),
        ("BLIP", 2022.0, 58.4, "generative"),
        ("Flamingo", 2022.4, 67.2, "generative"),
        ("BLIP-2", 2023.0, 65.2, "generative"),
        ("LLaVA-1.0", 2023.4, 71.0, "generative"),
        ("LLaVA-1.5", 2023.8, 80.0, "generative"),
        ("InternVL-1", 2024.1, 82.6, "generative"),
        ("Qwen2-VL-2B", 2024.7, 81.5, "generative"),
        ("InternVL-2", 2024.9, 84.3, "generative"),
        ("Qwen2-VL-7B", 2025.0, 83.0, "generative"),
        ("This thesis", 2026.4, 84.2, "explanation"),
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, year, score, kind in rows:
        color = (
            PALETTE["contrastive"]
            if kind == "contrastive"
            else PALETTE["generative"]
            if kind == "generative"
            else PALETTE["explanation"]
        )
        marker = "D" if kind == "explanation" else "o"
        ax.scatter(year, score, s=160, color=color, marker=marker, edgecolor="black", linewidth=0.8, zorder=3)
        dy = 2.0 if name != "This thesis" else 3.0
        ax.annotate(name, (year, score), xytext=(0, dy * 3), textcoords="offset points", ha="center", fontsize=8)

    ax.set_xlabel("Year")
    ax.set_ylabel("Mean benchmark score (VQAv2 / DocVQA / ChartQA)")
    ax.set_title("Figure 1.2 — Timeline of Vision–Language Alignment Milestones")
    ax.set_xlim(2020.8, 2026.8)
    ax.set_ylim(35, 92)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(
        handles=[
            mpatches.Patch(color=PALETTE["contrastive"], label="Contrastive"),
            mpatches.Patch(color=PALETTE["generative"], label="Generative"),
            mpatches.Patch(color=PALETTE["explanation"], label="Explanation-aware (this thesis)"),
        ],
        loc="lower right",
    )
    save(fig, "fig_1_2_timeline.png")


# --------------------------------------------------------------------------- #
# RQ2                                                                         #
# --------------------------------------------------------------------------- #


def fig_2_1_acc_gap() -> None:
    variants = ["Generative", "Generative +\ncontrastive"]
    frozen = [0.445, 0.445]
    tuned = [0.475, 0.480]

    x = np.arange(len(variants))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, frozen, width, color=PALETTE["baseline"], edgecolor="black", linewidth=0.6, label="Frozen reference")
    ax.bar(x + width / 2, tuned, width, color=PALETTE["contrastive"], edgecolor="black", linewidth=0.6, label="Fine-tuned")

    for i, score in enumerate(tuned):
        ax.text(i + width / 2, score + 0.015, f"{score:.3f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylabel("ScienceQA MC accuracy")
    ax.set_ylim(0, 0.60)
    ax.set_title("Figure 2.1 — BLIP-2 Objective Comparison Placeholder")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig_2_1_acc_gap.png")


def fig_2_2_attention() -> None:
    metrics = ["R@1", "R@5", "R@10", "MRR"]
    rows = {
        "Frozen BLIP-2": [0.010, 0.020, 0.035, 0.028],
        "BLIP-2 generative": [0.010, 0.020, 0.025, 0.030],
        "BLIP-2 contrastive": [0.010, 0.020, 0.030, 0.031],
        "Random rank": [0.005, 0.025, 0.050, 0.029],
    }
    colors = [PALETTE["baseline"], PALETTE["generative"], PALETTE["contrastive"], PALETTE["purple"]]
    x = np.arange(len(metrics))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, ((name, values), color) in enumerate(zip(rows.items(), colors)):
        ax.bar(x + (idx - 1.5) * width, values, width, label=name, color=color, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Retrieval score")
    ax.set_ylim(0, 0.08)
    ax.set_title("Figure 2.2 — BLIP-2 Retrieval Diagnostic Placeholder")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig_2_2_attention.png")


# --------------------------------------------------------------------------- #
# RQ3                                                                         #
# --------------------------------------------------------------------------- #


def fig_3_1_expl_acc() -> None:
    groups = ["ScienceQA", "A-OKVQA"]
    series = {
        "Zero-shot": [0.400, 0.355],
        "Answer-only": [0.800, 0.830],
        "Rationale+answer CE": [0.780, 0.790],
        "Explanation-aware": [0.765, 0.795],
        "Length-aware": [0.780, 0.790],
    }
    colors = [PALETTE["baseline"], PALETTE["contrastive"], PALETTE["generative"], PALETTE["explanation"], PALETTE["accent"]]
    x = np.arange(len(groups))
    width = 0.15
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, ((name, values), color) in enumerate(zip(series.items(), colors)):
        ax.bar(x + (idx - 2) * width, values, width, color=color, edgecolor="black", linewidth=0.5, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("MC accuracy")
    ax.set_ylim(0, 0.95)
    ax.set_title("Figure 3.1 — Controlled Strategy Comparison Placeholder")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig_3_1_expl_acc.png")


def fig_3_2_human_violin() -> None:
    alphas = [0.00, 0.10, 0.25, 0.50, 0.75, 1.00]
    acc = [0.725, 0.780, 0.775, 0.765, 0.690, 0.695]
    rouge = [0.417, 0.695, 0.686, 0.643, 0.604, np.nan]
    bleu = [0.299, 0.566, 0.548, 0.493, 0.443, np.nan]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(alphas, acc, marker="o", lw=2.3, color=PALETTE["contrastive"], label="MC accuracy")
    ax.plot(alphas, rouge, marker="s", lw=2.3, color=PALETTE["accent"], label="ROUGE-L")
    ax.plot(alphas, bleu, marker="^", lw=2.3, color=PALETTE["generative"], label="BLEU")
    ax.axhline(0.354, color="#999", linestyle="--", lw=1.2, label="Random-choice accuracy")
    ax.set_xlabel("alpha (answer-span weight)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.85)
    ax.set_title("Figure 3.2 — Alpha and Rationale Quality Placeholder")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    save(fig, "fig_3_2_human_violin.png")


# --------------------------------------------------------------------------- #
# RQ4                                                                         #
# --------------------------------------------------------------------------- #


def fig_4_1_ablation_heatmap() -> None:
    domains = ["ScienceQA\nalpha=0.50", "A-OKVQA\nalpha=0.50", "ChartQA\nfallback", "DocVQA\nfallback", "VQAv2\nfallback"]
    zero = [0.400, 0.355, 0.335, 0.349, 0.350]
    tuned = [0.765, 0.795, 0.655, 0.905, 0.730]
    x = np.arange(len(domains))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, zero, width, color=PALETTE["baseline"], edgecolor="black", linewidth=0.5, label="Zero-shot reference")
    ax.bar(x + width / 2, tuned, width, color=PALETTE["contrastive"], edgecolor="black", linewidth=0.5, label="Fine-tuned")
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylabel("Native headline metric")
    ax.set_ylim(0, 1.0)
    ax.set_title("Figure 4.1 — Domain-Level Adaptation Placeholder")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig_4_1_ablation_heatmap.png")


def fig_4_2_attention_entropy() -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    boxes = [
        ("Raw example", 0.04, 0.62, PALETTE["teal"]),
        ("Prompt template", 0.28, 0.62, PALETTE["contrastive"]),
        ("Collator", 0.52, 0.62, PALETTE["purple"]),
        ("Model input tokens", 0.76, 0.62, PALETTE["gold"]),
        ("Generated output", 0.28, 0.22, PALETTE["generative"]),
        ("Parsed answer", 0.52, 0.22, PALETTE["accent"]),
        ("Metric / loss", 0.76, 0.22, PALETTE["explanation"]),
    ]
    for text, x, y, color in boxes:
        rect = FancyBboxPatch((x, y), 0.18, 0.16, boxstyle="round,pad=0.02", facecolor=color, edgecolor="#333")
        ax.add_patch(rect)
        ax.text(x + 0.09, y + 0.08, text, ha="center", va="center", fontsize=10, color="white", fontweight="bold")
    for start, end in [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (5, 6)]:
        sx, sy = boxes[start][1] + 0.18, boxes[start][2] + 0.08
        ex, ey = boxes[end][1], boxes[end][2] + 0.08
        ax.add_patch(FancyArrowPatch((sx, sy), (ex, ey), arrowstyle="->", mutation_scale=14, linewidth=1.4, color="#555"))
    ax.set_title("Figure 4.2 — Pipeline Diagnostic Placeholder")
    save(fig, "fig_4_2_attention_entropy.png")


# --------------------------------------------------------------------------- #
# RQ5                                                                         #
# --------------------------------------------------------------------------- #


def fig_5_1_hallucination() -> None:
    runs = ["DocVQA fallback", "ChartQA fallback", "VQAv2 fallback", "ScienceQA answer-only", "A-OKVQA expl-aware", "ScienceQA alpha=0.50"]
    drift = [0.375, 0.224, 0.111, 0.090, 0.027, 0.020]
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(runs))
    ax.barh(y, drift, color=PALETTE["contrastive"], edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(runs)
    ax.invert_yaxis()
    ax.set_xlabel("Mean masking drift")
    ax.set_title("Figure 5.1 — Evidence-Masking Drift Placeholder")
    ax.grid(axis="x", alpha=0.25)
    save(fig, "fig_5_1_hallucination.png")


def fig_5_2_qualitative() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    base = np.ones((80, 80, 3)) * 0.96
    base[10:70, 12:68, :] = [0.98, 0.88, 0.78]
    for idx, h in enumerate([50, 38, 62, 44]):
        x0 = 18 + idx * 12
        base[70 - h : 70, x0 : x0 + 7, :] = [0.47, 0.25, 0.12]
    base[68:70, 12:68, :] = 0.25

    variants = []
    labels = [
        "Original\nscore: -2.04",
        "Top-drift mask\nscore: -2.88",
        "Random mask\nscore: -2.15",
        "Blank image\nscore: -3.31",
    ]
    variants.append(base.copy())
    top = base.copy()
    top[16:46, 41:62, :] = 0.55
    variants.append(top)
    random = base.copy()
    random[20:42, 16:34, :] = 0.55
    variants.append(random)
    blank = np.ones_like(base) * 0.55
    variants.append(blank)

    for ax, img, label in zip(axes, variants, labels):
        ax.imshow(img)
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Figure 5.2 — Masking Example Placeholder", fontsize=12, y=1.02)
    fig.text(0.5, 0.02, "Question: Which bar is highest? Gold answer: C. Drift is computed from full-image score minus masked score.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    save(fig, "fig_5_2_qualitative.png")


# --------------------------------------------------------------------------- #
# RQ6                                                                         #
# --------------------------------------------------------------------------- #


def fig_6_1_pareto() -> None:
    models = ["Qwen2-VL-2B\nQLoRA", "Qwen2-VL-7B\nQLoRA"]
    memory = [22, 40]
    trainable = [18.5, 43.0]
    x = np.arange(len(models))
    width = 0.36
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(x - width / 2, memory, width, color=PALETTE["contrastive"], edgecolor="black", label="Approx. GPU memory target")
    ax1.set_ylabel("GPU memory target (GB)")
    ax1.set_ylim(0, 48)
    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, trainable, width, color=PALETTE["accent"], edgecolor="black", label="Trainable params")
    ax2.set_ylabel("Trainable parameters (M)")
    ax2.set_ylim(0, 50)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_title("Figure 6.1 — Single-GPU Scale Context Placeholder")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=8)
    save(fig, "fig_6_1_pareto.png")


def fig_6_2_scaling() -> None:
    categories = ["MC accuracy", "ROUGE-L", "BLEU"]
    qlora_2b = [0.765, 0.643, 0.493]
    qlora_7b = [0.885, 0.738, 0.637]

    x = np.arange(len(categories))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, qlora_2b, width, color=PALETTE["generative"], edgecolor="black", linewidth=0.6, label="Qwen2-VL-2B (QLoRA)")
    ax.bar(x + width / 2, qlora_7b, width, color=PALETTE["explanation"], edgecolor="black", linewidth=0.6, label="Qwen2-VL-7B (QLoRA)")

    for i, (a, b) in enumerate(zip(qlora_2b, qlora_7b)):
        ax.text(i - width / 2, a + 0.03, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, b + 0.03, f"{b:.3f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Figure 6.2 — Qwen2-VL 2B versus 7B Scale Placeholder")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig_6_2_scaling.png")


# --------------------------------------------------------------------------- #
# RQ7                                                                         #
# --------------------------------------------------------------------------- #


def fig_7_1_transfer_matrix() -> None:
    train_domains = ["ScienceQA\nanswer-only", "ScienceQA\nexpl-aware", "A-OKVQA\nanswer-only", "DocVQA\nfallback", "VQAv2\nfallback"]
    eval_domains = ["ScienceQA", "A-OKVQA", "ChartQA", "DocVQA", "VQAv2"]
    data = np.array(
        [
            [0.800, 0.805, 0.645, 0.703, 0.630],
            [0.740, 0.765, 0.460, 0.491, 0.410],
            [0.690, 0.825, 0.635, 0.677, 0.700],
            [0.450, 0.435, 0.685, 0.910, 0.660],
            [0.440, 0.575, 0.660, 0.716, 0.730],
        ]
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = LinearSegmentedColormap.from_list("whblgn", ["#F7F7F7", "#A6CEE3", "#1F78B4"])
    im = ax.imshow(data, cmap=cmap, vmin=0.35, vmax=0.95)
    ax.set_xticks(range(len(eval_domains)))
    ax.set_xticklabels(eval_domains)
    ax.set_yticks(range(len(train_domains)))
    ax.set_yticklabels(train_domains)
    ax.set_xlabel("Evaluation domain")
    ax.set_ylabel("Training domain")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            color = "white" if data[i, j] > 0.75 else "black"
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=9, color=color)
    ax.set_title("Figure 7.1 — Cross-Domain Transfer Matrix Placeholder")
    fig.colorbar(im, ax=ax, label="Target-domain headline metric", shrink=0.8)
    save(fig, "fig_7_1_transfer_matrix.png")


def fig_7_2_forgetting() -> None:
    targets = ["Answer-only\nprompt", "Rationale\nprompt", "Fallback\nprompt"]
    direct = [0.80, 0.78, 0.65]
    shifted = [0.74, 0.70, 0.61]
    x = np.arange(len(targets))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, direct, width, color=PALETTE["contrastive"], edgecolor="black", label="Matched prompt")
    ax.bar(x + width / 2, shifted, width, color=PALETTE["baseline"], edgecolor="black", label="Shifted prompt")
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel("Target-domain score")
    ax.set_ylim(0, 0.9)
    ax.set_title("Figure 7.2 — Optional Transfer Stability Placeholder")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    save(fig, "fig_7_2_forgetting.png")


if __name__ == "__main__":
    fig_1_1_taxonomy()
    fig_1_2_timeline()
    fig_2_1_acc_gap()
    fig_2_2_attention()
    fig_3_1_expl_acc()
    fig_3_2_human_violin()
    fig_4_1_ablation_heatmap()
    fig_4_2_attention_entropy()
    fig_5_1_hallucination()
    fig_5_2_qualitative()
    fig_6_1_pareto()
    fig_6_2_scaling()
    fig_7_1_transfer_matrix()
    fig_7_2_forgetting()
    print(f"\nAll 14 figures saved to {FIG_DIR}")
