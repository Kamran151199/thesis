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
    domains = ["DocVQA\n(ANLS)", "ChartQA", "ScienceQA", "A-OKVQA", "VQAv2"]
    contr = [75.4, 63.2, 78.0, 60.1, 74.5]
    gener = [83.1, 71.8, 82.7, 62.5, 75.6]

    x = np.arange(len(domains))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width / 2, contr, width, color=PALETTE["contrastive"], edgecolor="black", linewidth=0.6, label="Contrastive-enhanced")
    b2 = ax.bar(x + width / 2, gener, width, color=PALETTE["generative"], edgecolor="black", linewidth=0.6, label="Generative")

    for bar, c, g in zip(x, contr, gener):
        ax.text(bar + width / 2, g + 0.6, f"+{g - c:.1f}", ha="center", fontsize=9, color=PALETTE["accent"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 95)
    ax.set_title("Figure 2.1 — Contrastive vs. Generative Alignment Across Domains")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig_2_1_acc_gap.png")


def fig_2_2_attention() -> None:
    rng = np.random.default_rng(2)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    domains = ["Document", "Chart", "Natural Image"]

    def make_attn(sharp: bool, cx: float, cy: float, size: int = 28) -> np.ndarray:
        y, x = np.mgrid[0:size, 0:size]
        sigma = 2.5 if sharp else 8.0
        attn = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
        attn += rng.normal(0, 0.02, size=attn.shape)
        return np.clip(attn, 0, None)

    centers = [(8, 7), (21, 12), (14, 16)]

    for col, (domain, (cx, cy)) in enumerate(zip(domains, centers)):
        ax = axes[0, col]
        ax.imshow(make_attn(False, cx, cy), cmap="viridis")
        ax.set_title(f"Contrastive — {domain}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[1, col]
        ax.imshow(make_attn(True, cx, cy), cmap="viridis")
        ax.set_title(f"Generative — {domain}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Figure 2.2 — Attention Maps: Contrastive vs. Generative", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig_2_2_attention.png")


# --------------------------------------------------------------------------- #
# RQ3                                                                         #
# --------------------------------------------------------------------------- #


def fig_3_1_expl_acc() -> None:
    domains = ["DocVQA", "ChartQA", "ScienceQA", "A-OKVQA", "VQAv2"]
    base = [83.1, 71.8, 82.7, 62.5, 75.6]
    expl = [84.9, 74.2, 87.8, 65.1, 76.3]

    x = np.arange(len(domains))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, base, width, color=PALETTE["generative"], edgecolor="black", linewidth=0.6, label="Generative (baseline)")
    ax.bar(x + width / 2, expl, width, color=PALETTE["explanation"], edgecolor="black", linewidth=0.6, label="Explanation-aware")

    for i, (b, e) in enumerate(zip(base, expl)):
        ax.text(i + width / 2, e + 0.6, f"+{e - b:.1f}", ha="center", fontsize=9, color=PALETTE["accent"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Figure 3.1 — Effect of Explanation-Aware Training on Accuracy")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig_3_1_expl_acc.png")


def fig_3_2_human_violin() -> None:
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    def draw(ax, title, baseline_mean, expl_mean):
        base = np.clip(rng.normal(baseline_mean, 0.8, 150), 1, 5)
        expl = np.clip(rng.normal(expl_mean, 0.7, 150), 1, 5)
        parts = ax.violinplot([base, expl], showmeans=True, showextrema=False, widths=0.85)
        for pc, color in zip(parts["bodies"], [PALETTE["generative"], PALETTE["explanation"]]):
            pc.set_facecolor(color)
            pc.set_edgecolor("black")
            pc.set_alpha(0.75)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Generative\n(baseline)", "Explanation-\naware"])
        ax.set_ylabel("Rating (1–5 Likert)")
        ax.set_ylim(0.8, 5.2)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

    draw(axes[0], "(a) Plausibility", 3.2, 4.1)
    draw(axes[1], "(b) Faithfulness", 2.7, 4.0)
    fig.suptitle("Figure 3.2 — Human-Rated Explanation Quality (n=150 per group)", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig_3_2_human_violin.png")


# --------------------------------------------------------------------------- #
# RQ4                                                                         #
# --------------------------------------------------------------------------- #


def fig_4_1_ablation_heatmap() -> None:
    ablations = [
        "Fixed 448×448 (vs. native res.)",
        "Q-Former-style proj. (vs. MLP)",
        "2B LM (vs. 7B LM)",
        "Single-domain train (vs. mixed)",
    ]
    domains = ["DocVQA", "ChartQA", "ScienceQA", "A-OKVQA", "VQAv2"]
    data = np.array(
        [
            [-6.8, -4.5, -1.2, -0.4, -0.2],
            [+0.4, +0.6, -0.1, +0.2, +0.3],
            [-2.2, -3.1, -4.0, -3.8, -3.5],
            [-1.7, -2.0, -1.5, -1.9, -1.1],
        ]
    )
    fig, ax = plt.subplots(figsize=(10, 4.5))
    cmap = LinearSegmentedColormap.from_list("rdwhgn", ["#B2182B", "#F7F7F7", "#1A9850"])
    im = ax.imshow(data, cmap=cmap, vmin=-8, vmax=8, aspect="auto")
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains)
    ax.set_yticks(range(len(ablations)))
    ax.set_yticklabels(ablations)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:+.1f}", ha="center", va="center", fontsize=10, color="black")
    ax.set_title("Figure 4.1 — Per-Domain Accuracy Impact of Architectural Ablations (Δ acc.)")
    fig.colorbar(im, ax=ax, label="Δ accuracy (%)", shrink=0.85)
    save(fig, "fig_4_1_ablation_heatmap.png")


def fig_4_2_attention_entropy() -> None:
    layers = np.arange(1, 29)
    rng = np.random.default_rng(4)

    def curve(final_min: float, sharpness: float) -> np.ndarray:
        y = 5.0 - (5.0 - final_min) * 1 / (1 + np.exp(-(layers - 22) / sharpness))
        return y + rng.normal(0, 0.05, len(layers))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, curve(2.1, 1.2), color=PALETTE["contrastive"], lw=2.2, marker="o", markersize=4, label="DocVQA")
    ax.plot(layers, curve(2.3, 1.6), color=PALETTE["generative"], lw=2.2, marker="s", markersize=4, label="ChartQA")
    ax.plot(layers, curve(2.9, 2.5), color=PALETTE["accent"], lw=2.2, marker="^", markersize=4, label="ScienceQA")
    ax.plot(layers, curve(3.4, 3.0), color=PALETTE["purple"], lw=2.2, marker="d", markersize=4, label="A-OKVQA")
    ax.plot(layers, curve(3.7, 3.5), color=PALETTE["gold"], lw=2.2, marker="v", markersize=4, label="VQAv2")

    ax.set_xlabel("Decoder layer index")
    ax.set_ylabel("Mean attention entropy (nats)")
    ax.set_title("Figure 4.2 — Per-Layer Attention Entropy by Visual Domain")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="lower left")
    save(fig, "fig_4_2_attention_entropy.png")


# --------------------------------------------------------------------------- #
# RQ5                                                                         #
# --------------------------------------------------------------------------- #


def fig_5_1_hallucination() -> None:
    domains = ["DocVQA\n(fabr. ent.)", "ChartQA\n(fabr. ent.)", "ScienceQA\n(POPE)", "A-OKVQA\n(POPE)", "VQAv2\n(CHAIR)"]
    base = [18.4, 21.2, 14.5, 16.0, 12.8]
    expl = [9.6, 11.5, 9.8, 12.0, 8.9]

    x = np.arange(len(domains))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, base, width, color=PALETTE["baseline"], edgecolor="black", linewidth=0.6, label="Baseline generative")
    ax.bar(x + width / 2, expl, width, color=PALETTE["explanation"], edgecolor="black", linewidth=0.6, label="Explanation-aware")

    for i, (b, e) in enumerate(zip(base, expl)):
        ax.text(i + width / 2, e + 0.3, f"-{b - e:.1f}", ha="center", fontsize=9, color=PALETTE["accent"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylabel("Hallucination rate (% — lower is better)")
    ax.set_ylim(0, 26)
    ax.set_title("Figure 5.1 — Hallucination Rates: Baseline vs. Explanation-Aware")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig_5_1_hallucination.png")


def fig_5_2_qualitative() -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11, 8))

    rng = np.random.default_rng(5)

    def draw_mock_image(ax, kind):
        size = 64
        if kind == "doc":
            img = np.ones((size, size)) * 0.95
            for row in [8, 16, 24, 32, 40, 48]:
                img[row : row + 2, 6 : 58] = 0.2 + rng.random() * 0.3
            img[20:28, 34:58] = 0.1
        elif kind == "chart":
            img = np.ones((size, size)) * 0.97
            bars = [38, 22, 48, 15, 30]
            for i, h in enumerate(bars):
                img[64 - h : 60, 8 + i * 10 : 14 + i * 10] = [0.25, 0.45, 0.35, 0.55, 0.40][i]
            img[58:60, 6:58] = 0
        else:
            img = rng.random((size, size)) * 0.4 + 0.3
            img[20:44, 20:44] = 0.75
        return ax.imshow(img, cmap="gray", vmin=0, vmax=1)

    rows = [
        ("doc", "baseline", "The total is $1,240, according to the summary field."),
        ("doc", "explain", "Line 4 shows 'Total: $1,240'. Therefore the answer is $1,240."),
        ("chart", "baseline", "The bar is about 45%, which is the largest."),
        ("chart", "explain", "Bar 'C' measures 48 against the y-axis scale (0–60). Answer: 48."),
        ("img", "baseline", "A dog is running in a park."),
        ("img", "explain", "Center region shows a brown quadruped with leash; scene is grass. Answer: dog."),
    ]

    for idx, (kind, mode, text) in enumerate(rows):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        draw_mock_image(ax, kind)
        title = ("Baseline: " if mode == "baseline" else "Explanation-aware: ") + {
            "doc": "Document",
            "chart": "Chart",
            "img": "Image",
        }[kind]
        ax.set_title(title, fontsize=10, color=(PALETTE["baseline"] if mode == "baseline" else PALETTE["explanation"]))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(text, fontsize=8, wrap=True)

    fig.suptitle("Figure 5.2 — Qualitative Examples: Confabulated vs. Grounded Rationales", fontsize=12, y=1.00)
    fig.tight_layout()
    save(fig, "fig_5_2_qualitative.png")


# --------------------------------------------------------------------------- #
# RQ6                                                                         #
# --------------------------------------------------------------------------- #


def fig_6_1_pareto() -> None:
    points = [
        ("Full FT × 2B", 18.4, 78.9, PALETTE["baseline"], "o"),
        ("LoRA × 2B", 7.6, 77.6, PALETTE["generative"], "s"),
        ("QLoRA × 2B", 6.9, 77.1, PALETTE["explanation"], "^"),
        ("QLoRA × 7B", 14.2, 82.4, PALETTE["accent"], "D"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, gpuh, acc, color, marker in points:
        ax.scatter(gpuh, acc, s=260, color=color, marker=marker, edgecolor="black", linewidth=0.9, zorder=3, label=name)
        ax.annotate(name, (gpuh, acc), xytext=(8, 4), textcoords="offset points", fontsize=9)

    pareto_x = [6.9, 7.6, 14.2]
    pareto_y = [77.1, 77.6, 82.4]
    ax.plot(pareto_x, pareto_y, "--", color=PALETTE["accent"], lw=1.8, label="Pareto frontier")

    ax.set_xlabel("Training GPU-hours (A100)")
    ax.set_ylabel("Mean multi-domain accuracy (%)")
    ax.set_title("Figure 6.1 — Efficiency–Accuracy Pareto Front")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "fig_6_1_pareto.png")


def fig_6_2_scaling() -> None:
    categories = ["Mean Accuracy", "Mean Faithfulness", "Inf. tokens/sec (/10)"]
    qlora_2b = [77.1, 64.0, 4.6]
    qlora_7b = [82.4, 72.0, 2.2]

    x = np.arange(len(categories))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, qlora_2b, width, color=PALETTE["generative"], edgecolor="black", linewidth=0.6, label="Qwen2-VL-2B (QLoRA)")
    ax.bar(x + width / 2, qlora_7b, width, color=PALETTE["explanation"], edgecolor="black", linewidth=0.6, label="Qwen2-VL-7B (QLoRA)")

    for i, (a, b) in enumerate(zip(qlora_2b, qlora_7b)):
        ax.text(i - width / 2, a + 1.0, f"{a:.1f}", ha="center", fontsize=9)
        ax.text(i + width / 2, b + 1.0, f"{b:.1f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 100)
    ax.set_title("Figure 6.2 — QLoRA: 2B vs. 7B Backbone Under Identical Training")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig_6_2_scaling.png")


# --------------------------------------------------------------------------- #
# RQ7                                                                         #
# --------------------------------------------------------------------------- #


def fig_7_1_transfer_matrix() -> None:
    train_domains = ["Documents", "Charts", "Natural Images"]
    eval_domains = ["Documents", "Charts", "Natural Images"]
    data = np.array(
        [
            [83.1, 54.8, 61.2],
            [56.3, 71.8, 60.0],
            [43.5, 45.2, 75.6],
        ]
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    cmap = LinearSegmentedColormap.from_list("whblgn", ["#F7F7F7", "#A6CEE3", "#1F78B4"])
    im = ax.imshow(data, cmap=cmap, vmin=40, vmax=90)
    ax.set_xticks(range(len(eval_domains)))
    ax.set_xticklabels(eval_domains)
    ax.set_yticks(range(len(train_domains)))
    ax.set_yticklabels(train_domains)
    ax.set_xlabel("Evaluation domain")
    ax.set_ylabel("Training domain")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            color = "white" if data[i, j] > 70 else "black"
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=11, color=color, fontweight="bold" if i == j else "normal")
    ax.set_title("Figure 7.1 — Cross-Domain Transfer Matrix (Accuracy %)")
    fig.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.8)
    save(fig, "fig_7_1_transfer_matrix.png")


def fig_7_2_forgetting() -> None:
    steps = np.linspace(0, 3000, 31)
    rng = np.random.default_rng(7)

    def decay(start: float, floor: float, rate: float) -> np.ndarray:
        y = floor + (start - floor) * np.exp(-steps / rate)
        return y + rng.normal(0, 0.25, len(steps))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, decay(83.1, 58.0, 900), color=PALETTE["baseline"], lw=2.2, label="Baseline generative")
    ax.plot(steps, decay(83.1, 65.0, 1200), color=PALETTE["contrastive"], lw=2.2, label="Contrastive-enhanced")
    ax.plot(steps, decay(83.1, 73.5, 1800), color=PALETTE["explanation"], lw=2.5, label="Explanation-aware")

    ax.set_xlabel("Fine-tuning steps on new domain")
    ax.set_ylabel("Accuracy on ORIGINAL domain (DocVQA, %)")
    ax.set_title("Figure 7.2 — Catastrophic Forgetting Curve After Adapting to a New Domain")
    ax.set_ylim(50, 90)
    ax.grid(alpha=0.25, linestyle="--")
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
