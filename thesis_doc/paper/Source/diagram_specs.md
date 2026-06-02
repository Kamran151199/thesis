# Conceptual diagram specs (hand-make in Canva / PPT / draw.io → PDF, ≥300 ppi)

Three diagrams the notebook can't auto-generate (they're conceptual, not data).
Each spec below is a paint-by-numbers blueprint: layout, every label, B&W styling,
the caption, and the paper section it belongs to. Export each to
`thesis_doc/paper/Figures/<name>.pdf`.

**Global style (professor's rules):** vector → PDF; B&W-friendly (distinguish boxes
by *shape/border/hatch*, never colour alone); ❄ = frozen, 🔥 = trained; one
sans-serif font; arrows = data flow.

---

## 1. Architecture — `fig_architecture.pdf`   →  §3.2 Methodology

Purpose: show how an image + question becomes a reasoned answer, and *where the
explanation-aware loss acts*. Primary = Qwen2-VL (the workhorse); BLIP-2 inset =
the alignment backbone used for RQ1/RQ2.

```
        IMAGE  (chart · diagram · document)
          │
          ▼
   ┌───────────────────────┐   ❄ FROZEN
   │  Vision Encoder (ViT)  │   dynamic resolution
   │                        │   → 64–1280 patch tokens
   └───────────┬────────────┘
               │   inserted as:  <|vision_start|> ⟨image_pad × N⟩ <|vision_end|>
               ▼
   ┌──────────────────────────────────────────────────────┐
   │            Qwen2-VL language decoder                   │
   │   [ image tokens ]  ⊕  [ "Question: … Options: …" ]    │
   │   ┌───────────────────────────────────────────────┐   │
   │   │  N× transformer blocks                         │   │  base = 4-bit NF4  ❄
   │   │     + QLoRA adapters 🔥 (q,k,v,o · gate,up,down)│   │  LoRA r=16 trains ~1%
   │   └───────────────────────────────────────────────┘   │
   └───────────────────────────┬────────────────────────────┘
                               │  autoregressive generation (greedy)
                               ▼
        "Reasoning: plants take in CO₂ …   Answer: B."
        └────── explanation span ──────┘ └─ answer span ─┘
                               │                 │
        ┌──────────────────────┴─────────────────┴───────────────────────┐
        │  EXPLANATION-AWARE LOSS:   L = (1−α)·CE_explanation + α·CE_answer │
        │  α = answer weight ∈ {0 … 1}   (α=1 ⇒ answer-only)               │
        └─────────────────────────────────────────────────────────────────┘

   ┌─ INSET: BLIP-2 (alignment backbone, RQ1/RQ2) ───────────────────────┐
   │  Image → ViT → Q-Former (32 learned queries) → OPT LLM              │
   │  Q-Former pretrained with ITC (contrastive) · ITM (match) · ITG (gen)│
   │  → exposes pooled image/text embeddings (what Qwen2-VL lacks)        │
   └──────────────────────────────────────────────────────────────────────┘
```

Caption: *"Explanation-aware VLM architecture. A frozen vision encoder feeds image
tokens, interleaved with the question, into the Qwen2-VL decoder; only rank-16
QLoRA adapters on the 4-bit base are trained. The generated target is split into an
explanation span and an answer span, weighted by α in the explanation-aware loss.
Inset: BLIP-2's Q-Former, whose ITC pretraining gives the pooled embeddings used
for the alignment study."*

---

## 2. Methodology / pipeline — `fig_pipeline.pdf`   →  §3.2 / §3.4

Purpose: show the config-driven 4-axis framework and the run→eval→artifact flow
(one experiment = one YAML).

```
   ┌──────────── ONE EXPERIMENT = one YAML config ────────────┐
   │  BACKBONE   │ Qwen2-VL-2B · Qwen2-VL-7B · BLIP-2          │
   │  DATASET    │ ScienceQA · ChartQA · DocVQA               │
   │  OBJECTIVE  │ generative · explanation-aware(α) · contrast│
   │  METRICS    │ MC-acc · relaxed-acc · ANLS · ROUGE-L · BLEU│
   └─────────────────────────┬─────────────────────────────────┘
                             ▼
        ┌───────────────────────────────────────────────┐
        │  baseline eval  →  QLoRA fine-tune  →  final eval│
        └───────────────────────────────────────────────┘
              │                    │                    │
              ▼                    ▼                    ▼
      generate-then-score   explanation quality   faithfulness
      (CoT, greedy)         vs gold rationale      (mask region →
      → accuracy            → ROUGE-L / BLEU        answer drift)
              │                    │                    │
              └────────────────────┴────────────────────┘
                                   ▼
                results.json  +  LoRA adapter   →  Drive (persistent)
                                   ▼
              tables · α-sweep curve · faithfulness heatmaps · transfer matrix
```

Caption: *"Experimental pipeline. Each run is a single configuration over four
pluggable axes; the runner measures a baseline, fine-tunes with QLoRA, and
re-evaluates along three axes — answer accuracy (generate-then-score), explanation
quality (ROUGE-L/BLEU vs gold), and faithfulness (evidence-masking drift) —
persisting results and adapters for reproducible analysis."*

---

## 3. Graphical abstract — `fig_graphical_abstract.pdf`   →  front matter

Purpose: one-glance story — input → method → the honest finding. Punchy, minimal.

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │     CROSS-MODAL GENERATIVE ALIGNMENT for Explanation-Aware VLMs        │
 ├──────────────────────────────────────────────────────────────────────┤
 │                                                                        │
 │  ┌─chart─┐                ┌──────────────┐     Reasoning: the 2020     │
 │  │▁▃▅█▂  │  + "What is    │  VLM + QLoRA  │ ──▶ bar is tallest …        │
 │  └───────┘   the max?"    │  + α-loss 🔥  │     Answer: 42%             │
 │                           └──────┬───────┘                             │
 │                                  ▼                                     │
 │              ┌── accuracy ──┐   ┌── explanation ──┐   ┌─ faithful? ─┐  │
 │              │  +30 pp lift │   │  ROUGE / BLEU   │   │ mask→drift  │  │
 │              └──────────────┘   └─────────────────┘   └─────────────┘  │
 │                                                                        │
 │  FINDING ▸ balancing the loss (α=0.5) does NOT beat plain generative   │
 │  CE — uniform CE already over-weights the (long) explanation. We map   │
 │  the quality↔accuracy trade-off across α, and probe faithfulness.      │
 │                                                                        │
 │         [ tiny α-sweep curve thumbnail: MC-acc flat, ROUGE ↓ as α↑ ]   │
 └──────────────────────────────────────────────────────────────────────┘
```

Caption: none (graphical abstract stands alone). Keep it ≤ ~1200×1200 px, legible
at thumbnail size; the α-sweep thumbnail is a shrunk copy of `rq3_alpha_sweep.pdf`.

---

### Build order / tips
- Reuse the **real** α-sweep figure (`rq3_alpha_sweep.pdf`) as the abstract's
  thumbnail — keeps the abstract honest and consistent with §4.
- Keep all three on one Canva page (shared style), export each artboard to PDF.
- Numbers shown (42%, +30 pp) are placeholders → swap for the final results.
