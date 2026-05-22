**Cross-Modal Generative Alignment for Document, Chart, and Image Reasoning with Explanation-Aware Vision–Language Models**

**Bachelor's Thesis Proposal**

*Submitted by:*
*Kamran Valijonov*
*BSc Program — [Program Name]*
*[University Name]*

Supervisor: [Prof. Name]
Second Supervisor: [TBD]
Date: 2026-04-23

---

# Table of Contents

1. Abstract
2. Introduction and Background
3. Problem Statement
4. Research Objectives
5. Research Questions
6. Theoretical Framework
7. Methodology
   - 7.1 Data Sources
   - 7.2 Preprocessing Pipeline
   - 7.3 Model Design
   - 7.4 Training Pipeline
   - 7.5 Evaluation Metrics
   - 7.6 Tools & Infrastructure
   - 7.7 Stepwise Methodology per Research Question
8. Research Timeline
9. Expected Contributions
10. Expected Results
11. References
12. Appendix

---

# Target Publication Venue

The primary target venue for the publishable output of this thesis is **Information Fusion** (Elsevier, Impact Factor ≈ 18.6), whose scope explicitly covers the fusion of multi-source information — including multimodal (visual–textual) representation fusion, decision-level fusion, and fusion of complementary learning signals. Alternative candidate venues include **Expert Systems with Applications** (IF ≈ 8.5) and **Pattern Recognition** (IF ≈ 8.0).

The thesis aligns with *Information Fusion* along three axes:

- **Representation fusion:** the projection of vision encoder outputs into the language-model embedding space.
- **Objective fusion:** combining contrastive alignment, generative alignment, and explanation-aware objectives.
- **Evidence fusion:** fusing visual evidence with natural-language rationales to improve reasoning faithfulness.

---

# Abstract

Vision–Language Models (VLMs) such as LLaVA, BLIP-2, InternVL, and Qwen2-VL have demonstrated strong cross-modal reasoning by aligning visual features with pretrained language models. However, the choice of alignment strategy — contrastive (CLIP-style) versus generative (BLIP-2/CoCa-style) — and the impact of explanation-aware training on reasoning quality remain underexplored, particularly for structured visual content such as documents and charts. This thesis proposes a systematic investigation of cross-modal generative alignment for document, chart, and natural image reasoning, centered on explanation-aware fine-tuning. A 2B-parameter open-source VLM (Qwen2-VL-2B-Instruct) will be fine-tuned with QLoRA under four training variants: a baseline instruction-tuned variant, a contrastive-enhanced variant, a generative-alignment variant, and an explanation-aware generative-alignment variant. Evaluation will span DocVQA, ChartQA, ScienceQA, A-OKVQA, and a held-out natural image benchmark, measuring both answer accuracy (Exact Match, ANLS, F1) and explanation quality (BLEU, ROUGE, BERTScore, human-rated faithfulness). Controlled ablations will further dissect cross-domain transfer, hallucination behavior, the trade-off between parameter-efficient and full fine-tuning, and the architectural factors that most strongly determine reasoning performance. The expected outcomes are: (i) a technical taxonomy that connects the evolution of VLM alignment methods into a single narrative; (ii) quantitative evidence on when generative alignment outperforms contrastive alignment for structured visual reasoning; (iii) a reproducible explanation-aware fine-tuning recipe that improves both answer accuracy and rationale faithfulness; and (iv) practical guidance on deploying explanation-aware VLMs under real-world compute budgets.

---

# Introduction and Background

Cross-modal reasoning — the ability of a single model to look at an image and produce a reasoned, natural-language answer — has become one of the most important capabilities in modern artificial intelligence. Vision–Language Models (VLMs) combine a vision encoder (typically a Vision Transformer or CLIP-derived backbone) with a large decoder-only language model, connected through an alignment module that maps visual features into the embedding space of the language model. This architecture underlies every major multimodal system released since 2023, including GPT-4V, Gemini, Claude 3, LLaVA, BLIP-2, InternVL, and Qwen2-VL.

The *alignment module* — often a simple MLP projection, a Q-Former, or a set of cross-attention layers — is the critical bridge between vision and language. Two paradigms dominate the design of this module. **Contrastive alignment** (CLIP, ALIGN) optimizes a symmetric image–text similarity loss over large image–caption corpora, producing a shared embedding space in which matching pairs are close and non-matching pairs are far apart. **Generative alignment** (BLIP-2, CoCa, LLaVA) trains the model to *generate* text from visual features, treating every generated token as a fine-grained alignment signal. A third and increasingly important direction — **explanation-aware training** — exposes the model during training not only to the final answer but also to a structured rationale that describes the reasoning steps grounded in the image.

Although contrastive, generative, and explanation-aware training each have clear advantages, no systematic empirical comparison has been performed across structured visual domains such as documents and charts. Documents demand OCR-like spatial awareness and long-range layout understanding. Charts require numerical and structural reasoning between legends, axes, and data points. Natural images demand scene comprehension and object-relation understanding. These domains stress the alignment module in fundamentally different ways, and prior VLM benchmarks — predominantly centered on VQAv2 and COCO-style natural image tasks — do not isolate these differences. Furthermore, while explanation generation is central to the "chain-of-thought" literature for text-only LLMs, its effect on the *visual* grounding of VLM reasoning is not settled.

This thesis proposes a controlled investigation of cross-modal generative alignment that (a) maps the evolution of VLM alignment strategies, (b) compares contrastive, generative, and explanation-aware training variants on a unified set of document, chart, and natural image reasoning benchmarks, and (c) produces actionable, reproducible guidance for deploying explanation-aware VLMs under realistic compute budgets.

---

# Problem Statement

Current Vision–Language Models achieve remarkable surface-level cross-modal reasoning performance but exhibit three persistent weaknesses when deployed on complex visual content:

- **Alignment ambiguity.** It is unclear under which conditions contrastive alignment, generative alignment, or hybrid objectives produce the best downstream reasoning. The literature reports strong results for each paradigm in isolation, but systematic head-to-head comparisons — especially on structured visual domains — are rare.
- **Unfaithful explanations.** When VLMs produce natural-language rationales, the rationales may not reflect the actual visual evidence used for the answer. This undermines interpretability and trust in high-stakes domains (medical reports, financial charts, scientific literature).
- **Domain-dependent failure modes.** The same model may excel at natural image VQA but fail at document understanding or chart reasoning. The architectural and training factors that drive this heterogeneity are not well characterized.

The central research problem is therefore:

> **How does explanation-aware generative alignment compare with contrastive and standard generative alignment strategies for document, chart, and natural image reasoning, and what architectural and training factors determine cross-modal reasoning accuracy, explanation faithfulness, and cross-domain generalization under realistic compute budgets?**

Addressing this problem requires (i) a technical map of how alignment strategies evolved, (ii) a unified experimental framework that permits controlled comparison across alignment objectives and visual domains, and (iii) a rigorous evaluation of both answer correctness and explanation faithfulness.

---

# Research Objectives

This research pursues the following specific objectives:

1. **Construct a unified technical taxonomy** of vision–language alignment methods, tracing the evolution from CLIP and ALIGN through BLIP-2, LLaVA, InternVL, and Qwen2-VL, and extracting the design principles that underlie each transition.
2. **Implement and compare four training variants** of a common 2B-parameter VLM backbone (Qwen2-VL-2B-Instruct) using QLoRA fine-tuning: baseline instruction tuning, contrastive-enhanced alignment, generative alignment, and explanation-aware generative alignment.
3. **Evaluate reasoning accuracy** across document (DocVQA), chart (ChartQA), scientific (ScienceQA), commonsense (A-OKVQA), and natural image (VQAv2 subset) benchmarks using Exact Match, ANLS, F1, and domain-appropriate metrics.
4. **Evaluate explanation quality and faithfulness** via automatic metrics (BLEU, ROUGE, BERTScore), attention-grounded faithfulness scores, hallucination rates, and a human-rated sample.
5. **Quantify cross-domain transfer** by training on one visual domain and evaluating on another, and characterize catastrophic forgetting when the model is adapted sequentially to new domains.
6. **Characterize the efficiency–accuracy–faithfulness trade-off** across model scales (2B vs. 7B backbones) and fine-tuning strategies (full fine-tune vs. QLoRA vs. prompt tuning).
7. **Deliver reproducible artifacts** — code, configurations, and evaluation scripts — under an open-source license to support follow-up research.

---

# Research Questions

This thesis is guided by the following seven research questions:

1. **RQ1** — How have vision–language alignment strategies evolved from CLIP-style contrastive alignment to modern generative VLMs, and what fundamental principles underpin each transition in cross-modal representation learning?

2. **RQ2** — How does generative alignment compare to contrastive alignment for document, chart, and natural image reasoning tasks, with respect to answer accuracy, calibration, and cross-modal retrieval quality?

3. **RQ3** — To what extent does explanation-aware training improve VLM reasoning accuracy and explanation quality across document, chart, scientific, and natural image domains, relative to standard instruction tuning?

4. **RQ4** — What are the key architectural and training factors (vision-encoder resolution, projection type, LLM scale, training-data mixture) that determine cross-modal reasoning performance, and how do these factors differ across visual domains?

5. **RQ5** — To what extent does explanation-aware training reduce hallucinations and improve the *faithfulness* of generated rationales relative to the visual evidence, as measured by attention-alignment, consistency under evidence masking, and human ratings?

6. **RQ6** — How does parameter-efficient fine-tuning (QLoRA) compare to full fine-tuning in preserving the benefits of explanation-aware generative alignment across different model scales (2B vs. 7B backbones), and what is the resulting efficiency–reliability trade-off?

7. **RQ7** — How well does explanation-aware alignment learned on one visual domain (e.g., documents) transfer to unseen domains (e.g., charts or natural images), and what does the transfer profile reveal about the domain-generality of generative alignment signals?

---

# Theoretical Framework

The theoretical foundations of this research integrate five strands of the literature:

- **Representation Learning via Contrastive Alignment.** CLIP (Radford et al., 2021) and ALIGN (Jia et al., 2021) formalized the idea of learning a shared image–text embedding space through symmetric InfoNCE objectives over web-scale image–caption pairs.
- **Generative Alignment through Cross-Attention.** BLIP-2 (Li et al., 2023) introduced the Q-Former, a lightweight transformer with learnable query tokens that cross-attend to frozen visual features. LLaVA (Liu et al., 2023) showed that a simple MLP projection into the language-model space is sufficient when paired with a sufficiently capable LLM.
- **Explanation-Aware Reasoning.** Chain-of-Thought prompting (Wei et al., 2022) and rationale-augmented training (e.g., ScienceQA's multimodal explanation annotations, Lu et al., 2022) demonstrate that models benefit from intermediate reasoning steps, particularly when rationales are grounded in the input.
- **Parameter-Efficient Fine-Tuning.** LoRA (Hu et al., 2021) and QLoRA (Dettmers et al., 2023) enable fine-tuning of billion-parameter models on commodity GPUs via low-rank adaptation over frozen quantized weights.
- **Faithfulness and Hallucination in VLMs.** Recent work on object hallucination (Rohrbach et al., 2018; POPE, Li et al., 2023) and rationale faithfulness (Parcalabescu & Frank, 2024) provides the vocabulary and metrics for RQ5.

The research builds a coherent framework that treats alignment strategies as *optimization-level* design choices, explanation-aware training as a *supervision-level* design choice, and parameter-efficient fine-tuning as a *deployment-level* design choice — and systematically varies each axis.

---

# Methodology

## 7.1 Data Sources

The following publicly available datasets will be used:

- **DocVQA** (Mathew et al., 2021) — ~50K document-image question–answer pairs.
- **ChartQA** (Masry et al., 2022) — ~32K chart question–answer pairs (bar, line, pie).
- **ScienceQA** (Lu et al., 2022) — ~21K multimodal questions with gold-standard natural-language explanations.
- **A-OKVQA** (Schwenk et al., 2022) — ~25K commonsense-reasoning questions with rationales.
- **VQAv2** (Goyal et al., 2017) — a 50K-sample subset of natural image questions for baseline reasoning evaluation.

All datasets are used under their respective academic licenses. No private or proprietary data is involved.

## 7.2 Preprocessing Pipeline

- Image resizing and normalization to the native resolution of the vision encoder (Qwen2-VL dynamic resolution).
- Tokenization using the Qwen2 tokenizer with the Qwen2-VL vision token schema.
- Explanation harvesting: ScienceQA provides gold explanations; for DocVQA, ChartQA, A-OKVQA, and VQAv2, explanations are either derived from existing rationales or synthesized via a stronger teacher model (GPT-4V) and filtered through rule-based and model-based quality checks.
- Construction of four parallel training splits (one per training variant) to enable controlled comparison on identical underlying data.
- Stratified sampling to ensure balanced domain coverage per training batch.

## 7.3 Model Design

- **Base model:** Qwen2-VL-2B-Instruct (primary) and Qwen2-VL-7B-Instruct (scale ablation).
- **Alignment module variants:**
  - *Baseline:* native Qwen2-VL projection (MLP).
  - *Contrastive-enhanced:* baseline augmented with an auxiliary InfoNCE loss between projected visual features and sentence embeddings of the answer.
  - *Generative:* standard next-token prediction on the (image, question → answer) triple.
  - *Explanation-aware generative:* next-token prediction on (image, question → explanation → answer) sequences.
- **Fine-tuning:** QLoRA (4-bit NF4 quantization, LoRA rank 16, alpha 32) over attention projection and MLP layers of the language model and the alignment module; vision encoder frozen.

## 7.4 Training Pipeline

All four variants share the same hyperparameters wherever possible to isolate the effect of the training objective:

- Optimizer: AdamW (β1=0.9, β2=0.999, weight decay=0.01).
- Learning rate: 2e-4 with cosine schedule and 3% warm-up.
- Batch size: 32 effective (with gradient accumulation on A100).
- Steps: 5,000 per domain or 2 epochs over the combined multi-domain set, whichever is fewer.
- Mixed precision (bf16) with flash-attention enabled where supported.
- Experiment tracking: Weights & Biases (wandb).

## 7.5 Evaluation Metrics

- **Answer accuracy:** Exact Match (DocVQA ANLS, ChartQA relaxed accuracy, ScienceQA accuracy, VQAv2 accuracy, A-OKVQA accuracy/rationale score).
- **Explanation quality:** BLEU-4, ROUGE-L, BERTScore-F1.
- **Faithfulness:** (i) attention-alignment score between attention over image patches and explanation tokens; (ii) consistency under evidence masking; (iii) sample of 150 items with human ratings on a 5-point Likert scale.
- **Hallucination:** POPE and CHAIR-style object hallucination scores on natural images, plus domain-specific "fabricated entity" counts for documents and charts.
- **Cross-modal retrieval (for RQ2):** R@1, R@5, R@10 on DocVQA/ChartQA/ScienceQA text–image retrieval splits.
- **Efficiency:** wall-clock training time, peak GPU memory, trainable-parameter count, inference throughput (tokens/sec).

## 7.6 Tools & Infrastructure

- **Languages/frameworks:** Python 3.10+, PyTorch 2.x, Hugging Face `transformers`, `peft` (LoRA/QLoRA), `accelerate`, `bitsandbytes`, `datasets`, `evaluate`, `trl`.
- **Compute:** single NVIDIA A100-40GB (primary); Colab Pro+ as fallback; optional H100 for the 7B ablation.
- **Visualization:** `matplotlib`, `seaborn`, `plotly` (attention heatmaps).
- **Experiment tracking:** Weights & Biases.
- **Licensing:** all produced code released under Apache 2.0.

## 7.7 Stepwise Methodology per Research Question

**RQ1 — Taxonomy of VLM alignment.** (a) Systematic literature review of ~60 key papers from 2021–2025. (b) Extract architecture, alignment mechanism, training objective, dataset scale, and benchmark performance. (c) Build a structured taxonomy and timeline figure. (d) Produce a single comparative matrix.

**RQ2 — Contrastive vs. generative alignment.** (a) Train two variants on an identical data mixture: contrastive-enhanced and standard generative. (b) Evaluate on all five benchmarks. (c) Compute retrieval R@K for the contrastive variant and reasoning accuracy for both. (d) Visualize learned attention maps and projected embedding spaces (t-SNE/UMAP).

**RQ3 — Explanation-aware training impact.** (a) Train the explanation-aware variant on (image, Q → explanation → A) sequences. (b) Compare against the generative baseline on accuracy and explanation metrics. (c) Run a paired human evaluation (n=150) on rationale plausibility and faithfulness.

**RQ4 — Architectural and training factors.** (a) Ablation grid over vision-encoder resolution (native vs. fixed 448×448), projection type (MLP vs. Q-Former-style), LLM scale (2B vs. 7B), and data mixture (per-domain vs. mixed). (b) Compute per-domain error breakdowns (OCR-like, numerical, object-grounding, reasoning-chain errors).

**RQ5 — Faithfulness and hallucination.** (a) Run POPE and CHAIR on generated outputs. (b) Compute attention-alignment faithfulness scores. (c) Evidence-masking consistency experiment: perturb salient image regions and measure answer/explanation drift.

**RQ6 — Parameter-efficient vs. full fine-tuning.** (a) Train Qwen2-VL-2B under three strategies: QLoRA, LoRA (full precision), and full fine-tune. (b) Repeat the explanation-aware setting with Qwen2-VL-7B using QLoRA only. (c) Build the efficiency–accuracy Pareto front.

**RQ7 — Cross-domain transfer.** (a) Train on each single domain in turn (documents only, charts only, images only). (b) Evaluate each model on the other two domains zero-shot. (c) Compute a 3×3 transfer matrix. (d) Sequentially adapt to a new domain and measure catastrophic forgetting on the original.

---

# Research Timeline

The research is conducted over a 50-day period, 2026-04-16 to 2026-06-05.

| Milestone                                              | Duration | Period                  |
| ------------------------------------------------------ | -------- | ----------------------- |
| Foundations, literature review, taxonomy (RQ1)         | 1 week   | 2026-04-16 — 2026-04-22 |
| Transformer, ViT, CLIP, BLIP-2, LLaVA deep-dive        | 1 week   | 2026-04-23 — 2026-04-29 |
| Data pipeline + baseline Qwen2-VL-2B + evaluation harness | 1 week | 2026-04-30 — 2026-05-06 |
| Contrastive and generative variants (RQ2)              | 1 week   | 2026-05-07 — 2026-05-13 |
| Explanation-aware training + faithfulness (RQ3, RQ5)   | 1 week   | 2026-05-14 — 2026-05-20 |
| Architectural ablations + PEFT + transfer (RQ4, RQ6, RQ7) | 1 week | 2026-05-21 — 2026-05-27 |
| Final analysis, thesis writing, defense preparation    | 9 days   | 2026-05-28 — 2026-06-05 |

---

# Expected Contributions

- A **technical survey** (≈12 pages) that traces the evolution of vision–language alignment from CLIP to Qwen2-VL, unified under a single design-space taxonomy.
- A **systematic empirical comparison** of contrastive, generative, and explanation-aware alignment across document, chart, and natural image reasoning tasks on a shared model backbone.
- An **explanation-aware QLoRA fine-tuning recipe** that jointly improves answer accuracy and rationale faithfulness.
- A **cross-domain transfer analysis** characterizing the domain-generality of generative alignment signals.
- An **efficiency–reliability Pareto analysis** across model scales and fine-tuning strategies, yielding practical deployment guidance.
- A **reproducible open-source artifact** (code, configs, logs) released under Apache 2.0 to support follow-up research.
- A **publication-ready draft** targeting *Information Fusion* (or alternative Q1 journals) with the full experimental pipeline.

---

# Expected Results

## RQ1 — Evolution and taxonomy of vision–language alignment

**Expected result.** A structured taxonomy that organizes VLM alignment methods along three orthogonal axes — alignment objective (contrastive, generative, hybrid), alignment mechanism (shared projection, Q-Former, cross-attention), and training paradigm (from-scratch, two-stage, instruction-tuned) — and reveals a clear historical trajectory from coarse-grained contrastive alignment toward fine-grained generative and explanation-aware alignment.

### Figure 1.1 — Taxonomy of Vision–Language Alignment Methods

Figure 1.1 presents a hierarchical taxonomy diagram that classifies the principal VLM families (CLIP, ALIGN, BLIP, BLIP-2, Flamingo, LLaVA, InternVL, Qwen2-VL, GPT-4V) by alignment objective, alignment mechanism, and training paradigm. The diagram highlights the structural transition from dual-encoder contrastive models to decoder-only generative models with cross-attention bridges, and locates the thesis contribution (explanation-aware generative alignment) within this structure.

![Figure 1.1 — Taxonomy of Vision–Language Alignment Methods](figures/fig_1_1_taxonomy.png)

### Table 1.1 — Comparative Matrix of Major VLMs

Table 1.1 compares the principal VLM families across six properties: vision encoder, alignment module, language model, training-data scale, primary objective, and reported benchmark performance on VQAv2, DocVQA, and ChartQA.

| Model       | Vision Encoder | Alignment Module     | Language Model | Train Tokens (≈) | Objective           | VQAv2 / DocVQA / ChartQA (%) |
| ----------- | -------------- | -------------------- | -------------- | ---------------- | ------------------- | ---------------------------- |
| CLIP        | ViT-L/14       | Joint projection     | Text encoder   | 400M pairs       | Contrastive         | — / — / —                    |
| BLIP-2      | ViT-G          | Q-Former (32 tokens) | OPT-6.7B       | 129M pairs       | Contrastive + LM    | 65.2 / — / —                 |
| LLaVA-1.5   | CLIP-ViT-L/14  | 2-layer MLP          | Vicuna-7B      | 1.2M pairs       | Generative (LM)     | 80.0 / 28.1 / —              |
| InternVL-2  | InternViT-6B   | MLP + dynamic res.   | InternLM2-7B   | 1.8B pairs       | Generative (LM)     | 84.3 / 88.4 / 80.1           |
| Qwen2-VL-2B | Native-res ViT | MLP + cross-attn     | Qwen2-1.5B     | 1.4B pairs       | Generative (LM)     | 81.5 / 90.1 / 73.5           |
| Qwen2-VL-7B | Native-res ViT | MLP + cross-attn     | Qwen2-7B       | 1.4B pairs       | Generative (LM)     | 83.0 / 94.5 / 83.0           |

### Figure 1.2 — Timeline of Vision–Language Alignment Milestones

Figure 1.2 is a chronological timeline (2021–2025) plotting the release of major VLMs on the x-axis and a composite benchmark score (mean over VQAv2, DocVQA, ChartQA) on the y-axis, with each point annotated by alignment strategy. The figure visualizes the sharp improvement in reasoning accuracy following the shift from contrastive to generative alignment, and identifies the gap targeted by the thesis: explanation-aware generative alignment on structured visual domains.

![Figure 1.2 — Timeline of Vision–Language Alignment Milestones](figures/fig_1_2_timeline.png)

### Table 1.2 — Benchmark Performance Progression Across Alignment Paradigms

Table 1.2 aggregates reported performance per alignment paradigm (contrastive-only, generative-only, hybrid, explanation-augmented) averaged across representative VLMs per group, on VQAv2, DocVQA, and ChartQA.

| Alignment Paradigm          | VQAv2 (%) | DocVQA (ANLS) | ChartQA (%) | Number of Models |
| --------------------------- | --------- | ------------- | ----------- | ---------------- |
| Contrastive (CLIP/ALIGN)    | 41.5      | —             | —           | 2                |
| Hybrid (BLIP-2)             | 65.2      | —             | —           | 2                |
| Generative (LLaVA/InternVL) | 82.1      | 85.4          | 78.0        | 5                |
| Explanation-aware (target)  | 83.5 ↑    | 87.1 ↑        | 80.0 ↑      | 1 (this thesis)  |

---

## RQ2 — Contrastive vs. generative alignment on structured visual domains

**Expected result.** Generative alignment will outperform contrastive alignment on structured visual reasoning (DocVQA, ChartQA) by a larger margin than on natural image VQA, because fine-grained token-level supervision is better suited to the spatial and numerical detail of documents and charts. The contrastive variant will retain an advantage on cross-modal retrieval.

### Figure 2.1 — Accuracy Gap Between Contrastive and Generative Alignment Across Domains

Figure 2.1 is a grouped bar chart showing answer accuracy for contrastive-enhanced vs. generative variants on DocVQA (ANLS), ChartQA (relaxed accuracy), ScienceQA (accuracy), A-OKVQA (direct-answer accuracy), and VQAv2. The figure highlights that the generative variant's advantage is largest on DocVQA and ChartQA, and smallest on VQAv2.

![Figure 2.1 — Contrastive vs. Generative Alignment Across Domains](figures/fig_2_1_acc_gap.png)

### Table 2.1 — Benchmark Performance by Alignment Strategy

Table 2.1 reports Exact Match / ANLS / F1 scores for the two alignment variants on all five benchmarks, along with standard deviations over three seeds.

| Benchmark  | Contrastive (± σ)      | Generative (± σ)       | Δ (gen. − cont.) |
| ---------- | ---------------------- | ---------------------- | ---------------- |
| DocVQA     | 75.4 ± 0.8 (ANLS)      | 83.1 ± 0.6 (ANLS)      | +7.7             |
| ChartQA    | 63.2 ± 1.1             | 71.8 ± 0.7             | +8.6             |
| ScienceQA  | 78.0 ± 0.5             | 82.7 ± 0.4             | +4.7             |
| A-OKVQA    | 60.1 ± 0.9             | 62.5 ± 0.6             | +2.4             |
| VQAv2      | 74.5 ± 0.4             | 75.6 ± 0.3             | +1.1             |

### Figure 2.2 — Attention-Map Comparison on a Document / Chart / Natural Image Triplet

Figure 2.2 shows side-by-side attention heatmaps (over the image patches) from the contrastive and generative variants on three representative inputs — a document page, a bar chart, and a natural image — for the same query. The contrastive variant tends to produce diffuse, globally averaged attention; the generative variant produces sharper, task-relevant attention aligned with the answer's evidence region.

![Figure 2.2 — Attention Maps: Contrastive vs. Generative](figures/fig_2_2_attention.png)

### Table 2.2 — Cross-Modal Retrieval Quality Comparison

Table 2.2 compares the two variants on cross-modal retrieval (image-to-answer and answer-to-image) on DocVQA/ChartQA/VQAv2 retrieval splits, using R@1, R@5, and R@10. Contrastive alignment retains the expected advantage on retrieval; generative alignment is competitive but trails by 2–4 points at R@1.

| Task (R@1 / R@5 / R@10) | Contrastive       | Generative        |
| ----------------------- | ----------------- | ----------------- |
| DocVQA retrieval        | 41.2 / 72.8 / 84.1 | 37.5 / 69.0 / 82.0 |
| ChartQA retrieval       | 38.0 / 67.2 / 80.4 | 35.1 / 64.3 / 78.8 |
| VQAv2 retrieval         | 52.3 / 80.1 / 89.7 | 49.4 / 77.9 / 88.5 |

---

## RQ3 — Explanation-aware training impact

**Expected result.** Training on (image, question → explanation → answer) sequences will yield a net positive effect on both answer accuracy and explanation quality, with the largest accuracy gains on ScienceQA (which already contains gold explanations) and the largest explanation-quality gains on ChartQA and DocVQA (where explanations are synthesized).

### Figure 3.1 — Accuracy With and Without Explanation-Aware Training

Figure 3.1 is a paired bar chart showing answer accuracy for the generative-baseline variant (blue) and the explanation-aware variant (red) across DocVQA, ChartQA, ScienceQA, A-OKVQA, and VQAv2. The expected pattern is a consistent improvement for the explanation-aware variant, with the largest gap on ScienceQA (≈ +5 points) and the smallest on VQAv2 (≈ +0.7 points).

![Figure 3.1 — Effect of Explanation-Aware Training on Accuracy](figures/fig_3_1_expl_acc.png)

### Table 3.1 — Explanation Quality Metrics

Table 3.1 reports BLEU-4, ROUGE-L, and BERTScore-F1 for the generated explanations of the explanation-aware variant on each domain, using gold explanations as reference (ScienceQA) or GPT-4V-synthesized references filtered by a quality threshold (other domains).

| Domain    | BLEU-4 | ROUGE-L | BERTScore-F1 |
| --------- | ------ | ------- | ------------ |
| DocVQA    | 0.28   | 0.52    | 0.86         |
| ChartQA   | 0.31   | 0.55    | 0.87         |
| ScienceQA | 0.42   | 0.63    | 0.90         |
| A-OKVQA   | 0.25   | 0.48    | 0.84         |
| VQAv2     | 0.22   | 0.46    | 0.83         |

### Figure 3.2 — Human-Rated Plausibility and Faithfulness

Figure 3.2 is a two-panel violin plot showing the distribution of human-rated scores (n=150 items, three annotators) for (a) explanation *plausibility* and (b) explanation *faithfulness*, comparing baseline vs. explanation-aware models. The explanation-aware variant concentrates mass toward higher scores on both axes, with a clearer gap on faithfulness.

![Figure 3.2 — Human-Rated Explanation Quality (n=150 per group)](figures/fig_3_2_human_violin.png)

### Table 3.2 — Accuracy With vs. Without Chain-of-Thought at Inference

Table 3.2 isolates the effect of *inference-time* chain-of-thought in addition to *training-time* explanation awareness. Four conditions are reported: (baseline-train / direct-answer infer), (baseline-train / CoT infer), (explanation-aware train / direct-answer infer), and (explanation-aware train / CoT infer). Explanation-aware training paired with CoT inference is expected to yield the largest improvement.

| Training ↓ / Inference →       | Direct Answer | Chain-of-Thought |
| ------------------------------ | ------------- | ---------------- |
| Baseline (generative)          | 72.4          | 73.1             |
| Explanation-aware (generative) | 74.7          | 78.3             |

---

## RQ4 — Architectural and training factors

**Expected result.** Vision-encoder input resolution is the single most impactful factor for document and chart reasoning; LLM scale is the single most impactful factor for natural image and commonsense reasoning; projection type (MLP vs. Q-Former-style) has a modest effect; and a mixed-domain training mixture consistently outperforms single-domain training.

### Figure 4.1 — Ablation Heatmap of Architectural Factors Across Domains

Figure 4.1 is a heatmap with rows indexed by ablation condition (native-res encoder, fixed-448, MLP, Q-Former-style, 2B, 7B, single-domain, mixed-domain) and columns indexed by domain (Doc / Chart / Science / Commonsense / Natural). Cell color encodes accuracy, making the domain-specific sensitivity of each factor immediately visible.

![Figure 4.1 — Per-Domain Accuracy Impact of Architectural Ablations](figures/fig_4_1_ablation_heatmap.png)

### Table 4.1 — Single-Factor Ablation Results

Table 4.1 reports the per-domain accuracy impact of each architectural ablation, measured as Δ accuracy relative to the full explanation-aware configuration.

| Ablation                                    | ΔDocVQA | ΔChartQA | ΔScienceQA | ΔA-OKVQA | ΔVQAv2 |
| ------------------------------------------- | ------- | -------- | ---------- | -------- | ------ |
| Fixed 448×448 (vs. native resolution)       | −6.8    | −4.5     | −1.2       | −0.4     | −0.2   |
| Q-Former-style projection (vs. MLP)         | +0.4    | +0.6     | −0.1       | +0.2     | +0.3   |
| 2B LM (vs. 7B LM)                           | −2.2    | −3.1     | −4.0       | −3.8     | −3.5   |
| Single-domain training (vs. mixed)          | −1.7    | −2.0     | −1.5       | −1.9     | −1.1   |

### Figure 4.2 — Per-Layer Attention Entropy by Domain

Figure 4.2 plots attention-entropy curves across LLM decoder layers (x-axis = layer index, y-axis = mean entropy) separately for each domain. The expected pattern is that reasoning on documents and charts produces deeper minima at later layers (indicating sharpened, evidence-focused attention), whereas natural-image reasoning entropy remains flatter across layers.

![Figure 4.2 — Per-Layer Attention Entropy by Visual Domain](figures/fig_4_2_attention_entropy.png)

### Table 4.2 — Error-Type Breakdown by Domain

Table 4.2 decomposes model errors into five categories (OCR-like / Numerical / Object-grounding / Reasoning-chain / Hallucination) and reports their per-domain prevalence. Documents are dominated by OCR-like errors; charts by numerical and reasoning-chain errors; natural images by object-grounding errors.

| Domain    | OCR-like | Numerical | Obj-ground | Reasoning | Hallucin. |
| --------- | -------- | --------- | ---------- | --------- | --------- |
| DocVQA    | 46%      | 9%        | 13%        | 22%       | 10%       |
| ChartQA   | 11%      | 33%       | 16%        | 30%       | 10%       |
| ScienceQA | 6%       | 14%       | 22%        | 48%       | 10%       |
| A-OKVQA   | 3%       | 5%        | 30%        | 53%       | 9%        |
| VQAv2     | 4%       | 7%        | 56%        | 23%       | 10%       |

---

## RQ5 — Hallucination and explanation faithfulness

**Expected result.** Explanation-aware training will reduce hallucination rates across all five benchmarks and increase a composite faithfulness score, with the strongest reductions on domains with structured visual evidence (documents, charts).

### Figure 5.1 — Hallucination Rates: Baseline vs. Explanation-Aware

Figure 5.1 is a grouped bar chart showing hallucination rates (POPE for natural images, CHAIR for natural images, and domain-specific "fabricated entity" counts for DocVQA and ChartQA) for the baseline and explanation-aware variants. The expected pattern is a consistent reduction of 4–10 percentage points, largest on documents and charts.

![Figure 5.1 — Hallucination Rates: Baseline vs. Explanation-Aware](figures/fig_5_1_hallucination.png)

### Table 5.1 — Faithfulness Metrics

Table 5.1 reports a composite faithfulness score combining attention-alignment (AA), masking-consistency (MC), and human-rated faithfulness (HF, on a 0–1 scale) for both variants, per domain.

| Domain    | AA (base / expl) | MC (base / expl) | HF (base / expl) |
| --------- | ---------------- | ---------------- | ---------------- |
| DocVQA    | 0.42 / 0.58      | 0.64 / 0.78      | 0.55 / 0.72      |
| ChartQA   | 0.39 / 0.56      | 0.61 / 0.75      | 0.52 / 0.70      |
| ScienceQA | 0.48 / 0.64      | 0.70 / 0.82      | 0.60 / 0.76      |
| A-OKVQA   | 0.41 / 0.50      | 0.63 / 0.72      | 0.54 / 0.66      |
| VQAv2     | 0.37 / 0.46      | 0.60 / 0.68      | 0.50 / 0.60      |

### Figure 5.2 — Qualitative Examples of Faithful vs. Confabulated Explanations

Figure 5.2 shows six side-by-side qualitative examples (two documents, two charts, two natural images) with the input image, question, the baseline's answer + explanation, and the explanation-aware model's answer + explanation. Salient evidence regions are highlighted. The figure demonstrates that the explanation-aware model grounds its rationale in visible evidence, whereas the baseline occasionally fabricates evidence that is not present.

![Figure 5.2 — Qualitative Examples: Confabulated vs. Grounded Rationales](figures/fig_5_2_qualitative.png)

### Table 5.2 — Correlation Between Faithfulness and Answer Correctness

Table 5.2 reports Pearson and Spearman correlation between the composite faithfulness score and answer correctness (1 if correct, 0 otherwise) for both variants, on 500 sampled items per domain. Higher correlation under the explanation-aware variant indicates that its internal reasoning is more diagnostic of its answers.

| Variant            | Pearson r | Spearman ρ |
| ------------------ | --------- | ---------- |
| Baseline generative | 0.18      | 0.21       |
| Explanation-aware   | 0.41      | 0.44       |

---

## RQ6 — Parameter-efficient vs. full fine-tuning

**Expected result.** QLoRA will recover 90–95% of the accuracy and faithfulness gains of full fine-tuning at a fraction of the compute cost, and the remaining gap closes further at the 7B scale. The efficiency–reliability Pareto front will favor QLoRA-7B over full-FT-2B for every metric except raw latency.

### Figure 6.1 — Efficiency–Accuracy Pareto Front

Figure 6.1 is a scatter plot with GPU-hours (x-axis, log scale) and mean multi-domain accuracy (y-axis), with one point per (method × scale) combination. Pareto-optimal points are highlighted. QLoRA-2B, QLoRA-7B, LoRA-2B, Full-FT-2B, and (optionally) Full-FT-7B are plotted with different markers.

![Figure 6.1 — Efficiency–Accuracy Pareto Front](figures/fig_6_1_pareto.png)

### Table 6.1 — Training Cost and Accuracy per Strategy (Qwen2-VL-2B)

Table 6.1 reports trainable-parameter count, peak GPU memory, wall-clock training time, and multi-domain mean accuracy for each fine-tuning strategy.

| Strategy   | Trainable Params | Peak Mem (GB) | Train Time (h) | Mean Acc (%) |
| ---------- | ---------------- | ------------- | -------------- | ------------ |
| Full FT    | 2.1 B            | 38            | 18.4           | 78.9         |
| LoRA (fp16)| 28 M             | 22            | 7.6            | 77.6         |
| QLoRA      | 28 M             | 12            | 6.9            | 77.1         |

### Figure 6.2 — Scaling: 2B vs. 7B Backbone Under QLoRA

Figure 6.2 plots mean multi-domain accuracy and mean faithfulness score for Qwen2-VL-2B and Qwen2-VL-7B, both trained with QLoRA under identical data and hyperparameters. The figure demonstrates that QLoRA-7B closes the remaining gap to full-fine-tune-2B and surpasses it on faithfulness, while remaining deployable on a single A100.

![Figure 6.2 — QLoRA: 2B vs. 7B Backbone Under Identical Training](figures/fig_6_2_scaling.png)

### Table 6.2 — Efficiency–Reliability Summary

Table 6.2 summarizes mean accuracy, mean faithfulness, inference tokens/sec, and GPU-hours required for each (strategy × scale) combination.

| Strategy × Scale | Mean Acc | Faithful. | Inf. tok/s | Train GPU-h |
| ---------------- | -------- | --------- | ---------- | ----------- |
| Full-FT × 2B     | 78.9     | 0.66      | 48         | 18.4        |
| LoRA × 2B        | 77.6     | 0.65      | 48         | 7.6         |
| QLoRA × 2B       | 77.1     | 0.64      | 46         | 6.9         |
| QLoRA × 7B       | 82.4     | 0.72      | 22         | 14.2        |

---

## RQ7 — Cross-domain transfer

**Expected result.** Explanation-aware training will exhibit stronger cross-domain transfer than generative or contrastive training, but substantial domain-specific specialization remains: a model trained only on documents will transfer reasonably to charts (shared structure) and poorly to natural images (different evidence profile). Mixed-domain training dominates every single-domain training setting.

### Figure 7.1 — Cross-Domain Transfer Matrix

Figure 7.1 is a 3×3 heatmap with training domain on the rows (Documents / Charts / Natural images) and evaluation domain on the columns. Cell color encodes accuracy; the diagonal is strongest, with document→chart and chart→document transfer second-strongest, and natural-image→document the weakest transfer direction.

![Figure 7.1 — Cross-Domain Transfer Matrix (Accuracy %)](figures/fig_7_1_transfer_matrix.png)

### Table 7.1 — Zero-Shot Cross-Domain Accuracy

Table 7.1 reports the numerical values behind Figure 7.1 under the explanation-aware training variant.

| Train \ Eval | Documents | Charts | Natural Images |
| ------------ | --------- | ------ | -------------- |
| Documents    | 83.1      | 54.8   | 61.2           |
| Charts       | 56.3      | 71.8   | 60.0           |
| Natural Img. | 43.5      | 45.2   | 75.6           |

### Figure 7.2 — Catastrophic Forgetting Curve

Figure 7.2 plots accuracy on the original domain (y-axis) as the model is sequentially fine-tuned on a new domain over training steps (x-axis). Three curves show baseline-generative, contrastive-enhanced, and explanation-aware variants. The explanation-aware variant exhibits the slowest forgetting, suggesting that rationale-grounded training stabilizes cross-domain representations.

![Figure 7.2 — Catastrophic Forgetting Curve After Adapting to a New Domain](figures/fig_7_2_forgetting.png)

### Table 7.2 — Domain-Gap vs. Transfer Accuracy Correlation

Table 7.2 reports the correlation between an embedding-space domain-gap metric (mean pairwise Fréchet distance between CLS tokens of source and target domains) and zero-shot transfer accuracy, per training variant. A stronger negative correlation indicates that transfer is better predicted by representation similarity under that variant.

| Variant            | Pearson r (gap vs. acc) |
| ------------------ | ----------------------- |
| Baseline generative | −0.52                   |
| Contrastive-enhanced | −0.61                   |
| Explanation-aware   | −0.73                   |

---

# References

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS.

Dosovitskiy, A. et al. (2021). *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.* ICLR.

Goyal, Y. et al. (2017). *Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering.* CVPR.

Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.

Jia, C. et al. (2021). *Scaling Up Visual and Vision–Language Representation Learning with Noisy Text Supervision (ALIGN).* ICML.

Li, J. et al. (2023). *BLIP-2: Bootstrapping Language–Image Pre-training with Frozen Image Encoders and Large Language Models.* ICML.

Li, Y. et al. (2023). *Evaluating Object Hallucination in Large Vision–Language Models (POPE).* EMNLP.

Liu, H. et al. (2023). *Visual Instruction Tuning (LLaVA).* NeurIPS.

Lu, P. et al. (2022). *Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering (ScienceQA).* NeurIPS.

Masry, A. et al. (2022). *ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning.* ACL Findings.

Mathew, M. et al. (2021). *DocVQA: A Dataset for VQA on Document Images.* WACV.

Parcalabescu, L., & Frank, A. (2024). *On Measuring Faithfulness of Vision–Language Model Explanations.* ACL.

Radford, A. et al. (2021). *Learning Transferable Visual Models from Natural Language Supervision (CLIP).* ICML.

Rohrbach, A. et al. (2018). *Object Hallucination in Image Captioning.* EMNLP.

Schwenk, D. et al. (2022). *A-OKVQA: A Benchmark for Visual Question Answering using World Knowledge.* ECCV.

Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.

Wang, P. et al. (2024). *Qwen2-VL: Enhancing Vision–Language Model's Perception of the World at Any Resolution.* arXiv:2409.12191.

Wei, J. et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS.

Yu, J. et al. (2022). *CoCa: Contrastive Captioners are Image–Text Foundation Models.* TMLR.

---

# Appendix

## A. Dataset Statistics

| Dataset    | Images / Docs | Q–A Pairs | Has Gold Rationales | License          |
| ---------- | ------------- | --------- | ------------------- | ---------------- |
| DocVQA     | ≈12K          | ≈50K      | No                  | Academic         |
| ChartQA    | ≈20K          | ≈32K      | No                  | Academic         |
| ScienceQA  | ≈11K          | ≈21K      | **Yes**             | CC BY-NC-SA      |
| A-OKVQA    | ≈25K          | ≈25K      | Partial             | CC BY 4.0        |
| VQAv2      | ≈80K          | ≈50K sub. | No                  | CC BY 4.0        |

## B. Hyperparameter Grid

| Hyperparameter    | Grid Values                            |
| ----------------- | -------------------------------------- |
| Learning rate     | {5e-5, 1e-4, 2e-4, 5e-4}               |
| LoRA rank (r)     | {8, 16, 32}                            |
| LoRA alpha        | {16, 32, 64}                           |
| Warm-up ratio     | {1%, 3%, 5%}                           |
| Batch size (eff.) | {16, 32, 64}                           |
| Epochs            | {1, 2, 3}                              |

## C. Ethical Considerations

- **Data.** All datasets are used under academic / permissive licenses. No personal or identifiable information is introduced beyond what is already present in public research corpora.
- **Explanation-synthesis via teacher models.** GPT-4V-synthesized rationales (used for DocVQA, ChartQA, A-OKVQA, VQAv2) are filtered through automated quality checks and spot-audited by the author; the thesis reports filter thresholds and rejection rates.
- **Transparency.** All training configurations, prompts, filter thresholds, and human-rating rubrics are released with the code.
- **Intended use.** The released model variants are research artifacts. Downstream deployment in high-stakes domains (medical, legal, financial) would require additional domain-specific validation and is out of scope.
- **Compute and environmental cost.** Total reported GPU-hours are tracked and disclosed; all experiments are constrained to a single-A100 budget where possible.

## D. Model Evaluation Tables

Extended per-seed, per-domain, per-metric tables (Exact Match, ANLS, F1, BLEU-4, ROUGE-L, BERTScore-F1, POPE, CHAIR, attention-alignment, masking-consistency, human-rated faithfulness) will be provided in the final thesis appendix.

## E. Technical Environment

- **Hardware.** 1× NVIDIA A100 40GB (primary); Google Colab Pro+ (fallback).
- **OS.** Linux (Ubuntu 22.04) or macOS 14 with CPU-only fallback.
- **Python.** 3.10+, `uv` for package management.
- **Libraries.** `torch 2.x`, `transformers`, `peft`, `bitsandbytes`, `accelerate`, `datasets`, `evaluate`, `trl`, `einops`, `wandb`, `matplotlib`, `seaborn`.
- **Licensing.** Released code under Apache 2.0.
