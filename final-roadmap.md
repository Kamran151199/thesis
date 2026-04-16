# Bachelor's Thesis: Complete Battle Plan (v4 — Revised)
## "Cross-Modal Generative Alignment for Document, Chart, and Image Reasoning with Explanation-Aware Vision–Language Models"

---

## THE THESIS IN ONE PAGE

### What This Thesis Is About
Modern Vision–Language Models (VLMs) like GPT-4V, Gemini, and Claude can look at images, documents, and charts and reason about them in natural language. But HOW do they align visual and textual representations? And can we make them EXPLAIN their reasoning rather than just outputting answers? This thesis traces the evolution of vision–language alignment, deeply analyzes contrastive vs. generative alignment strategies, fine-tunes an explanation-aware VLM on document/chart/image reasoning tasks, and systematically investigates what makes cross-modal reasoning work through controlled experiments.

### The Four Research Questions

Everything in this thesis exists to answer these four questions:

```
RQ1: How have vision–language alignment strategies evolved, and what
     fundamental principles underpin cross-modal representation learning?
     → Answered by: Technical survey of alignment methods (Phase 1)

RQ2: How does generative alignment compare to contrastive alignment for
     document, chart, and natural image reasoning tasks?
     → Answered by: Implementing and comparing both strategies (Phase 2 + 3)

RQ3: To what extent does explanation-aware training improve VLM reasoning
     accuracy and explanation faithfulness across different visual domains?
     → Answered by: Training with/without explanations, evaluating both (Phase 2 + 3)

RQ4: What are the key architectural and training factors that determine
     cross-modal reasoning performance, and how do they differ across
     visual domains (documents vs. charts vs. natural images)?
     → Answered by: Ablation studies and cross-domain analysis (Phase 3)
```

### Why These Questions Matter
- **RQ1** provides a structured map of a rapidly evolving field — alignment methods are the backbone of every VLM, yet no single source connects them clearly
- **RQ2** is an active debate — CLIP-style contrastive learning dominates, but generative approaches (BLIP-2, CoCa) are gaining ground. Empirical comparison on structured visual domains (documents/charts) is underexplored
- **RQ3** addresses interpretability — explanation-aware models are critical for trust in high-stakes domains (medical documents, financial charts), but whether explanations actually improve reasoning accuracy is not settled
- **RQ4** produces practical guidance — document understanding requires OCR-like spatial awareness, chart reasoning needs numerical/structural understanding, natural images need scene comprehension. Are these fundamentally different problems for a VLM?

### Thesis Contribution Summary
1. A technical survey connecting the evolution of vision–language alignment methods into one narrative
2. A systematic comparison of contrastive vs. generative alignment on document, chart, and image reasoning
3. An explanation-aware fine-tuning approach with evaluation of both accuracy and explanation quality
4. Cross-domain ablation studies revealing what architectural/training factors matter most per visual domain

---

## WHAT CURRENT VLMS ACTUALLY DO (The Target Understanding)

Before you build anything, you need a precise mental model of what these systems are doing under the hood.

### The Core Pipeline of Every VLM

```
                    ┌──────────────┐
   Image/Doc/Chart  │ Vision       │  Visual feature vectors
   (pixels)     ──→ │ Encoder      │ ──→ (sequence of patch embeddings)
                    │ (ViT/CLIP)   │
                    └──────────────┘
                                          ┌─────────────────┐
                                      ──→ │ Alignment Layer  │ ──→ Projected visual tokens
                                          │ (Projection/     │     (in LLM's embedding space)
                                          │  Q-Former/       │
                                          │  Cross-Attention) │
                                          └─────────────────┘
                    ┌──────────────┐              │
   Question/Prompt  │ Text         │              ▼
   (text)       ──→ │ Tokenizer +  │ ──→ [visual tokens] + [text tokens]
                    │ Embeddings   │              │
                    └──────────────┘              ▼
                                          ┌─────────────────┐
                                          │ Language Model   │ ──→ Generated answer
                                          │ (Decoder-only    │     (token by token)
                                          │  Transformer)    │
                                          └─────────────────┘
```

### How Each Major VLM Does It

**LLaVA (Meta/Wisconsin):**
- Vision encoder: CLIP-ViT-L/14 (frozen) → 576 patch embeddings
- Alignment: Simple 2-layer MLP projection → map visual features into LLM space
- LLM: Vicuna-7B/13B (fine-tuned LLaMA)
- Training: 2-stage — (1) pre-train projection on caption data, (2) instruction-tune end-to-end
- Key insight: simplest possible alignment (just an MLP) works surprisingly well

**BLIP-2 (Salesforce):**
- Vision encoder: ViT-G (frozen)
- Alignment: Q-Former — a small transformer with 32 learnable query tokens that cross-attend to visual features
- LLM: FlanT5 or OPT (frozen)
- Training: 3-stage — (1) contrastive + generative pre-training of Q-Former, (2) generative pre-training with LLM, (3) instruction tuning
- Key insight: Q-Former acts as an "information bottleneck" — extracts only the most relevant visual info

**InternVL-2 (Shanghai AI Lab):**
- Vision encoder: InternViT-6B (scaled-up ViT, sometimes frozen, sometimes unfrozen)
- Alignment: MLP projection with dynamic resolution (handles different image sizes/aspect ratios)
- LLM: InternLM2 or LLaMA3
- Key insight: Scaling the vision encoder matters as much as scaling the LLM

**Qwen2-VL (Alibaba):**
- Vision encoder: ViT with Naive Dynamic Resolution — processes images at their native resolution
- Alignment: Cross-attention layers + MLP projection
- LLM: Qwen2 language model
- Key insight: Native resolution processing is critical for document/chart understanding

**GPT-4V / GPT-4o (OpenAI):**
- Architecture: Not fully public, but known to use vision encoding → projection → large LLM
- Likely uses a strong ViT encoder with sophisticated projection into a massive decoder-only transformer
- Can handle documents, charts, images, and complex reasoning with explanations
- Represents the "north star" of what VLMs can do

**Document-Specific Models:**
- **Donut:** No OCR needed — encoder-decoder transformer reads document images directly
- **Pix2Struct:** Screenshot-to-structure — learns to parse visual layouts into structured text
- **LayoutLMv3:** Combines text tokens + visual patches + 2D position embeddings (layout-aware)

**Chart-Specific Models:**
- **DePlot:** Translates charts into linearized data tables, then reasons with an LLM
- **MatCha:** Fine-tunes Pix2Struct specifically for math reasoning and chart understanding
- **ChartLlama:** Fine-tunes LLaVA on chart-specific instruction data

### The Two Alignment Paradigms (Central to YOUR Thesis)

```
CONTRASTIVE ALIGNMENT (CLIP-style):
  ┌─────────┐     ┌─────────┐
  │ Image   │     │ Text    │
  │ Encoder │     │ Encoder │
  └────┬────┘     └────┬────┘
       │               │
       ▼               ▼
  [img_embed]     [txt_embed]      ← Both projected to same space
       │               │
       └──── cosine ───┘           ← Push matching pairs together,
             similarity                non-matching apart
  
  Loss: InfoNCE (contrastive)
  Result: Shared embedding space where similar concepts are nearby
  Limitation: Alignment is "coarse" — one vector per image, one per text

GENERATIVE ALIGNMENT (BLIP-2 / CoCa style):
  ┌─────────┐
  │ Image   │
  │ Encoder │
  └────┬────┘
       │
       ▼
  [visual features]
       │
       ▼
  ┌──────────────┐     "A dog playing    ← The model must GENERATE text
  │ Cross-attend │ ──→  fetch in a park"    that describes the image
  │ + Generate   │
  └──────────────┘
  
  Loss: Cross-entropy (language modeling)
  Result: Alignment through generation — model must deeply understand image to describe it
  Advantage: "Fine-grained" alignment — every generated token must match visual content
  YOUR THESIS HYPOTHESIS: Generating EXPLANATIONS is an even richer alignment signal
```

**Your thesis explores:** Does generative alignment through explanation generation create better cross-modal representations than contrastive alignment, specifically for complex structured visual content (documents, charts)?

---

## THESIS STRUCTURE (How Chapters Map to Research Questions)

```
Chapter 1: Introduction
  → Frames all four RQs, motivates why document/chart reasoning matters

Chapter 2: Background & Related Work ← ANSWERS RQ1
  → 2.1: Transformer Architecture (Encoder, Decoder, Encoder-Decoder)
  → 2.2: Vision Transformers and Visual Representation Learning
  → 2.3: Vision–Language Alignment: From CLIP to Generative Methods
  → 2.4: Document, Chart, and Image Understanding Models
  → 2.5: Explanation-Aware and Chain-of-Thought Reasoning in VLMs
  → Design principles extracted from each evolution

Chapter 3: Research Framework ← FRAMES RQ2, RQ3, RQ4
  → 3.1: Contrastive vs. Generative Alignment — The Design Space
  → 3.2: Explanation-Aware Training Objectives
  → 3.3: Cross-Domain Visual Reasoning Challenges
  → 3.4: Evaluation Framework (metrics for accuracy + explanation quality)

Chapter 4: System Design & Implementation ← METHODOLOGY FOR RQ2, RQ3, RQ4
  → 4.1: Base Model Selection and Architecture
  → 4.2: Dataset Preparation and Explanation Annotation
  → 4.3: Training Pipeline (baseline, contrastive, generative, explanation-aware)
  → 4.4: Evaluation Protocol

Chapter 5: Experiments & Analysis ← ANSWERS RQ2, RQ3, RQ4
  → 5.1: Contrastive vs. Generative Alignment Comparison (RQ2)
  → 5.2: Explanation-Aware Training Impact (RQ3)
  → 5.3: Cross-Domain Ablation Studies (RQ4)
  → 5.4: Attention and Representation Analysis
  → 5.5: Qualitative Analysis of Generated Explanations

Chapter 6: Discussion ← SYNTHESIZES ALL RQs
  → Key findings, implications, limitations

Chapter 7: Conclusion & Future Work
  → Summary of answers to each RQ, future directions
```

---

## PHASE 0: SETUP (Days 1-2)

### Hardware
- **Minimum:** Google Colab Pro+ ($50/month) with A100 — sufficient for QLoRA fine-tuning
- **Better:** Cloud GPU (Lambda Labs A100, Vast.ai, or GCP since you know it)
- **Ideal:** 1x A100 40GB or 1x H100 — can do full fine-tuning of 2B-7B models

### Software
```
Python 3.10+
PyTorch 2.x
Hugging Face ecosystem:
  - transformers (VLMs, ViT, CLIP, LLMs)
  - datasets (data loading)
  - accelerate (distributed/mixed-precision training)
  - peft (LoRA, QLoRA — parameter-efficient fine-tuning)
  - evaluate (metrics)
  - trl (training with reward/preference — optional)
bitsandbytes (4-bit/8-bit quantization for QLoRA)
einops (tensor reshaping)
wandb (experiment tracking)
Pillow / torchvision (image processing)
matplotlib / seaborn (visualization)
nltk / rouge-score (explanation evaluation metrics)
```

### Project Structure
```
thesis-vlm/
├── notebooks/
│   ├── 01_attention_from_scratch.ipynb
│   ├── 02_transformer_encoder_decoder.ipynb
│   ├── 03_vision_transformer.ipynb
│   ├── 04_clip_deep_dive.ipynb
│   ├── 05_blip2_q_former.ipynb
│   ├── 06_llava_inference_trace.ipynb
│   ├── 07_document_models.ipynb
│   ├── 08_chart_models.ipynb
│   ├── 09_explanation_generation.ipynb
│   └── 10_alignment_comparison.ipynb
├── src/
│   ├── data/
│   │   ├── docvqa_loader.py
│   │   ├── chartqa_loader.py
│   │   ├── scienceqa_loader.py
│   │   ├── vqa_loader.py
│   │   └── explanation_annotator.py
│   ├── models/
│   │   ├── base_vlm.py              # Base VLM wrapper
│   │   ├── explanation_head.py       # Explanation generation module
│   │   └── alignment_objectives.py   # Contrastive + Generative losses
│   ├── training/
│   │   ├── train_baseline.py         # Standard fine-tuning
│   │   ├── train_explanation.py      # Explanation-aware fine-tuning
│   │   ├── train_contrastive.py      # Contrastive alignment fine-tuning
│   │   └── train_generative.py       # Generative alignment fine-tuning
│   └── evaluation/
│       ├── accuracy_metrics.py
│       ├── explanation_metrics.py
│       ├── visualization.py
│       └── cross_domain_analysis.py
├── configs/
│   ├── baseline.yaml
│   ├── explanation_aware.yaml
│   ├── contrastive.yaml
│   └── generative.yaml
├── thesis/                           # LaTeX or Word documents
└── README.md
```

### Day 1: Environment Setup
- Set up GPU environment (Colab Pro+ or cloud)
- Install all dependencies
- Verify GPU access: `torch.cuda.is_available()`
- Download and verify you can load a small VLM (e.g., `Qwen/Qwen2-VL-2B-Instruct`)
- Set up wandb for experiment tracking
- Set up git repo

### Day 2: Data Exploration
- Download and explore each dataset you'll use:
  - DocVQA (~50K document questions)
  - ChartQA (~32K chart questions)
  - ScienceQA (~21K questions WITH explanations — gold mine for your thesis)
  - VQAv2 subset (~50K general image questions)
  - A-OKVQA (~25K questions with rationales)
- For each: load 10 examples, look at the images, read the questions and answers
- Note the format differences — this informs your data pipeline design

---

## PHASE 1: FOUNDATIONS — Deep Dive Into How It All Works (Days 3-21)
### → Answers RQ1, builds the mental model you need for everything else

This is the phase where you go from "I know NLP basics" to "I deeply understand how vision-language models work internally." You are simultaneously LEARNING and WRITING your survey chapters.

---

### Week 1: Transformer Internals + Vision Foundations (Days 3-9)

**Days 3-4: The Transformer — From First Principles**
**→ RQ1 foundation: This is THE architecture underlying everything**

You said you don't understand how transformer encoder/decoder works. After these 2 days, you will — from the inside out.

**The Core Insight to Internalize:**
A transformer is just a stack of two operations repeated over and over:
1. **Self-Attention:** Every token looks at every other token and decides how much to "pay attention" to each one
2. **Feed-Forward Network:** Each token independently processes its information through a small neural network

That's it. Everything else is optimization on top.

**Step-by-Step Build (Notebook 01):**

```python
# STEP 1: Understand what attention actually computes
# For each token, attention answers: "Which other tokens are relevant to me?"

def self_attention(x):
    """
    x: (batch, seq_len, d_model) — sequence of token embeddings

    Three projections create three roles for each token:
    - Query (Q): "What am I looking for?"
    - Key (K):   "What do I contain?"
    - Value (V): "What information do I carry?"
    """
    Q = x @ W_q  # (batch, seq_len, d_k) — each token's question
    K = x @ W_k  # (batch, seq_len, d_k) — each token's label
    V = x @ W_v  # (batch, seq_len, d_v) — each token's actual content

    # Attention scores: how much does each query match each key?
    scores = Q @ K.transpose(-2, -1) / sqrt(d_k)
    # scores shape: (batch, seq_len, seq_len) — every token's relevance to every other

    # Softmax → probabilities (each row sums to 1)
    weights = softmax(scores, dim=-1)

    # Weighted sum of values — each token becomes a mix of relevant tokens
    output = weights @ V
    return output

# WHY THIS MATTERS:
# - "Attention Is All You Need" showed this REPLACES recurrence entirely
# - Token at position 100 can directly attend to token at position 1
# - No sequential bottleneck like RNNs — everything is parallel
# - The (seq_len × seq_len) attention matrix is why transformers are O(n²)
```

**Multi-Head Attention — Why Multiple Heads?**
```python
# Single head: one "question" per token position
# Multi-head: multiple independent "questions" per position
# Head 1 might learn: "What is the subject of this sentence?"
# Head 2 might learn: "What adjectives modify me?"
# Head 3 might learn: "What is the next likely word pattern?"

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.d_k = d_model // num_heads  # each head gets a slice
        self.heads = nn.ModuleList([
            AttentionHead(d_model, self.d_k) for _ in range(num_heads)
        ])
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        head_outputs = [head(x, mask) for head in self.heads]
        concat = torch.cat(head_outputs, dim=-1)  # reassemble
        return self.output_proj(concat)
```

**The Three Transformer Flavors (THIS IS CRITICAL FOR YOUR THESIS):**

```
ENCODER-ONLY (BERT, ViT, CLIP's image/text encoders):
  ┌─────────────────────────────────┐
  │ Token 1 ←→ Token 2 ←→ Token 3  │  ← Bidirectional: every token sees all others
  │ Token 4 ←→ Token 5 ←→ Token 6  │  ← No masking — full context in both directions
  └─────────────────────────────────┘
  Used for: UNDERSTANDING. Encode input into a rich representation.
  - BERT: encode text → classify/extract
  - ViT: encode image patches → classify/understand
  - CLIP vision encoder: encode image → embedding for alignment

DECODER-ONLY (GPT, LLaMA, Vicuna — the LLMs in VLMs):
  ┌─────────────────────────────────┐
  │ Token 1 → Token 2 → Token 3    │  ← Causal mask: each token only sees PAST tokens
  │           Token 2 → Token 3    │  ← Cannot look forward (would be "cheating")
  │                      Token 3   │
  └─────────────────────────────────┘
  Used for: GENERATION. Predict the next token given all previous tokens.
  - GPT: generate text
  - LLaMA: generate text
  - In VLMs: takes visual tokens + question tokens → generates answer tokens

ENCODER-DECODER (T5, original Transformer, Donut):
  ┌──────────────┐     ┌──────────────┐
  │ ENCODER      │     │ DECODER      │
  │ (bidirection)│ ──→ │ (causal +    │
  │ Input tokens │     │  cross-attn) │
  └──────────────┘     └──────────────┘
  The decoder has TWO attention mechanisms:
  1. Self-attention (causal, looking at own generated tokens)
  2. Cross-attention (looking at encoder's output — "what did the input say?")
  Used for: SEQUENCE-TO-SEQUENCE. Transform input sequence to output sequence.
  - T5: text-to-text
  - Donut: image-to-text (document understanding)
  - BLIP-2's Q-Former uses cross-attention to extract from visual features

CROSS-ATTENTION (the bridge between modalities):
  The KEY mechanism for VLMs. In cross-attention:
  - Q (queries) come from one modality (e.g., text/learnable tokens)
  - K, V (keys, values) come from another modality (e.g., image features)
  - This is how the language model "looks at" the image

  cross_attention(text_tokens, image_features):
      Q = text_tokens @ W_q     # "What do I want to know from the image?"
      K = image_features @ W_k  # "What does each image patch contain?"
      V = image_features @ W_v  # "What information does each patch carry?"
      # Result: each text token attends to relevant image patches
```

**Build It (Notebook 01 + 02):**
1. Implement single-head attention from scratch — print the attention weights, visualize them
2. Implement multi-head attention
3. Build a full transformer block (attention + feed-forward + layer norm + residual)
4. Build an encoder-only model (like a tiny BERT)
5. Build a decoder-only model (like a tiny GPT — follow Karpathy's nanoGPT)
6. Build cross-attention and understand why it's the VLM bridge
7. Train the tiny GPT on Shakespeare text — watch it learn

**Key "aha" moments to reach:**
- Attention weights ARE interpretable — you can see what attends to what
- Positional encoding is necessary because attention itself has no notion of order
- Layer normalization + residual connections are what make deep transformers trainable
- The causal mask is what makes decoder-only models autoregressive

**Resources:**
- **Watch:** Karpathy "Let's build GPT from scratch" (YouTube, ~2hrs) — build along with him
- **Watch:** Karpathy "Let's build the GPT Tokenizer" (YouTube, ~2hrs) — understand BPE tokenization
- **Read:** "Attention Is All You Need" (Vaswani et al., 2017) — focus on Figures 1-2, Section 3
- **Read:** Jay Alammar "The Illustrated Transformer" (blog — the BEST visual explanation)

**Thesis Writing — Start Immediately:**
- Begin Chapter 2, Section 2.1: "The Transformer Architecture"
- Cover: self-attention, multi-head attention, encoder/decoder/encoder-decoder variants
- Create Figure 2.1: transformer architecture diagram (all three variants)
- Create Figure 2.2: attention weight visualization from your notebook

---

**Days 5-6: Vision Transformer (ViT) — How Transformers See**
**→ RQ1: The critical bridge from language to vision**

The insight that changed everything: an image is just a sequence of patches, and each patch is a "token."

```
Input image (224×224 pixels)
     │
     ▼
Split into patches (16×16 pixels each)
     │ → 14×14 = 196 patches (= 196 "tokens")
     ▼
Flatten each patch → 16×16×3 = 768 numbers per patch
     │
     ▼
Linear projection → 768-dim embedding per patch (same as a word embedding!)
     │
     ▼
Add [CLS] token (learnable, prepended) → 197 tokens
     │
     ▼
Add positional embeddings (learned, one per position)
     │
     ▼
Feed through standard Transformer Encoder (12 layers, same as BERT!)
     │
     ▼
[CLS] token's final embedding = image representation
```

**Why ViT matters for YOUR thesis:**
- Every VLM uses a ViT (or its descendant) as the vision encoder
- CLIP's image encoder IS a ViT
- Understanding ViT = understanding how images become the "visual tokens" that get aligned with text
- For documents/charts: patch size matters — small text in a document needs smaller patches or higher resolution

**Build It (Notebook 03):**
```python
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, d_model=768, num_layers=12, num_heads=12):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2  # 196
        patch_dim = patch_size * patch_size * 3       # 768

        # Convert patches to embeddings (like word embeddings but for image patches)
        self.patch_embedding = nn.Linear(patch_dim, d_model)

        # CLS token — aggregates information from all patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Position embeddings — tells the model where each patch is spatially
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, d_model))

        # Standard transformer encoder layers (SAME as BERT!)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=3072),
            num_layers=num_layers
        )

    def forward(self, image):
        # image: (batch, 3, 224, 224)
        patches = extract_patches(image, patch_size=16)   # (batch, 196, 768)
        x = self.patch_embedding(patches)                  # (batch, 196, 768)

        cls = self.cls_token.expand(x.shape[0], -1, -1)   # (batch, 1, 768)
        x = torch.cat([cls, x], dim=1)                     # (batch, 197, 768)
        x = x + self.pos_embedding                          # add position info

        x = self.transformer(x)                             # (batch, 197, 768)

        cls_output = x[:, 0]        # image-level representation
        patch_outputs = x[:, 1:]    # patch-level representations (used by VLMs!)
        return cls_output, patch_outputs
```

**Hands-on:**
- Build a tiny ViT, train it on CIFAR-10 image classification
- Visualize the attention maps — which patches attend to which?
- Visualize the position embeddings — the model learns a spatial grid!
- Load a pre-trained ViT from HuggingFace, compare with yours

**Key insight for your thesis:**
- In VLMs, the **patch_outputs** (not just CLS) are what get projected into the LLM's space
- Each patch output = a "visual token" that carries localized information
- For a document: patch at top-left carries header info, patch at center carries body text
- For a chart: patch near an axis carries axis label info, patch near bars carries data

**Resources:**
- **Read:** "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020) — the ViT paper
- **Watch:** Yannic Kilcher's ViT explanation (YouTube)
- **Read:** "DINOv2" paper intro (understanding self-supervised ViT training)

**Thesis Writing:**
- Chapter 2, Section 2.2: "Vision Transformers and Visual Representation Learning"
- Cover: patch embedding, position encoding, CLS token, relationship to NLP transformers
- Create Figure 2.3: ViT architecture diagram showing image → patches → tokens → transformer
- Key point: ViT proves that THE SAME transformer architecture works for both language and vision

---

**Days 7-8: CLIP — Where Vision Meets Language**
**→ RQ1 + RQ2: Understanding contrastive alignment (one side of your thesis comparison)**

CLIP is the most important model for your thesis to understand deeply, because it established the dominant paradigm for vision–language alignment.

```
CLIP TRAINING:
  Batch of (image, text) pairs:
    ("photo of a dog",  🐕 )
    ("a red sports car", 🏎️ )
    ("sunset over ocean", 🌅 )

  Image Encoder (ViT):  image → image_embedding (512-dim vector)
  Text Encoder (Transformer): text → text_embedding (512-dim vector)

  Similarity matrix (batch_size × batch_size):
       "dog"  "car"  "sunset"
  🐕  [ 0.9   0.1    0.2  ]  ← dog image should match "dog" text
  🏎️  [ 0.1   0.8    0.1  ]  ← car image should match "car" text
  🌅  [ 0.2   0.1    0.9  ]  ← sunset image should match "sunset" text

  InfoNCE Loss: Maximize diagonal (matching pairs), minimize off-diagonal
  This is CONTRASTIVE alignment — learn by comparing positives vs negatives
```

**Why CLIP matters for your thesis:**
1. It's the vision encoder backbone of LLaVA, many VLMs
2. It represents CONTRASTIVE alignment — one of the two paradigms you're comparing (RQ2)
3. Its embedding space is what gets "projected" into LLMs
4. Understanding its limitations explains why generative alignment emerged

**CLIP's Limitations (motivates your RQ2):**
- Alignment is at image-level, not region-level — "a dog" matches the whole image, not the dog region
- Can't do complex reasoning — "a dog to the LEFT of a cat" vs "a dog to the RIGHT of a cat" get similar scores
- Can't generate — only encodes, doesn't produce text or explanations
- These limitations are exactly why generative alignment (BLIP-2, your thesis direction) emerged

**Build It (Notebook 04):**
```python
# Don't build CLIP from scratch — load pre-trained and DEEPLY STUDY it

from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Experiment 1: Encode images and texts, compute similarity matrix
images = [load_image(path) for path in image_paths]  # 50 diverse images
texts = ["a photo of a dog", "a chart showing revenue", "a scanned document", ...]

inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
outputs = model(**inputs)

image_embeds = outputs.image_embeds  # (50, 512)
text_embeds = outputs.text_embeds    # (50, 512)

# Cosine similarity matrix
similarity = image_embeds @ text_embeds.T  # (50, 50)
# Visualize this matrix — it shows you what CLIP "understands"

# Experiment 2: Test CLIP on document/chart images
# Does CLIP understand "a bar chart showing Q3 revenue"? (spoiler: poorly)
# Does CLIP understand "a document about tax returns"? (somewhat)
# This motivates why specialized models exist for documents/charts

# Experiment 3: Extract patch-level features (not just CLS)
vision_outputs = model.vision_model(pixel_values=pixel_values)
patch_features = vision_outputs.last_hidden_state  # (batch, 257, 1024)
# These 257 patch-level features are what VLMs like LLaVA project into the LLM
```

**Resources:**
- **Read:** "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021) — CLIP paper
- **Watch:** Yannic Kilcher CLIP explanation (YouTube)
- **Read:** "SigLIP" paper (Google's improved contrastive alignment — simpler loss function)

**Thesis Writing:**
- Chapter 2, Section 2.3 begins: "Vision–Language Alignment: From CLIP to Generative Methods"
- Subsection on contrastive alignment: CLIP, SigLIP, ALIGN
- Create Figure 2.4: CLIP architecture with contrastive loss visualization
- Document CLIP's strengths and limitations → this naturally motivates generative alignment

---

**Day 9: Synthesis + Writing**
**→ Complete the foundation sections of Chapter 2**

Pull together what you've learned:
1. Transformers are the universal architecture (text AND vision)
2. ViT proved images can be processed exactly like text sequences
3. CLIP proved you can align image and text in a shared space via contrastive learning
4. But contrastive alignment has limits → motivating generative approaches

Write a synthesis paragraph connecting these dots. This becomes the bridge to Week 2's content.

**Checkpoint:** By end of Day 9, you should be able to:
- [ ] Explain self-attention, multi-head attention, and why they work — from code, not just words
- [ ] Explain the difference between encoder-only, decoder-only, and encoder-decoder transformers
- [ ] Explain how ViT turns an image into a sequence of tokens
- [ ] Explain how CLIP aligns vision and language contrastively
- [ ] Explain why cross-attention is the key mechanism in VLMs
- [ ] Have rough drafts of Chapter 2 sections 2.1, 2.2, and start of 2.3

---

### Week 2: VLM Architectures + Alignment Deep Dive (Days 10-16)

**Days 10-11: BLIP-2 and Generative Alignment**
**→ RQ2: Understanding the GENERATIVE side of alignment**

BLIP-2 is your most important reference model because it uses BOTH contrastive AND generative alignment, letting you study both paradigms within one framework.

**BLIP-2's Three-Stage Training (understand each stage deeply):**

```
STAGE 1: Representation Learning (Q-Former learns to extract from vision)
  ┌────────────────┐
  │ Frozen ViT     │ → image features (257 tokens × 1408 dims)
  └───────┬────────┘
          │
          ▼
  ┌────────────────────────────────────────────┐
  │ Q-Former (small transformer)               │
  │   32 learnable query tokens                │
  │   ↕ self-attend with each other            │
  │   ↕ cross-attend to image features         │
  │   → 32 output vectors (information bottleneck) │
  └────────────────────────────────────────────┘
          │
          ▼
  Three objectives trained simultaneously:
    1. Image-Text Contrastive (ITC): match Q-Former outputs ↔ text features
    2. Image-grounded Text Generation (ITG): generate caption from Q-Former outputs
    3. Image-Text Matching (ITM): binary — does this image match this text?

  KEY INSIGHT: Objective #2 is GENERATIVE ALIGNMENT
  The Q-Former must extract enough visual information to GENERATE the correct caption
  This is a much richer learning signal than just "match or don't match" (contrastive)

STAGE 2: Generative Pre-training (connect Q-Former to LLM)
  Q-Former outputs → linear projection → LLM input space
  Train the projection to produce embeddings the LLM can use to generate text

STAGE 3: Instruction Tuning
  Fine-tune for question-answering, reasoning, explanation generation
```

**The Q-Former Architecture (study carefully):**
```python
# The Q-Former is a transformer with a twist:
# It has LEARNABLE QUERIES that cross-attend to frozen image features

class QFormer(nn.Module):
    def __init__(self, num_queries=32, d_model=768, num_layers=12):
        super().__init__()
        # These 32 queries are LEARNED — the model discovers what to ask
        self.queries = nn.Parameter(torch.randn(1, num_queries, d_model))

        self.layers = nn.ModuleList([
            QFormerLayer(d_model) for _ in range(num_layers)
        ])

    def forward(self, image_features):
        # image_features: from frozen ViT (batch, 257, 1408)
        queries = self.queries.expand(batch_size, -1, -1)

        for layer in self.layers:
            # Step 1: queries self-attend (talk to each other)
            queries = layer.self_attention(queries)
            # Step 2: queries cross-attend to image (look at the image)
            queries = layer.cross_attention(
                queries=queries,       # Q from learnable queries
                keys=image_features,   # K from image patches
                values=image_features  # V from image patches
            )
            # Step 3: feed-forward
            queries = layer.ffn(queries)

        return queries  # (batch, 32, 768) — compressed visual information

# WHY 32 QUERIES?
# 257 patch features → 32 query outputs = 8x compression
# Forces the model to extract only the MOST RELEVANT visual information
# This is the "information bottleneck" principle
# For YOUR thesis: this bottleneck might lose fine-grained document/chart details
```

**For RQ2, note the key difference:**
- CLIP (contrastive): One embedding per image, one per text → coarse alignment
- BLIP-2 (generative): 32 query embeddings per image, must generate correct text → fine-grained alignment
- Your hypothesis: Generative alignment captures more nuanced visual information, especially for structured content (documents, charts)

**Hands-on (Notebook 05):**
- Load BLIP-2 from HuggingFace, trace the data flow step by step
- Feed in a document image → observe what Q-Former extracts
- Feed in a chart image → observe what Q-Former extracts
- Feed in a natural image → compare
- Visualize Q-Former cross-attention weights: which image patches do the queries attend to?

**Resources:**
- **Read:** "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models" (Li et al., 2023) — read ALL of it, this is core
- **Read:** "CoCa: Contrastive Captioners are Image-Text Foundation Models" (Yu et al., 2022) — hybrid alignment
- **Watch:** Umar Jamil BLIP-2 explanation if available

**Thesis Writing:**
- Chapter 2, Section 2.3 continues: generative alignment methods
- Create Figure 2.5: BLIP-2 Q-Former architecture diagram
- Create Figure 2.6: side-by-side comparison of contrastive vs. generative alignment
- This section directly motivates your RQ2

---

**Days 12-13: LLaVA and Modern VLMs — The Full Pipeline**
**→ Understanding the complete system you'll be fine-tuning**

LLaVA is important because it's the simplest effective VLM architecture — and the one you'll likely build on.

```
LLaVA ARCHITECTURE:
  Image → CLIP-ViT-L/14 (frozen) → 576 patch features (each 1024-dim)
                                          │
                                          ▼
                                    2-layer MLP projection
                                    (1024 → 4096 → 4096)
                                          │
                                          ▼
                                    576 visual tokens (in LLM space)
                                          │
  Question text → Tokenizer → Text tokens │
                                          │
                                          ▼
                           [visual tokens] + [text tokens]
                                          │
                                          ▼
                              LLaMA/Vicuna (7B/13B)
                              Decoder-only Transformer
                                          │
                                          ▼
                              Generated answer (token by token)

LLaVA TRAINING:
  Stage 1 — Feature Alignment (train projection only):
    Data: 558K image-caption pairs from CC3M
    Frozen: ViT + LLM
    Trainable: MLP projection ONLY
    Goal: teach the projection to map visual features into LLM-understandable tokens

  Stage 2 — Visual Instruction Tuning (train projection + LLM):
    Data: 158K visual instruction data (question-answer pairs about images)
    Frozen: ViT only
    Trainable: MLP projection + LLM (full fine-tune or LoRA)
    Goal: teach the LLM to reason about visual content and follow instructions
```

**Why LLaVA's simplicity matters for YOUR thesis:**
- The MLP projection is so simple that any performance differences you observe are due to the ALIGNMENT STRATEGY (contrastive vs. generative) and TRAINING DATA (with/without explanations) — not architectural complexity
- This is ideal for controlled experiments (RQ2, RQ3)

**Hands-on (Notebook 06):**
```python
from transformers import LlavaForConditionalGeneration, AutoProcessor

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# Feed it a document, chart, and natural image
# For each: ask a question, observe the answer quality
# KEY: Also ask it to EXPLAIN its reasoning
# "What is the total revenue shown in this chart? Explain how you determined this."
# → This tells you how well current VLMs explain their reasoning (baseline for RQ3)

# Trace the forward pass:
# 1. Image → vision_encoder → patch features
# 2. Patch features → projection → visual tokens
# 3. [visual tokens] + [text tokens] → LLM → generated tokens
# Inspect each intermediate representation's shape and content
```

**Also study more recent VLMs:**

| Model | Vision Encoder | Alignment | LLM | Key Innovation |
|-------|---------------|-----------|-----|----------------|
| LLaVA | CLIP-ViT | MLP | Vicuna | Simplest effective architecture |
| LLaVA-NeXT | CLIP-ViT + DynamicRes | MLP | Various | Higher resolution for documents |
| InternVL-2 | InternViT-6B | MLP | InternLM2 | Massive vision encoder |
| Qwen2-VL | ViT + NaiveDynRes | Cross-attn + MLP | Qwen2 | True native resolution |
| Phi-3-Vision | CLIP-ViT | MLP | Phi-3 | Small but capable (3.8B) |

**For YOUR thesis, the base model decision:**
- **Qwen2-VL-2B-Instruct** or **InternVL2-2B** — small enough for QLoRA fine-tuning on single A100, capable enough for meaningful results
- Alternative: **Phi-3-Vision** (3.8B) — great balance of size and capability
- These handle documents and charts better than original LLaVA due to dynamic resolution

**Resources:**
- **Read:** "Visual Instruction Tuning" (Liu et al., 2023) — LLaVA paper
- **Read:** "LLaVA-NeXT" / "LLaVA-OneVision" papers — resolution improvements
- **Watch:** Umar Jamil LLaVA explanation (YouTube)

---

**Days 14-15: Document and Chart Understanding Models**
**→ RQ4: Understanding domain-specific challenges**

Documents and charts present unique challenges that natural images don't:

```
NATURAL IMAGE:                    DOCUMENT:                      CHART:
┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
│  🐕 dog playing    │   │ TITLE: Invoice #42 │   │     Revenue ($M)   │
│  in a park with    │   │ Date: 2024-01-15   │   │   ┃█              │
│  green grass and   │   │ Item    Price      │   │   ┃█ █            │
│  blue sky          │   │ Widget  $42.00     │   │   ┃█ █ █          │
│                    │   │ Gadget  $18.50     │   │   ┃█ █ █ █        │
│                    │   │ Total:  $60.50     │   │   ┗━━━━━━━━━━━━━  │
└────────────────────┘   └────────────────────┘   │    Q1 Q2 Q3 Q4   │
                                                   └────────────────────┘
Challenge: Scene         Challenge: OCR +           Challenge: Spatial
understanding,           layout understanding,      reasoning, numerical
object recognition       text reading               extraction, trend analysis
```

**Document AI Models (study these):**

**LayoutLMv3:** Combines three input types
```
Input: Document image
  → Text tokens (from OCR) with 2D bounding box positions
  → Image patches (from ViT)
  → Layout information (x, y, width, height of each text region)

  The model learns that "Total: $60.50" at the BOTTOM RIGHT of an invoice
  means something different than "Total: $60.50" in the middle of a paragraph.
```

**Donut (Document Understanding Transformer):**
```
Input: Document IMAGE (no OCR needed!)
  → ViT encoder processes the image directly
  → Transformer decoder generates structured text output
  → End-to-end: image → structured answer

  Key insight: OCR is a bottleneck and error source. Skip it entirely.
  Relevant to YOUR thesis: end-to-end models avoid compounding OCR errors
```

**Pix2Struct:**
```
Input: Screenshot/document image
  → Variable-resolution ViT (handles different aspect ratios)
  → Generates structured text (HTML, markdown, answers)
  → Pre-trained on web screenshots → parse HTML structure

  Key insight: Pre-training on "screenshot → code" teaches visual structure parsing
  Directly relevant to chart understanding (MatCha is Pix2Struct fine-tuned for charts)
```

**Chart-Specific Models:**

**DePlot:** Chart → Data Table → LLM reasoning
```
Chart image → Pix2Struct → Linearized data table
  "Title: Revenue | x: Q1, Q2, Q3, Q4 | y: 10, 25, 30, 45"
    → Feed this text to an LLM for reasoning
  
  Key insight: Separate visual parsing from reasoning
  Limitation: Lossy — if the chart-to-table step fails, reasoning fails too
```

**MatCha:** Pix2Struct + Math reasoning
```
Chart image → fine-tuned Pix2Struct → answer
  Trained specifically on chart QA with mathematical operations
  Can compute: "What is the difference between Q4 and Q1 revenue?"
  
  Key for RQ4: Shows that chart reasoning requires numerical capabilities
  that standard VLMs may lack
```

**Hands-on (Notebook 07 + 08):**
- Load Donut, run it on a sample document → observe its encoder-decoder behavior
- Load DePlot or MatCha (via Pix2Struct), run on sample charts
- Load your chosen base VLM (Qwen2-VL-2B), run on the SAME documents and charts
- Compare: Where does the general VLM succeed? Where does it fail?
- This gives you a qualitative baseline for RQ4

**Resources:**
- **Read:** "Donut: Document Understanding Transformer" (Kim et al., 2022)
- **Read:** "Pix2Struct: Screenshot Parsing as Pretraining" (Lee et al., 2023)
- **Read:** "DePlot: One-shot visual language reasoning by plot-to-table translation" (Liu et al., 2023)
- **Read:** "MatCha: Enhancing Visual Language Pretraining with Math Reasoning and Chart Derendering" (Liu et al., 2023)

**Thesis Writing:**
- Chapter 2, Section 2.4: "Document, Chart, and Image Understanding Models"
- Create a taxonomy table of approaches (end-to-end vs. pipeline, OCR-dependent vs. OCR-free)
- Highlight what each domain requires that others don't (spatial layout for docs, numerical reasoning for charts)

---

**Day 16: Explanation-Aware Models + Chain-of-Thought**
**→ RQ3: Understanding how explanations improve reasoning**

This is the "explanation-aware" part of your thesis title. The key question: can forcing a model to EXPLAIN its reasoning actually IMPROVE the reasoning itself?

**Chain-of-Thought (CoT) in VLMs:**

```
WITHOUT EXPLANATION:
  Q: "What is the total revenue in Q4?"
  Image: [bar chart]
  A: "$45M"
  → The model just outputs the answer. Did it read the chart correctly? Got lucky?

WITH EXPLANATION (Chain-of-Thought):
  Q: "What is the total revenue in Q4? Explain your reasoning."
  Image: [bar chart]
  A: "Looking at the bar chart, the x-axis shows quarters (Q1-Q4) and the y-axis
      shows revenue in millions. The bar for Q4 reaches the 45 mark on the y-axis.
      Therefore, the total revenue in Q4 is $45M."
  → The model shows its work. We can verify its reasoning. And crucially:
     the process of generating the explanation FORCES the model to attend to
     the right parts of the image.
```

**Key Models/Papers:**

**Multimodal-CoT (Zhang et al., 2023):**
```
Two-stage approach:
  Stage 1: Generate rationale — "I see a bar chart with Q1-Q4..."
  Stage 2: Generate answer using the rationale as additional context
  
  Finding: Generating rationale FIRST, then using it for answering,
  outperforms direct answer generation
  
  FOR YOUR THESIS: This is evidence that explanation-aware training
  improves reasoning accuracy (supports your RQ3 hypothesis)
```

**ScienceQA (Lu et al., 2022):**
```
Dataset with built-in explanations:
  Question: "Which animal has a better sense of smell?"
  Image: [photo of dog and cat]
  Choices: (A) Dog (B) Cat
  Explanation: "Dogs have approximately 300 million olfactory receptors
               in their noses, compared to about 6 million in cats.
               The image shows a dog and a cat. Based on biological
               evidence, dogs have a superior sense of smell."
  Answer: (A)

  21K questions across science topics with images, choices,
  explanations, and answers
  
  FOR YOUR THESIS: This is your PRIMARY dataset for explanation-aware training
  It has ground-truth explanations you can train against and evaluate
```

**A-OKVQA (Schwenk et al., 2022):**
```
Visual QA with rationales:
  Image: [photo of a kitchen]
  Question: "What room is this?"
  Rationale: "The image shows cabinets, a stove, a refrigerator,
             and a sink, which are typical features of a kitchen."
  Answer: "Kitchen"
  
  ~25K questions with human-written rationales
  Good for training explanation generation on natural images
```

**Hands-on (Notebook 09):**
- Load ScienceQA dataset, explore examples across domains
- Run your base VLM on ScienceQA examples:
  - With prompt: "Answer the question." → measure accuracy
  - With prompt: "First explain your reasoning, then answer." → measure accuracy + explanation quality
- Compare accuracy with and without explanation prompting (zero-shot CoT)
- This gives you a baseline measurement for RQ3

**Your Thesis's Explanation-Aware Training Approach:**
```
STANDARD TRAINING:
  Input: [image] + "What is the value in Q4?"
  Target: "$45M"
  Loss: cross-entropy on answer tokens only

EXPLANATION-AWARE TRAINING (your approach):
  Input: [image] + "What is the value in Q4? Explain your reasoning."
  Target: "The bar chart shows quarterly revenue. The Q4 bar reaches $45M
           on the y-axis. Therefore, the answer is $45M."
  Loss: cross-entropy on BOTH explanation AND answer tokens

  WHY THIS IS GENERATIVE ALIGNMENT:
  To generate a correct explanation, the model must:
  1. Correctly attend to the relevant image regions (the Q4 bar, the y-axis)
  2. Extract the correct information (the numerical value)
  3. Verbalize its understanding (proves alignment between visual and textual)
  
  The explanation generation acts as a RICH ALIGNMENT SIGNAL between
  the visual features and the language model's understanding.
  This is "Cross-Modal Generative Alignment" — your thesis title.
```

**Resources:**
- **Read:** "Multimodal Chain-of-Thought Reasoning in Language Models" (Zhang et al., 2023)
- **Read:** "Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering" (Lu et al., 2022) — ScienceQA
- **Read:** "A-OKVQA: A Benchmark for Visual Question Answering using World Knowledge" (Schwenk et al., 2022)

**Thesis Writing:**
- Chapter 2, Section 2.5: "Explanation-Aware and Chain-of-Thought Reasoning in VLMs"
- Connect the dots: explanation generation = a form of generative alignment
- This is the novel angle of your thesis — frame it clearly

---

### Week 3: Survey Completion + Implementation Kickoff (Days 17-21)

**Days 17-18: Complete the Survey + Write Chapter 3**

By now you have deep understanding of:
- Transformers (encoder/decoder/encoder-decoder)
- Vision Transformers (how images become tokens)
- Contrastive alignment (CLIP)
- Generative alignment (BLIP-2)
- VLM architectures (LLaVA, modern VLMs)
- Document/Chart-specific models
- Explanation-aware reasoning (CoT, ScienceQA)

**Write Chapter 3: Research Framework**

This chapter FRAMES your experiments. It's not a survey — it's YOUR analytical framework.

**Section 3.1: Contrastive vs. Generative Alignment — The Design Space**

Create a comparison table that becomes the backbone of your RQ2 experiments:

| Dimension | Contrastive (CLIP-style) | Generative (BLIP-2/LLaVA-style) |
|-----------|-------------------------|----------------------------------|
| Alignment granularity | Image-level (one vector) | Token-level (generate each word) |
| Learning signal | Binary (match/no match) | Rich (generate correct sequence) |
| Computational cost | Lower (just embeddings) | Higher (full sequence generation) |
| Spatial understanding | Weak (global pooling) | Strong (patch-level attention) |
| Numerical reasoning | Very weak | Moderate (can generate numbers) |
| Explanation capability | None (only embeddings) | Natural (generative by design) |
| Document understanding | Poor (designed for natural images) | Better (can read text in images) |
| Chart understanding | Poor | Moderate (depends on training data) |

**Section 3.2: Explanation-Aware Training Objectives**

Define the specific training variants you'll compare:

```
Variant A — ANSWER-ONLY (Baseline):
  Target: just the answer
  Loss: L_answer = CrossEntropy(predicted_answer, ground_truth_answer)

Variant B — EXPLANATION-THEN-ANSWER:
  Target: explanation followed by answer
  Loss: L_explain = CrossEntropy(predicted_explanation, ground_truth_explanation)
       + CrossEntropy(predicted_answer, ground_truth_answer)

Variant C — EXPLANATION-WEIGHTED:
  Same as B, but with adjustable weighting:
  Loss: α * L_explanation + (1 - α) * L_answer
  Experiment with α = 0.3, 0.5, 0.7

Variant D — CONTRASTIVE + GENERATIVE HYBRID:
  Add a contrastive loss on the visual-textual embeddings
  alongside the generative explanation loss
  Loss: L_contrastive + β * L_generative_explain
```

**Section 3.3: Cross-Domain Visual Reasoning Challenges**
- Document: requires text reading (OCR-like), layout understanding, spatial relationships
- Chart: requires numerical extraction, trend identification, comparison operations
- Natural Image: requires scene understanding, object recognition, spatial reasoning
- Hypothesis: Explanation-aware training helps MOST on charts and documents (where reasoning steps are explicit)

**Section 3.4: Evaluation Framework**

| Metric | Measures | Applied to |
|--------|----------|-----------|
| Accuracy | Correct answers | All domains |
| BLEU-4 | N-gram overlap of explanation | Explanation quality |
| ROUGE-L | Longest common subsequence | Explanation quality |
| BERTScore | Semantic similarity of explanation | Explanation quality |
| Faithfulness | Does explanation match the visual evidence? | Explanation quality |
| Domain-Specific | WER (documents), numerical accuracy (charts) | Per-domain |

---

**Days 19-21: Implementation Kickoff — Data Pipeline + Base Model**
**→ Start building toward RQ2, RQ3, RQ4**

**Day 19: Dataset Preparation**

Prepare unified data loaders for all your datasets:

```python
# Your unified data format
{
    "id": "docvqa_001",
    "domain": "document",           # document | chart | image
    "image": PIL.Image,             # the visual input
    "question": "What is the total amount due?",
    "answer": "$1,250.00",
    "explanation": "The document shows an invoice. Looking at the bottom right,
                    there is a field labeled 'Total Due' with the value $1,250.00.",
    "has_explanation": True          # not all samples have ground-truth explanations
}

# Dataset sizes for your experiments:
# DocVQA:    ~5K training, ~1K eval (subset — full dataset is 50K)
# ChartQA:   ~5K training, ~1K eval (subset — full dataset is 32K)
# ScienceQA: ~10K training, ~2K eval (HAS explanations!)
# A-OKVQA:   ~5K training, ~1K eval (HAS rationales!)
# VQAv2:     ~5K training, ~1K eval (natural images, NO explanations)
```

**Creating Explanation Annotations:**
For DocVQA and ChartQA, which don't have explanations, you need to generate them:

```python
# Option 1: Use a strong VLM (GPT-4V or Claude) to generate explanations
# for your training data. This is common in research.
# Cost: ~$20-50 for 10K examples via API

prompt = """Look at this image and answer the question.
First, explain your reasoning step by step, then give the final answer.

Question: {question}
Ground truth answer: {answer}

Provide your explanation of how someone would arrive at this answer
by looking at the image."""

# Option 2: Template-based explanations for charts
# "The chart shows {chart_type}. The x-axis represents {x_label}
#  and the y-axis represents {y_label}. Reading the value at {query_point},
#  the answer is {answer}."

# Option 3: Use only ScienceQA + A-OKVQA for explanation-aware training
# and DocVQA + ChartQA only for answer-only evaluation
# This is cleaner methodologically (no synthetic explanations)
```

**Day 20-21: Load Base Model + Verify Pipeline**

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model

# Load base model (Qwen2-VL-2B as example — adjust based on your GPU)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# Apply QLoRA for efficient fine-tuning
lora_config = LoraConfig(
    r=16,                       # rank — lower = fewer parameters
    lora_alpha=32,              # scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# → trainable params: ~20M (vs 2B total) — this is what makes it feasible

# Verify forward pass works
image = load_sample_image()
inputs = processor(
    text="<|im_start|>user\n<|image|>What do you see in this image?<|im_end|>\n<|im_start|>assistant\n",
    images=image,
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256)
    print(processor.decode(outputs[0], skip_special_tokens=True))
```

**Run baseline evaluation (before any fine-tuning):**
- Evaluate the pre-trained model on ALL your test sets
- This gives you the "zero-shot" baseline
- Record: accuracy on DocVQA, ChartQA, ScienceQA, A-OKVQA, VQAv2
- Record: explanation quality when prompted to explain (even without explanation training)
- These numbers go directly into your Chapter 5 results tables

**Week 3 Deliverable:**
- Complete drafts of Chapters 2 and 3
- Working data pipeline loading all 5 datasets
- Base model loaded, QLoRA configured, forward pass verified
- Zero-shot baseline numbers for all domains
- Chapter 4 (Methodology) outline written

---

## PHASE 2: IMPLEMENTATION + TRAINING (Days 22-35)
### → Generates the evidence to answer RQ2, RQ3, RQ4

---

### Week 4: Training the Models (Days 22-28)

You will train FOUR model variants to answer your research questions:

```
MODEL A — BASELINE (answer-only fine-tuning):
  Training data: all domains, answer-only targets
  This is the control group for RQ3

MODEL B — EXPLANATION-AWARE (explanation + answer):
  Training data: explanation-annotated data (ScienceQA + A-OKVQA + generated explanations)
  This is the treatment group for RQ3

MODEL C — CONTRASTIVE-ENHANCED:
  Add a contrastive alignment loss during fine-tuning
  This is one arm of the RQ2 comparison

MODEL D — GENERATIVE-EXPLANATION (your thesis's main contribution):
  Generative alignment through explanation generation
  = Model B with emphasis on explanation as alignment signal
  This is the other arm of the RQ2 comparison
```

**Days 22-24: Train Model A (Baseline)**

```python
# Training configuration
training_config = {
    "model": "Qwen2-VL-2B + QLoRA",
    "learning_rate": 2e-4,          # standard for QLoRA
    "batch_size": 4,                # per GPU
    "gradient_accumulation": 8,     # effective batch 32
    "epochs": 3,
    "warmup_ratio": 0.03,
    "max_seq_length": 2048,
    "precision": "bf16",
    "optimizer": "adamw_8bit",      # memory-efficient
    "lr_scheduler": "cosine",
}

# Data mix for baseline:
# 30% DocVQA (answer only)
# 30% ChartQA (answer only)
# 20% ScienceQA (answer only — drop explanations)
# 20% VQAv2/A-OKVQA (answer only — drop rationales)

# Training prompt format (Qwen2-VL chat template):
"""
<|im_start|>user
<|image|>
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>
"""

# Train using HuggingFace Trainer or TRL's SFTTrainer
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir="./checkpoints/baseline",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
    ),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=processor,
)
trainer.train()
```

**→ Save checkpoint. This is your baseline for ALL comparisons.**

**Days 25-26: Train Model B (Explanation-Aware)**

Same setup as Model A, but with explanation targets:

```python
# Training prompt format WITH explanations:
"""
<|im_start|>user
<|image|>
{question} Explain your reasoning step by step, then give the final answer.<|im_end|>
<|im_start|>assistant
Reasoning: {explanation}

Answer: {answer}<|im_end|>
"""

# Data mix for explanation-aware:
# ScienceQA: use ground-truth explanations (10K examples)
# A-OKVQA: use ground-truth rationales (5K examples)
# DocVQA: use generated explanations or skip (see Day 19 options)
# ChartQA: use generated explanations or skip
# VQAv2: answer-only (no explanations available)

# Same hyperparameters as baseline for fair comparison
# ONLY difference: targets include explanations
```

**→ RQ3 Direct Test:** Compare Model B accuracy vs. Model A accuracy. Does training with explanations improve answer accuracy?

**Days 27-28: Train Models C and D (Alignment Variants)**

**Model C — Contrastive-Enhanced:**
```python
# Add a contrastive loss term during fine-tuning
# Pull the visual features and text features, apply InfoNCE loss
# alongside the standard language modeling loss

class ContrastiveVLMTrainer:
    def compute_loss(self, model, inputs):
        # Standard language modeling loss
        lm_outputs = model(**inputs)
        lm_loss = lm_outputs.loss

        # Extract visual and text representations
        visual_embeds = extract_visual_embeds(model, inputs)  # pool over visual tokens
        text_embeds = extract_text_embeds(model, inputs)      # pool over text tokens

        # Contrastive loss (InfoNCE)
        similarity = visual_embeds @ text_embeds.T / temperature
        labels = torch.arange(similarity.shape[0]).to(similarity.device)
        contrastive_loss = F.cross_entropy(similarity, labels)

        return lm_loss + 0.1 * contrastive_loss  # weighted combination
```

**Model D — Generative-Explanation Alignment (YOUR main contribution):**
```python
# This is Model B (explanation-aware) with explicit emphasis on the
# explanation as an alignment mechanism

# Key difference from Model B:
# 1. Higher weight on explanation tokens vs answer tokens
# 2. Explicitly track visual attention during explanation generation
# 3. Optional: add an auxiliary loss encouraging visual grounding

class ExplanationAlignmentTrainer:
    def compute_loss(self, model, inputs):
        outputs = model(**inputs)

        # Split loss into explanation portion and answer portion
        explanation_mask = create_explanation_mask(inputs)  # tokens in explanation
        answer_mask = create_answer_mask(inputs)            # tokens in answer

        explanation_loss = masked_cross_entropy(outputs.logits, inputs.labels, explanation_mask)
        answer_loss = masked_cross_entropy(outputs.logits, inputs.labels, answer_mask)

        # Weight explanation loss higher — it's the alignment signal
        total_loss = 0.7 * explanation_loss + 0.3 * answer_loss
        return total_loss
```

**Week 4 Deliverable:**
- 4 trained models (A: baseline, B: explanation-aware, C: contrastive, D: generative-explanation)
- Training logs with loss curves for all 4 models
- Preliminary evaluation on a small held-out set
- Start writing Chapter 4 (Implementation) with training details

---

### Week 5: Full Evaluation + Ablation Setup (Days 29-35)

**Days 29-31: Comprehensive Evaluation of All Models**

Run ALL four models on ALL test sets with ALL metrics:

```
EVALUATION MATRIX:

              DocVQA    ChartQA   ScienceQA   A-OKVQA   VQAv2
              (doc)     (chart)   (science)   (knowledge) (general)
Model A       acc       acc       acc         acc        acc
(baseline)

Model B       acc       acc       acc+expl    acc+expl   acc
(expl-aware)  (expl)    (expl)    quality     quality

Model C       acc       acc       acc         acc        acc
(contrastive)

Model D       acc       acc       acc+expl    acc+expl   acc
(gen-explain) (expl)    (expl)    quality     quality

Metrics per cell:
  acc = exact match accuracy or soft accuracy
  expl = BLEU-4, ROUGE-L, BERTScore of generated explanation
  quality = human evaluation of explanation faithfulness (on subset)
```

```python
# Evaluation script structure
def evaluate_model(model, processor, test_dataset, generate_explanation=False):
    results = {
        "accuracy": [],
        "bleu_scores": [],
        "rouge_scores": [],
        "bertscore": [],
        "predictions": [],  # save for qualitative analysis
    }

    for sample in test_dataset:
        image = sample["image"]
        question = sample["question"]

        if generate_explanation:
            prompt = f"{question} Explain your reasoning step by step, then give the final answer."
        else:
            prompt = question

        # Generate
        inputs = processor(text=format_prompt(prompt), images=image, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=512)
        prediction = processor.decode(outputs[0], skip_special_tokens=True)

        # Parse answer and explanation from prediction
        answer, explanation = parse_output(prediction)

        # Compute metrics
        results["accuracy"].append(compute_accuracy(answer, sample["answer"]))
        if generate_explanation and sample.get("explanation"):
            results["bleu_scores"].append(compute_bleu(explanation, sample["explanation"]))
            results["rouge_scores"].append(compute_rouge(explanation, sample["explanation"]))

    return results
```

**Days 32-34: Ablation Studies**
**→ Directly answers RQ4**

**Ablation 1: Domain-Specific Performance Breakdown (RQ4)**
- For each model, break down performance by domain
- Question: Does explanation-aware training help MORE on charts than on natural images?
- Expected finding: Charts and documents benefit most (reasoning is more explicit)

**Ablation 2: Explanation Weight Sensitivity (RQ3)**
- Train Model D variants with different α values (explanation weight):
  - α = 0.0 (no explanation loss — equivalent to Model A)
  - α = 0.3 (low explanation weight)
  - α = 0.5 (balanced)
  - α = 0.7 (high explanation weight — the default Model D)
  - α = 1.0 (explanation only, no answer-specific loss)
- This shows the optimal balance between explanation and answer training

**Ablation 3: Training Data Size Impact (RQ4)**
- Train Model D with increasing amounts of explanation data:
  - 1K explanation-annotated samples
  - 5K explanation-annotated samples
  - 10K explanation-annotated samples
  - 15K explanation-annotated samples
- Question: How much explanation data do you need for improvement?

**Ablation 4: Explanation Quality vs. Answer Accuracy Correlation (RQ3)**
- For Model D's predictions: plot explanation quality (BLEU/ROUGE) vs. answer accuracy
- Question: When the model generates a GOOD explanation, is it more likely to get the answer right?
- This tests whether explanation generation is genuinely helping reasoning or just adding text

**Ablation 5: Cross-Domain Transfer (RQ4)**
- Train on explanations from ONE domain, test on ALL domains
  - Train on ScienceQA explanations → test on ChartQA, DocVQA
  - Train on ChartQA explanations → test on ScienceQA, DocVQA
- Question: Does explanation-aware training transfer across visual domains?

**Day 35: Attention + Representation Analysis**

```python
# Extract and visualize attention patterns
def visualize_cross_modal_attention(model, image, question):
    """
    For each model variant, visualize which image patches the model
    attends to when generating the explanation vs. the answer.
    """
    # Hook into the cross-attention layers
    attention_weights = []
    def hook_fn(module, input, output):
        attention_weights.append(output[1])  # attention weights

    # Register hooks on cross-attention layers
    for layer in model.language_model.layers:
        layer.cross_attention.register_forward_hook(hook_fn)

    # Generate with attention output
    outputs = model.generate(inputs, output_attentions=True)

    # Reshape attention to image grid and overlay on original image
    # This shows: "When answering about Q4 revenue, does Model D attend
    # to the Q4 bar more than Model A?"
    plot_attention_overlay(image, attention_weights, question)
```

Visualizations to create:
1. Attention heatmaps: which patches each model focuses on for the same question
2. t-SNE/UMAP of internal representations: do explanation-trained models create better-separated clusters?
3. Attention pattern comparison: baseline vs. explanation-aware models

**Week 5 Deliverable:**
- Complete evaluation results for all models on all datasets
- All ablation study results
- Attention visualizations
- **Clear data tables answering RQ2, RQ3, RQ4**
- Chapter 4 (Implementation) complete draft

---

## PHASE 3: ANALYSIS + THESIS WRITING (Days 36-50)
### → Documents all findings and synthesizes the answers to RQ1-4

---

### Week 6: Deep Analysis + Write Core Chapters (Days 36-42)

**Days 36-38: Synthesize Results → Write Chapter 5**

**Section 5.1: Experimental Setup**
- Hardware, software, training details (reproducibility)
- Table of all hyperparameters
- Dataset statistics and splits

**Section 5.2: Contrastive vs. Generative Alignment (RQ2 Answer)**

```
RQ2 ANSWER FRAMEWORK — Fill with your actual results:

Alignment Comparison Summary:
┌─────────────────────────┬──────────────┬──────────────┬──────────────┐
│ Domain                  │ Model A      │ Model C      │ Model D      │
│                         │ (baseline)   │ (contrastive)│ (gen-explain)│
├─────────────────────────┼──────────────┼──────────────┼──────────────┤
│ DocVQA accuracy         │ ___%         │ ___%         │ ___%         │
│ ChartQA accuracy        │ ___%         │ ___%         │ ___%         │
│ ScienceQA accuracy      │ ___%         │ ___%         │ ___%         │
│ VQAv2 accuracy          │ ___%         │ ___%         │ ___%         │
│ Avg explanation quality  │ N/A          │ N/A          │ BLEU=___     │
│ Training time           │ ___hrs       │ ___hrs       │ ___hrs       │
└─────────────────────────┴──────────────┴──────────────┴──────────────┘

Key Finding for RQ2: [Generative alignment through explanation generation
produces ___% improvement on structured visual domains (documents, charts)
compared to contrastive enhancement, while performing ___% on natural images.
This suggests that generative alignment is particularly effective when
the visual content requires explicit reasoning steps.]
```

**Section 5.3: Explanation-Aware Training Impact (RQ3 Answer)**

```
RQ3 ANSWER FRAMEWORK:

Explanation Impact:
┌─────────────────────────┬──────────────┬──────────────┬──────────┐
│ Metric                  │ Model A      │ Model B      │ Δ        │
│                         │ (no explain) │ (explain)    │          │
├─────────────────────────┼──────────────┼──────────────┼──────────┤
│ Overall accuracy        │ ___%         │ ___%         │ +/-___%  │
│ DocVQA accuracy         │ ___%         │ ___%         │ +/-___%  │
│ ChartQA accuracy        │ ___%         │ ___%         │ +/-___%  │
│ ScienceQA accuracy      │ ___%         │ ___%         │ +/-___%  │
│ Explanation BLEU-4      │ N/A          │ ___          │ -        │
│ Explanation ROUGE-L     │ N/A          │ ___          │ -        │
│ Explanation faithfulness│ N/A          │ ___%         │ -        │
└─────────────────────────┴──────────────┴──────────────┴──────────┘

Key Finding for RQ3: [Training with explanations ___
(improves/does not improve) accuracy by ___%. The effect is strongest
on ___ domain, suggesting that explanation-aware training is most
beneficial when ___. Explanation quality (BLEU=___, ROUGE=___) correlates
with answer accuracy (r=___), providing evidence that the model
genuinely uses the explanation process for reasoning.]
```

**Section 5.4: Cross-Domain Analysis (RQ4 Answer)**

```
RQ4 ANSWER FRAMEWORK:

Domain-Specific Factor Analysis:
┌──────────────┬───────────────────┬──────────────────┬─────────────────┐
│ Factor       │ Documents         │ Charts           │ Natural Images  │
├──────────────┼───────────────────┼──────────────────┼─────────────────┤
│ Key challenge│ Text reading,     │ Numerical        │ Scene           │
│              │ layout awareness  │ extraction,      │ understanding,  │
│              │                   │ spatial reasoning│ object recog.   │
├──────────────┼───────────────────┼──────────────────┼─────────────────┤
│ Explanation  │ ___% impact       │ ___% impact      │ ___% impact     │
│ training     │                   │                  │                 │
│ impact       │                   │                  │                 │
├──────────────┼───────────────────┼──────────────────┼─────────────────┤
│ Best model   │ Model _           │ Model _          │ Model _         │
├──────────────┼───────────────────┼──────────────────┼─────────────────┤
│ Bottleneck   │ ___               │ ___              │ ___             │
└──────────────┴───────────────────┴──────────────────┴─────────────────┘

Key Finding for RQ4: [The dominant factor for document understanding is ___,
for chart reasoning is ___, and for natural image understanding is ___.
Explanation-aware training has the largest impact on ___ domain because ___.
Cross-domain transfer of explanation training is ___ (limited/moderate/strong),
suggesting that ___ is/is not shared across visual domains.]
```

**Section 5.5: Qualitative Analysis**
- Show 5-10 examples per domain where Model D generates explanations
- Include both successes AND failures
- For failures: analyze WHY the explanation was wrong (attended to wrong region? misread text? arithmetic error?)
- This qualitative analysis is often the most valuable part of the thesis

---

**Days 39-42: Write Remaining Chapters**

**Chapter 6: Discussion (5-7 pages)**

Section 6.1: Key Findings
- Summarize the answer to each RQ in 1-2 paragraphs with evidence

Section 6.2: Implications
- For practitioners: When should you use explanation-aware VLMs? (Answer: structured content)
- For researchers: Generative alignment via explanations is a viable alternative to contrastive methods
- For the field: Explanation generation is not just for interpretability — it's an alignment signal

Section 6.3: Relationship to Commercial Models
- GPT-4V/Gemini can explain their reasoning — your work provides evidence for WHY this matters
- Your findings suggest that their explanation capabilities may contribute to their accuracy, not just user trust

Section 6.4: Limitations
- Small model scale (2B-7B) may not reflect behavior of 70B+ models
- Generated explanations for DocVQA/ChartQA training may introduce noise
- Limited to English, single-image, specific domains
- QLoRA fine-tuning is not the same as full fine-tuning
- Evaluation metrics for explanations (BLEU/ROUGE) are imperfect measures of quality

---

### Week 7: Complete + Polish + Submit (Days 43-50)

**Days 43-44: Write Chapter 1 (Introduction)**

Write this LAST — you now know exactly what your thesis contains.

Structure:
- Hook: "When a vision–language model answers a question about a financial chart, does it truly understand the data, or is it pattern-matching? Can we know, if it doesn't explain itself?"
- Context: VLMs are increasingly used for document processing, chart analysis, visual reasoning
- Problem: Current alignment methods are coarse; explanation capability is underexplored as an alignment strategy
- State all four RQs explicitly
- Preview contributions (survey + 4 model variants + ablation studies + cross-domain analysis)
- Thesis structure overview

**Days 45-46: Create ALL Figures and Diagrams**

Essential figures:
1. Transformer architecture (encoder/decoder/encoder-decoder) — Chapter 2
2. ViT patch embedding process — Chapter 2
3. CLIP contrastive alignment — Chapter 2
4. BLIP-2 Q-Former generative alignment — Chapter 2
5. Contrastive vs. Generative alignment comparison diagram — Chapter 3
6. Your training pipeline (4 model variants) — Chapter 4
7. Results tables (filled with actual numbers) — Chapter 5
8. Attention heatmap visualizations — Chapter 5
9. Ablation study plots (line charts showing parameter sweeps) — Chapter 5
10. Qualitative examples grid — Chapter 5
11. t-SNE/UMAP representation visualizations — Chapter 5

**Days 47-48: Format + Citations**
- Ensure all citations are correct (BibTeX)
- Table of contents
- List of figures, list of tables
- Consistent formatting throughout
- Page numbers, headers
- Abstract (write this VERY LAST — it's a summary of the whole thesis)

**Days 49-50: Review + Submit**
- Day 49: Have someone else read for clarity
- Day 50: Final revisions + submit

**Target thesis length: ~60-75 pages**

---

## HOW EACH PHASE MAPS TO RESEARCH QUESTIONS

```
                        RQ1        RQ2        RQ3        RQ4
                      (Survey)   (Contrast   (Explain)  (Domain)
                                 vs Gen)
Phase 1: Foundations   ████████   ░░██░░     ░░██░░     ░░░░░░
Phase 2: Build+Train   ░░░░░░    ████████   ████████   ░░██░░
Phase 3: Experiments   ░░░░░░    ████████   ████████   ████████
Thesis Writing         ████████  ████████   ████████   ████████

████ = primary focus
░░██ = secondary/supporting
░░░░ = not directly addressed in this phase
```

---

## COMPLETE READING LIST

### Must-Read Papers (Priority Order)

**Transformer Foundations (RQ1):**
1. "Attention Is All You Need" (Vaswani et al., 2017) — THE transformer paper
2. "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020) — ViT
3. "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2019) — encoder-only

**Vision–Language Alignment (RQ1 + RQ2):**
4. "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021) — CLIP
5. "BLIP-2: Bootstrapping Language-Image Pre-training" (Li et al., 2023) — YOUR MOST IMPORTANT PAPER
6. "CoCa: Contrastive Captioners are Image-Text Foundation Models" (Yu et al., 2022) — hybrid alignment
7. "SigLIP: Sigmoid Loss for Language Image Pre-Training" (Zhai et al., 2023) — improved contrastive

**VLM Architectures (RQ2):**
8. "Visual Instruction Tuning" (Liu et al., 2023) — LLaVA
9. "LLaVA-NeXT: Improved reasoning, OCR, and world knowledge" (Liu et al., 2024)
10. "InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks" (Chen et al., 2024)
11. "Qwen-VL: A Versatile Vision-Language Model" (Bai et al., 2023)
12. "Flamingo: a Visual Language Model for Few-Shot Learning" (Alayrac et al., 2022) — cross-attention approach

**Document Understanding (RQ4):**
13. "Donut: Document Understanding Transformer" (Kim et al., 2022) — OCR-free document understanding
14. "Pix2Struct: Screenshot Parsing as Pretraining" (Lee et al., 2023)
15. "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking" (Huang et al., 2022)

**Chart Understanding (RQ4):**
16. "DePlot: One-shot visual language reasoning by plot-to-table translation" (Liu et al., 2023)
17. "MatCha: Enhancing Visual Language Pretraining with Math Reasoning and Chart Derendering" (Liu et al., 2023)
18. "ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning" (Masry et al., 2022)

**Explanation-Aware / CoT (RQ3):**
19. "Multimodal Chain-of-Thought Reasoning in Language Models" (Zhang et al., 2023) — CRITICAL for RQ3
20. "Learn to Explain: Multimodal Reasoning via Thought Chains for Science QA" (Lu et al., 2022) — ScienceQA
21. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (Wei et al., 2022)
22. "A-OKVQA: A Benchmark for Visual QA using World Knowledge" (Schwenk et al., 2022)

**Efficient Fine-Tuning:**
23. "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2022) — understand what you're using
24. "QLoRA: Efficient Finetuning of Quantized Language Models" (Dettmers et al., 2023)

### Must-Watch

1. **Andrej Karpathy** — "Let's build GPT from scratch" (YouTube, ~2hrs) — WEEK 1 ESSENTIAL
2. **Andrej Karpathy** — "Let's build the GPT Tokenizer" (YouTube, ~2hrs) — WEEK 1
3. **Umar Jamil** — "Attention is All You Need" explained (YouTube) — WEEK 1
4. **Umar Jamil** — "Vision Transformer (ViT)" explained (YouTube) — WEEK 1
5. **Umar Jamil** — "CLIP" explained (YouTube) — WEEK 1
6. **Umar Jamil** — "LLaVA" explained (YouTube) — WEEK 2
7. **Jay Alammar** — "The Illustrated Transformer" (blog) — WEEK 1 ESSENTIAL
8. **Yannic Kilcher** — CLIP, ViT, BLIP explanations (YouTube) — WEEKS 1-2
9. **3Blue1Brown** — "But what is a neural network?" series (YouTube) — if you need fundamentals refresh

### Codebases to Study
1. **LLaVA:** https://github.com/haotian-liu/LLaVA — simplest VLM architecture
2. **BLIP-2/LAVIS:** https://github.com/salesforce/LAVIS — generative alignment reference
3. **InternVL:** https://github.com/OpenGVLab/InternVL — modern VLM reference
4. **nanoGPT:** https://github.com/karpathy/nanoGPT — transformer from scratch
5. **Multimodal-CoT:** https://github.com/amazon-science/mm-cot — explanation-aware training

---

## RISK MITIGATION

| Risk | Impact | Mitigation | RQ Impact |
|------|--------|------------|-----------|
| Base model too large for GPU | High | Use 2B model + QLoRA, or switch to Phi-3-Vision (3.8B) | Minor — still answers RQs |
| Explanation generation doesn't improve accuracy | Medium | Document WHY — negative result IS a valid RQ3 answer. Analyze when it helps vs hurts. | Actually strengthens RQ3 |
| Generated explanation annotations are noisy | Medium | Rely primarily on ScienceQA + A-OKVQA (human-written). Use generated only as supplement. | RQ3 still answerable |
| Not enough time for all ablations | Medium | Priority: (1) Models A+B comparison (2) Models C+D comparison (3) Ablation on α (4) Domain transfer | See priority order below |
| Chart/Document results are poor | Low-Medium | Analyze WHY — resolution? OCR capability? This IS RQ4 analysis | Helps answer RQ4 |
| Running out of time | High | See priority cutoff below | Graceful degradation |

**If Time Gets Tight — Priority Order:**
1. Complete survey chapters (RQ1 answered) — this alone is 35% of thesis value
2. Train Model A (baseline) + Model B (explanation-aware) → directly answers RQ3
3. Evaluate on all domains → answers RQ4
4. Train Model D (generative-explanation) → answers RQ2
5. Train Model C (contrastive) → completes RQ2 comparison
6. Run ablation experiments → deepens RQ3 + RQ4
7. Attention visualizations and qualitative analysis → polish

Even completing only steps 1-3 gives you a solid thesis with clear answers to RQ1, RQ3, and partial RQ4.

---

## WEEK-BY-WEEK CHECKLIST (with RQ mapping)

- [ ] **Days 1-2:** Setup + Data exploration → Foundation
- [ ] **Days 3-9 (Week 1):** Transformer deep dive + ViT + CLIP → **RQ1 foundation**
- [ ] **Days 10-16 (Week 2):** BLIP-2 + LLaVA + Document/Chart models + CoT → **RQ1 + RQ2/RQ3 framing**
- [ ] **Days 17-21 (Week 3):** Complete survey writing + implementation kickoff → **RQ1 complete, methodology start**
- [ ] **Days 22-28 (Week 4):** Train all 4 model variants → **Evidence for RQ2, RQ3**
- [ ] **Days 29-35 (Week 5):** Full evaluation + ablation studies → **RQ2, RQ3, RQ4 evidence**
- [ ] **Days 36-42 (Week 6):** Analysis + write Chapters 4-6 → **All RQs documented**
- [ ] **Days 43-50 (Week 7):** Chapter 1, figures, formatting, review, submit

---

## DAILY RHYTHM

```
Monday-Friday:
  Morning (2-3 hrs):    Read papers / watch explanations / learn concepts
  Afternoon (3-4 hrs):  Code, implement, train, debug
  Evening (1-2 hrs):    Write thesis sections / document findings

Saturday:
  Long training runs. Write while GPU works.
  Ask yourself: "Can I answer each RQ better than last week?"

Sunday:
  Light review. Plan next week. Rest.
  Burnout kills theses faster than deadlines.
```

**Key Time-Saving Principles:**
- Write as you learn — don't wait until "writing phase" to start writing
- Use pre-trained models — fine-tune, don't build from scratch (except the learning notebooks)
- QLoRA is your friend — it makes 2B-7B models trainable on a single GPU
- ScienceQA has explanations built in — no need to generate them for your primary dataset
- Launch training runs before bed — GPUs don't sleep

---

## WHAT DEEP UNDERSTANDING LOOKS LIKE

After this thesis, you should be able to:

1. **Whiteboard a VLM from scratch** — draw the ViT, the projection layer, the LLM, explain attention, explain how image patches become tokens that the language model processes
2. **Explain WHY designs work** — not just "CLIP uses contrastive learning" but "contrastive learning pushes matching pairs together in a shared embedding space using InfoNCE loss, which has the property that..."
3. **Predict architectural tradeoffs** — "If I add cross-attention between the vision encoder and LLM instead of just projecting, I'll get better visual grounding but slower inference because..."
4. **Read any VLM paper** and immediately understand the architecture, training, and where it fits in the landscape
5. **Debug multimodal training** — "The model ignores the image because the visual tokens aren't in the right embedding space — the projection layer isn't trained enough"

This isn't about building the best model. It's about building the deepest understanding.

And that understanding is exactly what separates an ML/AI engineer from someone who calls `model.fit()`.
